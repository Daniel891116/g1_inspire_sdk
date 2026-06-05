"""Rerun-based motion preview.

Loads the G1+Inspire URDF in Pinocchio, runs forward-kinematics on every
command, and logs the resulting link transforms to a Rerun viewer.
Attach an instance to ``G1Arm`` / ``InspireHand`` (or to ``G1InspireRobot``)
and joint targets are mirrored to the viewer each tick — useful for
sanity-checking a trajectory before running it on hardware.

Hand visualization: the default URDF (``g1_body29_inspire_hand.urdf``)
includes the Inspire hand kinematics. The 6-DOF canonical hand command
is mapped onto the 6 actuated finger joints and the URDF mimic
relationships are applied manually (Pinocchio does not honour URDF
``<mimic>`` tags). Hand commands are also logged as scalar time-series
for graphing.

If a URDF without finger joints is supplied (e.g. ``g1_body29_hand14.urdf``)
the hand-FK path is silently skipped and only scalar logging fires.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Optional

import numpy as np
import pinocchio as pin
import rerun as rr

from ._config import (
    Config,
    DEFAULT_G1_INSPIRE_URDF,
    DEFAULT_G1_URDF,
    DEFAULT_G1_YAML,
    ASSETS_ROOT,
)


_ENTITY_ROOT = "robot"
_EE_TARGET_ROOT = "ee_target"
_HAND_SCALAR_ROOT = "hand"

_CANONICAL_HAND_NAMES = (
    "index_mcp",
    "middle_mcp",
    "ring_mcp",
    "little_mcp",
    "thumb_rotate",
    "thumb_flex",
)


def _hand_joint_specs(side: str) -> dict:
    """Mapping spec for one Inspire hand in the G1+Inspire URDF.

    Returns a dict with:
      * ``actuated``: {canonical_name: urdf_joint_name}  (6 entries)
      * ``mimic``: list of ``(driven_joint, source_canonical, multiplier)``
        — extracted from the URDF's ``<mimic>`` tags. Pinocchio drops
        mimic info on load, so the preview applies these relationships
        manually each FK update.
    """
    p = side  # "left" or "right"
    actuated = {
        "index_mcp": f"{p}_index_1_joint",
        "middle_mcp": f"{p}_middle_1_joint",
        "ring_mcp": f"{p}_ring_1_joint",
        "little_mcp": f"{p}_little_1_joint",
        "thumb_rotate": f"{p}_thumb_1_joint",
        "thumb_flex": f"{p}_thumb_2_joint",
    }
    mimic = [
        (f"{p}_index_2_joint", "index_mcp", 1.05),
        (f"{p}_middle_2_joint", "middle_mcp", 1.05),
        (f"{p}_ring_2_joint", "ring_mcp", 1.05),
        (f"{p}_little_2_joint", "little_mcp", 1.05),
        (f"{p}_thumb_3_joint", "thumb_flex", 0.40),
        (f"{p}_thumb_4_joint", "thumb_flex", 0.60),
    ]
    return {"actuated": actuated, "mimic": mimic}


class RerunPreview:
    """Rerun preview backend for G1 + Inspire control.

    Parameters
    ----------
    urdf_path
        URDF to load for the 3D view. Defaults to
        ``g1_body29_inspire_hand.urdf`` (full G1 body with the Inspire
        right + left hand kinematics embedded). Pass
        ``DEFAULT_G1_URDF`` for the simpler ``g1_body29_hand14.urdf``
        if you don't want the per-finger preview.
    spawn
        Launch the Rerun viewer subprocess. Set False to connect to an
        existing viewer or to record to file instead.
    application_id
        Rerun recording name.
    config_path
        G1 yaml — used to seed leg/waist joints at their default pose.
    """

    def __init__(
        self,
        *,
        urdf_path: "str | Path | None" = None,
        spawn: bool = True,
        application_id: str = "g1_inspire_preview",
        config_path: "str | Path | None" = None,
    ) -> None:
        self.urdf_path = Path(urdf_path) if urdf_path else Path(DEFAULT_G1_INSPIRE_URDF)
        self.mesh_dir = self.urdf_path.parent
        self.config = Config(config_path or DEFAULT_G1_YAML)

        rr.init(application_id, spawn=spawn)
        rr.log("/", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        # Load the URDF — Pinocchio gives us both model (kinematic) and
        # visual_model (mesh geometries with constant per-link offsets).
        # buildModelFromUrdf gives us a fixed-base model (29 DOF for the
        # G1 body; the hand14 URDF adds nothing extra at the actuated layer).
        self.robot = pin.RobotWrapper.BuildFromURDF(
            str(self.urdf_path), str(self.mesh_dir)
        )
        self.model = self.robot.model
        self.data = self.robot.data
        self.visual_model = self.robot.visual_model
        self.visual_data = self.robot.visual_data

        self._lock = threading.Lock()
        self._q = pin.neutral(self.model)
        self._seed_defaults()

        # Index lookup: joint-name -> q-vector position (nq layout). Used by
        # update_arm to write the 14 arm joints into the right slots.
        self._joint_q_idx = {
            self.model.names[j]: self.model.joints[j].idx_q
            for j in range(1, self.model.njoints)  # joint 0 is "universe"
        }

        self._left_arm_joint_names = [
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ]
        self._right_arm_joint_names = [
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]

        # Validate arm joints exist (helps catch URDF mismatches early).
        for n in self._left_arm_joint_names + self._right_arm_joint_names:
            if n not in self._joint_q_idx:
                raise RuntimeError(f"joint {n!r} not found in URDF {self.urdf_path}")

        # Build per-side hand-joint mapping IF the URDF has finger joints.
        # When the URDF is the simple hand14 variant the dicts stay empty
        # and update_hand only logs scalars.
        self._hand_actuated_idx: dict = {"left": {}, "right": {}}
        self._hand_mimic_idx: dict = {"left": [], "right": []}
        for side in ("left", "right"):
            spec = _hand_joint_specs(side)
            if all(n in self._joint_q_idx for n in spec["actuated"].values()):
                self._hand_actuated_idx[side] = {
                    canon: self._joint_q_idx[urdf] for canon, urdf in spec["actuated"].items()
                }
                self._hand_mimic_idx[side] = [
                    (self._joint_q_idx[driven], src_canon, mult)
                    for driven, src_canon, mult in spec["mimic"]
                    if driven in self._joint_q_idx
                ]

        self._log_static_meshes()
        self._update_link_transforms(self._q)

    # ------------------------------------------------------------------
    # Public update hooks (called by G1Arm / InspireHand)
    # ------------------------------------------------------------------

    def update_arm(self, q14: np.ndarray) -> None:
        """Update the 3D view from a 14-DOF arm joint target."""
        q14 = np.asarray(q14, dtype=np.float64).reshape(-1)
        if q14.size != 14:
            raise ValueError(f"q14 must have length 14, got {q14.shape}")
        with self._lock:
            for i, name in enumerate(self._left_arm_joint_names):
                self._q[self._joint_q_idx[name]] = float(q14[i])
            for i, name in enumerate(self._right_arm_joint_names):
                self._q[self._joint_q_idx[name]] = float(q14[i + 7])
            self._update_link_transforms(self._q)

    def update_hand(self, side: str, q6: np.ndarray) -> None:
        """Update the 3D hand pose AND log scalar time-series for a 6-DOF
        canonical Inspire command.

        If the URDF doesn't contain finger joints (e.g. the simpler
        ``g1_body29_hand14.urdf``) only the scalar logging fires.
        """
        q6 = np.asarray(q6, dtype=np.float64).reshape(-1)
        if q6.size != 6:
            raise ValueError(f"q6 must have length 6, got {q6.shape}")
        for i, n in enumerate(_CANONICAL_HAND_NAMES):
            rr.log(f"{_HAND_SCALAR_ROOT}/{side}/{n}", rr.Scalars(float(q6[i])))

        actuated = self._hand_actuated_idx.get(side, {})
        if not actuated:
            return

        canon = {name: float(q6[i]) for i, name in enumerate(_CANONICAL_HAND_NAMES)}
        with self._lock:
            for canon_name, q_idx in actuated.items():
                self._q[q_idx] = canon[canon_name]
            for q_idx, src_canon, mult in self._hand_mimic_idx.get(side, []):
                self._q[q_idx] = mult * canon[src_canon]
            self._update_link_transforms(self._q)

    def update_ee_targets(
        self,
        L_tf: "np.ndarray | None" = None,
        R_tf: "np.ndarray | None" = None,
    ) -> None:
        """Log an EE target frame (4x4 SE3) as a coordinate triad."""
        if L_tf is not None:
            self._log_frame(f"{_EE_TARGET_ROOT}/left", np.asarray(L_tf, dtype=np.float64))
        if R_tf is not None:
            self._log_frame(f"{_EE_TARGET_ROOT}/right", np.asarray(R_tf, dtype=np.float64))

    def log_message(self, text: str, level: str = "info") -> None:
        """Log a free-text status line into the recording."""
        rr.log(
            "log/message",
            rr.TextLog(text, level=getattr(rr.TextLogLevel, level.upper(), rr.TextLogLevel.INFO)),
        )

    def close(self) -> None:
        # Rerun has no explicit close on the SDK; viewer keeps running.
        pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _seed_defaults(self) -> None:
        # Map yaml leg/waist defaults into the URDF q layout. The motor-
        # index lists in the yaml match the URDF joint order: left leg
        # (6) then right leg (6) then waist (3), arms last.
        leg_names = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        ]
        waist_names = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
        for i, name in enumerate(leg_names):
            j = self.model.joints[self.model.getJointId(name)]
            self._q[j.idx_q] = float(self.config.default_angles[i])
        for i, name in enumerate(waist_names):
            j = self.model.joints[self.model.getJointId(name)]
            self._q[j.idx_q] = float(self.config.waist_target[i])
        for i, name in enumerate([
            "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint", "left_elbow_joint",
            "left_wrist_roll_joint", "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
        ]):
            j = self.model.joints[self.model.getJointId(name)]
            self._q[j.idx_q] = float(self.config.left_arm_target[i])
        for i, name in enumerate([
            "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint", "right_elbow_joint",
            "right_wrist_roll_joint", "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ]):
            j = self.model.joints[self.model.getJointId(name)]
            self._q[j.idx_q] = float(self.config.right_arm_target[i])

    def _log_static_meshes(self) -> None:
        # One Asset3D per visual geometry, logged once. Per-tick updates only
        # need to push new Transform3D values.
        for gm in self.visual_model.geometryObjects:
            mesh_path = str(gm.meshPath)
            entity = f"{_ENTITY_ROOT}/{gm.name}"
            try:
                rr.log(entity, rr.Asset3D(path=mesh_path), static=True)
            except Exception:
                # Mesh load failures (missing file, unsupported format) shouldn't
                # crash the preview — fall back to a small point marker.
                rr.log(entity, rr.Points3D([[0, 0, 0]]), static=True)

    def _update_link_transforms(self, q: np.ndarray) -> None:
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.updateGeometryPlacements(
            self.model, self.data, self.visual_model, self.visual_data, q
        )
        for i, gm in enumerate(self.visual_model.geometryObjects):
            M = self.visual_data.oMg[i]
            entity = f"{_ENTITY_ROOT}/{gm.name}"
            rr.log(
                entity,
                rr.Transform3D(
                    translation=np.asarray(M.translation, dtype=np.float64),
                    mat3x3=np.asarray(M.rotation, dtype=np.float64),
                ),
            )

    def _log_frame(self, entity: str, T: np.ndarray) -> None:
        T = T.reshape(4, 4)
        rr.log(
            entity,
            rr.Transform3D(
                translation=T[:3, 3],
                mat3x3=T[:3, :3],
                axis_length=0.1,
            ),
        )
