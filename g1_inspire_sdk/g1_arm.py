"""G1 arm control — modular interface.

Wraps the DDS lifecycle + IK solver behind a clean joint / EE-pose API.
Supports a `dry_run` mode that skips DDS entirely and forwards targets
to a Rerun preview backend so a motion can be visualized before deploy.

Typical lifecycle (live mode):
    arm = G1Arm("eth0", side="both")
    arm.connect()
    arm.initialize()         # zero torque -> move to default -> hold
    arm.start_streaming()    # 250 Hz publisher thread
    while ...:
        arm.set_ee_targets(L_tf, R_tf)   # or set_joint_targets(q14)
    arm.shutdown()

Dry-run mode is identical except `connect()` skips DDS and `set_*`
forwards to the preview instead of streaming to motors.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np
import pinocchio as pin

# unitree_sdk2py is imported lazily inside connect() / _send_cmd so that
# dry-run users can run the preview without the Unitree SDK installed.

from ._common.command_helper import (
    create_damping_cmd,
    create_zero_cmd,
    init_cmd_hg,
    MotorMode,
)
from ._common.remote_controller import RemoteController
from ._config import Config, DEFAULT_G1_YAML
from ._dds import init_dds
from ._ik import G1_29_ArmIK


class G1Arm:
    """Modular G1 14-DOF arm controller.

    Parameters
    ----------
    network_interface
        Ethernet interface for DDS (e.g. ``"eth0"``). Required in live mode,
        ignored when ``dry_run=True``.
    side
        ``"left"``, ``"right"``, or ``"both"`` — which arm(s) to actively
        command. The IK always solves both; only the active side is sent to
        the motors.
    config_path
        Optional yaml override; defaults to the SDK-bundled g1.yaml.
    dry_run
        When True, DDS is never touched. The internal joint state mirrors
        the most recent target so ``get_joint_state()`` and ``get_ee_pose()``
        remain meaningful for preview pipelines.
    preview
        Optional ``RerunPreview`` instance. When attached (live OR dry-run),
        joint targets are mirrored to the preview each tick.
    """

    NUM_ARM_JOINTS = 14

    def __init__(
        self,
        network_interface: Optional[str] = None,
        *,
        side: str = "both",
        config_path: "str | Path | None" = None,
        dry_run: bool = False,
        preview: Optional[object] = None,
    ) -> None:
        if side not in ("left", "right", "both"):
            raise ValueError(f"side must be left/right/both, got {side!r}")
        if not dry_run and network_interface is None:
            raise ValueError(
                "network_interface is required when dry_run=False"
            )

        self.config = Config(config_path or DEFAULT_G1_YAML)
        self.side = side
        self.dry_run = dry_run
        self._network_interface = network_interface
        self._preview = preview

        # Lazily built in connect() so the IK solve / URDF load happens
        # together with hardware bring-up rather than at import time.
        self.arm_ik: "G1_29_ArmIK | None" = None
        self.remote_controller = RemoteController()

        self._active_motor_indices = self.config.arm_indices_for_side(side)
        self._active_slice = self.config.arm_slice_for_side(side)
        self._arm_motor_indices = list(self.config.arm_joint2motor_idx)

        # IK output LPF state.
        self._arm_q_cmd_filtered: "np.ndarray | None" = None
        self._arm_filter_tau = self.config.arm_cmd_filter_tau
        self._last_ik_t: "float | None" = None

        # Streaming targets — written by `set_joint_targets`, read by the
        # 250 Hz publisher thread (live) or by the preview-only sim loop.
        self._q_target = np.zeros(self.NUM_ARM_JOINTS, dtype=np.float64)
        self._tauff_target = np.zeros(self.NUM_ARM_JOINTS, dtype=np.float64)
        self._target_lock = threading.Lock()

        # Live-mode publisher state.
        self._motor_dt = 1.0 / 250.0
        self._arm_velocity_limit = 20.0
        self._cmd_q_state: "np.ndarray | None" = None
        self._publish_thread: "threading.Thread | None" = None
        self._stop_publish = threading.Event()

        # Live DDS handles — populated in connect() when live. Kept None in
        # dry-run so unitree_sdk2py is never imported.
        self.low_cmd: Any = None
        self.low_state: Any = None
        self._mode_pr = MotorMode.PR
        self._mode_machine = 0
        self._lowcmd_pub: Any = None
        self._lowstate_sub: Any = None
        self._crc: Any = None

        # Dry-run virtual state — populated from defaults on connect().
        self._sim_q = np.zeros(self.NUM_ARM_JOINTS, dtype=np.float64)
        self._sim_dq = np.zeros(self.NUM_ARM_JOINTS, dtype=np.float64)
        self._sim_tau = np.zeros(self.NUM_ARM_JOINTS, dtype=np.float64)

        self._connected = False
        self._streaming = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Bring up DDS (if live) and build the IK solver."""
        if self._connected:
            return

        # IK loads the URDF + builds the CasADi solver — slow on cold start
        # (~5-15 s) but only done once.
        self.arm_ik = G1_29_ArmIK(Unit_Test=False, Visualization=False)

        if self.dry_run:
            self._sim_q = self.config.arm_target.astype(np.float64).copy()
            self._connected = True
            return

        # Hardware bring-up — local imports so the SDK is only required for
        # live mode.
        from unitree_sdk2py.core.channel import (  # noqa: PLC0415
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import (  # noqa: PLC0415
            unitree_hg_msg_dds__LowCmd_,
            unitree_hg_msg_dds__LowState_,
        )
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import (  # noqa: PLC0415
            LowCmd_ as LowCmdHG,
            LowState_ as LowStateHG,
        )

        self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        self.low_state = unitree_hg_msg_dds__LowState_()
        init_dds(self._network_interface)
        self._lowcmd_pub = ChannelPublisher(self.config.lowcmd_topic, LowCmdHG)
        self._lowcmd_pub.Init()
        self._lowstate_sub = ChannelSubscriber(self.config.lowstate_topic, LowStateHG)
        self._lowstate_sub.Init(self._low_state_handler, 10)
        self._wait_for_low_state()
        init_cmd_hg(self.low_cmd, self._mode_machine, self._mode_pr)
        self._connected = True

    def initialize(
        self,
        *,
        zero_torque_duration: float = 3.0,
        move_duration: float = 4.0,
        hold_duration: float = 2.0,
    ) -> None:
        """Run the safe startup sequence: zero torque -> ramp to default -> hold.

        In dry-run mode this just seeds the simulated joint state to the
        default arm pose so the preview matches what the real robot would
        be holding before streaming begins.
        """
        if not self._connected:
            raise RuntimeError("call connect() before initialize()")

        if self.dry_run:
            self._sim_q = self.config.arm_target.astype(np.float64).copy()
            self._sim_dq[:] = 0.0
            self._mirror_to_preview(self._sim_q)
            return

        self._zero_torque_state(zero_torque_duration)
        self._move_to_default_pos(move_duration)
        self._default_pos_state(hold_duration)

    def start_streaming(self, gradual_ramp_seconds: float = 5.0) -> None:
        """Start the 250 Hz publisher thread (live) or sim-loop (dry-run).

        After this point, the streamed q_target is what reaches the motors
        (or the preview). Call ``set_joint_targets`` / ``set_ee_targets``
        to drive the arm.
        """
        if self._streaming:
            return

        # Seed q_target from current measured / simulated q so the first
        # tick doesn't snap the arm anywhere.
        current_q = self._read_q()
        with self._target_lock:
            self._q_target = current_q.copy()
            self._tauff_target = np.zeros(self.NUM_ARM_JOINTS, dtype=np.float64)
        self._cmd_q_state = current_q.copy()
        self._arm_velocity_limit = 20.0 + (10.0 if gradual_ramp_seconds <= 0 else 0.0)

        self._stop_publish.clear()
        target = self._dry_run_loop if self.dry_run else self._live_publisher_loop
        self._publish_thread = threading.Thread(
            target=target,
            name="g1arm-publisher",
            daemon=True,
            args=(gradual_ramp_seconds,),
        )
        self._publish_thread.start()
        self._streaming = True

    def stop_streaming(self, timeout: float = 1.0) -> None:
        if not self._streaming:
            return
        self._stop_publish.set()
        if self._publish_thread is not None:
            self._publish_thread.join(timeout=timeout)
        self._publish_thread = None
        self._streaming = False

    def shutdown(self) -> None:
        """Stop streaming, send damping cmd (live), drop DDS handles."""
        self.stop_streaming()
        if self.dry_run:
            self._connected = False
            return

        if self.low_cmd is not None and self._lowcmd_pub is not None:
            try:
                create_damping_cmd(self.low_cmd)
                self._send_cmd(self.low_cmd)
            except Exception:
                pass
        self._lowcmd_pub = None
        self._lowstate_sub = None
        self._connected = False

    def __enter__(self) -> "G1Arm":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_joint_state(self) -> dict:
        """Return current 14-DOF arm state as ``{q, dq, tau}``."""
        if self.dry_run:
            return {
                "q": self._sim_q.copy(),
                "dq": self._sim_dq.copy(),
                "tau": self._sim_tau.copy(),
            }
        q = np.array(
            [self.low_state.motor_state[i].q for i in self._arm_motor_indices],
            dtype=np.float64,
        )
        dq = np.array(
            [self.low_state.motor_state[i].dq for i in self._arm_motor_indices],
            dtype=np.float64,
        )
        tau = np.array(
            [self.low_state.motor_state[i].tau_est for i in self._arm_motor_indices],
            dtype=np.float64,
        )
        return {"q": q, "dq": dq, "tau": tau}

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Forward-kinematics the L/R EE poses at the current joint config."""
        if self.arm_ik is None:
            raise RuntimeError("call connect() first")
        q = self._read_q()
        return self.arm_ik.compute_current_ee(q)

    def default_ee_targets(self) -> Tuple[np.ndarray, np.ndarray]:
        """4x4 home poses for L/R EE (identity rotation, xyz from yaml)."""
        L = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            np.asarray(self.config.left_hand_home_xyz, dtype=np.float64),
        ).homogeneous
        R = pin.SE3(
            pin.Quaternion(1, 0, 0, 0),
            np.asarray(self.config.right_hand_home_xyz, dtype=np.float64),
        ).homogeneous
        return L, R

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def set_joint_targets(
        self,
        q14: np.ndarray,
        tau_ff14: "np.ndarray | None" = None,
    ) -> None:
        """Stream a 14-DOF joint target (left[0:7] + right[7:14]) to the publisher."""
        q14 = np.asarray(q14, dtype=np.float64).reshape(-1)
        if q14.size != self.NUM_ARM_JOINTS:
            raise ValueError(
                f"expected 14-vec joint target, got shape {q14.shape}"
            )
        if tau_ff14 is None:
            tau = np.zeros(self.NUM_ARM_JOINTS, dtype=np.float64)
        else:
            tau = np.asarray(tau_ff14, dtype=np.float64).reshape(-1)
            if tau.size != self.NUM_ARM_JOINTS:
                raise ValueError(
                    f"expected 14-vec tauff, got shape {tau.shape}"
                )
        with self._target_lock:
            self._q_target = q14.copy()
            self._tauff_target = tau.copy()
        self._mirror_to_preview(q14)

    def set_ee_targets(
        self,
        L_tf: np.ndarray,
        R_tf: np.ndarray,
    ) -> None:
        """IK to ``L_tf`` / ``R_tf`` (4x4 each) and stream the result."""
        if self.arm_ik is None:
            raise RuntimeError("call connect() first")
        L_tf = np.asarray(L_tf, dtype=np.float64).reshape(4, 4)
        R_tf = np.asarray(R_tf, dtype=np.float64).reshape(4, 4)
        q_now = self._read_q()
        dq_now = self._read_dq()
        sol_q, sol_tauff = self.arm_ik.solve_ik(L_tf, R_tf, q_now, dq_now)

        # Output LPF — same dt-aware coefficient as the upstream controller.
        if self._arm_q_cmd_filtered is None:
            self._arm_q_cmd_filtered = q_now.copy()
        now = time.time()
        if self._last_ik_t is None:
            dt = float(self.config.control_dt)
        else:
            dt = max(1e-4, min(0.1, now - self._last_ik_t))
        self._last_ik_t = now
        alpha = dt / (self._arm_filter_tau + dt)
        self._arm_q_cmd_filtered = (
            1.0 - alpha
        ) * self._arm_q_cmd_filtered + alpha * sol_q

        self.set_joint_targets(self._arm_q_cmd_filtered, sol_tauff)

    # ------------------------------------------------------------------
    # Internals — live DDS path
    # ------------------------------------------------------------------

    def _low_state_handler(self, msg: Any) -> None:
        self.low_state = msg
        self._mode_machine = msg.mode_machine
        self.remote_controller.set(msg.wireless_remote)

    def _send_cmd(self, cmd: Any) -> None:
        if self._crc is None:
            from unitree_sdk2py.utils.crc import CRC  # noqa: PLC0415
            self._crc = CRC()
        cmd.crc = self._crc.Crc(cmd)
        self._lowcmd_pub.Write(cmd)

    def _wait_for_low_state(self) -> None:
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)

    def _set_motor_command(
        self, motor_idx: int, q: float, qd: float, kp: float, kd: float, tau: float
    ) -> None:
        motor_cmd = self.low_cmd.motor_cmd[motor_idx]
        motor_cmd.q = q
        motor_cmd.qd = qd
        motor_cmd.kp = kp
        motor_cmd.kd = kd
        motor_cmd.tau = tau

    def _read_q(self) -> np.ndarray:
        if self.dry_run:
            return self._sim_q.copy()
        return np.array(
            [self.low_state.motor_state[i].q for i in self._arm_motor_indices],
            dtype=np.float64,
        )

    def _read_dq(self) -> np.ndarray:
        if self.dry_run:
            return self._sim_dq.copy()
        return np.array(
            [self.low_state.motor_state[i].dq for i in self._arm_motor_indices],
            dtype=np.float64,
        )

    def _clip_to_velocity_limit(
        self, target_q: np.ndarray, velocity_limit: float
    ) -> np.ndarray:
        if self._cmd_q_state is None:
            self._cmd_q_state = self._read_q().copy()
        delta = target_q - self._cmd_q_state
        max_delta = np.max(np.abs(delta)) if delta.size else 0.0
        motion_scale = max_delta / max(velocity_limit * self._motor_dt, 1e-9)
        self._cmd_q_state = self._cmd_q_state + delta / max(motion_scale, 1.0)
        return self._cmd_q_state

    def _live_publisher_loop(self, gradual_ramp_seconds: float) -> None:
        active = self._active_slice
        kps = np.asarray(self.config.arm_kps, dtype=np.float64)[active]
        kds = np.asarray(self.config.arm_kds, dtype=np.float64)[active]
        ramp_start = time.time()
        ramp_active = gradual_ramp_seconds > 0
        ramp_time = max(1e-3, gradual_ramp_seconds)

        while not self._stop_publish.is_set():
            t0 = time.time()
            with self._target_lock:
                q_target = self._q_target.copy()
                tauff_target = self._tauff_target.copy()
            clipped = self._clip_to_velocity_limit(q_target, self._arm_velocity_limit)
            active_q = clipped[active]
            active_tau = tauff_target[active]
            for local_idx, motor_idx in enumerate(self._active_motor_indices):
                self._set_motor_command(
                    motor_idx=motor_idx,
                    q=float(active_q[local_idx]),
                    qd=0.0,
                    kp=float(kps[local_idx]),
                    kd=float(kds[local_idx]),
                    tau=float(active_tau[local_idx]),
                )
            self._send_cmd(self.low_cmd)

            if ramp_active:
                t_elapsed = t0 - ramp_start
                self._arm_velocity_limit = 20.0 + 10.0 * min(1.0, t_elapsed / ramp_time)
                if t_elapsed >= ramp_time:
                    ramp_active = False

            sleep_t = self._motor_dt - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    def _dry_run_loop(self, gradual_ramp_seconds: float) -> None:
        """Sim-publisher: steps `_sim_q` toward `_q_target` at the velocity cap.

        Matches the real publisher's clipping so the preview shows roughly
        the trajectory the real arm would follow, not an instantaneous snap.
        """
        while not self._stop_publish.is_set():
            t0 = time.time()
            with self._target_lock:
                q_target = self._q_target.copy()
            clipped = self._clip_to_velocity_limit(q_target, 30.0)
            self._sim_q = clipped.copy()
            self._sim_dq[:] = 0.0
            self._mirror_to_preview(self._sim_q)
            sleep_t = self._motor_dt - (time.time() - t0)
            if sleep_t > 0:
                time.sleep(sleep_t)

    # ------------------------------------------------------------------
    # Preview hook
    # ------------------------------------------------------------------

    def attach_preview(self, preview: object) -> None:
        self._preview = preview

    def _mirror_to_preview(self, q14: np.ndarray) -> None:
        if self._preview is None:
            return
        fn = getattr(self._preview, "update_arm", None)
        if fn is None:
            return
        try:
            fn(q14)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Live-only startup helpers (ported from the upstream Controller class)
    # ------------------------------------------------------------------

    def _zero_torque_state(self, duration: float) -> None:
        t_end = time.time() + duration
        while time.time() < t_end:
            create_zero_cmd(self.low_cmd)
            self._send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def _move_to_default_pos(self, total_time: float) -> None:
        num_step = int(total_time / self.config.control_dt)
        dof_idx = (
            list(self.config.leg_joint2motor_idx)
            + list(self.config.waist_joint2motor_idx)
            + list(self.config.arm_joint2motor_idx)
        )
        kps = (
            list(self.config.kps)
            + list(self.config.waist_kps)
            + list(self.config.arm_kps)
        )
        kds = (
            list(self.config.kds)
            + list(self.config.waist_kds)
            + list(self.config.arm_kds)
        )
        default_pos = np.concatenate(
            (self.config.default_angles, self.config.waist_target, self.config.arm_target),
            axis=0,
        )
        dof_size = len(dof_idx)
        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        for i in range(num_step):
            alpha = i / num_step
            s = alpha * alpha * (3.0 - 2.0 * alpha)
            arm_state = self.get_joint_state()
            arm_tauff, *_ = self.arm_ik.compute_arm_dynamics(
                arm_state["q"], arm_state["dq"], arm_state["tau"]
            )
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                interpolated_q = init_dof_pos[j] * (1 - s) + default_pos[j] * s
                tauff = 0.0
                arm_j = j - len(self.config.leg_joint2motor_idx) - len(
                    self.config.waist_joint2motor_idx
                )
                if 0 <= arm_j < arm_tauff.size:
                    tauff = float(arm_tauff[arm_j])
                self._set_motor_command(
                    motor_idx=motor_idx,
                    q=float(interpolated_q),
                    qd=0.0,
                    kp=float(kps[j]),
                    kd=float(kds[j]),
                    tau=tauff,
                )
            self._send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)

    def _default_pos_state(self, duration: float) -> None:
        arm_state = self.get_joint_state()
        arm_tauff, *_ = self.arm_ik.compute_arm_dynamics(
            arm_state["q"], arm_state["dq"], arm_state["tau"]
        )
        t_end = time.time() + duration
        while time.time() < t_end:
            for i, motor_idx in enumerate(self.config.leg_joint2motor_idx):
                self._set_motor_command(motor_idx, 0.0, 0.0, 0.0, 8.0, 0.0)
            for i, motor_idx in enumerate(self.config.waist_joint2motor_idx):
                self._set_motor_command(
                    motor_idx,
                    float(self.config.waist_target[i]),
                    0.0,
                    float(self.config.waist_kps[i]),
                    float(self.config.waist_kds[i]),
                    0.0,
                )
            for i, motor_idx in enumerate(self.config.arm_joint2motor_idx):
                self._set_motor_command(
                    motor_idx,
                    float(self.config.arm_target[i]),
                    0.0,
                    float(self.config.arm_kps[i]),
                    float(self.config.arm_kds[i]),
                    float(arm_tauff[i]),
                )
            self._send_cmd(self.low_cmd)
            time.sleep(self.config.control_dt)
