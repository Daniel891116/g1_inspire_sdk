"""End-effector admittance controller for the G1 arm.

A small, side-symmetric controller that reads the EE residual wrench via
`G1_29_ArmIK.compute_arm_dynamics()`, removes the hand's gravity bias,
projects the horizontal pelvis-frame force through a damped pseudo-inverse
of the Jacobian (shifted to the hand COM), and returns a joint-velocity
command for one side's 7 DOF.

Pure controller: doesn't read DDS, doesn't write DDS. The caller decides
when to step it, what to do with `qd` (integrate into q_target, feed as a
soft goal into a QP, etc.), and how to send commands.

Designed to drop into:
  * `scripts/test_admittance.py`  -- standalone bench test
  * `scripts/handover_qp.py`      -- replaces the in-QP admittance term
  * any future controller that wants compliant-EE behaviour

Conventions match what's already in this repo:
  * Side layout in 14-DOF arm vectors: left=[0:7], right=[7:14]
  * F_ext "world" frame is pelvis frame
  * Hand mass is subtracted in z: F_world[2] -= -m_hand * g
    (matches the F.z-negative-when-carrying-load convention used in
    handover_qp.py for beta_current)
"""

from __future__ import annotations

import numpy as np
import pinocchio as pin


GRAVITY = 9.81


def damped_pinv(J: np.ndarray, lam: float = 0.05) -> np.ndarray:
    """Damped least-squares pseudo-inverse, stable near singular poses."""
    return J.T @ np.linalg.inv(J @ J.T + (lam * lam) * np.eye(J.shape[0]))


class ArmAdmittance:
    """Per-side admittance step.

    Construct once per side; call `step(q, dq, tau_meas)` each tick.

    Args:
        arm_ik: a `G1_29_ArmIK` instance (or anything with the same
                interface: `R_hand_id`/`L_hand_id`, `reduced_robot.model`,
                `reduced_robot.data`, `compute_arm_dynamics`, and
                `compute_current_ee`).
        side: 'left' or 'right'.
        hand_mass_kg: mass to subtract from the gravity-aligned residual.
                      Pull from `Config.hand_mass_left/right`.
        hand_com_local: hand COM offset (m) from the pinocchio L_ee/R_ee
                        frame, expressed in the local EE frame. The
                        admittance Jacobian is shifted to this point so
                        the arm "yields at the hand," not at the wrist.
        gain: admittance gain (m/s per N of horizontal force).
        deadband_n: ignore ||F_xy|| below this (hard cutoff).
        qdot_max: per-joint velocity cap (rad/s).
        damped_pinv_lambda: damping for the LS pseudo-inverse.
        gravity: m/s^2.
    """

    def __init__(
        self,
        arm_ik,
        *,
        side: str = "right",
        hand_mass_kg: float = 0.0,
        hand_com_local=(0.045, 0.0, 0.0),
        gain: float = 0.005,
        deadband_n: float = 2.0,
        qdot_max: float = 0.3,
        damped_pinv_lambda: float = 0.05,
        gravity: float = GRAVITY,
    ) -> None:
        if side not in ("left", "right"):
            raise ValueError(f"side must be 'left' or 'right', got {side!r}")
        self.ik = arm_ik
        self.side = side
        if side == "right":
            self.frame_id = arm_ik.R_hand_id
            self.joint_slice = slice(7, 14)
        else:
            self.frame_id = arm_ik.L_hand_id
            self.joint_slice = slice(0, 7)

        self.m_hand = float(hand_mass_kg)
        r = np.asarray(hand_com_local, dtype=np.float64).reshape(3)
        self.hand_com_local = r
        self._r_cross = np.array([
            [0.0, -r[2], r[1]],
            [r[2], 0.0, -r[0]],
            [-r[1], r[0], 0.0],
        ])
        self.gain = float(gain)
        self.deadband_n = float(deadband_n)
        self.qdot_max = float(qdot_max)
        self.lam = float(damped_pinv_lambda)
        self.gravity = float(gravity)

    def step(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        tau_meas: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One admittance tick.

        Args:
            q: 14D arm joint positions (left[0:7] + right[7:14])
            dq: 14D arm joint velocities
            tau_meas: 14D measured joint torques

        Returns:
            qd_side: 7D joint velocities for THIS side's arm
            F_world: 3D EE force in pelvis frame (hand weight subtracted;
                     gives you a useful 'how hard is it being pulled'
                     signal for higher-level logic)
            tau_rnea: 14D RNEA torques (caller can hand this to the
                      publisher as `tauff_target` for gravity comp)
        """
        tau_rnea, _, _, F_l_local, F_r_local = self.ik.compute_arm_dynamics(
            q, dq, tau_meas,
        )
        F_local = F_r_local if self.side == "right" else F_l_local

        L_tf, R_tf = self.ik.compute_current_ee(q)
        T_ee = R_tf if self.side == "right" else L_tf
        R = np.asarray(T_ee[:3, :3], dtype=np.float64)

        # Pelvis-frame force; subtract hand's gravity contribution from z.
        F_world = R @ np.asarray(F_local[:3], dtype=np.float64)
        F_world[2] -= -self.m_hand * self.gravity

        # Horizontal pull only, hard deadband.
        F_xy = np.array([F_world[0], F_world[1], 0.0])
        f_norm = float(np.linalg.norm(F_xy[:2]))
        if f_norm < self.deadband_n:
            F_xy[:] = 0.0

        # Jacobian shifted to the hand COM:
        #   v_COM = v_R_ee + omega × r_local
        #   J_lin_COM = J_lin_R_ee - [r]× @ J_ang_R_ee   (all in local frame)
        J_full = pin.computeFrameJacobian(
            self.ik.reduced_robot.model,
            self.ik.reduced_robot.data,
            q,
            self.frame_id,
        )
        J_lin = np.asarray(J_full[:3, self.joint_slice])
        J_ang = np.asarray(J_full[3:6, self.joint_slice])
        J_lin_com = J_lin - self._r_cross @ J_ang

        v_des_local = R.T @ (self.gain * F_xy)
        qd = damped_pinv(J_lin_com, self.lam) @ v_des_local
        qd = np.clip(qd, -self.qdot_max, self.qdot_max)
        return qd, F_world, tau_rnea
