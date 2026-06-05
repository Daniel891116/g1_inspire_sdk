"""Dry-run an end-effector (Cartesian) trajectory.

The arm reaches in xyz space; the IK solver inside ``G1Arm`` picks joint
angles each tick. EE target frames are mirrored to the Rerun viewer so
you can see both the requested target and the arm pose that tracks it.

Run::
    python examples/02_preview_ee_trajectory.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pinocchio as pin

from g1_inspire_sdk import G1InspireRobot, RerunPreview


def make_pose(xyz, R=None) -> np.ndarray:
    T = np.eye(4)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    if R is not None:
        T[:3, :3] = R
    return T


def main() -> None:
    preview = RerunPreview(spawn=True)
    robot = G1InspireRobot(arm_side="both", dry_run=True, preview=preview)
    robot.connect()
    robot.initialize()
    robot.start_streaming()
    print("Streaming dry-run EE trajectory. Watch the Rerun viewer.")

    # Trace a horizontal arc with each hand. The y-axis swings between the
    # home y and a wider/narrower offset; z and x track a small lift.
    L_home, R_home = robot.arm.default_ee_targets()
    L_home_xyz = L_home[:3, 3].copy()
    R_home_xyz = R_home[:3, 3].copy()

    duration = 8.0
    rate_hz = 50.0
    n_steps = int(duration * rate_hz)
    for i in range(n_steps + 1):
        t = i / n_steps
        phase = 2.0 * np.pi * t
        # Lift + spread arc.
        L_xyz = L_home_xyz + np.array(
            [0.05 * np.sin(phase), 0.10 * np.sin(0.5 * phase), 0.08 * (1.0 - np.cos(phase))]
        )
        R_xyz = R_home_xyz + np.array(
            [0.05 * np.sin(phase), -0.10 * np.sin(0.5 * phase), 0.08 * (1.0 - np.cos(phase))]
        )
        L_tf = make_pose(L_xyz)
        R_tf = make_pose(R_xyz)
        robot.step(ee_targets=(L_tf, R_tf))
        time.sleep(1.0 / rate_hz)

    robot.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
