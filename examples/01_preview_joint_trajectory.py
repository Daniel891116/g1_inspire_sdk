"""Dry-run a joint-space trajectory and watch it in Rerun.

No DDS, no robot — just the preview. Useful as the first sanity check for
a new motion before running it on hardware.

Run::
    python examples/01_preview_joint_trajectory.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Make the workspace root importable so this script works without `pip install`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from g1_inspire_sdk import G1InspireRobot, RerunPreview, HAND_SOFT_LOWER_LIMITS


def main() -> None:
    preview = RerunPreview(spawn=True)
    robot = G1InspireRobot(
        arm_side="both",
        hand_side="right",
        dry_run=True,
        preview=preview,
    )
    robot.connect()
    robot.initialize()
    robot.start_streaming()
    print("Streaming dry-run trajectory. Watch the Rerun viewer.")

    # Two-keyframe trajectory: home -> reach up + close hand -> back to home.
    q_home = robot.arm.config.arm_target.astype(np.float64).copy()
    q_reach = q_home.copy()
    q_reach[3] = 1.2           # left elbow
    q_reach[10] = -1.2         # right elbow (mirror)
    q_reach[0] = -0.5          # left shoulder pitch
    q_reach[7] = -0.5          # right shoulder pitch

    h_open = HAND_SOFT_LOWER_LIMITS.copy()
    h_closed = np.array([1.2, 1.2, 1.2, 1.2, 0.9, 0.4])

    waypoints = [
        (q_home, h_open, 1.0),
        (q_reach, h_closed, 2.0),
        (q_reach, h_closed, 1.0),
        (q_home, h_open, 2.0),
    ]

    cur_q = q_home.copy()
    cur_h = h_open.copy()
    rate_hz = 60.0

    for target_q, target_h, duration in waypoints:
        n_steps = max(1, int(duration * rate_hz))
        for s in range(n_steps + 1):
            alpha = s / n_steps
            t = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep
            q = (1.0 - t) * cur_q + t * target_q
            h = (1.0 - t) * cur_h + t * target_h
            robot.step(q_arm=q, q_hand_right=h)
            time.sleep(1.0 / rate_hz)
        cur_q = target_q.copy()
        cur_h = target_h.copy()

    robot.shutdown()
    print("Done. Trajectory finished.")


if __name__ == "__main__":
    main()
