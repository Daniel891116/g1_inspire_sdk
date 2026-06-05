"""Live control with a Rerun mirror.

Same as ``03_live_control.py`` but also attaches a ``RerunPreview`` so the
viewer shows what's being commanded to the motors in real time. Useful
for trajectory debugging and demos.

Run::
    python examples/04_live_with_preview.py --net eth0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from g1_inspire_sdk import G1InspireRobot, RerunPreview, HAND_SOFT_LOWER_LIMITS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", required=True)
    parser.add_argument("--arm-side", default="both")
    parser.add_argument("--hand-side", default="right")
    args = parser.parse_args()

    preview = RerunPreview(spawn=True)
    robot = G1InspireRobot(
        network_interface=args.net,
        arm_side=args.arm_side,
        hand_side=args.hand_side,
        dry_run=False,
        preview=preview,
    )

    print("Connecting...")
    robot.connect()
    print("Initializing arm + hand.")
    robot.initialize()
    print("Starting publisher.")
    robot.start_streaming()

    try:
        # Move arms to a slight wave pose then return.
        q_home = robot.arm.config.arm_target.astype(np.float64).copy()
        q_wave = q_home.copy()
        q_wave[2] = 0.5     # left shoulder yaw out
        q_wave[9] = -0.5    # right shoulder yaw out

        rate_hz = 60.0
        for q_target in (q_wave, q_home):
            duration = 3.0
            n_steps = int(duration * rate_hz)
            cur = robot.arm.get_joint_state()["q"]
            for s in range(n_steps + 1):
                a = s / n_steps
                t = a * a * (3.0 - 2.0 * a)
                q = (1.0 - t) * cur + t * q_target
                robot.step(q_arm=q, q_hand_right=HAND_SOFT_LOWER_LIMITS)
                time.sleep(1.0 / rate_hz)
        print("Trajectory done.")
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
