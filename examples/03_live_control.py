"""Live control of the G1 + Inspire right hand.

WARNING: this script actually drives the robot. Before running it:
  * Hang the robot from the rigging or have an e-stop in reach.
  * Make sure the arms have room to move to the default pose.
  * Confirm the network interface name (default ``eth0``).

The example:
  1. Connects to DDS, brings up arm + hand.
  2. Runs the safe startup (zero torque -> move to default -> hold).
  3. Starts the 250 Hz publisher and the Inspire state reader.
  4. Holds the default pose while opening/closing the right hand twice.
  5. Sends a damping cmd and exits.

Run::
    python examples/03_live_control.py --net eth0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from g1_inspire_sdk import G1InspireRobot, HAND_SOFT_HIGHER_LIMITS, HAND_SOFT_LOWER_LIMITS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", required=True, help="DDS network interface (e.g. eth0)")
    parser.add_argument("--arm-side", default="both", choices=["left", "right", "both"])
    parser.add_argument("--hand-side", default="right", choices=["left", "right", "both"])
    args = parser.parse_args()

    robot = G1InspireRobot(
        network_interface=args.net,
        arm_side=args.arm_side,
        hand_side=args.hand_side,
        dry_run=False,
    )

    print("Connecting...")
    robot.connect()
    print("Initializing (zero torque -> default pose)...")
    robot.initialize()
    print("Starting 250 Hz publisher.")
    robot.start_streaming()

    try:
        # Hold default arm; open + close the right hand twice.
        q_arm = robot.arm.config.arm_target.astype(np.float64).copy()
        for _ in range(2):
            robot.step(q_arm=q_arm, q_hand_right=HAND_SOFT_LOWER_LIMITS)
            time.sleep(1.5)
            robot.step(q_arm=q_arm, q_hand_right=HAND_SOFT_HIGHER_LIMITS)
            time.sleep(1.5)

        print("Done. Sending damping cmd.")
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
