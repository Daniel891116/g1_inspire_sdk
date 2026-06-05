"""Live EE-pose control of the G1 arm.

Drives the arm by streaming Cartesian (4x4 SE3) targets instead of raw
joint angles. The IK solver inside ``G1Arm`` picks joint angles each
tick; the 250 Hz publisher streams them to the motors.

WARNING: this script actually drives the robot. Before running it:
  * Hang the robot from the rigging or have an e-stop in reach.
  * Make sure the arms have room to move ~10 cm in xyz around the
    default home pose.
  * Confirm the network interface name (default below assumes ``eth0``).

The example:
  1. Connects to DDS, brings up arm + hand.
  2. Runs the safe startup (zero torque -> move to default -> hold).
  3. Reads the *current* EE pose with FK and uses it as the trajectory
     start so the first IK target equals where the arm physically is
     (no startup jerk).
  4. Lifts both hands 8 cm, holds, returns to start. Rotation held fixed.
  5. Sends a damping cmd and exits.

Run::
    python examples/05_live_ee_control.py --net eth0
    python examples/05_live_ee_control.py --net eth0 --preview

Example::
    # Bare minimum
    python examples/05_live_ee_control.py --net eth0

    # With Rerun mirror — viewer shows the commanded EE frames + arm pose live
    python examples/05_live_ee_control.py --net eth0 --preview

    # Tweak the motion
    python examples/05_live_ee_control.py --net eth0 --lift 0.05 --segment-duration 4.0
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from g1_inspire_sdk import G1InspireRobot, RerunPreview


def lerp_pose(T_a: np.ndarray, T_b: np.ndarray, t: float) -> np.ndarray:
    """Translation-only interpolation between two 4x4 SE3 poses.

    Rotation is held at T_a; for these short bench moves that's safe and
    avoids needing a slerp dependency. For longer motions or pose changes
    swap this for a proper SE3 interpolation.
    """
    T = T_a.copy()
    T[:3, 3] = (1.0 - t) * T_a[:3, 3] + t * T_b[:3, 3]
    return T


def smoothstep(alpha: float) -> float:
    return alpha * alpha * (3.0 - 2.0 * alpha)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--net", required=True, help="DDS network interface (e.g. eth0)")
    parser.add_argument("--arm-side", default="both", choices=["left", "right", "both"])
    parser.add_argument("--hand-side", default="right", choices=["left", "right", "both"])
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Mirror commanded EE targets and arm joints to a Rerun viewer.",
    )
    parser.add_argument(
        "--lift",
        type=float,
        default=0.08,
        help="How far to lift each EE in z (metres). Default 0.08.",
    )
    parser.add_argument(
        "--segment-duration",
        type=float,
        default=3.0,
        help="Seconds for each lift / return segment.",
    )
    parser.add_argument(
        "--hold-duration",
        type=float,
        default=1.0,
        help="Seconds to hold the top of the lift.",
    )
    parser.add_argument(
        "--rate",
        type=float,
        default=60.0,
        help="EE target update rate (Hz). The 250 Hz publisher runs underneath.",
    )
    args = parser.parse_args()

    preview = RerunPreview(spawn=True) if args.preview else None

    robot = G1InspireRobot(
        network_interface=args.net,
        arm_side=args.arm_side,
        hand_side=args.hand_side,
        dry_run=False,
        preview=preview,
    )

    print("Connecting...")
    robot.connect()
    print("Initializing (zero torque -> default pose)...")
    robot.initialize()
    print("Starting 250 Hz publisher.")
    robot.start_streaming()

    try:
        # Read the actual EE pose AFTER initialize so the trajectory starts
        # exactly where the arm is. Using the yaml-defined home as the start
        # would jerk on the first tick if the arm settled even a few mm off.
        L_start, R_start = robot.arm.get_ee_pose()
        print(f"  Left EE start xyz:  {L_start[:3, 3]}")
        print(f"  Right EE start xyz: {R_start[:3, 3]}")

        # Lifted target: same rotation, z + lift.
        lift_vec = np.array([0.0, 0.0, args.lift], dtype=np.float64)
        L_up = L_start.copy(); L_up[:3, 3] += lift_vec
        R_up = R_start.copy(); R_up[:3, 3] += lift_vec

        segments = [
            ("lift",   L_start, R_start, L_up,    R_up,    args.segment_duration),
            ("hold",   L_up,    R_up,    L_up,    R_up,    args.hold_duration),
            ("return", L_up,    R_up,    L_start, R_start, args.segment_duration),
        ]

        period = 1.0 / max(args.rate, 1.0)
        for name, La, Ra, Lb, Rb, duration in segments:
            print(f"  segment: {name} ({duration:.1f}s)")
            n_steps = max(1, int(duration * args.rate))
            for s in range(n_steps + 1):
                t = smoothstep(s / n_steps)
                L_tf = lerp_pose(La, Lb, t)
                R_tf = lerp_pose(Ra, Rb, t)
                robot.step(ee_targets=(L_tf, R_tf))
                time.sleep(period)

        print("Trajectory done. Sending damping cmd.")
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
