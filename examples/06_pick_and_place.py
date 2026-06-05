"""Coordinated live arm + Inspire hand pick-and-place gesture.

Shows how to drive the right arm via EE-pose targets while timing hand
open/close commands against arm phases — the standard pick-and-place
recipe distilled to one script, running on actual hardware.

WARNING: this script actually drives the robot. Before running it:
  * Hang the robot from the rigging or have an e-stop in reach.
  * Make sure the workspace below the right hand is clear — the
    default trajectory lowers the EE by ``--grasp-depth`` then
    translates by ``--translate-y`` then lowers again.
  * Confirm the network interface name (default below assumes ``eth0``).

Phases (right arm + right hand):
  1. start at home, hand open
  2. reach down to grasp height (EE z -= grasp_depth), hand stays open
  3. close hand (grasp pose)
  4. lift up (EE z += lift), hand stays closed
  5. translate sideways (EE y -= translate_y), hand stays closed
  6. lower to place height (EE z -= grasp_depth), hand stays closed
  7. open hand (release)
  8. retract up + return home

Run::
    python examples/06_pick_and_place_preview.py --net eth0
    python examples/06_pick_and_place_preview.py --net eth0 --preview
    python examples/06_pick_and_place_preview.py --net eth0 \\
        --grasp-depth 0.05 --lift 0.15 --translate-y 0.20
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from g1_inspire_sdk import (
    G1InspireRobot,
    HAND_SOFT_LOWER_LIMITS,
    RerunPreview,
)


def lerp_pose(T_a: np.ndarray, T_b: np.ndarray, t: float) -> np.ndarray:
    """Translation-only SE3 interpolation (rotation held at T_a)."""
    T = T_a.copy()
    T[:3, 3] = (1.0 - t) * T_a[:3, 3] + t * T_b[:3, 3]
    return T


def lerp_hand(q_a: np.ndarray, q_b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * q_a + t * q_b


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
        "--grasp-depth",
        type=float,
        default=0.01,
        help="How far to lower the EE for grasp + place (metres). Default 0.01.",
    )
    parser.add_argument(
        "--lift",
        type=float,
        default=0.10,
        help="How far to lift the EE between grasp and translate (metres).",
    )
    parser.add_argument(
        "--translate-y",
        type=float,
        default=0.10,
        help="How far to translate sideways (metres, -y direction).",
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
        # Anchor the trajectory on the measured EE pose so the first IK
        # target equals where the arm actually is — no startup jerk.
        L_home, R_home = robot.arm.get_ee_pose()
        print(f"  Left EE home xyz:  {L_home[:3, 3]}")
        print(f"  Right EE home xyz: {R_home[:3, 3]}")

        R_grasp = R_home.copy(); R_grasp[:3, 3] += [0.0, 0.0, -args.grasp_depth]
        R_lift  = R_home.copy(); R_lift[:3, 3]  += [0.0, 0.0, +args.lift]
        R_above_place = R_lift.copy(); R_above_place[:3, 3] += [0.0, -args.translate_y, 0.0]
        R_place = R_above_place.copy(); R_place[:3, 3] += [0.0, 0.0, -args.grasp_depth]
        R_after_place = R_above_place.copy()

        h_open = HAND_SOFT_LOWER_LIMITS.copy()
        h_grasp = np.array([1.0, 1.0, 1.0, 1.0, 0.8, 0.4])

        # Each phase: (label, R_start, R_end, hand_start, hand_end, duration_s)
        # The left arm is pinned at L_home every tick so it doesn't drift.
        phases = [
            ("reach down",     R_home,         R_grasp,        h_open,  h_open,  2.5),
            ("grasp",          R_grasp,        R_grasp,        h_open,  h_grasp, 1.0),
            ("lift",           R_grasp,        R_lift,         h_grasp, h_grasp, 2.0),
            ("translate",      R_lift,         R_above_place,  h_grasp, h_grasp, 2.5),
            ("lower to place", R_above_place,  R_place,        h_grasp, h_grasp, 2.0),
            ("release",        R_place,        R_place,        h_grasp, h_open,  1.0),
            ("retract up",     R_place,        R_after_place,  h_open,  h_open,  2.0),
            ("return home",    R_after_place,  R_home,         h_open,  h_open,  3.0),
        ]

        period = 1.0 / max(args.rate, 1.0)
        for label, Ra, Rb, ha, hb, duration in phases:
            n_steps = max(1, int(duration * args.rate))
            print(f"  phase: {label} ({duration:.1f}s)")
            for s in range(n_steps + 1):
                t = smoothstep(s / n_steps)
                R_tf = lerp_pose(Ra, Rb, t)
                q_hand = lerp_hand(ha, hb, t)
                robot.step(ee_targets=(L_home, R_tf), q_hand_right=q_hand)
                time.sleep(period)

        print("Trajectory done. Sending damping cmd.")
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
