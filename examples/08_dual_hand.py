"""Drive the G1 arm and BOTH Inspire hands at the same time.

Runs the arm (both sides, held at their measured home pose so they don't
drift) while cycling both hands open <-> grasp in sync. This exercises the
full pipeline: the 250 Hz arm publisher and both per-hand DDS->Modbus
bridges, all at once.

Networking recap (see the README "Control the real robot" + troubleshooting):
  * The arm speaks DDS over ``--net`` to the G1.
  * Each hand is driven by a local DDS->Modbus bridge; the laptop must reach
    each hand's IP over TCP :6000. On the shared-switch setup the two hands
    sit on one L2 segment and MUST have distinct IPs.

The defaults below match the discovered wiring on this rig:
    left  hand -> 192.168.123.210  (on the G1 network, via the G1 back port)
    right hand -> 192.168.11.210   (on the switch, laptop side)
Override with --left-ip / --right-ip if yours differ. Note these are the
OPPOSITE of the SDK's built-in defaults, so passing them explicitly matters.

WARNING: this drives real hardware.
  * Hang the robot / keep an e-stop in reach.
  * By default the arms are held at home (no Cartesian motion). If you pass
    --arm-bob > 0 the arms gently lower and rise in sync with the grasp;
    keep the workspace under both hands clear.

Run::
    python examples/08_dual_hand.py --net enx6c6e072d3ca1
    python examples/08_dual_hand.py --net enx6c6e072d3ca1 --cycles 5 --arm-bob 0.05
    python examples/08_dual_hand.py --dry-run --preview   # no hardware
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
    parser.add_argument("--net", default="enx6c6e072d3ca1",
                        help="DDS network interface to the G1 (default: enx6c6e072d3ca1).")
    parser.add_argument("--left-ip", default="192.168.123.210",
                        help="Modbus IP of the LEFT hand (default: G1-side 192.168.123.210).")
    parser.add_argument("--right-ip", default="192.168.11.210",
                        help="Modbus IP of the RIGHT hand (default: switch-side 192.168.11.210).")
    parser.add_argument("--arm-side", default="both", choices=["left", "right", "both"])
    parser.add_argument("--hand-side", default="both", choices=["left", "right", "both"])
    parser.add_argument("--cycles", type=int, default=3,
                        help="How many open<->grasp cycles to run per hand.")
    parser.add_argument("--arm-bob", type=float, default=0.0,
                        help="Vertical arm bob amplitude (metres) synced to the grasp. "
                             "0 (default) keeps both arms pinned at home.")
    parser.add_argument("--rate", type=float, default=60.0,
                        help="Command update rate (Hz). The 250 Hz publisher runs underneath.")
    parser.add_argument("--preview", action="store_true",
                        help="Mirror arm joints / EE targets / hand joints to a Rerun viewer.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip all DDS/Modbus IO; run the logic against the preview only.")
    args = parser.parse_args()

    preview = RerunPreview(spawn=True) if args.preview else None

    robot = G1InspireRobot(
        network_interface=args.net,
        arm_side=args.arm_side,
        hand_side=args.hand_side,
        left_ip=args.left_ip,
        right_ip=args.right_ip,
        dry_run=args.dry_run,
        preview=preview,
    )

    print("Connecting...")
    robot.connect()
    print("Initializing (zero torque -> default pose; both hands open + force-cal)...")
    robot.initialize()
    print("Starting 250 Hz publisher.")
    robot.start_streaming()

    try:
        # Anchor arm targets on the measured home pose so the first IK target
        # equals where the arms actually are -- no startup jerk. Both arms are
        # pinned here every tick; only --arm-bob perturbs the z height.
        L_home, R_home = robot.arm.get_ee_pose()
        print(f"  Left EE home xyz:  {L_home[:3, 3]}")
        print(f"  Right EE home xyz: {R_home[:3, 3]}")

        L_low = L_home.copy(); L_low[:3, 3] += [0.0, 0.0, -args.arm_bob]
        R_low = R_home.copy(); R_low[:3, 3] += [0.0, 0.0, -args.arm_bob]

        h_open = HAND_SOFT_LOWER_LIMITS.copy()
        h_grasp = np.array([1.0, 1.0, 1.0, 1.0, 0.8, 0.4])

        # Each phase: (label, hand_start, hand_end, arm_alpha_start, arm_alpha_end, seconds)
        # arm_alpha blends home(0) <-> low(1) so the optional bob tracks the grasp.
        half = [
            ("close both", h_open,  h_grasp, 0.0, 1.0, 1.5),
            ("open both",  h_grasp, h_open,  1.0, 0.0, 1.5),
        ]

        period = 1.0 / max(args.rate, 1.0)
        for c in range(args.cycles):
            print(f"cycle {c + 1}/{args.cycles}")
            for label, ha, hb, a0, a1, duration in half:
                n_steps = max(1, int(duration * args.rate))
                print(f"  phase: {label} ({duration:.1f}s)")
                for s in range(n_steps + 1):
                    t = smoothstep(s / n_steps)
                    q_hand = lerp_hand(ha, hb, t)
                    a = a0 + (a1 - a0) * t
                    L_tf = lerp_pose(L_home, L_low, a)
                    R_tf = lerp_pose(R_home, R_low, a)
                    robot.step(
                        ee_targets=(L_tf, R_tf),
                        q_hand_left=q_hand,
                        q_hand_right=q_hand,
                    )
                    time.sleep(period)

        print("Done. Sending damping cmd.")
    finally:
        robot.shutdown()


if __name__ == "__main__":
    main()
