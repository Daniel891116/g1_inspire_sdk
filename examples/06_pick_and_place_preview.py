"""Coordinated arm + Inspire hand pick-and-place gesture (preview / dry-run).

Shows how to drive the right arm via EE-pose targets while timing hand
open/close commands against arm phases — the standard pick-and-place
recipe distilled to one script.

Phases (right arm + right hand):
  1. start at home, hand open
  2. reach down to grasp height (EE z -= 0.15 m), hand stays open
  3. close hand (grasp pose)
  4. lift up (EE z += 0.20 m), hand stays closed
  5. translate sideways (EE y -= 0.20 m), hand stays closed
  6. lower to place height (EE z -= 0.15 m), hand stays closed
  7. open hand (release)
  8. retract to home (z up then back to start)

Watch the Rerun viewer: the arm meshes track EE targets and the four
finger MCPs + thumb track the canonical 6-DOF hand command.

Run::
    python examples/06_pick_and_place_preview.py
"""

from __future__ import annotations

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
    preview = RerunPreview(spawn=True)
    robot = G1InspireRobot(
        arm_side="both",        # the left arm holds its default pose
        hand_side="right",      # only the right hand is wired up
        dry_run=True,
        preview=preview,
    )
    robot.connect()
    robot.initialize()
    robot.start_streaming()
    print("Streaming pick-and-place. Watch the Rerun viewer.")

    # Anchor the trajectory on the *measured* (simulated, in dry-run) EE
    # pose so the first IK target equals where the arm is, not the yaml
    # home — same pattern as the live EE example.
    L_home, R_home = robot.arm.get_ee_pose()

    R_grasp = R_home.copy(); R_grasp[:3, 3] += [0.0, 0.0, -0.01]
    R_lift  = R_home.copy(); R_lift[:3, 3]  += [0.0, 0.0, +0.10]
    R_above_place = R_lift.copy(); R_above_place[:3, 3] += [0.0, -0.20, 0.0]
    R_place = R_above_place.copy(); R_place[:3, 3] += [0.0, 0.0, -0.15]
    R_after_place = R_above_place.copy()

    # Right hand: open vs. a moderately closed grasp pose.
    h_open = HAND_SOFT_LOWER_LIMITS.copy()
    h_grasp = np.array([1.0, 1.0, 1.0, 1.0, 0.8, 0.4])

    # Each phase: (label, R_start, R_end, hand_start, hand_end, duration_s)
    # Left arm holds R_home is not needed — left side is held at its home
    # pose by the publisher loop since we never command a new left target.
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

    rate_hz = 60.0
    period = 1.0 / rate_hz
    for label, Ra, Rb, ha, hb, duration in phases:
        n_steps = max(1, int(duration * rate_hz))
        print(f"  phase: {label} ({duration:.1f}s)")
        for s in range(n_steps + 1):
            t = smoothstep(s / n_steps)
            R_tf = lerp_pose(Ra, Rb, t)
            q_hand = lerp_hand(ha, hb, t)
            robot.step(ee_targets=(L_home, R_tf), q_hand_right=q_hand)
            time.sleep(period)

    robot.shutdown()
    print("Done.")


if __name__ == "__main__":
    main()
