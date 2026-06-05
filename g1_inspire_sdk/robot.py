"""Unified G1 + Inspire facade.

Owns one ``G1Arm`` and one ``InspireHand``, plus an optional
``RerunPreview``. Provides a single connect / initialize / shutdown
lifecycle and a convenience step() that drives both subsystems together.

Use this when you want to control the arm and hand together with a single
object and a single dry-run / live-mode switch.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .g1_arm import G1Arm
from .inspire_hand import InspireHand, DEFAULT_LEFT_IP, DEFAULT_RIGHT_IP


class G1InspireRobot:
    """G1 arm + Inspire hand bundled behind one lifecycle and one preview.

    Parameters
    ----------
    network_interface
        Ethernet interface for DDS. Required in live mode.
    arm_side
        ``"left"``, ``"right"``, or ``"both"`` for the arm. The IK always
        solves both arms but only the chosen side is sent to the motors.
    hand_side
        ``"left"``, ``"right"``, or ``"both"`` for the hand. Independent of
        the arm side so you can preview a right-arm reach with no hand
        attached, for example.
    config_path
        Override the bundled g1.yaml.
    left_ip, right_ip
        Modbus IPs for the Inspire state reader.
    dry_run
        Skip all DDS / modbus IO. Both subsystems still expose state and
        accept commands; the preview shows what would have been sent.
    preview
        Optional ``RerunPreview``. Attached to both subsystems if provided.
    """

    def __init__(
        self,
        network_interface: Optional[str] = None,
        *,
        arm_side: str = "both",
        hand_side: str = "right",
        config_path: "str | Path | None" = None,
        left_ip: str = DEFAULT_LEFT_IP,
        right_ip: str = DEFAULT_RIGHT_IP,
        dry_run: bool = False,
        preview: Optional[object] = None,
    ) -> None:
        self.arm = G1Arm(
            network_interface=network_interface,
            side=arm_side,
            config_path=config_path,
            dry_run=dry_run,
            preview=preview,
        )
        self.hand = InspireHand(
            side=hand_side,
            left_ip=left_ip,
            right_ip=right_ip,
            network_interface=network_interface,
            dry_run=dry_run,
            preview=preview,
        )
        self.preview = preview
        self.dry_run = dry_run

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect both subsystems in the safe order (arm first)."""
        self.arm.connect()
        self.hand.connect()

    def initialize(self) -> None:
        """Run the startup sequence on the arm, then open the hand."""
        self.arm.initialize()
        self.hand.initialize()
        if not self.dry_run:
            self.hand.wait_for_state()

    def start_streaming(self) -> None:
        """Start the arm publisher (the hand has no streaming thread)."""
        self.arm.start_streaming()

    def shutdown(self) -> None:
        """Stop streaming, send damping cmd, drop DDS handles."""
        self.arm.shutdown()
        self.hand.shutdown()

    def __enter__(self) -> "G1InspireRobot":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # Combined command convenience
    # ------------------------------------------------------------------

    def step(
        self,
        *,
        q_arm: "np.ndarray | None" = None,
        ee_targets: "tuple[np.ndarray, np.ndarray] | None" = None,
        q_hand_left: "np.ndarray | None" = None,
        q_hand_right: "np.ndarray | None" = None,
    ) -> None:
        """Push one set of commands to whichever subsystems have new targets.

        ``q_arm`` and ``ee_targets`` are mutually exclusive — pass at most
        one. Any ``None`` argument leaves that subsystem's last command in
        place.
        """
        if q_arm is not None and ee_targets is not None:
            raise ValueError("pass q_arm OR ee_targets, not both")
        if q_arm is not None:
            self.arm.set_joint_targets(q_arm)
        elif ee_targets is not None:
            L, R = ee_targets
            self.arm.set_ee_targets(L, R)
            if self.preview is not None:
                fn = getattr(self.preview, "update_ee_targets", None)
                if fn is not None:
                    fn(L, R)
        if q_hand_left is not None or q_hand_right is not None:
            self.hand.set_joint_targets(q_left=q_hand_left, q_right=q_hand_right)

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_state(self) -> dict:
        """Combined state snapshot."""
        out = {"arm": self.arm.get_joint_state()}
        try:
            L, R = self.arm.get_ee_pose()
            out["arm_ee_left"] = L
            out["arm_ee_right"] = R
        except Exception:
            pass
        for s in self.hand.sides:
            out[f"hand_{s}"] = self.hand.get_state(s)
        return out
