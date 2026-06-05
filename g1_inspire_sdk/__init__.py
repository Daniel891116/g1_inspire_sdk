"""g1_inspire_sdk — modular control interface for the Unitree G1 humanoid
and the Inspire RH56 dexterous hand.

Top-level entry points::

    from g1_inspire_sdk import G1Arm, InspireHand, G1InspireRobot, RerunPreview

For a quick joint trajectory preview::

    preview = RerunPreview()
    robot = G1InspireRobot(dry_run=True, preview=preview)
    robot.connect()
    robot.initialize()
    robot.start_streaming()
    # ... robot.step(q_arm=..., q_hand_right=...) ...
"""

from .g1_arm import G1Arm
from .inspire_hand import (
    InspireHand,
    DEFAULT_LEFT_IP,
    DEFAULT_RIGHT_IP,
    SOFT_LOWER_LIMITS as HAND_SOFT_LOWER_LIMITS,
    SOFT_HIGHER_LIMITS as HAND_SOFT_HIGHER_LIMITS,
    JOINT_NAMES as HAND_JOINT_NAMES,
)
from .robot import G1InspireRobot
from .preview import RerunPreview
from ._dds import init_dds
from ._ik import (
    G1_29_ArmIK,
    ArmAdmittance,
    HybridForcePositionController,
    AdaptiveForceController,
    TransportGraspController,
)
from ._config import (
    Config,
    DEFAULT_G1_YAML,
    DEFAULT_G1_URDF,
    DEFAULT_G1_INSPIRE_URDF,
    DEFAULT_INSPIRE_RIGHT_URDF,
    DEFAULT_INSPIRE_LEFT_URDF,
)

__all__ = [
    "G1Arm",
    "InspireHand",
    "G1InspireRobot",
    "RerunPreview",
    "init_dds",
    "Config",
    "G1_29_ArmIK",
    "ArmAdmittance",
    "HybridForcePositionController",
    "AdaptiveForceController",
    "TransportGraspController",
    "DEFAULT_LEFT_IP",
    "DEFAULT_RIGHT_IP",
    "HAND_SOFT_LOWER_LIMITS",
    "HAND_SOFT_HIGHER_LIMITS",
    "HAND_JOINT_NAMES",
    "DEFAULT_G1_YAML",
    "DEFAULT_G1_URDF",
    "DEFAULT_G1_INSPIRE_URDF",
    "DEFAULT_INSPIRE_RIGHT_URDF",
    "DEFAULT_INSPIRE_LEFT_URDF",
]
