from pathlib import Path

from .config import Config

PACKAGE_ROOT: Path = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT: Path = PACKAGE_ROOT.parent
ASSETS_ROOT: Path = WORKSPACE_ROOT / "assets"
DEFAULT_G1_YAML: Path = Path(__file__).resolve().parent / "g1.yaml"

DEFAULT_G1_URDF: Path = ASSETS_ROOT / "g1" / "g1_body29_hand14.urdf"
DEFAULT_G1_INSPIRE_URDF: Path = ASSETS_ROOT / "g1" / "g1_body29_inspire_hand.urdf"
DEFAULT_INSPIRE_RIGHT_URDF: Path = (
    ASSETS_ROOT / "hands" / "inspire_gen4_hand" / "inspire_gen4_hand_right.urdf"
)
DEFAULT_INSPIRE_LEFT_URDF: Path = (
    ASSETS_ROOT / "hands" / "inspire_gen4_hand" / "inspire_gen4_hand_left.urdf"
)

__all__ = [
    "Config",
    "PACKAGE_ROOT",
    "WORKSPACE_ROOT",
    "ASSETS_ROOT",
    "DEFAULT_G1_YAML",
    "DEFAULT_G1_URDF",
    "DEFAULT_G1_INSPIRE_URDF",
    "DEFAULT_INSPIRE_RIGHT_URDF",
    "DEFAULT_INSPIRE_LEFT_URDF",
]
