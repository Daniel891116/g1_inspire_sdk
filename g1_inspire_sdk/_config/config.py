import numpy as np
import yaml


class Config:
    """Parsed G1 yaml config. See g1.yaml for field documentation."""

    def __init__(self, file_path) -> None:
        with open(file_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        self.control_dt = config["control_dt"]

        self.lowcmd_topic = config["lowcmd_topic"]
        self.lowstate_topic = config["lowstate_topic"]

        self.leg_joint2motor_idx = config["leg_joint2motor_idx"]
        self.kps = config["kps"]
        self.kds = config["kds"]
        self.default_angles = np.array(config["default_angles"], dtype=np.float32)

        self.waist_joint2motor_idx = config["waist_joint2motor_idx"]
        self.waist_kps = config["waist_kps"]
        self.waist_kds = config["waist_kds"]
        self.waist_target = np.array(config["waist_target"], dtype=np.float32)

        self.left_arm_joint2motor_idx = list(config["left_arm_joint2motor_idx"])
        self.right_arm_joint2motor_idx = list(config["right_arm_joint2motor_idx"])
        self.left_arm_kps = list(config["left_arm_kps"])
        self.right_arm_kps = list(config["right_arm_kps"])
        self.left_arm_kds = list(config["left_arm_kds"])
        self.right_arm_kds = list(config["right_arm_kds"])
        self.left_arm_target = np.array(config["left_arm_target"], dtype=np.float32)
        self.right_arm_target = np.array(config["right_arm_target"], dtype=np.float32)

        self.arm_joint2motor_idx = (
            self.left_arm_joint2motor_idx + self.right_arm_joint2motor_idx
        )
        self.arm_kps = self.left_arm_kps + self.right_arm_kps
        self.arm_kds = self.left_arm_kds + self.right_arm_kds
        self.arm_target = np.concatenate(
            (self.left_arm_target, self.right_arm_target)
        )

        self.arm_cmd_filter_tau = config.get("arm_cmd_filter_tau", 0.30)

        self.left_hand_home_xyz = np.array(
            config.get("left_hand_home_xyz", [0.25, 0.15, 0.25]),
            dtype=np.float64,
        )
        self.right_hand_home_xyz = np.array(
            config.get("right_hand_home_xyz", [0.25, -0.15, 0.25]),
            dtype=np.float64,
        )

        hand_cfg = config.get("hand_config", {}) or {}
        hand_models = hand_cfg.get("models", {}) or {}
        hand_active = hand_cfg.get("active", {}) or {}
        self.hand_model_left = str(hand_active.get("left", "none"))
        self.hand_model_right = str(hand_active.get("right", "none"))

        def _resolve_hand(name: str, side: str) -> tuple:
            if name not in hand_models:
                raise ValueError(
                    f"hand_config.active.{side} = {name!r}, but no entry "
                    f"hand_config.models.{name} was found. "
                    f"Known: {sorted(hand_models)}"
                )
            entry = hand_models[name]
            mass = float(entry.get("mass_kg", 0.0))
            com = np.asarray(
                entry.get("com_local", [0.0, 0.0, 0.0]),
                dtype=np.float64,
            ).reshape(3)
            return mass, com

        if hand_models:
            self.hand_mass_left, self.hand_com_left = _resolve_hand(
                self.hand_model_left, "left"
            )
            self.hand_mass_right, self.hand_com_right = _resolve_hand(
                self.hand_model_right, "right"
            )
        else:
            self.hand_mass_left = 0.0
            self.hand_mass_right = 0.0
            self.hand_com_left = np.zeros(3, dtype=np.float64)
            self.hand_com_right = np.zeros(3, dtype=np.float64)

    def arm_indices_for_side(self, side: str) -> list:
        """Motor-index list driven for the given side ('left'/'right'/'both')."""
        if side == "left":
            return list(self.left_arm_joint2motor_idx)
        if side == "right":
            return list(self.right_arm_joint2motor_idx)
        return list(self.arm_joint2motor_idx)

    def arm_slice_for_side(self, side: str) -> slice:
        """Slice into a 14-DOF arm vector (left 0..6, right 7..13) for the given side."""
        if side == "left":
            return slice(0, 7)
        if side == "right":
            return slice(7, 14)
        return slice(0, 14)
