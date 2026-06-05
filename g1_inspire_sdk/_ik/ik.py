"""G1 arm IK solver (CasADi + Pinocchio + IPOPT) for the 14-DOF arm chain.

Copied from the upstream g1_data_collection project with two changes for
reuse as an SDK module:
  * URDF/mesh paths default to package-absolute (no `cd` dependency).
  * `logging_mp` replaced with stdlib `logging`.
  * `meshcat` import moved into the `Visualization=True` branch so it is
    not a hard dependency for headless control.
"""

import logging
import os
import pickle
from pathlib import Path

import casadi
import numpy as np
import pinocchio as pin
from pinocchio import casadi as cpin

from .._common.weighted_moving_filter import WeightedMovingFilter
from .._config import ASSETS_ROOT

logger = logging.getLogger(__name__)

_CACHE_PATH = Path(__file__).resolve().parent / "g1_29_model_cache.pkl"
_DEFAULT_URDF = ASSETS_ROOT / "g1" / "g1_body29_hand14.urdf"
_DEFAULT_MESH_DIR = ASSETS_ROOT / "g1"


class G1_29_ArmIK:
    """Solves for 14-DOF arm joint angles to reach left/right EE target poses.

    The reduced model locks legs, waist, and finger joints. L_ee and R_ee
    frames are offset 5 cm forward from the wrist-yaw joint to put the IK
    target at the palm centre rather than the wrist mechanism.
    """

    def __init__(
        self,
        urdf_path: "str | Path | None" = None,
        mesh_dir: "str | Path | None" = None,
        Unit_Test: bool = False,
        Visualization: bool = False,
    ):
        np.set_printoptions(precision=5, suppress=True, linewidth=200)

        self.Unit_Test = Unit_Test
        self.Visualization = Visualization

        self.cache_path = str(_CACHE_PATH)
        self.urdf_path = str(urdf_path) if urdf_path is not None else str(_DEFAULT_URDF)
        self.model_dir = str(mesh_dir) if mesh_dir is not None else str(_DEFAULT_MESH_DIR)

        if os.path.exists(self.cache_path) and (not self.Visualization):
            logger.info(f"[G1_29_ArmIK] loading cached robot model: {self.cache_path}")
            self.robot, self.reduced_robot = self.load_cache()
        else:
            logger.info(f"[G1_29_ArmIK] loading URDF: {self.urdf_path}")
            self.robot = pin.RobotWrapper.BuildFromURDF(self.urdf_path, self.model_dir)

            self.mixed_jointsToLockIDs = [
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
                "left_hand_thumb_0_joint",
                "left_hand_thumb_1_joint",
                "left_hand_thumb_2_joint",
                "left_hand_middle_0_joint",
                "left_hand_middle_1_joint",
                "left_hand_index_0_joint",
                "left_hand_index_1_joint",
                "right_hand_thumb_0_joint",
                "right_hand_thumb_1_joint",
                "right_hand_thumb_2_joint",
                "right_hand_index_0_joint",
                "right_hand_index_1_joint",
                "right_hand_middle_0_joint",
                "right_hand_middle_1_joint",
            ]

            # Drop any joint names that aren't actually in the loaded URDF.
            # The hand sub-joints (left_hand_*, right_hand_*) are present in
            # the merged G1+Inspire URDF but absent in g1_body29_hand14.urdf.
            # Without this filter pinocchio raises "two identical indexes"
            # when missing names all collapse to joint id 0 (universe).
            existing_lock = [
                n for n in self.mixed_jointsToLockIDs
                if self.robot.model.existJointName(n)
            ]
            self.reduced_robot = self.robot.buildReducedRobot(
                list_of_joints_to_lock=existing_lock,
                reference_configuration=np.array([0.0] * self.robot.model.nq),
            )

            self.reduced_robot.model.addFrame(
                pin.Frame(
                    "L_ee",
                    self.reduced_robot.model.getJointId("left_wrist_yaw_joint"),
                    pin.SE3(np.eye(3), np.array([0.05, 0, 0]).T),
                    pin.FrameType.OP_FRAME,
                )
            )
            self.reduced_robot.model.addFrame(
                pin.Frame(
                    "R_ee",
                    self.reduced_robot.model.getJointId("right_wrist_yaw_joint"),
                    pin.SE3(np.eye(3), np.array([0.05, 0, 0]).T),
                    pin.FrameType.OP_FRAME,
                )
            )
            if not os.path.exists(self.cache_path):
                self.save_cache()
                logger.info(f"[G1_29_ArmIK] cache saved to {self.cache_path}")

        self.cmodel = cpin.Model(self.reduced_robot.model)
        self.cdata = self.cmodel.createData()

        self.cq = casadi.SX.sym("q", self.reduced_robot.model.nq, 1)
        self.cTf_l = casadi.SX.sym("tf_l", 4, 4)
        self.cTf_r = casadi.SX.sym("tf_r", 4, 4)
        cpin.framesForwardKinematics(self.cmodel, self.cdata, self.cq)

        self.L_hand_id = self.reduced_robot.model.getFrameId("L_ee")
        self.R_hand_id = self.reduced_robot.model.getFrameId("R_ee")

        self.translational_error = casadi.Function(
            "translational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    self.cdata.oMf[self.L_hand_id].translation - self.cTf_l[:3, 3],
                    self.cdata.oMf[self.R_hand_id].translation - self.cTf_r[:3, 3],
                )
            ],
        )
        self.rotational_error = casadi.Function(
            "rotational_error",
            [self.cq, self.cTf_l, self.cTf_r],
            [
                casadi.vertcat(
                    cpin.log3(
                        self.cdata.oMf[self.L_hand_id].rotation @ self.cTf_l[:3, :3].T
                    ),
                    cpin.log3(
                        self.cdata.oMf[self.R_hand_id].rotation @ self.cTf_r[:3, :3].T
                    ),
                )
            ],
        )

        self.opti = casadi.Opti()
        self.var_q = self.opti.variable(self.reduced_robot.model.nq)
        self.var_q_last = self.opti.parameter(self.reduced_robot.model.nq)
        self.param_tf_l = self.opti.parameter(4, 4)
        self.param_tf_r = self.opti.parameter(4, 4)
        self.translational_cost = casadi.sumsqr(
            self.translational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.rotation_cost = casadi.sumsqr(
            self.rotational_error(self.var_q, self.param_tf_l, self.param_tf_r)
        )
        self.regularization_cost = casadi.sumsqr(self.var_q)
        self.smooth_cost = casadi.sumsqr(self.var_q - self.var_q_last)

        self.opti.subject_to(
            self.opti.bounded(
                self.reduced_robot.model.lowerPositionLimit,
                self.var_q,
                self.reduced_robot.model.upperPositionLimit,
            )
        )
        self.opti.minimize(
            50 * self.translational_cost
            + self.rotation_cost
            + 0.02 * self.regularization_cost
            + 0.1 * self.smooth_cost
        )

        opts = {
            "expand": True,
            "detect_simple_bounds": True,
            "calc_lam_p": False,
            "print_time": False,
            "ipopt.sb": "yes",
            "ipopt.print_level": 0,
            "ipopt.max_iter": 10,
            "ipopt.tol": 1e-3,
            "ipopt.acceptable_tol": 5e-3,
            "ipopt.acceptable_iter": 3,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.mu_strategy": "adaptive",
            "ipopt.derivative_test": "none",
            "ipopt.jacobian_approximation": "exact",
        }
        self.opti.solver("ipopt", opts)

        self.init_data: "np.ndarray | None" = None
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
        self.vis = None
        self._ik_call_count = 0
        self._ik_fail_count = 0

        if self.Visualization:
            # Meshcat is only required when Visualization=True — keep the
            # import lazy so headless users don't need it installed.
            import meshcat.geometry as mg
            from pinocchio.visualize import MeshcatVisualizer

            self.vis = MeshcatVisualizer(
                self.reduced_robot.model,
                self.reduced_robot.collision_model,
                self.reduced_robot.visual_model,
            )
            self.vis.initViewer(open=True)
            self.vis.loadViewerModel("pinocchio")
            self.vis.displayFrames(
                True, frame_ids=[107, 108], axis_length=0.15, axis_width=5
            )
            self.vis.display(pin.neutral(self.reduced_robot.model))

            frame_viz_names = ["L_ee_target", "R_ee_target"]
            FRAME_AXIS_POSITIONS = (
                np.array(
                    [[0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]]
                )
                .astype(np.float32)
                .T
            )
            FRAME_AXIS_COLORS = (
                np.array(
                    [
                        [1, 0, 0],
                        [1, 0.6, 0],
                        [0, 1, 0],
                        [0.6, 1, 0],
                        [0, 0, 1],
                        [0, 0.6, 1],
                    ]
                )
                .astype(np.float32)
                .T
            )
            axis_length = 0.1
            axis_width = 20
            for frame_viz_name in frame_viz_names:
                self.vis.viewer[frame_viz_name].set_object(
                    mg.LineSegments(
                        mg.PointsGeometry(
                            position=axis_length * FRAME_AXIS_POSITIONS,
                            color=FRAME_AXIS_COLORS,
                        ),
                        mg.LineBasicMaterial(linewidth=axis_width, vertexColors=True),
                    )
                )

    def save_cache(self):
        data = {
            "robot_model": self.robot.model,
            "reduced_model": self.reduced_robot.model,
        }
        with open(self.cache_path, "wb") as f:
            pickle.dump(data, f)

    def load_cache(self):
        with open(self.cache_path, "rb") as f:
            data = pickle.load(f)
        robot = pin.RobotWrapper()
        robot.model = data["robot_model"]
        robot.data = robot.model.createData()
        reduced_robot = pin.RobotWrapper()
        reduced_robot.model = data["reduced_model"]
        reduced_robot.data = reduced_robot.model.createData()
        return robot, reduced_robot

    def compute_current_ee(self, q: np.ndarray):
        """Forward-kinematics the L/R end-effectors at joint config `q`.

        Returns:
            L_tf, R_tf : (4,4) homogeneous SE3 in the robot pelvis frame.
        """
        model = self.reduced_robot.model
        data = self.reduced_robot.data
        pin.forwardKinematics(model, data, np.asarray(q, dtype=np.float64))
        pin.updateFramePlacements(model, data)
        return (
            data.oMf[self.L_hand_id].homogeneous.copy(),
            data.oMf[self.R_hand_id].homogeneous.copy(),
        )

    def compute_arm_dynamics(
        self,
        q: np.ndarray,
        dq: np.ndarray,
        tau_meas: np.ndarray,
    ):
        """RNEA joint torques + per-side EE wrenches from measured state.

        Returns (tau_rnea, F_ee_l, F_ee_r, F_ee_l_ext, F_ee_r_ext). The `_ext`
        variants have the model-predicted contribution subtracted so they
        approach zero when nothing touches the hand.
        """
        model = self.reduced_robot.model
        data = self.reduced_robot.data

        a_zero = np.zeros(model.nv)
        tau_rnea = pin.rnea(model, data, q, dq, a_zero)

        J_at_L = pin.computeFrameJacobian(model, data, q, self.L_hand_id)
        J_at_R = pin.computeFrameJacobian(model, data, q, self.R_hand_id)

        J_l = J_at_L[:, :7]
        J_r = J_at_R[:, 7:]
        J_l_T_pinv = np.linalg.pinv(J_l.T)
        J_r_T_pinv = np.linalg.pinv(J_r.T)

        F_ee_l = J_l_T_pinv @ tau_meas[:7]
        F_ee_r = J_r_T_pinv @ tau_meas[7:]
        F_ee_l_ext = J_l_T_pinv @ (tau_meas[:7] - tau_rnea[:7])
        F_ee_r_ext = J_r_T_pinv @ (tau_meas[7:] - tau_rnea[7:])

        return tau_rnea, F_ee_l, F_ee_r, F_ee_l_ext, F_ee_r_ext

    def solve_ik(
        self,
        left_wrist,
        right_wrist,
        current_lr_arm_motor_q=None,
        current_lr_arm_motor_dq=None,
    ):
        self._ik_call_count += 1

        if self.init_data is None or (
            current_lr_arm_motor_q is not None
            and np.max(np.abs(self.init_data - current_lr_arm_motor_q)) > 0.5
        ):
            self.init_data = (
                current_lr_arm_motor_q.copy()
                if current_lr_arm_motor_q is not None
                else np.zeros(self.reduced_robot.model.nq)
            )
        self.opti.set_initial(self.var_q, self.init_data)

        if self.Visualization:
            self.vis.viewer["L_ee_target"].set_transform(left_wrist)
            self.vis.viewer["R_ee_target"].set_transform(right_wrist)

        self.opti.set_value(self.param_tf_l, left_wrist)
        self.opti.set_value(self.param_tf_r, right_wrist)
        self.opti.set_value(self.var_q_last, self.init_data)

        try:
            self.opti.solve()
            sol_q = self.opti.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                sol_q,
                v,
                np.zeros(self.reduced_robot.model.nv),
            )

            if self.Visualization:
                self.vis.display(sol_q)

            return sol_q, sol_tauff

        except KeyboardInterrupt:
            raise
        except Exception as e:
            msg = str(e)
            if "NonIpopt_Exception_Thrown" in msg or "KeyboardInterrupt" in msg:
                raise KeyboardInterrupt() from e

            self._ik_fail_count += 1
            if self._ik_fail_count % 20 == 0:
                logger.warning(
                    f"[IK] {self._ik_fail_count} convergence failures so far "
                    f"({self._ik_call_count} total calls). Latest: {e}"
                )

            sol_q = self.opti.debug.value(self.var_q)
            self.smooth_filter.add_data(sol_q)
            sol_q = self.smooth_filter.filtered_data

            if current_lr_arm_motor_dq is not None:
                v = current_lr_arm_motor_dq * 0.0
            else:
                v = (sol_q - self.init_data) * 0.0

            self.init_data = sol_q

            sol_tauff = pin.rnea(
                self.reduced_robot.model,
                self.reduced_robot.data,
                sol_q,
                v,
                np.zeros(self.reduced_robot.model.nv),
            )

            if self.Visualization:
                self.vis.display(sol_q)

            return sol_q, sol_tauff
