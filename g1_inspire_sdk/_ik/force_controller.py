"""PID force regulator on top of position tracking (release-only).

    error  = |τ| - tau_des                               # signed: +over, -under
    P      = kp · max(error, 0)                          # only on overshoot side
    I      = ki · clip(∫error dt, 0, offset_max/ki)      # bidirectional
    D      = kd · de/dt
    offset = -clip(P + I + D, 0, offset_max)             # release-only output
    q_cmd  = q_ref + offset

The integrator is bidirectional but bounded to ≥ 0, so it can grow under
overpressure (drives offset more negative = retreat) and can climb back when
the force drops below tau_des (drives offset toward 0 = relax). Anti-windup
at offset_max/ki prevents I from accumulating past what the offset cap can
deliver. The release-only clip on the final output ensures the controller
can never close past q_ref.

Steady-state fixed point: |τ| = tau_des. The loop holds contact at the
desired pressure instead of either fighting the operator or running off it.
"""

import numpy as np


class HybridForcePositionController:
    def __init__(
        self,
        n_joints: int,
        tau_des=0.30,
        kp: float = 0.20,
        ki: float = 1.0,
        kd: float = 0.0,
        offset_max: float = 1.5,
    ):
        # tau_des may be a scalar (broadcast to all joints) or a length-
        # n_joints array (per-joint target). For multi-joint chains like
        # wuji's 4-DOF fingers, the per-joint variant lets the proximal
        # and distal joints have different thresholds derived from a single
        # fingertip force via lever-arm scaling.
        arr = np.asarray(tau_des, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            self.tau_des = float(arr.item())
        elif arr.size == int(n_joints):
            self.tau_des = arr.copy()
        else:
            raise ValueError(
                f"tau_des must be scalar or length-{n_joints}, got length {arr.size}"
            )
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.offset_max = float(offset_max)
        self.integral = np.zeros(int(n_joints), dtype=np.float64)
        self.prev_error = np.zeros(int(n_joints), dtype=np.float64)
        # observability shims:
        #   InspireStepper reads .K / .D for telemetry.
        #   WujiStepper writes to .x / .q_cmd / .q_out at DISABLED_JOINTS_2D
        #   to zero internal admittance state — these are no-ops here.
        self.K = 0.0
        self.D = 0.0
        self.x = np.zeros(int(n_joints), dtype=np.float64)
        self.q_cmd: np.ndarray | None = None
        self.q_out: np.ndarray | None = None

    def reset(self) -> None:
        self.integral[:] = 0.0
        self.prev_error[:] = 0.0

    def step(self, tau, q_ref, q, dt):
        tau = np.asarray(tau, dtype=np.float64)
        if not np.all(np.isfinite(tau)):
            tau = np.where(np.isfinite(tau), tau, 0.0)
        q_ref = np.asarray(q_ref, dtype=np.float64)
        dt = max(float(dt), 1e-6)

        # signed force error on raw tau (no LPF)
        error = np.abs(tau) - self.tau_des

        # bidirectional integrator with release-only anti-windup
        max_int = self.offset_max / max(self.ki, 1e-9)
        self.integral = np.clip(self.integral + error * dt, 0.0, max_int)

        # derivative on error
        deriv = (error - self.prev_error) / dt
        self.prev_error = error.copy()

        # PID retreat (≥ 0); P only on overshoot side
        retreat = (
            self.kp * np.maximum(error, 0.0) + self.ki * self.integral + self.kd * deriv
        )
        offset = -np.clip(retreat, 0.0, self.offset_max)
        return q_ref + offset


class AdaptiveForceController(HybridForcePositionController):
    """Outer loop that adapts tau_des from wrist-force disturbance.

    Inner loop: HybridForcePositionController regulates fingertip force to
    tau_des (release-only).
    Outer loop: drives tau_des so that |F_wrist(t) - F_wrist(0)| tracks
    `disturbance_setpoint`, clamped to [f_des, max_f_des].

    Intuition: f_des is the baseline grip needed to manipulate the compliant
    object; max_f_des is the crush ceiling. If the wrist sees more external
    disturbance than the setpoint (slipping, operator pulling), tau_des is
    increased toward max_f_des. If disturbance is comfortably below the
    setpoint, tau_des relaxes back toward f_des.

    Scalar-only: this adapter assumes a scalar tau_des and scales it
    uniformly. For per-joint targets, use the base class directly.
    """

    def __init__(
        self,
        n_joints: int,
        f_des: float = 0.30,
        max_f_des: float = 1.00,
        disturbance_setpoint: float = 1.0,
        gain: float = 0.5,
        kp: float = 0.20,
        ki: float = 1.0,
        kd: float = 0.0,
        offset_max: float = 1.5,
    ):
        super().__init__(
            n_joints,
            tau_des=float(f_des),
            kp=kp,
            ki=ki,
            kd=kd,
            offset_max=offset_max,
        )
        if not (float(f_des) <= float(max_f_des)):
            raise ValueError(f"f_des ({f_des}) must be <= max_f_des ({max_f_des})")
        self.f_des = float(f_des)
        self.max_f_des = float(max_f_des)
        self.disturbance_setpoint = float(disturbance_setpoint)
        self.gain = float(gain)
        self.wrist_baseline: np.ndarray | None = None
        self.current_f_des = float(f_des)

    def reset(self) -> None:
        super().reset()
        self.wrist_baseline = None
        self.current_f_des = self.f_des
        self.tau_des = self.f_des

    def update_f_des(self, f_wrist, dt) -> float:
        """Snapshot baseline on first call, then run one outer-loop step."""
        f_wrist = np.asarray(f_wrist, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(f_wrist)):
            return self.current_f_des
        if self.wrist_baseline is None:
            self.wrist_baseline = f_wrist.copy()
            return self.current_f_des

        disturbance = float(np.linalg.norm(f_wrist - self.wrist_baseline))
        error = disturbance - self.disturbance_setpoint
        self.current_f_des = float(
            np.clip(
                self.current_f_des + self.gain * error * max(float(dt), 1e-6),
                self.f_des,
                self.max_f_des,
            )
        )
        self.tau_des = self.current_f_des
        return self.current_f_des

    def step(self, tau, q_ref, q, dt, f_wrist=None):
        if f_wrist is not None:
            self.update_f_des(f_wrist, dt)
        return super().step(tau, q_ref, q, dt)


class TransportGraspController:
    """Steady-grasp law for transporting compliant objects.

        Δq^{i+1} = clip_{[0, Δq_max]}( Δq^i + κ·(τ_des - |τ^i|) )
        q_cmd    = q_g + Δq

    q_g is the grasp pose snapshotted when the operator first closes
    the hand around the object (set_grasp). Δq is per-joint closure
    relative to q_g, clipped non-negative (cannot open past the grasp
    pose) and bounded above by Δq_max (crush limit).

    τ_des is the scalar grip target the outer loop chooses, e.g.
    τ* + k_a·||a|| where ||a|| is arm acceleration magnitude — grip
    self-tightens as transport loads the object.

    dt is absorbed into κ (no per-tick scaling), matching the iteration
    form of the paper. The step() signature mirrors
    HybridForcePositionController so the inspire stepper can use this
    as a drop-in inner loop. Until set_grasp is called the controller
    is idle and step() returns q_ref unchanged (passthrough teleop).
    """

    def __init__(
        self,
        n_joints: int,
        tau_des=0.30,
        kappa: float = 0.05,
        delta_q_max=1.5,
    ):
        self.n_joints = int(n_joints)
        self.kappa = float(kappa)
        # tau_des / delta_q_max each accept scalar (broadcast) or a
        # length-n_joints array (per-joint). Wuji uses per-joint tau_des
        # via lever-arm scaling; inspire normally uses a scalar.
        self.tau_des = self._coerce("tau_des", tau_des)
        self.delta_q_max = self._coerce("delta_q_max", delta_q_max)

        self.q_g: np.ndarray | None = None
        self.delta_q = np.zeros(self.n_joints, dtype=np.float64)
        # observability shims (same as HybridForcePositionController):
        # InspireStepper reads .K / .D for telemetry; clutch path writes
        # to .x / .q_cmd / .q_out — no-ops here.
        self.K = 0.0
        self.D = 0.0
        self.x = np.zeros(self.n_joints, dtype=np.float64)
        self.q_cmd: np.ndarray | None = None
        self.q_out: np.ndarray | None = None

    def _coerce(self, name: str, v) -> np.ndarray | float:
        arr = np.asarray(v, dtype=np.float64).reshape(-1)
        if arr.size == 1:
            return float(arr.item())
        if arr.size == self.n_joints:
            return arr.copy()
        raise ValueError(
            f"{name} must be scalar or length-{self.n_joints}, got length {arr.size}"
        )

    def reset(self) -> None:
        self.q_g = None
        self.delta_q[:] = 0.0

    def set_grasp(self, q_g) -> None:
        q_g = np.asarray(q_g, dtype=np.float64).reshape(-1).copy()
        if q_g.size != self.n_joints:
            raise ValueError(
                f"q_g length {q_g.size} != n_joints {self.n_joints}"
            )
        self.q_g = q_g
        self.delta_q[:] = 0.0

    def step(self, tau, q_ref, q, dt):
        if self.q_g is None:
            return np.asarray(q_ref, dtype=np.float64)
        tau = np.asarray(tau, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(tau)):
            tau = np.where(np.isfinite(tau), tau, 0.0)
        tau_mag = np.abs(tau)
        update = self.kappa * (self.tau_des - tau_mag)
        self.delta_q = np.clip(self.delta_q + update, 0.0, self.delta_q_max)
        return self.q_g + self.delta_q
