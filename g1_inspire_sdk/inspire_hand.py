"""Inspire RH56 hand control — modular interface.

Strips the Manus retargeting / torque-prediction NN / live plot stack from
the upstream ``inspire_manus.py`` so the hand can be driven by raw joint
commands. The "canonical" 6-DOF joint order is::

    [index_MCP, middle_MCP, ring_MCP, little_MCP, thumb_rotate, thumb_flex]

with soft limits roughly::

    [0,    1.47]  finger MCPs (closed -> open)
    [0.3,  1.20]  thumb rotate
    [0.14, 0.56]  thumb flex

Hardware uses a different finger order (HW <-> canonical mapping is
``[3, 2, 1, 0, 5, 4]``) and an *angle-set* scalar 0..1000 where 0 = fully
closed and 1000 = fully open. The conversion is handled internally.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Dict, Optional

import numpy as np

# unitree_sdk2py / inspire_sdkpy / pymodbus are imported lazily inside
# connect() / _reader_loop / _calibrate_force_sensor so dry-run users can
# run the preview without those SDKs installed.

from ._dds import init_dds


HW_READ_TO_CANONICAL = np.array([3, 2, 1, 0, 5, 4])
CANONICAL_TO_HW_WRITE = np.array([3, 2, 1, 0, 5, 4])

SOFT_LOWER_LIMITS = np.array([0.0, 0.0, 0.0, 0.0, 0.3, 0.14])
SOFT_HIGHER_LIMITS = np.array([1.47, 1.47, 1.47, 1.47, 1.2, 0.56])
HW_UPPER_LIMITS = np.array([1.47, 1.47, 1.47, 1.47, 1.5, 0.7])

DEFAULT_LEFT_IP = "192.168.11.210"
DEFAULT_RIGHT_IP = "192.168.123.210"

HISTORY_LEN = 7
NUM_HAND_JOINTS = 6

JOINT_NAMES = (
    "index_mcp",
    "middle_mcp",
    "ring_mcp",
    "little_mcp",
    "thumb_rotate",
    "thumb_flex",
)


def _qpos_to_angle_set(qpos_canonical: np.ndarray) -> list:
    pct = 1.0 - np.clip(np.asarray(qpos_canonical, float) / HW_UPPER_LIMITS, 0.0, 1.0)
    return (pct[CANONICAL_TO_HW_WRITE] * 1000).astype(int).tolist()


def _calibrate_force_sensor(ip: str, port: int = 6000, device_id: int = 1, settle: float = 3.0) -> None:
    """Write the FSR zero-out register (1009 <- 1) and wait for it to settle."""
    from pymodbus.client import ModbusTcpClient  # noqa: PLC0415

    client = ModbusTcpClient(ip, port=port)
    if not client.connect():
        raise RuntimeError(f"Cannot connect to Inspire hand at {ip}:{port}")
    try:
        client.write_register(1009, 1, device_id)
        time.sleep(settle)
    finally:
        client.close()


class _SideState:
    """Per-side rolling buffer fed by a background modbus reader."""

    __slots__ = ("latest", "seq", "pos", "vel", "force", "last_pos", "last_t", "last_seq")

    def __init__(self) -> None:
        self.latest: "tuple[np.ndarray, np.ndarray] | None" = None
        self.seq = 0
        self.pos: deque = deque(maxlen=HISTORY_LEN)
        self.vel: deque = deque(maxlen=HISTORY_LEN)
        self.force: deque = deque(maxlen=HISTORY_LEN)
        self.last_pos: "np.ndarray | None" = None
        self.last_t: "float | None" = None
        self.last_seq = -1

    def push_latest(self) -> None:
        """Promote the latest modbus reading into the rolling history."""
        if self.latest is None or self.seq == self.last_seq:
            return
        pos, force = self.latest
        now = time.monotonic()
        vel = (
            (pos - self.last_pos) / max(now - self.last_t, 1e-6)
            if self.last_t is not None
            else np.zeros_like(pos)
        )
        self.pos.append(pos)
        self.vel.append(vel)
        self.force.append(force)
        self.last_pos, self.last_t, self.last_seq = pos, now, self.seq


class InspireHand:
    """Modular Inspire hand controller — one instance handles 1 or 2 sides.

    Parameters
    ----------
    side
        ``"left"``, ``"right"``, or ``"both"``.
    left_ip, right_ip
        Modbus IPs for the force/position state stream. Only read for sides
        you enable.
    network_interface
        Ethernet interface for DDS (forwarded to ``init_dds``). Required
        when ``dry_run=False``.
    dry_run
        Skip DDS / modbus entirely. ``get_state`` returns the last commanded
        joint vector, ``get_force`` returns zeros. ``set_joint_targets``
        forwards to the preview if attached.
    preview
        Optional ``RerunPreview`` to mirror joint targets to.
    """

    def __init__(
        self,
        *,
        side: str = "right",
        left_ip: str = DEFAULT_LEFT_IP,
        right_ip: str = DEFAULT_RIGHT_IP,
        network_interface: Optional[str] = None,
        dry_run: bool = False,
        preview: Optional[object] = None,
    ) -> None:
        if side not in ("left", "right", "both"):
            raise ValueError(f"side must be left/right/both, got {side!r}")
        if not dry_run and network_interface is None:
            raise ValueError("network_interface is required when dry_run=False")

        self.side = side
        self.dry_run = dry_run
        self._left_ip = left_ip
        self._right_ip = right_ip
        self._network_interface = network_interface
        self._preview = preview

        self._sides = [s for s, on in (("left", side in ("left", "both")),
                                       ("right", side in ("right", "both"))) if on]
        self._state: Dict[str, _SideState] = {s: _SideState() for s in self._sides}

        # DDS handles (None in dry-run). Typed as Any so unitree_sdk2py is
        # not required at import time.
        self._publishers: Dict[str, Any] = {s: None for s in self._sides}
        self._cmd: Any = None

        # Last commanded joint vector (canonical order), for state queries
        # in dry-run AND for fallback after a modbus read glitch.
        self._last_target: Dict[str, np.ndarray] = {
            s: SOFT_LOWER_LIMITS.copy() for s in self._sides
        }

        self._reader_threads: list[threading.Thread] = []
        self._stop_readers = threading.Event()
        self._connected = False

    @property
    def sides(self) -> list[str]:
        return list(self._sides)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Spin up state readers, init DDS, build publishers (live mode)."""
        if self._connected:
            return

        if self.dry_run:
            self._connected = True
            return

        # Live-mode imports — kept local so dry-run users can run without
        # unitree_sdk2py / inspire_sdkpy installed.
        from unitree_sdk2py.core.channel import ChannelPublisher  # noqa: PLC0415
        from inspire_sdkpy import inspire_dds, inspire_hand_defaut  # noqa: PLC0415

        init_dds(self._network_interface)

        for s in self._sides:
            ip = self._right_ip if s == "right" else self._left_ip
            lr = "r" if s == "right" else "l"
            t = threading.Thread(
                target=self._reader_loop,
                args=(ip, lr, s),
                name=f"inspire-state-{s}",
                daemon=True,
            )
            t.start()
            self._reader_threads.append(t)
            # Stagger so simultaneous modbus connects on a shared switch
            # don't race for ARP — mirrors the upstream pattern.
            time.sleep(0.3)

        for s in self._sides:
            lr = "r" if s == "right" else "l"
            pub = ChannelPublisher(f"rt/inspire_hand/ctrl/{lr}", inspire_dds.inspire_hand_ctrl)
            pub.Init()
            self._publishers[s] = pub

        self._cmd = inspire_hand_defaut.get_inspire_hand_ctrl()
        self._connected = True

    def initialize(
        self,
        *,
        target: "np.ndarray | None" = None,
        force_set: int = 1000,
        speed_set: int = 250,
        calibrate_force: bool = True,
    ) -> None:
        """Send mode/force/speed init and move to the given joint target.

        Defaults to ``SOFT_LOWER_LIMITS`` (fully open) when ``target`` is None.
        """
        if not self._connected:
            raise RuntimeError("call connect() before initialize()")

        q = SOFT_LOWER_LIMITS.copy() if target is None else np.asarray(target, dtype=np.float64).reshape(-1)
        if q.size != NUM_HAND_JOINTS:
            raise ValueError(f"target must be length-{NUM_HAND_JOINTS}, got {q.shape}")

        if self.dry_run:
            for s in self._sides:
                self._last_target[s] = q.copy()
            self._mirror_to_preview()
            return

        # Set max-force budget.
        self._cmd.force_set = [int(force_set)] * 6
        self._cmd.mode = 0b0100
        self._write_all()
        time.sleep(1.0)

        # Move to target.
        self._cmd.angle_set = _qpos_to_angle_set(q)
        self._cmd.mode = 0b0001
        self._write_all()

        # Set max-speed budget.
        self._cmd.speed_set = [int(speed_set)] * 6
        self._cmd.mode = 0b1000
        self._write_all()
        time.sleep(2.0)

        if calibrate_force:
            ip = self._right_ip if "right" in self._sides else self._left_ip
            _calibrate_force_sensor(ip)

        for s in self._sides:
            self._last_target[s] = q.copy()
        self._mirror_to_preview()

    def wait_for_state(self, min_samples: int = HISTORY_LEN, timeout: float = 5.0) -> None:
        """Block until each enabled side has filled its rolling buffer."""
        if self.dry_run:
            return
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for s in self._sides:
                self._state[s].push_latest()
            if all(len(self._state[s].pos) >= min_samples for s in self._sides):
                return
            time.sleep(0.01)
        raise TimeoutError("Inspire state buffer did not fill before timeout")

    def shutdown(self) -> None:
        """Stop reader threads, drop DDS handles."""
        self._stop_readers.set()
        for t in self._reader_threads:
            t.join(timeout=0.5)
        self._reader_threads.clear()
        self._publishers = {s: None for s in self._sides}
        self._cmd = None
        self._connected = False

    def __enter__(self) -> "InspireHand":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def get_state(self, side: Optional[str] = None) -> dict:
        """Return ``{q, force}`` in canonical 6-DOF order for one side.

        In dry-run, ``q`` is the last commanded target and ``force`` is zeros.
        """
        s = self._resolve_side(side)
        if self.dry_run:
            return {
                "q": self._last_target[s].copy(),
                "force": np.zeros(NUM_HAND_JOINTS, dtype=np.float64),
            }
        st = self._state[s]
        st.push_latest()
        if not st.pos:
            return {
                "q": self._last_target[s].copy(),
                "force": np.zeros(NUM_HAND_JOINTS, dtype=np.float64),
            }
        return {
            "q": np.asarray(st.pos[-1], dtype=np.float64),
            "force": np.asarray(st.force[-1], dtype=np.float64),
        }

    # ------------------------------------------------------------------
    # Commands
    # ------------------------------------------------------------------

    def set_joint_targets(
        self,
        q_left: "np.ndarray | None" = None,
        q_right: "np.ndarray | None" = None,
    ) -> None:
        """Send a 6-DOF canonical joint command to one or both sides.

        Pass ``None`` to leave that side untouched. Targets are soft-clipped
        to ``[SOFT_LOWER_LIMITS, SOFT_HIGHER_LIMITS]`` before writing.
        """
        per_side = {"left": q_left, "right": q_right}
        for s in self._sides:
            q = per_side[s]
            if q is None:
                continue
            q = np.asarray(q, dtype=np.float64).reshape(-1)
            if q.size != NUM_HAND_JOINTS:
                raise ValueError(
                    f"{s} target must be length-{NUM_HAND_JOINTS}, got {q.shape}"
                )
            q = np.clip(q, SOFT_LOWER_LIMITS, SOFT_HIGHER_LIMITS)
            self._last_target[s] = q.copy()

            if self.dry_run:
                continue

            self._cmd.angle_set = _qpos_to_angle_set(q)
            self._cmd.mode = 0b0001
            pub = self._publishers[s]
            if pub is not None and not pub.Write(self._cmd):
                # Lossy DDS write — log once-per-second-ish, don't raise.
                pass

        self._mirror_to_preview()

    def open(self) -> None:
        """Convenience: fully open both enabled sides."""
        q = SOFT_LOWER_LIMITS.copy()
        self.set_joint_targets(
            q_left=q if "left" in self._sides else None,
            q_right=q if "right" in self._sides else None,
        )

    def close_fist(self) -> None:
        """Convenience: fully close both enabled sides."""
        q = SOFT_HIGHER_LIMITS.copy()
        self.set_joint_targets(
            q_left=q if "left" in self._sides else None,
            q_right=q if "right" in self._sides else None,
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _resolve_side(self, side: Optional[str]) -> str:
        if side is None:
            if len(self._sides) != 1:
                raise ValueError(
                    "side argument is required when controlling multiple sides"
                )
            return self._sides[0]
        if side not in self._sides:
            raise ValueError(f"side {side!r} not enabled (enabled: {self._sides})")
        return side

    def _write_all(self) -> None:
        for s in self._sides:
            pub = self._publishers[s]
            if pub is not None:
                pub.Write(self._cmd)

    def _reader_loop(self, ip: str, lr: str, side: str) -> None:
        from inspire_sdkpy import inspire_sdk  # noqa: PLC0415

        handler = inspire_sdk.ModbusDataHandler(network=None, ip=ip, LR=lr, device_id=1)
        time.sleep(0.5)
        buf = self._state[side]
        last_err_t = 0.0
        while not self._stop_readers.is_set():
            try:
                states = handler.read()["states"]
                angle = np.asarray(states["ANGLE_ACT"], float)[HW_READ_TO_CANONICAL]
                forces = np.asarray(states["FORCE_ACT"], float)[HW_READ_TO_CANONICAL]
                buf.latest = ((1.0 - angle / 1000.0) * HW_UPPER_LIMITS, forces)
                buf.seq += 1
            except Exception as e:
                now = time.monotonic()
                if now - last_err_t > 1.0:
                    print(f"[inspire:{side}] modbus read failed: {e}")
                    last_err_t = now
                time.sleep(0.05)

    # ------------------------------------------------------------------
    # Preview hook
    # ------------------------------------------------------------------

    def attach_preview(self, preview: object) -> None:
        self._preview = preview

    def _mirror_to_preview(self) -> None:
        if self._preview is None:
            return
        fn = getattr(self._preview, "update_hand", None)
        if fn is None:
            return
        for s in self._sides:
            try:
                fn(s, self._last_target[s])
            except Exception:
                pass
