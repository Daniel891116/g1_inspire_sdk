"""Intel RealSense D435i capture — modular interface.

Wraps the ``pyrealsense2`` pipeline behind a clean ``connect / capture /
shutdown`` API matching the rest of the SDK. ``pyrealsense2`` is imported
lazily inside ``connect()`` so dry-run users can run without it.

Typical usage::

    cam = RealSenseCamera(width=640, height=480, fps=30, preview=preview)
    cam.connect()
    while True:
        frame = cam.capture()
        # frame["color"]: HxWx3 uint8 BGR
        # frame["depth"]: HxW uint16, units are `cam.depth_scale` metres
        # frame["points"], frame["colors"]: optional pointcloud
    cam.shutdown()
"""

from __future__ import annotations

import os
import time
from typing import Any, Optional

import numpy as np


class RealSenseCamera:
    """RealSense D435i wrapper.

    Parameters
    ----------
    width, height, fps
        Color/depth stream dimensions. Both streams use the same size when
        ``align_depth=True`` (the depth frame is reprojected into the color
        sensor's image plane).
    serial
        Optional device serial number. If ``None`` the first connected
        D435/D435i is selected.
    align_depth
        Reproject the depth stream into the color frame so ``depth[v, u]``
        corresponds to the pixel at ``color[v, u]``.
    pointcloud
        Compute a coloured point cloud each capture. Defaults to the value
        of the ``G1_CAMERA_POINTCLOUD`` env var, or ``False``.
    dry_run
        Skip the hardware bring-up. ``capture()`` returns a synthetic
        gradient frame so downstream code (e.g. preview hooks) can be
        exercised without a camera.
    preview
        Optional ``RerunPreview`` instance. When attached, each
        ``capture()`` call also pushes the RGB + depth frame to Rerun.
    """

    def __init__(
        self,
        *,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial: Optional[str] = None,
        align_depth: bool = True,
        pointcloud: Optional[bool] = None,
        dry_run: bool = False,
        preview: Any = None,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.fps = int(fps)
        self._serial = serial
        self._align_depth = bool(align_depth)
        self.dry_run = bool(dry_run)
        self._preview = preview

        if pointcloud is None:
            pointcloud = os.environ.get("G1_CAMERA_POINTCLOUD") == "1"
        self.pointcloud_enabled = bool(pointcloud)

        self.pipe: Any = None
        self.align: Any = None
        self.pc: Any = None
        self.profile: Any = None

        self.depth_scale: float = 0.001  # default for D435i (mm per unit)
        self.device_info: dict = {}
        self.intrinsics: dict = {}

        self._connected = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Bring up the pipeline (or seed dry-run state)."""
        if self._connected:
            return

        if self.dry_run:
            self.intrinsics = {
                "width": self.width,
                "height": self.height,
                "fx": 600.0,
                "fy": 600.0,
                "ppx": self.width / 2.0,
                "ppy": self.height / 2.0,
                "model": "synthetic",
                "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0],
            }
            self.device_info = {
                "name": "synthetic D435i (dry-run)",
                "serial": "0",
                "usb": "n/a",
                "firmware": "n/a",
            }
            self._connected = True
            return

        # Lazy import so non-camera users (preview-only, etc.) don't need
        # pyrealsense2 installed.
        import pyrealsense2 as rs  # noqa: PLC0415

        serial = self._serial or self._find_d435i_serial(rs)
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        cfg.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, self.fps)
        self.profile = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.color) if self._align_depth else None
        self.pc = rs.pointcloud() if self.pointcloud_enabled else None

        dev = self.profile.get_device()
        depth_sensor = dev.first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        self.device_info = {
            "name": self._info(dev, rs.camera_info.name) or "unknown device",
            "serial": self._info(dev, rs.camera_info.serial_number) or "unknown",
            "usb": self._info(dev, rs.camera_info.usb_type_descriptor) or "unknown",
            "firmware": self._info(dev, rs.camera_info.firmware_version) or "unknown",
        }
        intr = (
            self.profile.get_stream(rs.stream.color)
            .as_video_stream_profile()
            .get_intrinsics()
        )
        self.intrinsics = {
            "width": intr.width,
            "height": intr.height,
            "fx": intr.fx,
            "fy": intr.fy,
            "ppx": intr.ppx,
            "ppy": intr.ppy,
            "model": str(intr.model),
            "coeffs": list(intr.coeffs),
        }
        self._connected = True

        self._publish_intrinsics_to_preview()

    def shutdown(self) -> None:
        if self.pipe is not None:
            try:
                self.pipe.stop()
            except Exception:
                pass
        self.pipe = None
        self.align = None
        self.pc = None
        self.profile = None
        self._connected = False

    def __enter__(self) -> "RealSenseCamera":
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.shutdown()

    # ------------------------------------------------------------------
    # Capture
    # ------------------------------------------------------------------

    def capture(self) -> dict:
        """Return one frame as ``{color, depth, points, colors, timestamp}``.

        ``color`` is BGR uint8 ``(H, W, 3)``. ``depth`` is uint16
        ``(H, W)`` in units of ``cam.depth_scale`` metres. ``points`` and
        ``colors`` are ``(N, 3)`` and ``(N, 3)`` and empty unless
        ``pointcloud_enabled``.

        Also mirrors RGB + depth to the attached preview if set.
        """
        if not self._connected:
            raise RuntimeError("call connect() before capture()")

        if self.dry_run:
            out = self._synthetic_frame()
            self._mirror_to_preview(out)
            return out

        # Hardware path.
        import pyrealsense2 as rs  # noqa: PLC0415

        frames = self.pipe.wait_for_frames()
        if self.align is not None:
            frames = self.align.process(frames)

        depth = frames.get_depth_frame()
        color = frames.get_color_frame()
        if not depth or not color:
            raise RuntimeError("missing color or depth frame")

        color_img = np.asanyarray(color.get_data())
        depth_img = np.asanyarray(depth.get_data())
        out = {
            "color": color_img,
            "depth": depth_img,
            "timestamp": time.time(),
            "points": np.empty((0, 3), dtype=np.float32),
            "colors": np.empty((0, 3), dtype=np.uint8),
        }

        if self.pc is not None:
            self.pc.map_to(color)
            points = self.pc.calculate(depth)
            verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
            tex = (
                np.asanyarray(points.get_texture_coordinates())
                .view(np.float32)
                .reshape(-1, 2)
            )
            h, w = color_img.shape[:2]
            u = np.clip((tex[:, 0] * w).astype(np.int32), 0, w - 1)
            v = np.clip((tex[:, 1] * h).astype(np.int32), 0, h - 1)
            rgb = color_img[v, u][:, ::-1]  # BGR -> RGB
            valid = verts[:, 2] > 0
            out["points"] = verts[valid]
            out["colors"] = rgb[valid]

        self._mirror_to_preview(out)
        return out

    # ------------------------------------------------------------------
    # Preview hook
    # ------------------------------------------------------------------

    def attach_preview(self, preview: Any) -> None:
        self._preview = preview
        if self._connected:
            self._publish_intrinsics_to_preview()

    def _publish_intrinsics_to_preview(self) -> None:
        if self._preview is None:
            return
        fn = getattr(self._preview, "set_camera_intrinsics", None)
        if fn is None:
            return
        try:
            fn(self.intrinsics)
        except Exception:
            pass

    def _mirror_to_preview(self, frame: dict) -> None:
        if self._preview is None:
            return
        fn = getattr(self._preview, "update_camera", None)
        if fn is not None:
            try:
                fn(
                    color=frame["color"],
                    depth=frame["depth"],
                    depth_scale=self.depth_scale,
                )
            except Exception:
                pass
        if self.pointcloud_enabled and frame.get("points", np.empty(0)).size:
            pc_fn = getattr(self._preview, "update_pointcloud", None)
            if pc_fn is not None:
                try:
                    pc_fn(points=frame["points"], colors=frame["colors"])
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _info(dev: Any, key: Any) -> "str | None":
        try:
            if dev.supports(key):
                return dev.get_info(key)
        except RuntimeError:
            pass
        return None

    @staticmethod
    def _find_d435i_serial(rs: Any) -> str:
        ctx = rs.context()
        for dev in ctx.query_devices():
            name = RealSenseCamera._info(dev, rs.camera_info.name) or ""
            if "D435" in name:
                return dev.get_info(rs.camera_info.serial_number)
        devs = ", ".join(
            f"{RealSenseCamera._info(dev, rs.camera_info.name) or 'unknown'}@"
            f"{RealSenseCamera._info(dev, rs.camera_info.serial_number) or 'unknown'}"
            for dev in ctx.query_devices()
        ) or "(none)"
        raise RuntimeError(f"no D435/D435i device found. Connected: {devs}")

    def _synthetic_frame(self) -> dict:
        """Gradient-color + radial-depth synthetic frame for dry-run testing."""
        h, w = self.height, self.width
        xs = np.linspace(0, 255, w, dtype=np.uint8)
        ys = np.linspace(0, 255, h, dtype=np.uint8)
        gx, gy = np.meshgrid(xs, ys)
        color = np.stack([gx, gy, np.full_like(gx, 128)], axis=-1)
        # Radial depth: closer at center, farther at edges (1000-3000 mm).
        u = np.linspace(-1, 1, w)
        v = np.linspace(-1, 1, h)
        uu, vv = np.meshgrid(u, v)
        radial = np.sqrt(uu ** 2 + vv ** 2)
        depth = (1000.0 + 2000.0 * np.clip(radial, 0.0, 1.0)).astype(np.uint16)
        return {
            "color": color,
            "depth": depth,
            "timestamp": time.time(),
            "points": np.empty((0, 3), dtype=np.float32),
            "colors": np.empty((0, 3), dtype=np.uint8),
        }
