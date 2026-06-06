"""RealSense D435i capture + Rerun preview.

Streams color, depth, and (optionally) a coloured point cloud from a
connected D435i into a Rerun viewer. Works against real hardware or as a
synthetic dry-run.

Run::
    # Real camera (auto-discovers first connected D435/D435i)
    python examples/07_camera_preview.py

    # With a coloured point cloud
    python examples/07_camera_preview.py --pointcloud

    # No hardware — synthetic gradient frames, useful for testing the pipeline
    python examples/07_camera_preview.py --dry-run

    # Specific device serial + lower resolution
    python examples/07_camera_preview.py --serial 123456 --width 424 --height 240
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1_inspire_sdk import RealSenseCamera, RerunPreview


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--serial", default=None, help="Specific D435i serial number")
    parser.add_argument(
        "--pointcloud",
        action="store_true",
        help="Compute + stream a coloured point cloud (slower).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use synthetic frames instead of real hardware.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="How long to stream before exiting.",
    )
    args = parser.parse_args()

    preview = RerunPreview(spawn=True)
    cam = RealSenseCamera(
        width=args.width,
        height=args.height,
        fps=args.fps,
        serial=args.serial,
        pointcloud=args.pointcloud,
        dry_run=args.dry_run,
        preview=preview,
    )
    cam.connect()
    print(f"[camera] {cam.device_info}")
    print(f"[camera] depth_scale={cam.depth_scale:.6f} m/unit")
    print(f"[camera] streaming for {args.seconds:.1f}s — watch the Rerun viewer.")

    t_end = time.time() + args.seconds
    frames = 0
    try:
        while time.time() < t_end:
            cam.capture()
            frames += 1
    finally:
        cam.shutdown()

    print(f"[camera] captured {frames} frames.")


if __name__ == "__main__":
    main()
