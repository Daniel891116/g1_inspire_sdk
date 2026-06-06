# g1_inspire_sdk

Modular control interface for the Unitree G1 humanoid and the Inspire RH56
dexterous hand. Drives the arm + hand together with a clean API and
provides a Rerun-based dry-run preview so you can sanity-check a motion
in 3D before sending it to hardware.

## What's included

- `G1Arm` — DDS lifecycle, 14-DOF joint commands, EE-pose commands via IK,
  optional 250 Hz publisher thread.
- `InspireHand` — DDS publisher for one or both Inspire hands, modbus state
  reader, soft-limit clipping, `open()` / `close_fist()` convenience.
- `RealSenseCamera` — Intel RealSense D435i wrapper. Color + depth + optional
  coloured point cloud. Lazy `pyrealsense2` import, dry-run synthetic frames.
- `G1InspireRobot` — facade owning one arm + one hand + optional camera with
  a single `connect / initialize / start_streaming / shutdown` lifecycle
  and a `step()` helper to drive both subsystems in one call.
- `RerunPreview` — loads the G1 URDF in Pinocchio, mirrors every joint
  command to a Rerun 3D view, plus EE target frames, hand joint scalar
  time-series, and camera RGB / depth / point-cloud streams.
- `dry_run=True` on any of the above skips DDS/modbus/camera I/O so the
  preview shows what *would* have been sent — same code runs against the
  preview before pointing it at hardware.

Intentionally **not** here: Vive trackers, Manus gloves, dex retargeting,
episode recorders, torque-prediction NNs, Wuji hand support, policy
inference. Add them externally if needed; the SDK exposes joint-command
entry points everything else can hang off.

## Requirements

### Always required

- Python ≥ 3.10
- `numpy`, `pyyaml`
- `pinocchio` ≥ 3.0 **built with CasADi bindings** (URDF kinematics + the IK)
- `casadi` (IK NLP)
- `rerun-sdk` ≥ 0.20 (3D preview)

> **Use conda for Pinocchio — pip will not work for the IK.**
> The IK does `from pinocchio import casadi as cpin` ([_ik/ik.py](g1_inspire_sdk/_ik/ik.py#L19)),
> which needs a Pinocchio build that ships the CasADi bindings. Two gotchas:
> 1. On **PyPI** the Pinocchio library is named **`pin`**, *not* `pinocchio`
>    (the PyPI project called `pinocchio` is an unrelated package that stops at
>    v0.4 — this is the "pinocchio 3.0 not available for Python 3.10" symptom).
> 2. Even the correct PyPI wheel (`pin`) **does not include** the
>    `pinocchio.casadi` module, so the IK fails to import under a pip-only
>    install. Only the **conda-forge `pinocchio`** package ships them.
>
> The import name is always `import pinocchio` regardless of how it was installed.

### Recommended: one-command conda environment

The repo ships a pinned, reproducible [environment.yml](environment.yml):

```bash
git clone <your-repo-url> g1_inspire_sdk
cd g1_inspire_sdk
conda env create -f environment.yml
conda activate g1_inspire_sdk
pip install -e .            # installs this SDK + the pure-pip deps
```

This gets you everything for the **dry-run preview** (`dry_run=True`) and the
IK. For live hardware, continue with the section below.

<details>
<summary>Manual conda setup (equivalent, if you don't want environment.yml)</summary>

```bash
conda create -n g1_inspire_sdk -c conda-forge python=3.12 \
    "pinocchio>=3.0" casadi numpy pyyaml
conda activate g1_inspire_sdk
pip install "rerun-sdk>=0.20"
pip install -e .
```
</details>

### Required for live hardware control only

(Not needed if you only want to use `dry_run=True` for previewing.)

- `unitree_sdk2py` — Unitree's DDS bindings, open-source (BSD-3). It depends
  on **CycloneDDS 0.10.x**, which has no self-contained PyPI wheel, so you must
  build the CycloneDDS C library first and point `CYCLONEDDS_HOME` at it
  *before* installing the Python package:

  ```bash
  # 1. Build the CycloneDDS C library (one-time)
  git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
  cd cyclonedds && mkdir build install && cd build
  cmake .. -DCMAKE_INSTALL_PREFIX=../install -DBUILD_IDLC=ON
  cmake --build . --target install
  cd ../..
  export CYCLONEDDS_HOME="$PWD/cyclonedds/install"   # add to ~/.bashrc to persist

  # 2. Install the Python bindings (picks up CYCLONEDDS_HOME)
  pip install cyclonedds==0.10.2
  pip install git+https://github.com/unitreerobotics/unitree_sdk2_python.git
  ```

  Verified working combo: `cyclonedds==0.10.2`. If `import cyclonedds` fails
  with a missing `libddsc` error, `CYCLONEDDS_HOME` wasn't set when `pip
  install cyclonedds` ran — reinstall it with the env var exported.

- `inspire_sdkpy` — Inspire hand bindings (BSD-3, shipped by Unitree):

  ```bash
  # Install from the Inspire SDK checkout you already use with the robot.
  # If you don't have one yet, contact the vendor for the wheel.
  pip install /path/to/inspire_hand_sdk
  ```

- `pymodbus` — used by the Inspire FSR zero-out. Pulled in via the
  `[hand]` extra:

  ```bash
  pip install "g1_inspire_sdk[hand]"
  ```

### Required for the RealSense camera only

(Not needed for arm/hand-only workflows or `RealSenseCamera(dry_run=True)`.)

- `pyrealsense2` — Intel RealSense SDK Python bindings. Pulled in via the
  `[camera]` extra:

  ```bash
  pip install "g1_inspire_sdk[camera]"
  ```

  (Or just `pip install pyrealsense2` if you're not using `pip install -e .`.)

  Lazy-imported inside `RealSenseCamera.connect()` so the SDK still
  imports cleanly without it.

### Optional

- `meshcat` — only needed if you set `Visualization=True` on the raw IK
  solver. The Rerun preview path does not need it.

## Install

Use the conda environment from [Requirements](#recommended-one-command-conda-environment)
above — `conda env create -f environment.yml` then `pip install -e .`. The
`pip install -e .` step makes `g1_inspire_sdk` importable and pulls the pure-pip
deps, but it deliberately does **not** install Pinocchio (that comes from
conda-forge — see the note above). A pip-only install will import fine but the
IK path will fail at `from pinocchio import casadi`.

Alternatively, the example scripts insert the workspace root into `sys.path` on
startup, so you can run them directly without `pip install` (still need the
conda env active for Pinocchio/CasADi).

## Layout

```
g1_inspire_sdk/
├── g1_inspire_sdk/             # the package
│   ├── g1_arm.py               # G1Arm
│   ├── inspire_hand.py         # InspireHand
│   ├── robot.py                # G1InspireRobot
│   ├── preview.py              # RerunPreview
│   ├── camera.py               # RealSenseCamera (D435i)
│   ├── _common/                # command_helper, remote_controller, weighted_moving_filter
│   ├── _config/                # Config loader + g1.yaml
│   └── _ik/                    # G1_29_ArmIK + cached reduced-model pickle, ArmAdmittance, force controllers
├── assets/
│   ├── g1/                     # URDF + STL meshes (~52 MB)
│   └── hands/inspire_gen4_hand/  # URDF + GLB/OBJ meshes (~62 MB)
├── examples/
│   ├── 01_preview_joint_trajectory.py
│   ├── 02_preview_ee_trajectory.py
│   ├── 03_live_control.py
│   ├── 04_live_with_preview.py
│   ├── 05_live_ee_control.py
│   ├── 06_pick_and_place_preview.py
│   └── 07_camera_preview.py
├── pyproject.toml
├── .gitignore
└── README.md
```

## Quick start

### Preview a joint trajectory (no hardware, no Unitree SDK)

```python
from g1_inspire_sdk import G1InspireRobot, RerunPreview
import numpy as np

preview = RerunPreview(spawn=True)
robot = G1InspireRobot(arm_side="both", hand_side="right",
                      dry_run=True, preview=preview)
robot.connect()
robot.initialize()
robot.start_streaming()

q = robot.arm.config.arm_target.astype(np.float64).copy()
q[3] = 1.2  # bend left elbow
robot.step(q_arm=q, q_hand_right=np.array([1.0, 1.0, 1.0, 1.0, 0.8, 0.4]))
```

Run the bundled example:

```bash
python examples/01_preview_joint_trajectory.py
```

### Drive the real robot

```python
robot = G1InspireRobot(network_interface="eth0",
                      arm_side="both", hand_side="right")
robot.connect()         # DDS bring-up
robot.initialize()      # zero torque -> default pose -> hold
robot.start_streaming() # 250 Hz publisher

robot.step(ee_targets=(L_tf, R_tf), q_hand_right=hand_q)
# ...
robot.shutdown()        # damping cmd + close
```

`G1InspireRobot` is also a context manager:

```python
with G1InspireRobot("eth0", arm_side="both") as robot:
    robot.initialize()
    robot.start_streaming()
    ...
```

### Add a RealSense camera to the loop

```python
robot = G1InspireRobot(
    network_interface="eth0",
    arm_side="both",
    hand_side="right",
    camera=True,                     # default-configured D435i
    # or: camera={"width": 1280, "height": 720, "fps": 15, "pointcloud": True}
    # or: camera=RealSenseCamera(...)  for full control
    preview=RerunPreview(spawn=True),
)
robot.connect()       # DDS + camera bring-up
robot.initialize()
robot.start_streaming()
frame = robot.capture_camera()
# frame["color"] (HxWx3 BGR), frame["depth"] (HxW uint16, units = cam.depth_scale m)
```

Or use the camera standalone:

```python
from g1_inspire_sdk import RealSenseCamera, RerunPreview

preview = RerunPreview(spawn=True)
with RealSenseCamera(width=640, height=480, fps=30, preview=preview) as cam:
    for _ in range(300):
        cam.capture()    # auto-streams RGB + depth to preview
```

## Control modes

| API | Input | Output | Notes |
|-----|-------|--------|-------|
| `arm.set_joint_targets(q14)` | 14-D joint vec | 250 Hz DDS publish | Left[0:7] + right[7:14] |
| `arm.set_ee_targets(L_tf, R_tf)` | 4×4 each | Runs IK, then joint targets | Cartesian via Pinocchio + IPOPT |
| `hand.set_joint_targets(q_left=, q_right=)` | 6-D canonical each | Inspire DDS publish | Soft-clipped |
| `hand.open() / close_fist()` | — | Inspire DDS publish | Convenience |
| `robot.step(q_arm=, q_hand_right=, ...)` | mix | Drives both subsystems | Pass `ee_targets=(L, R)` instead of `q_arm` for Cartesian |
| `cam.capture()` | — | Returns `{color, depth, points, colors, timestamp}` | Auto-mirrors to preview if attached |
| `robot.capture_camera()` | — | Same dict (or `None` if no camera) | Convenience shortcut |

Joint order for the hand (canonical 6-DOF, soft limits):

| idx | name | low | high |
|-----|------|-----|------|
| 0 | index MCP | 0.00 | 1.47 |
| 1 | middle MCP | 0.00 | 1.47 |
| 2 | ring MCP | 0.00 | 1.47 |
| 3 | little MCP | 0.00 | 1.47 |
| 4 | thumb rotate | 0.30 | 1.20 |
| 5 | thumb flex | 0.14 | 0.56 |

## Dry-run preview

`RerunPreview` is a `g1_body29_hand14.urdf`-driven Rerun scene. Each call
to `arm.set_joint_targets` / `arm.set_ee_targets` runs FK and pushes new
link transforms to the viewer. Hand commands log as 6 scalar streams per
side. EE targets log as coordinate triads on `ee_target/{left,right}`.

In dry-run mode the arm's 250 Hz publisher thread becomes a simulated
publisher that integrates `q_target` toward the streamed command at the
real velocity cap — so a preview trajectory looks close to what the real
arm would actually do, not an instantaneous snap.

To attach a preview to a live session, pass `preview=` on the
constructor — the same hooks fire in both modes.

## Optional controllers

- `ArmAdmittance` — Cartesian admittance controller (subtracts hand
  gravity, damped pseudo-inverse to joint velocity). Used as a building
  block for handover / contact tasks; not in the default control path.
- `HybridForcePositionController` — release-only PID on `|τ|` for the
  Inspire fingers. Use when you want grip-force regulation instead of
  pure position tracking.
- `TransportGraspController` — steady-grasp law for carrying compliant
  objects; grip self-tightens with arm acceleration.

All three are importable from `g1_inspire_sdk`.

## Caveats

- `g1.yaml` ships with `hand_mass_kg = 0.0` placeholders. If you use
  `ArmAdmittance` you'll want to measure your actual hand and update
  `hand_config.models.inspire.mass_kg`.
- The IK solver loads a pre-built reduced-model cache from
  `_ik/g1_29_model_cache.pkl` (shipped in the repo so cold-start IK is
  instant). The from-scratch build path only works if the URDF contains
  the hand sub-joints listed in `ik.py`; the bundled `g1_body29_hand14.urdf`
  does not, so deleting the pickle and re-running on this URDF will fail.
  Either keep the shipped cache or swap to a hand-merged URDF before
  rebuilding.

## License

TODO: add a `LICENSE` file. The two embedded SDKs that need to be installed
separately (`unitree_sdk2py`, `inspire_sdkpy`) are both BSD-3.
