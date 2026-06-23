# inspire_hand_sdk (vendored)

Python bindings for the Inspire RH56 dexterous hand. The importable module is
**`inspire_sdkpy`**; this folder is the distribution package (`inspire_sdkpy`
v1.0.0).

- **Author / origin:** Unitree (`unitree@unitree.com`)
- **License:** BSD-3-Clause
- **Why it's here:** it is not on PyPI, so it is vendored into this repo so a
  fresh machine can install it without hunting for a separate download.

## Install

From the repository root:

```bash
pip install -e third_party/inspire_hand_sdk
```

This is only needed for **live Inspire hand control** — the dry-run preview
(`dry_run=True`) does not import it.

## Dependencies it pulls in

Its `setup.py` installs `cyclonedds==0.10.2`, `numpy`, `PyQt5`, `pyqtgraph`,
`colorcet`, `pymodbus==3.6.9`, and `pyserial`. Note `cyclonedds` needs the
CycloneDDS C library on the system — see the main project README for the
CycloneDDS bring-up steps.
