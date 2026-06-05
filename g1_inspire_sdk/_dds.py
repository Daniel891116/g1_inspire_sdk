"""Process-wide DDS factory init. Idempotent.

The ``unitree_sdk2py`` import is deferred so that dry-run users who only
want the preview can run without the Unitree SDK installed.
"""

from __future__ import annotations

import threading

_lock = threading.Lock()
_initialized = False
_iface: "str | None" = None


def init_dds(network_interface: "str | None" = None) -> None:
    """Initialize the Unitree DDS factory exactly once per process.

    Call this before constructing any controller that publishes/subscribes
    to ``rt/lowcmd`` / ``rt/inspire_hand/*``. Repeated calls with the same
    interface are no-ops. Conflicting calls raise.

    Raises:
        ImportError: if ``unitree_sdk2py`` is not installed.
    """
    global _initialized, _iface
    with _lock:
        if _initialized:
            if network_interface != _iface:
                raise RuntimeError(
                    f"DDS already initialized on interface {_iface!r}, "
                    f"cannot re-init on {network_interface!r}"
                )
            return
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize  # noqa: PLC0415

        if network_interface is None:
            ChannelFactoryInitialize(0)
        else:
            ChannelFactoryInitialize(0, network_interface)
        _initialized = True
        _iface = network_interface


def is_initialized() -> bool:
    return _initialized
