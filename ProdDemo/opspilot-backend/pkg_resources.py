"""
Compatibility shim for `pkg_resources` to provide `get_distribution()` used
by some third-party packages. This uses `importlib.metadata` when available.
"""
from importlib import metadata as _metadata


class _Dist:
    def __init__(self, version: str | None):
        self.version = version or "0.0.0"


def get_distribution(name: str):
    try:
        ver = _metadata.version(name)
    except Exception:
        ver = "0.0.0"
    return _Dist(ver)


__all__ = ["get_distribution"]
