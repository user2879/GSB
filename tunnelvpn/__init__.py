"""Deterministic loopback tunnel lab.

The package intentionally uses only the Python standard library at runtime.
It is a local traffic-shaping and persistence exercise, not a production VPN.
"""

from .errors import ErrorCode, TunnelError

__all__ = ["ErrorCode", "TunnelError"]
__version__ = "1.0.0"
