"""shardcast: deterministic multi-node shard dispatch over loopback JSON lines."""

from .errors import (
    DivergeError,
    FieldError,
    LeaseError,
    PortError,
    ShardcastError,
    SpecError,
    TimeoutError,
)
from .spec import Spec, load_spec, parse_spec

__all__ = [
    "Spec",
    "ShardcastError",
    "SpecError",
    "PortError",
    "TimeoutError",
    "FieldError",
    "LeaseError",
    "DivergeError",
    "load_spec",
    "parse_spec",
]

__version__ = "0.1.0"
