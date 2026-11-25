"""Local shard-route drift and shard-lease diagnostics.

The package deliberately uses only local deterministic fixtures, loopback TCP
sockets, child processes, and SQLite.  It never contacts a network service.
"""

from .errors import DiagnosticError, ErrorCode
from .invalidation import InvalidationEngine
from .ledger import Ledger, RecoveryReport
from .leases import Lease, LeaseManager, LeaseRequest
from .routing import compare_route, generate_sample_keys
from .scheduler import ProbeScheduler
from .session import DiagnosticSession
from .spec import Event, Spec, load_spec

__all__ = [
    "DiagnosticError",
    "ErrorCode",
    "Event",
    "InvalidationEngine",
    "Lease",
    "LeaseManager",
    "LeaseRequest",
    "Ledger",
    "ProbeScheduler",
    "RecoveryReport",
    "Spec",
    "compare_route",
    "generate_sample_keys",
    "load_spec",
    "DiagnosticSession",
]
__version__ = "0.1.0"
