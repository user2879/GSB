"""GSB local multi-node shard diagnostics.

The public API is intentionally small.  See :mod:`gsb.spec` for schema loading,
:mod:`gsb.runtime` for the subprocess/TCP runtime, and :mod:`gsb.ledger` for
the durable SQLite state machine.
"""

from .errors import (
    DependencyError,
    DivergenceError,
    GSBError,
    GSBTimeoutError,
    LeaseStateError,
    PortError,
    ProtocolError,
    ScheduleError,
    SnapshotError,
    SpecError,
)
from .arbitration import WaterCandidate, arbitrate_budget, arbitrate_water
from .fixed import FixedErrorModel, WaterLevel, analyze_fixed_error, water_level
from .invalidation import InvalidationPlan, plan_invalidation
from .ledger import Ledger, compare_record_sets, inspect_snapshot, restore_snapshot
from .model import NodeSpec, ShardSpec, Spec
from .runtime import ClusterRuntime
from .samples import generate_samples, sample_digest
from .scheduler import LeaseManager, RunResult, Scheduler
from .spec import load_spec, parse_spec

__all__ = [
    "DependencyError",
    "DivergenceError",
    "FixedErrorModel",
    "GSBError",
    "GSBTimeoutError",
    "LeaseStateError",
    "NodeSpec",
    "InvalidationPlan",
    "LeaseManager",
    "Ledger",
    "PortError",
    "ProtocolError",
    "ScheduleError",
    "ShardSpec",
    "SnapshotError",
    "Scheduler",
    "Spec",
    "SpecError",
    "WaterLevel",
    "WaterCandidate",
    "ClusterRuntime",
    "RunResult",
    "arbitrate_budget",
    "arbitrate_water",
    "analyze_fixed_error",
    "compare_record_sets",
    "generate_samples",
    "inspect_snapshot",
    "load_spec",
    "parse_spec",
    "plan_invalidation",
    "restore_snapshot",
    "sample_digest",
    "water_level",
]
