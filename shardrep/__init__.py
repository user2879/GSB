"""Public API for the local shard replica repair middleware."""

from .canonical import canonical_json_bytes, sha256_object
from .errors import (
    CycleError,
    IncrementalMismatch,
    IntegrityError,
    ReplayError,
    ShardRepairError,
    SnapshotError,
    SpecValidationError,
)
from .ledger import Ledger
from .models import BlockDecl, LogicalShard, NodeDecl, RelationDecl, ShardDecl, Spec
from .packing import PackStore, PackTask, build_pack_tasks, compare_artifact_sets
from .propagation import PropagationResult, propagate_invalidation
from .registry import InjectionRecord, ReplicaRecord, ReplicaRegistry
from .repair import RepairPlan, apply_and_report, block_replica_digest, plan_repair
from .scheduler import PackScheduler, ScheduleEntry, ScheduleResult
from .snapshot import SnapshotManager, compare_replay
from .spec import compute_block_checksum, load_spec, stable_topological_order

__all__ = [
    "BlockDecl",
    "CycleError",
    "IncrementalMismatch",
    "InjectionRecord",
    "IntegrityError",
    "Ledger",
    "LogicalShard",
    "NodeDecl",
    "PackScheduler",
    "PackStore",
    "PackTask",
    "PropagationResult",
    "RelationDecl",
    "ReplayError",
    "RepairPlan",
    "ReplicaRecord",
    "ReplicaRegistry",
    "ScheduleEntry",
    "ScheduleResult",
    "ShardDecl",
    "ShardRepairError",
    "SnapshotError",
    "SnapshotManager",
    "Spec",
    "SpecValidationError",
    "apply_and_report",
    "block_replica_digest",
    "build_pack_tasks",
    "canonical_json_bytes",
    "compare_artifact_sets",
    "compare_replay",
    "compute_block_checksum",
    "load_spec",
    "plan_repair",
    "propagate_invalidation",
    "sha256_object",
    "stable_topological_order",
]

