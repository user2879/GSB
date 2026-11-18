"""Slow-request deterministic stack profiling.

The public API intentionally keeps aggregation free of module-level mutable
state so independent profilers can safely run concurrently.
"""

__version__ = "0.1.0"

from .aggregate import AggregationResult, aggregate_file, aggregate_samples
from .errors import (
    E_DEPTH,
    E_IO,
    E_PARAM,
    E_REPLAY,
    E_SAMPLE,
    E_VERSION,
    SlowProfError,
)
from .models import (
    CURRENT_METRIC_VERSION,
    INTERVAL_BOUNDS,
    SNAPSHOT_SCHEMA_VERSION,
    FrameStat,
    Interval,
    ProfilePolicy,
    Sample,
)
from .ndjson import NDJSONMetadata, load_samples, scan_ndjson
from .ranking import build_ranking, canonical_core_bytes
from .replay import replay_snapshot
from .session import ProfileSession, profile_file, profile_samples
from .snapshot import (
    RestoreOutcome,
    SnapshotStore,
    load_snapshot,
    restore_snapshot,
    save_snapshot,
)

__all__ = [
    "AggregationResult",
    "CURRENT_METRIC_VERSION",
    "E_DEPTH",
    "E_IO",
    "E_PARAM",
    "E_REPLAY",
    "E_SAMPLE",
    "E_VERSION",
    "FrameStat",
    "INTERVAL_BOUNDS",
    "Interval",
    "NDJSONMetadata",
    "ProfilePolicy",
    "ProfileSession",
    "RestoreOutcome",
    "SNAPSHOT_SCHEMA_VERSION",
    "Sample",
    "SlowProfError",
    "SnapshotStore",
    "__version__",
    "aggregate_file",
    "aggregate_samples",
    "build_ranking",
    "canonical_core_bytes",
    "load_samples",
    "load_snapshot",
    "profile_file",
    "profile_samples",
    "replay_snapshot",
    "restore_snapshot",
    "save_snapshot",
    "scan_ndjson",
]
