"""Deterministic segmented range-index component for local document collections."""

from .engine import (
    apply_changes,
    diagnose,
    full_scan,
    initialize_session,
    manifest,
    query_index,
)
from .errors import (
    BoundError,
    DocDBError,
    DocumentError,
    LedgerError,
    ReplayError,
    SegmentError,
    SnapshotError,
    StaleError,
    StateError,
)
from .snapshot import create_snapshot, list_snapshots, restore_snapshot
from .validation import load_declarations, validate_declarations

__all__ = [
    "BoundError",
    "DocDBError",
    "DocumentError",
    "LedgerError",
    "ReplayError",
    "SegmentError",
    "SnapshotError",
    "StaleError",
    "StateError",
    "apply_changes",
    "create_snapshot",
    "diagnose",
    "full_scan",
    "initialize_session",
    "list_snapshots",
    "load_declarations",
    "manifest",
    "query_index",
    "restore_snapshot",
    "validate_declarations",
]

__version__ = "1.0.0"
