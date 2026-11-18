"""Local SQLite migrations with a deterministic fixed-point precision ledger."""

from .errors import MigrationError
from .analysis import compute_error_report, derivation_graph_payload
from .fixedpoint import QuantizationSpec, QuantizedValue, quantize_decimal
from .manifest import Manifest, MigrationStep, load_manifest, validate_manifest
from .migrations import MigrationRunner, MigrationRunResult, perturb_scale
from .replay import replay_snapshot
from .snapshot import SnapshotManager, restore_snapshot
from .verify import verify_drift

__all__ = [
    "Manifest",
    "MigrationError",
    "MigrationRunResult",
    "MigrationRunner",
    "MigrationStep",
    "QuantizationSpec",
    "QuantizedValue",
    "SnapshotManager",
    "compute_error_report",
    "derivation_graph_payload",
    "load_manifest",
    "perturb_scale",
    "quantize_decimal",
    "replay_snapshot",
    "restore_snapshot",
    "validate_manifest",
    "verify_drift",
]

__version__ = "0.1.0"
