"""Public package API for deterministic fixed-point QoS diagnostics."""

from __future__ import annotations

from .demo import format_demo_summary, run_demo
from .engine import CascadeEngine, EngineRunResult
from .error_propagation import compare_configurations, propagate_fixed_errors
from .errors import LedgerError, PolicyError, ReplayError, SnapshotError
from .fixed import FixedPointSpec, ceil_div
from .generator import (
    DeterministicUdpGenerator,
    decode_packet,
    encode_packet,
    parse_generator_config,
)
from .ledger import DecisionLedger, LedgerRecovery
from .models import (
    DecisionEvent,
    ErrorPropagationResult,
    GeneratorConfig,
    IndistinguishabilityResult,
    Packet,
    PolicySet,
    RunSummary,
)
from .policy import load_policy
from .replay import ReplayVerificationResult, verify_snapshot_replay
from .samples import write_sample_files
from .snapshot import SnapshotDocument, SnapshotManager, SnapshotRestoreResult
from .transport import InMemoryTransport, LoopbackTransport
from .ui_model import DiagnosticView, build_view


__version__ = "1.0.0"

__all__ = [
    "CascadeEngine",
    "DecisionEvent",
    "DecisionLedger",
    "DeterministicUdpGenerator",
    "DiagnosticView",
    "EngineRunResult",
    "ErrorPropagationResult",
    "FixedPointSpec",
    "GeneratorConfig",
    "InMemoryTransport",
    "IndistinguishabilityResult",
    "LedgerError",
    "LedgerRecovery",
    "LoopbackTransport",
    "Packet",
    "PolicyError",
    "PolicySet",
    "ReplayError",
    "ReplayVerificationResult",
    "RunSummary",
    "SnapshotDocument",
    "SnapshotError",
    "SnapshotManager",
    "SnapshotRestoreResult",
    "build_view",
    "ceil_div",
    "compare_configurations",
    "decode_packet",
    "encode_packet",
    "format_demo_summary",
    "load_policy",
    "parse_generator_config",
    "propagate_fixed_errors",
    "run_demo",
    "verify_snapshot_replay",
    "write_sample_files",
]


def launch_ui(*args, **kwargs):
    """Lazily import Tkinter only when the desktop UI is requested."""
    from .ui import launch_ui as _launch_ui

    return _launch_ui(*args, **kwargs)


__all__.append("launch_ui")
