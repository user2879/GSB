"""Chaos engineering fault-injection and reproducible-verification component.

Public API surface covering declaration loading, timeline generation,
injection execution, observation aggregation, ledger save/recovery and replay
verification. Only the Python standard library is required.
"""
from __future__ import annotations

from .engine import ChaosEngine, InjectionResult
from .errors import (
    ChaosError,
    LedgerError,
    ReplayError,
    SpecError,
    TargetError,
    TimelineError,
)
from .ledger import Ledger, LEDGER_FORMAT_VERSION, open_or_replace_ledger
from .observe import ObservationSummary, RequestRecord, aggregate, collect_phase
from .replay import verify_identity, verify_observations, verify_timeline_entries
from .rng import SplitMix64, mix_u64
from .spec import (
    CURRENT_SPEC_VERSION,
    SUPPORTED_INJECTION_TYPES,
    ExperimentSpec,
    load_spec,
    parse_spec,
)
from .timeline import TimelineEntry, build_timeline, timeline_digest, verify_timeline

__all__ = [
    "ChaosEngine",
    "InjectionResult",
    "ChaosError",
    "SpecError",
    "TimelineError",
    "TargetError",
    "ReplayError",
    "LedgerError",
    "Ledger",
    "LEDGER_FORMAT_VERSION",
    "open_or_replace_ledger",
    "ObservationSummary",
    "RequestRecord",
    "aggregate",
    "collect_phase",
    "verify_identity",
    "verify_observations",
    "verify_timeline_entries",
    "SplitMix64",
    "mix_u64",
    "CURRENT_SPEC_VERSION",
    "SUPPORTED_INJECTION_TYPES",
    "ExperimentSpec",
    "load_spec",
    "parse_spec",
    "TimelineEntry",
    "build_timeline",
    "timeline_digest",
    "verify_timeline",
]

__version__ = "1.0.0"
