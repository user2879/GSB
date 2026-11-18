"""tier_mover: deterministic hot/cold tiering and migration component."""

from __future__ import annotations

from .engine import Engine
from .errors import (
    BUSINESS_EXIT_CODE,
    CapacityError,
    IntegrityError,
    ReplayError,
    SpecError,
    TierMoverError,
    VersionConflictError,
)
from .faults import Injection, parse_injection
from .heat import HEAT_TIME_SCALE, heat_value
from .ledger import Ledger
from .logging_util import JsonlLogger
from .manifest import load_manifest
from .planner import derive_cycle_plan, derive_initial_plan
from .replay import replay
from .samplegen import build_sample

__all__ = [
    "Engine",
    "Ledger",
    "JsonlLogger",
    "Injection",
    "parse_injection",
    "load_manifest",
    "build_sample",
    "derive_initial_plan",
    "derive_cycle_plan",
    "replay",
    "heat_value",
    "HEAT_TIME_SCALE",
    "TierMoverError",
    "SpecError",
    "IntegrityError",
    "CapacityError",
    "VersionConflictError",
    "ReplayError",
    "BUSINESS_EXIT_CODE",
]

__version__ = "1.0.0"
