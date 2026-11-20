"""GPU quota and shard scheduling domain component.

Public entry points are re-exported here so an integrator can rely on a stable
package surface instead of reaching into individual modules.
"""
from .errors import (
    E_CYCLE,
    E_DEPEND,
    E_LEDGER,
    E_MEMORY,
    E_QUOTA,
    E_REPLAY,
    E_SPEC,
    Problem,
    SchedulerError,
)
from .engine import SimulationEngine
from .ledger import LEDGER_FORMAT_VERSION, Ledger
from .loader import load_declaration
from .scheduler import SchedulingSession, resume_session

__all__ = [
    "E_CYCLE", "E_DEPEND", "E_LEDGER", "E_MEMORY", "E_QUOTA", "E_REPLAY",
    "E_SPEC", "Problem", "SchedulerError", "SimulationEngine", "Ledger",
    "LEDGER_FORMAT_VERSION", "load_declaration", "SchedulingSession",
    "resume_session",
]
