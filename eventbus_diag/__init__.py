"""Local event-bus streaming integration diagnostic bench.

The package deliberately keeps its public entry points small.  The command
line interface is available through ``python -m eventbus_diag`` and the
Tkinter workbench through ``eventbus_diag.ui.run_ui``.
"""

from .errors import (
    AckError,
    BusDiagnosticError,
    LeaseError,
    PortError,
    ProtocolTimeoutError,
    ReplayError,
    SpecError,
)
from .models import Channel, Declaration, Event, PlanBatch, Subscriber

__all__ = [
    "AckError",
    "BusDiagnosticError",
    "Channel",
    "Declaration",
    "Event",
    "LeaseError",
    "PlanBatch",
    "PortError",
    "ProtocolTimeoutError",
    "ReplayError",
    "SpecError",
    "Subscriber",
]

__version__ = "1.0.0"
