"""Long-context position-encoding extrapolation diagnostics.

The package is intentionally dependency-light: NumPy is used only for the
deterministic vector/matrix operations.  Public entry points are re-exported
here so an application can embed the diagnostic engine without the CLI.
"""

from .cache import ResultCache, build_stage_keys
from .engine import DiagnosticEngine, run_full_diagnosis
from .errors import BusinessError, Failure
from .loader import load_commands, load_position_config, load_samples
from .models import AttentionSample, PositionEncodingConfig
from .session import InteractiveSession, run_command_sequence

__all__ = [
    "AttentionSample",
    "BusinessError",
    "DiagnosticEngine",
    "Failure",
    "InteractiveSession",
    "PositionEncodingConfig",
    "ResultCache",
    "build_stage_keys",
    "load_commands",
    "load_position_config",
    "load_samples",
    "run_command_sequence",
    "run_full_diagnosis",
]

