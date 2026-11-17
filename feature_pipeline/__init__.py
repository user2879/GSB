"""Multi-source feature exchange pipeline (v1 fixed-width / v2 JSON / v3 binary).

The package turns three non-equivalent feature exchange formats into one unified
feature table, merges them deterministically and emits auditable artifacts.
"""

from .errors import PipelineError, StageFailure

__all__ = ["PipelineError", "StageFailure"]
__version__ = "1.0.0"
