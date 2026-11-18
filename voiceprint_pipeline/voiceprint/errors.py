"""Business error codes and structured failure records."""
from __future__ import annotations

import traceback
from dataclasses import dataclass, asdict
from typing import Any


class PipelineError(Exception):
    """Base error carrying a stable business code."""

    code = "E_PIPELINE"

    def __init__(self, message: str = "", **context: Any) -> None:
        super().__init__(message or self.code)
        self.context = context


class AudioFormatError(PipelineError):
    code = "E_AUDIO_FORMAT"


class InsufficientFramesError(PipelineError):
    code = "E_INSUFFICIENT_FRAMES"


class BoundarySensitiveError(PipelineError):
    code = "E_BOUNDARY_SENSITIVE"


class ConfigMismatchError(PipelineError):
    code = "E_CONFIG_MISMATCH"


class PairDependencyError(PipelineError):
    code = "E_PAIR_DEPENDENCY"


class SignalShutdown(Exception):
    """Raised inside workers when a graceful shutdown has been requested."""

    code = "E_SIGNAL_SHUTDOWN"


@dataclass(frozen=True)
class FailureRecord:
    item_id: str
    stage: str
    code: str
    exception_type: str
    message: str
    sort_key: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_failure(item_id: str, stage: str, code: str, exc: BaseException) -> FailureRecord:
    return FailureRecord(
        item_id=item_id,
        stage=stage,
        code=code,
        exception_type=type(exc).__name__,
        message=str(exc) or code,
        sort_key=f"{stage}|{item_id}",
    )


def format_traceback(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
