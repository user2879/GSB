"""Business errors used by the deterministic extraction workbench."""

from __future__ import annotations

from typing import Any


class WorkbenchError(Exception):
    """Base class for errors that map to a stable business code."""

    code = "E_RUNTIME"
    exit_code = 5

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigError(WorkbenchError):
    code = "E_CONFIG_INVALID"
    exit_code = 2


class ModelShapeError(WorkbenchError):
    code = "E_MODEL_SHAPE_MISMATCH"
    exit_code = 3


class AnnotationRangeError(WorkbenchError):
    code = "E_ANNOTATION_RANGE"
    exit_code = 4


class ConfigMismatchError(WorkbenchError):
    code = "E_CONFIG_MISMATCH"
    exit_code = 6


class SnapshotError(WorkbenchError):
    code = "E_SNAPSHOT_INVALID"
    exit_code = 7


def failure_from_exception(
    sentence_id: str,
    stage: str,
    exc: BaseException,
    *,
    injection_ids: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Convert an exception into the report's stable failure record shape."""

    code = exc.code if isinstance(exc, WorkbenchError) else "E_RUNTIME"
    return {
        "sentence_id": sentence_id,
        "stage": stage,
        "code": code,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "injection_ids": list(injection_ids),
    }
