"""Stable business error codes shared by every pipeline stage."""

from __future__ import annotations


class WorkbenchError(Exception):
    """Raised for classified business failures.

    ``code`` is one of the ``E_*`` constants and ``stage`` records the pipeline
    stage that detected the problem (load / forward / train / validate /
    snapshot).
    """

    def __init__(self, code: str, message: str, stage: str = "load") -> None:
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.stage = stage


# Sample / task manifest validation failures (business rule 1).
E_SAMPLE_INVALID = "E_SAMPLE_INVALID"
E_TASK_LIST_INVALID = "E_TASK_LIST_INVALID"
# Base parameter package validation failures (business rule 1).
E_BASE_SHAPE_MISMATCH = "E_BASE_SHAPE_MISMATCH"
E_BASE_FINITE_MISMATCH = "E_BASE_FINITE_MISMATCH"
# Configuration validation failures.
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_TARGET_LAYER_UNKNOWN = "E_TARGET_LAYER_UNKNOWN"
E_RANK_INVALID = "E_RANK_INVALID"
E_VALIDATION_EMPTY = "E_VALIDATION_EMPTY"
# Snapshot / resume reconciliation failures (business rule 5).
E_CONFIG_MISMATCH = "E_CONFIG_MISMATCH"
E_SNAPSHOT_CORRUPT = "E_SNAPSHOT_CORRUPT"
# Per-task runtime failures kept in the diagnostic report (business rule 8).
E_TRAIN_TASK_FAULT = "E_TRAIN_TASK_FAULT"
E_VALIDATE_TASK_FAULT = "E_VALIDATE_TASK_FAULT"


def failure_record(code: str, stage: str, message: str, **extra: object) -> dict:
    """Build a canonical, JSON-serialisable failure entry."""
    record = {"code": code, "stage": stage, "message": message}
    record.update(extra)
    return record
