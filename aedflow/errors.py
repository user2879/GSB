"""Stable business codes shared by diagnostics, snapshots and tests."""

from __future__ import annotations


class AedError(Exception):
    """Base application error carrying a stable business code."""

    code = "E_INTERNAL"

    def __init__(self, message: str, *, code: str | None = None):
        super().__init__(message)
        if code is not None:
            self.code = code


# Fixed business-code catalogue. Order here is the canonical ordering used
# when serializing code summaries in the diagnostic report.
BUSINESS_CODES: dict[str, str] = {
    "OK": "processing succeeded",
    "E_SHARD_ID_DUPLICATE": "duplicate shard identifier",
    "E_SHARD_LENGTH_MISMATCH": "frame probability length does not match shard span",
    "E_PROBABILITY_RANGE": "frame probability outside the inclusive [0, 1] range",
    "E_TIMELINE_GAP_OR_OVERLAP": "shard spans are not strictly continuous on the global axis",
    "E_FRAME_GRID_MISALIGN": "shard boundary is not an integer multiple of the frame shift",
    "E_CLASS_MAP_UNMAPPED": "class code has no semantic name in the current codebook",
    "E_GRID_VERSION_RESAMPLE": "foreign frame grid version resampled onto the current grid",
    "E_TASK_FAILURE": "a single boundary or trajectory task raised an exception",
    "E_SNAPSHOT_STALE": "snapshot identity no longer matches current inputs and configuration",
    "I_BACKGROUND_FRAME": "frame posterior below threshold, classified as background",
    "I_SMOOTH_MERGED": "boundary discontinuity smoothed while merging a cut event",
}


def failure_entry(
    subject_id: str,
    stage: str,
    code: str,
    detail: str,
    *,
    exception_type: str | None = None,
) -> dict:
    """Build a failure record with a fixed field order."""

    entry = {
        "subject_id": subject_id,
        "stage": stage,
        "code": code,
        "detail": detail,
    }
    if exception_type is not None:
        entry["exception_type"] = exception_type
    return entry
