"""Stable error codes and ordered failure list entries."""

E_DEVICE_UNKNOWN = "E_DEVICE_UNKNOWN"
E_WINDOW_INVALID = "E_WINDOW_INVALID"
E_SPECTRUM_TOO_SHORT = "E_SPECTRUM_TOO_SHORT"
E_SAMPLE_NONFINITE = "E_SAMPLE_NONFINITE"
E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"

ALL_ERROR_CODES = (
    E_DEVICE_UNKNOWN,
    E_WINDOW_INVALID,
    E_SPECTRUM_TOO_SHORT,
    E_SAMPLE_NONFINITE,
    E_POINT_BUDGET_PENDING,
)


class PipelineError(Exception):
    def __init__(self, code, message, device=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.device = device


def failure_entry(code, message, device=None, stage=None, position=None):
    """Build one failure-list entry with a stable, content-derived key."""
    entry = {
        "code": code,
        "message": message,
        "device": device,
        "stage": stage,
        "position": position,
    }
    entry["key"] = stable_hash([code, message, device, stage, position])
    return entry


def stable_hash(obj):
    # local import avoids a circular import at module import time
    from .hashing import stable_hash as _stable_hash

    return _stable_hash(obj)
