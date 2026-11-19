"""Business error codes and the exception used to carry their context."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


class ErrorCode(str, Enum):
    E_CUBE_FORMAT = "E_CUBE_FORMAT"
    E_DARK_GAIN = "E_DARK_GAIN"
    E_UNMIX_COND = "E_UNMIX_COND"
    E_DRIFT_BIAS = "E_DRIFT_BIAS"
    E_CLOSURE_MISMATCH = "E_CLOSURE_MISMATCH"
    E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
    E_INJECT_CONFIG = "E_INJECT_CONFIG"


@dataclass
class WorkbenchError(RuntimeError):
    code: ErrorCode
    message: str
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.code.value}] {self.message}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "details": self.details,
        }


def fail(code: ErrorCode, message: str, **details: Any) -> None:
    raise WorkbenchError(code=code, message=message, details=details)


def first_divergence(
    expected,
    actual,
    *,
    frame_axis: int = 0,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> Optional[Dict[str, Any]]:
    """Return the first frame/pixel divergence in C-order, if any."""
    import numpy as np

    exp = np.asarray(expected)
    act = np.asarray(actual)
    if exp.shape != act.shape:
        return {"kind": "shape", "expected": list(exp.shape), "actual": list(act.shape)}
    if exp.size == 0:
        return None
    moved = np.moveaxis(exp, frame_axis, 0)
    moved_actual = np.moveaxis(act, frame_axis, 0)
    for frame in range(moved.shape[0]):
        delta = np.abs(moved[frame].astype(np.float64) - moved_actual[frame].astype(np.float64))
        threshold = atol + rtol * np.abs(moved[frame].astype(np.float64))
        bad = delta > threshold
        if not np.any(bad):
            continue
        flat_index = int(np.flatnonzero(bad.ravel(order="C"))[0])
        coordinates = np.unravel_index(flat_index, bad.shape)
        return {
            "kind": "value",
            "frame": frame,
            "pixel": [int(v) for v in coordinates],
            "flat_index": flat_index,
            "expected": float(moved[frame][coordinates]),
            "actual": float(moved_actual[frame][coordinates]),
            "absolute_delta": float(delta[coordinates]),
            "atol": float(atol),
            "rtol": float(rtol),
        }
    return None
