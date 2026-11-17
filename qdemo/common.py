"""Shared utilities: business codes, canonical fingerprints, atomic IO, rounding."""
from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile

import numpy as np

# ---------------------------------------------------------------------------
# Business codes
# ---------------------------------------------------------------------------
OK = "OK"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_WEIGHT_SHAPE_MISMATCH = "E_WEIGHT_SHAPE_MISMATCH"
E_INTERACT_BUDGET_PENDING = "E_INTERACT_BUDGET_PENDING"
E_LAYER_FAILED = "E_LAYER_FAILED"
E_DEPENDENCY_FAILED = "E_DEPENDENCY_FAILED"
E_REPLAY_MISMATCH = "E_REPLAY_MISMATCH"
E_UNKNOWN_COMMAND = "E_UNKNOWN_COMMAND"

ALLOWED_BITS = (2, 3, 4, 5, 6, 7, 8)
SCHEMES = ("symmetric", "asymmetric")

# Exit codes
EXIT_OK = 0
EXIT_CONFIG_INVALID = 2
EXIT_WEIGHT_MISMATCH = 3
EXIT_REPLAY_MISMATCH = 4


class BusinessError(Exception):
    """Error carrying a business code."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Canonical JSON / fingerprints
# ---------------------------------------------------------------------------
def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def fingerprint(obj) -> str:
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def array_fingerprint(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(a.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Deterministic rounding to significant digits (recorded rounding parameter)
# ---------------------------------------------------------------------------
def round_sig(arr, digits: int) -> np.ndarray:
    a = np.asarray(arr, dtype=np.float64)
    out = np.zeros_like(a)
    nz = a != 0
    if np.any(nz):
        expo = np.floor(np.log10(np.abs(a[nz]))).astype(np.int64)
        factor = np.power(10.0, expo - (digits - 1))
        out[nz] = np.round(a[nz] / factor) * factor
    return out


# ---------------------------------------------------------------------------
# JSON encoding / atomic write
# ---------------------------------------------------------------------------
class NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.bool_):
            return bool(o)
        return super().default(o)


def to_jsonable(obj):
    return json.loads(json.dumps(obj, cls=NumpyEncoder))


def atomic_write_json(path: str, obj) -> None:
    """Write JSON via temp file + atomic os.replace in the same directory."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(to_jsonable(obj), fh, ensure_ascii=False, indent=2, sort_keys=True)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def read_json(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
