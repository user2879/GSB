"""Domain error codes and structured logging for the middleware workbench.

Every rejected input or violated invariant maps to one stable error code so
that experiment results remain machine-checkable:

E_SPEC      event stream / rules / spec file malformed or fails validation
E_WATER     watermark rollback request rejected (monotonicity invariant)
E_LATE      late event registered (informational record, not a hard failure)
E_VIEW      incremental cache view differs from authoritative replay view
E_LAT       UI query exceeded its declared latency budget
E_ORDER     timestamp regression beyond the allowed out-of-order allowance
E_SNAPSHOT  snapshot format version / content digest / parameter mismatch
E_REPLAY    post-recovery replay diverged from snapshotted state
"""
from __future__ import annotations

import json
import os
import time
from typing import Any


class BenchError(Exception):
    """Base class carrying a stable error code and a structured payload."""

    code = "E_BENCH"

    def __init__(self, message: str, **details: Any) -> None:
        super().__init__(message)
        self.message = message
        self.details = details

    def to_record(self) -> dict[str, Any]:
        record = {"code": self.code, "message": self.message}
        record.update(self.details)
        return record


class SpecError(BenchError):
    code = "E_SPEC"

    def __init__(self, message: str, *, line: int | None = None,
                 field: str | None = None, **details: Any) -> None:
        super().__init__(message, line=line, field=field, **details)


class WatermarkError(BenchError):
    code = "E_WATER"


class ViewMismatchError(BenchError):
    code = "E_VIEW"


class OrderError(BenchError):
    code = "E_ORDER"


class SnapshotError(BenchError):
    code = "E_SNAPSHOT"


class ReplayError(BenchError):
    code = "E_REPLAY"


# E_LATE and E_LAT are observational records rather than hard failures;
# they are written into the ledger/structured log but never raised.
LATE_CODE = "E_LATE"
LAT_CODE = "E_LAT"


class StructuredLogger:
    """Append-only JSON Lines (NDJSON) structured logger.

    Each call appends one canonical JSON object terminated by '\n' and
    followed by an fsync so that a forced termination leaves either a whole
    record or no record at all (no torn JSON lines).
    """

    def __init__(self, path: str) -> None:
        self.path = path
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")

    def emit(self, code: str, message: str, **fields: Any) -> dict[str, Any]:
        record: dict[str, Any] = {
            "ts_ms": int(time.time() * 1000),
            "code": code,
            "message": message,
        }
        for key, value in fields.items():
            record[key] = _jsonable(value)
        line = json.dumps(record, sort_keys=True, ensure_ascii=False,
                          separators=(",", ":"))
        self._fh.write(line + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())
        return record

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()

    def __enter__(self) -> "StructuredLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return str(value)
