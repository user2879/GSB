"""Stable error codes and exception hierarchy for batchdiag."""

E_SPEC = "E_SPEC"
E_FIELD = "E_FIELD"
E_CHAIN = "E_CHAIN"
E_TIMEOUT = "E_TIMEOUT"
E_REPLAY = "E_REPLAY"
E_SNAPSHOT = "E_SNAPSHOT"


class BatchDiagError(Exception):
    """Base error.

    Every diagnostic error carries a stable code, a dotted field path, an
    optional source file/line and a deterministic human readable reason so
    that failures are reproducible byte for byte across runs.
    """

    code = "E_GENERIC"

    def __init__(self, reason, field=None, line=None, source=None, extra=None):
        self.reason = reason
        self.field = field
        self.line = line
        self.source = source
        self.extra = extra or {}
        loc = ""
        if source is not None:
            loc = f"{source}:"
        if line is not None:
            loc += f"L{line}:"
        if field:
            loc += f"{field} "
        message = f"[{self.code}] {loc}{reason}".strip()
        super().__init__(message)

    def to_dict(self):
        data = {
            "code": self.code,
            "reason": self.reason,
            "field": self.field,
            "line": self.line,
            "source": self.source,
        }
        data.update(self.extra)
        return data


class SpecError(BatchDiagError):
    code = E_SPEC


class FieldError(BatchDiagError):
    code = E_FIELD


class ChainError(BatchDiagError):
    code = E_CHAIN


class TimeoutError_(BatchDiagError):
    code = E_TIMEOUT


class ReplayError(BatchDiagError):
    code = E_REPLAY


class SnapshotError(BatchDiagError):
    code = E_SNAPSHOT
