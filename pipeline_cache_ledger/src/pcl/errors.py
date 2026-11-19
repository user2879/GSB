"""Stable error codes and structured exceptions for the cache ledger."""


class PclError(Exception):
    code = "E_INTERNAL"

    def __init__(self, message, details=None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self):
        return {"code": self.code, "message": self.message, "details": self.details}


class SpecError(PclError):
    """Manifest specification rejection. ``details['paths']`` lists field paths."""

    code = "E_SPEC"

    def __init__(self, message, paths):
        if isinstance(paths, str):
            paths = [paths]
        super().__init__(message, {"paths": sorted(set(paths))})


class CorruptEntryError(PclError):
    code = "E_CORRUPT"

    def __init__(self, message, key, offset):
        super().__init__(message, {"key": key, "offset": offset})


class DigestMismatchError(PclError):
    code = "E_DIGEST"

    def __init__(self, message, key, expected, actual):
        super().__init__(
            message,
            {"key": key, "expected_digest": expected, "actual_digest": actual},
        )


class ReferenceError(PclError):
    code = "E_REF"

    def __init__(self, message, key, referrers):
        super().__init__(message, {"key": key, "referrers": sorted(set(referrers))})


class CycleError(PclError):
    code = "E_CYCLE"

    def __init__(self, message, cycle):
        super().__init__(message, {"cycle": list(cycle)})


class SnapshotError(PclError):
    code = "E_SNAPSHOT"
