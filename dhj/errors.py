"""Typed errors with machine-readable codes and structured field paths."""


class DHJError(Exception):
    """Base error. ``code`` is the stable machine-readable error code."""

    code = "E_DHJ"

    def __init__(self, message, details=None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def to_dict(self):
        out = {"code": self.code, "message": self.message}
        if self.details:
            out["details"] = self.details
        return out


class SchemaError(DHJError):
    code = "E_SCHEMA"

    def __init__(self, message, paths=None):
        paths = list(paths or [])
        super().__init__(message, {"field_paths": paths} if paths else {})
        self.field_paths = paths


class BucketError(DHJError):
    code = "E_BUCKET"

    def __init__(self, message, path=None, version=None, offset=None):
        details = {}
        if path is not None:
            details["path"] = path
        if version is not None:
            details["version"] = version
        if offset is not None:
            details["offset"] = offset
        super().__init__(message, details)


class ResumeError(DHJError):
    code = "E_RESUME"

    def __init__(self, message, partition=None, row_id=None, detail=None):
        details = {}
        if partition is not None:
            details["partition"] = partition
        if row_id is not None:
            details["row_id"] = row_id
        if detail is not None:
            details["detail"] = detail
        super().__init__(message, details)


class VerifyError(DHJError):
    code = "E_VERIFY"

    def __init__(self, message, partition=None, row_id=None, detail=None):
        details = {}
        if partition is not None:
            details["partition"] = partition
        if row_id is not None:
            details["row_id"] = row_id
        if detail is not None:
            details["detail"] = detail
        super().__init__(message, details)


class SnapshotError(DHJError):
    code = "E_SNAPSHOT"

    def __init__(self, message, details=None):
        super().__init__(message, details)
