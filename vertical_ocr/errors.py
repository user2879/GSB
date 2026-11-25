from __future__ import annotations

from .codes import INPUT_INVALID


class BusinessError(Exception):
    def __init__(self, code: str, message: str, *, field: str | None = None,
                 version: str | None = None, details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.field = field
        self.version = version
        self.details = details or {}

    def to_failure(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "version": self.version,
            "details": self.details,
        }


class InputInvalid(BusinessError):
    def __init__(self, message: str, *, field: str | None = None, details: dict | None = None):
        super().__init__(INPUT_INVALID, message, field=field, details=details)


class FieldTruncated(BusinessError):
    def __init__(self, version: str, field: str, message: str):
        from .codes import FIELD_TRUNCATED
        super().__init__(FIELD_TRUNCATED, message, field=field, version=version)
