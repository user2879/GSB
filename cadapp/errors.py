"""Business error codes and the CadError exception type.

Every abnormal input path raises CadError with a stable machine-readable
code and a location string pointing at the offending field.  No code path
silently degrades or partially writes a model.
"""

from __future__ import annotations


class ErrorCode:
    E_SCHEMA_VERSION = "E_SCHEMA_VERSION"
    E_INDEX_OVERFLOW = "E_INDEX_OVERFLOW"
    E_NORMAL_COUNT = "E_NORMAL_COUNT"
    E_KNOT_MONOTONIC = "E_KNOT_MONOTONIC"
    E_UNDO_EMPTY = "E_UNDO_EMPTY"
    E_FORMAT_ENDIAN = "E_FORMAT_ENDIAN"
    # Additional format-level codes (header/magic/truncation).
    E_FORMAT_MAGIC = "E_FORMAT_MAGIC"
    E_FORMAT_HEADER = "E_FORMAT_HEADER"
    E_FORMAT_TRUNCATED = "E_FORMAT_TRUNCATED"


class CadError(Exception):
    """Business error carrying a stable code and a field location."""

    def __init__(self, code: str, message: str, location: str = ""):
        super().__init__(f"[{code}] {message} (at {location or '<root>'})")
        self.code = code
        self.message = message
        self.location = location

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "location": self.location}
