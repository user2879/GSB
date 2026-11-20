"""Business error codes shared by every pipeline stage.

Each error carries a stable ``code`` string plus a structured ``context``
mapping so the demo report can render the full exception chain.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


class AnimError(Exception):
    """Base class for all pipeline business errors."""

    code: str = "E_UNKNOWN"

    def __init__(self, message: str, context: Optional[Mapping[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context or {})

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}


class SchemaVersionError(AnimError):
    code = "E_SCHEMA_VERSION"


class QuatInvalidError(AnimError):
    code = "E_QUAT_INVALID"


class IndexOverflowError(AnimError):
    code = "E_INDEX_OVERFLOW"


class TimeNonMonotonicError(AnimError):
    code = "E_TIME_NONMONOTONIC"


class RetargetDivergeError(AnimError):
    code = "E_RETARGET_DIVERGE"


class BudgetDeniedError(AnimError):
    code = "E_BUDGET_DENIED"


class ClosureMismatchError(AnimError):
    code = "E_CLOSURE_MISMATCH"


E_SCHEMA_VERSION = SchemaVersionError
E_QUAT_INVALID = QuatInvalidError
E_INDEX_OVERFLOW = IndexOverflowError
E_TIME_NONMONOTONIC = TimeNonMonotonicError
E_RETARGET_DIVERGE = RetargetDivergeError
E_BUDGET_DENIED = BudgetDeniedError
E_CLOSURE_MISMATCH = ClosureMismatchError
