from __future__ import annotations

from typing import Any


class ErrorCode:
    IMAGE_FORMAT = "E_IMAGE_FORMAT"
    TRUTH_INVALID = "E_TRUTH_INVALID"
    CONFIG_INVALID = "E_CONFIG_INVALID"
    FIELD_TRUNCATED = "E_FIELD_TRUNCATED"
    CELL_DEGRADED = "E_CELL_DEGRADED"
    SAMPLE_BUDGET_PENDING = "E_SAMPLE_BUDGET_PENDING"


class LedgerError(Exception):
    code = "E_LEDGER"

    def __init__(self, message: str, **context: Any) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def to_issue(self) -> dict[str, Any]:
        return make_issue(self.code, self.message, **self.context)


class ImageFormatError(LedgerError):
    code = ErrorCode.IMAGE_FORMAT


class TruthInvalidError(LedgerError):
    code = ErrorCode.TRUTH_INVALID


class ConfigInvalidError(LedgerError):
    code = ErrorCode.CONFIG_INVALID


class FieldTruncatedError(LedgerError):
    code = ErrorCode.FIELD_TRUNCATED


def make_issue(code: str, message: str, **context: Any) -> dict[str, Any]:
    issue: dict[str, Any] = {"code": code, "message": message}
    for key in sorted(context):
        if context[key] is not None:
            issue[key] = context[key]
    return issue
