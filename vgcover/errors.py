"""Business error codes and structured exceptions for vgcover."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class ErrorCode:
    """Stable public error-code constants.

    Codes are strings on purpose so they survive JSON/CLI boundaries without a
    shared enum module on the caller side.
    """

    SCENE_INVALID = "E_SCENE_INVALID"
    FILL_RULE = "E_FILL_RULE"
    TOLERANCE_RANGE = "E_TOLERANCE_RANGE"
    ALPHA_RANGE = "E_ALPHA_RANGE"
    CONTOUR_UNCLOSED = "E_CONTOUR_UNCLOSED"
    CLOSURE_MISMATCH = "E_CLOSURE_MISMATCH"
    LEDGER_MISMATCH = "E_LEDGER_MISMATCH"
    COVERAGE_RANGE = "E_COVERAGE_RANGE"


class CoverageError(Exception):
    """A business error with locator fields and a JSON-safe representation."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        region_id: int | None = None,
        contour_index: int | None = None,
        segment_index: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.region_id = region_id
        self.contour_index = contour_index
        self.segment_index = segment_index
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.region_id is not None:
            payload["region_id"] = self.region_id
        if self.contour_index is not None:
            payload["contour_index"] = self.contour_index
        if self.segment_index is not None:
            payload["segment_index"] = self.segment_index
        if self.details:
            payload["details"] = self.details
        return payload

    def __str__(self) -> str:
        locators = []
        if self.region_id is not None:
            locators.append(f"region_id={self.region_id}")
        if self.contour_index is not None:
            locators.append(f"contour_index={self.contour_index}")
        if self.segment_index is not None:
            locators.append(f"segment_index={self.segment_index}")
        suffix = f" ({', '.join(locators)})" if locators else ""
        return f"{self.code}: {self.message}{suffix}"


@dataclass(frozen=True)
class ErrorInfo:
    """Small immutable view useful when composing higher-level APIs."""

    code: str
    region_id: int | None = None
    contour_index: int | None = None
    segment_index: int | None = None

