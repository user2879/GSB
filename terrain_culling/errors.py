"""Business error codes and typed exception chain for the terrain culling component.

Every failure raised by the component carries one stable business code so that the
demo and the test-suite can assert the exact error branch instead of matching text.
"""
from __future__ import annotations

from enum import Enum


class ErrorCode(Enum):
    E_HEIGHT_RANGE = "E_HEIGHT_RANGE"
    E_BLOCK_MISSING = "E_BLOCK_MISSING"
    E_HIZ_LEVEL = "E_HIZ_LEVEL"
    E_DEPTH_BIAS = "E_DEPTH_BIAS"
    E_BUDGET_DENIED = "E_BUDGET_DENIED"
    E_RESIDENT_LEAK = "E_RESIDENT_LEAK"
    E_TRACK_DIVERGE = "E_TRACK_DIVERGE"


class CullingError(Exception):
    """Base exception carrying a :class:`ErrorCode` and structured context."""

    code: ErrorCode = ErrorCode.E_BLOCK_MISSING

    def __init__(self, message: str, *, context: dict | None = None):
        super().__init__(message)
        self.message = message
        self.context: dict = dict(context or {})

    def to_dict(self) -> dict:
        return {"code": self.code.value, "message": self.message, "context": self.context}


class HeightRangeError(CullingError):
    code = ErrorCode.E_HEIGHT_RANGE


class BlockMissingError(CullingError):
    code = ErrorCode.E_BLOCK_MISSING


class HiZLevelError(CullingError):
    code = ErrorCode.E_HIZ_LEVEL


class DepthBiasError(CullingError):
    code = ErrorCode.E_DEPTH_BIAS


class BudgetDeniedError(CullingError):
    code = ErrorCode.E_BUDGET_DENIED


class ResidentLeakError(CullingError):
    code = ErrorCode.E_RESIDENT_LEAK


class TrackDivergenceError(CullingError):
    code = ErrorCode.E_TRACK_DIVERGE
