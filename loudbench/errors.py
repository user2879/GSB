"""业务错误码与可复算异常链路。

所有业务异常都带稳定错误码、发生位置（任务键/样本位置）以及触发输入片段
（截取的样本或参数），保证错误本身可以被离线复算。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


E_AUDIO_FORMAT = "E_AUDIO_FORMAT"
E_BLOCK_BOUNDS = "E_BLOCK_BOUNDS"
E_BUDGET_DENIED = "E_BUDGET_DENIED"
E_WINDOW_INVALID = "E_WINDOW_INVALID"
E_MERGE_ORDER = "E_MERGE_ORDER"
E_PEAK_OVER = "E_PEAK_OVER"
E_THREAD_RANGE = "E_THREAD_RANGE"
E_TIME_BUDGET = "E_TIME_BUDGET"


class LBError(Exception):
    """带稳定错误码的工作台业务异常。"""

    code: str = "E_LB"

    def __init__(
        self,
        message: str,
        *,
        location: Optional[Dict[str, Any]] = None,
        evidence: Optional[Dict[str, Any]] = None,
        cause: Optional[BaseException] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.location = dict(location or {})
        self.evidence = dict(evidence or {})
        if cause is not None:
            self.__cause__ = cause

    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "evidence": self.evidence,
        }


class AudioFormatError(LBError):
    code = E_AUDIO_FORMAT


class BlockBoundsError(LBError):
    code = E_BLOCK_BOUNDS


class BudgetDeniedError(LBError):
    code = E_BUDGET_DENIED


class WindowInvalidError(LBError):
    code = E_WINDOW_INVALID


class MergeOrderError(LBError):
    code = E_MERGE_ORDER


class PeakOverError(LBError):
    code = E_PEAK_OVER


class ThreadRangeError(LBError):
    code = E_THREAD_RANGE


class TimeBudgetError(LBError):
    code = E_TIME_BUDGET


@dataclass
class ErrorRecord:
    """降级/拒绝事件，进入报告的可追溯记录。"""

    code: str = ""
    message: str = ""
    location: Dict[str, Any] = field(default_factory=dict)
    evidence: Dict[str, Any] = field(default_factory=dict)
    recovery: str = ""

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "location": self.location,
            "evidence": self.evidence,
        }
        if self.recovery:
            out["recovery"] = self.recovery
        return out
