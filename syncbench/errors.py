"""业务错误码与异常链路。

所有业务可恢复错误以 :class:`BusinessError` 携带稳定错误码返回/抛出，
错误码字符串固定，便于清单、日志与测试断言。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# --- 稳定错误码（任务要求 8 个） -------------------------------------------------
E_CH_FORMAT = "E_CH_FORMAT"            # 通道素材格式不合规
E_PULSE_MISSING = "E_PULSE_MISSING"    # 同步脉冲/同步帧缺失
E_DRIFT_EXCEED = "E_DRIFT_EXCEED"      # 注入/估计漂移超出声明容差
E_GAP_TOO_LONG = "E_GAP_TOO_LONG"      # 空洞长度超出可接受阈值
E_SYNC_MISMATCH = "E_SYNC_MISMATCH"    # 音视频双域映射不一致
E_DEPENDENCY_STALE = "E_DEPENDENCY_STALE"  # 依赖节点版本过期
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"  # 视图渲染的快照版本过期
E_BUDGET_EXCEEDED = "E_BUDGET_EXCEEDED"  # 交互预算超限（降级跳帧）

ALL_CODES = (
    E_CH_FORMAT,
    E_PULSE_MISSING,
    E_DRIFT_EXCEED,
    E_GAP_TOO_LONG,
    E_SYNC_MISMATCH,
    E_DEPENDENCY_STALE,
    E_SNAPSHOT_STALE,
    E_BUDGET_EXCEEDED,
)


@dataclass(frozen=True)
class BusinessError(Exception):
    """携带稳定错误码的业务异常（可序列化进结果清单）。"""

    code: str
    message: str
    channel: int | None = None
    segment: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - 仅调试展示
        loc = ""
        if self.channel is not None:
            loc += f" ch={self.channel}"
        if self.segment is not None:
            loc += f" seg={self.segment}"
        return f"[{self.code}]{loc} {self.message}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "channel": self.channel,
            "segment": self.segment,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class ErrorRecord:
    """结果对象中携带的结构化错误记录（不抛异常时使用）。"""

    code: str
    message: str
    channel: int | None = None
    segment: int | None = None
    detail: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_error(cls, err: BusinessError) -> "ErrorRecord":
        return cls(err.code, err.message, err.channel, err.segment, dict(err.detail))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "channel": self.channel,
            "segment": self.segment,
            "detail": self.detail,
        }


def make_record(code: str, message: str, channel: int | None = None,
                segment: int | None = None, **detail: Any) -> ErrorRecord:
    return ErrorRecord(code, message, channel, segment, dict(detail))
