"""核心数据模型与业务常量（全部使用整数/有理数，杜绝浮点累计误差）。"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from fractions import Fraction
from typing import Final

# 万分比基数：目标可用率、燃烧率阈值均以该基数表达
PERMYRIAD: Final[int] = 10_000
# 燃烧率 1.0 对应万分比数值
BURN_RATE_ONE: Final[int] = PERMYRIAD
# 合法区间
TARGET_MIN: Final[int] = 1
TARGET_MAX: Final[int] = 9_999
BURN_MIN: Final[int] = 1
BURN_MAX: Final[int] = 1_000_000
BUDGET_FLOOR_MIN: Final[int] = 0
WINDOW_MIN: Final[int] = 1

# 路径选择显式规则阈值（常量，便于审计）
SELECT_WINDOW_MINUTES: Final[int] = 60
SELECT_BATCH_COUNT: Final[int] = 12
SELECT_INCREMENTAL_STATE_RATIO: Final[float] = 0.5
SELECT_CHUNK_CAPACITY: Final[int] = 8

PATH_EXACT: Final[str] = "exact"
PATH_INCREMENTAL: Final[str] = "incremental"


@dataclass(frozen=True)
class ServiceSpec:
    """服务声明（某一参数版本）。

    target_permyriad  目标可用率，万分比整数，[1, 9999]
    window_minutes    滚动窗口长度（分钟，正整数）
    burn_threshold    燃烧率告警阈值，万分比整数（BURN_RATE_ONE 表示 1.0）
    budget_floor      错误预算下限（缩放整数，剩余低于该值触发告警）
    param_version     参数版本号，单调递增
    """

    service_id: str
    target_permyriad: int
    window_minutes: int
    burn_threshold: int
    budget_floor: int
    param_version: int = 1

    def with_version(self, new_version: int, **changes: int) -> "ServiceSpec":
        return replace(self, param_version=new_version, **changes)


@dataclass(frozen=True)
class MinuteBatch:
    """分钟级 SLI 观测批次。

    line      在 NDJSON 文件中的物理行号（1 基），用于 E_SPEC 定位
    batch_seq 批次滚动序号（同文件内单调），迟到批通过 stream 区分
    minute    整数分钟序号
    successes 成功计数（非负整数）
    failures  失败计数（非负整数）
    stream    regular=常规滚动流；late=迟到/重发流（允许分钟乱序、重复）
    """

    service_id: str
    line: int
    batch_seq: int
    minute: int
    successes: int
    failures: int
    stream: str = "regular"


@dataclass(frozen=True)
class MinutePoint:
    """归并去重后某一分钟的唯一采纳观测。"""

    minute: int
    successes: int
    failures: int
    source_line: int
    source_batch_seq: int
    source_stream: str


@dataclass(frozen=True)
class WindowConclusion:
    """单服务单分钟的滚动窗口结论（账本最小复核单元）。"""

    service_id: str
    minute: int
    window_minutes: int
    param_version: int
    path: str
    total: int
    failures: int
    budget_total_scaled: int
    remaining_budget_scaled: int
    burn_rate_num: int
    burn_rate_den: int
    alert: bool
    alert_reasons: tuple[str, ...]
    chunk_capacity: int | None = None

    @property
    def burn_rate(self) -> Fraction:
        if self.burn_rate_den == 0:
            return Fraction(0)
        return Fraction(self.burn_rate_num, self.burn_rate_den)

    def comparable(self) -> dict[str, object]:
        """逐字段一致性比较视图（路径、版本等元数据不参与结论比较）。"""
        return {
            "remaining_budget_scaled": self.remaining_budget_scaled,
            "burn_rate_num": self.burn_rate_num,
            "burn_rate_den": self.burn_rate_den,
            "alert": self.alert,
        }


@dataclass
class PathDecision:
    """默认路径选择结论与可解释理由。"""

    path: str
    reason: str
    window_minutes: int
    batch_count: int
    incremental_state_ratio: float
    chunk_capacity: int

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "reason": self.reason,
            "window_minutes": self.window_minutes,
            "batch_count": self.batch_count,
            "incremental_state_ratio": self.incremental_state_ratio,
            "chunk_capacity": self.chunk_capacity,
        }


@dataclass
class DivergenceRecord:
    service_id: str
    minute: int
    field: str
    exact_value: object
    incremental_value: object
    bound: object
    exceeded: bool
    basis: str


@dataclass
class LateImpact:
    """迟到批重算后的受影响集合条目（稳定排序输出）。"""

    service_id: str
    minute: int
    changed_fields: tuple[str, ...]
    path: str
