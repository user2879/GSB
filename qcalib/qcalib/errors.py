"""业务错误码与失败明细。

所有加载/解析/运行阶段的非法条目都被收集为 :class:`Failure`，
不会因为单条非法而中止整体流程。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Optional


# ---- 业务码（按任务要求命名）------------------------------------------------
E_CONFIG_INVALID = "E_CONFIG_INVALID"          # 配置/字段取值非法
E_GROUP_UNKNOWN = "E_GROUP_UNKNOWN"            # 引用了未声明的列组
E_FIELD_TRUNCATED = "E_FIELD_TRUNCATED"        # 定宽行字段宽度不足/错位
E_SCALE_INVALID = "E_SCALE_INVALID"            # 缩放因子非有限正数
E_DUPLICATE_ROW = "E_DUPLICATE_ROW"            # 行标识重复
E_MISSING_COLUMN = "E_MISSING_COLUMN"          # 列在表内不存在
E_EMPTY_GROUP = "E_EMPTY_GROUP"                # 组内列集合为空
E_ENUM_INVALID = "E_ENUM_INVALID"              # 枚举值非法（方案/舍入/位宽）
E_ZERO_OUT_OF_RANGE = "E_ZERO_OUT_OF_RANGE"    # 零点超出位宽允许区间
E_ACC_OVERFLOW = "E_ACC_OVERFLOW"              # 越界累加
E_CALIB_BUDGET_PENDING = "E_CALIB_BUDGET_PENDING"  # 预算不足，任务挂起


@dataclass
class Failure:
    """单条失败明细。"""

    code: str
    message: str
    source: str = ""
    location: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class FailureCollector:
    """收集失败明细，保持追加顺序稳定。"""

    def __init__(self, source: str = "") -> None:
        self.source = source
        self._items: list[Failure] = []

    def add(
        self,
        code: str,
        message: str,
        location: Optional[str] = None,
        **detail: Any,
    ) -> Failure:
        failure = Failure(
            code=code,
            message=message,
            source=self.source,
            location=location,
            detail=detail,
        )
        self._items.append(failure)
        return failure

    def extend(self, failures: "list[Failure] | FailureCollector") -> None:
        if isinstance(failures, FailureCollector):
            self._items.extend(failures._items)
        else:
            self._items.extend(failures)

    @property
    def items(self) -> list[Failure]:
        return list(self._items)

    @property
    def codes(self) -> list[str]:
        return [f.code for f in self._items]

    def has(self, code: str) -> bool:
        return any(f.code == code for f in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __bool__(self) -> bool:
        return bool(self._items)

    def to_list(self) -> list[Dict[str, Any]]:
        return [f.to_dict() for f in self._items]
