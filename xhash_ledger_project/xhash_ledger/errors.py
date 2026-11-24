"""错误码与失败清单数据结构。

所有校验/调度失败都被收集为 :class:`Failure`，进入报告的 ``failures`` 字段，
单条非法数据不会中断其余样本的处理。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

# 样本侧字段非法
E_SAMPLE_INVALID = "E_SAMPLE_INVALID"
# 配置或命令流字段非法
E_CONFIG_INVALID = "E_CONFIG_INVALID"
# 单样本签名计算点数超出配额
E_SIGNATURE_QUOTA_EXCEEDED = "E_SIGNATURE_QUOTA_EXCEEDED"
# 全局签名计算点预算不足，样本进入待办
E_SIGNATURE_BUDGET_PENDING = "E_SIGNATURE_BUDGET_PENDING"


@dataclass
class Failure:
    """单条失败记录，可直接 JSON 序列化。"""

    code: str
    category: str
    subject: str
    field: str
    message: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def make_failure(code: str, category: str, subject: str, field: str, message: str) -> Failure:
    return Failure(code=code, category=category, subject=subject, field=field, message=message)
