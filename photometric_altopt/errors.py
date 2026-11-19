"""业务错误码与异常定义。

所有可预期的求解/管线异常都使用 :class:`BusinessError` 抛出，
``code`` 为稳定的业务码字符串，``details`` 携带受影响像素或视点等诊断信息。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ---- 业务码（字符串稳定，禁止随意改名） ------------------------------------
E_NONFINITE_ALBEDO = "E_NONFINITE_ALBEDO"
E_ZERO_LIGHT = "E_ZERO_LIGHT"
E_DEGENERATE_VIEWS = "E_DEGENERATE_VIEWS"
E_ILL_CONDITIONED = "E_ILL_CONDITIONED"
E_VIEW_SKIPPED = "E_VIEW_SKIPPED"
E_RENDER_DIVERGE = "E_RENDER_DIVERGE"
E_NO_CHECKPOINT = "E_NO_CHECKPOINT"

# 异常类别：参数层 / 管线层
LAYER_PARAMETER = "parameter"
LAYER_PIPELINE = "pipeline"


@dataclass
class BusinessError(Exception):
    """带业务码的异常。

    Attributes:
        code: 业务码，见本模块常量。
        message: 人类可读说明。
        layer: ``parameter`` 或 ``pipeline``。
        details: 结构化诊断信息（受影响像素坐标、视点索引等），可 JSON 序列化。
    """

    code: str
    message: str
    layer: str = LAYER_PARAMETER
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:  # pragma: no cover - 调试用
        return f"[{self.code}] {self.message} :: {self.details}"


def is_business_code(value: Any) -> bool:
    """判断字符串是否为已声明的业务码。"""
    return isinstance(value, str) and value in {
        E_NONFINITE_ALBEDO,
        E_ZERO_LIGHT,
        E_DEGENERATE_VIEWS,
        E_ILL_CONDITIONED,
        E_VIEW_SKIPPED,
        E_RENDER_DIVERGE,
        E_NO_CHECKPOINT,
    }
