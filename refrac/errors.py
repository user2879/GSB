"""业务错误码与异常类型。

错误码总表（需求第 7 条）：
- E_IOR_RANGE            折射率非法（<=0 或超出声明范围）
- E_RAY_DEGENERATE       光线方向向量为零
- E_DEPTH_BOUNDS         递归深度参数非法（<1）或触及深度上限
- E_ABSORB_NEGATIVE      吸收系数为负
- E_INTERSECT_DEGENERATE 退化相交（如半径为负）
- E_PIXEL_ISOLATED       单像素失败被隔离（帧级归类码）
- E_INJECT_CONFIG        注入配置本身非法
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

ERROR_CODES = (
    "E_IOR_RANGE",
    "E_RAY_DEGENERATE",
    "E_DEPTH_BOUNDS",
    "E_ABSORB_NEGATIVE",
    "E_INTERSECT_DEGENERATE",
    "E_PIXEL_ISOLATED",
    "E_INJECT_CONFIG",
)


class RefracError(Exception):
    """所有可归类业务异常的基类。"""

    code: str = "E_PIXEL_ISOLATED"

    def __init__(self, message: str, context: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.context: dict[str, Any] = dict(context or {})

    def to_record(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "context": self.context}


class IorRangeError(RefracError):
    code = "E_IOR_RANGE"


class RayDegenerateError(RefracError):
    code = "E_RAY_DEGENERATE"


class DepthBoundsError(RefracError):
    code = "E_DEPTH_BOUNDS"


class AbsorbNegativeError(RefracError):
    code = "E_ABSORB_NEGATIVE"


class IntersectDegenerateError(RefracError):
    code = "E_INTERSECT_DEGENERATE"


class InjectConfigError(RefracError):
    code = "E_INJECT_CONFIG"


@dataclass
class PixelIsolatedError(RefracError):
    """单像素隔离记录：包装底层错误码与坐标。"""

    row: int = -1
    col: int = -1
    origin_code: str = "E_PIXEL_ISOLATED"

    def __post_init__(self) -> None:
        super().__init__(
            f"pixel ({self.row},{self.col}) isolated due to {self.origin_code}",
            {"row": self.row, "col": self.col, "origin_code": self.origin_code},
        )
        self.code = "E_PIXEL_ISOLATED"

    def to_record(self) -> dict[str, Any]:
        record = super().to_record()
        record["row"] = self.row
        record["col"] = self.col
        record["origin_code"] = self.origin_code
        return record


def classify_exception(exc: BaseException) -> str:
    """返回异常对应的业务错误码；未知异常归到 E_PIXEL_ISOLATED。"""
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code in ERROR_CODES:
        return code
    return "E_PIXEL_ISOLATED"
