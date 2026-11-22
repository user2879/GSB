"""统一错误码与异常类型。

平台所有可预期的拒绝场景都使用 :class:`PlatformError` 的子类抛出，
携带稳定的机器可读错误码与违规条目列表，便于账本与测试复核。
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping


class PlatformError(Exception):
    """平台错误基类，``code`` 为稳定错误码。"""

    code = "E_PLATFORM"

    def __init__(self, message: str, violations: Iterable[Mapping[str, Any]] | None = None):
        super().__init__(message)
        self.message = message
        self.violations = list(violations or [])

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "violations": self.violations}


class ManifestError(PlatformError):
    """配置清单 / 节点注册表强校验失败。"""

    code = "E_MANIFEST"


class RegressionError(PlatformError):
    """候选版相对基线版超出允许回归预算。"""

    code = "E_REGRESSION"


class ReplayError(PlatformError):
    """重放结果与已提交账本不一致。"""

    code = "E_REPLAY"

    @property
    def first_mismatch(self) -> Mapping[str, Any] | None:
        return self.violations[0] if self.violations else None
