"""平台统一业务错误码。

每个业务码对应任务中规定的一个错误类别，结构化日志与 CLI 输出均以
``code`` 字段为准，避免依赖人类可读的错误文本做判定。
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class PlatformError(Exception):
    """所有平台业务异常的基类。"""

    code: str = "E_GENERIC"

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details: Dict[str, Any] = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {"code": self.code, "message": self.message, "details": self.details}


class ManifestError(PlatformError):
    """清单/注册表强校验失败：整体拒绝。"""

    code = "E_MANIFEST"


class PortInUseError(PlatformError):
    """回环端口被占用或无法绑定。"""

    code = "E_PORT"


class NodeStartError(PlatformError):
    """节点子进程启动失败或异常退出。"""

    code = "E_NODE"


class HandshakeTimeoutError(PlatformError):
    """节点握手超时。"""

    code = "E_HANDSHAKE"


class ReferenceError(PlatformError):
    """删除条目仍被引用 / 引用了不存在的键。"""

    code = "E_REF"


class ConvergenceError(PlatformError):
    """回执缺失或派生摘要不一致。"""

    code = "E_CONVERGE"


class ReplayError(PlatformError):
    """恢复后重放结果与快照不一致。"""

    code = "E_REPLAY"


class RangeError(PlatformError):
    """配置项量化结果超出定点可表示范围（异常注入 E_RANGE）。"""

    code = "E_RANGE"


class CycleError(PlatformError):
    """派生依赖图存在环（异常注入 E_CYCLE）。"""

    code = "E_CYCLE"


class GridError(PlatformError):
    """快照缺失精度格度参数（异常注入 E_GRID）。"""

    code = "E_GRID"


ERROR_BY_CODE = {
    cls.code: cls
    for cls in (
        ManifestError,
        PortInUseError,
        NodeStartError,
        HandshakeTimeoutError,
        ReferenceError,
        ConvergenceError,
        ReplayError,
        RangeError,
        CycleError,
        GridError,
    )
}
