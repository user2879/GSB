"""业务错误码与统一异常类型。

所有非法输入与运行期可诊断错误都抛出 :class:`WorkbenchError`，
CLI 边界捕获后输出带 ``E_xxx`` 业务码的 JSON，且不产生任何持久化输出。
"""
from __future__ import annotations


class WorkbenchError(Exception):
    """携带业务码的可诊断异常。"""

    def __init__(self, code: str, message: str, detail: dict | None = None):
        if not code.startswith("E_"):
            raise ValueError("业务码必须以 E_")
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail or {}

    def to_dict(self) -> dict:
        return {"ok": False, "code": self.code, "message": self.message, "detail": self.detail}


# 解析与校验类
E_PARSE = "E_PARSE"
E_SCHEMA = "E_SCHEMA"
E_PITCH = "E_PITCH"
E_BEAT = "E_BEAT"
E_VOICE = "E_VOICE"
E_OVERLAP_SAME_VOICE = "E_OVERLAP_SAME_VOICE"
E_TEMPO = "E_TEMPO"
E_VELOCITY_CURVE = "E_VELOCITY_CURVE"
E_EDIT_OP = "E_EDIT_OP"
E_EDIT_TARGET = "E_EDIT_TARGET"
# 持久化与一致性类
E_DIGEST = "E_DIGEST"
E_VERSION = "E_VERSION"
E_BUDGET = "E_BUDGET"
# 资源类
E_IO = "E_IO"
