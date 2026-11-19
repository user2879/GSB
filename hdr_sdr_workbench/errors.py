"""业务错误码与异常链路。

所有可预期业务错误都抛出 :class:`WorkbenchError`，携带稳定错误码字符串。
上层（CLI / 交互式工作台）捕获后转换为返回码，工作区保持在可继续操作
的状态，不会留下半提交快照。
"""

from __future__ import annotations

# 错误码常量（对外稳定，禁止改名）
E_FRAME_FORMAT = "E_FRAME_FORMAT"
E_STATE_VERSION = "E_STATE_VERSION"
E_HASH_MISMATCH = "E_HASH_MISMATCH"
E_CLOSURE_DIVERGE = "E_CLOSURE_DIVERGE"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"
E_TONEMAP_PARAM = "E_TONEMAP_PARAM"
E_RECOVER_ORPHAN = "E_RECOVER_ORPHAN"

ALL_ERROR_CODES = (
    E_FRAME_FORMAT,
    E_STATE_VERSION,
    E_HASH_MISMATCH,
    E_CLOSURE_DIVERGE,
    E_SNAPSHOT_STALE,
    E_TONEMAP_PARAM,
    E_RECOVER_ORPHAN,
)


class WorkbenchError(Exception):
    """带稳定业务错误码的异常。"""

    def __init__(self, code: str, message: str, *, details: dict | None = None):
        if code not in ALL_ERROR_CODES:
            raise ValueError(f"unknown error code: {code}")
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "details": self.details}
