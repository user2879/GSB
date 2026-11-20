"""业务错误码。所有可预期的领域错误均携带稳定业务码与出错字段路径。"""

from __future__ import annotations


E_CHANNEL = "E_CHANNEL"
E_ESTIMATE = "E_ESTIMATE"
E_DIGEST = "E_DIGEST"
E_SNAPSHOT = "E_SNAPSHOT"
E_REPLAY = "E_REPLAY"
E_PROTOCOL = "E_PROTOCOL"


class BusinessError(Exception):
    """领域异常基类。

    code   稳定业务码
    path   出错字段路径（如 channels[1].window_ms），可为空
    detail 人类可读说明
    """

    def __init__(self, code: str, detail: str, path: str | None = None) -> None:
        self.code = code
        self.path = path
        self.detail = detail
        location = f" @ {path}" if path else ""
        super().__init__(f"[{code}]{location} {detail}")

    def to_dict(self) -> dict:
        return {"code": self.code, "path": self.path, "detail": self.detail}
