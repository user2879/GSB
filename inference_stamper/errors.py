"""统一业务异常与错误码。"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any


@dataclass
class ITSError(Exception):
    """所有业务异常的基类，携带稳定错误码与定位信息。"""

    code: str
    message: str
    item_id: str = ""
    field: str = ""
    detail: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["detail"] = self.detail or {}
        return data


class SpecError(ITSError):
    """声明或请求集校验失败（E_SPEC）。"""

    def __init__(self, message: str, item_id: str = "", field: str = "",
                 detail: dict[str, Any] | None = None) -> None:
        super().__init__("E_SPEC", message, item_id, field, detail)


class StaleError(ITSError):
    """验章时身份摘要或令牌口径版本演进导致旧章失效（E_STALE）。"""

    def __init__(self, message: str, item_id: str = "", field: str = "",
                 detail: dict[str, Any] | None = None) -> None:
        super().__init__("E_STALE", message, item_id, field, detail)


class IdentityError(ITSError):
    """身份摘要/盖章校验值被篡改（E_IDENTITY）。"""

    def __init__(self, message: str, item_id: str = "", field: str = "",
                 detail: dict[str, Any] | None = None) -> None:
        super().__init__("E_IDENTITY", message, item_id, field, detail)


class SnapshotError(ITSError):
    """账本格式版本或内容摘要不符（E_SNAPSHOT）。"""

    def __init__(self, message: str, item_id: str = "", field: str = "",
                 detail: dict[str, Any] | None = None) -> None:
        super().__init__("E_SNAPSHOT", message, item_id, field, detail)


class ReplayError(ITSError):
    """强制终止恢复后的一致性重放发现差异（E_REPLAY）。"""

    def __init__(self, message: str, item_id: str = "", field: str = "",
                 detail: dict[str, Any] | None = None) -> None:
        super().__init__("E_REPLAY", message, item_id, field, detail)


class AuditError(ITSError):
    """独立核对工具发现账本与重算结论不一致（E_AUDIT）。"""

    def __init__(self, message: str, item_id: str = "", field: str = "",
                 detail: dict[str, Any] | None = None) -> None:
        super().__init__("E_AUDIT", message, item_id, field, detail)
