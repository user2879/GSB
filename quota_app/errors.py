from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_FORMAT = "E_FORMAT"
E_UPGRADE = "E_UPGRADE"
E_DERIVE = "E_DERIVE"
E_ORDER = "E_ORDER"
LIMIT = "LIMIT"
E_PREVIEW = "E_PREVIEW"
E_LATENCY = "E_LATENCY"
E_REPLAY = "E_REPLAY"
E_GUI = "E_GUI"


@dataclass
class AppError(Exception):
    """A business error with a stable machine-readable shape.

    ``volume_id`` and ``field_path`` are deliberately optional because errors
    such as an unreadable JSON document cannot always identify a volume.
    """

    code: str
    message: str
    volume_id: str | None = None
    field_path: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
            "volume_id": self.volume_id,
            "field_path": self.field_path,
        }
        if self.details:
            result["details"] = self.details
        return result


class ConfigFormatError(AppError):
    def __init__(
        self,
        message: str,
        *,
        volume_id: str | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(E_FORMAT, message, volume_id, field_path, details or {})


class UpgradeError(AppError):
    def __init__(
        self,
        message: str,
        *,
        volume_id: str | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(E_UPGRADE, message, volume_id, field_path, details or {})


class DeriveError(AppError):
    def __init__(
        self,
        message: str,
        *,
        volume_id: str | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(E_DERIVE, message, volume_id, field_path, details or {})


class OrderError(AppError):
    def __init__(
        self,
        message: str,
        *,
        volume_id: str | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(E_ORDER, message, volume_id, field_path, details or {})


class PreviewError(AppError):
    def __init__(
        self,
        message: str,
        *,
        volume_id: str | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(E_PREVIEW, message, volume_id, field_path, details or {})


class ReplayError(AppError):
    def __init__(
        self,
        message: str,
        *,
        volume_id: str | None = None,
        field_path: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(E_REPLAY, message, volume_id, field_path, details or {})


class GuiError(AppError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(E_GUI, message, None, None, details or {})
