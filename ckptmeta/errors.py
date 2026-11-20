from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EXIT_CODES: dict[str, int] = {
    "E_CONFIG": 10,
    "E_FORMAT": 11,
    "E_INTEGRITY": 12,
    "E_CHAIN": 13,
    "E_UPGRADE": 14,
    "E_RESTORE": 15,
    "E_DIVERGE": 16,
    "E_REPLAY": 17,
    "E_IO": 18,
    "E_INTERNAL": 19,
}


@dataclass
class InteropError(Exception):
    """A business error with a stable code and machine-readable details."""

    code: str
    message: str
    path: str | None = None
    manifest_path: str | None = None
    shard_index: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    @property
    def exit_code(self) -> int:
        return EXIT_CODES.get(self.code, EXIT_CODES["E_INTERNAL"])

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": False,
            "business_code": self.code,
            "message": self.message,
            "path": self.path,
            "manifest_path": self.manifest_path,
            "shard_index": self.shard_index,
            "details": self.details,
        }
        return payload


def fail(
    code: str,
    message: str,
    *,
    path: str | None = None,
    manifest_path: str | None = None,
    shard_index: int | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    raise InteropError(
        code=code,
        message=message,
        path=path,
        manifest_path=manifest_path,
        shard_index=shard_index,
        details=details or {},
    )
