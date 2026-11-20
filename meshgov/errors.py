from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


E_RULE = "E_RULE"
E_UPSTREAM = "E_UPSTREAM"
E_CIRCUIT = "E_CIRCUIT"
E_SNAPSHOT = "E_SNAPSHOT"
E_VERIFY = "E_VERIFY"
E_RECOVERY = "E_RECOVERY"


@dataclass
class MeshError(Exception):
    code: str
    message: str
    field_path: str = "$"
    context: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(f"{self.code}: {self.message} ({self.field_path})")

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "field_path": self.field_path,
            "context": dict(self.context),
        }


class RuleError(MeshError):
    def __init__(
        self,
        message: str,
        field_path: str = "$",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(E_RULE, message, field_path, context or {})


class UpstreamError(MeshError):
    def __init__(
        self,
        message: str,
        field_path: str = "$",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(E_UPSTREAM, message, field_path, context or {})


class SnapshotError(MeshError):
    def __init__(
        self,
        message: str,
        field_path: str = "$",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(E_SNAPSHOT, message, field_path, context or {})


class VerifyError(MeshError):
    def __init__(
        self,
        message: str,
        field_path: str = "$",
        context: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(E_VERIFY, message, field_path, context or {})

