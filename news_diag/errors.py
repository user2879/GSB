"""Business error types used by the workbench."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


E_DOC_INVALID = "E_DOC_INVALID"
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_COMMAND_INVALID = "E_COMMAND_INVALID"
E_SNAPSHOT_STALE = "E_SNAPSHOT_STALE"


@dataclass
class WorkbenchError(Exception):
    code: str
    message: str
    stage: str
    document_id: str | None = None
    exception_type: str = "WorkbenchError"
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_failure(self, **extra: Any) -> dict[str, Any]:
        failure: dict[str, Any] = {
            "document_id": self.document_id or "",
            "stage": self.stage,
            "code": self.code,
            "exception_type": self.exception_type,
            "message": self.message,
        }
        if self.details:
            failure["details"] = self.details
        failure.update(extra)
        return failure


class DocumentInvalid(WorkbenchError):
    def __init__(self, message: str, *, stage: str = "load", document_id: str | None = None,
                 details: dict[str, Any] | None = None) -> None:
        super().__init__(E_DOC_INVALID, message, stage, document_id,
                         self.__class__.__name__, details or {})


class ConfigInvalid(WorkbenchError):
    def __init__(self, message: str, *, stage: str = "config", document_id: str | None = None,
                 details: dict[str, Any] | None = None) -> None:
        super().__init__(E_CONFIG_INVALID, message, stage, document_id,
                         self.__class__.__name__, details or {})


class CommandInvalid(WorkbenchError):
    def __init__(self, message: str, *, document_id: str | None = None,
                 details: dict[str, Any] | None = None) -> None:
        super().__init__(E_COMMAND_INVALID, message, "command", document_id,
                         self.__class__.__name__, details or {})


class SnapshotStale(WorkbenchError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(E_SNAPSHOT_STALE, message, "snapshot", None,
                         self.__class__.__name__, details or {})

