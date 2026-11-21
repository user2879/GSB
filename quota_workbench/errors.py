"""Business error codes for the quota arbitration workbench."""

from __future__ import annotations

from typing import Any, Optional


class WorkbenchError(Exception):
    """Base error carrying a stable business code.

    Codes:
      E_SPEC      - declaration / request-stream validation failure
      E_PORT      - loopback port orchestration failure
      E_QUOTA     - tenant share sum exceeds declared cap
      E_SNAPSHOT  - snapshot version / digest mismatch
      E_REPLAY    - replay verification mismatch
    """

    code = "E_SPEC"

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        line: Optional[int] = None,
        details: Optional[Any] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.field = field
        self.line = line
        self.details = details

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "field": self.field,
            "line": self.line,
            "details": self.details,
        }


class SpecError(WorkbenchError):
    code = "E_SPEC"


class PortError(WorkbenchError):
    code = "E_PORT"


class QuotaError(WorkbenchError):
    code = "E_QUOTA"


class SnapshotError(WorkbenchError):
    code = "E_SNAPSHOT"


class ReplayError(WorkbenchError):
    code = "E_REPLAY"
