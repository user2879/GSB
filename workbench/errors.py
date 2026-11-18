"""Business error codes and structured workbench errors.

All failures produced by the workbench carry one of the documented business
codes.  The workbench never raises an anonymous traceback to the CLI layer;
failures are converted into :class:`WorkbenchError` instances with an explicit
stage id and JSON-pointer style field path.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional


# Business code registry.  The numeric prefix groups the failure domain:
#   1xxx -> specification / loading
#   2xxx -> runtime data / parameter
#   3xxx -> determinism / replay / shard binding
BUSINESS_CODES: Mapping[str, str] = {
    "E_SPEC": "1001 specification rejected: schema, reference, cycle or bound violation",
    "E_DATA": "2001 sample data unreadable, malformed or type incompatible",
    "E_PARAM": "2002 stage integer parameter outside its declared bounds",
    "E_OUTLIER": "2003 outlier ratio above the stage threshold",
    "E_SHARD": "3001 shard fingerprint/content binding mismatch",
    "E_REPLAY": "3002 replay comparison mismatch against committed ledger records",
}


class WorkbenchError(Exception):
    """Structured failure carrying a business code and a diagnostic location."""

    def __init__(
        self,
        code: str,
        message: str,
        stage_id: Optional[str] = None,
        field_path: str = "$",
        details: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if code not in BUSINESS_CODES:
            raise KeyError(f"unknown business code: {code}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.stage_id = stage_id
        self.field_path = field_path
        self.details = dict(details or {})

    def as_event(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "stage_id": self.stage_id or "",
            "field_path": self.field_path,
            "details": self.details,
        }

    def __str__(self) -> str:  # pragma: no cover - trivial
        loc = self.stage_id or "-"
        return f"{self.code} stage={loc} path={self.field_path} {self.message}"
