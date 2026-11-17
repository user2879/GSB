"""Business error codes and structured error type for graymirror.

All validation / replay / version failures raise :class:`GrayError` carrying a
stable business code so the CLI can render machine readable output and append a
structured JSON log line.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


E_RULE = "E_RULE"        # rule file validation failure
E_STREAM = "E_STREAM"    # request stream (NDJSON) validation failure
E_GRID = "E_GRID"        # fixed point grid / scale factor failure
E_VERSION = "E_VERSION"  # snapshot / rule version alignment failure
E_REPLAY = "E_REPLAY"    # replay consistency failure

ALL_CODES = (E_RULE, E_STREAM, E_GRID, E_VERSION, E_REPLAY)


@dataclass
class GrayError(Exception):
    code: str
    message: str
    field_path: str = ""
    line: int | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["name"] = "GrayError"
        return payload
