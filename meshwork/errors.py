from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any


@dataclass
class MeshError(Exception):
    """A structured, user-facing workbench error.

    ``field`` is a dotted field name and ``location`` is its JSON path, so a
    caller can always identify both the offending field and its position.
    """

    code: str
    message: str
    field: str | None = None
    location: str | None = None
    request_id: str | None = None
    service: str | None = None
    details: dict[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.field is not None:
            result["field"] = self.field
        if self.location is not None:
            result["location"] = self.location
        if self.request_id is not None:
            result["request_id"] = self.request_id
        if self.service is not None:
            result["service"] = self.service
        if self.details:
            result["details"] = self.details
        return result


def spec_error(message: str, field: str, location: str) -> MeshError:
    return MeshError("E_SPEC", message, field=field, location=location)
