"""Business errors and stable error codes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

E_CONFIG = "E_CONFIG"
E_CONNECT = "E_CONNECT"
E_SCRIPT = "E_SCRIPT"
E_PORT = "E_PORT"
E_VERSION = "E_VERSION"
E_REPLAY = "E_REPLAY"
E_STATE = "E_STATE"
E_BUDGET = "E_BUDGET"


@dataclass
class BusinessError(Exception):
    """An expected, structured business failure.

    ``fields`` are stable dotted paths.  They are always sorted before being
    exposed so logs and UI rows are deterministic.
    """

    code: str
    message: str
    fields: list[str] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.fields = sorted(dict.fromkeys(self.fields))
        Exception.__init__(self, self.message)

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "fields": list(self.fields),
            "context": dict(self.context),
        }


def config_error(
    fields: Iterable[str], message: str = "invalid session configuration"
) -> BusinessError:
    return BusinessError(E_CONFIG, message, list(fields))


def version_error(fields: Iterable[str], message: str, **context: Any) -> BusinessError:
    return BusinessError(E_VERSION, message, list(fields), context)
