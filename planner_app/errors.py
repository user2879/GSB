"""Structured business errors used by every public entry point."""

from __future__ import annotations

from typing import Any


E_SPEC = "E_SPEC"
E_SYNTAX = "E_SYNTAX"
E_REF = "E_REF"
E_DIVERGE = "E_DIVERGE"
E_LATENCY = "E_LATENCY"
E_SNAPSHOT = "E_SNAPSHOT"
E_REPLAY = "E_REPLAY"
E_LEDGER = "E_LEDGER"


class AppError(Exception):
    """A deterministic, serialisable application error.

    ``json_path`` points at the first offending JSON location. ``position`` is
    a zero-based character offset for SQL syntax/validation errors. Both are
    optional but are emitted when known.
    """

    def __init__(
        self,
        code: str,
        message: str,
        *,
        json_path: str | None = None,
        position: int | None = None,
        edit_seq: int | None = None,
        node_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.json_path = json_path
        self.position = position
        self.edit_seq = edit_seq
        self.node_id = node_id
        self.details = details or {}

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": False,
            "code": self.code,
            "message": self.message,
        }
        if self.json_path is not None:
            result["json_path"] = self.json_path
        if self.position is not None:
            result["position"] = self.position
        if self.edit_seq is not None:
            result["edit_seq"] = self.edit_seq
        if self.node_id is not None:
            result["node_id"] = self.node_id
        if self.details:
            result["details"] = self.details
        return result

    def __str__(self) -> str:
        suffix = ""
        if self.json_path:
            suffix += f" at {self.json_path}"
        elif self.position is not None:
            suffix += f" at character {self.position}"
        return f"{self.code}: {self.message}{suffix}"


def syntax_error(message: str, position: int, **kwargs: Any) -> AppError:
    return AppError(E_SYNTAX, message, position=position, **kwargs)


def spec_error(message: str, json_path: str, **kwargs: Any) -> AppError:
    return AppError(E_SPEC, message, json_path=json_path, **kwargs)


def ref_error(message: str, *, edit_seq: int | None = None, node_id: str | None = None) -> AppError:
    return AppError(E_REF, message, edit_seq=edit_seq, node_id=node_id)
