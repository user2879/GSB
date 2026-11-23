"""Stable error codes for the multi-cluster configuration diagnostics console."""

from __future__ import annotations


class DiagError(Exception):
    """Domain error carrying a stable code and a structured location.

    ``location`` points at the offending entity, for example the revision
    version and field path that violated the specification.
    """

    def __init__(self, code: str, message: str, location: dict | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.location = location or {}

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "location": self.location}

    def __str__(self) -> str:
        loc = ""
        if self.location:
            loc = " location=" + canonical_json(self.location)
        return f"{self.code}: {self.message}{loc}"


def canonical_json(obj) -> str:
    """Stable JSON text used for byte-for-byte digests and serialization."""
    import json

    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def raise_spec(message: str, location: dict) -> None:
    raise DiagError("E_SPEC", message, location)


def raise_ref(message: str, location: dict) -> None:
    raise DiagError("E_REF", message, location)


def raise_snapshot(message: str, location: dict) -> None:
    raise DiagError("E_SNAPSHOT", message, location)
