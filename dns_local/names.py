"""Canonical DNS presentation-name handling."""

from __future__ import annotations

import re

from .errors import BusinessError


_LABEL_RE = re.compile(r"^[A-Za-z0-9_](?:[A-Za-z0-9_-]{0,61}[A-Za-z0-9_])?$")


def _validate_label(label: str, *, wildcard_allowed: bool, index: int) -> None:
    if wildcard_allowed and index == 0 and label == "*":
        return
    if not label or len(label.encode("ascii", "strict")) > 63:
        raise ValueError(f"label {index} must contain 1..63 ASCII bytes")
    if not _LABEL_RE.match(label):
        raise ValueError(f"label {index} contains unsupported characters")


def normalize_name(name: str, *, allow_wildcard: bool = False, allow_root: bool = True) -> str:
    """Return an absolute, ASCII-lower-cased presentation name.

    The deliberately small character set is appropriate for local laboratory
    zones and keeps JSON/wire round trips deterministic.
    """

    if not isinstance(name, str):
        raise ValueError("name must be a string")
    value = name.strip()
    if not value:
        raise ValueError("name must not be empty")
    if value == ".":
        if not allow_root:
            raise ValueError("root name is not allowed here")
        return "."
    if value.endswith("."):
        value = value[:-1]
    if not value:
        if not allow_root:
            raise ValueError("root name is not allowed here")
        return "."
    labels = value.split(".")
    if any(label == "" for label in labels):
        raise ValueError("name contains an empty label")
    for index, label in enumerate(labels):
        _validate_label(label, wildcard_allowed=allow_wildcard, index=index)
    wire_len = 1 + sum(1 + len(label.encode("ascii")) for label in labels)
    if wire_len > 255:
        raise ValueError("name exceeds 255 wire bytes")
    return ".".join(label.lower() for label in labels) + "."


def try_normalize_name(name: str, **kwargs: object) -> str:
    return normalize_name(name, **kwargs)  # type: ignore[arg-type]


def normalize_zone_name(name: str) -> str:
    normalized = normalize_name(name, allow_wildcard=False, allow_root=False)
    return normalized


def is_subdomain(name: str, zone: str) -> bool:
    normalized = normalize_name(name, allow_wildcard=True)
    apex = normalize_zone_name(zone)
    return normalized == apex or normalized.endswith("." + apex)


def labels(name: str) -> list[str]:
    normalized = normalize_name(name)
    if normalized == ".":
        return []
    return normalized[:-1].split(".")


def parent_name(name: str) -> str:
    parts = labels(name)
    if len(parts) <= 1:
        return "."
    return ".".join(parts[1:]) + "."


def cname_target_from_targets(target: str) -> str:
    return normalize_name(target, allow_wildcard=False, allow_root=True)


def wire_name_to_text(raw: bytes) -> str:
    if not raw:
        return "."
    return raw.decode("ascii") + "."


def zone_field_error(message: str, record_index: int, field: str) -> BusinessError:
    return BusinessError("E_ZONE", message, record_index=record_index, field=field)
