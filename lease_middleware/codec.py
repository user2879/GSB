"""Encode / decode / normalize the three historical lease record formats.

r1: lease_id, config_key, expires_at_ms
r2: r1 fields + lease_version, client_id
r3: r2 fields + renewal_count, value_digest, parent_lease_id

Normalization fills missing fields with *explicit* defaults and never changes
``lease_version``; therefore the same lease represented in r1/r2/r3 normalizes
to one identical internal view (same version number and digest).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .errors import E_LEASE, MiddlewareError
from .spec import CURRENT_FORMAT, SUPPORTED_FORMATS, Spec

DEFAULT_LEASE_VERSION = 1
DEFAULT_RENEWAL_COUNT = 0
DEFAULT_PARENT = None

_REQUIRED = {
    "r1": ("lease_id", "config_key", "expires_at_ms"),
    "r2": ("lease_id", "config_key", "expires_at_ms", "lease_version", "client_id"),
    "r3": ("lease_id", "config_key", "expires_at_ms", "lease_version", "client_id",
           "renewal_count", "value_digest", "parent_lease_id"),
}


@dataclass(frozen=True)
class LeaseRecord:
    lease_id: str
    config_key: str
    expires_at_ms: int
    lease_version: int
    client_id: str
    renewal_count: int
    value_digest: str
    parent_lease_id: str | None
    format: str = CURRENT_FORMAT

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "lease_id": self.lease_id,
            "config_key": self.config_key,
            "expires_at_ms": self.expires_at_ms,
            "lease_version": self.lease_version,
            "client_id": self.client_id,
            "renewal_count": self.renewal_count,
            "value_digest": self.value_digest,
            "parent_lease_id": self.parent_lease_id,
        }


def make_lease_id(client_id: str, config_key: str, seq: int) -> str:
    return f"L:{client_id}:{config_key}:{seq:04d}"


def parse_client_id(lease_id: str) -> str:
    parts = lease_id.split(":", 3)
    if len(parts) != 4 or parts[0] != "L":
        raise MiddlewareError(E_LEASE, "cannot derive client id from lease id",
                              "lease_id", lease_id=lease_id)
    return parts[1]


def _require_fields(raw: dict[str, Any], fmt: str) -> None:
    for name in _REQUIRED[fmt]:
        if name not in raw:
            raise MiddlewareError(E_LEASE, f"{fmt} record missing required field {name!r}",
                                  name, lease_id=raw.get("lease_id"))


def _check_int(name: str, value: Any, lease_id: str | None) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise MiddlewareError(E_LEASE, f"field {name!r} must be a positive integer",
                              name, lease_id=lease_id)


def decode_record(raw: Any, spec: Spec, known_ids: set[str] | None = None,
                  version_override: int | None = None) -> LeaseRecord:
    """Decode one raw record dict into the normalized internal view."""
    if not isinstance(raw, dict):
        raise MiddlewareError(E_LEASE, "lease record must be a JSON object", "record")
    fmt = raw.get("format")
    if fmt not in SUPPORTED_FORMATS:
        raise MiddlewareError(E_LEASE, f"unsupported record format {fmt!r}", "format",
                              lease_id=raw.get("lease_id"))
    _require_fields(raw, fmt)
    lease_id = raw["lease_id"]
    if not isinstance(lease_id, str) or not lease_id:
        raise MiddlewareError(E_LEASE, "lease_id must be a non-empty string", "lease_id")
    config_key = raw["config_key"]
    if not isinstance(config_key, str) or config_key not in spec.configs:
        raise MiddlewareError(E_LEASE, f"unknown config key {config_key!r}",
                              "config_key", lease_id=lease_id)
    _check_int("expires_at_ms", raw["expires_at_ms"], lease_id)

    lease_version = raw.get("lease_version", version_override or DEFAULT_LEASE_VERSION)
    _check_int("lease_version", lease_version, lease_id)
    config = spec.configs[config_key]
    try:
        expected_digest = config.digest_at(lease_version)
    except KeyError:
        raise MiddlewareError(
            E_LEASE, f"lease_version {lease_version} has no matching config value",
            "lease_version", lease_id=lease_id,
        )

    if fmt == "r1":
        client_id = parse_client_id(lease_id)
        renewal_count = DEFAULT_RENEWAL_COUNT
        digest = expected_digest                 # r1: digest derived, never invented version
        parent = DEFAULT_PARENT
    else:
        client_id = raw["client_id"]
        if not isinstance(client_id, str) or not client_id:
            raise MiddlewareError(E_LEASE, "client_id must be a non-empty string",
                                  "client_id", lease_id=lease_id)
        if fmt == "r2":
            renewal_count = DEFAULT_RENEWAL_COUNT
            digest = expected_digest
            parent = DEFAULT_PARENT
        else:
            renewal_count = raw["renewal_count"]
            if not isinstance(renewal_count, int) or isinstance(renewal_count, bool) or renewal_count < 0:
                raise MiddlewareError(E_LEASE, "renewal_count must be a non-negative integer",
                                      "renewal_count", lease_id=lease_id)
            digest = raw["value_digest"]
            if not isinstance(digest, str) or digest != expected_digest:
                raise MiddlewareError(
                    E_LEASE, "value_digest does not match the config item value",
                    "value_digest", lease_id=lease_id, client_id=client_id,
                    context={"expected": expected_digest, "actual": digest},
                )
            parent = raw["parent_lease_id"]
            if parent is not None:
                if not isinstance(parent, str):
                    raise MiddlewareError(E_LEASE, "parent_lease_id must be string or null",
                                          "parent_lease_id", lease_id=lease_id)
                if known_ids is not None and parent not in known_ids:
                    raise MiddlewareError(E_LEASE, f"parent lease {parent!r} does not exist",
                                          "parent_lease_id", lease_id=lease_id,
                                          client_id=client_id)
    return LeaseRecord(
        lease_id=lease_id,
        config_key=config_key,
        expires_at_ms=raw["expires_at_ms"],
        lease_version=lease_version,
        client_id=client_id,
        renewal_count=renewal_count,
        value_digest=digest,
        parent_lease_id=parent,
        format=fmt,
    )


def decode_records(raws: Iterable[Any], spec: Spec) -> list[LeaseRecord]:
    """Decode a batch, validating r3 parent references across the whole set.

    Invalid records raise E_LEASE; the caller is responsible for keeping any
    previously accepted valid record untouched.
    """
    raws = list(raws)
    known = {r["lease_id"] for r in raws if isinstance(r, dict) and isinstance(r.get("lease_id"), str)}
    # Resolve the authoritative version for a lease from its richer r2/r3
    # representations so an r1 row normalizes to the identical view.
    version_by_lease: dict[str, int] = {}
    for raw in raws:
        if isinstance(raw, dict) and "lease_version" in raw:
            version_by_lease[raw["lease_id"]] = raw["lease_version"]
    return [decode_record(raw, spec, known_ids=known,
                          version_override=version_by_lease.get(
                              raw.get("lease_id") if isinstance(raw, dict) else None))
            for raw in raws]


def encode_record(record: LeaseRecord, fmt: str) -> dict[str, Any]:
    if fmt not in SUPPORTED_FORMATS:
        raise MiddlewareError(E_LEASE, f"cannot encode to unsupported format {fmt!r}", "format")
    full = record.to_dict()
    keep = ["format"] + list(_REQUIRED[fmt])
    out = {key: full[key] for key in keep}
    out["format"] = fmt
    return out


def upgrade_record(record: LeaseRecord) -> LeaseRecord:
    """Record-level in-memory upgrade to r3; lease_version is preserved."""
    if record.format == CURRENT_FORMAT:
        return record
    return LeaseRecord(
        lease_id=record.lease_id,
        config_key=record.config_key,
        expires_at_ms=record.expires_at_ms,
        lease_version=record.lease_version,
        client_id=record.client_id,
        renewal_count=record.renewal_count,
        value_digest=record.value_digest,
        parent_lease_id=record.parent_lease_id,
        format=CURRENT_FORMAT,
    )
