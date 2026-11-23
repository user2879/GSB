from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable

from .canonical import canonical_json, is_int, sha256_text
from .errors import LabError


SUPPORTED_FORMATS = {"h1", "h2", "h3"}
DESTINATIONS = {"stable", "canary"}
SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
PARAM_DIGEST_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
IDENTIFIER_RE = re.compile(r"^[^\x00-\x1f\x7f]{1,256}$")

FORMAT_BASE_FIELDS = {"h", "batch_id", "destination", "output_digest"}
FORMAT_H2_FIELDS = FORMAT_BASE_FIELDS | {"client_bucket", "weight_bp", "duration_ms"}
FORMAT_H3_FIELDS = FORMAT_H2_FIELDS | {
    "sample_ids",
    "backend_param_digest",
    "merge_seq",
}
FORMAT_FIELDS = {"h1": FORMAT_BASE_FIELDS, "h2": FORMAT_H2_FIELDS, "h3": FORMAT_H3_FIELDS}
KNOWN_FIELDS = {
    "h1": FORMAT_BASE_FIELDS,
    "h2": FORMAT_H2_FIELDS,
    "h3": FORMAT_H3_FIELDS | {"backend_param_version"},
}
FORMAT_IMPLIED_SEQUENCE = {"h1": 0, "h2": 1}


def _format_error(message: str, field: str | None = None, batch_id: str | None = None, **details: Any) -> LabError:
    return LabError("E_FORMAT", message, field=field, batch_id=batch_id, details=details)


@dataclass(frozen=True)
class Receipt:
    format: str
    batch_id: str
    destination: str
    output_digest: str
    client_bucket: int | None = None
    weight_bp: int | None = None
    duration_ms: int | None = None
    sample_ids: tuple[str, ...] = ()
    backend_param_digest: str | None = None
    backend_param_version: int | None = None
    merge_seq: int | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def effective_merge_seq(self) -> int:
        if self.format == "h3":
            if self.merge_seq is None:
                raise _format_error("h3 merge_seq is missing", "merge_seq", self.batch_id)
            return self.merge_seq
        return FORMAT_IMPLIED_SEQUENCE[self.format]

    def identity_digest(self) -> str:
        return sha256_text(canonical_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        obj: dict[str, Any] = {
            "h": self.format,
            "batch_id": self.batch_id,
            "destination": self.destination,
            "output_digest": self.output_digest,
        }
        if self.format in {"h2", "h3"}:
            obj.update(
                {
                    "client_bucket": self.client_bucket,
                    "weight_bp": self.weight_bp,
                    "duration_ms": self.duration_ms,
                }
            )
        if self.format == "h3":
            obj.update(
                {
                    "sample_ids": list(self.sample_ids),
                    "backend_param_digest": self.backend_param_digest,
                    "merge_seq": self.merge_seq,
                }
            )
            if self.backend_param_version is not None:
                obj["backend_param_version"] = self.backend_param_version
        for key in sorted(self.extras):
            if key not in obj:
                obj[key] = self.extras[key]
        return obj


def _require_nonempty_string(obj: dict[str, Any], key: str, batch_id: str | None, *, identifier: bool = False) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        raise _format_error(f"required field {key!r} must be a non-empty string", key, batch_id)
    if identifier and not IDENTIFIER_RE.fullmatch(value):
        raise _format_error(f"field {key!r} has an invalid identifier format", key, batch_id)
    return value


def _require_digest(obj: dict[str, Any], key: str, batch_id: str | None, regex: re.Pattern[str]) -> str:
    value = _require_nonempty_string(obj, key, batch_id)
    if not regex.fullmatch(value):
        raise _format_error(f"field {key!r} is not a valid sha256 digest", key, batch_id)
    return value


def _require_int(
    obj: dict[str, Any],
    key: str,
    batch_id: str | None,
    minimum: int,
    maximum: int | None = None,
) -> int:
    value = obj.get(key)
    if not is_int(value) or value < minimum or (maximum is not None and value > maximum):
        upper = "" if maximum is None else f"..{maximum}"
        raise _format_error(f"field {key!r} must be an integer in {minimum}{upper}", key, batch_id)
    return int(value)


def decode_receipt(raw: Any) -> Receipt:
    if not isinstance(raw, dict):
        raise _format_error("receipt must be a JSON object", "$")
    version = raw.get("h")
    batch_id = raw.get("batch_id") if isinstance(raw.get("batch_id"), str) else None
    if not isinstance(version, str) or version not in SUPPORTED_FORMATS:
        raise _format_error("unsupported or missing receipt version field 'h'", "h", batch_id)
    required = FORMAT_FIELDS[version]
    missing = sorted(field for field in required if field not in raw)
    extras = {key: raw[key] for key in sorted(raw) if key not in KNOWN_FIELDS[version]}
    if missing:
        raise _format_error(
            f"required field {missing[0]!r} is missing",
            missing[0],
            batch_id,
            pending_candidates=sorted(extras),
        )

    batch_id = _require_nonempty_string(raw, "batch_id", batch_id, identifier=True)
    destination = _require_nonempty_string(raw, "destination", batch_id)
    if destination not in DESTINATIONS:
        raise _format_error("destination must be 'stable' or 'canary'", "destination", batch_id)
    output_digest = _require_digest(raw, "output_digest", batch_id, SHA256_RE)

    client_bucket = weight_bp = duration_ms = None
    sample_ids: tuple[str, ...] = ()
    param_digest = None
    param_version = None
    merge_seq = None
    if version in {"h2", "h3"}:
        client_bucket = _require_int(raw, "client_bucket", batch_id, 0, 10000)
        weight_bp = _require_int(raw, "weight_bp", batch_id, 0, 10000)
        duration_ms = _require_int(raw, "duration_ms", batch_id, 0)
    if version == "h3":
        sample_values = raw.get("sample_ids")
        if not isinstance(sample_values, list) or not sample_values:
            raise _format_error("field 'sample_ids' must be a non-empty array", "sample_ids", batch_id)
        normalized_ids: list[str] = []
        seen: set[str] = set()
        for index, item in enumerate(sample_values):
            if not isinstance(item, str) or not item or not IDENTIFIER_RE.fullmatch(item):
                raise _format_error(
                    "sample_ids entries must be non-empty identifiers", f"sample_ids[{index}]", batch_id
                )
            if item in seen:
                raise _format_error("sample_ids must be unique within a receipt", f"sample_ids[{index}]", batch_id)
            seen.add(item)
            normalized_ids.append(item)
        sample_ids = tuple(sorted(normalized_ids))
        param_digest = _require_digest(raw, "backend_param_digest", batch_id, PARAM_DIGEST_RE)
        merge_seq = _require_int(raw, "merge_seq", batch_id, 0)
        if "backend_param_version" in raw:
            param_version = _require_int(raw, "backend_param_version", batch_id, 1)

    return Receipt(
        format=version,
        batch_id=batch_id,
        destination=destination,
        output_digest=output_digest,
        client_bucket=client_bucket,
        weight_bp=weight_bp,
        duration_ms=duration_ms,
        sample_ids=sample_ids,
        backend_param_digest=param_digest,
        backend_param_version=param_version,
        merge_seq=merge_seq,
        extras=extras,
    )


def decode_receipts(raw_receipts: Iterable[Any]) -> list[Receipt]:
    receipts = [
        item if isinstance(item, Receipt) else decode_receipt(item) for item in raw_receipts
    ]
    seen: set[tuple[str, int]] = set()
    for receipt in receipts:
        key = (receipt.batch_id, receipt.effective_merge_seq)
        if key in seen:
            raise _format_error(
                "duplicate (batch_id, merge_seq) within receipt collection",
                "merge_seq",
                receipt.batch_id,
                merge_seq=receipt.effective_merge_seq,
            )
        seen.add(key)
    return receipts


def make_receipt(
    version: str,
    batch_id: str,
    destination: str,
    output_digest: str,
    *,
    client_bucket: int | None = None,
    weight_bp: int | None = None,
    duration_ms: int | None = None,
    sample_ids: Iterable[str] = (),
    backend_param_digest: str | None = None,
    backend_param_version: int | None = None,
    merge_seq: int | None = None,
    extras: dict[str, Any] | None = None,
) -> Receipt:
    if version == "h1":
        raw = {
            "h": version,
            "batch_id": batch_id,
            "destination": destination,
            "output_digest": output_digest,
        }
    elif version == "h2":
        raw = {
            "h": version,
            "batch_id": batch_id,
            "destination": destination,
            "output_digest": output_digest,
            "client_bucket": client_bucket,
            "weight_bp": weight_bp,
            "duration_ms": duration_ms,
        }
    elif version == "h3":
        raw = {
            "h": version,
            "batch_id": batch_id,
            "destination": destination,
            "output_digest": output_digest,
            "client_bucket": client_bucket,
            "weight_bp": weight_bp,
            "duration_ms": duration_ms,
            "sample_ids": list(sample_ids),
            "backend_param_digest": backend_param_digest,
            "merge_seq": merge_seq,
        }
        if backend_param_version is not None:
            raw["backend_param_version"] = backend_param_version
    else:
        raise _format_error("unsupported receipt version", "h", batch_id)
    if extras:
        for key, value in extras.items():
            raw.setdefault(key, value)
    return decode_receipt(raw)


def encode_receipt(receipt: Receipt, version: str | None = None) -> dict[str, Any]:
    target = version or receipt.format
    if target not in SUPPORTED_FORMATS:
        raise _format_error("unsupported receipt version", "h", receipt.batch_id)
    if target in {"h2", "h3"}:
        if receipt.client_bucket is None or receipt.weight_bp is None or receipt.duration_ms is None:
            raise _format_error(
                "receipt lacks h2 fields required for the target format", "client_bucket", receipt.batch_id
            )
    if target == "h3":
        if (
            not receipt.sample_ids
            or receipt.backend_param_digest is None
            or receipt.merge_seq is None
        ):
            raise _format_error(
                "receipt lacks h3 fields required for the target format", "sample_ids", receipt.batch_id
            )
    clone = Receipt(
        format=target,
        batch_id=receipt.batch_id,
        destination=receipt.destination,
        output_digest=receipt.output_digest,
        client_bucket=receipt.client_bucket,
        weight_bp=receipt.weight_bp,
        duration_ms=receipt.duration_ms,
        sample_ids=receipt.sample_ids if target == "h3" else (),
        backend_param_digest=receipt.backend_param_digest if target == "h3" else None,
        backend_param_version=receipt.backend_param_version if target == "h3" else None,
        merge_seq=receipt.merge_seq if target == "h3" else None,
        extras=dict(receipt.extras),
    )
    return clone.to_dict()
