"""Deterministic segment capacity proof construction."""

from __future__ import annotations

from typing import Any, Iterable

from .canonical import canonical_bytes, canonical_text, digest_value, utf8_key


PROOF_FORMAT_VERSION = 1


def _proof_payload(
    *,
    segment_id: str,
    capacity_bytes: int,
    used_bytes: int,
    remaining_bytes: int,
    segment_set_version: int,
    segment_version: int,
    entries: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "format_version": PROOF_FORMAT_VERSION,
        "segment_id": segment_id,
        "capacity_bytes": capacity_bytes,
        "used_bytes": used_bytes,
        "remaining_bytes": remaining_bytes,
        "entry_count": len(entries),
        "segment_set_version": segment_set_version,
        "segment_version": segment_version,
        "entries": entries,
    }


def build_segment_proof(
    *,
    segment_id: str,
    capacity_bytes: int,
    segment_set_version: int,
    segment_version: int,
    entries: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    normalized_entries = [
        {
            "key": str(entry["entry_key"]),
            "size_bytes": int(entry["size_bytes"]),
            "digest": str(entry["digest"]),
        }
        for entry in entries
    ]
    normalized_entries.sort(key=lambda item: utf8_key(item["key"]))
    used_bytes = sum(entry["size_bytes"] for entry in normalized_entries)
    payload = _proof_payload(
        segment_id=segment_id,
        capacity_bytes=capacity_bytes,
        used_bytes=used_bytes,
        remaining_bytes=capacity_bytes - used_bytes,
        segment_set_version=segment_set_version,
        segment_version=segment_version,
        entries=normalized_entries,
    )
    payload["content_digest"] = digest_value(payload)
    return payload


def proof_json(proof: dict[str, Any]) -> str:
    return canonical_text(proof)


def validate_proof_shape(proof: Any) -> dict[str, Any]:
    from .errors import DiagnosticError

    if not isinstance(proof, dict):
        raise DiagnosticError(
            "E_INTEGRITY", "capacity proof must be an object", field="proof"
        )
    segment_id = proof.get("segment_id")
    field_prefix = "proof"
    if not isinstance(segment_id, str) or not segment_id:
        raise DiagnosticError(
            "E_INTEGRITY",
            "capacity proof has no segment_id",
            field=f"{field_prefix}.segment_id",
        )
    entries = proof.get("entries")
    if not isinstance(entries, list):
        raise DiagnosticError(
            "E_INTEGRITY",
            "capacity proof entries must be a list",
            segment_id=segment_id,
            field=f"{field_prefix}.entries",
        )
    for field_name, minimum in (
        ("capacity_bytes", 1),
        ("used_bytes", 0),
        ("remaining_bytes", 0),
        ("entry_count", 0),
        ("segment_set_version", 0),
        ("segment_version", 0),
    ):
        value = proof.get(field_name)
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise DiagnosticError(
                "E_INTEGRITY",
                "capacity proof field has an invalid integer value",
                segment_id=segment_id,
                field=f"{field_prefix}.{field_name}",
                details={"value": value, "minimum": minimum},
            )
    if proof["entry_count"] != len(entries):
        raise DiagnosticError(
            "E_INTEGRITY",
            "capacity proof entry_count does not match its entry list",
            segment_id=segment_id,
            field=f"{field_prefix}.entry_count",
            details={
                "entry_count": proof["entry_count"],
                "entries_length": len(entries),
            },
        )
    if proof["used_bytes"] + proof["remaining_bytes"] != proof["capacity_bytes"]:
        raise DiagnosticError(
            "E_INTEGRITY",
            "capacity proof bytes do not sum to capacity",
            segment_id=segment_id,
            field=f"{field_prefix}.used_bytes",
            details={
                "used_bytes": proof["used_bytes"],
                "remaining_bytes": proof["remaining_bytes"],
                "capacity_bytes": proof["capacity_bytes"],
            },
        )
    previous_key: str | None = None
    total_size = 0
    for entry in entries:
        if not isinstance(entry, dict):
            raise DiagnosticError(
                "E_INTEGRITY",
                "capacity proof entry must be an object",
                segment_id=segment_id,
                field=f"{field_prefix}.entries",
            )
        key = entry.get("key")
        if not isinstance(key, str) or not key:
            raise DiagnosticError(
                "E_INTEGRITY",
                "capacity proof entry key is invalid",
                segment_id=segment_id,
                field=f"{field_prefix}.entries.key",
            )
        size_bytes = entry.get("size_bytes")
        if (
            not isinstance(size_bytes, int)
            or isinstance(size_bytes, bool)
            or size_bytes < 0
        ):
            raise DiagnosticError(
                "E_INTEGRITY",
                "capacity proof entry size is invalid",
                segment_id=segment_id,
                entry_key=key,
                field=f"{field_prefix}.entries.size_bytes",
            )
        digest = entry.get("digest")
        if not isinstance(digest, str) or not digest.startswith("sha256:"):
            raise DiagnosticError(
                "E_INTEGRITY",
                "capacity proof entry digest is invalid",
                segment_id=segment_id,
                entry_key=key,
                field=f"{field_prefix}.entries.digest",
            )
        if previous_key is not None and utf8_key(key) <= utf8_key(previous_key):
            raise DiagnosticError(
                "E_INTEGRITY",
                "capacity proof entries are not strictly sorted",
                segment_id=segment_id,
                entry_key=key,
                field=f"{field_prefix}.entries",
            )
        total_size += size_bytes
        previous_key = key
    if total_size != proof["used_bytes"]:
        raise DiagnosticError(
            "E_INTEGRITY",
            "capacity proof used bytes do not match the entry sizes",
            segment_id=segment_id,
            field=f"{field_prefix}.used_bytes",
            details={"used_bytes": proof["used_bytes"], "entry_size_sum": total_size},
        )
    return proof


def verify_proof_digest(proof: dict[str, Any]) -> str:
    validate_proof_shape(proof)
    stored_digest = proof.get("content_digest")
    payload = {key: value for key, value in proof.items() if key != "content_digest"}
    expected_digest = digest_value(payload)
    if stored_digest != expected_digest:
        from .errors import DiagnosticError

        raise DiagnosticError(
            "E_INTEGRITY",
            "capacity proof content digest mismatch",
            segment_id=proof.get("segment_id"),
            field="proof.content_digest",
            details={
                "stored_digest": stored_digest,
                "computed_digest": expected_digest,
            },
        )
    return expected_digest


def proof_content_bytes(proof: dict[str, Any]) -> bytes:
    payload = {key: value for key, value in proof.items() if key != "content_digest"}
    return canonical_bytes(payload)
