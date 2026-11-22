"""m1/m2/m3 snapshot manifest codecs and lossless normalization metadata."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol

from .bitmap import bitmap_checksum, bitmap_from_indices, bitmap_length, set_bits, validate_bitmap
from .errors import DiagnosticError, E_FORMAT
from .util import canonical_json_bytes, normalize_digest


FORMAT_VERSIONS = ("m1", "m2", "m3")

_COMMON_TOP_FIELDS = {"format_version", "volume_block_count", "block_size", "snapshots"}
_TOP_FIELDS = {
    "m1": _COMMON_TOP_FIELDS,
    "m2": _COMMON_TOP_FIELDS,
    "m3": _COMMON_TOP_FIELDS | {"manifest_digest"},
}
_COMMON_ENTRY_FIELDS = {
    "snapshot_id",
    "volume_block_count",
    "bitmap_length",
    "bitmap",
}
_ENTRY_FIELDS = {
    "m1": _COMMON_ENTRY_FIELDS,
    "m2": _COMMON_ENTRY_FIELDS | {"digest_table", "bitmap_checksum", "snapshot_version"},
    "m3": _COMMON_ENTRY_FIELDS
    | {
        "digest_table",
        "bitmap_checksum",
        "snapshot_version",
        "parent_snapshot_id",
        "block_refs",
        "content_digest",
    },
}
_REQUIRED_ENTRY_FIELDS = {
    "m1": {"snapshot_id", "volume_block_count", "bitmap_length"},
    "m2": _ENTRY_FIELDS["m2"],
    "m3": _ENTRY_FIELDS["m3"],
}


class ManifestResolver(Protocol):
    """Disk-state lookup used to normalize fields absent from old formats."""

    def resolve_snapshot(self, snapshot_id: str) -> Mapping[str, Any] | None:
        ...


class DictManifestResolver:
    """Small resolver useful in tests and for normalized in-memory sessions."""

    def __init__(self, records: Mapping[str, Mapping[str, Any]]):
        self.records = records

    def resolve_snapshot(self, snapshot_id: str) -> Mapping[str, Any] | None:
        return self.records.get(snapshot_id)


@dataclass(frozen=True)
class SnapshotManifest:
    snapshot_id: str
    block_count: int
    bitmap: bytes
    bitmap_length: int
    digest_table: dict[int, str]
    block_refs: dict[int, str]
    parent_snapshot_id: str | None
    bitmap_checksum: str
    content_digest: str
    snapshot_version: int
    source_format: str
    filled_fields: tuple[str, ...] = ()
    fill_basis: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "volume_block_count": self.block_count,
            "bitmap_length": self.bitmap_length,
            "bitmap": bytes(self.bitmap).hex(),
            "digest_table": [[index, digest] for index, digest in sorted(self.digest_table.items())],
            "bitmap_checksum": self.bitmap_checksum,
            "snapshot_version": self.snapshot_version,
            "parent_snapshot_id": self.parent_snapshot_id,
            "block_refs": [[index, ref] for index, ref in sorted(self.block_refs.items())],
            "content_digest": self.content_digest,
            "source_format": self.source_format,
            "filled_fields": list(self.filled_fields),
            "fill_basis": dict(sorted(self.fill_basis.items())),
        }


@dataclass(frozen=True)
class NormalizedManifest:
    source_format: str
    block_size: int | None
    block_count: int
    snapshots: tuple[SnapshotManifest, ...]
    manifest_digest: str
    filled_fields: tuple[str, ...] = ()
    fill_basis: dict[str, str] = field(default_factory=dict)

    def by_id(self) -> dict[str, SnapshotManifest]:
        return {snapshot.snapshot_id: snapshot for snapshot in self.snapshots}

    def fingerprint(self) -> str:
        payload = {
            "block_size": self.block_size,
            "block_count": self.block_count,
            "snapshots": [
                {
                    "snapshot_id": snapshot.snapshot_id,
                    "volume_block_count": snapshot.block_count,
                    "bitmap": bytes(snapshot.bitmap).hex(),
                    "digest_table": [[index, digest] for index, digest in sorted(snapshot.digest_table.items())],
                    "block_refs": [[index, ref] for index, ref in sorted(snapshot.block_refs.items())],
                    "parent_snapshot_id": snapshot.parent_snapshot_id,
                    "snapshot_version": snapshot.snapshot_version,
                }
                for snapshot in sorted(self.snapshots, key=lambda item: item.snapshot_id)
            ],
        }
        return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_format": self.source_format,
            "block_size": self.block_size,
            "volume_block_count": self.block_count,
            "snapshots": [snapshot.as_dict() for snapshot in self.snapshots],
            "manifest_digest": self.manifest_digest,
            "filled_fields": list(self.filled_fields),
            "fill_basis": dict(sorted(self.fill_basis.items())),
            "fingerprint": self.fingerprint(),
        }


def peek_format_version(data: bytes | str | Mapping[str, Any]) -> str | None:
    try:
        if isinstance(data, Mapping):
            value = data.get("format_version")
        else:
            raw = data.decode("utf-8") if isinstance(data, bytes) else data
            value = json.loads(raw).get("format_version")
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, str) else None


def _first_snapshot_id(document: Mapping[str, Any]) -> str | None:
    snapshots = document.get("snapshots")
    if isinstance(snapshots, list):
        for item in snapshots:
            if isinstance(item, Mapping):
                value = item.get("snapshot_id")
                if isinstance(value, str) and value:
                    return value
    return None


def _format_error(
    message: str,
    *,
    snapshot_id: str | None,
    field_name: str,
    **details: Any,
) -> DiagnosticError:
    return DiagnosticError(
        E_FORMAT,
        message,
        snapshot_id=snapshot_id,
        field=field_name,
        details=details,
    )


def _require_int(value: Any, *, snapshot_id: str | None, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _format_error(
            f"{field_name} must be an integer",
            snapshot_id=snapshot_id,
            field_name=field_name,
            actual=value,
        )
    return value


def _parse_block_map(
    value: Any,
    *,
    snapshot_id: str,
    field_name: str,
    block_count: int,
    value_field: str,
) -> dict[int, str]:
    result: dict[int, str] = {}
    if isinstance(value, dict):
        items = list(value.items())
        for raw_index, raw_digest in items:
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                raise _format_error(
                    f"{field_name} contains a non-integer block index",
                    snapshot_id=snapshot_id,
                    field_name=field_name,
                    actual=raw_index,
                )
            if str(index) != str(raw_index).strip() and not isinstance(raw_index, int):
                raise _format_error(
                    f"{field_name} block index must be a canonical integer",
                    snapshot_id=snapshot_id,
                    field_name=field_name,
                    actual=raw_index,
                )
            digest = normalize_digest(raw_digest, snapshot_id=snapshot_id, field_name=field_name)
            if index in result:
                raise _format_error(
                    f"{field_name} contains a duplicate block index",
                    snapshot_id=snapshot_id,
                    field_name=field_name,
                    block_index=index,
                )
            result[index] = digest
    elif isinstance(value, list):
        for item in value:
            if not isinstance(item, Mapping):
                raise _format_error(
                    f"{field_name} entries must be objects",
                    snapshot_id=snapshot_id,
                    field_name=field_name,
                )
            index_value = item.get("block", item.get("block_index", item.get("index")))
            digest_value = item.get(value_field, item.get("digest", item.get("ref")))
            index = _require_int(index_value, snapshot_id=snapshot_id, field_name=field_name)
            digest = normalize_digest(digest_value, snapshot_id=snapshot_id, field_name=field_name)
            if index in result:
                raise _format_error(
                    f"{field_name} contains a duplicate block index",
                    snapshot_id=snapshot_id,
                    field_name=field_name,
                    block_index=index,
                )
            result[index] = digest
    else:
        raise _format_error(
            f"{field_name} must be a list or object",
            snapshot_id=snapshot_id,
            field_name=field_name,
        )
    for index in sorted(result):
        if index < 0 or index >= block_count:
            raise _format_error(
                f"{field_name} block index is outside the volume",
                snapshot_id=snapshot_id,
                field_name=field_name,
                block_index=index,
                block_count=block_count,
            )
    return result


def _encoded_entry(snapshot: SnapshotManifest, version: str) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "snapshot_id": snapshot.snapshot_id,
        "volume_block_count": snapshot.block_count,
        "bitmap_length": snapshot.bitmap_length,
        "bitmap": bytes(snapshot.bitmap).hex(),
    }
    if version in ("m2", "m3"):
        entry["digest_table"] = [
            {"block": index, "digest": digest}
            for index, digest in sorted(snapshot.digest_table.items())
        ]
        entry["bitmap_checksum"] = snapshot.bitmap_checksum
        entry["snapshot_version"] = snapshot.snapshot_version
    if version == "m3":
        entry["parent_snapshot_id"] = snapshot.parent_snapshot_id
        entry["block_refs"] = [
            {"block": index, "ref": ref} for index, ref in sorted(snapshot.block_refs.items())
        ]
        entry["content_digest"] = snapshot.content_digest
    return entry


def _entry_content_digest(entry: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in entry.items() if key != "content_digest"}
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _snapshot_content_payload(
    *,
    snapshot_id: str,
    block_count: int,
    bitmap: bytes,
    digest_table: Mapping[int, str],
    block_refs: Mapping[int, str],
    parent_snapshot_id: str | None,
    snapshot_version: int,
) -> dict[str, Any]:
    return {
        "snapshot_id": snapshot_id,
        "volume_block_count": block_count,
        "bitmap_length": bitmap_length(block_count),
        "bitmap": bytes(bitmap).hex(),
        "digest_table": [
            {"block": index, "digest": digest} for index, digest in sorted(digest_table.items())
        ],
        "bitmap_checksum": bitmap_checksum(bitmap),
        "snapshot_version": snapshot_version,
        "parent_snapshot_id": parent_snapshot_id,
        "block_refs": [
            {"block": index, "ref": ref} for index, ref in sorted(block_refs.items())
        ],
    }


def compute_content_digest(
    *,
    snapshot_id: str,
    block_count: int,
    bitmap: bytes,
    digest_table: Mapping[int, str],
    block_refs: Mapping[int, str],
    parent_snapshot_id: str | None,
    snapshot_version: int,
) -> str:
    payload = _snapshot_content_payload(
        snapshot_id=snapshot_id,
        block_count=block_count,
        bitmap=bitmap,
        digest_table=digest_table,
        block_refs=block_refs,
        parent_snapshot_id=parent_snapshot_id,
        snapshot_version=snapshot_version,
    )
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _resolved_fill(
    resolver: ManifestResolver | None,
    snapshot_id: str,
) -> Mapping[str, Any] | None:
    if resolver is None:
        return None
    record = resolver.resolve_snapshot(snapshot_id)
    if record is None:
        raise _format_error(
            "manifest entry cannot be normalized because disk state is missing",
            snapshot_id=snapshot_id,
            field_name="snapshot_id",
        )
    return record


def _fill_digest_map(
    record: Mapping[str, Any] | None,
    *,
    key: str,
    snapshot_id: str,
    block_count: int,
) -> dict[int, str] | None:
    if record is None:
        return None
    raw = record.get(key)
    if raw is None:
        return None
    return _parse_block_map(
        raw,
        snapshot_id=snapshot_id,
        field_name=key,
        block_count=block_count,
        value_field="digest" if key == "digest_table" else "ref",
    )


def _decode_entry(
    raw: Any,
    *,
    version: str,
    block_count: int,
    resolver: ManifestResolver | None,
) -> SnapshotManifest:
    if not isinstance(raw, Mapping):
        raise _format_error(
            "snapshot entry must be an object",
            snapshot_id=None,
            field_name="snapshots",
        )
    snapshot_id_value = raw.get("snapshot_id")
    snapshot_id = snapshot_id_value if isinstance(snapshot_id_value, str) and snapshot_id_value else None
    if snapshot_id is None:
        raise _format_error(
            "snapshot_id must be a non-empty string",
            snapshot_id=None,
            field_name="snapshot_id",
            actual=snapshot_id_value,
        )
    allowed = _ENTRY_FIELDS[version]
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise _format_error(
            "snapshot entry contains fields not supported by this manifest version",
            snapshot_id=snapshot_id,
            field_name=unknown[0],
            unknown_fields=unknown,
            format_version=version,
        )
    missing = sorted(_REQUIRED_ENTRY_FIELDS[version] - set(raw))
    if missing:
        raise _format_error(
            "snapshot entry is missing required fields",
            snapshot_id=snapshot_id,
            field_name=missing[0],
            missing_fields=missing,
            format_version=version,
        )
    entry_block_count = _require_int(
        raw.get("volume_block_count"),
        snapshot_id=snapshot_id,
        field_name="volume_block_count",
    )
    if entry_block_count != block_count:
        raise _format_error(
            "snapshot volume_block_count does not match the manifest",
            snapshot_id=snapshot_id,
            field_name="volume_block_count",
            expected=block_count,
            actual=entry_block_count,
        )
    length_value = _require_int(
        raw.get("bitmap_length"),
        snapshot_id=snapshot_id,
        field_name="bitmap_length",
    )
    record = _resolved_fill(resolver, snapshot_id) if version in ("m1", "m2") else None
    bitmap_hex = raw.get("bitmap")
    bitmap_filled = False
    if bitmap_hex is None and version == "m1" and record is not None:
        resolver_bitmap = record.get("bitmap")
        if isinstance(resolver_bitmap, (bytes, bytearray)):
            bitmap_hex = bytes(resolver_bitmap).hex()
        elif isinstance(resolver_bitmap, str):
            bitmap_hex = resolver_bitmap
        if bitmap_hex is None and record is not None:
            source_indices = record.get("digest_table", record.get("block_refs"))
            if isinstance(source_indices, Mapping):
                try:
                    bitmap_hex = bytes(
                        bitmap_from_indices(block_count, (int(index) for index in source_indices))
                    ).hex()
                except (TypeError, ValueError):
                    bitmap_hex = None
        if bitmap_hex is not None:
            bitmap_filled = True
    if not isinstance(bitmap_hex, str):
        raise _format_error(
            "bitmap must be a hexadecimal string or available from disk state for m1",
            snapshot_id=snapshot_id,
            field_name="bitmap",
            fill_source="snapshot_bitmap" if version == "m1" else None,
        )
    try:
        bitmap = bytes.fromhex(bitmap_hex)
    except ValueError as exc:
        raise _format_error(
            "bitmap contains non-hexadecimal characters",
            snapshot_id=snapshot_id,
            field_name="bitmap",
            reason=str(exc),
        )
    validate_bitmap(
        bitmap,
        block_count,
        snapshot_id=snapshot_id,
        bitmap_length_value=length_value,
    )
    bits = set_bits(bitmap, block_count)
    filled: list[str] = []
    basis: dict[str, str] = {}
    if bitmap_filled:
        filled.append("bitmap")
        basis["bitmap"] = "sqlite_snapshots"

    if version == "m1":
        digest_table = _fill_digest_map(
            record, key="digest_table", snapshot_id=snapshot_id, block_count=block_count
        )
        if digest_table is None:
            digest_table = _fill_digest_map(
                record, key="block_refs", snapshot_id=snapshot_id, block_count=block_count
            )
        if digest_table is None:
            raise _format_error(
                "m1 has no digest table; normalization requires disk block references",
                snapshot_id=snapshot_id,
                field_name="digest_table",
                fill_source="snapshot_blocks",
            )
        block_refs = _fill_digest_map(
            record, key="block_refs", snapshot_id=snapshot_id, block_count=block_count
        ) or dict(digest_table)
        parent = record.get("parent_snapshot_id") if record is not None else None
        snapshot_version = int(record.get("snapshot_version", 1)) if record is not None else 1
        checksum = bitmap_checksum(bitmap)
        filled.extend(["digest_table", "block_refs", "parent_snapshot_id", "bitmap_checksum", "content_digest", "snapshot_version"])
        basis.update(
            {
                "digest_table": "sqlite_snapshot_blocks",
                "block_refs": "sqlite_snapshot_blocks",
                "parent_snapshot_id": "sqlite_snapshots",
                "bitmap_checksum": "computed_sha256(bitmap)",
                "content_digest": "computed_from_normalized_m3_fields",
                "snapshot_version": "sqlite_snapshots",
            }
        )
    elif version == "m2":
        digest_table = _parse_block_map(
            raw.get("digest_table"),
            snapshot_id=snapshot_id,
            field_name="digest_table",
            block_count=block_count,
            value_field="digest",
        )
        if len(digest_table) != len(bits):
            raise _format_error(
                "digest_table item count must equal the number of set bitmap bits",
                snapshot_id=snapshot_id,
                field_name="digest_table",
                expected=len(bits),
                actual=len(digest_table),
                missing_indices=sorted(set(bits) - set(digest_table)),
                extra_indices=sorted(set(digest_table) - set(bits)),
            )
        if set(digest_table) != set(bits):
            raise _format_error(
                "digest_table block indices must exactly match set bitmap bits",
                snapshot_id=snapshot_id,
                field_name="digest_table",
                missing_indices=sorted(set(bits) - set(digest_table)),
                extra_indices=sorted(set(digest_table) - set(bits)),
            )
        checksum_value = normalize_digest(
            raw.get("bitmap_checksum"),
            snapshot_id=snapshot_id,
            field_name="bitmap_checksum",
        )
        expected_checksum = bitmap_checksum(bitmap)
        if checksum_value != expected_checksum:
            raise _format_error(
                "bitmap_checksum does not match bitmap content",
                snapshot_id=snapshot_id,
                field_name="bitmap_checksum",
                expected=f"sha256:{expected_checksum}",
                actual=f"sha256:{checksum_value}",
            )
        checksum = checksum_value
        parent = record.get("parent_snapshot_id") if record is not None else None
        block_refs = _fill_digest_map(
            record, key="block_refs", snapshot_id=snapshot_id, block_count=block_count
        )
        if block_refs is None:
            block_refs = dict(digest_table)
            basis["block_refs"] = "copied_from_m2_digest_table"
        else:
            basis["block_refs"] = "sqlite_snapshot_blocks"
        snapshot_version = _require_int(
            raw.get("snapshot_version"),
            snapshot_id=snapshot_id,
            field_name="snapshot_version",
        )
        filled.extend(["parent_snapshot_id", "block_refs", "content_digest"])
        basis.setdefault("parent_snapshot_id", "sqlite_snapshots" if record is not None else "m2_has_no_parent")
        basis.setdefault("content_digest", "computed_from_normalized_m3_fields")
    else:
        digest_table = _parse_block_map(
            raw.get("digest_table"),
            snapshot_id=snapshot_id,
            field_name="digest_table",
            block_count=block_count,
            value_field="digest",
        )
        block_refs = _parse_block_map(
            raw.get("block_refs"),
            snapshot_id=snapshot_id,
            field_name="block_refs",
            block_count=block_count,
            value_field="ref",
        )
        if set(digest_table) != set(bits):
            raise _format_error(
                "digest_table block indices must exactly match set bitmap bits",
                snapshot_id=snapshot_id,
                field_name="digest_table",
                missing_indices=sorted(set(bits) - set(digest_table)),
                extra_indices=sorted(set(digest_table) - set(bits)),
            )
        if set(block_refs) != set(bits):
            raise _format_error(
                "block_refs block indices must exactly match set bitmap bits",
                snapshot_id=snapshot_id,
                field_name="block_refs",
                missing_indices=sorted(set(bits) - set(block_refs)),
                extra_indices=sorted(set(block_refs) - set(bits)),
            )
        if set(digest_table) != set(block_refs):
            raise _format_error(
                "block_refs indices must exactly match digest_table indices",
                snapshot_id=snapshot_id,
                field_name="block_refs",
                missing_indices=sorted(set(digest_table) - set(block_refs)),
                extra_indices=sorted(set(block_refs) - set(digest_table)),
            )
        if len(digest_table) != len(bits):
            raise _format_error(
                "digest_table item count must equal the number of set bitmap bits",
                snapshot_id=snapshot_id,
                field_name="digest_table",
                expected=len(bits),
                actual=len(digest_table),
                missing_indices=sorted(set(bits) - set(digest_table)),
                extra_indices=sorted(set(digest_table) - set(bits)),
            )
        if set(digest_table) != set(bits):
            raise _format_error(
                "digest_table and block_refs indices must match set bitmap bits",
                snapshot_id=snapshot_id,
                field_name="digest_table",
                missing_indices=sorted(set(bits) - set(digest_table)),
                extra_indices=sorted(set(digest_table) - set(bits)),
            )
        checksum_value = normalize_digest(
            raw.get("bitmap_checksum"),
            snapshot_id=snapshot_id,
            field_name="bitmap_checksum",
        )
        expected_checksum = bitmap_checksum(bitmap)
        if checksum_value != expected_checksum:
            raise _format_error(
                "bitmap_checksum does not match bitmap content",
                snapshot_id=snapshot_id,
                field_name="bitmap_checksum",
                expected=f"sha256:{expected_checksum}",
                actual=f"sha256:{checksum_value}",
            )
        checksum = checksum_value
        parent_value = raw.get("parent_snapshot_id")
        if parent_value is not None and not isinstance(parent_value, str):
            raise _format_error(
                "parent_snapshot_id must be a string or null",
                snapshot_id=snapshot_id,
                field_name="parent_snapshot_id",
            )
        parent = parent_value or None
        snapshot_version = _require_int(
            raw.get("snapshot_version"),
            snapshot_id=snapshot_id,
            field_name="snapshot_version",
        )
        content_digest_value = normalize_digest(
            raw.get("content_digest"),
            snapshot_id=snapshot_id,
            field_name="content_digest",
        )
        expected_content_digest = compute_content_digest(
            snapshot_id=snapshot_id,
            block_count=block_count,
            bitmap=bitmap,
            digest_table=digest_table,
            block_refs=block_refs,
            parent_snapshot_id=parent,
            snapshot_version=snapshot_version,
        )
        if content_digest_value != expected_content_digest:
            raise _format_error(
                "content_digest does not match normalized manifest content",
                snapshot_id=snapshot_id,
                field_name="content_digest",
                expected=f"sha256:{expected_content_digest}",
                actual=f"sha256:{content_digest_value}",
            )
        content_digest = content_digest_value

    if set(digest_table) != set(bits):
        raise _format_error(
            "digest_table block indices must exactly match set bitmap bits",
            snapshot_id=snapshot_id,
            field_name="digest_table",
            missing_indices=sorted(set(bits) - set(digest_table)),
            extra_indices=sorted(set(digest_table) - set(bits)),
        )
    if set(block_refs) != set(bits):
        raise _format_error(
            "block_refs block indices must exactly match set bitmap bits",
            snapshot_id=snapshot_id,
            field_name="block_refs",
            missing_indices=sorted(set(bits) - set(block_refs)),
            extra_indices=sorted(set(block_refs) - set(bits)),
        )
    if len(digest_table) != len(bits) or len(block_refs) != len(bits):
        raise _format_error(
            "digest_table and block_refs counts must equal the number of set bitmap bits",
            snapshot_id=snapshot_id,
            field_name="digest_table",
            expected=len(bits),
            digest_table_count=len(digest_table),
            block_refs_count=len(block_refs),
        )

    if version in ("m1", "m2"):
        content_digest = compute_content_digest(
            snapshot_id=snapshot_id,
            block_count=block_count,
            bitmap=bitmap,
            digest_table=digest_table,
            block_refs=block_refs,
            parent_snapshot_id=parent,
            snapshot_version=snapshot_version,
        )

    return SnapshotManifest(
        snapshot_id=snapshot_id,
        block_count=block_count,
        bitmap=bitmap,
        bitmap_length=length_value,
        digest_table=dict(sorted(digest_table.items())),
        block_refs=dict(sorted(block_refs.items())),
        parent_snapshot_id=parent,
        bitmap_checksum=checksum,
        content_digest=content_digest,
        snapshot_version=snapshot_version,
        source_format=version,
        filled_fields=tuple(sorted(set(filled))),
        fill_basis=dict(sorted(basis.items())),
    )


def validate_chain(
    snapshots: list[SnapshotManifest] | tuple[SnapshotManifest, ...],
    *,
    error_code: str = E_FORMAT,
) -> None:
    by_id = {snapshot.snapshot_id: snapshot for snapshot in snapshots}
    if len(by_id) != len(snapshots):
        seen: set[str] = set()
        duplicate = ""
        for snapshot in snapshots:
            if snapshot.snapshot_id in seen:
                duplicate = snapshot.snapshot_id
                break
            seen.add(snapshot.snapshot_id)
        raise DiagnosticError(
            error_code,
            "duplicate snapshot identifier",
            snapshot_id=duplicate,
            field="snapshot_id",
        )
    for snapshot in snapshots:
        if snapshot.parent_snapshot_id is not None and snapshot.parent_snapshot_id not in by_id:
            raise DiagnosticError(
                error_code,
                "snapshot chain is broken: parent reference is missing",
                snapshot_id=snapshot.snapshot_id,
                field="parent_snapshot_id",
                details={
                    "missing_parent": snapshot.parent_snapshot_id,
                    "snapshot_id": snapshot.snapshot_id,
                },
            )
    state: dict[str, int] = {}

    def visit(snapshot_id: str, path: tuple[str, ...]) -> None:
        marker = state.get(snapshot_id, 0)
        if marker == 2:
            return
        if marker == 1:
            cycle_start = path.index(snapshot_id) if snapshot_id in path else 0
            cycle = list(path[cycle_start:]) + [snapshot_id]
            raise DiagnosticError(
                error_code,
                "snapshot chain contains a cycle",
                snapshot_id=snapshot_id,
                field="parent_snapshot_id",
                details={"cycle": cycle},
            )
        state[snapshot_id] = 1
        parent = by_id[snapshot_id].parent_snapshot_id
        if parent is not None:
            visit(parent, path + (snapshot_id,))
        state[snapshot_id] = 2

    for snapshot_id in sorted(by_id):
        visit(snapshot_id, ())


def decode_manifest(
    data: bytes | str | Mapping[str, Any],
    *,
    resolver: ManifestResolver | None = None,
) -> NormalizedManifest:
    snapshot_hint: str | None = None
    if isinstance(data, (bytes, str)):
        raw_text = data.decode("utf-8") if isinstance(data, bytes) else data
        match = re.search(r'"snapshot_id"\s*:\s*"([^"]+)"', raw_text)
        if match:
            snapshot_hint = match.group(1)
        try:
            document = json.loads(raw_text)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise DiagnosticError(
                E_FORMAT,
                "snapshot manifest is truncated or invalid JSON",
                snapshot_id=snapshot_hint,
                field="json",
                details={"reason": str(exc)},
            )
    else:
        document = data
    if not isinstance(document, Mapping):
        raise _format_error(
            "manifest must be a JSON object",
            snapshot_id=snapshot_hint,
            field_name="manifest",
        )
    if snapshot_hint is None:
        snapshot_hint = _first_snapshot_id(document)
    version = document.get("format_version")
    if version not in FORMAT_VERSIONS:
        first_match = re.search(r'"snapshot_id"\s*:\s*"([^"]+)"', json.dumps(document))
        raise _format_error(
            "unknown snapshot manifest version",
            snapshot_id=first_match.group(1) if first_match else snapshot_hint,
            field_name="format_version",
            actual=version,
            supported=list(FORMAT_VERSIONS),
        )
    allowed_top = _TOP_FIELDS[version]
    unknown = sorted(set(document) - allowed_top)
    if unknown:
        raise _format_error(
            "manifest contains fields not supported by this version",
            snapshot_id=snapshot_hint,
            field_name=unknown[0],
            unknown_fields=unknown,
            format_version=version,
        )
    missing = sorted(allowed_top - set(document))
    if missing:
        raise _format_error(
            "manifest is missing required fields",
            snapshot_id=snapshot_hint,
            field_name=missing[0],
            missing_fields=missing,
            format_version=version,
        )
    block_count = _require_int(
        document.get("volume_block_count"),
        snapshot_id=snapshot_hint,
        field_name="volume_block_count",
    )
    if block_count <= 0:
        raise _format_error(
            "volume_block_count must be positive",
            snapshot_id=snapshot_hint,
            field_name="volume_block_count",
            actual=block_count,
        )
    block_size_value = document.get("block_size")
    block_size = None
    if block_size_value is not None:
        block_size = _require_int(
            block_size_value,
            snapshot_id=snapshot_hint,
            field_name="block_size",
        )
        from .geometry import SUPPORTED_BLOCK_SIZES

        if block_size not in SUPPORTED_BLOCK_SIZES:
            raise _format_error(
                "block_size is not supported",
                snapshot_id=snapshot_hint,
                field_name="block_size",
                actual=block_size,
                supported=sorted(SUPPORTED_BLOCK_SIZES),
            )
    raw_snapshots = document.get("snapshots")
    if not isinstance(raw_snapshots, list):
        raise _format_error(
            "snapshots must be a list",
            snapshot_id=snapshot_hint,
            field_name="snapshots",
        )
    snapshots = [
        _decode_entry(item, version=version, block_count=block_count, resolver=resolver)
        for item in raw_snapshots
    ]
    validate_chain(snapshots)
    aggregate_filled: list[str] = ["manifest_digest"] if version != "m3" else []
    aggregate_basis: dict[str, str] = {}
    for snapshot in snapshots:
        aggregate_filled.extend(f"snapshots.{snapshot.snapshot_id}.{field}" for field in snapshot.filled_fields)
        for field_name, source in snapshot.fill_basis.items():
            aggregate_basis[f"snapshots.{snapshot.snapshot_id}.{field_name}"] = source
    aggregate_basis["manifest_digest"] = "computed_from_normalized_m3_manifest"

    if version == "m3":
        manifest_digest_value = normalize_digest(
            document.get("manifest_digest"),
            snapshot_id=snapshot_hint,
            field_name="manifest_digest",
        )
    else:
        manifest_digest_value = ""

    provisional = NormalizedManifest(
        source_format=version,
        block_size=block_size,
        block_count=block_count,
        snapshots=tuple(snapshots),
        manifest_digest=manifest_digest_value,
        filled_fields=tuple(sorted(set(aggregate_filled))),
        fill_basis=dict(sorted(aggregate_basis.items())),
    )
    computed_manifest_digest = compute_manifest_digest(provisional)
    if version == "m3" and manifest_digest_value != computed_manifest_digest:
        raise _format_error(
            "manifest_digest does not match the normalized manifest content",
            snapshot_id=snapshot_hint,
            field_name="manifest_digest",
            expected=f"sha256:{computed_manifest_digest}",
            actual=f"sha256:{manifest_digest_value}",
        )
    return NormalizedManifest(
        source_format=version,
        block_size=block_size,
        block_count=block_count,
        snapshots=provisional.snapshots,
        manifest_digest=computed_manifest_digest,
        filled_fields=provisional.filled_fields,
        fill_basis=provisional.fill_basis,
    )


def encode_manifest(model: NormalizedManifest, version: str = "m3") -> dict[str, Any]:
    if version not in FORMAT_VERSIONS:
        raise _format_error(
            "unknown snapshot manifest version",
            snapshot_id=None,
            field_name="format_version",
            actual=version,
            supported=list(FORMAT_VERSIONS),
        )
    ordered = sorted(model.snapshots, key=lambda item: item.snapshot_id)
    document: dict[str, Any] = {
        "format_version": version,
        "volume_block_count": model.block_count,
        "block_size": model.block_size,
        "snapshots": [_encoded_entry(snapshot, version) for snapshot in ordered],
    }
    if version == "m3":
        for entry in document["snapshots"]:
            entry["content_digest"] = _entry_content_digest(entry)
        document["manifest_digest"] = compute_manifest_digest(model)
    return document


def compute_manifest_digest(model: NormalizedManifest) -> str:
    document: dict[str, Any] = {
        "format_version": "m3",
        "volume_block_count": model.block_count,
        "block_size": model.block_size,
        "snapshots": [],
    }
    for snapshot in sorted(model.snapshots, key=lambda item: item.snapshot_id):
        entry = _encoded_entry(snapshot, "m3")
        entry["content_digest"] = _entry_content_digest(entry)
        document["snapshots"].append(entry)
    return hashlib.sha256(canonical_json_bytes(document)).hexdigest()


def encode_manifest_text(model: NormalizedManifest, version: str = "m3") -> str:
    return json.dumps(encode_manifest(model, version), ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def load_manifest_file(
    path: str | Path,
    *,
    resolver: ManifestResolver | None = None,
) -> NormalizedManifest:
    manifest_path = Path(path)
    try:
        data = manifest_path.read_bytes()
    except FileNotFoundError:
        raise DiagnosticError(
            E_FORMAT,
            "snapshot manifest does not exist",
            snapshot_id="<manifest>",
            field="file",
            details={"path": str(manifest_path)},
        )
    return decode_manifest(data, resolver=resolver)


def build_manifest_model(
    *,
    block_size: int | None,
    block_count: int,
    snapshots: list[SnapshotManifest],
    source_format: str = "m3",
) -> NormalizedManifest:
    ordered = tuple(sorted(snapshots, key=lambda item: item.snapshot_id))
    provisional = NormalizedManifest(
        source_format=source_format,
        block_size=block_size,
        block_count=block_count,
        snapshots=ordered,
        manifest_digest="",
    )
    digest = compute_manifest_digest(provisional)
    return NormalizedManifest(
        source_format=source_format,
        block_size=block_size,
        block_count=block_count,
        snapshots=ordered,
        manifest_digest=digest,
    )


def make_snapshot_manifest(
    *,
    snapshot_id: str,
    block_count: int,
    bitmap: bytes,
    digest_table: Mapping[int, str],
    block_refs: Mapping[int, str],
    parent_snapshot_id: str | None,
    snapshot_version: int,
    source_format: str = "m3",
    filled_fields: tuple[str, ...] = (),
    fill_basis: Mapping[str, str] | None = None,
) -> SnapshotManifest:
    checksum = bitmap_checksum(bitmap)
    content_digest = compute_content_digest(
        snapshot_id=snapshot_id,
        block_count=block_count,
        bitmap=bitmap,
        digest_table=digest_table,
        block_refs=block_refs,
        parent_snapshot_id=parent_snapshot_id,
        snapshot_version=snapshot_version,
    )
    return SnapshotManifest(
        snapshot_id=snapshot_id,
        block_count=block_count,
        bitmap=bytes(bitmap),
        bitmap_length=bitmap_length(block_count),
        digest_table=dict(sorted(digest_table.items())),
        block_refs=dict(sorted(block_refs.items())),
        parent_snapshot_id=parent_snapshot_id,
        bitmap_checksum=checksum,
        content_digest=content_digest,
        snapshot_version=snapshot_version,
        source_format=source_format,
        filled_fields=filled_fields,
        fill_basis=dict(fill_basis or {}),
    )
