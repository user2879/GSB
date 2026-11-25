"""Three-version sampling-evidence exchange.

Formats
-------
p1
  78-column ASCII rows. Columns: magic/prefix, 32-byte bucket id, 6-digit
  slot, 24-digit per-mille weight, 1-digit admitted flag, CRC32.
p2
  Deterministic JSON rows carrying the p1 fields plus bucket arrival ordinal
  and random-source digest.
p3
  Binary rows carrying every p2 field plus log id, replaced-record reference,
  bucket-count snapshot, record fingerprint, arrival cursor and raw metric.

When downgrading, fields that the target cannot express are written to a
JSON sidecar keyed by row index. Upgrading reads those fields from the sidecar;
it never recomputes them.
"""

from __future__ import annotations

import json
import hashlib
import math
import struct
import zlib
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable

from .canonical import canonical_json, sha256_bytes

E_SAMPLE_FORMAT = "E_SAMPLE_FORMAT"
E_SAMPLE_METADATA = "E_SAMPLE_METADATA"
P1_WIDTH = 78
P3_MAGIC = b"P3SAMP01"
P3_VERSION = 1
P3_ROW = struct.Struct("<32sHQB I32s32sI32sI64sd".replace(" ", ""))
P3_HEADER = struct.Struct("<8sII")
P3_DIGEST_SIZE = 32


@dataclass
class EvidenceIssue:
    code: str
    message: str
    row: int | None = None
    field: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass
class EvidenceSlot:
    bucket_id: str
    slot_no: int
    weight: int
    admitted: int
    arrival_ordinal: int | None = None
    random_digest: str | None = None
    replaced_record_ref: str | None = None
    bucket_count_snapshot: int | None = None
    log_id: str | None = None
    arrival_seq: int | None = None
    record_fingerprint: str | None = None
    raw_metric: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceSlot":
        known = {field.name for field in fields(cls)}
        return cls(**{key: value.get(key) for key in known})


@dataclass
class EvidenceLoadResult:
    version: str
    slots: list[EvidenceSlot]
    issues: list[EvidenceIssue]
    sidecar_rows: dict[int, dict[str, Any]]


@dataclass
class EvidenceConversionResult:
    target_version: str
    payload: bytes
    sidecar: dict[str, Any]
    issues: list[EvidenceIssue]
    slots: list[EvidenceSlot]


def slot_from_history(event: dict[str, Any], bucket_id: str) -> EvidenceSlot:
    return EvidenceSlot(
        bucket_id=bucket_id,
        slot_no=int(event["slot_no"]),
        weight=int(event["quantized_weight"]),
        admitted=1 if event["admitted"] else 0,
        arrival_ordinal=int(event["arrival_ordinal"]),
        random_digest=str(event["random_digest"]),
        replaced_record_ref=event.get("replaced_log_id"),
        bucket_count_snapshot=int(event["arrival_ordinal"]),
        log_id=str(event["log_id"]),
        arrival_seq=int(event["arrival_seq"]),
        record_fingerprint=str(event["record_fingerprint"]),
        raw_metric=float(event["metric"]),
    )


def build_evidence_slots(buckets: dict[str, dict[str, Any]]) -> list[EvidenceSlot]:
    slots: list[EvidenceSlot] = []
    for bucket_id, bucket in sorted(buckets.items()):
        for event in bucket["history"]:
            slots.append(slot_from_history(event, bucket_id))
    return slots


def _truncate_ascii(value: str, limit: int, field: str, issues: list[EvidenceIssue]) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) > limit:
        issues.append(
            EvidenceIssue(
                code=E_SAMPLE_FORMAT,
                message=f"{field} exceeds fixed width {limit}",
                field=field,
            )
        )
        return encoded[:limit].decode("utf-8", errors="replace")
    return value


def _p1_line(slot: EvidenceSlot) -> bytes:
    if not slot.bucket_id.isascii() or len(slot.bucket_id) > 32:
        raise ValueError("p1 bucket_id must be <= 32 ASCII characters")
    if slot.weight < 0 or slot.weight >= 10 ** 24:
        raise ValueError("p1 weight must fit 24 decimal digits")
    if slot.admitted not in (0, 1):
        raise ValueError("p1 admitted must be 0 or 1")
    body = (
        f"P1|{slot.bucket_id:<32}|{slot.slot_no:06d}|"
        f"{slot.weight:024d}|{slot.admitted:01d}|"
    ).encode("ascii")
    crc = f"{zlib.crc32(body) & 0xFFFFFFFF:08x}".encode("ascii")
    return body + crc


def _p1_parse(data: bytes, sidecar_rows: dict[int, dict[str, Any]]) -> tuple[list[EvidenceSlot], list[EvidenceIssue]]:
    slots: list[EvidenceSlot] = []
    issues: list[EvidenceIssue] = []
    text = data.decode("ascii", errors="replace")
    lines = text.splitlines()
    for row_index, line in enumerate(lines):
        if len(line) != P1_WIDTH:
            issues.append(
                EvidenceIssue(
                    code=E_SAMPLE_FORMAT,
                    message=f"p1 row width {len(line)} != {P1_WIDTH}",
                    row=row_index,
                    field="width",
                )
            )
            continue
        if (
            line[:3] != "P1|"
            or line[35:36] != "|"
            or line[42:43] != "|"
            or line[67:68] != "|"
            or line[69:70] != "|"
        ):
            issues.append(
                EvidenceIssue(
                    code=E_SAMPLE_FORMAT,
                    message="p1 prefix/separator mismatch",
                    row=row_index,
                )
            )
            continue
        body = line[:70]
        expected_crc = f"{zlib.crc32(body.encode('ascii')) & 0xFFFFFFFF:08x}"
        if line[70:78] != expected_crc:
            issues.append(
                EvidenceIssue(
                    code=E_SAMPLE_FORMAT,
                    message="p1 CRC32 mismatch",
                    row=row_index,
                    field="crc32",
                )
            )
            continue
        bucket_id = line[3:35].rstrip()
        try:
            slot_no = int(line[36:42])
            weight = int(line[43:67])
            admitted = int(line[68:69])
        except ValueError:
            issues.append(
                EvidenceIssue(
                    code=E_SAMPLE_FORMAT,
                    message="p1 numeric field is malformed",
                    row=row_index,
                )
            )
            continue
        if not bucket_id or slot_no < 0 or weight < 0 or admitted not in (0, 1):
            issues.append(
                EvidenceIssue(
                    code=E_SAMPLE_FORMAT,
                    message="p1 field is outside its allowed range",
                    row=row_index,
                )
            )
            continue
        slot = EvidenceSlot(
            bucket_id=bucket_id,
            slot_no=slot_no,
            weight=weight,
            admitted=admitted,
        )
        _merge_sidecar(slot, sidecar_rows.get(row_index), row_index, issues)
        slots.append(slot)
    return slots, issues


def _merge_sidecar(
    slot: EvidenceSlot,
    row: dict[str, Any] | None,
    row_index: int,
    issues: list[EvidenceIssue],
) -> None:
    if not row:
        return
    sidecar_bucket = row.get("bucket_id")
    sidecar_slot = row.get("slot_no")
    if sidecar_bucket is not None and sidecar_bucket != slot.bucket_id:
        issues.append(
            EvidenceIssue(
                code=E_SAMPLE_METADATA,
                message="sidecar bucket_id does not match serialized row",
                row=row_index,
                field="bucket_id",
            )
        )
        return
    if sidecar_slot is not None and int(sidecar_slot) != slot.slot_no:
        issues.append(
            EvidenceIssue(
                code=E_SAMPLE_METADATA,
                message="sidecar slot_no does not match serialized row",
                row=row_index,
                field="slot_no",
            )
        )
        return
    for attribute in (
        "arrival_ordinal",
        "random_digest",
        "replaced_record_ref",
        "bucket_count_snapshot",
        "log_id",
        "arrival_seq",
        "record_fingerprint",
        "raw_metric",
    ):
        if getattr(slot, attribute) is None and attribute in row:
            setattr(slot, attribute, row[attribute])


def _validate_slot(slot: EvidenceSlot, row: int, issues: list[EvidenceIssue]) -> bool:
    before = len(issues)
    if not isinstance(slot.bucket_id, str) or not slot.bucket_id:
        issues.append(
            EvidenceIssue(E_SAMPLE_FORMAT, "bucket_id must be a non-empty string", row, "bucket_id")
        )
    if not isinstance(slot.slot_no, int) or slot.slot_no < 0:
        issues.append(
            EvidenceIssue(E_SAMPLE_FORMAT, "slot_no must be a non-negative integer", row, "slot_no")
        )
    if not isinstance(slot.weight, int) or slot.weight < 0:
        issues.append(
            EvidenceIssue(E_SAMPLE_FORMAT, "weight must be a non-negative integer", row, "weight")
        )
    if slot.admitted not in (0, 1, True, False):
        issues.append(
            EvidenceIssue(E_SAMPLE_FORMAT, "admitted must be 0 or 1", row, "admitted")
        )
    if slot.arrival_ordinal is not None and (
        not isinstance(slot.arrival_ordinal, int) or slot.arrival_ordinal < 1
    ):
        issues.append(
            EvidenceIssue(E_SAMPLE_FORMAT, "arrival_ordinal must be positive", row, "arrival_ordinal")
        )
    if slot.random_digest is not None and not _valid_digest(slot.random_digest):
        issues.append(
            EvidenceIssue(E_SAMPLE_FORMAT, "random_digest must be 64 hex characters", row, "random_digest")
        )
    return len(issues) == before


def _valid_digest(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    try:
        bytes.fromhex(value)
    except ValueError:
        return False
    return True


def _p2_payload(slots: Iterable[EvidenceSlot]) -> bytes:
    rows = []
    for slot in slots:
        if slot.arrival_ordinal is None or slot.random_digest is None:
            raise ValueError("p2 requires arrival_ordinal and random_digest")
        rows.append(
            {
                "bucket_id": slot.bucket_id,
                "slot_no": slot.slot_no,
                "weight": slot.weight,
                "admitted": int(slot.admitted),
                "arrival_ordinal": slot.arrival_ordinal,
                "random_digest": slot.random_digest,
            }
        )
    payload = {
        "format": "p2",
        "version": 1,
        "rows": rows,
    }
    return (canonical_json(payload) + "\n").encode("utf-8")


def _p2_parse(data: bytes, sidecar_rows: dict[int, dict[str, Any]]) -> tuple[list[EvidenceSlot], list[EvidenceIssue]]:
    slots: list[EvidenceSlot] = []
    issues: list[EvidenceIssue] = []
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return [], [EvidenceIssue(E_SAMPLE_FORMAT, f"p2 JSON is malformed: {exc}")]
    if not isinstance(payload, dict) or payload.get("format") != "p2" or not isinstance(payload.get("rows"), list):
        return [], [EvidenceIssue(E_SAMPLE_FORMAT, "p2 must be an object with format=p2 and rows list")]
    for row_index, row in enumerate(payload["rows"]):
        if not isinstance(row, dict):
            issues.append(EvidenceIssue(E_SAMPLE_FORMAT, "p2 row must be an object", row_index))
            continue
        required = ("bucket_id", "slot_no", "weight", "admitted", "arrival_ordinal", "random_digest")
        missing = [key for key in required if key not in row]
        if missing:
            issues.append(
                EvidenceIssue(
                    E_SAMPLE_FORMAT,
                    f"p2 row is missing {','.join(missing)}",
                    row_index,
                )
            )
            continue
        try:
            slot = EvidenceSlot(
                bucket_id=row["bucket_id"],
                slot_no=int(row["slot_no"]),
                weight=int(row["weight"]),
                admitted=int(row["admitted"]),
                arrival_ordinal=int(row["arrival_ordinal"]),
                random_digest=row["random_digest"],
            )
        except (TypeError, ValueError, OverflowError):
            issues.append(EvidenceIssue(E_SAMPLE_FORMAT, "p2 row has a non-integer numeric field", row_index))
            continue
        if not _validate_slot(slot, row_index, issues):
            continue
        _merge_sidecar(slot, sidecar_rows.get(row_index), row_index, issues)
        slots.append(slot)
    return slots, issues


def _fixed(limit: int) -> Any:
    return lambda value: struct.pack(f"<{limit}s", value)


def _pad_ascii(value: str | None, limit: int) -> bytes:
    if value is None:
        return b"\x00" * limit
    encoded = value.encode("utf-8")
    if len(encoded) > limit:
        raise ValueError(f"field is longer than fixed p3 width {limit}")
    return encoded + b"\x00" * (limit - len(encoded))


def _unpad_ascii(value: bytes) -> str | None:
    stripped = value.split(b"\x00", 1)[0]
    if not stripped:
        return None
    return stripped.decode("utf-8")


def _p3_payload(slots: Iterable[EvidenceSlot]) -> bytes:
    rows = bytearray()
    count = 0
    for slot in slots:
        if slot.arrival_ordinal is None:
            ordinal = 0
        else:
            ordinal = int(slot.arrival_ordinal)
        if ordinal >= 1 << 32 or slot.slot_no >= 1 << 16:
            raise ValueError("p3 ordinal/slot overflow")
        if slot.weight < 0 or slot.weight >= 1 << 64:
            raise ValueError("p3 weight overflow")
        if slot.random_digest is None:
            digest = b"\x00" * 32
        else:
            digest = bytes.fromhex(slot.random_digest)
        record_fp = ""
        if slot.record_fingerprint is not None:
            record_fp = slot.record_fingerprint
        rows.extend(
            P3_ROW.pack(
                _pad_ascii(slot.bucket_id, 32),
                int(slot.slot_no),
                int(slot.weight),
                1 if slot.admitted else 0,
                ordinal,
                digest,
                _pad_ascii(slot.replaced_record_ref, 32),
                int(slot.bucket_count_snapshot or 0),
                _pad_ascii(slot.log_id, 32),
                int(slot.arrival_seq or 0),
                _pad_ascii(record_fp, 64),
                float(slot.raw_metric) if slot.raw_metric is not None else math.nan,
            )
        )
        count += 1
    header = P3_HEADER.pack(P3_MAGIC, P3_VERSION, count)
    body = header + bytes(rows)
    return body + hashlib.sha256(body).digest()


def _p3_parse(data: bytes) -> tuple[list[EvidenceSlot], list[EvidenceIssue]]:
    if len(data) < P3_HEADER.size + P3_DIGEST_SIZE:
        return [], [EvidenceIssue(E_SAMPLE_FORMAT, "p3 payload is too short")]
    body, digest = data[:-P3_DIGEST_SIZE], data[-P3_DIGEST_SIZE:]
    if hashlib.sha256(body).digest() != digest:
        return [], [EvidenceIssue(E_SAMPLE_FORMAT, "p3 SHA-256 integrity check failed", field="sha256")]
    try:
        magic, version, count = P3_HEADER.unpack(body[: P3_HEADER.size])
    except struct.error as exc:
        return [], [EvidenceIssue(E_SAMPLE_FORMAT, f"p3 header is malformed: {exc}")]
    if magic != P3_MAGIC or version != P3_VERSION:
        return [], [EvidenceIssue(E_SAMPLE_FORMAT, "p3 magic/version mismatch")]
    expected = P3_HEADER.size + count * P3_ROW.size
    if len(body) != expected:
        return [], [
            EvidenceIssue(
                E_SAMPLE_FORMAT,
                f"p3 row area length {len(body) - P3_HEADER.size} != {count * P3_ROW.size}",
                field="row_count",
            )
        ]
    slots: list[EvidenceSlot] = []
    issues: list[EvidenceIssue] = []
    offset = P3_HEADER.size
    for row_index in range(count):
        raw = P3_ROW.unpack(body[offset : offset + P3_ROW.size])
        offset += P3_ROW.size
        bucket_id = _unpad_ascii(raw[0])
        random_digest = raw[5].hex() if raw[5] != b"\x00" * 32 else None
        slot = EvidenceSlot(
            bucket_id=bucket_id or "",
            slot_no=int(raw[1]),
            weight=int(raw[2]),
            admitted=int(raw[3]),
            arrival_ordinal=int(raw[4]) or None,
            random_digest=random_digest,
            replaced_record_ref=_unpad_ascii(raw[6]),
            bucket_count_snapshot=int(raw[7]) or None,
            log_id=_unpad_ascii(raw[8]),
            arrival_seq=int(raw[9]) or None,
            record_fingerprint=_unpad_ascii(raw[10]),
            raw_metric=None if math.isnan(raw[11]) else raw[11],
        )
        if _validate_slot(slot, row_index, issues):
            slots.append(slot)
    return slots, issues


def _sidecar_for(slots: list[EvidenceSlot], version: str) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    if version == "p3":
        return {"format": version, "schema_version": 1, "rows": rows}
    for row_index, slot in enumerate(slots):
        row: dict[str, Any] = {
            "row_index": row_index,
            "bucket_id": slot.bucket_id,
            "slot_no": slot.slot_no,
        }
        if version == "p1":
            row.update(
                {
                    "arrival_ordinal": slot.arrival_ordinal,
                    "random_digest": slot.random_digest,
                    "replaced_record_ref": slot.replaced_record_ref,
                    "bucket_count_snapshot": slot.bucket_count_snapshot,
                    "log_id": slot.log_id,
                    "arrival_seq": slot.arrival_seq,
                    "record_fingerprint": slot.record_fingerprint,
                    "raw_metric": slot.raw_metric,
                }
            )
        else:
            row.update(
                {
                    "replaced_record_ref": slot.replaced_record_ref,
                    "bucket_count_snapshot": slot.bucket_count_snapshot,
                    "log_id": slot.log_id,
                    "arrival_seq": slot.arrival_seq,
                    "record_fingerprint": slot.record_fingerprint,
                    "raw_metric": slot.raw_metric,
                }
            )
        rows.append(row)
    return {"format": version, "schema_version": 1, "rows": rows}


def serialize_evidence(
    slots: list[EvidenceSlot], version: str
) -> tuple[bytes, dict[str, Any]]:
    if version == "p1":
        payload = b"\n".join(_p1_line(slot) for slot in slots)
        if slots:
            payload += b"\n"
    elif version == "p2":
        payload = _p2_payload(slots)
    elif version == "p3":
        payload = _p3_payload(slots)
    else:
        raise ValueError(f"unknown evidence version {version}")
    return payload, _sidecar_for(slots, version)


def parse_evidence(
    data: bytes,
    version: str,
    sidecar: dict[str, Any] | None = None,
) -> EvidenceLoadResult:
    sidecar_rows: dict[int, dict[str, Any]] = {}
    issues: list[EvidenceIssue] = []
    if sidecar is not None:
        rows = sidecar.get("rows") if isinstance(sidecar, dict) else None
        if not isinstance(rows, list):
            issues.append(
                EvidenceIssue(E_SAMPLE_METADATA, "sidecar must contain a rows list")
            )
        else:
            for fallback_index, row in enumerate(rows):
                if not isinstance(row, dict):
                    issues.append(
                        EvidenceIssue(
                            E_SAMPLE_METADATA,
                            "sidecar row must be an object",
                            fallback_index,
                        )
                    )
                    continue
                row_index = int(row.get("row_index", fallback_index))
                sidecar_rows[row_index] = row
    if version == "p1":
        slots, format_issues = _p1_parse(data, sidecar_rows)
    elif version == "p2":
        slots, format_issues = _p2_parse(data, sidecar_rows)
    elif version == "p3":
        slots, format_issues = _p3_parse(data)
    else:
        return EvidenceLoadResult(
            version=version,
            slots=[],
            issues=[EvidenceIssue(E_SAMPLE_FORMAT, f"unknown version {version}")],
            sidecar_rows=sidecar_rows,
        )
    return EvidenceLoadResult(
        version=version,
        slots=slots,
        issues=issues + format_issues,
        sidecar_rows=sidecar_rows,
    )


def _missing_for_upgrade(
    slots: list[EvidenceSlot],
    source: str,
    target: str,
    sidecar_rows: dict[int, dict[str, Any]],
) -> list[EvidenceIssue]:
    issues: list[EvidenceIssue] = []
    if source == "p1" and target in {"p2", "p3"}:
        for row_index, slot in enumerate(slots):
            if slot.arrival_ordinal is None:
                issues.append(
                    EvidenceIssue(
                        E_SAMPLE_METADATA,
                        "arrival_ordinal is unavailable; supply p1 sidecar",
                        row_index,
                        "arrival_ordinal",
                    )
                )
            if slot.random_digest is None:
                issues.append(
                    EvidenceIssue(
                        E_SAMPLE_METADATA,
                        "random_digest is unavailable; supply p1 sidecar",
                        row_index,
                        "random_digest",
                    )
                )
    if target == "p3" and source in {"p1", "p2"}:
        for row_index, slot in enumerate(slots):
            sidecar_row = sidecar_rows.get(row_index, {})
            if (
                slot.replaced_record_ref is None
                and "replaced_record_ref" not in sidecar_row
            ):
                issues.append(
                    EvidenceIssue(
                        E_SAMPLE_METADATA,
                        "replaced_record_ref is unavailable; supply source sidecar",
                        row_index,
                        "replaced_record_ref",
                    )
                )
            if slot.bucket_count_snapshot is None:
                issues.append(
                    EvidenceIssue(
                        E_SAMPLE_METADATA,
                        "bucket_count_snapshot is unavailable; supply source sidecar",
                        row_index,
                        "bucket_count_snapshot",
                    )
                )
    return issues


def convert_evidence(
    data: bytes,
    source_version: str,
    target_version: str,
    sidecar: dict[str, Any] | None = None,
) -> EvidenceConversionResult:
    loaded = parse_evidence(data, source_version, sidecar)
    upgrade_issues = _missing_for_upgrade(
        loaded.slots,
        source_version,
        target_version,
        loaded.sidecar_rows,
    )
    issues = loaded.issues + upgrade_issues
    try:
        payload, target_sidecar = serialize_evidence(loaded.slots, target_version)
    except ValueError as exc:
        issues.append(
            EvidenceIssue(E_SAMPLE_FORMAT, f"target {target_version} serialization failed: {exc}")
        )
        payload = b""
        target_sidecar = _sidecar_for(loaded.slots, target_version)
    return EvidenceConversionResult(
        target_version=target_version,
        payload=payload,
        sidecar=target_sidecar,
        issues=issues,
        slots=loaded.slots,
    )


def load_evidence_file(
    path: str | Path,
    version: str,
    sidecar_path: str | Path | None = None,
) -> EvidenceLoadResult:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        return EvidenceLoadResult(
            version,
            [],
            [EvidenceIssue(E_SAMPLE_FORMAT, f"evidence file cannot be read: {exc}")],
            {},
        )
    sidecar = None
    if sidecar_path is not None:
        try:
            sidecar = json.loads(Path(sidecar_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return EvidenceLoadResult(
                version,
                [],
                [EvidenceIssue(E_SAMPLE_METADATA, f"sidecar cannot be read: {exc}")],
                {},
            )
    return parse_evidence(data, version, sidecar)
