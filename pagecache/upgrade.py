from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import os
from pathlib import Path
import struct
from typing import Any
import zlib

from .canonical import atomic_write_bytes, atomic_write_json, sha256_hex
from .errors import EngineError, ErrorCode
from .manifest import MANIFEST_MAGIC, MANIFEST_TRAILER, _HEADER, _TRAILER, decode_manifest, encode_record, read_manifest
from .model import CURRENT_FORMAT_VERSION, PageEntry
from .pagefile import page_checksum, page_path, read_page_file, summaries_for_page
from .segment import read_segment_file, segment_path


class SimulatedUpgradeCrash(RuntimeError):
    def __init__(self, completed_records: int) -> None:
        super().__init__(f"simulated crash after {completed_records} completed records")
        self.completed_records = completed_records


@dataclass(slots=True)
class UpgradeResult:
    manifest_path: str
    backup_path: str
    checkpoint_path: str
    report_path: str
    source_versions: list[int]
    target_version: int
    resumed: bool
    already_current: bool
    entries: list[dict[str, Any]] = field(default_factory=list)
    processed_record_indices: list[int] = field(default_factory=list)

    @property
    def filled_fields(self) -> list[str]:
        return sorted({name for entry in self.entries for name in entry.get("filled_fields", [])})

    def to_dict(self) -> dict[str, Any]:
        filled = self.filled_fields
        return {
            "status": "already_current" if self.already_current else "upgraded",
            "manifest_path": self.manifest_path,
            "backup_path": self.backup_path,
            "checkpoint_path": self.checkpoint_path,
            "report_path": self.report_path,
            "source_versions": self.source_versions,
            "target_version": self.target_version,
            "resumed": self.resumed,
            "processed_record_indices": self.processed_record_indices,
            "entries": self.entries,
            "filled_fields": filled,
        }


def upgrade_manifest(
    manifest_path: str | Path,
    *,
    pages_dir: str | Path,
    overflow_dir: str | Path,
    crash_after_checkpoint: int | None = None,
) -> UpgradeResult:
    manifest = Path(manifest_path)
    backup_path = manifest.with_name(f"{manifest.name}.pre-upgrade.backup")
    work_path = manifest.with_name(f"{manifest.name}.upgrade.work")
    checkpoint_path = manifest.with_name(f"{manifest.name}.upgrade-checkpoint.json")
    report_path = manifest.with_name(f"{manifest.name}.upgrade-report.json")

    if checkpoint_path.exists():
        checkpoint = _read_json(checkpoint_path)
        if checkpoint.get("stage") == "complete":
            current = read_manifest(manifest)
            if current.header_version != CURRENT_FORMAT_VERSION or any(
                record.record_version != CURRENT_FORMAT_VERSION for record in current.records
            ):
                raise EngineError(
                    ErrorCode.E_FORMAT,
                    "checkpoint says complete but the manifest is not current",
                    path=str(checkpoint_path),
                )
            return UpgradeResult(
                manifest_path=str(manifest),
                backup_path=str(checkpoint["backup_path"]),
                checkpoint_path=str(checkpoint_path),
                report_path=str(checkpoint["report_path"]),
                source_versions=[int(value) for value in checkpoint["source_versions"]],
                target_version=CURRENT_FORMAT_VERSION,
                resumed=True,
                already_current=True,
                entries=list(checkpoint.get("entries", [])),
                processed_record_indices=[int(value) for value in checkpoint.get("processed_record_indices", [])],
            )
        if checkpoint.get("stage") == "upgrading":
            return _resume_upgrade(
                manifest=manifest,
                backup_path=backup_path,
                work_path=work_path,
                checkpoint_path=checkpoint_path,
                report_path=report_path,
                checkpoint=checkpoint,
                pages_dir=Path(pages_dir),
                overflow_dir=Path(overflow_dir),
                crash_after_checkpoint=crash_after_checkpoint,
            )
        raise EngineError(ErrorCode.E_FORMAT, "unknown checkpoint stage", path=str(checkpoint_path))

    source = read_manifest(manifest)
    needs_upgrade = source.header_version < CURRENT_FORMAT_VERSION or any(
        record.record_version < CURRENT_FORMAT_VERSION for record in source.records
    )
    if not needs_upgrade:
        return UpgradeResult(
            manifest_path=str(manifest),
            backup_path="",
            checkpoint_path="",
            report_path="",
            source_versions=list(source.format_versions),
            target_version=CURRENT_FORMAT_VERSION,
            resumed=False,
            already_current=True,
        )

    source_bytes = manifest.read_bytes()
    source_sha = hashlib.sha256(source_bytes).hexdigest()
    _write_backup(backup_path, source_bytes, source_sha)
    header = _HEADER.pack(MANIFEST_MAGIC, CURRENT_FORMAT_VERSION, 0, len(source.records))
    atomic_write_bytes(work_path, header)
    checkpoint: dict[str, Any] = {
        "stage": "upgrading",
        "source_path": str(manifest),
        "source_sha256": source_sha,
        "backup_path": str(backup_path),
        "work_path": str(work_path),
        "report_path": str(report_path),
        "source_versions": list(source.format_versions),
        "target_version": CURRENT_FORMAT_VERSION,
        "total_records": len(source.records),
        "done_records": 0,
        "work_size": len(header),
        "work_sha256": sha256_hex(header),
        "work_crc32": zlib.crc32(header) & 0xFFFFFFFF,
        "entries": [],
        "processed_record_indices": [],
    }
    atomic_write_json(checkpoint_path, checkpoint)
    if crash_after_checkpoint == 0:
        raise SimulatedUpgradeCrash(0)
    return _continue_upgrade(
        manifest=manifest,
        source=source,
        work_path=work_path,
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        backup_path=backup_path,
        checkpoint=checkpoint,
        pages_dir=Path(pages_dir),
        overflow_dir=Path(overflow_dir),
        resumed=False,
        crash_after_checkpoint=crash_after_checkpoint,
    )


def _resume_upgrade(
    *,
    manifest: Path,
    backup_path: Path,
    work_path: Path,
    checkpoint_path: Path,
    report_path: Path,
    checkpoint: dict[str, Any],
    pages_dir: Path,
    overflow_dir: Path,
    crash_after_checkpoint: int | None,
) -> UpgradeResult:
    if not backup_path.exists():
        raise EngineError(ErrorCode.E_FORMAT, "upgrade backup is missing", path=str(backup_path))
    source_bytes = backup_path.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != checkpoint["source_sha256"]:
        raise EngineError(ErrorCode.E_FORMAT, "upgrade backup checksum differs from checkpoint", path=str(backup_path))
    source = decode_manifest(source_bytes, source_path=str(backup_path))
    if len(source.records) != int(checkpoint["total_records"]):
        raise EngineError(ErrorCode.E_FORMAT, "checkpoint record count differs from backup", path=str(checkpoint_path))
    if not work_path.exists():
        raise EngineError(ErrorCode.E_FORMAT, "upgrade work file is missing", path=str(work_path))
    done_records = int(checkpoint["done_records"])
    if not 0 <= done_records <= len(source.records):
        raise EngineError(ErrorCode.E_FORMAT, "checkpoint done_records is out of range", path=str(checkpoint_path))
    if len(checkpoint.get("entries", [])) != done_records:
        raise EngineError(ErrorCode.E_FORMAT, "checkpoint entry report length is inconsistent", path=str(checkpoint_path))

    expected_size = int(checkpoint["work_size"])
    actual_size = work_path.stat().st_size
    if actual_size < expected_size:
        raise EngineError(
            ErrorCode.E_FORMAT,
            "upgrade work file is shorter than its checkpoint",
            path=str(work_path),
            details={"checkpoint_size": expected_size, "actual_size": actual_size},
        )
    if actual_size > expected_size:
        with work_path.open("r+b") as handle:
            handle.truncate(expected_size)
            handle.flush()
            os.fsync(handle.fileno())
    prefix = work_path.read_bytes()
    if hashlib.sha256(prefix).hexdigest() != checkpoint["work_sha256"]:
        raise EngineError(ErrorCode.E_FORMAT, "upgrade work prefix checksum mismatch", path=str(checkpoint_path))
    if zlib.crc32(prefix) & 0xFFFFFFFF != int(checkpoint["work_crc32"]):
        raise EngineError(ErrorCode.E_FORMAT, "upgrade work CRC differs from checkpoint", path=str(checkpoint_path))
    return _continue_upgrade(
        manifest=manifest,
        source=source,
        work_path=work_path,
        checkpoint_path=checkpoint_path,
        report_path=report_path,
        backup_path=backup_path,
        checkpoint=checkpoint,
        pages_dir=pages_dir,
        overflow_dir=overflow_dir,
        resumed=True,
        crash_after_checkpoint=crash_after_checkpoint,
    )


def _continue_upgrade(
    *,
    manifest: Path,
    source: Any,
    work_path: Path,
    checkpoint_path: Path,
    report_path: Path,
    backup_path: Path,
    checkpoint: dict[str, Any],
    pages_dir: Path,
    overflow_dir: Path,
    resumed: bool,
    crash_after_checkpoint: int | None,
) -> UpgradeResult:
    done_records = int(checkpoint["done_records"])
    work_crc = int(checkpoint["work_crc32"])
    entries: list[dict[str, Any]] = list(checkpoint.get("entries", []))
    processed: list[int] = [int(value) for value in checkpoint.get("processed_record_indices", [])]

    with work_path.open("ab") as handle:
        for record_index in range(done_records, len(source.records)):
            source_entry = source.records[record_index]
            upgraded, report = upgrade_record(
                source_entry,
                record_index=record_index,
                pages_dir=pages_dir,
                overflow_dir=overflow_dir,
            )
            record_bytes = encode_record(upgraded, format_version=CURRENT_FORMAT_VERSION)
            handle.write(record_bytes)
            handle.flush()
            os.fsync(handle.fileno())
            work_crc = zlib.crc32(record_bytes, work_crc) & 0xFFFFFFFF
            done_records = record_index + 1
            if report is not None:
                entries.append(report)
                processed.append(record_index)
            work_size = work_path.stat().st_size
            work_sha = _sha256_prefix(work_path, work_size)
            checkpoint.update(
                {
                    "done_records": done_records,
                    "work_size": work_size,
                    "work_sha256": work_sha,
                    "work_crc32": work_crc,
                    "entries": entries,
                    "processed_record_indices": processed,
                }
            )
            atomic_write_json(checkpoint_path, checkpoint)
            if crash_after_checkpoint is not None and done_records == crash_after_checkpoint:
                raise SimulatedUpgradeCrash(done_records)

    if done_records != len(source.records):
        raise EngineError(ErrorCode.E_FORMAT, "upgrade did not process every record", path=str(checkpoint_path))
    trailer = _TRAILER.pack(MANIFEST_TRAILER, work_crc)
    with work_path.open("ab") as handle:
        handle.write(trailer)
        handle.flush()
        os.fsync(handle.fileno())
    final_bytes = work_path.read_bytes()
    decode_manifest(final_bytes, source_path=str(work_path))
    os.replace(work_path, manifest)
    _fsync_dir(manifest.parent)
    report_payload = {
        "status": "complete",
        "source_versions": list(source.format_versions),
        "target_version": CURRENT_FORMAT_VERSION,
        "entries": entries,
        "processed_record_indices": processed,
    }
    atomic_write_json(report_path, report_payload)
    checkpoint.update(
        {
            "stage": "complete",
            "done_records": done_records,
            "manifest_sha256": hashlib.sha256(final_bytes).hexdigest(),
            "entries": entries,
            "processed_record_indices": processed,
        }
    )
    atomic_write_json(checkpoint_path, checkpoint)
    return UpgradeResult(
        manifest_path=str(manifest),
        backup_path=str(backup_path),
        checkpoint_path=str(checkpoint_path),
        report_path=str(report_path),
        source_versions=list(source.format_versions),
        target_version=CURRENT_FORMAT_VERSION,
        resumed=resumed,
        already_current=False,
        entries=entries,
        processed_record_indices=processed,
    )


def upgrade_record(
    entry: PageEntry,
    *,
    record_index: int,
    pages_dir: str | Path,
    overflow_dir: str | Path,
) -> tuple[PageEntry, dict[str, Any] | None]:
    if entry.record_version >= CURRENT_FORMAT_VERSION:
        return _copy_current_entry(entry), None
    pages = Path(pages_dir)
    overflow = Path(overflow_dir)
    filled_fields: list[str] = []
    values: dict[str, Any] = {}
    column_summaries = entry.column_summaries
    payload_checksum = entry.payload_checksum

    payload = read_page_file(page_path(pages, entry.page_id), expected_page_id=entry.page_id)
    if payload.row_count != entry.row_count:
        raise EngineError(
            ErrorCode.E_INTEGRITY,
            "page row count does not match the manifest",
            path=f"$manifest.records[{record_index}].row_count",
            record_index=record_index,
            page_id=entry.page_id,
            details={"manifest_rows": entry.row_count, "page_rows": payload.row_count},
        )
    actual_checksum = page_checksum(payload.raw_bytes)
    actual_summaries = summaries_for_page(payload)
    if entry.record_version == 1:
        column_summaries = actual_summaries
        payload_checksum = actual_checksum
        filled_fields.extend(["column_summaries", "payload_checksum"])
        values["column_summaries"] = [summary.to_dict() for summary in column_summaries]
        values["payload_checksum"] = payload_checksum
    else:
        if entry.payload_checksum != actual_checksum:
            raise EngineError(
                ErrorCode.E_INTEGRITY,
                "v2 payload checksum does not match the page file",
                path=f"$manifest.records[{record_index}].payload_checksum",
                record_index=record_index,
                page_id=entry.page_id,
                details={"expected": entry.payload_checksum, "actual": actual_checksum},
            )
        if entry.column_summaries != actual_summaries:
            raise EngineError(
                ErrorCode.E_INTEGRITY,
                "v2 column summaries do not match the page file",
                path=f"$manifest.records[{record_index}].column_summaries",
                record_index=record_index,
                page_id=entry.page_id,
            )

    segment_ref = f"{entry.page_id}.seg"
    candidate = segment_path(overflow, segment_ref, entry.page_id)
    if candidate.exists():
        segment = read_segment_file(candidate, expected_page_id=entry.page_id)
        if segment.row_count != entry.row_count or segment.payload != payload.raw_bytes:
            raise EngineError(
                ErrorCode.E_INTEGRITY,
                "overflow segment does not match the page file during upgrade",
                path=str(candidate),
                record_index=record_index,
                page_id=entry.page_id,
            )
        compressed = segment.compressed
        overflow_ref: str | None = segment_ref
        rule = "existing_segment_header"
    else:
        compressed = False
        overflow_ref = None
        rule = "no_existing_segment_default_uncompressed_null_ref"
    filled_fields.extend(["compressed", "overflow_ref"])
    values["compressed"] = compressed
    values["overflow_ref"] = overflow_ref

    upgraded = PageEntry(
        page_id=entry.page_id,
        row_count=entry.row_count,
        record_version=CURRENT_FORMAT_VERSION,
        column_summaries=column_summaries,
        payload_checksum=payload_checksum,
        compressed=compressed,
        overflow_ref=overflow_ref,
    )
    report = {
        "record_index": record_index,
        "page_id": entry.page_id,
        "from_version": entry.record_version,
        "to_version": CURRENT_FORMAT_VERSION,
        "filled_fields": filled_fields,
        "values": values,
        "overflow_rule": rule,
    }
    return upgraded, report


def _copy_current_entry(entry: PageEntry) -> PageEntry:
    return PageEntry(
        page_id=entry.page_id,
        row_count=entry.row_count,
        record_version=CURRENT_FORMAT_VERSION,
        column_summaries=entry.column_summaries,
        payload_checksum=entry.payload_checksum,
        compressed=entry.compressed,
        overflow_ref=entry.overflow_ref,
    )


def _write_backup(path: Path, data: bytes, expected_sha: str) -> None:
    if path.exists():
        existing = path.read_bytes()
        if hashlib.sha256(existing).hexdigest() != expected_sha:
            raise EngineError(ErrorCode.E_FORMAT, "existing upgrade backup differs from source", path=str(path))
        return
    atomic_write_bytes(path, data, mode=0o444)


def _sha256_prefix(path: Path, size: int) -> str:
    digest = hashlib.sha256()
    remaining = size
    with path.open("rb") as handle:
        while remaining:
            chunk = handle.read(min(1024 * 1024, remaining))
            if not chunk:
                raise EngineError(ErrorCode.E_FORMAT, "cannot hash complete upgrade prefix", path=str(path))
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    import json

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EngineError(ErrorCode.E_FORMAT, f"cannot read checkpoint: {exc}", path=str(path)) from exc
    if not isinstance(value, dict):
        raise EngineError(ErrorCode.E_FORMAT, "checkpoint root must be an object", path=str(path))
    return value


def _fsync_dir(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
