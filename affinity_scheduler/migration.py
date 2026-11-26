from __future__ import annotations

import os
import shutil
import stat
import time
from pathlib import Path
from typing import Any

from .db import connect, ensure_schema
from .errors import MigrateError
from .model import Spec
from .records import (
    NormalizedRecord,
    encode_v3,
    load_records,
)
from .util import atomic_write_text, digest_file, digest_value, monotonic_ms


COMPARE_FIELDS = (
    "node_name",
    "plugin_scores",
    "plugin_contributions",
    "veto_code",
    "total_score",
    "error_bound",
    "tie_break",
)


def _migration_batch_id(source_sha: str, spec: Spec) -> str:
    return "mig-" + digest_value(
        {
            "source_sha": source_sha,
            "parameter_version": spec.parameter_version,
            "scale_exponent": spec.scale_exponent,
            "weights": dict(spec.weights_myriad),
        }
    )[:24]


def _ensure_backup(source: Path, backup: Path) -> str:
    source_sha = digest_file(source)
    if backup.exists():
        if digest_file(backup) != source_sha:
            raise MigrateError(
                "read-only migration backup already exists with different content",
                details={"backup": str(backup)},
            )
        os.chmod(backup, 0o444)
        return source_sha
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, backup)
    os.chmod(backup, 0o444)
    if digest_file(backup) != source_sha:
        raise MigrateError(
            "read-only migration backup verification failed",
            details={"backup": str(backup)},
        )
    return source_sha


def _first_difference(
    source: NormalizedRecord, target: NormalizedRecord
) -> str | None:
    for field_name in COMPARE_FIELDS:
        if getattr(source, field_name) != getattr(target, field_name):
            return field_name
    return None


def migrate_records(
    records_path: str | Path,
    spec: Spec,
    db_path: str | Path,
    *,
    batch_size: int = 2,
    backup_path: str | Path | None = None,
    abort_after_batches: int | None = None,
) -> dict[str, Any]:
    source = Path(records_path)
    if batch_size < 1:
        raise MigrateError("batch_size must be positive")
    normalized = load_records(
        source, params=None
    )
    # Normalize once more with the declared parameter set so every legacy row is
    # upgraded under the same target parameters.
    from .records import NormalizationParams

    params = NormalizationParams.from_spec(spec)
    normalized = load_records(source, params=params)
    if normalized.format_versions == (3,):
        return {
            "status": "already-current",
            "source": str(source),
            "record_count": len(normalized.records),
            "checkpoint": len(normalized.records),
            "batch_id": None,
        }

    backup = Path(backup_path) if backup_path is not None else Path(str(source) + ".readonly.bak")
    original_mode = stat.S_IMODE(source.stat().st_mode)
    source_sha = _ensure_backup(source, backup)
    batch_id = _migration_batch_id(source_sha, spec)

    connection = connect(db_path)
    ensure_schema(connection)
    try:
        row = connection.execute(
            "SELECT * FROM migration_batches WHERE batch_id = ?", (batch_id,)
        ).fetchone()
        if row is None:
            now = monotonic_ms()
            connection.execute(
                """
                INSERT INTO migration_batches
                (batch_id, source_path, source_sha, target_path, total_records,
                 checkpoint, status, started_mono_ms, updated_mono_ms)
                VALUES (?, ?, ?, ?, ?, 0, 'running', ?, ?)
                """,
                (
                    batch_id,
                    str(source),
                    source_sha,
                    str(source),
                    len(normalized.records),
                    now,
                    now,
                ),
            )
            checkpoint = 0
        else:
            if row["source_sha"] != source_sha:
                raise MigrateError(
                    "migration checkpoint belongs to different source bytes",
                    details={"batch_id": batch_id},
                )
            checkpoint = int(row["checkpoint"])
            if row["status"] == "complete":
                return {
                    "status": "complete",
                    "source": str(source),
                    "record_count": int(row["total_records"]),
                    "checkpoint": checkpoint,
                    "batch_id": batch_id,
                }

        # Ordinal order is source format version, then original line number.
        ordered = sorted(
            normalized.records,
            key=lambda item: (item.source_format_version, item.source_line),
        )
        batches_since_abort = 0
        target_rows: dict[int, str] = {}
        for ordinal, record in enumerate(ordered, start=1):
            if ordinal <= checkpoint:
                existing = connection.execute(
                    "SELECT target_json FROM migration_records WHERE batch_id=? AND ordinal=?",
                    (batch_id, ordinal),
                ).fetchone()
                if existing is None:
                    raise MigrateError(
                        "checkpoint is ahead of committed migration records",
                        pod_id=record.pod_id,
                        details={"batch_id": batch_id, "checkpoint": checkpoint},
                    )
                target_rows[ordinal] = str(existing["target_json"])
                continue
            target_json = encode_v3(record)
            try:
                target_record = normalize_current_json(
                    target_json, params=params, source_line=record.source_line
                )
            except Exception as exc:
                raise MigrateError(
                    "migrated v3 record cannot be normalized",
                    pod_id=record.pod_id,
                    details={"error": str(exc)},
                ) from exc
            mismatch = _first_difference(record, target_record)
            if mismatch is not None:
                raise MigrateError(
                    "migration changed the binding conclusion",
                    pod_id=record.pod_id,
                    details={
                        "field": mismatch,
                        "source_value": getattr(record, mismatch),
                        "target_value": getattr(target_record, mismatch),
                    },
                )
            target_rows[ordinal] = target_json

            if ordinal % batch_size == 0 or ordinal == len(ordered):
                committed_end = ordinal
                connection.execute("BEGIN IMMEDIATE")
                try:
                    for item_ordinal in range(checkpoint + 1, committed_end + 1):
                        item = ordered[item_ordinal - 1]
                        payload = target_rows[item_ordinal]
                        connection.execute(
                            """
                            INSERT OR REPLACE INTO migration_records
                            (batch_id, ordinal, pod_id, source_format_version,
                             source_line, source_sha, target_json, target_sha)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                batch_id,
                                item_ordinal,
                                item.pod_id,
                                item.source_format_version,
                                item.source_line,
                                digest_value(item.conclusion()),
                                payload,
                                digest_value(payload),
                            ),
                        )
                    connection.execute(
                        """
                        UPDATE migration_batches
                        SET checkpoint=?, updated_mono_ms=? WHERE batch_id=?
                        """,
                        (committed_end, monotonic_ms(), batch_id),
                    )
                    connection.execute("COMMIT")
                except BaseException:
                    connection.execute("ROLLBACK")
                    raise
                checkpoint = committed_end
                batches_since_abort += 1
                if (
                    abort_after_batches is not None
                    and batches_since_abort >= abort_after_batches
                ):
                    os._exit(137)

        rows = connection.execute(
            "SELECT ordinal, target_json FROM migration_records WHERE batch_id=? ORDER BY ordinal",
            (batch_id,),
        ).fetchall()
        if len(rows) != len(ordered):
            raise MigrateError(
                "committed migration rows do not cover the full record set",
                details={
                    "batch_id": batch_id,
                    "records": len(ordered),
                    "committed": len(rows),
                },
            )
        output = "".join(str(row["target_json"]) + "\n" for row in rows)
        atomic_write_text(source, output, mode=original_mode)
        connection.execute("BEGIN IMMEDIATE")
        try:
            connection.execute(
                "UPDATE migration_batches SET status='complete', checkpoint=?, updated_mono_ms=? WHERE batch_id=?",
                (len(ordered), monotonic_ms(), batch_id),
            )
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
        return {
            "status": "migrated",
            "source": str(source),
            "backup": str(backup),
            "backup_mode": "0444",
            "records": len(ordered),
            "checkpoint": len(ordered),
            "batch_id": batch_id,
            "source_sha": source_sha,
            "target_sha": digest_file(source),
            "source_format_versions": list(normalized.format_versions),
            "target_format_versions": [3],
        }
    finally:
        connection.close()


def normalize_current_json(
    target_json: str,
    *,
    params: Any,
    source_line: int,
) -> NormalizedRecord:
    from .records import decode_line

    return decode_line(
        target_json,
        source_line=source_line,
        source_path="<migrated-v3>",
        params=params,
    )
