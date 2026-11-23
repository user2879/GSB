"""Versioned session snapshots with self-digest and whole-state replacement."""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path
from typing import Any, Iterable

from .content import canonical_json_bytes, digest_bytes, digest_file, digest_json
from .errors import BusinessError, monotonic_ms, utc_now_iso
from .storage import GENESIS_HASH, Storage


SNAPSHOT_FORMAT_VERSION = 1

TABLE_COLUMNS: dict[str, tuple[str, ...]] = {
    "pages": (
        "page_id",
        "size_bytes",
        "initial_version",
        "initial_digest",
        "declaration_order",
        "page_table_version",
    ),
    "budget_params": (
        "singleton",
        "global_slots",
        "per_tx_quota",
        "page_byte_budget",
        "checkpoint_interval",
        "params_version",
        "params_digest",
    ),
    "wal_entries": (
        "lsn",
        "txid",
        "page_id",
        "content_digest",
        "commit_flag",
        "persist_flag",
        "target_version",
        "generated_digest",
        "visible",
        "tx_group",
        "loaded_at_ms",
    ),
    "page_versions": (
        "id",
        "page_id",
        "version",
        "lsn",
        "txid",
        "digest",
        "content_path",
        "visible",
        "kind",
        "batch_id",
        "committed_ms",
    ),
    "checkpoints": (
        "checkpoint_id",
        "run_id",
        "interval",
        "ledger_cursor",
        "page_count_since_checkpoint",
        "state_digest",
        "state_json",
        "created_ms",
    ),
    "replay_runs": (
        "run_id",
        "cursor_start",
        "cursor_end",
        "status",
        "budget_json",
        "started_ms",
        "finished_ms",
        "note",
    ),
    "replay_batches": (
        "batch_id",
        "run_id",
        "start_lsn",
        "end_lsn",
        "page_count",
        "bytes_consumed",
        "ledger_cursor",
        "checkpoint_id",
        "committed_ms",
        "payload_json",
        "prev_hash",
        "ledger_hash",
    ),
    "replay_tasks": (
        "task_id",
        "run_id",
        "txid",
        "page_id",
        "chunk_index",
        "lsn_start",
        "lsn_end",
        "entry_count",
        "planned_bytes",
        "state",
        "queued_ms",
        "wait_started_ms",
        "wait_ended_ms",
        "started_ms",
        "finished_ms",
        "committed_ms",
        "consumed_bytes",
        "batch_id",
        "detail_json",
    ),
    "ledger_events": (
        "event_id",
        "event_type",
        "cursor",
        "batch_id",
        "task_id",
        "prev_hash",
        "event_hash",
        "payload_json",
        "created_ms",
    ),
    "snapshot_meta": (
        "snapshot_id",
        "file_path",
        "format_version",
        "content_digest",
        "log_digest",
        "pages_digest",
        "budget_digest",
        "ledger_cursor",
        "created_ms",
    ),
}


def _resolve_snapshot_path(storage: Storage, path: str | os.PathLike[str]) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = storage.workspace / candidate
    return candidate.resolve()


def _relative_to_workspace(storage: Storage, path: Path) -> str:
    try:
        return path.relative_to(storage.workspace).as_posix()
    except ValueError:
        return str(path)


def _rows(storage: Storage, table: str) -> list[dict[str, Any]]:
    columns = ", ".join(TABLE_COLUMNS[table])
    with storage.read_connection() as conn:
        return [
            dict(row)
            for row in conn.execute(f"SELECT {columns} FROM {table} ORDER BY rowid").fetchall()
        ]


def _app_meta(storage: Storage) -> dict[str, str]:
    with storage.read_connection() as conn:
        return {
            row["key"]: row["value"]
            for row in conn.execute("SELECT key, value FROM app_meta ORDER BY key").fetchall()
        }


def _page_files(storage: Storage, page_version_rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    files: dict[str, Any] = {}
    for row in page_version_rows:
        content_path = row["content_path"]
        if content_path in files:
            continue
        target = storage.path_for(content_path)
        try:
            data = target.read_bytes()
        except OSError as exc:
            raise BusinessError(
                "E_SNAPSHOT",
                "cannot read page file for snapshot",
                {"page_id": row["page_id"], "version": row["version"], "path": content_path, "error": str(exc)},
            )
        actual = digest_bytes(data)
        if actual != row["digest"]:
            raise BusinessError(
                "E_SNAPSHOT",
                "page file digest differs from the committed metadata",
                {
                    "page_id": row["page_id"],
                    "version": row["version"],
                    "path": content_path,
                    "field": "page_file_digest",
                    "expected": row["digest"],
                    "actual": actual,
                },
            )
        files[content_path] = {
            "sha256": actual,
            "data_b64": base64.b64encode(data).decode("ascii"),
        }
    return dict(sorted(files.items()))


def _snapshot_payload(document: dict[str, Any]) -> bytes:
    payload = dict(document)
    payload.pop("content_digest", None)
    return canonical_json_bytes(payload)


def create_snapshot(storage: Storage, path: str | os.PathLike[str]) -> dict[str, Any]:
    if not storage.is_initialized():
        raise BusinessError("E_STATE", "workspace is not initialized")
    target = _resolve_snapshot_path(storage, path)
    target.parent.mkdir(parents=True, exist_ok=True)

    wal_path = storage.inputs_dir / "wal.ndjson"
    sample_path = storage.inputs_dir / "sample.json"
    try:
        wal_bytes = wal_path.read_bytes()
        wal_text = wal_bytes.decode("utf-8")
        sample_text = sample_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BusinessError("E_SNAPSHOT", "workspace input files are missing", {"error": str(exc)})
    except UnicodeDecodeError as exc:
        raise BusinessError("E_SNAPSHOT", "active WAL is not UTF-8", {"error": str(exc)})
    log_digest = digest_bytes(wal_bytes)
    if log_digest != storage.get_meta("active_wal_digest"):
        raise BusinessError(
            "E_SNAPSHOT",
            "active WAL file has changed since it was loaded",
            {
                "field": "log_digest",
                "expected": storage.get_meta("active_wal_digest"),
                "actual": log_digest,
            },
        )

    tables = {table: _rows(storage, table) for table in TABLE_COLUMNS}
    checkpoint_set = [
        {
            "checkpoint_id": row["checkpoint_id"],
            "ledger_cursor": row["ledger_cursor"],
            "state_digest": row["state_digest"],
        }
        for row in tables["checkpoints"]
    ]
    page_table_version = int(_app_meta(storage).get("page_table_version", "1"))
    budget_params_version = int(tables["budget_params"][0]["params_version"])
    created_ms = monotonic_ms()
    snapshot_id = f"snap-{time_ns_id()}"
    document: dict[str, Any] = {
        "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
        "snapshot_id": snapshot_id,
        "created_at": utc_now_iso(),
        "created_ms": created_ms,
        "log_digest": log_digest,
        "pages_digest": storage.get_meta("active_pages_digest"),
        "budget_digest": storage.get_meta("active_budget_digest"),
        "budget_params_version": budget_params_version,
        "page_table_version": page_table_version,
        "ledger_cursor": storage.ledger_cursor(),
        "checkpoint_set": checkpoint_set,
        "wal_text": wal_text,
        "inputs_sample": sample_text,
        "tables": tables,
        "app_meta": _app_meta(storage),
        "page_files": _page_files(storage, tables["page_versions"]),
    }
    content_digest = digest_json(json.loads(_snapshot_payload(document)))
    document["content_digest"] = content_digest
    serialized = json.dumps(document, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    temp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temp.write_text(serialized, encoding="utf-8", newline="\n")
    os.replace(temp, target)

    registered = {
        "snapshot_id": snapshot_id,
        "file_path": _relative_to_workspace(storage, target),
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "content_digest": content_digest,
        "log_digest": log_digest,
        "pages_digest": document["pages_digest"],
        "budget_digest": document["budget_digest"],
        "ledger_cursor": document["ledger_cursor"],
        "created_ms": created_ms,
    }
    with storage.transaction() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO snapshot_meta(snapshot_id, file_path, format_version, "
            "content_digest, log_digest, pages_digest, budget_digest, ledger_cursor, created_ms) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                registered["snapshot_id"],
                registered["file_path"],
                registered["format_version"],
                registered["content_digest"],
                registered["log_digest"],
                registered["pages_digest"],
                registered["budget_digest"],
                registered["ledger_cursor"],
                registered["created_ms"],
            ),
        )
    return {
        "ok": True,
        "snapshot_id": snapshot_id,
        "path": _relative_to_workspace(storage, target),
        "format_version": SNAPSHOT_FORMAT_VERSION,
        "content_digest": content_digest,
        "ledger_cursor": document["ledger_cursor"],
        "checkpoint_count": len(checkpoint_set),
        "page_file_count": len(document["page_files"]),
    }


def time_ns_id() -> str:
    return f"{__import__('time').time_ns():x}"


def _load_document(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise BusinessError("E_SNAPSHOT", "cannot read snapshot", {"path": str(path), "error": str(exc)})
    except json.JSONDecodeError as exc:
        raise BusinessError("E_SNAPSHOT", "snapshot is not valid JSON", {"path": str(path), "error": str(exc)})
    if not isinstance(document, dict):
        raise BusinessError("E_SNAPSHOT", "snapshot root must be an object", {"path": str(path)})
    return document


def _verify_snapshot_digest(document: dict[str, Any]) -> str:
    supplied = document.get("content_digest")
    expected = digest_json(json.loads(_snapshot_payload(document)))
    if not isinstance(supplied, str) or supplied != expected:
        raise BusinessError(
            "E_SNAPSHOT",
            "snapshot content digest does not match its payload",
            {
                "field": "content_digest",
                "expected": expected,
                "actual": supplied,
            },
        )
    return expected


def _validate_embedded(document: dict[str, Any]) -> None:
    version = document.get("snapshot_format_version")
    if isinstance(version, bool) or not isinstance(version, int) or version != SNAPSHOT_FORMAT_VERSION:
        raise BusinessError(
            "E_SNAPSHOT",
            "unsupported snapshot format version",
            {
                "field": "snapshot_format_version",
                "expected": SNAPSHOT_FORMAT_VERSION,
                "actual": version,
            },
        )
    required = (
        "snapshot_id",
        "log_digest",
        "pages_digest",
        "budget_digest",
        "budget_params_version",
        "page_table_version",
        "ledger_cursor",
        "checkpoint_set",
        "wal_text",
        "tables",
        "app_meta",
        "page_files",
    )
    missing = [key for key in required if key not in document]
    if missing:
        raise BusinessError("E_SNAPSHOT", "snapshot is missing required fields", {"missing": missing})
    if not isinstance(document["snapshot_id"], str) or not document["snapshot_id"].strip():
        raise BusinessError("E_SNAPSHOT", "snapshot_id must be a non-empty string", {"field": "snapshot_id"})
    if (
        isinstance(document["page_table_version"], bool)
        or not isinstance(document["page_table_version"], int)
        or document["page_table_version"] < 1
    ):
        raise BusinessError(
            "E_SNAPSHOT",
            "page_table_version must be a positive integer",
            {"field": "page_table_version", "actual": document["page_table_version"]},
        )
    if (
        isinstance(document["budget_params_version"], bool)
        or not isinstance(document["budget_params_version"], int)
        or document["budget_params_version"] < 1
    ):
        raise BusinessError(
            "E_SNAPSHOT",
            "budget_params_version must be a positive integer",
            {"field": "budget_params_version", "actual": document["budget_params_version"]},
        )
    if (
        isinstance(document["ledger_cursor"], bool)
        or not isinstance(document["ledger_cursor"], int)
        or document["ledger_cursor"] < 0
    ):
        raise BusinessError(
            "E_SNAPSHOT",
            "ledger_cursor must be a non-negative integer",
            {"field": "ledger_cursor", "actual": document["ledger_cursor"]},
        )
    wal_text = document["wal_text"]
    if not isinstance(wal_text, str) or digest_bytes(wal_text.encode("utf-8")) != document["log_digest"]:
        raise BusinessError(
            "E_SNAPSHOT",
            "snapshot WAL bytes do not match its log digest",
            {"field": "log_digest", "expected": document["log_digest"]},
        )
    tables = document["tables"]
    if not isinstance(tables, dict) or any(table not in tables for table in TABLE_COLUMNS):
        raise BusinessError("E_SNAPSHOT", "snapshot table set is incomplete", {"field": "tables"})
    pages_payload = [
        {
            "page_id": row["page_id"],
            "size_bytes": row["size_bytes"],
            "initial_version": row["initial_version"],
        }
        for row in sorted(tables["pages"], key=lambda row: row["page_id"])
    ]
    if digest_json(pages_payload) != document["pages_digest"]:
        raise BusinessError(
            "E_SNAPSHOT",
            "snapshot page-table digest is inconsistent",
            {"field": "pages_digest", "expected": document["pages_digest"]},
        )
    if not tables["pages"]:
        raise BusinessError("E_SNAPSHOT", "snapshot must contain at least one page", {"field": "pages"})
    if any(
        row["page_table_version"] != document["page_table_version"] for row in tables["pages"]
    ):
        raise BusinessError(
            "E_SNAPSHOT",
            "snapshot page_table_version is inconsistent with page rows",
            {"field": "page_table_version", "expected": document["page_table_version"]},
        )
    budget_rows = tables["budget_params"]
    if len(budget_rows) != 1:
        raise BusinessError("E_SNAPSHOT", "snapshot must contain exactly one budget row", {"field": "budget_params"})
    budget_payload = {
        "global_slots": budget_rows[0]["global_slots"],
        "per_tx_quota": budget_rows[0]["per_tx_quota"],
        "page_byte_budget": budget_rows[0]["page_byte_budget"],
        "checkpoint_interval": budget_rows[0]["checkpoint_interval"],
    }
    if digest_json(budget_payload) != document["budget_digest"]:
        raise BusinessError(
            "E_SNAPSHOT",
            "snapshot budget digest is inconsistent",
            {"field": "budget_digest", "expected": document["budget_digest"]},
        )
    if document["budget_params_version"] != budget_rows[0]["params_version"]:
        raise BusinessError(
            "E_SNAPSHOT",
            "snapshot budget parameter version is inconsistent",
            {
                "field": "budget_params_version",
                "expected": budget_rows[0]["params_version"],
                "actual": document["budget_params_version"],
            },
        )
    checkpoint_set = [
        {
            "checkpoint_id": row["checkpoint_id"],
            "ledger_cursor": row["ledger_cursor"],
            "state_digest": row["state_digest"],
        }
        for row in tables["checkpoints"]
    ]
    if checkpoint_set != document["checkpoint_set"]:
        raise BusinessError(
            "E_SNAPSHOT",
            "snapshot checkpoint set is inconsistent with checkpoint rows",
            {"field": "checkpoint_set"},
        )
    page_files = document["page_files"]
    if not isinstance(page_files, dict):
        raise BusinessError("E_SNAPSHOT", "snapshot page_files must be an object", {"field": "page_files"})
    for row in tables["page_versions"]:
        file_info = page_files.get(row["content_path"])
        if not isinstance(file_info, dict) or file_info.get("sha256") != row["digest"]:
            raise BusinessError(
                "E_SNAPSHOT",
                "snapshot page-version rows are not covered by matching page files",
                {
                    "field": "page_files",
                    "page_id": row["page_id"],
                    "version": row["version"],
                    "path": row["content_path"],
                },
            )
    app_meta = document["app_meta"]
    if not isinstance(app_meta, dict) or int(app_meta.get("ledger_cursor", -1)) != int(document["ledger_cursor"]):
        raise BusinessError(
            "E_SNAPSHOT",
            "snapshot ledger cursor is inconsistent with app metadata",
            {"field": "ledger_cursor"},
        )

    # Verify the embedded hash chain before any current state is touched.
    previous = GENESIS_HASH
    for event in tables["ledger_events"]:
        if event["prev_hash"] != previous:
            raise BusinessError(
                "E_SNAPSHOT",
                "snapshot ledger chain is broken",
                {"field": "prev_hash", "event_id": event["event_id"]},
            )
        recomputed = Storage._event_hash(
            event["event_type"],
            event["cursor"],
            event["batch_id"],
            event["task_id"],
            event["prev_hash"],
            event["payload_json"],
            event["created_ms"],
        )
        if recomputed != event["event_hash"]:
            raise BusinessError(
                "E_SNAPSHOT",
                "snapshot ledger event hash is invalid",
                {"field": "event_hash", "event_id": event["event_id"]},
            )
        previous = event["event_hash"]


def _current_page_signatures(storage: Storage) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for item in storage.visible_page_state():
        disk_digest: str | None
        try:
            disk_digest = digest_file(storage.path_for(item["content_path"]))
        except (OSError, TypeError):
            disk_digest = None
        result[item["page_id"]] = {
            "page_id": item["page_id"],
            "version": item["visible_version"],
            "digest": item["visible_digest"],
            "disk_digest": disk_digest,
        }
    return result


def _target_page_signatures(document: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = document["tables"]["page_versions"]
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not row["visible"]:
            continue
        current = latest.get(row["page_id"])
        if current is None or row["version"] > current["version"]:
            latest[row["page_id"]] = row
    return {
        page_id: {
            "page_id": page_id,
            "version": row["version"],
            "digest": row["digest"],
        }
        for page_id, row in sorted(latest.items())
    }


def _page_diff(
    current: dict[str, dict[str, Any]], target: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    current_ids = set(current)
    target_ids = set(target)
    changed = []
    for page_id in sorted(current_ids & target_ids):
        before = current[page_id]
        after = target[page_id]
        if (
            before["version"] != after["version"]
            or before["digest"] != after["digest"]
            or before.get("disk_digest") != after["digest"]
        ):
            changed.append(
                {
                    "page_id": page_id,
                    "current_version": before["version"],
                    "current_digest": before["digest"],
                    "current_disk_digest": before.get("disk_digest"),
                    "snapshot_version": after["version"],
                    "snapshot_digest": after["digest"],
                }
            )
    return {
        "added": sorted(target_ids - current_ids),
        "deleted": sorted(current_ids - target_ids),
        "changed": changed,
    }


def _prepare_page_files(storage: Storage, document: dict[str, Any]) -> None:
    for content_path, file_info in sorted(document["page_files"].items()):
        if not isinstance(file_info, dict) or "data_b64" not in file_info:
            raise BusinessError("E_SNAPSHOT", "invalid page file entry", {"path": content_path})
        try:
            data = base64.b64decode(file_info["data_b64"], validate=True)
        except Exception as exc:
            raise BusinessError(
                "E_SNAPSHOT",
                "snapshot page file is not valid base64",
                {"path": content_path, "error": str(exc)},
            )
        actual = digest_bytes(data)
        if actual != file_info.get("sha256"):
            raise BusinessError(
                "E_SNAPSHOT",
                "snapshot page file digest is invalid",
                {
                    "path": content_path,
                    "field": "page_file_digest",
                    "expected": file_info.get("sha256"),
                    "actual": actual,
                },
            )
        storage.write_page_file(content_path, data)


def _insert_rows(conn: Any, table: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    columns = TABLE_COLUMNS[table]
    placeholders = ", ".join("?" for _ in columns)
    sql = f"INSERT INTO {table}({', '.join(columns)}) VALUES({placeholders})"
    conn.executemany(sql, [tuple(row.get(column) for column in columns) for row in rows])


def _cleanup_orphan_page_files(storage: Storage) -> None:
    with storage.read_connection() as conn:
        referenced = {
            row["content_path"]
            for row in conn.execute("SELECT content_path FROM page_versions").fetchall()
        }
    root = storage.pages_dir
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file() and path.relative_to(storage.workspace).as_posix() not in referenced:
            try:
                path.unlink()
            except OSError:
                pass
    for directory in sorted((path for path in root.rglob("*") if path.is_dir()), reverse=True):
        try:
            directory.rmdir()
        except OSError:
            pass


def restore_snapshot(storage: Storage, path: str | os.PathLike[str]) -> dict[str, Any]:
    if not storage.is_initialized():
        raise BusinessError("E_STATE", "workspace is not initialized")
    target = _resolve_snapshot_path(storage, path)
    document = _load_document(target)
    content_digest = _verify_snapshot_digest(document)
    _validate_embedded(document)

    current_signatures = _current_page_signatures(storage)
    target_signatures = _target_page_signatures(document)
    difference = _page_diff(current_signatures, target_signatures)
    metadata_log_digest = storage.get_meta("active_wal_digest")
    try:
        current_log_digest = digest_file(storage.inputs_dir / "wal.ndjson")
    except OSError:
        current_log_digest = None
    log_changed = current_log_digest != document["log_digest"]

    # Validate all embedded binary assets before mutating any current state.
    _prepare_page_files(storage, document)
    storage.inputs_dir.mkdir(parents=True, exist_ok=True)
    (storage.inputs_dir / "wal.ndjson").write_text(
        document["wal_text"], encoding="utf-8", newline="\n"
    )
    sample_text = document.get("inputs_sample")
    if not isinstance(sample_text, str) or not sample_text.strip():
        raise BusinessError("E_SNAPSHOT", "snapshot inputs_sample is missing", {"field": "inputs_sample"})
    (storage.inputs_dir / "sample.json").write_text(sample_text, encoding="utf-8", newline="\n")

    tables = document["tables"]
    conn = storage.connect()
    try:
        with storage.transaction(conn):
            # Whole-state replacement, not a row-by-row merge.
            for table in (
                "ledger_events",
                "replay_tasks",
                "replay_batches",
                "replay_runs",
                "checkpoints",
                "wal_entries",
                "page_versions",
                "pages",
                "budget_params",
                "app_meta",
                "snapshot_meta",
            ):
                conn.execute(f"DELETE FROM {table}")
            for table in (
                "pages",
                "budget_params",
                "wal_entries",
                "page_versions",
                "checkpoints",
                "replay_runs",
                "replay_batches",
                "replay_tasks",
                "ledger_events",
                "snapshot_meta",
            ):
                _insert_rows(conn, table, tables.get(table, []))
            for key, value in sorted(document["app_meta"].items()):
                conn.execute(
                    "INSERT OR REPLACE INTO app_meta(key, value) VALUES(?, ?)", (key, value)
                )
            conn.execute(
                "INSERT OR REPLACE INTO snapshot_meta(snapshot_id, file_path, format_version, "
                "content_digest, log_digest, pages_digest, budget_digest, ledger_cursor, created_ms) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    document["snapshot_id"],
                    _relative_to_workspace(storage, target),
                    document["snapshot_format_version"],
                    content_digest,
                    document["log_digest"],
                    document["pages_digest"],
                    document["budget_digest"],
                    document["ledger_cursor"],
                    int(document.get("created_ms", monotonic_ms())),
                ),
            )
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    _cleanup_orphan_page_files(storage)
    ledger_check = storage.verify_ledger()
    if not ledger_check.get("ok"):
        raise BusinessError(
            "E_SNAPSHOT",
            "restored ledger failed independent verification",
            {"ledger_check": ledger_check},
        )
    return {
        "ok": True,
        "snapshot_id": document["snapshot_id"],
        "path": _relative_to_workspace(storage, target),
        "format_version": document["snapshot_format_version"],
        "content_digest": content_digest,
        "restore_mode": "whole_state_replacement",
        "log_changed": log_changed,
        "current_log_digest": current_log_digest,
        "current_metadata_log_digest": metadata_log_digest,
        "snapshot_log_digest": document["log_digest"],
        "page_diff": difference,
        "restored": {
            "pages": len(tables["pages"]),
            "page_versions": len(tables["page_versions"]),
            "checkpoints": len(tables["checkpoints"]),
            "replay_batches": len(tables["replay_batches"]),
            "replay_tasks": len(tables["replay_tasks"]),
            "ledger_events": len(tables["ledger_events"]),
            "ledger_cursor": document["ledger_cursor"],
        },
        "ledger": ledger_check,
    }
