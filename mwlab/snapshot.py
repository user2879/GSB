"""Version-identified SQLite session snapshots with wholesale restoration."""

from __future__ import annotations

import os
import re
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from .canonical import make_read_only, read_json, utc_now, write_json
from .errors import LabError, io_error, state_error
from .ledger import Ledger, state_content_digest


SNAPSHOT_NAME_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,78}[A-Za-z0-9])?$")


def _timestamp_token() -> str:
    return utc_now().replace(":", "").replace("-", "").replace(".", "")


def _validate_name(name: str) -> None:
    if not SNAPSHOT_NAME_RE.fullmatch(name):
        raise state_error(
            "invalid snapshot name",
            name=name,
            rule="letters, digits, dot, underscore and dash; first and last characters alphanumeric",
        )


def _readonly_backup(source: Path, backup_root: Path, prefix: str) -> Path:
    backup_root.mkdir(parents=True, exist_ok=True)
    target = backup_root / f"{prefix}.{_timestamp_token()}"
    if target.exists():
        target = backup_root / f"{prefix}.{_timestamp_token()}.{os.getpid()}"
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
    make_read_only(target)
    return target


def save_snapshot(
    ledger_path: str | Path,
    snapshot_root: str | Path,
    backup_root: str | Path,
    name: str,
) -> dict[str, Any]:
    _validate_name(name)
    ledger = Ledger(ledger_path)
    try:
        state = ledger.export_state()
        if state is None:
            raise state_error("ledger has no current execution")
        if state["execution_status"] != "complete":
            raise state_error(
                "incomplete executions cannot be snapshotted",
                execution_id=state["execution_id"],
                status=state["execution_status"],
                rerun_required=True,
            )
        computed_digest = state_content_digest(state)
        stored_digest = state.get("content_digest")
        if stored_digest is not None and stored_digest != computed_digest:
            raise LabError(
                "E_SNAPSHOT",
                "ledger content digest does not match exported state",
                {"stored": stored_digest, "computed": computed_digest},
            )
        snapshot_dir = Path(snapshot_root) / name
        backup_path = None
        if snapshot_dir.exists():
            backup_path = _readonly_backup(snapshot_dir, Path(backup_root), f"snapshot-{name}")
            shutil.rmtree(snapshot_dir)
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        database_path = snapshot_dir / "ledger.sqlite3"
        destination = sqlite3.connect(str(database_path))
        try:
            ledger.connection.backup(destination)
        finally:
            destination.close()
        document = {
            "snapshot_schema_version": 1,
            "name": name,
            "created_at_utc": utc_now(),
            "format_versions": state["format_versions"],
            "parameter_version": state["parameter_version"],
            "content_digest": computed_digest,
            "state": state,
        }
        write_json(snapshot_dir / "snapshot.json", document)
    finally:
        ledger.close()
    return {
        "status": "saved",
        "name": name,
        "snapshot_dir": str(snapshot_dir),
        "content_digest": computed_digest,
        "format_versions": state["format_versions"],
        "parameter_version": state["parameter_version"],
        "previous_snapshot_backup": None if backup_path is None else str(backup_path),
    }


def load_snapshot(
    snapshot_root: str | Path,
    name: str,
    *,
    inject_digest_mismatch: bool = False,
) -> dict[str, Any]:
    _validate_name(name)
    snapshot_dir = Path(snapshot_root) / name
    document_path = snapshot_dir / "snapshot.json"
    if not document_path.exists():
        raise io_error("snapshot not found", name=name, path=str(document_path))
    document = read_json(document_path)
    state = document.get("state")
    if not isinstance(state, dict):
        raise LabError("E_SNAPSHOT", "snapshot has no state object", {"name": name})
    computed = state_content_digest(state)
    stored = str(document.get("content_digest", ""))
    if inject_digest_mismatch:
        stored = ("0" * 64) if stored != "0" * 64 else ("1" * 64)
    if stored != computed:
        raise LabError(
            "E_SNAPSHOT",
            "snapshot content digest mismatch",
            {
                "name": name,
                "stored": stored,
                "computed": computed,
                "injected": inject_digest_mismatch,
            },
        )
    if document.get("format_versions") != state.get("format_versions"):
        raise LabError(
            "E_SNAPSHOT",
            "snapshot format version set is internally inconsistent",
            {
                "name": name,
                "document_formats": document.get("format_versions"),
                "state_formats": state.get("format_versions"),
            },
        )
    database_path = snapshot_dir / "ledger.sqlite3"
    if not database_path.exists():
        raise LabError("E_SNAPSHOT", "snapshot database is missing", {"name": name})
    return {
        "name": name,
        "snapshot_dir": str(snapshot_dir),
        "document": document,
        "state": state,
        "content_digest": computed,
        "database_path": str(database_path),
    }


def restore_snapshot(
    ledger_path: str | Path,
    snapshot_root: str | Path,
    backup_root: str | Path,
    name: str,
    *,
    inject_digest_mismatch: bool = False,
) -> dict[str, Any]:
    loaded = load_snapshot(
        snapshot_root,
        name,
        inject_digest_mismatch=inject_digest_mismatch,
    )
    ledger_path = Path(ledger_path)
    mismatch_basis: list[str] = []
    current_digest = None
    current_formats = None
    if ledger_path.exists():
        current = Ledger(ledger_path)
        try:
            current_state = current.export_state()
            if current_state is not None:
                current_digest = state_content_digest(current_state)
                current_formats = current_state["format_versions"]
                if sorted(current_formats) != sorted(loaded["state"]["format_versions"]):
                    mismatch_basis.append("format_versions")
                if current_digest != loaded["content_digest"]:
                    mismatch_basis.append("content_digest")
        finally:
            current.close()
    else:
        mismatch_basis.append("ledger_absent")

    backup_path = None
    if ledger_path.exists():
        backup_path = _readonly_backup(ledger_path, Path(backup_root), "ledger-before-restore")
    for suffix in ("-wal", "-shm", "-journal"):
        sidecar = Path(str(ledger_path) + suffix)
        if sidecar.exists():
            sidecar.unlink()
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = ledger_path.with_name(f".{ledger_path.name}.restore-{os.getpid()}.tmp")
    if temporary.exists():
        temporary.unlink()
    shutil.copy2(loaded["database_path"], temporary)
    os.replace(temporary, ledger_path)

    restored = Ledger(ledger_path)
    try:
        restored_state = restored.export_state()
        if restored_state is None:
            raise LabError("E_SNAPSHOT", "restored ledger has no current execution", {"name": name})
        restored_digest = state_content_digest(restored_state)
        if restored_digest != loaded["content_digest"]:
            raise LabError(
                "E_SNAPSHOT",
                "restored ledger content digest differs from snapshot",
                {
                    "name": name,
                    "expected": loaded["content_digest"],
                    "actual": restored_digest,
                },
            )
    finally:
        restored.close()
    return {
        "status": "restored",
        "name": name,
        "replacement_mode": "wholesale_copy_no_merge",
        "compared_format_versions": {
            "current": current_formats,
            "snapshot": loaded["state"]["format_versions"],
        },
        "compared_content_digest": {
            "current": current_digest,
            "snapshot": loaded["content_digest"],
        },
        "mismatch_basis": mismatch_basis,
        "replaced_due_to_mismatch": bool(mismatch_basis),
        "previous_ledger_backup": None if backup_path is None else str(backup_path),
        "content_digest": loaded["content_digest"],
        "format_versions": loaded["state"]["format_versions"],
        "parameter_version": loaded["state"]["parameter_version"],
    }


def list_snapshots(snapshot_root: str | Path) -> list[dict[str, Any]]:
    root = Path(snapshot_root)
    if not root.exists():
        return []
    rows: list[dict[str, Any]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        document_path = child / "snapshot.json"
        database_path = child / "ledger.sqlite3"
        if not document_path.exists() or not database_path.exists():
            continue
        try:
            document = read_json(document_path)
        except LabError:
            continue
        rows.append(
            {
                "name": child.name,
                "created_at_utc": document.get("created_at_utc"),
                "parameter_version": document.get("parameter_version"),
                "format_versions": document.get("format_versions"),
                "content_digest": document.get("content_digest"),
            }
        )
    return rows
