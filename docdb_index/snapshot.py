"""Versioned, content-addressed whole-session snapshots and replacement restore."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
from pathlib import Path
from typing import Any

from .canonical import canonical_json, digest_json
from .errors import SnapshotError, StateError
from .storage import (
    ledger_digest,
    ledger_rows,
    open_session,
    query_history_digest,
    query_history_rows,
    segment_manifest,
    session_db_path,
    state_manifest,
)


SNAPSHOT_FORMAT = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest_without_digest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "content_digest"}


def _attach_content_digest(manifest: dict[str, Any]) -> dict[str, Any]:
    result = dict(manifest)
    result.pop("content_digest", None)
    result["content_digest"] = digest_json(result)
    return result


def snapshot_state(session_dir: str | Path) -> dict[str, Any]:
    conn = open_session(session_dir)
    try:
        state = state_manifest(conn)
        ledger = ledger_rows(conn)
        history = query_history_rows(conn)
        return {
            "collection_versions": state["collection_versions"],
            "index_versions": state["index_versions"],
            "segments": state["segments"],
            "segment_digests": state["segment_digests"],
            "ledger_digest": ledger_digest(conn),
            "ledger": ledger,
            "query_history_digest": query_history_digest(conn),
            "query_history_count": len(history),
        }
    finally:
        conn.close()


def _next_label(snapshot_dir: Path) -> str:
    maximum = 0
    for path in snapshot_dir.iterdir() if snapshot_dir.exists() else []:
        if not path.is_dir():
            continue
        name = path.name
        if name.startswith("snap-") and name[5:].isdigit():
            maximum = max(maximum, int(name[5:]))
    return f"snap-{maximum + 1:06d}"


def _make_readonly(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        try:
            if child.is_dir():
                child.chmod(0o555)
            else:
                child.chmod(0o444)
        except OSError:
            pass
    try:
        path.chmod(0o555)
    except OSError:
        pass


def _make_writable(path: Path) -> None:
    for child in path.rglob("*"):
        try:
            if child.is_dir():
                child.chmod(0o755)
            else:
                child.chmod(0o644)
        except OSError:
            pass
    try:
        path.chmod(0o755)
    except OSError:
        pass


def create_snapshot(
    session_dir: str | Path,
    *,
    label: str | None = None,
) -> dict[str, Any]:
    session_path = Path(session_dir)
    db_path = session_db_path(session_path)
    if not db_path.is_file():
        raise StateError("session database does not exist", session=str(session_path))
    snapshot_root = session_path / "snapshots"
    snapshot_root.mkdir(parents=True, exist_ok=True)
    chosen = label or _next_label(snapshot_root)
    if (
        not chosen
        or "/" in chosen
        or "\\" in chosen
        or chosen in {".", ".."}
        or any(ord(char) < 32 for char in chosen)
    ):
        raise SnapshotError("invalid snapshot label", label=chosen)
    target = snapshot_root / chosen
    if target.exists():
        raise SnapshotError("snapshot already exists", label=chosen, path=str(target))
    target.mkdir(parents=False, exist_ok=False)
    temp_db = target / ".snapshot.db.tmp"
    source = open_session(session_path)
    destination: sqlite3.Connection | None = None
    try:
        source.execute("PRAGMA wal_checkpoint(FULL)")
        destination = sqlite3.connect(str(temp_db))
        source.backup(destination)
        destination.close()
        destination = None
        db_sha256 = _sha256_file(temp_db)
        state = state_manifest(source)
        manifest = _attach_content_digest(
            {
                "format": SNAPSHOT_FORMAT,
                "label": chosen,
                "collection_versions": state["collection_versions"],
                "index_versions": state["index_versions"],
                "segments": state["segments"],
                "segment_digests": state["segment_digests"],
                "ledger_digest": ledger_digest(source),
                "ledger": ledger_rows(source),
                "query_history_digest": query_history_digest(source),
                "query_history_count": len(query_history_rows(source)),
                "db_sha256": db_sha256,
            }
        )
    finally:
        if destination is not None:
            destination.close()
        source.close()

    os.replace(temp_db, target / "snapshot.db")
    (target / "manifest.json").write_text(
        canonical_json(manifest) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    for sibling in snapshot_root.iterdir():
        if sibling.is_dir() and sibling != target and (sibling / "manifest.json").is_file():
            _make_readonly(sibling)
    _make_readonly(target)
    return {
        "ok": True,
        "label": chosen,
        "path": str(target.resolve()),
        "content_digest": manifest["content_digest"],
        "db_sha256": manifest["db_sha256"],
        "read_only": True,
    }


def load_snapshot_manifest(path: str | Path) -> dict[str, Any]:
    snapshot_path = Path(path)
    manifest_path = snapshot_path / "manifest.json"
    db_path = snapshot_path / "snapshot.db"
    if not manifest_path.is_file() or not db_path.is_file():
        raise SnapshotError("snapshot is incomplete", path=str(snapshot_path))
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise SnapshotError("snapshot manifest cannot be read", path=str(manifest_path)) from exc
    claimed = manifest.get("content_digest")
    computed = digest_json(_manifest_without_digest(manifest))
    if claimed != computed:
        raise SnapshotError(
            "snapshot content digest does not match its manifest",
            path=str(snapshot_path),
            expected=computed,
            actual=claimed,
        )
    actual_db_digest = _sha256_file(db_path)
    if actual_db_digest != manifest.get("db_sha256"):
        raise SnapshotError(
            "snapshot database digest does not match its manifest",
            path=str(snapshot_path),
            expected=manifest.get("db_sha256"),
            actual=actual_db_digest,
        )
    return manifest


def list_snapshots(session_dir: str | Path) -> list[dict[str, Any]]:
    root = Path(session_dir) / "snapshots"
    result: list[dict[str, Any]] = []
    if not root.exists():
        return result
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_dir() or not (path / "manifest.json").is_file():
            continue
        manifest = load_snapshot_manifest(path)
        result.append(
            {
                "label": manifest["label"],
                "path": str(path.resolve()),
                "collection_versions": manifest["collection_versions"],
                "index_versions": manifest["index_versions"],
                "segment_count": len(manifest["segments"]),
                "ledger_count": len(manifest["ledger"]),
                "content_digest": manifest["content_digest"],
            }
        )
    return result


def _snapshot_differences(snapshot: dict[str, Any], current: dict[str, Any]) -> list[dict[str, Any]]:
    dimensions = (
        "collection_versions",
        "index_versions",
        "segment_digests",
        "ledger_digest",
        "query_history_digest",
    )
    differences: list[dict[str, Any]] = []
    for dimension in dimensions:
        if snapshot.get(dimension) != current.get(dimension):
            differences.append(
                {
                    "dimension": dimension,
                    "snapshot": snapshot.get(dimension),
                    "current": current.get(dimension),
                }
            )
    snapshot_segments = {
        f"{item['collection']}|{item['index_path']}|{item['segment_id']}": item
        for item in snapshot.get("segments", [])
    }
    current_segments = {
        f"{item['collection']}|{item['index_path']}|{item['segment_id']}": item
        for item in current.get("segments", [])
    }
    detail: list[dict[str, Any]] = []
    for key in sorted(set(snapshot_segments) | set(current_segments)):
        old = snapshot_segments.get(key)
        new = current_segments.get(key)
        if old == new:
            continue
        detail.append(
            {
                "segment_key": key,
                "segment_id": key.rsplit("|", 1)[-1],
                "snapshot_digest": None if old is None else old.get("digest"),
                "current_digest": None if new is None else new.get("digest"),
                "snapshot_version": None if old is None else old.get("version"),
                "current_version": None if new is None else new.get("version"),
                "snapshot_entry_count": None if old is None else old.get("entry_count"),
                "current_entry_count": None if new is None else new.get("entry_count"),
            }
        )
    if detail:
        differences.append({"dimension": "segment_changes", "details": detail})
    return differences


def restore_snapshot(
    session_dir: str | Path,
    snapshot_path: str | Path,
) -> dict[str, Any]:
    session_path = Path(session_dir)
    snapshot_path = Path(snapshot_path)
    manifest = load_snapshot_manifest(snapshot_path)
    current = snapshot_state(session_path)
    differences = _snapshot_differences(manifest, current)
    same = not differences
    source_db = snapshot_path / "snapshot.db"
    current_db = session_db_path(session_path)

    rollback_dir = session_path / "rollback"
    rollback_dir.mkdir(parents=True, exist_ok=True)
    rollback_path = rollback_dir / f"pre-restore-{manifest['label']}.db"
    if rollback_path.exists():
        try:
            rollback_path.chmod(0o644)
            rollback_path.unlink()
        except OSError:
            pass
    if current_db.is_file():
        shutil.copyfile(current_db, rollback_path)
        try:
            rollback_path.chmod(0o444)
        except OSError:
            pass

    temp_restore = session_path / f".restore.db.tmp-{os.getpid()}"
    for suffix in ("", "-wal", "-shm"):
        try:
            Path(str(temp_restore) + suffix).unlink()
        except FileNotFoundError:
            pass
    try:
        shutil.copyfile(source_db, temp_restore)
        os.replace(temp_restore, current_db)
        for suffix in ("-wal", "-shm"):
            try:
                Path(str(current_db) + suffix).unlink()
            except FileNotFoundError:
                pass
        reopened = snapshot_state(session_path)
        verify_diff = _snapshot_differences(manifest, reopened)
        if verify_diff:
            raise SnapshotError(
                "restored session does not match the snapshot manifest",
                differences=verify_diff,
            )
    finally:
        for path in (temp_restore, Path(str(temp_restore) + "-wal"), Path(str(temp_restore) + "-shm")):
            try:
                path.unlink()
            except FileNotFoundError:
                pass
    _make_readonly(snapshot_path)
    return {
        "ok": True,
        "label": manifest["label"],
        "restored_from": str(snapshot_path.resolve()),
        "replaced": True,
        "same_state": same,
        "differences": differences,
        "rollback_copy": str(rollback_path.resolve()) if rollback_path.exists() else None,
        "content_digest": manifest["content_digest"],
    }
