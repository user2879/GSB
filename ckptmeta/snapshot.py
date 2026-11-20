from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .canonical import canonical_json_text, digest_json, load_json, read_bytes, sha256_hex
from .catalog import CHECKPOINT_NAME, Catalog, scan_catalog
from .config import InteropConfig, load_config
from .errors import fail
from .ledger import Ledger, build_ledger
from .normalize import normalize_manifest
from .view import first_difference


SNAPSHOT_SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _manifest_disk_facts(catalog: Catalog) -> tuple[list[dict[str, Any]], list[int], str]:
    facts: list[dict[str, Any]] = []
    versions: set[int] = set()
    manifests = scan_catalog(catalog.directory)
    for manifest in manifests:
        assert manifest.source_path is not None
        relative = manifest.source_path.relative_to(catalog.directory).as_posix()
        facts.append({"path": relative, "sha256": sha256_hex(read_bytes(manifest.source_path))})
        versions.add(manifest.version)
    facts.sort(key=lambda item: item["path"])
    return facts, sorted(versions), digest_json(facts)


def read_checkpoint_trace(catalog: Catalog) -> tuple[list[dict[str, Any]], int]:
    path = catalog.checkpoint_path
    if not path.exists():
        return [], 0
    try:
        raw = load_json(path, code="E_UPGRADE")
    except BaseException as exc:
        if getattr(exc, "code", None):
            fail("E_REPLAY", f"cannot read upgrade checkpoint for replay: {exc}", manifest_path=str(path))
        raise
    if not isinstance(raw, dict) or not isinstance(raw.get("items"), list):
        fail("E_REPLAY", "upgrade checkpoint has no item trace", manifest_path=str(path))
    trace: list[dict[str, Any]] = []
    done = 0
    for item in raw["items"]:
        status = item.get("status")
        if status == "done":
            done += 1
        trace.append(
            {
                "manifest_path": item.get("path"),
                "from_version": item.get("source_version"),
                "to_version": raw.get("target_version"),
                "status": status,
            }
        )
    return trace, done


def capture_state(
    catalog: Catalog,
    *,
    upgrade_trace: list[dict[str, Any]] | None = None,
    processed_cursor: int | None = None,
) -> dict[str, Any]:
    facts, versions, content_digest = _manifest_disk_facts(catalog)
    items: list[dict[str, Any]] = []
    manifests = scan_catalog(catalog.directory)
    for manifest in manifests:
        assert manifest.source_path is not None
        normalized = normalize_manifest(manifest, catalog.ledger)
        relative = manifest.source_path.relative_to(catalog.directory).as_posix()
        items.append(
            {
                "manifest_path": relative,
                "manifest_id": normalized.view["manifest_id"],
                "source_version": manifest.version,
                "target_version": 3,
                "normalized_json": canonical_json_text(normalized.view),
                "view_digest": normalized.view_digest,
            }
        )
    items.sort(key=lambda item: item["manifest_path"])

    if upgrade_trace is None or processed_cursor is None:
        checkpoint_trace, checkpoint_cursor = read_checkpoint_trace(catalog)
        if upgrade_trace is None:
            upgrade_trace = checkpoint_trace
        if processed_cursor is None:
            processed_cursor = checkpoint_cursor
    normalized_digest = digest_json(
        [{"path": item["manifest_path"], "view_digest": item["view_digest"]} for item in items]
    )
    return {
        "schema_version": SNAPSHOT_SCHEMA_VERSION,
        "catalog": str(catalog.directory),
        "config": catalog.config.to_dict(),
        "ledger_digest": catalog.ledger.ledger_digest,
        "format_versions": versions,
        "content_digest": content_digest,
        "normalized_view_digest": normalized_digest,
        "upgrade_trace": upgrade_trace,
        "processed_cursor": int(processed_cursor),
        "items": items,
        "disk_facts": facts,
    }


@dataclass(frozen=True)
class Snapshot:
    id: int
    created_at: str
    catalog: str
    state: dict[str, Any]
    readonly: bool
    active: bool
    items: tuple[dict[str, Any], ...]

    @property
    def format_versions(self) -> list[int]:
        return list(self.state["format_versions"])

    @property
    def content_digest(self) -> str:
        return self.state["content_digest"]

    @property
    def normalized_view_digest(self) -> str:
        return self.state["normalized_view_digest"]


class SnapshotStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.connection = sqlite3.connect(str(self.path), timeout=30)
            self.connection.row_factory = sqlite3.Row
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA journal_mode = WAL")
            self._create_schema()
        except sqlite3.Error as exc:
            fail("E_IO", f"cannot open snapshot SQLite database: {exc}", manifest_path=str(self.path))

    def _create_schema(self) -> None:
        with self.connection:
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    catalog TEXT NOT NULL,
                    format_versions_json TEXT NOT NULL,
                    content_digest TEXT NOT NULL,
                    normalized_view_digest TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    active INTEGER NOT NULL,
                    readonly INTEGER NOT NULL
                )
                """
            )
            self.connection.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshot_items (
                    snapshot_id INTEGER NOT NULL,
                    sequence INTEGER NOT NULL,
                    manifest_path TEXT NOT NULL,
                    manifest_id TEXT NOT NULL,
                    source_version INTEGER NOT NULL,
                    target_version INTEGER NOT NULL,
                    normalized_json TEXT NOT NULL,
                    view_digest TEXT NOT NULL,
                    PRIMARY KEY (snapshot_id, sequence),
                    FOREIGN KEY (snapshot_id) REFERENCES snapshots(id) ON DELETE CASCADE
                )
                """
            )

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SnapshotStore":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def save(
        self,
        catalog: Catalog,
        *,
        upgrade_trace: list[dict[str, Any]] | None = None,
        processed_cursor: int | None = None,
    ) -> Snapshot:
        state = capture_state(
            catalog,
            upgrade_trace=upgrade_trace,
            processed_cursor=processed_cursor,
        )
        return self._insert_state(state, active=True, readonly=False)

    def _insert_state(self, state: dict[str, Any], *, active: bool, readonly: bool) -> Snapshot:
        items = state["items"]
        try:
            with self.connection:
                if active:
                    self.connection.execute("UPDATE snapshots SET active = 0 WHERE active = 1")
                cursor = self.connection.execute(
                    """
                    INSERT INTO snapshots (
                        created_at, catalog, format_versions_json, content_digest,
                        normalized_view_digest, state_json, active, readonly
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        _utc_now(),
                        state["catalog"],
                        canonical_json_text(state["format_versions"]),
                        state["content_digest"],
                        state["normalized_view_digest"],
                        canonical_json_text(state),
                        1 if active else 0,
                        1 if readonly else 0,
                    ),
                )
                snapshot_id = int(cursor.lastrowid)
                for sequence, item in enumerate(items):
                    self.connection.execute(
                        """
                        INSERT INTO snapshot_items (
                            snapshot_id, sequence, manifest_path, manifest_id,
                            source_version, target_version, normalized_json, view_digest
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            snapshot_id,
                            sequence,
                            item["manifest_path"],
                            item["manifest_id"],
                            item["source_version"],
                            item["target_version"],
                            item["normalized_json"],
                            item["view_digest"],
                        ),
                    )
        except sqlite3.Error as exc:
            fail("E_IO", f"cannot persist snapshot: {exc}", manifest_path=str(self.path))
        loaded = self.get(snapshot_id)
        assert loaded is not None
        return loaded

    def get(self, snapshot_id: int | None = None) -> Snapshot | None:
        try:
            if snapshot_id is None:
                row = self.connection.execute(
                    "SELECT * FROM snapshots WHERE active = 1 ORDER BY id DESC LIMIT 1"
                ).fetchone()
            else:
                row = self.connection.execute(
                    "SELECT * FROM snapshots WHERE id = ?", (snapshot_id,)
                ).fetchone()
            if row is None:
                return None
            item_rows = self.connection.execute(
                "SELECT * FROM snapshot_items WHERE snapshot_id = ? ORDER BY sequence",
                (row["id"],),
            ).fetchall()
        except sqlite3.Error as exc:
            fail("E_IO", f"cannot read snapshot: {exc}", manifest_path=str(self.path))
        state = json.loads(row["state_json"])
        items = tuple(
            {
                "manifest_path": item["manifest_path"],
                "manifest_id": item["manifest_id"],
                "source_version": item["source_version"],
                "target_version": item["target_version"],
                "normalized_json": item["normalized_json"],
                "view_digest": item["view_digest"],
            }
            for item in item_rows
        )
        return Snapshot(
            id=int(row["id"]),
            created_at=row["created_at"],
            catalog=row["catalog"],
            state=state,
            readonly=bool(row["readonly"]),
            active=bool(row["active"]),
            items=items,
        )

    def list(self) -> list[dict[str, Any]]:
        try:
            rows = self.connection.execute(
                "SELECT id, created_at, active, readonly, content_digest, normalized_view_digest FROM snapshots ORDER BY id"
            ).fetchall()
        except sqlite3.Error as exc:
            fail("E_IO", f"cannot list snapshots: {exc}", manifest_path=str(self.path))
        return [dict(row) for row in rows]

    def mark_readonly(self, snapshot_id: int) -> None:
        try:
            with self.connection:
                self.connection.execute(
                    "UPDATE snapshots SET readonly = 1, active = 0 WHERE id = ?", (snapshot_id,)
                )
        except sqlite3.Error as exc:
            fail("E_IO", f"cannot mark snapshot read-only: {exc}", manifest_path=str(self.path))

    def restore(
        self,
        catalog: Catalog,
        *,
        snapshot_id: int | None = None,
        upgrade_trace: list[dict[str, Any]] | None = None,
        verify_replay: bool = True,
    ) -> dict[str, Any]:
        snapshot = self.get(snapshot_id)
        if snapshot is None:
            fail("E_REPLAY", "no snapshot is available to restore", manifest_path=str(self.path))
        facts, versions, content_digest = _manifest_disk_facts(catalog)
        differences: list[str] = []
        if versions != snapshot.format_versions:
            differences.append("format_versions")
        if content_digest != snapshot.content_digest:
            differences.append("content_digest")

        if not differences:
            state = snapshot.state
            restore_report = {
                "ok": True,
                "operation": "snapshot_restore",
                "matched": True,
                "replaced": False,
                "snapshot_id": snapshot.id,
                "old_snapshot_readonly": snapshot.readonly,
                "differences": [],
                "format_versions": versions,
                "content_digest": content_digest,
                "normalized_view_digest": snapshot.normalized_view_digest,
                "processed_cursor": state.get("processed_cursor", 0),
            }
        else:
            current_state = capture_state(catalog, upgrade_trace=upgrade_trace)
            self.mark_readonly(snapshot.id)
            replacement = self._insert_state(current_state, active=True, readonly=False)
            state = current_state
            restore_report = {
                "ok": True,
                "operation": "snapshot_restore",
                "matched": False,
                "replaced": True,
                "snapshot_id": replacement.id,
                "replaced_snapshot_id": snapshot.id,
                "old_snapshot_readonly": True,
                "differences": differences,
                "format_versions": versions,
                "content_digest": content_digest,
                "normalized_view_digest": replacement.normalized_view_digest,
                "processed_cursor": current_state.get("processed_cursor", 0),
            }

        if verify_replay:
            active = self.get()
            assert active is not None
            replay_report = self.replay(catalog, snapshot_id=active.id)
            restore_report["replay"] = replay_report
        restore_report["session_state"] = {
            "catalog": state["catalog"],
            "config": state["config"],
            "format_versions": state["format_versions"],
            "content_digest": state["content_digest"],
            "normalized_view_digest": state["normalized_view_digest"],
            "upgrade_trace": state["upgrade_trace"],
            "processed_cursor": state["processed_cursor"],
        }
        restore_report["disk_facts_sha256"] = digest_json(facts)
        return restore_report

    def replay(self, catalog: Catalog, *, snapshot_id: int | None = None) -> dict[str, Any]:
        snapshot = self.get(snapshot_id)
        if snapshot is None:
            fail("E_REPLAY", "no snapshot is available to replay", manifest_path=str(self.path))
        facts, versions, content_digest = _manifest_disk_facts(catalog)
        if versions != snapshot.format_versions:
            fail(
                "E_REPLAY",
                "snapshot format version set does not match disk",
                path="format_versions",
                manifest_path=str(self.path),
                details={
                    "reason": "format_versions",
                    "expected": snapshot.format_versions,
                    "actual": versions,
                    "snapshot_id": snapshot.id,
                },
            )
        if content_digest != snapshot.content_digest:
            fail(
                "E_REPLAY",
                "snapshot content digest does not match disk",
                path="content_digest",
                manifest_path=str(self.path),
                details={
                    "reason": "content_digest",
                    "expected": snapshot.content_digest,
                    "actual": content_digest,
                    "snapshot_id": snapshot.id,
                },
            )
        try:
            embedded_config = load_config(snapshot.state["config"])
        except BaseException as exc:
            if getattr(exc, "code", None):
                fail(
                    "E_REPLAY",
                    f"snapshot configuration cannot be loaded: {exc}",
                    path="config",
                    manifest_path=str(self.path),
                    details={"reason": "config"},
                )
            raise
        embedded_ledger = build_ledger(embedded_config)
        if embedded_ledger.ledger_digest != snapshot.state.get("ledger_digest"):
            fail(
                "E_REPLAY",
                "snapshot ledger truth cannot be reproduced from its configuration",
                path="ledger_digest",
                manifest_path=str(self.path),
                details={
                    "reason": "ledger_digest",
                    "expected": snapshot.state.get("ledger_digest"),
                    "actual": embedded_ledger.ledger_digest,
                },
            )

        try:
            embedded_catalog = Catalog(catalog.directory, embedded_config, embedded_ledger)
            current_state = capture_state(embedded_catalog, upgrade_trace=[], processed_cursor=0)
        except BaseException as exc:
            if getattr(exc, "code", None):
                fail(
                    "E_REPLAY",
                    f"current disk state cannot be re-normalized: {exc}",
                    manifest_path=str(self.path),
                    details={"reason": "renormalize", "business_code": getattr(exc, "code", None)},
                )
            raise

        current_by_path = {item["manifest_path"]: item for item in current_state["items"]}
        snapshot_by_path = {item["manifest_path"]: item for item in snapshot.items}
        for path in sorted(set(current_by_path) | set(snapshot_by_path)):
            current_item = current_by_path.get(path)
            snapshot_item = snapshot_by_path.get(path)
            if current_item is None or snapshot_item is None:
                fail(
                    "E_REPLAY",
                    "snapshot item set does not match disk",
                    path="items",
                    manifest_path=path,
                    details={
                        "reason": "item_set",
                        "snapshot_has_item": snapshot_item is not None,
                        "disk_has_item": current_item is not None,
                    },
                )
            if current_item["view_digest"] != snapshot_item["view_digest"]:
                expected = json.loads(snapshot_item["normalized_json"])
                actual = json.loads(current_item["normalized_json"])
                mismatch = first_difference(expected, actual) or first_difference(actual, expected) or {}
                fail(
                    "E_REPLAY",
                    "re-normalized item does not reproduce the snapshot",
                    path=mismatch.get("field_path"),
                    manifest_path=path,
                    shard_index=mismatch.get("shard_index"),
                    details={
                        "reason": "normalized_item",
                        "snapshot_id": snapshot.id,
                        "expected_view_digest": snapshot_item["view_digest"],
                        "actual_view_digest": current_item["view_digest"],
                        "first_mismatch": mismatch,
                    },
                )

        snapshot_trace = snapshot.state.get("upgrade_trace", [])
        current_trace, current_cursor = read_checkpoint_trace(catalog)
        if snapshot_trace or current_trace:
            comparable_snapshot = [
                {
                    "manifest_path": item.get("manifest_path"),
                    "from_version": item.get("from_version"),
                    "to_version": item.get("to_version"),
                    "status": item.get("status"),
                }
                for item in snapshot_trace
            ]
            if comparable_snapshot != current_trace:
                fail(
                    "E_REPLAY",
                    "snapshot upgrade trace does not match the current checkpoint",
                    path="upgrade_trace",
                    manifest_path=str(self.path),
                    details={
                        "reason": "upgrade_trace",
                        "expected": comparable_snapshot,
                        "actual": current_trace,
                    },
                )
            if int(snapshot.state.get("processed_cursor", 0)) != int(current_cursor):
                fail(
                    "E_REPLAY",
                    "snapshot processed cursor does not match the current checkpoint",
                    path="processed_cursor",
                    manifest_path=str(self.path),
                    details={
                        "reason": "processed_cursor",
                        "expected": snapshot.state.get("processed_cursor", 0),
                        "actual": current_cursor,
                    },
                )

        return {
            "ok": True,
            "operation": "snapshot_replay",
            "reproduced": True,
            "snapshot_id": snapshot.id,
            "format_versions": versions,
            "content_digest": content_digest,
            "normalized_view_digest": snapshot.normalized_view_digest,
            "compared_item_count": len(snapshot.items),
            "processed_cursor": snapshot.state.get("processed_cursor", 0),
            "config_digest": embedded_ledger.config_digest,
        }
