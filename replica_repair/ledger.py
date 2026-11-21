"""Versioned SQLite ledger with whole-view replacement and recovery semantics."""

from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from .errors import StateError
from .manifest_repo import compute_manifest_diff, compute_state_digest
from .model import Manifest
from .planner import RepairBudget, RepairPlan, RepairTask, TaskResult
from .truth import TruthReport
from .util import canonical_json_dumps


LEDGER_FORMAT_VERSION = 1


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS snapshots (
    snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation INTEGER NOT NULL,
    group_id TEXT NOT NULL,
    manifest_version INTEGER NOT NULL,
    state_digest TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    phase TEXT NOT NULL,
    is_active INTEGER NOT NULL,
    is_stale INTEGER NOT NULL,
    parent_snapshot_id INTEGER,
    config_json TEXT NOT NULL,
    FOREIGN KEY(parent_snapshot_id) REFERENCES snapshots(snapshot_id)
);
CREATE TABLE IF NOT EXISTS distributions (
    snapshot_id INTEGER NOT NULL,
    block_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    size_bytes INTEGER,
    digest TEXT,
    expected_size_bytes INTEGER,
    expected_digest TEXT,
    priority INTEGER NOT NULL,
    status TEXT NOT NULL,
    manifest_mismatch INTEGER NOT NULL,
    PRIMARY KEY(snapshot_id, block_id, node_id),
    FOREIGN KEY(snapshot_id) REFERENCES snapshots(snapshot_id)
);
CREATE TABLE IF NOT EXISTS truth_results (
    snapshot_id INTEGER NOT NULL,
    block_id TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    truth_digest TEXT NOT NULL,
    truth_nodes_json TEXT NOT NULL,
    ranking_json TEXT NOT NULL,
    PRIMARY KEY(snapshot_id, block_id),
    FOREIGN KEY(snapshot_id) REFERENCES snapshots(snapshot_id)
);
CREATE TABLE IF NOT EXISTS mismatches (
    snapshot_id INTEGER NOT NULL,
    block_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    deviation_digest TEXT,
    status TEXT NOT NULL,
    PRIMARY KEY(snapshot_id, block_id, node_id),
    FOREIGN KEY(snapshot_id) REFERENCES snapshots(snapshot_id)
);
CREATE TABLE IF NOT EXISTS repair_tasks (
    snapshot_id INTEGER NOT NULL,
    task_id TEXT NOT NULL,
    wave INTEGER NOT NULL,
    block_id TEXT NOT NULL,
    source_node TEXT NOT NULL,
    target_node TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    repair_kb INTEGER NOT NULL,
    repair_weight INTEGER NOT NULL,
    source_priority INTEGER NOT NULL,
    target_priority INTEGER NOT NULL,
    cost_units INTEGER NOT NULL,
    status TEXT NOT NULL,
    error_code TEXT,
    completion_digest TEXT,
    PRIMARY KEY(snapshot_id, task_id),
    FOREIGN KEY(snapshot_id) REFERENCES snapshots(snapshot_id)
);
CREATE TABLE IF NOT EXISTS progress (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    cursor INTEGER NOT NULL,
    current_snapshot_id INTEGER,
    completed_tasks INTEGER NOT NULL,
    total_tasks INTEGER NOT NULL,
    status TEXT NOT NULL,
    FOREIGN KEY(current_snapshot_id) REFERENCES snapshots(snapshot_id)
);
CREATE TABLE IF NOT EXISTS stale_impacts (
    impact_id INTEGER PRIMARY KEY AUTOINCREMENT,
    generation INTEGER NOT NULL,
    previous_snapshot_id INTEGER,
    block_id TEXT NOT NULL,
    node_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    UNIQUE(generation, previous_snapshot_id, block_id, node_id, reason)
);
"""


@dataclass(frozen=True)
class RecoveryInfo:
    replaced: bool
    first_open: bool
    reason: str | None
    previous_content_digest: str | None
    current_content_digest: str
    generation: int
    unknown_tasks: int
    previous_snapshot_id: int | None
    affected_blocks: tuple[str, ...]
    affected_nodes: tuple[str, ...]
    details: tuple[dict[str, str], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "replaced": self.replaced,
            "first_open": self.first_open,
            "reason": self.reason,
            "previous_content_digest": self.previous_content_digest,
            "current_content_digest": self.current_content_digest,
            "generation": self.generation,
            "unknown_tasks": self.unknown_tasks,
            "previous_snapshot_id": self.previous_snapshot_id,
            "affected_blocks": list(self.affected_blocks),
            "affected_nodes": list(self.affected_nodes),
            "details": list(self.details),
        }


class Ledger:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.connection = sqlite3.connect(str(self.path), isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.connection.execute("PRAGMA synchronous = FULL")
        self._format_mismatch = self._initialize()

    def close(self) -> None:
        self.connection.close()

    def bind_source(self, root: Path | str, manifest_path: Path | str) -> None:
        """Remember the local source so report/reopen can verify current disk state."""

        with self._lock, self._transaction():
            self._meta_set("source_root", str(Path(root).resolve()))
            self._meta_set("manifest_path", str(Path(manifest_path).resolve()))

    def reopen_from_disk(self) -> RecoveryInfo | None:
        """Compare a bound local source to the ledger and recover if it changed."""

        root_value = self._meta_get("source_root")
        manifest_value = self._meta_get("manifest_path")
        if not root_value or not manifest_value:
            return None
        manifest_path = Path(manifest_value)
        if not manifest_path.exists():
            raise StateError(
                "ledger is bound to a missing manifest",
                path="$",
                details={"manifest_path": manifest_value},
            )
        manifest = Manifest.from_file(manifest_path)
        return self.open_for_state(manifest, compute_state_digest(root_value, manifest))

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _initialize(self) -> bool:
        existing_user_version = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        existed = self.connection.execute(
            "SELECT count(*) FROM sqlite_master WHERE type='table' AND name='meta'"
        ).fetchone()[0] > 0
        self.connection.executescript(SCHEMA)
        if not existed:
            self.connection.execute("PRAGMA user_version = 1")
            self.connection.execute(
                "INSERT INTO meta(key, value) VALUES('format_version', ?)",
                (str(LEDGER_FORMAT_VERSION),),
            )
            self.connection.execute(
                "INSERT INTO meta(key, value) VALUES('generation', '0')"
            )
            self.connection.execute(
                "INSERT INTO progress(id, cursor, current_snapshot_id, completed_tasks, total_tasks, status) "
                "VALUES(1, 0, NULL, 0, 0, 'idle')"
            )
            return False
        meta_version = self._meta_get("format_version")
        mismatch = existing_user_version not in (0, LEDGER_FORMAT_VERSION)
        if meta_version is not None and meta_version != str(LEDGER_FORMAT_VERSION):
            mismatch = True
        if meta_version is None:
            self._meta_set("format_version", str(LEDGER_FORMAT_VERSION))
        if self._meta_get("generation") is None:
            self._meta_set("generation", "0")
        return mismatch

    def open_for_state(self, manifest: Manifest, state_digest: str) -> RecoveryInfo:
        with self._lock, self._transaction():
            content_digest = self._meta_get("content_digest")
            generation = int(self._meta_get("generation") or "0")
            active = self.connection.execute(
                "SELECT * FROM snapshots WHERE is_active=1 AND is_stale=0 ORDER BY snapshot_id DESC LIMIT 1"
            ).fetchone()
            progress = self.connection.execute("SELECT * FROM progress WHERE id=1").fetchone()
            assert progress is not None
            state_mismatch = active is not None and active["state_digest"] != state_digest
            content_mismatch = content_digest is not None and content_digest != state_digest
            interrupted = progress["status"] in {"running", "recovering", "failed"}
            format_mismatch = self._format_mismatch

            if not (state_mismatch or content_mismatch or interrupted or format_mismatch):
                first_open = content_digest is None
                if first_open:
                    self._meta_set("content_digest", state_digest)
                return RecoveryInfo(
                    replaced=False,
                    first_open=first_open,
                    reason=None,
                    previous_content_digest=content_digest,
                    current_content_digest=state_digest,
                    generation=generation,
                    unknown_tasks=0,
                    previous_snapshot_id=active["snapshot_id"] if active else None,
                    affected_blocks=(),
                    affected_nodes=(),
                    details=(),
                )

            previous_snapshot = active
            if previous_snapshot is None:
                previous_snapshot = self.connection.execute(
                    "SELECT * FROM snapshots ORDER BY snapshot_id DESC LIMIT 1"
                ).fetchone()
            previous_snapshot_id = previous_snapshot["snapshot_id"] if previous_snapshot else None
            old_manifest: Manifest | None = None
            if previous_snapshot is not None:
                old_manifest = Manifest.from_dict(json.loads(previous_snapshot["manifest_json"]))

            reasons: list[str] = []
            if format_mismatch:
                reasons.append("ledger_format_version")
            if content_mismatch or state_mismatch:
                reasons.append("content_digest")
            if interrupted and not reasons:
                reasons.append("unfinished_run")
            reason = "+".join(reasons) if reasons else "recovery"

            unknown_count = 0
            unfinished_before_update = self.connection.execute(
                "SELECT snapshot_id, block_id, target_node, task_id FROM repair_tasks "
                "WHERE status IN ('planned', 'running', 'queued') ORDER BY task_id"
            ).fetchall()
            for row in unfinished_before_update:
                unknown_count += self.connection.execute(
                    "UPDATE repair_tasks SET status='unknown', error_code='E_RECOVERED_UNKNOWN' "
                    "WHERE snapshot_id=? AND task_id=?",
                    (row["snapshot_id"], row["task_id"]),
                ).rowcount

            self.connection.execute(
                "UPDATE snapshots SET is_active=0, is_stale=1 WHERE is_active=1"
            )
            self.connection.execute(
                "UPDATE progress SET current_snapshot_id=NULL, status='recovered' WHERE id=1"
            )

            if old_manifest is not None:
                changes = compute_manifest_diff(old_manifest, manifest)
                details = list(changes.details)
            else:
                details = []
            if content_mismatch or state_mismatch or format_mismatch:
                if not details:
                    previous_rows = []
                    if previous_snapshot_id is not None:
                        previous_rows = self.connection.execute(
                            "SELECT block_id, node_id FROM distributions WHERE snapshot_id=? "
                            "ORDER BY block_id, node_id",
                            (previous_snapshot_id,),
                        ).fetchall()
                    if previous_rows:
                        details = [
                            {"block_id": row["block_id"], "node_id": row["node_id"], "reason": "content_digest"}
                            for row in previous_rows
                        ]
                    else:
                        for node in sorted(manifest.nodes, key=lambda item: item.node_id):
                            for block in sorted(node.blocks, key=lambda item: item.block_id):
                                details.append(
                                    {
                                        "block_id": block.block_id,
                                        "node_id": node.node_id,
                                        "reason": "content_digest",
                                    }
                                )
            if interrupted:
                for row in unfinished_before_update:
                    details.append(
                        {
                            "block_id": row["block_id"],
                            "node_id": row["target_node"],
                            "reason": "unfinished_repair",
                        }
                    )
            unique_details = {
                (item["block_id"], item["node_id"], item["reason"]): item for item in details
            }
            details = [unique_details[key] for key in sorted(unique_details)]

            generation += 1
            for item in details:
                self.connection.execute(
                    "INSERT OR IGNORE INTO stale_impacts"
                    "(generation, previous_snapshot_id, block_id, node_id, reason) "
                    "VALUES(?, ?, ?, ?, ?)",
                    (
                        generation,
                        previous_snapshot_id,
                        item["block_id"],
                        item["node_id"],
                        item["reason"],
                    ),
                )
            self._meta_set("generation", str(generation))
            self._meta_set("format_version", str(LEDGER_FORMAT_VERSION))
            self.connection.execute("PRAGMA user_version = 1")
            self._meta_set("content_digest", state_digest)
            self._meta_set("recovery_reason", reason)
            self._format_mismatch = False
            return RecoveryInfo(
                replaced=True,
                first_open=False,
                reason=reason,
                previous_content_digest=content_digest,
                current_content_digest=state_digest,
                generation=generation,
                unknown_tasks=unknown_count,
                previous_snapshot_id=previous_snapshot_id,
                affected_blocks=tuple(sorted({item["block_id"] for item in details})),
                affected_nodes=tuple(sorted({item["node_id"] for item in details})),
                details=tuple(details),
            )

    def record_before(
        self,
        manifest: Manifest,
        state_digest: str,
        truth: TruthReport,
        plan: RepairPlan,
        budget: RepairBudget,
    ) -> int:
        with self._lock, self._transaction():
            generation = int(self._meta_get("generation") or "0")
            active = self.connection.execute(
                "SELECT * FROM snapshots WHERE is_active=1 AND is_stale=0 ORDER BY snapshot_id DESC LIMIT 1"
            ).fetchone()
            if active is not None:
                self.connection.execute(
                    "UPDATE snapshots SET is_active=0, is_stale=1 WHERE snapshot_id=?",
                    (active["snapshot_id"],),
                )
                old_manifest = Manifest.from_dict(json.loads(active["manifest_json"]))
                changes = compute_manifest_diff(old_manifest, manifest)
                for item in changes.details:
                    self.connection.execute(
                        "INSERT OR IGNORE INTO stale_impacts"
                        "(generation, previous_snapshot_id, block_id, node_id, reason) "
                        "VALUES(?, ?, ?, ?, ?)",
                        (
                            generation,
                            active["snapshot_id"],
                            item["block_id"],
                            item["node_id"],
                            item["reason"],
                        ),
                    )
            config = {"budget": budget.to_dict()}
            snapshot_id = self._insert_snapshot(
                generation=generation,
                manifest=manifest,
                state_digest=state_digest,
                phase="before",
                parent_snapshot_id=None,
                config=config,
                is_active=True,
            )
            self._insert_truth_rows(snapshot_id, truth)
            self._insert_tasks(snapshot_id, plan.tasks, status_override=None)
            self.connection.execute(
                "UPDATE progress SET cursor=0, current_snapshot_id=?, completed_tasks=0, "
                "total_tasks=?, status='running' WHERE id=1",
                (snapshot_id, len(plan.tasks)),
            )
            self._meta_set("content_digest", state_digest)
            self._meta_set("active_before_snapshot_id", str(snapshot_id))
            return snapshot_id

    def update_task_progress(self, result: TaskResult, completed_count: int) -> None:
        with self._lock, self._transaction():
            progress = self.connection.execute("SELECT * FROM progress WHERE id=1").fetchone()
            assert progress is not None
            snapshot_id = progress["current_snapshot_id"]
            if snapshot_id is None:
                raise StateError("ledger progress has no active snapshot")
            updated = self.connection.execute(
                "UPDATE repair_tasks SET status='completed', completion_digest=?, error_code=NULL "
                "WHERE snapshot_id=? AND task_id=?",
                (result.digest, snapshot_id, result.task_id),
            ).rowcount
            if updated != 1:
                raise StateError("repair task is absent from the ledger", details={"task_id": result.task_id})
            self.connection.execute(
                "UPDATE progress SET cursor=?, completed_tasks=?, status='running' WHERE id=1",
                (completed_count, completed_count),
            )

    def mark_task_failed(self, task_id: str, code: str, message: str) -> None:
        with self._lock, self._transaction():
            progress = self.connection.execute("SELECT * FROM progress WHERE id=1").fetchone()
            assert progress is not None
            snapshot_id = progress["current_snapshot_id"]
            if snapshot_id is None:
                return
            self.connection.execute(
                "UPDATE repair_tasks SET status='failed', error_code=? WHERE snapshot_id=? AND task_id=?",
                (code, snapshot_id, task_id),
            )
            self.connection.execute("UPDATE progress SET status='failed' WHERE id=1")

    def mark_run_failed(self, code: str) -> None:
        with self._lock, self._transaction():
            self.connection.execute("UPDATE progress SET status='failed' WHERE id=1")

    def record_after(
        self,
        manifest: Manifest,
        state_digest: str,
        truth: TruthReport,
        before_snapshot_id: int,
        budget: RepairBudget,
    ) -> int:
        with self._lock, self._transaction():
            generation = int(self._meta_get("generation") or "0")
            before = self.connection.execute(
                "SELECT * FROM snapshots WHERE snapshot_id=?",
                (before_snapshot_id,),
            ).fetchone()
            if before is None:
                raise StateError("before snapshot is absent", details={"snapshot_id": before_snapshot_id})
            self.connection.execute(
                "UPDATE snapshots SET is_active=0, is_stale=1 WHERE is_active=1"
            )
            after_id = self._insert_snapshot(
                generation=generation,
                manifest=manifest,
                state_digest=state_digest,
                phase="after",
                parent_snapshot_id=before_snapshot_id,
                config={"budget": budget.to_dict()},
                is_active=True,
            )
            self._insert_truth_rows(after_id, truth)
            self.connection.execute(
                "INSERT INTO repair_tasks(snapshot_id, task_id, wave, block_id, source_node, target_node, "
                "size_bytes, repair_kb, repair_weight, source_priority, target_priority, cost_units, "
                "status, error_code, completion_digest) "
                "SELECT ?, task_id, wave, block_id, source_node, target_node, size_bytes, repair_kb, "
                "repair_weight, source_priority, target_priority, cost_units, status, error_code, "
                "completion_digest FROM repair_tasks WHERE snapshot_id=? ORDER BY task_id",
                (after_id, before_snapshot_id),
            )
            task_count = self.connection.execute(
                "SELECT count(*) FROM repair_tasks WHERE snapshot_id=?", (after_id,)
            ).fetchone()[0]
            self.connection.execute(
                "UPDATE progress SET cursor=?, current_snapshot_id=?, completed_tasks=?, "
                "total_tasks=?, status='completed' WHERE id=1",
                (task_count, after_id, task_count, task_count),
            )
            self._meta_set("content_digest", state_digest)
            self._meta_set("active_after_snapshot_id", str(after_id))
            return after_id

    def current_report(self) -> dict[str, Any]:
        with self._lock:
            active = self.connection.execute(
                "SELECT * FROM snapshots WHERE is_active=1 AND is_stale=0 ORDER BY snapshot_id DESC LIMIT 1"
            ).fetchone()
            stale_count = self.connection.execute(
                "SELECT count(*) FROM snapshots WHERE is_stale=1"
            ).fetchone()[0]
            recovered = self.connection.execute(
                "SELECT value FROM meta WHERE key='recovery_reason'"
            ).fetchone()
            generation = int(self._meta_get("generation") or "0")
            impacts = self.connection.execute(
                "SELECT block_id, node_id, reason FROM stale_impacts WHERE generation=? "
                "ORDER BY block_id, node_id, reason",
                (generation,),
            ).fetchall()
            report: dict[str, Any] = {
                "ok": True,
                "ledger_format_version": LEDGER_FORMAT_VERSION,
                "generation": generation,
                "content_digest": self._meta_get("content_digest"),
                "recovery_reason": recovered["value"] if recovered else None,
                "current": None,
                "baseline": None,
                "stale": {
                    "snapshot_count": stale_count,
                    "affected_blocks": sorted({row["block_id"] for row in impacts}),
                    "affected_nodes": sorted({row["node_id"] for row in impacts}),
                    "details": [
                        {"block_id": row["block_id"], "node_id": row["node_id"], "reason": row["reason"]}
                        for row in impacts
                    ],
                },
                "progress": self._progress_dict(),
            }
            if active is not None:
                report["current"] = self._snapshot_data(active["snapshot_id"])
                if active["parent_snapshot_id"] is not None:
                    report["baseline"] = self._snapshot_data(active["parent_snapshot_id"])
            return report

    def snapshot_data(self, snapshot_id: int) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_data(snapshot_id)

    def active_snapshot_row(self) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM snapshots WHERE is_active=1 AND is_stale=0 ORDER BY snapshot_id DESC LIMIT 1"
            ).fetchone()
            return dict(row) if row is not None else None

    def snapshot_row(self, snapshot_id: int) -> dict[str, Any] | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM snapshots WHERE snapshot_id=?", (snapshot_id,)
            ).fetchone()
            return dict(row) if row is not None else None

    def _snapshot_data(self, snapshot_id: int) -> dict[str, Any]:
        snapshot = self.connection.execute(
            "SELECT * FROM snapshots WHERE snapshot_id=?", (snapshot_id,)
        ).fetchone()
        if snapshot is None:
            raise StateError("snapshot is absent", details={"snapshot_id": snapshot_id})
        distributions = [
            {
                "block_id": row["block_id"],
                "node_id": row["node_id"],
                "size_bytes": row["size_bytes"],
                "digest": row["digest"],
                "expected_size_bytes": row["expected_size_bytes"],
                "expected_digest": row["expected_digest"],
                "priority": row["priority"],
                "status": row["status"],
                "manifest_mismatch": bool(row["manifest_mismatch"]),
            }
            for row in self.connection.execute(
                "SELECT * FROM distributions WHERE snapshot_id=? ORDER BY block_id, node_id",
                (snapshot_id,),
            )
        ]
        truths = [
            {
                "block_id": row["block_id"],
                "size_bytes": row["size_bytes"],
                "truth_digest": row["truth_digest"],
                "truth_nodes": json.loads(row["truth_nodes_json"]),
                "ranking": json.loads(row["ranking_json"]),
            }
            for row in self.connection.execute(
                "SELECT * FROM truth_results WHERE snapshot_id=? ORDER BY block_id",
                (snapshot_id,),
            )
        ]
        mismatches = [
            {
                "block_id": row["block_id"],
                "node_id": row["node_id"],
                "deviation_digest": row["deviation_digest"],
                "status": row["status"],
            }
            for row in self.connection.execute(
                "SELECT * FROM mismatches WHERE snapshot_id=? ORDER BY block_id, node_id",
                (snapshot_id,),
            )
        ]
        tasks = [
            {
                "task_id": row["task_id"],
                "wave": row["wave"],
                "block_id": row["block_id"],
                "source_node": row["source_node"],
                "target_node": row["target_node"],
                "size_bytes": row["size_bytes"],
                "repair_kb": row["repair_kb"],
                "repair_weight": row["repair_weight"],
                "source_priority": row["source_priority"],
                "target_priority": row["target_priority"],
                "cost_units": row["cost_units"],
                "status": row["status"],
                "error_code": row["error_code"],
                "completion_digest": row["completion_digest"],
            }
            for row in self.connection.execute(
                "SELECT * FROM repair_tasks WHERE snapshot_id=? ORDER BY task_id",
                (snapshot_id,),
            )
        ]
        by_node: dict[str, int] = {}
        for task in tasks:
            by_node[task["target_node"]] = by_node.get(task["target_node"], 0) + task["repair_kb"]
        return {
            "snapshot_id": snapshot["snapshot_id"],
            "parent_snapshot_id": snapshot["parent_snapshot_id"],
            "generation": snapshot["generation"],
            "group_id": snapshot["group_id"],
            "manifest_version": snapshot["manifest_version"],
            "state_digest": snapshot["state_digest"],
            "phase": snapshot["phase"],
            "is_stale": bool(snapshot["is_stale"]),
            "config": json.loads(snapshot["config_json"]),
            "distributions": distributions,
            "truth": truths,
            "mismatches": mismatches,
            "tasks": tasks,
            "repair_stats": {
                "task_count": len(tasks),
                "completed_tasks": sum(1 for task in tasks if task["status"] == "completed"),
                "unknown_tasks": sum(1 for task in tasks if task["status"] == "unknown"),
                "total_repair_kb": sum(task["repair_kb"] for task in tasks if task["status"] == "completed"),
                "planned_repair_kb": sum(task["repair_kb"] for task in tasks),
                "total_cost_units": sum(
                    task["cost_units"] for task in tasks if task["status"] == "completed"
                ),
                "per_node_repair_kb": dict(sorted(by_node.items())),
            },
        }

    def _insert_snapshot(
        self,
        *,
        generation: int,
        manifest: Manifest,
        state_digest: str,
        phase: str,
        parent_snapshot_id: int | None,
        config: dict[str, Any],
        is_active: bool,
    ) -> int:
        cursor = self.connection.execute(
            "INSERT INTO snapshots(generation, group_id, manifest_version, state_digest, "
            "manifest_json, phase, is_active, is_stale, parent_snapshot_id, config_json) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, 0, ?, ?)",
            (
                generation,
                manifest.group_id,
                manifest.manifest_version,
                state_digest,
                canonical_json_dumps(manifest.to_dict()),
                phase,
                1 if is_active else 0,
                parent_snapshot_id,
                canonical_json_dumps(config),
            ),
        )
        return int(cursor.lastrowid)

    def _insert_truth_rows(self, snapshot_id: int, truth: TruthReport) -> None:
        for block in truth.blocks:
            self.connection.execute(
                "INSERT INTO truth_results(snapshot_id, block_id, size_bytes, truth_digest, "
                "truth_nodes_json, ranking_json) VALUES(?, ?, ?, ?, ?, ?)",
                (
                    snapshot_id,
                    block.block_id,
                    block.size_bytes,
                    block.truth_digest,
                    canonical_json_dumps(list(block.truth_nodes)),
                    canonical_json_dumps([candidate.to_dict() for candidate in block.ranking]),
                ),
            )
            for copy in block.copies:
                self.connection.execute(
                    "INSERT INTO distributions(snapshot_id, block_id, node_id, size_bytes, digest, "
                    "expected_size_bytes, expected_digest, priority, status, manifest_mismatch) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        snapshot_id,
                        copy.block_id,
                        copy.node_id,
                        copy.size_bytes,
                        copy.digest,
                        copy.expected_size_bytes,
                        copy.expected_digest,
                        copy.priority,
                        copy.status,
                        1 if copy.manifest_mismatch else 0,
                    ),
                )
            for mismatch in block.mismatches:
                self.connection.execute(
                    "INSERT INTO mismatches(snapshot_id, block_id, node_id, deviation_digest, status) "
                    "VALUES(?, ?, ?, ?, ?)",
                    (
                        snapshot_id,
                        mismatch.block_id,
                        mismatch.node_id,
                        mismatch.digest,
                        mismatch.status,
                    ),
                )

    def _insert_tasks(
        self,
        snapshot_id: int,
        tasks: Sequence[RepairTask],
        *,
        status_override: str | None,
    ) -> None:
        for task in tasks:
            self.connection.execute(
                "INSERT INTO repair_tasks(snapshot_id, task_id, wave, block_id, source_node, "
                "target_node, size_bytes, repair_kb, repair_weight, source_priority, target_priority, "
                "cost_units, status, error_code, completion_digest) "
                "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    snapshot_id,
                    task.task_id,
                    task.wave,
                    task.block_id,
                    task.source_node,
                    task.target_node,
                    task.size_bytes,
                    task.repair_kb,
                    task.repair_weight,
                    task.source_priority,
                    task.target_priority,
                    task.cost_units,
                    status_override or task.status,
                    task.error_code,
                    None,
                ),
            )

    def _progress_dict(self) -> dict[str, Any]:
        row = self.connection.execute("SELECT * FROM progress WHERE id=1").fetchone()
        assert row is not None
        return {
            "cursor": row["cursor"],
            "current_snapshot_id": row["current_snapshot_id"],
            "completed_tasks": row["completed_tasks"],
            "total_tasks": row["total_tasks"],
            "status": row["status"],
        }

    def _meta_get(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return str(row[0]) if row is not None else None

    def _meta_set(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")
