"""SQLite decision ledger with atomic round commits and snapshot recovery."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes, canonical_text
from .errors import ReplayError, SnapshotError

LEDGER_FORMAT_VERSION = 1
SNAPSHOT_FORMAT_VERSION = 1


def scene_digest(scene) -> str:
    import hashlib

    payload = {
        "scene_id": scene.scene_id,
        "scene_version": scene.scene_version,
        "services": {
            service_id: {
                "strategy": config.strategy,
                "cooldown_ms": config.cooldown_ms,
                "min_healthy_replicas": config.min_healthy_replicas,
                "success_threshold": config.success_threshold,
                "failure_threshold": config.failure_threshold,
            }
            for service_id, config in sorted(scene.services.items())
        },
        "instances": {
            instance_id: {
                "service_id": instance.service_id,
                "role": instance.role,
                "version": instance.version,
            }
            for instance_id, instance in sorted(scene.instances.items())
        },
        "observations": [
            {
                "service_id": item.service_id,
                "instance_id": item.instance_id,
                "round_index": item.round_index,
                "observed_at_ms": item.observed_at_ms,
                "result": item.result,
            }
            for item in sorted(
                scene.observations,
                key=lambda item: (
                    item.instance_id,
                    item.round_index,
                    item.observed_at_ms,
                ),
            )
        ],
    }
    return hashlib.sha256(canonical_bytes(payload)).hexdigest()


def conclusion_digest(conclusion: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(canonical_bytes(conclusion)).hexdigest()


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS rounds (
    round_index INTEGER PRIMARY KEY,
    path_name TEXT NOT NULL,
    conclusion_json TEXT NOT NULL,
    conclusion_digest TEXT NOT NULL,
    business_codes_json TEXT NOT NULL,
    switch_targets_json TEXT NOT NULL,
    state_digest TEXT NOT NULL,
    available_digest TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS transactions (
    round_index INTEGER NOT NULL,
    service_id TEXT NOT NULL,
    txn_json TEXT NOT NULL,
    PRIMARY KEY (round_index, service_id)
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    level TEXT NOT NULL,
    code TEXT NOT NULL,
    event_json TEXT NOT NULL
);
"""


@dataclass
class RecoveryState:
    recovered_rounds: list[int]
    last_committed_round: int | None
    pending_rounds: list[int]
    replaced: bool
    replace_reason: str | None
    warnings: list[str]


class Ledger:
    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self._seq = 0

    def close(self) -> None:
        self.conn.close()

    def initialize_session(
        self,
        scene,
        path_name: str,
        *,
        replace: bool = False,
        replace_reason: str | None = None,
    ) -> RecoveryState:
        digest = scene_digest(scene)
        existing = self.get_meta()
        warnings: list[str] = []
        replaced = False
        has_existing = bool(existing)
        mismatch = has_existing and (
            replace
            or existing.get("scene_digest") != digest
            or existing.get("scene_version") != str(scene.scene_version)
        )
        if mismatch:
            reason = replace_reason or "scene digest mismatch; wholesale replacement"
            with self.conn:
                self.conn.execute("DELETE FROM rounds")
                self.conn.execute("DELETE FROM transactions")
                self.conn.execute("DELETE FROM events")
                self._write_meta(scene, path_name, digest)
                self.log_event("WARN", "E_SNAPSHOT", {"reason": reason})
            warnings.append(reason)
            replaced = True
        elif not has_existing:
            with self.conn:
                self._write_meta(scene, path_name, digest)
        pending = [
            row[0]
            for row in self.conn.execute(
                "SELECT round_index FROM rounds WHERE status != 'committed' ORDER BY round_index"
            )
        ]
        if pending:
            with self.conn:
                self.conn.execute(
                    "DELETE FROM rounds WHERE status != 'committed'"
                )
                self.conn.execute(
                    "DELETE FROM transactions WHERE round_index IN (%s)"
                    % ",".join("?" for _ in pending),
                    pending,
                )
                self.log_event(
                    "WARN",
                    "E_SNAPSHOT",
                    {"reason": "uncommitted round marked unknown and removed", "rounds": pending},
                )
            warnings.append(f"uncommitted rounds removed for rerun: {pending}")
        committed = [
            row[0]
            for row in self.conn.execute(
                "SELECT round_index FROM rounds WHERE status = 'committed' ORDER BY round_index"
            )
        ]
        return RecoveryState(
            recovered_rounds=committed,
            last_committed_round=committed[-1] if committed else None,
            pending_rounds=pending,
            replaced=replaced,
            replace_reason=replace_reason,
            warnings=warnings,
        )

    def _write_meta(self, scene, path_name: str, digest: str) -> None:
        meta = {
            "format_version": str(LEDGER_FORMAT_VERSION),
            "snapshot_format_version": str(SNAPSHOT_FORMAT_VERSION),
            "path_name": path_name,
            "scene_id": scene.scene_id,
            "scene_version": str(scene.scene_version),
            "scene_digest": digest,
        }
        self.conn.executemany(
            "INSERT OR REPLACE INTO meta(key, value) VALUES (?, ?)",
            sorted(meta.items()),
        )

    def get_meta(self) -> dict[str, str]:
        return {row[0]: row[1] for row in self.conn.execute("SELECT key, value FROM meta")}

    def log_event(self, level: str, code: str, payload: dict[str, Any]) -> None:
        self.conn.execute(
            "INSERT INTO events(level, code, event_json) VALUES (?, ?, ?)",
            (level, code, canonical_text(payload)),
        )

    def begin_round(self, round_index: int, path_name: str) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO rounds"
                "(round_index, path_name, conclusion_json, conclusion_digest, "
                "business_codes_json, switch_targets_json, state_digest, "
                "available_digest, status, created_at_seq) VALUES (?, ?, '', '', '', '', '', '', 'pending', ?)",
                (round_index, path_name, self._next_seq()),
            )

    def commit_round(
        self,
        round_index: int,
        path_name: str,
        conclusion: dict[str, Any],
    ) -> str:
        transactions = [
            decision["switch"]
            for decision in conclusion["service_decisions"].values()
            if decision["switch"] is not None
        ]
        digest = conclusion_digest(conclusion)
        state_digest = conclusion_digest(conclusion["instance_states"])
        available_digest = conclusion_digest(conclusion["available_instances"])
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO rounds"
                "(round_index, path_name, conclusion_json, conclusion_digest, "
                "business_codes_json, switch_targets_json, state_digest, "
                "available_digest, status, created_at_seq) VALUES "
                "(?, ?, ?, ?, ?, ?, ?, ?, 'committed', ?)",
                (
                    round_index,
                    path_name,
                    canonical_text(conclusion),
                    digest,
                    canonical_text(conclusion["business_codes"]),
                    canonical_text(conclusion["switch_targets"]),
                    state_digest,
                    available_digest,
                    self._next_seq(),
                ),
            )
            self.conn.executemany(
                "INSERT OR REPLACE INTO transactions(round_index, service_id, txn_json) "
                "VALUES (?, ?, ?)",
                [
                    (
                        item["round_index"],
                        item["service_id"],
                        canonical_text(item),
                    )
                    for item in transactions
                ],
            )
        return digest

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    def load_round(self, round_index: int) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT conclusion_json FROM rounds WHERE round_index = ? AND status = 'committed'",
            (round_index,),
        ).fetchone()
        if row is None:
            raise ReplayError(
                f"no committed ledger record for round {round_index}",
                f"rounds[{round_index}]",
            )
        return json.loads(row[0])

    def committed_rounds(self) -> list[int]:
        return [
            row[0]
            for row in self.conn.execute(
                "SELECT round_index FROM rounds WHERE status = 'committed' ORDER BY round_index"
            )
        ]

    def verify_digest(self, round_index: int, conclusion: dict[str, Any]) -> None:
        row = self.conn.execute(
            "SELECT conclusion_digest FROM rounds WHERE round_index = ? AND status = 'committed'",
            (round_index,),
        ).fetchone()
        if row is None:
            raise ReplayError(
                f"missing committed record for round {round_index}",
                f"rounds[{round_index}]",
            )
        actual = conclusion_digest(conclusion)
        if actual != row[0]:
            raise ReplayError(
                f"replay digest mismatch at round {round_index}",
                f"rounds[{round_index}]",
                {"expected": row[0], "actual": actual},
            )


def write_snapshot(snapshot_dir: str | Path, scene, ledger: Ledger) -> Path:
    snapshot_path = Path(snapshot_dir)
    snapshot_path.mkdir(parents=True, exist_ok=True)
    meta = ledger.get_meta()
    if not meta:
        raise SnapshotError("cannot snapshot an uninitialized ledger", "meta")
    payload = {
        "snapshot_format_version": SNAPSHOT_FORMAT_VERSION,
        "ledger_format_version": LEDGER_FORMAT_VERSION,
        "scene_id": scene.scene_id,
        "scene_version": scene.scene_version,
        "scene_digest": scene_digest(scene),
        "committed_rounds": ledger.committed_rounds(),
        "meta": meta,
    }
    target = snapshot_path / f"{scene.scene_id}.snapshot.json"
    target.write_text(canonical_text(payload), encoding="utf-8")
    return target


def read_snapshot(snapshot_dir: str | Path, scene_id: str) -> dict[str, Any]:
    target = Path(snapshot_dir) / f"{scene_id}.snapshot.json"
    if not target.exists():
        raise SnapshotError("snapshot file does not exist", str(target))
    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SnapshotError(f"invalid snapshot JSON: {exc.msg}", str(target)) from exc
    if "snapshot_format_version" not in payload or "ledger_format_version" not in payload:
        raise SnapshotError("snapshot version fields are missing", str(target))
    if not isinstance(payload.get("snapshot_format_version"), int) or not isinstance(
        payload.get("ledger_format_version"), int
    ):
        raise SnapshotError("snapshot version fields must be integers", str(target))
    return payload
