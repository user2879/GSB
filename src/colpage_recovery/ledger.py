from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

from .engine import BatchReplay
from .models import Bundle


SCHEMA = """
PRAGMA journal_mode=DELETE;
CREATE TABLE IF NOT EXISTS meta (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS spec_generation (
  generation INTEGER PRIMARY KEY,
  spec_hash TEXT NOT NULL,
  pages_json TEXT NOT NULL,
  budget_json TEXT NOT NULL,
  created_unix_ms INTEGER NOT NULL,
  stale INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS wal_entry (
  generation INTEGER NOT NULL,
  lsn INTEGER NOT NULL,
  seq_index INTEGER NOT NULL,
  tx_id TEXT NOT NULL,
  page_id TEXT NOT NULL,
  row_start INTEGER NOT NULL,
  row_end INTEGER NOT NULL,
  seed INTEGER NOT NULL,
  commit_flag INTEGER NOT NULL,
  persist_flag INTEGER NOT NULL,
  entry_json TEXT NOT NULL,
  PRIMARY KEY(generation, lsn)
);
CREATE TABLE IF NOT EXISTS page_version (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  generation INTEGER NOT NULL,
  page_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  lsn INTEGER NOT NULL,
  tx_id TEXT NOT NULL,
  visible INTEGER NOT NULL,
  after_crash INTEGER NOT NULL,
  content_path TEXT NOT NULL,
  byte_length INTEGER NOT NULL,
  content_digest TEXT NOT NULL,
  version_digest TEXT NOT NULL,
  patch_digest TEXT NOT NULL,
  batch_id INTEGER,
  round_no INTEGER,
  attempt_no INTEGER,
  stale INTEGER NOT NULL DEFAULT 0,
  UNIQUE(generation, page_id, version)
);
CREATE INDEX IF NOT EXISTS idx_page_version_lsn ON page_version(generation, lsn);
CREATE TABLE IF NOT EXISTS digest_registry (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  generation INTEGER NOT NULL,
  kind TEXT NOT NULL,
  page_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  lsn INTEGER NOT NULL,
  digest TEXT NOT NULL,
  created_unix_ms INTEGER NOT NULL,
  UNIQUE(generation, kind, page_id, version, lsn)
);
CREATE TABLE IF NOT EXISTS replay_batch (
  batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
  generation INTEGER NOT NULL,
  round_no INTEGER NOT NULL,
  attempt_no INTEGER NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('running','committed','unknown')),
  start_lsn INTEGER NOT NULL,
  end_lsn INTEGER NOT NULL,
  entry_count INTEGER NOT NULL,
  global_slots INTEGER NOT NULL,
  per_tx_quota INTEGER NOT NULL,
  page_byte_budget INTEGER NOT NULL,
  checkpoint_interval INTEGER NOT NULL,
  ledger_batch_entries INTEGER NOT NULL,
  slot_peak INTEGER,
  byte_consumed INTEGER,
  wait_total_ms REAL,
  run_total_ms REAL,
  unknown_reason TEXT,
  stale INTEGER NOT NULL DEFAULT 0,
  started_unix_ms INTEGER NOT NULL,
  committed_unix_ms INTEGER,
  UNIQUE(generation, round_no, attempt_no)
);
CREATE TABLE IF NOT EXISTS task_outcome (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  batch_id INTEGER NOT NULL REFERENCES replay_batch(batch_id),
  generation INTEGER NOT NULL,
  lsn INTEGER NOT NULL,
  tx_id TEXT NOT NULL,
  page_id TEXT NOT NULL,
  thread_name TEXT,
  wait_started_ms REAL,
  wait_ended_ms REAL,
  worker_started_ms REAL,
  worker_ended_ms REAL,
  wait_ms REAL,
  run_ms REAL,
  global_active_at_start INTEGER,
  tx_active_at_start INTEGER,
  byte_count INTEGER,
  patch_digest TEXT,
  status TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_task_batch ON task_outcome(batch_id);
CREATE TABLE IF NOT EXISTS budget_usage (
  generation INTEGER NOT NULL,
  page_id TEXT NOT NULL,
  bytes_consumed INTEGER NOT NULL,
  byte_budget INTEGER NOT NULL,
  last_batch_id INTEGER,
  updated_unix_ms INTEGER NOT NULL,
  PRIMARY KEY(generation, page_id)
);
CREATE TABLE IF NOT EXISTS progress_cursor (
  generation INTEGER PRIMARY KEY,
  last_committed_lsn INTEGER NOT NULL,
  last_committed_round INTEGER NOT NULL,
  last_batch_id INTEGER,
  current_revision INTEGER NOT NULL,
  updated_unix_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS checkpoint (
  checkpoint_id INTEGER PRIMARY KEY AUTOINCREMENT,
  generation INTEGER NOT NULL,
  checkpoint_no INTEGER NOT NULL,
  at_lsn INTEGER NOT NULL,
  batch_id INTEGER,
  aggregate_digest TEXT NOT NULL,
  stale INTEGER NOT NULL DEFAULT 0,
  created_unix_ms INTEGER NOT NULL,
  UNIQUE(generation, at_lsn)
);
CREATE TABLE IF NOT EXISTS checkpoint_page (
  checkpoint_id INTEGER NOT NULL REFERENCES checkpoint(checkpoint_id),
  generation INTEGER NOT NULL,
  page_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  lsn INTEGER NOT NULL,
  content_digest TEXT NOT NULL,
  version_digest TEXT NOT NULL,
  content_path TEXT NOT NULL,
  PRIMARY KEY(checkpoint_id, page_id)
);
CREATE TABLE IF NOT EXISTS ledger_revision (
  revision_id INTEGER PRIMARY KEY AUTOINCREMENT,
  generation INTEGER NOT NULL,
  revision_no INTEGER NOT NULL,
  batch_id INTEGER NOT NULL,
  round_no INTEGER NOT NULL,
  attempt_no INTEGER NOT NULL,
  cursor_lsn INTEGER NOT NULL,
  aggregate_digest TEXT NOT NULL,
  created_unix_ms INTEGER NOT NULL,
  UNIQUE(generation, revision_no)
);
CREATE TABLE IF NOT EXISTS snapshot (
  snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
  generation INTEGER NOT NULL,
  kind TEXT NOT NULL,
  label TEXT NOT NULL,
  at_lsn INTEGER NOT NULL,
  digest TEXT NOT NULL,
  content_path TEXT,
  stale INTEGER NOT NULL DEFAULT 0,
  created_unix_ms INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS view_snapshot (
  view_snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
  generation INTEGER NOT NULL,
  revision_no INTEGER NOT NULL,
  cursor_lsn INTEGER NOT NULL,
  status TEXT NOT NULL CHECK(status IN ('current','stale')),
  created_unix_ms INTEGER NOT NULL,
  summary_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS page_view (
  page_view_id INTEGER PRIMARY KEY AUTOINCREMENT,
  view_snapshot_id INTEGER NOT NULL REFERENCES view_snapshot(view_snapshot_id),
  generation INTEGER NOT NULL,
  page_id TEXT NOT NULL,
  version INTEGER NOT NULL,
  lsn INTEGER NOT NULL,
  visible INTEGER NOT NULL,
  after_crash INTEGER NOT NULL,
  content_digest TEXT NOT NULL,
  version_digest TEXT NOT NULL,
  byte_length INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_page_view_snapshot ON page_view(view_snapshot_id);
CREATE TABLE IF NOT EXISTS crash_decision (
  generation INTEGER PRIMARY KEY,
  crash_after_lsn INTEGER NOT NULL,
  criterion TEXT NOT NULL,
  decision_json TEXT NOT NULL,
  updated_unix_ms INTEGER NOT NULL
);
"""


def unix_ms() -> int:
    return time.time_ns() // 1_000_000


def canonical_spec_hash(bundle: Bundle) -> str:
    spec = {
        "pages": [
            {
                "page_id": page.page_id,
                "row_count": page.row_count,
                "columns": [asdict(column) for column in page.columns],
                "null_bitmap": list(page.null_bitmap),
            }
            for page in bundle.pages
        ],
        "budget": asdict(bundle.budget),
    }
    encoded = json.dumps(spec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class Ledger:
    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys=ON")

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if exc_type is not None:
            self.conn.rollback()
        self.close()

    def migrate(self) -> None:
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def set_meta(self, key: str, value: Any) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )
        self.conn.commit()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else row["value"]

    def current_generation(self) -> int:
        value = self.get_meta("current_generation")
        if value is None:
            raise RuntimeError("ledger generation has not been initialized")
        return int(value)

    def start_generation(self, bundle: Bundle) -> tuple[int, bool]:
        spec_hash = canonical_spec_hash(bundle)
        current_value = self.get_meta("current_generation")
        if current_value is not None:
            current = int(current_value)
            current_row = self.conn.execute(
                "SELECT spec_hash FROM spec_generation WHERE generation=?",
                (current,),
            ).fetchone()
            if current_row is not None and current_row["spec_hash"] == spec_hash:
                self.conn.execute(
                    "UPDATE spec_generation SET stale=0 WHERE generation=?",
                    (current,),
                )
                self.conn.commit()
                return current, False

        latest_row = self.conn.execute(
            "SELECT COALESCE(MAX(generation),0) AS generation FROM spec_generation"
        ).fetchone()
        generation = int(latest_row["generation"]) + 1
        self.conn.execute("UPDATE spec_generation SET stale=1 WHERE generation<>?", (generation,))
        for table in ("checkpoint", "replay_batch", "snapshot"):
            self.conn.execute(
                f"UPDATE {table} SET stale=1 WHERE generation<>?", (generation,)
            )
        self.conn.execute("UPDATE view_snapshot SET status='stale' WHERE generation<>?", (generation,))
        self.conn.execute(
            "INSERT INTO spec_generation("
            "generation,spec_hash,pages_json,budget_json,created_unix_ms,stale"
            ") VALUES(?,?,?,?,?,0)",
            (
                generation,
                spec_hash,
                json.dumps(bundle.pages_payload, ensure_ascii=False, sort_keys=True),
                json.dumps(asdict(bundle.budget), ensure_ascii=False, sort_keys=True),
                unix_ms(),
            ),
        )
        self.conn.execute(
            "INSERT INTO progress_cursor("
            "generation,last_committed_lsn,last_committed_round,last_batch_id,"
            "current_revision,updated_unix_ms) VALUES(?,0,0,NULL,0,?)",
            (generation, unix_ms()),
        )
        self.set_meta("current_generation", generation)
        self.conn.commit()
        return generation, True

    def store_wal_entries(self, generation: int, bundle: Bundle) -> None:
        rows = []
        for index, entry in enumerate(sorted(bundle.entries, key=lambda item: item.lsn)):
            rows.append(
                (
                    generation,
                    entry.lsn,
                    index,
                    entry.tx_id,
                    entry.page_id,
                    entry.row_start,
                    entry.row_end,
                    entry.seed,
                    int(entry.commit_flag),
                    int(entry.persist_flag),
                    json.dumps(asdict(entry), sort_keys=True),
                )
            )
        self.conn.executemany(
            "INSERT OR REPLACE INTO wal_entry("
            "generation,lsn,seq_index,tx_id,page_id,row_start,row_end,seed,"
            "commit_flag,persist_flag,entry_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            rows,
        )
        self.conn.commit()

    def insert_baseline_version(
        self,
        generation: int,
        page_id: str,
        content_path: str,
        byte_length: int,
        digest: str,
        vdigest: str,
    ) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO page_version("
            "generation,page_id,version,lsn,tx_id,visible,after_crash,content_path,"
            "byte_length,content_digest,version_digest,patch_digest,round_no,attempt_no"
            ") VALUES(?,?,0,0,'',1,0,?,?,?,?,'baseline',0,0)",
            (generation, page_id, content_path, byte_length, digest, vdigest),
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO digest_registry("
            "generation,kind,page_id,version,lsn,digest,created_unix_ms"
            ") VALUES(?, 'page_content', ?,0,0,?,?)",
            (generation, page_id, digest, unix_ms()),
        )
        self.conn.commit()

    def next_attempt_no(self, generation: int, round_no: int) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(attempt_no),0)+1 AS attempt_no "
            "FROM replay_batch WHERE generation=? AND round_no=?",
            (generation, round_no),
        ).fetchone()
        return int(row["attempt_no"])

    def begin_batch(
        self,
        generation: int,
        round_no: int,
        attempt_no: int,
        batch: BatchReplay,
        bundle: Bundle,
    ) -> int:
        first = batch.entries[0]
        last = batch.entries[-1]
        cursor = self.get_cursor(generation)
        row = self.conn.execute(
            "INSERT INTO replay_batch("
            "generation,round_no,attempt_no,status,start_lsn,end_lsn,entry_count,"
            "global_slots,per_tx_quota,page_byte_budget,checkpoint_interval,"
            "ledger_batch_entries,started_unix_ms) "
            "VALUES(?,?,?,'running',?,?,?,?,?,?,?,?,?)",
            (
                generation,
                round_no,
                attempt_no,
                first.lsn,
                last.lsn,
                len(batch.entries),
                bundle.budget.global_slots,
                bundle.budget.per_tx_quota,
                bundle.budget.page_byte_budget,
                bundle.budget.checkpoint_interval,
                bundle.budget.ledger_batch_entries,
                unix_ms(),
            ),
        )
        batch_id = int(row.lastrowid)
        self.conn.commit()
        assert cursor is not None
        return batch_id

    def get_cursor(self, generation: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM progress_cursor WHERE generation=?", (generation,)
        ).fetchone()
        return None if row is None else dict(row)

    def complete_batch(
        self,
        *,
        generation: int,
        batch_id: int,
        round_no: int,
        attempt_no: int,
        batch: BatchReplay,
        content_paths: dict[int, str],
        before_commit: Callable[[], None] | None = None,
    ) -> int:
        wait_total = sum(float(item.get("wait_ms") or 0.0) for item in batch.timeline)
        run_total = sum(float(item.get("run_ms") or 0.0) for item in batch.timeline)
        self.conn.execute(
            "UPDATE replay_batch SET status='committed',slot_peak=?,byte_consumed=?,"
            "wait_total_ms=?,run_total_ms=?,committed_unix_ms=? WHERE batch_id=?",
            (
                batch.slot_peak,
                batch.byte_consumed,
                wait_total,
                run_total,
                unix_ms(),
                batch_id,
            ),
        )
        for event in batch.timeline:
            self.conn.execute(
                "INSERT INTO task_outcome("
                "batch_id,generation,lsn,tx_id,page_id,thread_name,wait_started_ms,"
                "wait_ended_ms,worker_started_ms,worker_ended_ms,wait_ms,run_ms,"
                "global_active_at_start,tx_active_at_start,byte_count,patch_digest,status"
                ") VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    batch_id,
                    generation,
                    event["lsn"],
                    event["tx_id"],
                    event["page_id"],
                    event.get("thread_name"),
                    event.get("wait_started_ms"),
                    event.get("wait_ended_ms"),
                    event.get("worker_started_ms"),
                    event.get("worker_ended_ms"),
                    event.get("wait_ms"),
                    event.get("run_ms"),
                    event.get("global_active_at_start"),
                    event.get("tx_active_at_start"),
                    event.get("byte_count"),
                    event.get("patch_digest"),
                    event.get("status", "committed"),
                ),
            )

        for record in batch.records:
            content_path = content_paths[record.lsn]
            self.conn.execute(
                "INSERT INTO page_version("
                "generation,page_id,version,lsn,tx_id,visible,after_crash,content_path,"
                "byte_length,content_digest,version_digest,patch_digest,batch_id,"
                "round_no,attempt_no) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    generation,
                    record.page_id,
                    record.version,
                    record.lsn,
                    record.tx_id,
                    int(record.visible),
                    int(record.after_crash),
                    content_path,
                    record.byte_length,
                    record.content_digest,
                    record.version_digest,
                    record.patch_digest,
                    batch_id,
                    round_no,
                    attempt_no,
                ),
            )
            self.conn.execute(
                "INSERT INTO digest_registry("
                "generation,kind,page_id,version,lsn,digest,created_unix_ms"
                ") VALUES(?, 'page_content', ?,?,?,?,?) "
                "ON CONFLICT(generation,kind,page_id,version,lsn) "
                "DO UPDATE SET digest=excluded.digest",
                (
                    generation,
                    record.page_id,
                    record.version,
                    record.lsn,
                    record.content_digest,
                    unix_ms(),
                ),
            )

        for page_id, consumed in batch.page_bytes.items():
            self.conn.execute(
                "INSERT INTO budget_usage("
                "generation,page_id,bytes_consumed,byte_budget,last_batch_id,updated_unix_ms"
                ") VALUES(?,?,?,?,?,?) ON CONFLICT(generation,page_id) DO UPDATE SET "
                "bytes_consumed=budget_usage.bytes_consumed+excluded.bytes_consumed,"
                "byte_budget=excluded.byte_budget,last_batch_id=excluded.last_batch_id,"
                "updated_unix_ms=excluded.updated_unix_ms",
                (
                    generation,
                    page_id,
                    consumed,
                    self.conn.execute(
                        "SELECT page_byte_budget FROM replay_batch WHERE batch_id=?",
                        (batch_id,),
                    ).fetchone()["page_byte_budget"],
                    batch_id,
                    unix_ms(),
                ),
            )

        cursor_lsn = batch.entries[-1].lsn
        revision_row = self.conn.execute(
            "SELECT COALESCE(MAX(revision_no),0)+1 AS revision_no "
            "FROM ledger_revision WHERE generation=?",
            (generation,),
        ).fetchone()
        revision_no = int(revision_row["revision_no"])
        aggregate = hashlib.sha256(
            "|".join(record.version_digest for record in batch.records).encode()
        ).hexdigest()
        self.conn.execute(
            "INSERT INTO ledger_revision("
            "generation,revision_no,batch_id,round_no,attempt_no,cursor_lsn,"
            "aggregate_digest,created_unix_ms) VALUES(?,?,?,?,?,?,?,?)",
            (
                generation,
                revision_no,
                batch_id,
                round_no,
                attempt_no,
                cursor_lsn,
                aggregate,
                unix_ms(),
            ),
        )
        self.conn.execute(
            "UPDATE progress_cursor SET last_committed_lsn=?,last_committed_round=?,"
            "last_batch_id=?,current_revision=?,updated_unix_ms=? WHERE generation=?",
            (cursor_lsn, round_no, batch_id, revision_no, unix_ms(), generation),
        )
        self.conn.execute(
            "INSERT INTO snapshot(generation,kind,label,at_lsn,digest,stale,created_unix_ms)"
            " VALUES(?, 'ledger_batch',?,?,?,0,?)",
            (
                generation,
                f"R{round_no}.{attempt_no}/rev{revision_no}",
                cursor_lsn,
                aggregate,
                unix_ms(),
            ),
        )

        if batch.checkpoint is not None:
            checkpoint = batch.checkpoint
            cp_digest = hashlib.sha256(
                "|".join(page.version_digest for page in checkpoint.pages).encode()
            ).hexdigest()
            cp_cursor = self.conn.execute(
                "INSERT INTO checkpoint("
                "generation,checkpoint_no,at_lsn,batch_id,aggregate_digest,created_unix_ms"
                ") VALUES(?,?,?,?,?,?)",
                (
                    generation,
                    checkpoint.checkpoint_no,
                    checkpoint.at_lsn,
                    batch_id,
                    cp_digest,
                    unix_ms(),
                ),
            )
            checkpoint_id = int(cp_cursor.lastrowid)
            for cp_page in checkpoint.pages:
                version_row = self.conn.execute(
                    "SELECT content_path FROM page_version WHERE generation=? "
                    "AND page_id=? AND version=?",
                    (generation, cp_page.page_id, cp_page.version),
                ).fetchone()
                content_path = version_row["content_path"] if version_row else ""
                self.conn.execute(
                    "INSERT INTO checkpoint_page("
                    "checkpoint_id,generation,page_id,version,lsn,content_digest,"
                    "version_digest,content_path) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        checkpoint_id,
                        generation,
                        cp_page.page_id,
                        cp_page.version,
                        cp_page.lsn,
                        cp_page.content_digest,
                        cp_page.version_digest,
                        content_path,
                    ),
                )
            self.conn.execute(
                "INSERT INTO snapshot(generation,kind,label,at_lsn,digest,stale,created_unix_ms)"
                " VALUES(?, 'checkpoint',?,?,?,0,?)",
                (
                    generation,
                    f"checkpoint#{checkpoint.checkpoint_no}@lsn{checkpoint.at_lsn}",
                    checkpoint.at_lsn,
                    cp_digest,
                    unix_ms(),
                ),
            )

        if before_commit is not None:
            before_commit()
        self.conn.commit()
        return revision_no

    def mark_running_unknown(self, generation: int) -> list[dict[str, Any]]:
        rows = [
            dict(row)
            for row in self.conn.execute(
                "SELECT batch_id,round_no,attempt_no,start_lsn,end_lsn "
                "FROM replay_batch WHERE generation=? AND status='running' "
                "ORDER BY round_no,attempt_no",
                (generation,),
            ).fetchall()
        ]
        self.conn.execute(
            "UPDATE replay_batch SET status='unknown', unknown_reason=? "
            "WHERE generation=? AND status='running'",
            (
                "process terminated after begin and before SQLite completion commit; rerun required",
                generation,
            ),
        )
        self.conn.commit()
        return rows

    def latest_checkpoint(self, generation: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM checkpoint WHERE generation=? AND stale=0 "
            "ORDER BY at_lsn DESC, checkpoint_id DESC LIMIT 1",
            (generation,),
        ).fetchone()
        if row is None:
            return None
        checkpoint = dict(row)
        checkpoint["pages"] = [
            dict(page_row)
            for page_row in self.conn.execute(
                "SELECT * FROM checkpoint_page WHERE checkpoint_id=? ORDER BY page_id",
                (checkpoint["checkpoint_id"],),
            ).fetchall()
        ]
        return checkpoint

    def checkpoint_at_lsn(self, generation: int, lsn: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM checkpoint WHERE generation=? AND at_lsn=? ORDER BY checkpoint_id DESC LIMIT 1",
            (generation, lsn),
        ).fetchone()
        if row is None:
            return None
        checkpoint = dict(row)
        checkpoint["pages"] = [
            dict(page_row)
            for page_row in self.conn.execute(
                "SELECT * FROM checkpoint_page WHERE checkpoint_id=? ORDER BY page_id",
                (checkpoint["checkpoint_id"],),
            ).fetchall()
        ]
        return checkpoint

    def committed_batches(self, generation: int) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.conn.execute(
                "SELECT * FROM replay_batch WHERE generation=? AND status='committed' "
                "ORDER BY round_no,attempt_no",
                (generation,),
            ).fetchall()
        ]

    def latest_page_versions(self, generation: int) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.conn.execute(
                "SELECT pv.* FROM page_version pv "
                "JOIN (SELECT page_id, MAX(version) AS max_version FROM page_version "
                "WHERE generation=? GROUP BY page_id) latest "
                "ON pv.page_id=latest.page_id AND pv.version=latest.max_version "
                "WHERE pv.generation=? ORDER BY pv.page_id",
                (generation, generation),
            ).fetchall()
        ]

    def latest_visible_page_versions(self, generation: int) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.conn.execute(
                "SELECT pv.* FROM page_version pv "
                "JOIN (SELECT page_id, MAX(version) AS max_version FROM page_version "
                "WHERE generation=? AND visible=1 GROUP BY page_id) latest "
                "ON pv.page_id=latest.page_id AND pv.version=latest.max_version "
                "WHERE pv.generation=? AND pv.visible=1 ORDER BY pv.page_id",
                (generation, generation),
            ).fetchall()
        ]

    def replace_current_view(
        self,
        generation: int,
        revision_no: int,
        cursor_lsn: int,
        rows: list[dict[str, Any]],
        crash_decision: dict[str, Any],
    ) -> int:
        now = unix_ms()
        self.conn.execute(
            "UPDATE view_snapshot SET status='stale' WHERE generation=? AND status='current'",
            (generation,),
        )
        summary = {
            "page_count": len(rows),
            "visible_count": sum(1 for row in rows if row["visible"]),
            "invisible_count": sum(1 for row in rows if not row["visible"]),
            "cursor_lsn": cursor_lsn,
        }
        cursor = self.conn.execute(
            "INSERT INTO view_snapshot("
            "generation,revision_no,cursor_lsn,status,created_unix_ms,summary_json"
            ") VALUES(?,?,?,'current',?,?)",
            (generation, revision_no, cursor_lsn, now, json.dumps(summary, sort_keys=True)),
        )
        view_id = int(cursor.lastrowid)
        for row in rows:
            self.conn.execute(
                "INSERT INTO page_view("
                "view_snapshot_id,generation,page_id,version,lsn,visible,after_crash,"
                "content_digest,version_digest,byte_length) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (
                    view_id,
                    generation,
                    row["page_id"],
                    row["version"],
                    row["lsn"],
                    int(row["visible"]),
                    int(row["after_crash"]),
                    row["content_digest"],
                    row["version_digest"],
                    row["byte_length"],
                ),
            )
        self.conn.execute(
            "INSERT INTO crash_decision(generation,crash_after_lsn,criterion,"
            "decision_json,updated_unix_ms) VALUES(?,?,?,?,?) "
            "ON CONFLICT(generation) DO UPDATE SET crash_after_lsn=excluded.crash_after_lsn,"
            "criterion=excluded.criterion,decision_json=excluded.decision_json,"
            "updated_unix_ms=excluded.updated_unix_ms",
            (
                generation,
                crash_decision["crash_after_lsn"],
                crash_decision["criterion"],
                json.dumps(crash_decision, sort_keys=True, ensure_ascii=False),
                now,
            ),
        )
        self.conn.commit()
        return view_id

    def current_view(self, generation: int) -> dict[str, Any] | None:
        row = self.conn.execute(
            "SELECT * FROM view_snapshot WHERE generation=? AND status='current' "
            "ORDER BY view_snapshot_id DESC LIMIT 1",
            (generation,),
        ).fetchone()
        if row is None:
            return None
        view = dict(row)
        view["summary"] = json.loads(view.pop("summary_json"))
        view["rows"] = [
            dict(item)
            for item in self.conn.execute(
                "SELECT * FROM page_view WHERE view_snapshot_id=? ORDER BY page_id",
                (view["view_snapshot_id"],),
            ).fetchall()
        ]
        return view

    def tamper_content_digest(self, generation: int) -> dict[str, Any]:
        row = self.conn.execute(
            "SELECT page_id,version,lsn,content_digest FROM page_version "
            "WHERE generation=? AND version>0 ORDER BY lsn LIMIT 1",
            (generation,),
        ).fetchone()
        if row is None:
            raise RuntimeError("no page version available to tamper")
        bad_digest = "0" * 64
        self.conn.execute(
            "UPDATE page_version SET content_digest=? WHERE generation=? AND page_id=? AND version=?",
            (bad_digest, generation, row["page_id"], row["version"]),
        )
        self.conn.commit()
        return dict(row)

    def query_dicts(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        return [dict(row) for row in self.conn.execute(sql, params).fetchall()]
