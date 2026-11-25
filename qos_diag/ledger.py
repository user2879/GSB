from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from . import FORMAT_VERSION
from .canonical import canonical_text, sha256_hex
from .errors import LedgerError
from .runtime import monotonic_ms


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS revisions (
    revision INTEGER PRIMARY KEY,
    parent_revision INTEGER,
    created_monotonic_ms INTEGER NOT NULL,
    reason TEXT NOT NULL,
    spec_digest TEXT NOT NULL,
    change_json TEXT NOT NULL,
    propagation_json TEXT NOT NULL,
    selection_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batches (
    batch_id TEXT PRIMARY KEY,
    revision INTEGER NOT NULL,
    kind TEXT NOT NULL,
    seq_start INTEGER NOT NULL,
    seq_end INTEGER NOT NULL,
    status TEXT NOT NULL,
    planned_at_ms INTEGER NOT NULL,
    committed_at_ms INTEGER,
    unknown_marked_at_ms INTEGER,
    recovered_from TEXT,
    note TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS event_results (
    revision INTEGER NOT NULL,
    event_seq INTEGER NOT NULL,
    pipeline_id TEXT NOT NULL,
    event_json TEXT NOT NULL,
    event_sha256 TEXT NOT NULL,
    PRIMARY KEY(revision, event_seq)
);
CREATE TABLE IF NOT EXISTS judgments (
    revision INTEGER NOT NULL,
    event_seq INTEGER NOT NULL,
    pipeline_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    shaper_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    evaluated INTEGER NOT NULL,
    allowed INTEGER NOT NULL,
    verdict TEXT NOT NULL,
    decision TEXT NOT NULL,
    required_q INTEGER NOT NULL,
    balance_before_refill_q INTEGER NOT NULL,
    refill_q INTEGER NOT NULL,
    balance_before_decision_q INTEGER NOT NULL,
    balance_after_decision_q INTEGER NOT NULL,
    state_clock_ms INTEGER NOT NULL,
    wait_start_ms INTEGER,
    wait_end_ms INTEGER,
    wait_ms INTEGER,
    receipt_digest TEXT,
    receipt_json TEXT,
    record_json TEXT NOT NULL,
    PRIMARY KEY(revision, event_seq, stage)
);
CREATE TABLE IF NOT EXISTS revision_artifacts (
    revision INTEGER NOT NULL,
    kind TEXT NOT NULL,
    content_json TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    PRIMARY KEY(revision, kind)
);
CREATE TABLE IF NOT EXISTS cursor_state (
    id INTEGER PRIMARY KEY CHECK(id = 1),
    revision INTEGER NOT NULL,
    sealed_seq INTEGER NOT NULL,
    updated_ms INTEGER NOT NULL
);
"""


class Ledger:
    def __init__(self, path: str | Path, conn: sqlite3.Connection, replaced: bool = False, replacement_reason: str | None = None):
        self.path = Path(path)
        self.conn = conn
        self.replaced = replaced
        self.replacement_reason = replacement_reason

    @classmethod
    def open(
        cls,
        path: str | Path,
        *,
        spec_digest: str | None = None,
        replace_on_mismatch: bool = True,
    ) -> "Ledger":
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            status, reason = cls._inspect(target)
            if status == "ok":
                conn = cls._connect(target)
                ledger = cls(target, conn)
                stored_digest = ledger.get_meta("content_digest")
                computed = ledger.compute_content_digest()
                if stored_digest == computed and ledger.get_meta("format_version") == str(FORMAT_VERSION):
                    return ledger
                reason = f"content digest/version mismatch: stored={stored_digest}, computed={computed}"
                ledger.close()
            if not replace_on_mismatch:
                raise LedgerError(reason or "invalid ledger")
            cls._quarantine(target, reason or "invalid ledger")
            cls.create_fresh(target, spec_digest=spec_digest)
            conn = cls._connect(target)
            ledger = cls(target, conn, replaced=True, replacement_reason=reason)
            return ledger
        cls.create_fresh(target, spec_digest=spec_digest)
        conn = cls._connect(target)
        return cls(target, conn)

    @staticmethod
    def _connect(path: Path) -> sqlite3.Connection:
        conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=DELETE")
        conn.execute("PRAGMA synchronous=FULL")
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @classmethod
    def _inspect(cls, path: Path) -> tuple[str, str | None]:
        try:
            conn = cls._connect(path)
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            tables = {
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            }
            required = {"meta", "revisions", "batches", "judgments", "event_results", "cursor_state"}
            meta = {}
            if "meta" in tables:
                meta = {row[0]: row[1] for row in conn.execute("SELECT key,value FROM meta").fetchall()}
            conn.close()
            if version not in (0, FORMAT_VERSION):
                return "invalid", f"unsupported SQLite user_version {version}"
            if not required.issubset(tables):
                return "invalid", "ledger schema is missing required tables"
            if meta.get("format_version") != str(FORMAT_VERSION):
                return "invalid", f"unsupported meta format_version {meta.get('format_version')}"
            if not meta.get("content_digest"):
                return "invalid", "missing content_digest"
            return "ok", None
        except sqlite3.DatabaseError as exc:
            return "invalid", f"database error: {exc}"

    @classmethod
    def create_fresh(cls, path: str | Path, *, spec_digest: str | None = None, parameter_revision: int = 1) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_name(f".{target.name}.fresh-{os.getpid()}-{monotonic_ms()}.tmp")
        conn = cls._connect(tmp)
        try:
            conn.executescript(SCHEMA_SQL)
            conn.execute("PRAGMA user_version=" + str(FORMAT_VERSION))
            now = monotonic_ms()
            conn.execute(
                "INSERT INTO meta(key,value) VALUES('format_version',?),('parameter_revision',?),('created_ms',?),('spec_digest',?)",
                (str(FORMAT_VERSION), str(parameter_revision), str(now), spec_digest or ""),
            )
            conn.execute(
                "INSERT INTO cursor_state(id,revision,sealed_seq,updated_ms) VALUES(1,?, ?,?)",
                (parameter_revision, -1, now),
            )
            digest = cls._compute_digest_for_connection(conn)
            conn.execute("INSERT INTO meta(key,value) VALUES('content_digest',?)", (digest,))
            conn.commit()
        finally:
            conn.close()
        os.replace(tmp, target)

    @staticmethod
    def _quarantine(path: Path, reason: str) -> None:
        quarantine = path.with_name(f"{path.name}.invalid-{monotonic_ms()}")
        try:
            os.replace(path, quarantine)
        except FileNotFoundError:
            return
        try:
            (quarantine.with_suffix(".reason.txt")).write_text(reason, encoding="utf-8")
        except OSError:
            pass

    def close(self) -> None:
        self.conn.close()

    def get_meta(self, key: str) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def set_meta(self, key: str, value: Any) -> None:
        with self.conn:
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, str(value)),
            )
            self.update_content_digest()

    def compute_content_digest(self) -> str:
        return self._compute_digest_for_connection(self.conn)

    @staticmethod
    def _compute_digest_for_connection(conn: sqlite3.Connection) -> str:
        payload: dict[str, Any] = {"format_version": FORMAT_VERSION}
        meta_rows = conn.execute("SELECT key,value FROM meta ORDER BY key").fetchall()
        payload["meta"] = {row["key"]: row["value"] for row in meta_rows if row["key"] != "content_digest"}
        tables = (
            "revisions",
            "batches",
            "event_results",
            "judgments",
            "revision_artifacts",
            "cursor_state",
        )
        for table in tables:
            try:
                rows = conn.execute(f"SELECT * FROM {table} ORDER BY 1,2,3").fetchall()
            except sqlite3.OperationalError:
                payload[table] = []
                continue
            items: list[dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                for json_field in ("change_json", "propagation_json", "selection_json", "content_json", "event_json"):
                    if json_field in item and isinstance(item[json_field], str) and item[json_field]:
                        try:
                            item[json_field] = json.loads(item[json_field])
                        except json.JSONDecodeError:
                            pass
                items.append(item)
            payload[table] = items
        return sha256_hex(payload)

    def update_content_digest(self) -> str:
        digest = self.compute_content_digest()
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES('content_digest',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (digest,),
        )
        return digest

    def current_revision(self) -> int:
        value = self.get_meta("parameter_revision")
        return int(value or "1")

    def cursor(self) -> dict[str, Any]:
        row = self.conn.execute("SELECT revision,sealed_seq,updated_ms FROM cursor_state WHERE id=1").fetchone()
        if row is None:
            return {"revision": self.current_revision(), "sealed_seq": -1, "updated_ms": None}
        return {"revision": int(row["revision"]), "sealed_seq": int(row["sealed_seq"]), "updated_ms": int(row["updated_ms"])}

    def plan_batch(
        self,
        batch_id: str,
        revision: int,
        seq_start: int,
        seq_end: int,
        *,
        kind: str = "run",
        recovered_from: str | None = None,
    ) -> None:
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO batches(batch_id,revision,kind,seq_start,seq_end,status,planned_at_ms,recovered_from)
                VALUES(?,?,?,?,?,'planned',?,?)
                ON CONFLICT(batch_id) DO UPDATE SET
                  revision=excluded.revision, kind=excluded.kind, seq_start=excluded.seq_start,
                  seq_end=excluded.seq_end, status='planned', planned_at_ms=excluded.planned_at_ms,
                  unknown_marked_at_ms=NULL, recovered_from=excluded.recovered_from
                """,
                (batch_id, revision, kind, seq_start, seq_end, monotonic_ms(), recovered_from),
            )

    def mark_unfinished_unknown(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT batch_id,revision,seq_start,seq_end,status FROM batches WHERE status IN ('planned','running') ORDER BY seq_start"
        ).fetchall()
        items = [dict(row) for row in rows]
        with self.conn:
            self.conn.execute(
                "UPDATE batches SET status='unknown', unknown_marked_at_ms=? WHERE status IN ('planned','running')",
                (monotonic_ms(),),
            )
            self.update_content_digest()
        return items

    def unknown_batches(self) -> list[dict[str, Any]]:
        rows = self.conn.execute(
            "SELECT batch_id,revision,seq_start,seq_end,status FROM batches WHERE status='unknown' ORDER BY seq_start"
        ).fetchall()
        return [dict(row) for row in rows]

    def commit_event_batch(
        self,
        *,
        batch_id: str,
        revision: int,
        results: list[dict[str, Any]],
        call_results: dict[str, dict[str, Any]],
        revision_info: dict[str, Any] | None = None,
        artifacts: dict[str, Any] | None = None,
        sealed_seq: int | None = None,
        recovered_from: str | None = None,
        copy_from_revision: int | None = None,
        only_stage_keys: set[tuple[int, str]] | None = None,
    ) -> None:
        if not results:
            raise LedgerError("cannot commit an empty batch", object_id=batch_id)
        seq_start = min(int(item["event_seq"]) for item in results)
        seq_end = max(int(item["event_seq"]) for item in results)
        now = monotonic_ms()
        with self.conn:
            if revision_info is not None:
                self.conn.execute(
                    """
                    INSERT INTO revisions(revision,parent_revision,created_monotonic_ms,reason,spec_digest,change_json,propagation_json,selection_json)
                    VALUES(?,?,?,?,?,?,?,?)
                    ON CONFLICT(revision) DO NOTHING
                    """,
                    (
                        int(revision),
                        revision_info.get("parent_revision"),
                        now,
                        str(revision_info.get("reason", "run")),
                        str(revision_info.get("spec_digest", "")),
                        canonical_text(revision_info.get("change", {})),
                        canonical_text(revision_info.get("propagation", {})),
                        canonical_text(revision_info.get("selection", {})),
                    ),
                )
            else:
                row = self.conn.execute("SELECT revision FROM revisions WHERE revision=?", (revision,)).fetchone()
                if row is None:
                    self.conn.execute(
                        """
                        INSERT INTO revisions(revision,parent_revision,created_monotonic_ms,reason,spec_digest,change_json,propagation_json,selection_json)
                        VALUES(?,?,?,?,?,?,?,?)
                        """,
                        (revision, max(0, revision - 1), now, "run", "", canonical_text({}), canonical_text({}), canonical_text({})),
                )

            if copy_from_revision is not None and int(copy_from_revision) != int(revision):
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO event_results(revision,event_seq,pipeline_id,event_json,event_sha256)
                    SELECT ?,event_seq,pipeline_id,event_json,event_sha256
                    FROM event_results WHERE revision=?
                    """,
                    (int(revision), int(copy_from_revision)),
                )
                judgment_columns = (
                    "event_seq,pipeline_id,stage,shaper_id,batch_id,evaluated,allowed,verdict,decision,"
                    "required_q,balance_before_refill_q,refill_q,balance_before_decision_q,"
                    "balance_after_decision_q,state_clock_ms,wait_start_ms,wait_end_ms,wait_ms,"
                    "receipt_digest,receipt_json,record_json"
                )
                self.conn.execute(
                    f"""
                    INSERT OR IGNORE INTO judgments(revision,{judgment_columns})
                    SELECT ?,{judgment_columns}
                    FROM judgments WHERE revision=?
                    """,
                    (int(revision), int(copy_from_revision)),
                )

            self.conn.execute(
                """
                INSERT INTO batches(batch_id,revision,kind,seq_start,seq_end,status,planned_at_ms,committed_at_ms,recovered_from)
                VALUES(?,?,?,? ,?, 'committed', ?, ?, ?)
                ON CONFLICT(batch_id) DO UPDATE SET
                  revision=excluded.revision, seq_start=excluded.seq_start, seq_end=excluded.seq_end,
                  status='committed', committed_at_ms=excluded.committed_at_ms, recovered_from=excluded.recovered_from
                """,
                (batch_id, revision, "judgment", seq_start, seq_end, now, now, recovered_from),
            )

            for result in sorted(results, key=lambda item: int(item["event_seq"])):
                seq = int(result["event_seq"])
                self.conn.execute(
                    "INSERT OR REPLACE INTO event_results(revision,event_seq,pipeline_id,event_json,event_sha256) VALUES(?,?,?,?,?)",
                    (revision, seq, result["pipeline_id"], canonical_text(result), sha256_hex(result)),
                )
                for record in result["stages"]:
                    if only_stage_keys is not None and (seq, record["stage"]) not in only_stage_keys:
                        continue
                    key = f"{seq}:{record['stage']}"
                    call = call_results.get(key)
                    receipt_json = canonical_text(call["receipt"]) if call and call.get("receipt") is not None else None
                    self.conn.execute(
                        """
                        INSERT OR REPLACE INTO judgments(
                          revision,event_seq,pipeline_id,stage,shaper_id,batch_id,evaluated,allowed,verdict,decision,
                          required_q,balance_before_refill_q,refill_q,balance_before_decision_q,
                          balance_after_decision_q,state_clock_ms,wait_start_ms,wait_end_ms,wait_ms,
                          receipt_digest,receipt_json,record_json)
                        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        (
                            revision,
                            seq,
                            result["pipeline_id"],
                            record["stage"],
                            record["shaper_id"],
                            batch_id,
                            1 if record["evaluated"] else 0,
                            1 if record["verdict"] == "allow" else 0,
                            record["verdict"],
                            record["decision"],
                            int(record["required_q"]),
                            int(record["balance_before_refill_q"]),
                            int(record["refill_q"]),
                            int(record["balance_before_decision_q"]),
                            int(record["balance_after_decision_q"]),
                            int(record["state_clock_ms"]),
                            0 if call is None or call.get("wait_start_ms") is None else int(call["wait_start_ms"]),
                            0 if call is None or call.get("wait_end_ms") is None else int(call["wait_end_ms"]),
                            0 if call is None or call.get("wait_ms") is None else int(call["wait_ms"]),
                            None if call is None else call.get("receipt_digest"),
                            receipt_json,
                            canonical_text(record),
                        ),
                    )

            if artifacts:
                for kind, content in artifacts.items():
                    self.conn.execute(
                        "INSERT OR REPLACE INTO revision_artifacts(revision,kind,content_json,content_sha256) VALUES(?,?,?,?)",
                        (revision, kind, canonical_text(content), sha256_hex(content)),
                    )

            new_sealed = seq_end if sealed_seq is None else int(sealed_seq)
            current = self.cursor()
            sealed_value = max(int(current["sealed_seq"]), new_sealed)
            self.conn.execute(
                "INSERT INTO cursor_state(id,revision,sealed_seq,updated_ms) VALUES(1,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET revision=excluded.revision, sealed_seq=excluded.sealed_seq, updated_ms=excluded.updated_ms",
                (revision, sealed_value, now),
            )
            self.conn.execute(
                "INSERT INTO meta(key,value) VALUES('parameter_revision',?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(revision),),
            )
            self.update_content_digest()

    def load_event_results(self, revision: int | None = None) -> list[dict[str, Any]]:
        revision = int(revision or self.current_revision())
        rows = self.conn.execute(
            "SELECT event_json FROM event_results WHERE revision=? ORDER BY event_seq",
            (revision,),
        ).fetchall()
        return [json.loads(row["event_json"]) for row in rows]

    def load_judgment_rows(self, revision: int | None = None) -> list[dict[str, Any]]:
        revision = int(revision or self.current_revision())
        rows = self.conn.execute(
            "SELECT * FROM judgments WHERE revision=? ORDER BY event_seq, stage",
            (revision,),
        ).fetchall()
        return [dict(row) for row in rows]

    def load_artifacts(self, revision: int | None = None) -> dict[str, Any]:
        revision = int(revision or self.current_revision())
        rows = self.conn.execute(
            "SELECT kind,content_json FROM revision_artifacts WHERE revision=?",
            (revision,),
        ).fetchall()
        return {row["kind"]: json.loads(row["content_json"]) for row in rows}

    def verify_stored_digest(self) -> bool:
        return self.get_meta("content_digest") == self.compute_content_digest()
