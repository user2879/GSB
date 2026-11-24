"""SQLite ledger and session snapshot with format versions and digests.

A *batch* is the all-or-nothing persistence unit for one stage: its summary,
per-block digests, feature stats and error-chain rows are written in a single
transaction. A crashed process therefore never leaves a half-written stage;
reopening marks rows whose batch commit flag is missing as ``unknown`` and
requires them to be rerun.
"""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass

from . import LEDGER_FORMAT_VERSION, SNAPSHOT_FORMAT_VERSION
from .errors import SnapshotError
from .util import canonical_json, content_digest


def _row_digest(table: str, payload: dict) -> str:
    return content_digest({"table": table, "payload": payload})


class Ledger:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fresh = not os.path.exists(path)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self._create()
        if fresh:
            self._set_meta("format_version", LEDGER_FORMAT_VERSION)
        self._check_format()

    def _create(self) -> None:
        self.conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS stage_batches(
                spec_version INTEGER NOT NULL,
                stage_id TEXT NOT NULL,
                seq INTEGER NOT NULL,
                status TEXT NOT NULL,
                summary_json TEXT NOT NULL,
                block_digests_json TEXT NOT NULL,
                digest TEXT NOT NULL,
                PRIMARY KEY(spec_version, stage_id));
            CREATE TABLE IF NOT EXISTS feature_stats(
                spec_version INTEGER NOT NULL,
                stage_id TEXT NOT NULL,
                window TEXT NOT NULL,
                stats_json TEXT NOT NULL,
                digest TEXT NOT NULL,
                PRIMARY KEY(spec_version, stage_id, window));
            CREATE TABLE IF NOT EXISTS error_chains(
                spec_version INTEGER NOT NULL,
                stage_id TEXT NOT NULL,
                chain_json TEXT NOT NULL,
                digest TEXT NOT NULL,
                PRIMARY KEY(spec_version, stage_id));
            CREATE TABLE IF NOT EXISTS drift_verdicts(
                spec_version INTEGER NOT NULL,
                stage_id TEXT NOT NULL,
                verdict_json TEXT NOT NULL,
                digest TEXT NOT NULL,
                PRIMARY KEY(spec_version, stage_id));
            """)
        self.conn.commit()

    def _set_meta(self, key: str, value) -> None:
        self.conn.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, canonical_json(value)))
        self.conn.commit()

    def _check_format(self) -> None:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key='format_version'").fetchone()
        if row is None or json.loads(row["value"]) != LEDGER_FORMAT_VERSION:
            raise SnapshotError("ledger format version mismatch",
                                location="<ledger>", field="format_version")

    # -- batch lifecycle ---------------------------------------------------
    def begin_batch(self, spec_version: int, stage_id: str, seq: int) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO stage_batches"
            "(spec_version,stage_id,seq,status,summary_json,block_digests_json,digest)"
            " VALUES(?,?,?,?,'','','')",
            (spec_version, stage_id, seq, "unknown"))
        self.conn.commit()

    def commit_stage(self, spec_version: int, stage_id: str, seq: int,
                     summary: dict, block_digests: list[str]) -> None:
        payload = {"seq": seq, "summary": summary, "block_digests": block_digests}
        digest = _row_digest("stage_batches", payload)
        with self.conn:
            # Upsert: a downstream stage whose dependency crashed never got a
            # begin_batch row, so UPDATE alone would silently do nothing.
            self.conn.execute(
                "INSERT INTO stage_batches"
                "(spec_version,stage_id,seq,status,summary_json,block_digests_json,digest)"
                " VALUES(?,?,?,'committed',?,?,?) "
                "ON CONFLICT(spec_version,stage_id) DO UPDATE SET "
                "status='committed',summary_json=excluded.summary_json,"
                "block_digests_json=excluded.block_digests_json,digest=excluded.digest",
                (spec_version, stage_id, seq, canonical_json(summary),
                 canonical_json(block_digests), digest))

    def put_feature_stats(self, spec_version: int, stage_id: str,
                          window: str, stats: dict) -> None:
        digest = _row_digest("feature_stats", stats)
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO feature_stats"
                "(spec_version,stage_id,window,stats_json,digest)"
                " VALUES(?,?,?,?,?)",
                (spec_version, stage_id, window, canonical_json(stats), digest))

    def put_chain(self, spec_version: int, stage_id: str, chain: dict) -> None:
        digest = _row_digest("error_chains", chain)
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO error_chains"
                "(spec_version,stage_id,chain_json,digest) VALUES(?,?,?,?)",
                (spec_version, stage_id, canonical_json(chain), digest))

    def put_verdict(self, spec_version: int, stage_id: str, verdict: dict) -> None:
        digest = _row_digest("drift_verdicts", verdict)
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO drift_verdicts"
                "(spec_version,stage_id,verdict_json,digest) VALUES(?,?,?,?)",
                (spec_version, stage_id, canonical_json(verdict), digest))

    # -- reads -------------------------------------------------------------
    def batch_status(self, spec_version: int, stage_id: str):
        return self.conn.execute(
            "SELECT status, digest FROM stage_batches WHERE spec_version=? AND stage_id=?",
            (spec_version, stage_id)).fetchone()

    def committed_stages(self, spec_version: int) -> set[str]:
        rows = self.conn.execute(
            "SELECT stage_id FROM stage_batches WHERE spec_version=? AND status='committed'",
            (spec_version,)).fetchall()
        return {r["stage_id"] for r in rows}

    def unknown_stages(self, spec_version: int) -> set[str]:
        rows = self.conn.execute(
            "SELECT stage_id FROM stage_batches WHERE spec_version=? AND status='unknown'",
            (spec_version,)).fetchall()
        return {r["stage_id"] for r in rows}

    def get_summary(self, spec_version: int, stage_id: str):
        r = self.conn.execute(
            "SELECT summary_json, digest FROM stage_batches WHERE spec_version=? AND stage_id=?",
            (spec_version, stage_id)).fetchone()
        return (json.loads(r["summary_json"]), r["digest"]) if r else (None, None)

    def get_feature_stats(self, spec_version: int, stage_id: str, window: str):
        r = self.conn.execute(
            "SELECT stats_json, digest FROM feature_stats WHERE spec_version=? AND stage_id=? AND window=?",
            (spec_version, stage_id, window)).fetchone()
        return (json.loads(r["stats_json"]), r["digest"]) if r else (None, None)

    def get_chain(self, spec_version: int, stage_id: str):
        r = self.conn.execute(
            "SELECT chain_json, digest FROM error_chains WHERE spec_version=? AND stage_id=?",
            (spec_version, stage_id)).fetchone()
        return (json.loads(r["chain_json"]), r["digest"]) if r else (None, None)

    def get_verdict(self, spec_version: int, stage_id: str):
        r = self.conn.execute(
            "SELECT verdict_json, digest FROM drift_verdicts WHERE spec_version=? AND stage_id=?",
            (spec_version, stage_id)).fetchone()
        return (json.loads(r["verdict_json"]), r["digest"]) if r else (None, None)

    def replace_all(self) -> None:
        """Whole-ledger replacement used when snapshot identity mismatches."""
        with self.conn:
            for tbl in ("stage_batches", "feature_stats", "error_chains",
                        "drift_verdicts"):
                self.conn.execute(f"DELETE FROM {tbl}")

    def close(self) -> None:
        self.conn.close()


@dataclass
class Snapshot:
    spec_version: int
    spec_digest: str
    sources_digest: str
    scale_policy_version: int
    ledger_cursor: dict
    format_version: int = SNAPSHOT_FORMAT_VERSION

    def to_dict(self) -> dict:
        return {"format_version": self.format_version,
                "spec_version": self.spec_version,
                "spec_digest": self.spec_digest,
                "sources_digest": self.sources_digest,
                "scale_policy_version": self.scale_policy_version,
                "ledger_cursor": self.ledger_cursor}

    def digest(self) -> str:
        return _row_digest("snapshot", self.to_dict())

    def write(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        doc = {"data": self.to_dict(), "digest": self.digest()}
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(canonical_json(doc))

    @classmethod
    def read(cls, path: str) -> "Snapshot":
        with open(path, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        snap = cls(**doc["data"])
        if snap.format_version != SNAPSHOT_FORMAT_VERSION:
            raise SnapshotError("snapshot format version mismatch",
                                location="<snapshot>", field="format_version")
        if snap.digest() != doc["digest"]:
            raise SnapshotError("snapshot content digest mismatch",
                                location="<snapshot>", field="digest")
        return snap
