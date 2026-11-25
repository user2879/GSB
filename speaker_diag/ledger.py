"""SQLite ledger with per-batch transactions, snapshots, and replay checks."""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .canonical import (
    atomic_write_json,
    canonical_bytes,
    canonical_sha256,
    sha256_hex,
)
from .errors import DiagnosticError
from .invalidation import REPLAY_IGNORED_FIELDS, find_first_receipt_difference
from .receipts import NormalizedReceipt
from .synthetic import WindowTruth


LEDGER_SCHEMA_VERSION = 1
LEDGER_SCHEMA_NAME = "speaker-diag-ledger"


def _connect(path: str | Path) -> sqlite3.Connection:
    connection = sqlite3.connect(str(path), timeout=10.0, isolation_level=None)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = FULL")
    connection.execute("PRAGMA busy_timeout = 10000")
    return connection


class Ledger:
    def __init__(
        self,
        path: str | Path,
        *,
        session_version: int,
        spec_digest: str,
        session_snapshot: Mapping[str, Any],
        allow_replace: bool = True,
        reset: bool = False,
    ) -> None:
        if type(session_version) is not int or session_version < 1:
            raise DiagnosticError("E_REPLAY", "session_version must be a positive integer", field="session_version")
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.session_version = session_version
        self.spec_digest = spec_digest
        self.session_snapshot = dict(session_snapshot)
        self._initialize(allow_replace=allow_replace, reset=reset)

    @property
    def database_path(self) -> str:
        return str(self.path)

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.executescript(
            f"""
            PRAGMA user_version = {LEDGER_SCHEMA_VERSION};
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS session_snapshots (
                session_version INTEGER PRIMARY KEY,
                spec_digest TEXT NOT NULL,
                snapshot_json TEXT NOT NULL,
                snapshot_sha256 TEXT NOT NULL,
                replaced_from INTEGER,
                created_ns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS batches (
                batch_id TEXT PRIMARY KEY,
                ordinal INTEGER NOT NULL UNIQUE,
                worker_id TEXT NOT NULL,
                param_version INTEGER NOT NULL,
                source_format TEXT NOT NULL,
                receipt_json TEXT NOT NULL,
                receipt_sha256 TEXT NOT NULL,
                attribution_digest TEXT NOT NULL,
                participant_json TEXT NOT NULL,
                elapsed_ms INTEGER NOT NULL,
                content_digest TEXT NOT NULL,
                committed_ns INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS windows (
                batch_id TEXT NOT NULL,
                window_id TEXT NOT NULL,
                stable_rank INTEGER NOT NULL,
                attribution TEXT NOT NULL,
                duplicate_of TEXT,
                truth_speaker TEXT NOT NULL,
                truth_source TEXT NOT NULL,
                overlap_bps INTEGER NOT NULL,
                active_speakers_json TEXT NOT NULL,
                score_margin INTEGER,
                confidence_q12 INTEGER,
                PRIMARY KEY(batch_id, window_id),
                FOREIGN KEY(batch_id) REFERENCES batches(batch_id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS unknown_batches (
                batch_id TEXT PRIMARY KEY,
                expected_ordinal INTEGER NOT NULL,
                detected_ns INTEGER NOT NULL,
                reason TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_windows_window_id ON windows(window_id);
            """
        )

    def _initialize(self, *, allow_replace: bool, reset: bool) -> None:
        connection = _connect(self.path)
        try:
            self._create_schema(connection)
            connection.execute("BEGIN IMMEDIATE")
            if reset:
                for table in ("windows", "batches", "unknown_batches", "session_snapshots", "meta"):
                    connection.execute(f"DELETE FROM {table}")
            meta = {row["key"]: row["value"] for row in connection.execute("SELECT key, value FROM meta")}
            existing_schema = meta.get("ledger_schema_version")
            if existing_schema is not None and int(existing_schema) != LEDGER_SCHEMA_VERSION:
                raise DiagnosticError(
                    "E_REPLAY",
                    "ledger schema version is not supported",
                    field="ledger_schema_version",
                    details={"found": existing_schema, "supported": LEDGER_SCHEMA_VERSION},
                )
            existing_session = meta.get("session_version")
            existing_spec_digest = meta.get("spec_digest")
            replaced_from: int | None = None
            if existing_session is not None and int(existing_session) != self.session_version:
                if not allow_replace:
                    raise DiagnosticError(
                        "E_REPLAY",
                        "ledger session version differs; replacement was not allowed",
                        field="session_version",
                        details={"existing": int(existing_session), "requested": self.session_version},
                    )
                replaced_from = int(existing_session)
                for table in ("windows", "batches", "unknown_batches", "session_snapshots"):
                    connection.execute(f"DELETE FROM {table}")
                connection.execute("DELETE FROM meta")
                meta = {}
            elif existing_session is not None and existing_spec_digest != self.spec_digest:
                raise DiagnosticError(
                    "E_REPLAY",
                    "same session version has a different specification digest",
                    field="spec_digest",
                    details={"existing": existing_spec_digest, "requested": self.spec_digest},
                )

            initial_chain = sha256_hex(
                canonical_bytes(
                    {
                        "ledger_schema_version": LEDGER_SCHEMA_VERSION,
                        "session_version": self.session_version,
                        "spec_digest": self.spec_digest,
                    }
                )
            )
            defaults = {
                "ledger_schema_version": str(LEDGER_SCHEMA_VERSION),
                "ledger_schema_name": LEDGER_SCHEMA_NAME,
                "session_version": str(self.session_version),
                "spec_digest": self.spec_digest,
                "chain_digest": initial_chain,
                "batch_count": "0",
            }
            for key, value in defaults.items():
                connection.execute(
                    "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO NOTHING",
                    (key, value),
                )
            snapshot_json = canonical_bytes(dict(self.session_snapshot)).decode("utf-8")
            snapshot_sha = sha256_hex(snapshot_json)
            connection.execute(
                """
                INSERT INTO session_snapshots(
                    session_version, spec_digest, snapshot_json, snapshot_sha256, replaced_from, created_ns
                ) VALUES(?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_version) DO UPDATE SET
                    spec_digest=excluded.spec_digest,
                    snapshot_json=excluded.snapshot_json,
                    snapshot_sha256=excluded.snapshot_sha256,
                    replaced_from=COALESCE(session_snapshots.replaced_from, excluded.replaced_from),
                    created_ns=excluded.created_ns
                """,
                (
                    self.session_version,
                    self.spec_digest,
                    snapshot_json,
                    snapshot_sha,
                    replaced_from,
                    time.time_ns(),
                ),
            )
            connection.execute("COMMIT")
        except BaseException:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()

    def _meta(self, connection: sqlite3.Connection) -> dict[str, str]:
        return {row["key"]: row["value"] for row in connection.execute("SELECT key, value FROM meta")}

    def commit_batch(
        self,
        normalized: NormalizedReceipt,
        truth_by_window: Mapping[str, WindowTruth],
        *,
        ordinal: int,
    ) -> dict[str, Any]:
        if type(ordinal) is not int or ordinal < 0:
            raise DiagnosticError("E_REPLAY", "batch ordinal must be a non-negative integer", field="ordinal")
        data = normalized.data
        windows = list(data["windows"])
        missing_truth = [window["window_id"] for window in windows if window["window_id"] not in truth_by_window]
        if missing_truth:
            raise DiagnosticError(
                "E_REPLAY",
                "window truth is required in the same transaction as receipt attribution",
                field="window_id",
                details={"window_id": missing_truth[0]},
            )
        connection = _connect(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            meta = self._meta(connection)
            if int(meta["session_version"]) != self.session_version:
                raise DiagnosticError("E_REPLAY", "ledger session changed concurrently", field="session_version")
            existing = connection.execute(
                "SELECT batch_id FROM batches WHERE batch_id=?", (data["batch_id"],)
            ).fetchone()
            if existing is not None:
                raise DiagnosticError(
                    "E_REPLAY",
                    "batch has already been committed",
                    field="batch_id",
                    details={"batch_id": data["batch_id"]},
                )
            ordinal_row = connection.execute(
                "SELECT batch_id FROM batches WHERE ordinal=?", (ordinal,)
            ).fetchone()
            if ordinal_row is not None:
                raise DiagnosticError(
                    "E_REPLAY",
                    "batch ordinal is already occupied",
                    field="ordinal",
                    details={"ordinal": ordinal, "batch_id": ordinal_row["batch_id"]},
                )
            window_rows = []
            for window in sorted(windows, key=lambda item: item["rank"]):
                truth = truth_by_window[window["window_id"]]
                window_rows.append(
                    {
                        "window_id": window["window_id"],
                        "stable_rank": window["rank"],
                        "attribution": window["attribution"],
                        "duplicate_of": window["duplicate_of"],
                        "truth_speaker": truth.truth_speaker,
                        "truth_source": truth.truth_source,
                        "overlap_bps": truth.overlap_bps,
                        "active_speakers": list(truth.active_speakers),
                        "score_margin": window.get("score_margin"),
                        "confidence_q12": window.get("confidence_q12"),
                    }
                )
            record = {
                "batch_id": data["batch_id"],
                "ordinal": ordinal,
                "worker_id": data["worker_id"],
                "param_version": data["param_version"],
                "source_format": data["source_format"],
                "receipt_sha256": normalized.sha256,
                "attribution_digest": normalized.attribution_digest,
                "participant_window_ids": list(data["participant_window_ids"]),
                "windows": window_rows,
            }
            content_digest = sha256_hex(
                bytes.fromhex(meta["chain_digest"]) + canonical_bytes(record)
            )
            now_ns = time.time_ns()
            receipt_json = normalized.canonical.decode("utf-8")
            connection.execute(
                """
                INSERT INTO batches(
                    batch_id, ordinal, worker_id, param_version, source_format,
                    receipt_json, receipt_sha256, attribution_digest, participant_json,
                    elapsed_ms, content_digest, committed_ns
                ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["batch_id"],
                    ordinal,
                    data["worker_id"],
                    data["param_version"],
                    data["source_format"],
                    receipt_json,
                    normalized.sha256,
                    normalized.attribution_digest,
                    canonical_bytes(list(data["participant_window_ids"])).decode("utf-8"),
                    data["elapsed_ms"],
                    content_digest,
                    now_ns,
                ),
            )
            for row in window_rows:
                connection.execute(
                    """
                    INSERT INTO windows(
                        batch_id, window_id, stable_rank, attribution, duplicate_of,
                        truth_speaker, truth_source, overlap_bps, active_speakers_json,
                        score_margin, confidence_q12
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        data["batch_id"],
                        row["window_id"],
                        row["stable_rank"],
                        row["attribution"],
                        row["duplicate_of"],
                        row["truth_speaker"],
                        row["truth_source"],
                        row["overlap_bps"],
                        canonical_bytes(row["active_speakers"]).decode("utf-8"),
                        row["score_margin"],
                        row["confidence_q12"],
                    ),
                )
            connection.execute("DELETE FROM unknown_batches WHERE batch_id=?", (data["batch_id"],))
            batch_count = int(meta["batch_count"]) + 1
            connection.execute("UPDATE meta SET value=? WHERE key='chain_digest'", (content_digest,))
            connection.execute("UPDATE meta SET value=? WHERE key='batch_count'", (str(batch_count),))
            connection.execute("UPDATE meta SET value=? WHERE key='last_batch_id'", (data["batch_id"],)) if connection.execute(
                "SELECT 1 FROM meta WHERE key='last_batch_id'"
            ).fetchone() else connection.execute(
                "INSERT INTO meta(key, value) VALUES('last_batch_id', ?)", (data["batch_id"],)
            )
            connection.execute("COMMIT")
            return {
                "batch_id": data["batch_id"],
                "ordinal": ordinal,
                "content_digest": content_digest,
                "receipt_sha256": normalized.sha256,
                "window_count": len(window_rows),
            }
        except BaseException:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()

    def commit_many(
        self,
        receipts: Mapping[str, NormalizedReceipt],
        truth_by_window: Mapping[str, WindowTruth],
        ordinal_by_batch: Mapping[str, int],
    ) -> list[dict[str, Any]]:
        results = []
        for batch_id in sorted(receipts, key=lambda item: ordinal_by_batch[item]):
            results.append(
                self.commit_batch(
                    receipts[batch_id],
                    truth_by_window,
                    ordinal=ordinal_by_batch[batch_id],
                )
            )
        return results

    def batch_rows(self) -> list[dict[str, Any]]:
        connection = _connect(self.path)
        try:
            rows = connection.execute("SELECT * FROM batches ORDER BY ordinal").fetchall()
            result = []
            for row in rows:
                windows = connection.execute(
                    "SELECT * FROM windows WHERE batch_id=? ORDER BY stable_rank", (row["batch_id"],)
                ).fetchall()
                result.append({**dict(row), "windows": [dict(window) for window in windows]})
            return result
        finally:
            connection.close()

    def load_normalized_receipts(self) -> dict[str, NormalizedReceipt]:
        connection = _connect(self.path)
        try:
            rows = connection.execute("SELECT * FROM batches ORDER BY ordinal").fetchall()
            result: dict[str, NormalizedReceipt] = {}
            for row in rows:
                data = json.loads(row["receipt_json"])
                encoded = row["receipt_json"].encode("utf-8")
                result[row["batch_id"]] = NormalizedReceipt(
                    data=data,
                    canonical=encoded,
                    sha256=row["receipt_sha256"],
                    attribution_digest=row["attribution_digest"],
                    source_format=row["source_format"],
                    filled_fields=tuple(data.get("filled_fields", [])),
                )
            return result
        finally:
            connection.close()

    def recover(
        self,
        expected_batch_ids: Sequence[str],
        *,
        reason: str = "process terminated before batch commit",
    ) -> dict[str, Any]:
        expected = list(expected_batch_ids)
        if len(set(expected)) != len(expected):
            raise DiagnosticError("E_REPLAY", "expected batch order contains duplicates", field="expected_batch_ids")
        connection = _connect(self.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute("SELECT batch_id, ordinal FROM batches ORDER BY ordinal").fetchall()
            committed = [row["batch_id"] for row in rows]
            committed_set = set(committed)
            expected_set = set(expected)
            missing = [batch_id for batch_id in expected if batch_id not in committed_set]
            unexpected = [batch_id for batch_id in committed if batch_id not in expected_set]
            now_ns = time.time_ns()
            for ordinal, batch_id in enumerate(expected):
                if batch_id in missing:
                    connection.execute(
                        """
                        INSERT INTO unknown_batches(batch_id, expected_ordinal, detected_ns, reason)
                        VALUES(?, ?, ?, ?)
                        ON CONFLICT(batch_id) DO UPDATE SET
                            expected_ordinal=excluded.expected_ordinal,
                            detected_ns=excluded.detected_ns,
                            reason=excluded.reason
                        """,
                        (batch_id, ordinal, now_ns, reason),
                    )
            connection.execute("COMMIT")
            unknown_rows = connection.execute(
                "SELECT batch_id, expected_ordinal, detected_ns, reason FROM unknown_batches ORDER BY expected_ordinal"
            ).fetchall()
            return {
                "schema": "speaker-diag-recovery-v1",
                "session_version": self.session_version,
                "committed_batches": committed,
                "missing_batches": missing,
                "unknown_batches": [dict(row) for row in unknown_rows],
                "unexpected_batches": unexpected,
                "rerun_order": missing,
                "rerun_required": bool(missing),
                "recovery_rule": "only committed rows are trusted; missing batches are unknown and must be rerun",
            }
        except BaseException:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()

    def verify_replay(
        self,
        regenerated: Mapping[str, NormalizedReceipt],
        *,
        expected_order: Sequence[str] | None = None,
        ignore_fields: Iterable[str] = REPLAY_IGNORED_FIELDS,
    ) -> dict[str, Any]:
        committed = self.load_normalized_receipts()
        rows = self.batch_rows()
        ledger_order = [row["batch_id"] for row in rows]
        order = list(expected_order) if expected_order is not None else ledger_order
        if set(order) != set(committed):
            extra_committed = sorted(set(committed) - set(order))
            extra_expected = sorted(set(order) - set(committed))
            raise DiagnosticError(
                "E_REPLAY",
                "replay batch set does not match the ledger",
                field="batch_id",
                details={"extra_committed": extra_committed, "missing_committed": extra_expected},
            )
        if order != ledger_order:
            first_index = next(
                index for index, (left, right) in enumerate(zip(order, ledger_order)) if left != right
            )
            raise DiagnosticError(
                "E_REPLAY",
                "replay batch order does not match the committed ledger order",
                field="ordinal",
                details={
                    "reason": "batch_order_mismatch",
                    "window_id": None,
                    "batch_id": order[first_index],
                    "ordinal": first_index,
                    "expected_batch_id": order[first_index],
                    "actual_batch_id": ledger_order[first_index],
                },
            )
        for index, batch_id in enumerate(order):
            if batch_id not in committed:
                raise DiagnosticError(
                    "E_REPLAY",
                    "replay is missing a committed ledger batch",
                    field="batch_id",
                    details={"batch_id": batch_id, "ordinal": index},
                )
            if batch_id not in regenerated:
                raise DiagnosticError(
                    "E_REPLAY",
                    "regenerated replay result is missing a batch",
                    field="batch_id",
                    details={"batch_id": batch_id, "ordinal": index},
                )
            difference = find_first_receipt_difference(
                committed[batch_id].data,
                regenerated[batch_id].data,
                ignore_fields=ignore_fields,
            )
            if difference is not None:
                raise DiagnosticError(
                    "E_REPLAY",
                    "replay differs from the committed ledger",
                    field=difference["field"],
                    details={
                        "reason": "first_replay_mismatch",
                        "batch_id": difference["batch_id"],
                        "window_id": difference["window_id"],
                        "expected": difference["expected"],
                        "actual": difference["actual"],
                        "ignored_jitter_fields": sorted(ignore_fields),
                    },
                )
        return {
            "schema": "speaker-diag-replay-verification-v1",
            "verified_batches": len(order),
            "first_mismatch": None,
            "ignored_jitter_fields": sorted(ignore_fields),
            "order": order,
        }

    def content_manifest(self) -> dict[str, Any]:
        connection = _connect(self.path)
        try:
            meta = self._meta(connection)
            rows = connection.execute(
                "SELECT batch_id, ordinal, worker_id, param_version, source_format, receipt_sha256, attribution_digest, content_digest FROM batches ORDER BY ordinal"
            ).fetchall()
            unknown = connection.execute(
                "SELECT batch_id, expected_ordinal FROM unknown_batches ORDER BY expected_ordinal"
            ).fetchall()
            connection.execute("PRAGMA wal_checkpoint(FULL)")
        finally:
            connection.close()
        database_sha = sha256_hex(self.path.read_bytes())
        return {
            "schema": "speaker-diag-content-manifest-v1",
            "ledger_schema_version": int(meta["ledger_schema_version"]),
            "session_version": int(meta["session_version"]),
            "spec_digest": meta["spec_digest"],
            "chain_digest": meta["chain_digest"],
            "batch_count": int(meta["batch_count"]),
            "database_sha256": database_sha,
            "batches": [dict(row) for row in rows],
            "unknown_batches": [dict(row) for row in unknown],
        }

    def write_content_manifest(self, path: str | Path) -> dict[str, Any]:
        manifest = self.content_manifest()
        manifest["manifest_digest"] = canonical_sha256(
            {key: value for key, value in manifest.items() if key not in ("manifest_digest", "database_sha256")}
        )
        atomic_write_json(path, manifest)
        return manifest

    def export_snapshot(self, path: str | Path) -> dict[str, Any]:
        rows = self.batch_rows()
        connection = _connect(self.path)
        try:
            chain_digest = self._meta(connection)["chain_digest"]
        finally:
            connection.close()
        core = {
            "schema": "speaker-diag-session-snapshot-v1",
            "ledger_schema_version": LEDGER_SCHEMA_VERSION,
            "session_version": self.session_version,
            "spec_digest": self.spec_digest,
            "source_snapshot": dict(self.session_snapshot),
            "batches": rows,
            "chain_digest": chain_digest,
        }
        core["content_digest"] = canonical_sha256(core)
        atomic_write_json(path, core)
        return core

    @classmethod
    def restore_snapshot(
        cls,
        database_path: str | Path,
        snapshot_path: str | Path,
    ) -> "Ledger":
        snapshot = json.loads(Path(snapshot_path).read_text(encoding="utf-8"))
        if snapshot.get("schema") != "speaker-diag-session-snapshot-v1":
            raise DiagnosticError("E_REPLAY", "unrecognized session snapshot schema", field="schema")
        if int(snapshot.get("ledger_schema_version", -1)) != LEDGER_SCHEMA_VERSION:
            raise DiagnosticError(
                "E_REPLAY",
                "snapshot ledger schema version is unsupported",
                field="ledger_schema_version",
            )
        digest = snapshot.get("content_digest")
        core = {key: value for key, value in snapshot.items() if key != "content_digest"}
        if digest != canonical_sha256(core):
            raise DiagnosticError("E_REPLAY", "snapshot content digest mismatch", field="content_digest")
        session_version = int(snapshot["session_version"])
        spec_digest = str(snapshot["spec_digest"])
        ledger = cls(
            database_path,
            session_version=session_version,
            spec_digest=spec_digest,
            session_snapshot=snapshot.get("source_snapshot", {}),
            allow_replace=True,
            reset=True,
        )
        connection = _connect(ledger.path)
        try:
            connection.execute("BEGIN IMMEDIATE")
            for table in ("windows", "batches", "unknown_batches"):
                connection.execute(f"DELETE FROM {table}")
            for batch in sorted(snapshot["batches"], key=lambda item: int(item["ordinal"])):
                connection.execute(
                    """
                    INSERT INTO batches(
                        batch_id, ordinal, worker_id, param_version, source_format,
                        receipt_json, receipt_sha256, attribution_digest, participant_json,
                        elapsed_ms, content_digest, committed_ns
                    ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        batch["batch_id"],
                        batch["ordinal"],
                        batch["worker_id"],
                        batch["param_version"],
                        batch["source_format"],
                        batch["receipt_json"],
                        batch["receipt_sha256"],
                        batch["attribution_digest"],
                        batch["participant_json"],
                        batch["elapsed_ms"],
                        batch["content_digest"],
                        batch["committed_ns"],
                    ),
                )
                for window in batch["windows"]:
                    connection.execute(
                        """
                        INSERT INTO windows(
                            batch_id, window_id, stable_rank, attribution, duplicate_of,
                            truth_speaker, truth_source, overlap_bps, active_speakers_json,
                            score_margin, confidence_q12
                        ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            batch["batch_id"],
                            window["window_id"],
                            window["stable_rank"],
                            window["attribution"],
                            window["duplicate_of"],
                            window["truth_speaker"],
                            window["truth_source"],
                            window["overlap_bps"],
                            window["active_speakers_json"],
                            window["score_margin"],
                            window["confidence_q12"],
                        ),
                    )
            batch_count = len(snapshot["batches"])
            connection.execute("UPDATE meta SET value=? WHERE key='chain_digest'", (snapshot["chain_digest"],))
            connection.execute("UPDATE meta SET value=? WHERE key='batch_count'", (str(batch_count),))
            connection.execute("COMMIT")
        except BaseException:
            try:
                connection.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            connection.close()
        return ledger
