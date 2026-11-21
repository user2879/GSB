"""Transactional SQLite decision ledger with crash recovery semantics."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .canonical import canonical_json, digest
from .errors import LedgerError
from .generator import DeterministicUdpGenerator
from .models import DecisionEvent, PolicySet
from .policy import cascade_digest, policy_fingerprint, shaper_digest


LEDGER_FORMAT_VERSION = 1


@dataclass(frozen=True)
class LedgerRecovery:
    cursor_seq: int
    unknown_packet_ids: tuple[str, ...]
    unknown_batch_no: int | None
    requires_rerun: bool
    digest_verified: bool
    committed_batches: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "cursor_seq": self.cursor_seq,
            "unknown_packet_ids": list(self.unknown_packet_ids),
            "unknown_batch_no": self.unknown_batch_no,
            "requires_rerun": self.requires_rerun,
            "digest_verified": self.digest_verified,
            "committed_batches": self.committed_batches,
        }


@dataclass(frozen=True)
class LedgerProgress:
    cursor_seq: int
    committed_batches: int
    next_event_seq: int
    state: dict[str, Any] | None
    summary: dict[str, Any] | None
    content_digest: str


class DecisionLedger:
    """SQLite journal whose batch commits are atomic and digest-verified."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.connection = sqlite3.connect(
                str(self.path), isolation_level=None, timeout=10.0
            )
        except sqlite3.Error as exc:
            raise LedgerError(f"cannot open ledger {self.path}: {exc}") from exc
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._initialized = False
        self.session_signature = ""

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS progress (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                cursor_seq INTEGER NOT NULL,
                committed_batches INTEGER NOT NULL,
                next_event_seq INTEGER NOT NULL,
                state_json TEXT,
                summary_json TEXT,
                content_digest TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS decisions (
                event_seq INTEGER PRIMARY KEY,
                batch_no INTEGER NOT NULL,
                packet_id TEXT NOT NULL,
                packet_seq INTEGER NOT NULL,
                stage_id TEXT NOT NULL,
                kind TEXT NOT NULL,
                decision TEXT NOT NULL,
                time_tick INTEGER NOT NULL,
                time_ns INTEGER NOT NULL,
                tokens_before INTEGER NOT NULL,
                tokens_after INTEGER NOT NULL,
                queue_len_before INTEGER NOT NULL,
                queue_len_after INTEGER NOT NULL,
                delay_ticks INTEGER NOT NULL,
                release_tick INTEGER,
                forwarded_to_json TEXT NOT NULL,
                reason TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS decisions_packet_stage
                ON decisions(packet_id, stage_id, event_seq);
            CREATE TABLE IF NOT EXISTS inflight (
                batch_no INTEGER PRIMARY KEY,
                packet_ids_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS unknown_packets (
                packet_id TEXT PRIMARY KEY,
                batch_no INTEGER NOT NULL,
                reason TEXT NOT NULL,
                resolved INTEGER NOT NULL DEFAULT 0,
                resolved_batch INTEGER
            );
            CREATE TABLE IF NOT EXISTS checkpoints (
                batch_no INTEGER PRIMARY KEY,
                cursor_seq INTEGER NOT NULL,
                next_event_seq INTEGER NOT NULL,
                state_json TEXT NOT NULL,
                summary_json TEXT,
                content_digest TEXT NOT NULL
            );
            """
        )

    def initialize(
        self,
        *,
        policy: PolicySet,
        scale: int,
        generator: DeterministicUdpGenerator,
    ) -> None:
        self._create_schema()
        signature = digest(
            {
                "format_version": LEDGER_FORMAT_VERSION,
                "policy_version": policy.policy_version,
                "policy_fingerprint": policy_fingerprint(policy),
                "shaper_digest": shaper_digest(policy),
                "cascade_digest": cascade_digest(policy),
                "scale": scale,
                "generator_digest": generator.sequence_digest,
            }
        )
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key='session_signature'"
        ).fetchone()
        if row is None:
            initial_state = None
            initial_digest = self._compute_digest([], 0, initial_state)
            with self._transaction():
                self.connection.execute(
                    "INSERT INTO meta(key, value) VALUES(?, ?)",
                    ("format_version", canonical_json(LEDGER_FORMAT_VERSION)),
                )
                self.connection.execute(
                    "INSERT INTO meta(key, value) VALUES(?, ?)",
                    ("policy_version", canonical_json(policy.policy_version)),
                )
                self.connection.execute(
                    "INSERT INTO meta(key, value) VALUES(?, ?)",
                    ("scale", canonical_json(scale)),
                )
                self.connection.execute(
                    "INSERT INTO meta(key, value) VALUES(?, ?)",
                    ("generator_digest", canonical_json(generator.sequence_digest)),
                )
                self.connection.execute(
                    "INSERT INTO meta(key, value) VALUES(?, ?)",
                    ("session_signature", canonical_json(signature)),
                )
                self.connection.execute(
                    "INSERT INTO progress(id, cursor_seq, committed_batches, "
                    "next_event_seq, state_json, summary_json, content_digest) "
                    "VALUES(1, 0, 0, 1, NULL, NULL, ?)",
                    (initial_digest,),
                )
            self.connection.execute(f"PRAGMA user_version={LEDGER_FORMAT_VERSION}")
        else:
            if json.loads(row["value"]) != signature:
                raise LedgerError(
                    "ledger session signature differs from policy/scale/generator"
                )
            format_row = self.connection.execute(
                "SELECT value FROM meta WHERE key='format_version'"
            ).fetchone()
            if format_row is None or json.loads(format_row["value"]) != LEDGER_FORMAT_VERSION:
                raise LedgerError("unsupported ledger format version")
        self._initialized = True
        self.session_signature = signature
        self.verify()

    def __enter__(self) -> "DecisionLedger":
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def close(self) -> None:
        try:
            self.connection.close()
        except sqlite3.Error:
            pass

    class _Tx:
        def __init__(self, ledger: "DecisionLedger") -> None:
            self.ledger = ledger

        def __enter__(self) -> "DecisionLedger._Tx":
            self.ledger.connection.execute("BEGIN IMMEDIATE")
            return self

        def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
            if exc_type is None:
                self.ledger.connection.execute("COMMIT")
            else:
                self.ledger.connection.execute("ROLLBACK")

    def _transaction(self) -> "DecisionLedger._Tx":
        return DecisionLedger._Tx(self)

    def _decision_dicts(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM decisions ORDER BY event_seq"
        ).fetchall()
        return [self._row_to_dict(row) for row in rows]

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        value = dict(row)
        value["forwarded_to"] = json.loads(value.pop("forwarded_to_json"))
        return value

    def _compute_digest(
        self,
        decision_dicts: Iterable[Mapping[str, Any]],
        cursor_seq: int,
        state: Mapping[str, Any] | None,
    ) -> str:
        digest_state = None if state is None else dict(state)
        if digest_state is not None:
            # The heap tie-break counter is intentionally non-semantic and can
            # differ after a crash restore without changing any decision.
            digest_state.pop("event_order", None)
        return digest(
            {
                "format_version": LEDGER_FORMAT_VERSION,
                "cursor_seq": cursor_seq,
                "decisions": list(decision_dicts),
                "state": digest_state,
            }
        )

    def progress(self) -> LedgerProgress:
        row = self.connection.execute("SELECT * FROM progress WHERE id=1").fetchone()
        if row is None:
            raise LedgerError("ledger has not been initialized")
        return LedgerProgress(
            cursor_seq=int(row["cursor_seq"]),
            committed_batches=int(row["committed_batches"]),
            next_event_seq=int(row["next_event_seq"]),
            state=None if row["state_json"] is None else json.loads(row["state_json"]),
            summary=None if row["summary_json"] is None else json.loads(row["summary_json"]),
            content_digest=str(row["content_digest"]),
        )

    def verify(self) -> bool:
        if not self._initialized:
            raise LedgerError("ledger has not been initialized")
        progress = self.progress()
        expected = self._compute_digest(
            self._decision_dicts(), progress.cursor_seq, progress.state
        )
        if expected != progress.content_digest:
            raise LedgerError(
                "ledger content digest mismatch: committed content is not trustworthy"
            )
        return True

    def begin_batch(self, batch_no: int, packet_ids: Iterable[str]) -> None:
        encoded = canonical_json(sorted(packet_ids))
        with self._transaction():
            self.connection.execute(
                "INSERT INTO inflight(batch_no, packet_ids_json) VALUES(?, ?) "
                "ON CONFLICT(batch_no) DO UPDATE SET packet_ids_json=excluded.packet_ids_json",
                (batch_no, encoded),
            )

    def commit_batch(
        self,
        *,
        batch_no: int,
        events: Iterable[DecisionEvent],
        state: Mapping[str, Any],
        cursor_seq: int,
        next_event_seq: int,
        summary: Mapping[str, Any] | None,
        packet_ids: Iterable[str],
    ) -> str:
        event_list = sorted(events, key=lambda event: event.event_seq)
        batch_packet_ids = sorted(set(packet_ids))
        with self._transaction():
            for event in event_list:
                self._insert_event(event)
            for packet_id in batch_packet_ids:
                self.connection.execute(
                    "UPDATE unknown_packets SET resolved=1, resolved_batch=? "
                    "WHERE packet_id=?",
                    (batch_no, packet_id),
                )
            all_decisions = self._decision_dicts()
            state_dict = dict(state)
            summary_dict = None if summary is None else dict(summary)
            new_digest = self._compute_digest(all_decisions, cursor_seq, state_dict)
            self.connection.execute(
                "INSERT INTO checkpoints(batch_no, cursor_seq, next_event_seq, "
                "state_json, summary_json, content_digest) VALUES(?, ?, ?, ?, ?, ?)",
                (
                    batch_no,
                    cursor_seq,
                    next_event_seq,
                    canonical_json(state_dict),
                    None if summary_dict is None else canonical_json(summary_dict),
                    new_digest,
                ),
            )
            self.connection.execute(
                "UPDATE progress SET cursor_seq=?, committed_batches=?, "
                "next_event_seq=?, state_json=?, summary_json=?, content_digest=? "
                "WHERE id=1",
                (
                    cursor_seq,
                    batch_no + 1,
                    next_event_seq,
                    canonical_json(state_dict),
                    None if summary_dict is None else canonical_json(summary_dict),
                    new_digest,
                ),
            )
            self.connection.execute("DELETE FROM inflight WHERE batch_no=?", (batch_no,))
        return new_digest

    def _insert_event(self, event: DecisionEvent) -> None:
        values = event.as_dict()
        self.connection.execute(
            "INSERT INTO decisions(event_seq, batch_no, packet_id, packet_seq, "
            "stage_id, kind, decision, time_tick, time_ns, tokens_before, "
            "tokens_after, queue_len_before, queue_len_after, delay_ticks, "
            "release_tick, forwarded_to_json, reason) "
            "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                values["event_seq"],
                values["batch_no"],
                values["packet_id"],
                values["packet_seq"],
                values["stage_id"],
                values["kind"],
                values["decision"],
                values["time_tick"],
                values["time_ns"],
                values["tokens_before"],
                values["tokens_after"],
                values["queue_len_before"],
                values["queue_len_after"],
                values["delay_ticks"],
                values["release_tick"],
                canonical_json(values["forwarded_to"]),
                values["reason"],
            ),
        )

    def restore_from_snapshot(self, ledger_payload: Mapping[str, Any]) -> LedgerProgress:
        """Atomically replace all local ledger content from a snapshot ledger.

        This is a replacement primitive, never a merge. The snapshot's internal
        ledger digest is recomputed from the imported decisions, cursor and state
        before anything is committed.
        """

        if int(ledger_payload.get("format_version", -1)) != LEDGER_FORMAT_VERSION:
            raise LedgerError("unsupported snapshot ledger format version")
        cursor = int(ledger_payload.get("cursor_seq", 0))
        committed_batches = int(ledger_payload.get("committed_batches", 0))
        next_event_seq = int(ledger_payload.get("next_event_seq", 1))
        state = ledger_payload.get("state")
        summary = ledger_payload.get("summary")
        decision_values = list(ledger_payload.get("decisions", []))
        events = [DecisionEvent.from_dict(dict(item)) for item in decision_values]
        expected_digest = self._compute_digest(decision_values, cursor, state)
        recorded_digest = str(ledger_payload.get("content_digest", ""))
        if expected_digest != recorded_digest:
            raise LedgerError("snapshot ledger content digest mismatch")
        unknown_rows = list(ledger_payload.get("unknown_unresolved", []))
        with self._transaction():
            self.connection.execute("DELETE FROM decisions")
            self.connection.execute("DELETE FROM checkpoints")
            self.connection.execute("DELETE FROM inflight")
            self.connection.execute("DELETE FROM unknown_packets")
            for event in events:
                self._insert_event(event)
            for item in unknown_rows:
                self.connection.execute(
                    "INSERT INTO unknown_packets(packet_id, batch_no, reason, resolved) "
                    "VALUES(?, ?, ?, 0)",
                    (
                        str(item["packet_id"]),
                        int(item.get("batch_no", 0)),
                        str(item.get("reason", "restored unresolved unknown packet")),
                    ),
                )
            state_json = None if state is None else canonical_json(state)
            summary_json = None if summary is None else canonical_json(summary)
            self.connection.execute(
                "UPDATE progress SET cursor_seq=?, committed_batches=?, "
                "next_event_seq=?, state_json=?, summary_json=?, content_digest=? "
                "WHERE id=1",
                (
                    cursor,
                    committed_batches,
                    next_event_seq,
                    state_json,
                    summary_json,
                    expected_digest,
                ),
            )
            if committed_batches > 0:
                self.connection.execute(
                    "INSERT INTO checkpoints(batch_no, cursor_seq, next_event_seq, "
                    "state_json, summary_json, content_digest) VALUES(?, ?, ?, ?, ?, ?)",
                    (
                        committed_batches - 1,
                        cursor,
                        next_event_seq,
                        state_json or "null",
                        summary_json,
                        expected_digest,
                    ),
                )
        self.verify()
        return self.progress()

    def recover(self) -> LedgerRecovery:
        """Mark the durable in-flight batch as unknown and require a rerun.

        A hard process kill cannot execute cleanup code. The marker is committed
        before processing, while decisions are committed only after the entire
        batch. On reopen any surviving marker therefore identifies packets whose
        uncommitted decisions are unknowable and must be regenerated.
        """

        self.verify()
        progress = self.progress()
        inflight_rows = self.connection.execute(
            "SELECT batch_no, packet_ids_json FROM inflight ORDER BY batch_no"
        ).fetchall()
        unknown_ids: list[str] = []
        unknown_batch: int | None = None
        if inflight_rows:
            with self._transaction():
                for row in inflight_rows:
                    packet_ids = json.loads(row["packet_ids_json"])
                    unknown_ids.extend(packet_ids)
                    unknown_batch = max(
                        unknown_batch if unknown_batch is not None else int(row["batch_no"]),
                        int(row["batch_no"]),
                    )
                    for packet_id in packet_ids:
                        self.connection.execute(
                            "INSERT INTO unknown_packets(packet_id, batch_no, reason, resolved) "
                            "VALUES(?, ?, ?, 0) "
                            "ON CONFLICT(packet_id) DO UPDATE SET "
                            "batch_no=excluded.batch_no, reason=excluded.reason, resolved=0, "
                            "resolved_batch=NULL",
                            (
                                packet_id,
                                int(row["batch_no"]),
                                "process ended before batch decision transaction committed",
                            ),
                        )
                    self.connection.execute(
                        "DELETE FROM inflight WHERE batch_no=?", (int(row["batch_no"]),)
                    )
        unresolved_rows = self.connection.execute(
            "SELECT packet_id, batch_no FROM unknown_packets WHERE resolved=0 "
            "ORDER BY batch_no, packet_id"
        ).fetchall()
        for row in unresolved_rows:
            if row["packet_id"] not in unknown_ids:
                unknown_ids.append(str(row["packet_id"]))
            if unknown_batch is None:
                unknown_batch = int(row["batch_no"])
        return LedgerRecovery(
            cursor_seq=progress.cursor_seq,
            unknown_packet_ids=tuple(sorted(set(unknown_ids))),
            unknown_batch_no=unknown_batch,
            requires_rerun=bool(unknown_ids),
            digest_verified=True,
            committed_batches=progress.committed_batches,
        )

    def decisions(self) -> tuple[DecisionEvent, ...]:
        try:
            rows = self.connection.execute(
                "SELECT * FROM decisions ORDER BY event_seq"
            ).fetchall()
        except sqlite3.Error as exc:
            raise LedgerError(f"cannot read decisions: {exc}") from exc
        return tuple(DecisionEvent.from_dict(self._row_to_dict(row)) for row in rows)

    def export_decisions(self) -> list[dict[str, Any]]:
        return [event.as_dict() for event in self.decisions()]

    def unknown_packets(self, *, unresolved_only: bool = True) -> tuple[dict[str, Any], ...]:
        where = " WHERE resolved=0" if unresolved_only else ""
        rows = self.connection.execute(
            "SELECT packet_id, batch_no, reason, resolved, resolved_batch "
            "FROM unknown_packets" + where + " ORDER BY batch_no, packet_id"
        ).fetchall()
        return tuple(dict(row) for row in rows)

    def meta_dict(self) -> dict[str, Any]:
        rows = self.connection.execute("SELECT key, value FROM meta ORDER BY key").fetchall()
        return {str(row["key"]): json.loads(row["value"]) for row in rows}
