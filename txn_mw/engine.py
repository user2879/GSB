"""Transactional message state machine and background query driver."""

from __future__ import annotations

import ipaddress
import json
import math
import os
import socket
import sqlite3
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable

from . import FORMAT_VERSION
from .canonical import body_digest, canonical_json, sha256_hex
from .codec import decode_record, validate_sequence
from .errors import ConflictError, DuplicateTxn, MWError, ProtocolError, StateError
from .logging import NullLogger, StructuredLogger
from .store import Store


MONO_MS = 1_000_000


@dataclass
class EngineConfig:
    query_budget_ms: int = 250
    pull_budget_ms: int = 200
    query_default_max: int = 3
    query_retry_ms: int = 100
    query_tick_ms: int = 20
    query_workers: int = 8
    idem_max_entries: int = 1024
    idem_ttl_ms: int = 60_000
    max_line_bytes: int = 8 * 1024 * 1024

    def validate(self) -> None:
        for name in ("query_budget_ms", "pull_budget_ms", "query_default_max", "query_retry_ms", "query_tick_ms", "query_workers", "idem_max_entries", "idem_ttl_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")


def _now() -> int:
    return time.monotonic_ns()


def _mono_ms(ns: int) -> int:
    return ns // MONO_MS


def _duration_us(start_ns: int, end_ns: int) -> int:
    return max(0, (end_ns - start_ns) // 1_000)


def _as_int(value: Any, path: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolError("field must be an integer", path)
    if positive and value <= 0:
        raise ProtocolError("field must be positive", path)
    return value


def _as_str(value: Any, path: str, *, max_len: int = 512) -> str:
    if not isinstance(value, str) or not value or len(value) > max_len:
        raise ProtocolError(f"field must be a non-empty string of at most {max_len} chars", path)
    return value


def _loopback_host(value: Any) -> str:
    host = _as_str(value, "$.query.host", max_len=255)
    if host == "localhost":
        return "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ProtocolError("query host must be a loopback IP address", "$.query.host") from exc
    if not address.is_loopback:
        raise ProtocolError("query host must be loopback", "$.query.host")
    return str(address)


@dataclass
class TxState:
    txn_id: str
    version: int
    state: str
    disposition: str
    body: Any
    body_json: str
    body_digest: str
    half_at_wall_ms: int
    deadline_mono_ns: int
    query_host: str
    query_port: int
    query_max: int
    query_count: int = 0
    first_query_mono_ns: int | None = None
    last_query_mono_ns: int | None = None
    next_query_mono_ns: int | None = None
    query_inflight: bool = False
    last_query_result: str | None = None
    ack_sent: bool = False
    pending_command: str | None = None
    pending_command_seq: int | None = None
    effect_id: str | None = None
    consumed_at_mono_ns: int | None = None
    terminal_reason: str | None = None
    conflict_reason: str | None = None
    updated_mono_ns: int = field(default_factory=_now)

    @property
    def terminal(self) -> bool:
        return self.state in {"committable", "consumed"} or self.disposition in {"rolled_back", "manual_review"}

    @property
    def deliverable(self) -> bool:
        return self.state == "committable" and self.disposition == "committable"

    def as_row(self) -> dict[str, Any]:
        return {
            "txn_id": self.txn_id,
            "version": self.version,
            "state": self.state,
            "disposition": self.disposition,
            "body_json": self.body_json,
            "body_digest": self.body_digest,
            "half_at_wall_ms": self.half_at_wall_ms,
            "deadline_mono_ns": self.deadline_mono_ns,
            "query_host": self.query_host,
            "query_port": self.query_port,
            "query_max": self.query_max,
            "query_count": self.query_count,
            "first_query_mono_ns": self.first_query_mono_ns,
            "last_query_mono_ns": self.last_query_mono_ns,
            "next_query_mono_ns": self.next_query_mono_ns,
            "query_inflight": int(self.query_inflight),
            "last_query_result": self.last_query_result,
            "ack_sent": int(self.ack_sent),
            "pending_command": self.pending_command,
            "pending_command_seq": self.pending_command_seq,
            "effect_id": self.effect_id,
            "consumed_at_mono_ns": self.consumed_at_mono_ns,
            "terminal_reason": self.terminal_reason,
            "conflict_reason": self.conflict_reason,
            "updated_mono_ns": self.updated_mono_ns,
        }


def _tx_from_row(row: sqlite3.Row | dict[str, Any]) -> TxState:
    return TxState(
        txn_id=row["txn_id"],
        version=int(row["version"]),
        state=row["state"],
        disposition=row["disposition"],
        body=json.loads(row["body_json"]),
        body_json=row["body_json"],
        body_digest=row["body_digest"],
        half_at_wall_ms=int(row["half_at_wall_ms"]),
        deadline_mono_ns=int(row["deadline_mono_ns"]),
        query_host=row["query_host"],
        query_port=int(row["query_port"]),
        query_max=int(row["query_max"]),
        query_count=int(row["query_count"]),
        first_query_mono_ns=row["first_query_mono_ns"],
        last_query_mono_ns=row["last_query_mono_ns"],
        next_query_mono_ns=row["next_query_mono_ns"],
        query_inflight=bool(row["query_inflight"]),
        last_query_result=row["last_query_result"],
        ack_sent=bool(row["ack_sent"]),
        pending_command=row["pending_command"],
        pending_command_seq=row["pending_command_seq"],
        effect_id=row["effect_id"],
        consumed_at_mono_ns=row["consumed_at_mono_ns"],
        terminal_reason=row["terminal_reason"],
        conflict_reason=row["conflict_reason"],
        updated_mono_ns=int(row["updated_mono_ns"]),
    )


class Engine:
    def __init__(
        self,
        db_path: str | os.PathLike[str],
        *,
        config: EngineConfig | None = None,
        logger: StructuredLogger | None = None,
        start_worker: bool = True,
    ):
        self.config = config or EngineConfig()
        self.config.validate()
        self.logger = logger or NullLogger()
        self.store = Store(db_path)
        self._runtime_txns: dict[str, dict[str, Any]] = {}
        self._runtime_idempotency: dict[str, dict[str, Any]] = {}
        self._stop = threading.Event()
        self._worker: threading.Thread | None = None
        self._query_pool = ThreadPoolExecutor(
            max_workers=self.config.query_workers, thread_name_prefix="txn-query"
        )
        self._futures: set[Future[Any]] = set()
        self._futures_lock = threading.Lock()
        self.recovery_report = self.recover()
        self._refresh_runtime_cache()
        if start_worker:
            self.start()

    # ------------------------------------------------------------------
    # lifecycle and common persistence helpers
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._query_loop, name="txn-query-loop", daemon=True)
        self._worker.start()

    def shutdown(self, *, report: bool = True) -> dict[str, Any] | None:
        self._stop.set()
        self.store.lock.acquire()
        self.store.lock.release()
        if self._worker:
            self._worker.join(timeout=2.0)
        # Running queries use a finite socket timeout derived from the declared
        # query budget.  Wait for them before closing SQLite so a graceful stop
        # cannot race a query thread against a closed connection.
        self._query_pool.shutdown(wait=True, cancel_futures=True)
        if not report:
            self.store.close()
            return None
        summary = self.inventory()
        summary.update({"shutdown_report": True, "generated_at_mono_ns": _now()})
        self.store.close()
        return summary

    def _get_tx(self, conn: sqlite3.Connection, txn_id: str) -> TxState:
        row = conn.execute("SELECT * FROM txns WHERE txn_id=?", (txn_id,)).fetchone()
        if row is None:
            raise ProtocolError("transaction is not registered", "$.txn_id", {"txn_id": txn_id})
        return _tx_from_row(row)

    def _save_tx(self, conn: sqlite3.Connection, tx: TxState) -> None:
        tx.updated_mono_ns = _now()
        row = tx.as_row()
        columns = list(row)
        placeholders = ",".join("?" for _ in columns)
        updates = ",".join(f"{column}=excluded.{column}" for column in columns if column != "txn_id")
        conn.execute(
            f"INSERT INTO txns({','.join(columns)}) VALUES({placeholders}) "
            f"ON CONFLICT(txn_id) DO UPDATE SET {updates}",
            tuple(row[column] for column in columns),
        )
        if hasattr(self, "_runtime_txns"):
            self._runtime_txns[tx.txn_id] = dict(row)

    def _refresh_runtime_cache(self) -> None:
        """Reload the whole in-memory view; never merge two ledger versions."""
        with self.store.read() as conn:
            tx_rows = conn.execute("SELECT * FROM txns ORDER BY txn_id").fetchall()
            idem_rows = conn.execute("SELECT * FROM idempotency ORDER BY idem_key").fetchall()
        self._runtime_txns = {row["txn_id"]: dict(row) for row in tx_rows}
        self._runtime_idempotency = {row["idem_key"]: dict(row) for row in idem_rows}

    def replace_from_ledger(self, ledger_path: str | os.PathLike[str]) -> dict[str, Any]:
        from .ledger import replace_ledger

        with self.store.lock:
            self._runtime_txns.clear()
            self._runtime_idempotency.clear()
            result = replace_ledger(self.store, ledger_path, logger=self.logger)
            self._refresh_runtime_cache()
            result["runtime_memory_replacement"] = True
            result["runtime_transaction_count"] = len(self._runtime_txns)
            result["runtime_idempotency_count"] = len(self._runtime_idempotency)
            return result

    def inject_conflict_fault(self, txn_id: str = "__fault_conflict__") -> dict[str, Any]:
        """Inject a replayable conflict for the explicit startup fault mode.

        This is an administrative fault trigger, not the normal query path.  It
        deliberately creates both evidence items and routes them through the
        same deterministic arbitration and E_CONFLICT logging code.
        """
        try:
            self.persist_half(
                {
                    "txn_id": txn_id,
                    "version": 1,
                    "body": {"injected_fault": "query_committed_vs_rollback"},
                    "txn_timeout_ms": 60_000,
                    "query": {"host": "127.0.0.1", "port": 9},
                    "query_max": 3,
                }
            )
            self.mark_half_ack(txn_id)
        except DuplicateTxn:
            pass
        with self.store.transaction() as conn:
            tx = self._get_tx(conn, txn_id)
            if tx.terminal:
                return {"ok": True, "injected": False, "reason": "already terminal", "txn_id": txn_id}
            tx.query_inflight = True
            self._save_tx(conn, tx)
            evidence = {
                "kind": "query_committed_vs_rollback",
                "query_result": "committed",
                "command": "rollback",
                "winner": "commit_precedence",
                "injected": True,
            }
            self._record_conflict(conn, tx, evidence, winner="commit_precedence")
            tx.query_inflight = False
            self._decide(conn, tx, "commit", "conflict_commit_precedence", evidence)
            self.store.refresh_digest(conn)
        return {
            "ok": True,
            "injected": True,
            "txn_id": txn_id,
            "state": tx.state,
            "disposition": tx.disposition,
            "terminal_reason": tx.terminal_reason,
        }

    def _record_latency(
        self,
        conn: sqlite3.Connection,
        *,
        operation: str,
        txn_id: str,
        start_ns: int,
        end_ns: int,
        budget_us: int,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        duration = _duration_us(start_ns, end_ns)
        exceeded = duration > budget_us
        conn.execute(
            """INSERT INTO latency(operation,txn_id,request_id,start_mono_ns,end_mono_ns,duration_us,budget_us,exceeded)
               VALUES(?,?,?,?,?,?,?,?)""",
            (operation, txn_id, request_id, start_ns, end_ns, duration, budget_us, int(exceeded)),
        )
        record = {
            "operation": operation,
            "txn_id": txn_id,
            "request_id": request_id,
            "start_mono_ns": start_ns,
            "end_mono_ns": end_ns,
            "duration_us": duration,
            "budget_us": budget_us,
            "exceeded": exceeded,
        }
        return record

    def _emit_latency(self, record: dict[str, Any]) -> None:
        if record["exceeded"]:
            self.logger.error(
                "E_LATENCY",
                "interaction exceeded its explicit latency budget",
                operation=record["operation"],
                txn_id=record["txn_id"],
                request_id=record.get("request_id"),
                duration_us=record["duration_us"],
                budget_us=record["budget_us"],
            )

    def _append(self, conn: sqlite3.Connection, tx: TxState, event_type: str, data: dict[str, Any] | None = None) -> int:
        return self.store.append_event(
            conn,
            txn_id=tx.txn_id,
            version=tx.version,
            state=tx.state,
            body_digest=tx.body_digest,
            event_type=event_type,
            data=data,
        )

    # ------------------------------------------------------------------
    # half message ingestion
    # ------------------------------------------------------------------
    def validate_half(self, request: dict[str, Any]) -> dict[str, Any]:
        txn_id = _as_str(request.get("txn_id"), "$.txn_id", max_len=128)
        version = _as_int(request.get("version"), "$.version", positive=True)
        if "body" not in request:
            raise ProtocolError("missing mandatory field", "$.body")
        body = request["body"]
        try:
            canonical_json(body)
        except (TypeError, ValueError) as exc:
            raise ProtocolError(
                "body must be deterministic standard JSON (NaN/Infinity are not allowed)",
                "$.body",
            ) from exc
        timeout_ms = _as_int(request.get("txn_timeout_ms"), "$.txn_timeout_ms", positive=True)
        if timeout_ms > 365 * 24 * 60 * 60 * 1000:
            raise ProtocolError("txn_timeout_ms exceeds the one-year safety limit", "$.txn_timeout_ms")
        query = request.get("query")
        if not isinstance(query, dict):
            raise ProtocolError("field must be an object", "$.query")
        host = _loopback_host(query.get("host"))
        port = _as_int(query.get("port"), "$.query.port", positive=True)
        if port > 65535:
            raise ProtocolError("port must be at most 65535", "$.query.port")
        query_max = _as_int(
            request.get("query_max", self.config.query_default_max), "$.query_max", positive=True
        )
        return {
            "txn_id": txn_id,
            "version": version,
            "body": body,
            "body_json": canonical_json(body),
            "body_digest": body_digest(body),
            "timeout_ms": timeout_ms,
            "query_host": host,
            "query_port": port,
            "query_max": query_max,
        }

    def persist_half(self, request: dict[str, Any]) -> dict[str, Any]:
        """Persist a half message but do not make it query-eligible yet."""
        values = self.validate_half(request)
        now = _now()
        with self.store.transaction() as conn:
            if conn.execute("SELECT 1 FROM txns WHERE txn_id=?", (values["txn_id"],)).fetchone():
                self.logger.error(
                    "E_DUP", "transaction identifier is already registered", txn_id=values["txn_id"]
                )
                raise DuplicateTxn(values["txn_id"])
            tx = TxState(
                txn_id=values["txn_id"],
                version=values["version"],
                state="half",
                disposition="pending",
                body=values["body"],
                body_json=values["body_json"],
                body_digest=values["body_digest"],
                half_at_wall_ms=int(time.time() * 1000),
                deadline_mono_ns=now + values["timeout_ms"] * MONO_MS,
                query_host=values["query_host"],
                query_port=values["query_port"],
                query_max=values["query_max"],
                next_query_mono_ns=now + values["timeout_ms"] * MONO_MS,
                updated_mono_ns=now,
            )
            self._save_tx(conn, tx)
            seq = self.store.append_event(
                conn,
                txn_id=tx.txn_id,
                version=tx.version,
                state="half",
                body_digest=tx.body_digest,
                event_type="half",
                data={
                    "body": values["body"],
                    "timeout_ms": values["timeout_ms"],
                    "deadline_mono_ns": tx.deadline_mono_ns,
                    "query_host": tx.query_host,
                    "query_port": tx.query_port,
                    "query_max": tx.query_max,
                    "half_at_wall_ms": tx.half_at_wall_ms,
                },
                created_mono_ns=now,
            )
            self.store.refresh_digest(conn)
        return {
            "ok": True,
            "op": "half",
            "txn_id": tx.txn_id,
            "state": "half",
            "persisted": True,
            "event_seq": seq,
            "query_eligible": False,
        }

    def mark_half_ack(self, txn_id: str) -> dict[str, Any]:
        """Commit the acknowledgment marker after the response bytes were sent.

        A crash between the socket send and this marker intentionally leaves the
        half non-queryable.  This is the conservative side of the atomicity
        boundary and satisfies the no-query-before-ack rule.
        """
        now = _now()
        with self.store.transaction() as conn:
            tx = self._get_tx(conn, txn_id)
            if not tx.ack_sent:
                tx.ack_sent = True
                self._save_tx(conn, tx)
                seq = self._append(conn, tx, "half_ack", {"ack_sent": True, "acked_mono_ns": now})
                self.store.refresh_digest(conn)
            else:
                seq = None
        return {"ok": True, "txn_id": txn_id, "ack_sent": True, "event_seq": seq, "query_eligible": True}

    # ------------------------------------------------------------------
    # producer command and deterministic arbitration
    # ------------------------------------------------------------------
    def producer_command(self, request: dict[str, Any], command: str) -> dict[str, Any]:
        if command not in {"commit", "rollback"}:
            raise ProtocolError("unknown command", "$.op")
        txn_id = _as_str(request.get("txn_id"), "$.txn_id", max_len=128)
        requested_version = request.get("version")
        if requested_version is not None:
            requested_version = _as_int(requested_version, "$.version", positive=True)
        conflict_error: ConflictError | None = None
        final: dict[str, Any] | None = None
        with self.store.transaction() as conn:
            tx = self._get_tx(conn, txn_id)
            if requested_version is not None and requested_version != tx.version:
                raise ProtocolError(
                    "local transaction version does not match persisted half message",
                    "$.version",
                    {"expected": tx.version, "actual": requested_version},
                )
            seq = self._append(
                conn,
                tx,
                "command",
                {"command": command, "deferred": bool(tx.query_inflight), "request_version": requested_version},
            )
            if tx.state == "consumed":
                self._record_conflict(
                    conn,
                    tx,
                    {"kind": "command_after_terminal", "command": command, "terminal_reason": tx.terminal_reason},
                    winner="terminal_locked",
                )
                self.store.refresh_digest(conn)
                conflict_error = ConflictError(
                    "transaction is already consumed and cannot be changed",
                    {"txn_id": txn_id, "command": command, "final_state": tx.state, "winner": tx.terminal_reason},
                )
            elif tx.terminal:
                if self._command_agrees(tx, command):
                    self.store.refresh_digest(conn)
                    return {
                        "ok": True,
                        "op": command,
                        "txn_id": txn_id,
                        "deferred": False,
                        "idempotent": True,
                        "state": tx.state,
                        "disposition": tx.disposition,
                        "terminal_reason": tx.terminal_reason,
                        "event_seq": seq,
                    }
                self._record_conflict(
                    conn,
                    tx,
                    {"kind": "command_vs_terminal", "command": command, "terminal_reason": tx.terminal_reason},
                    winner="terminal_locked",
                )
                self.store.refresh_digest(conn)
                conflict_error = ConflictError(
                    "ending command conflicts with a locked terminal result",
                    {"txn_id": txn_id, "command": command, "final_state": tx.state, "winner": tx.terminal_reason},
                )
            elif tx.query_inflight:
                existing_command = tx.pending_command
                tx.pending_command = self._merge_pending_command(existing_command, command)
                tx.pending_command_seq = seq
                if existing_command is not None and existing_command != command:
                    evidence = {
                        "kind": "conflicting_commands_during_query",
                        "commands": [existing_command, command],
                        "winner": "commit_precedence",
                    }
                    self._record_conflict(conn, tx, evidence, winner="commit_precedence")
                    conflict_error = ConflictError(
                        "conflicting producer commands arrived while a query was in flight",
                        {
                            "txn_id": txn_id,
                            "commands": [existing_command, command],
                            "provisional_pending_command": tx.pending_command,
                            "winner": "commit_precedence",
                        },
                    )
                self._save_tx(conn, tx)
                self.store.refresh_digest(conn)
                if conflict_error is None:
                    return {
                        "ok": True,
                        "op": command,
                        "txn_id": txn_id,
                        "deferred": True,
                        "pending_command": tx.pending_command,
                        "state": tx.state,
                        "disposition": tx.disposition,
                        "event_seq": seq,
                    }
            else:
                self._decide(conn, tx, command, f"explicit_{command}", {"source": "producer_command", "command": command})
                self.store.refresh_digest(conn)
                final = {
                    "ok": True,
                    "op": command,
                    "txn_id": txn_id,
                    "deferred": False,
                    "state": tx.state,
                    "disposition": tx.disposition,
                    "terminal_reason": tx.terminal_reason,
                    "event_seq": seq,
                }
        if conflict_error is not None:
            raise conflict_error
        assert final is not None
        return final

    @staticmethod
    def _merge_pending_command(existing: str | None, new: str) -> str:
        # Commit has fixed precedence.  This makes conflict resolution
        # independent of the order in which two deferred commands arrived.
        if existing == "commit" or new == "commit":
            return "commit"
        return new

    @staticmethod
    def _command_agrees(tx: TxState, command: str) -> bool:
        if command == "commit":
            return tx.state in {"committable", "consumed"} or tx.disposition == "committable"
        return tx.disposition == "rolled_back"

    def _record_conflict(
        self,
        conn: sqlite3.Connection,
        tx: TxState,
        evidence: dict[str, Any],
        *,
        winner: str,
    ) -> None:
        now = _now()
        tx.conflict_reason = winner
        tx.terminal_reason = tx.terminal_reason or winner
        self._save_tx(conn, tx)
        conn.execute(
            "INSERT INTO conflicts(txn_id,evidence_json,winner,created_mono_ns) VALUES(?,?,?,?)",
            (tx.txn_id, canonical_json(evidence), winner, now),
        )
        self._append(conn, tx, "conflict", {"evidence": evidence, "winner": winner})
        self.logger.error(
            "E_CONFLICT",
            "conflicting transaction evidence was observed",
            txn_id=tx.txn_id,
            winner=winner,
            evidence=evidence,
        )

    def _decide(
        self,
        conn: sqlite3.Connection,
        tx: TxState,
        outcome: str,
        reason: str,
        evidence: dict[str, Any],
    ) -> bool:
        if outcome not in {"commit", "rollback"}:
            raise ValueError(f"invalid outcome {outcome}")
        if tx.terminal:
            agrees = self._command_agrees(tx, "commit" if outcome == "commit" else "rollback")
            if agrees:
                return True
            self._record_conflict(conn, tx, {**evidence, "attempted_outcome": outcome}, winner="terminal_locked")
            return False
        tx.query_inflight = False
        tx.pending_command = None
        tx.pending_command_seq = None
        if outcome == "commit":
            tx.state = "committable"
            tx.disposition = "committable"
            tx.terminal_reason = reason
        else:
            tx.state = "half"
            tx.disposition = "rolled_back"
            tx.terminal_reason = reason
        self._save_tx(conn, tx)
        self._append(
            conn,
            tx,
            "decision",
            {
                "outcome": outcome,
                "target_state": tx.state,
                "disposition": tx.disposition,
                "reason": reason,
                "evidence": evidence,
            },
        )
        return True

    def _mark_manual(self, conn: sqlite3.Connection, tx: TxState, reason: str) -> None:
        tx.query_inflight = False
        tx.state = "half"
        tx.disposition = "manual_review"
        tx.terminal_reason = reason
        self._save_tx(conn, tx)
        self._append(conn, tx, "manual_review", {"reason": reason, "query_count": tx.query_count})

    # ------------------------------------------------------------------
    # query loop and actual loopback query
    # ------------------------------------------------------------------
    def _query_loop(self) -> None:
        while not self._stop.is_set():
            try:
                claims, exhausted = self._claim_due_queries()
                for txn_id, attempt, started_ns in claims:
                    future = self._query_pool.submit(self._run_query, txn_id, attempt, started_ns)
                    with self._futures_lock:
                        self._futures.add(future)
                    future.add_done_callback(self._forget_future)
                for txn_id, used in exhausted:
                    self.logger.error(
                        "E_QUERY",
                        "transaction query limit exhausted; manual review required",
                        txn_id=txn_id,
                        query_count=used,
                    )
            except Exception as exc:  # keep the worker alive; errors are logged
                self.logger.error("E_INTERNAL", "query loop iteration failed", error=repr(exc))
            self._stop.wait(max(0.005, self.config.query_tick_ms / 1000))

    def _forget_future(self, future: Future[Any]) -> None:
        with self._futures_lock:
            self._futures.discard(future)

    def _claim_due_queries(self) -> tuple[list[tuple[str, int, int]], list[tuple[str, int]]]:
        now = _now()
        claims: list[tuple[str, int, int]] = []
        exhausted: list[tuple[str, int]] = []
        with self.store.transaction() as conn:
            rows = conn.execute(
                """SELECT * FROM txns
                   WHERE state='half' AND disposition='pending' AND ack_sent=1
                     AND query_inflight=0
                     AND (next_query_mono_ns IS NULL OR next_query_mono_ns<=?)
                   ORDER BY deadline_mono_ns, txn_id""",
                (now,),
            ).fetchall()
            for row in rows:
                tx = _tx_from_row(row)
                if tx.deadline_mono_ns > now:
                    continue
                if tx.query_count >= tx.query_max:
                    if tx.pending_command in {"commit", "rollback"}:
                        self._decide(
                            conn,
                            tx,
                            tx.pending_command,
                            f"deferred_{tx.pending_command}_at_query_limit",
                            {
                                "source": "producer_command",
                                "query_count": tx.query_count,
                                "query_max": tx.query_max,
                            },
                        )
                    else:
                        self._mark_manual(conn, tx, "query_limit_exhausted")
                        self._append(
                            conn,
                            tx,
                            "query_exhausted",
                            {"query_count": tx.query_count, "query_max": tx.query_max},
                        )
                        exhausted.append((tx.txn_id, tx.query_count))
                    continue
                tx.query_count += 1
                if tx.first_query_mono_ns is None:
                    tx.first_query_mono_ns = now
                tx.last_query_mono_ns = now
                tx.query_inflight = True
                tx.next_query_mono_ns = None
                self._save_tx(conn, tx)
                self._append(
                    conn,
                    tx,
                    "query_started",
                    {
                        "attempt": tx.query_count,
                        "started_mono_ns": now,
                        "first_query_mono_ns": tx.first_query_mono_ns,
                        "deadline_mono_ns": tx.deadline_mono_ns,
                    },
                )
                claims.append((tx.txn_id, tx.query_count, now))
            if claims or exhausted:
                self.store.refresh_digest(conn)
        return claims, exhausted

    def _read_query_snapshot(self, txn_id: str, attempt: int) -> dict[str, Any]:
        with self.store.read() as conn:
            row = conn.execute("SELECT * FROM txns WHERE txn_id=?", (txn_id,)).fetchone()
            if row is None:
                raise ProtocolError("transaction disappeared", "$.txn_id")
            tx = _tx_from_row(row)
            return {
                "txn_id": tx.txn_id,
                "version": tx.version,
                "host": tx.query_host,
                "port": tx.query_port,
                "attempt": attempt,
            }

    def _run_query(self, txn_id: str, attempt: int, claimed_ns: int) -> None:
        snapshot = self._read_query_snapshot(txn_id, attempt)
        result = "unknown"
        error: str | None = None
        response_txn_id: str | None = None
        # Latency and the socket timeout budget begin at the actual interaction,
        # not while this task waits in the worker queue.
        interaction_start_ns = _now()
        deadline_ns = interaction_start_ns + self.config.query_budget_ms * MONO_MS

        def remaining_timeout() -> float:
            remaining_ns = deadline_ns - _now()
            if remaining_ns <= 0:
                raise TimeoutError("query budget exhausted before the next socket operation")
            return remaining_ns / 1_000_000_000

        try:
            with socket.create_connection(
                (snapshot["host"], snapshot["port"]), timeout=remaining_timeout()
            ) as sock:
                sock.settimeout(remaining_timeout())
                request = {
                    "op": "query_txn",
                    "txn_id": txn_id,
                    "version": snapshot["version"],
                    "attempt": attempt,
                }
                sock.sendall((canonical_json(request) + "\n").encode("utf-8"))
                buffer = bytearray()
                while len(buffer) <= self.config.max_line_bytes:
                    sock.settimeout(remaining_timeout())
                    chunk = sock.recv(4096)
                    if not chunk:
                        raise ConnectionError("query endpoint closed before a complete JSON line")
                    buffer.extend(chunk)
                    if b"\n" in buffer:
                        break
                if b"\n" not in buffer:
                    raise ValueError("query response line exceeds maximum size")
                line = bytes(buffer).split(b"\n", 1)[0]
                reply = json.loads(line.decode("utf-8"))
                if not isinstance(reply, dict):
                    raise ValueError("query reply must be a JSON object")
                response_txn_id = reply.get("txn_id")
                if response_txn_id is not None and response_txn_id != txn_id:
                    raise ValueError("query reply txn_id mismatch")
                candidate = reply.get("result")
                if candidate not in {"committed", "rolled_back", "unknown"}:
                    raise ValueError("query result must be committed, rolled_back or unknown")
                result = candidate
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            result = "unknown"
        ended_ns = _now()
        self._resolve_query(
            txn_id,
            attempt,
            interaction_start_ns,
            ended_ns,
            result,
            error,
            response_txn_id,
        )

    def _resolve_query(
        self,
        txn_id: str,
        attempt: int,
        started_ns: int,
        ended_ns: int,
        result: str,
        error: str | None,
        response_txn_id: str | None,
    ) -> None:
        latency: dict[str, Any] | None = None
        conflict: dict[str, Any] | None = None
        exhausted: tuple[str, int] | None = None
        with self.store.transaction() as conn:
            tx = self._get_tx(conn, txn_id)
            was_inflight = tx.query_inflight
            tx.query_inflight = False
            tx.last_query_result = result
            tx.last_query_mono_ns = ended_ns
            duration_us = _duration_us(started_ns, ended_ns)
            self._append(
                conn,
                tx,
                "query_result" if was_inflight else "late_query_result",
                {
                    "attempt": attempt,
                    "result": result,
                    "error": error,
                    "response_txn_id": response_txn_id,
                    "started_mono_ns": started_ns,
                    "ended_mono_ns": ended_ns,
                    "duration_us": duration_us,
                },
            )
            latency = self._record_latency(
                conn,
                operation="query",
                txn_id=tx.txn_id,
                start_ns=started_ns,
                end_ns=ended_ns,
                budget_us=self.config.query_budget_ms * 1000,
            )
            if tx.terminal:
                if result in {"committed", "rolled_back"}:
                    desired = "commit" if result == "committed" else "rollback"
                    if not self._command_agrees(tx, desired):
                        conflict = {
                            "kind": "late_query_result_vs_terminal",
                            "query_result": result,
                            "terminal_reason": tx.terminal_reason,
                            "attempt": attempt,
                            "winner": "terminal_locked",
                        }
                        self._record_conflict(conn, tx, conflict, winner="terminal_locked")
                self._save_tx(conn, tx)
            else:
                pending = tx.pending_command
                if result == "committed" and pending == "rollback":
                    conflict = {
                        "kind": "query_committed_vs_rollback",
                        "query_result": "committed",
                        "command": "rollback",
                        "attempt": attempt,
                        "winner": "commit_precedence",
                    }
                    self._record_conflict(conn, tx, conflict, winner="commit_precedence")
                    self._decide(conn, tx, "commit", "conflict_commit_precedence", conflict)
                elif result == "rolled_back" and pending == "commit":
                    conflict = {
                        "kind": "query_rolled_back_vs_commit",
                        "query_result": "rolled_back",
                        "command": "commit",
                        "attempt": attempt,
                        "winner": "commit_precedence",
                    }
                    self._record_conflict(conn, tx, conflict, winner="commit_precedence")
                    self._decide(conn, tx, "commit", "conflict_commit_precedence", conflict)
                elif result == "committed":
                    self._decide(conn, tx, "commit", "query_committed", {"source": "query", "attempt": attempt})
                elif result == "rolled_back":
                    self._decide(conn, tx, "rollback", "query_rolled_back", {"source": "query", "attempt": attempt})
                elif pending in {"commit", "rollback"}:
                    self._decide(
                        conn,
                        tx,
                        pending,
                        f"deferred_{pending}_after_query_unknown",
                        {"source": "producer_command", "query_result": result, "attempt": attempt},
                    )
                elif tx.query_count >= tx.query_max:
                    self._mark_manual(conn, tx, "query_limit_exhausted")
                    self._append(
                        conn,
                        tx,
                        "query_exhausted",
                        {"query_count": tx.query_count, "query_max": tx.query_max},
                    )
                    exhausted = (tx.txn_id, tx.query_count)
                else:
                    tx.next_query_mono_ns = ended_ns + self.config.query_retry_ms * MONO_MS
                    self._save_tx(conn, tx)
            self.store.refresh_digest(conn)
        if latency:
            self._emit_latency(latency)
        if exhausted:
            self.logger.error(
                "E_QUERY",
                "transaction query limit exhausted; manual review required",
                txn_id=exhausted[0],
                query_count=exhausted[1],
            )

    # ------------------------------------------------------------------
    # idempotent pull and dual-dimension key window
    # ------------------------------------------------------------------
    @staticmethod
    def _effect_id(tx: TxState) -> str:
        return "eff_" + sha256_hex(f"{tx.txn_id}\0{tx.version}\0{tx.body_digest}".encode("ascii"))[:32]

    def pull(self, request: dict[str, Any], *, started_ns: int | None = None) -> dict[str, Any]:
        started_ns = started_ns or _now()
        txn_id = _as_str(request.get("txn_id"), "$.txn_id", max_len=128)
        consumer_id = _as_str(request.get("consumer_id"), "$.consumer_id", max_len=128)
        request_id = request.get("request_id")
        if request_id is None:
            request_id = "pull_" + uuid.uuid4().hex
        else:
            request_id = _as_str(request_id, "$.request_id", max_len=160)
        pending_created = False
        try:
            with self.store.transaction() as conn:
                tx = self._get_tx(conn, txn_id)
                try:
                    conn.execute(
                        "INSERT INTO pulls(request_id,txn_id,consumer_id,started_mono_ns,status) VALUES(?,?,?,?, 'pending')",
                        (request_id, txn_id, consumer_id, started_ns),
                    )
                except sqlite3.IntegrityError as exc:
                    raise ProtocolError("duplicate pull request_id", "$.request_id") from exc
                self._append(
                    conn,
                    tx,
                    "pull_started",
                    {"request_id": request_id, "consumer_id": consumer_id, "started_mono_ns": started_ns},
                )
                self.store.refresh_digest(conn)
            pending_created = True
            result, latency = self._execute_pull(
                txn_id=txn_id,
                consumer_id=consumer_id,
                request_id=request_id,
                started_ns=started_ns,
            )
        except Exception as exc:
            if pending_created:
                self._finish_pull_error(
                    request_id,
                    txn_id,
                    consumer_id,
                    started_ns,
                    getattr(exc, "code", "E_INTERNAL"),
                )
            raise
        self._emit_latency(latency)
        return result

    def _finish_pull_error(
        self,
        request_id: str,
        txn_id: str,
        consumer_id: str,
        started_ns: int,
        error_code: str,
    ) -> None:
        """Best-effort completion record; the message itself is never partially consumed."""
        latency: dict[str, Any] | None = None
        try:
            with self.store.transaction() as conn:
                end = _now()
                conn.execute(
                    "UPDATE pulls SET completed_mono_ns=?,status='failed',result_json=? WHERE request_id=?",
                    (end, canonical_json({"ok": False}), request_id),
                )
                row = conn.execute("SELECT * FROM txns WHERE txn_id=?", (txn_id,)).fetchone()
                if row:
                    tx = _tx_from_row(row)
                    self._append(
                        conn,
                        tx,
                        "pull_failed",
                        {
                            "request_id": request_id,
                            "consumer_id": consumer_id,
                            "error_code": error_code,
                        },
                    )
                    recorded = self._record_latency(
                        conn,
                        operation="pull",
                        txn_id=txn_id,
                        start_ns=started_ns,
                        end_ns=end,
                        budget_us=self.config.pull_budget_ms * 1000,
                        request_id=request_id,
                    )
                self.store.refresh_digest(conn)
                latency = recorded if row else None
        except Exception:
            pass
        if latency is not None:
            self._emit_latency(latency)

    def _execute_pull(
        self,
        *,
        txn_id: str,
        consumer_id: str,
        request_id: str,
        started_ns: int,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        evictions: list[dict[str, Any]] = []
        with self.store.transaction() as conn:
            tx = self._get_tx(conn, txn_id)
            if tx.state == "half":
                raise StateError(
                    "half message is not deliverable",
                    {"txn_id": txn_id, "state": tx.state, "disposition": tx.disposition},
                )
            existing = conn.execute(
                "SELECT * FROM idempotency WHERE idem_key=?", (txn_id,)
            ).fetchone()
            if tx.state == "consumed":
                if not tx.effect_id:
                    raise StateError("consumed transaction is missing its effect identifier", {"txn_id": txn_id})
                effect_id = tx.effect_id
                duplicate = True
                consumption_count = 1
                access_ns = _now()
                if existing is not None:
                    conn.execute(
                        "UPDATE idempotency SET last_access_mono_ns=? WHERE idem_key=?",
                        (access_ns, txn_id),
                    )
                    self._runtime_idempotency[txn_id]["last_access_mono_ns"] = access_ns
                self._append(
                    conn,
                    tx,
                    "pull_duplicate",
                    {
                        "request_id": request_id,
                        "consumer_id": consumer_id,
                        "effect_id": effect_id,
                        "idempotency_window_hit": existing is not None,
                        "accessed_mono_ns": access_ns,
                    },
                )
            else:
                if not tx.deliverable:
                    raise StateError("transaction is not committable", {"txn_id": txn_id, "state": tx.state})
                if existing is not None:
                    # A repair path for a crash after the ledger key was written
                    # but before the transaction row reflected consumption.
                    effect_id = existing["effect_id"]
                    duplicate = True
                else:
                    effect_id = self._effect_id(tx)
                    duplicate = False
                now = _now()
                tx.state = "consumed"
                tx.disposition = "consumed"
                tx.effect_id = effect_id
                tx.consumed_at_mono_ns = now
                tx.terminal_reason = tx.terminal_reason or "consumed"
                self._save_tx(conn, tx)
                result = {
                    "effect_id": effect_id,
                    "consumer_id": consumer_id,
                    "txn_id": txn_id,
                    "version": tx.version,
                    "body_digest": tx.body_digest,
                    "created_mono_ns": now,
                    "last_access_mono_ns": now,
                }
                if existing is None:
                    conn.execute(
                        """INSERT INTO idempotency(idem_key,txn_id,effect_id,consumer_id,created_mono_ns,last_access_mono_ns,consume_count,result_json)
                           VALUES(?,?,?,?,?,?,1,?)""",
                        (txn_id, txn_id, effect_id, consumer_id, now, now, canonical_json(result)),
                    )
                    self._runtime_idempotency[txn_id] = {
                        "idem_key": txn_id,
                        "txn_id": txn_id,
                        "effect_id": effect_id,
                        "consumer_id": consumer_id,
                        "created_mono_ns": now,
                        "last_access_mono_ns": now,
                        "consume_count": 1,
                        "result_json": canonical_json(result),
                    }
                self._append(
                    conn,
                    tx,
                    "consume",
                    {
                        "request_id": request_id,
                        "consumer_id": consumer_id,
                        "effect_id": effect_id,
                        "result": result,
                        "duplicate_repair": duplicate,
                    },
                )
                duplicate = False
                consumption_count = 1
            evictions = self._evict_idempotency(conn, now_ns=_now())
            end_ns = _now()
            response = {
                "ok": True,
                "op": "pull",
                "txn_id": txn_id,
                "consumer_id": consumer_id,
                "request_id": request_id,
                "effect_id": effect_id,
                "duplicate": duplicate,
                "consumption_count": consumption_count,
                "state": "consumed",
                "idempotency_window_hit": existing is not None,
            }
            conn.execute(
                """UPDATE pulls SET completed_mono_ns=?,status='completed',effect_id=?,duplicate=?,result_json=?
                   WHERE request_id=?""",
                (end_ns, effect_id, int(duplicate), canonical_json(response), request_id),
            )
            latency = self._record_latency(
                conn,
                operation="pull",
                txn_id=txn_id,
                start_ns=started_ns,
                end_ns=end_ns,
                budget_us=self.config.pull_budget_ms * 1000,
                request_id=request_id,
            )
            self.store.refresh_digest(conn)
        for eviction in evictions:
            self.logger.info(
                "idempotency key evicted",
                txn_id=eviction["txn_id"],
                reason=eviction["reason"],
                detail=eviction["detail"],
            )
        return response, latency

    def _evict_idempotency(self, conn: sqlite3.Connection, *, now_ns: int) -> list[dict[str, Any]]:
        evicted: list[dict[str, Any]] = []
        ttl_ns = self.config.idem_ttl_ms * MONO_MS
        expired = conn.execute(
            "SELECT * FROM idempotency WHERE last_access_mono_ns < ? ORDER BY last_access_mono_ns, idem_key",
            (now_ns - ttl_ns,),
        ).fetchall()
        for row in expired:
            evicted.append(
                self._evict_one(
                    conn,
                    row,
                    "ttl_exceeded",
                    {"idle_ms": (now_ns - int(row["last_access_mono_ns"])) // MONO_MS, "ttl_ms": self.config.idem_ttl_ms},
                )
            )
        count = conn.execute("SELECT COUNT(*) AS n FROM idempotency").fetchone()["n"]
        while count > self.config.idem_max_entries:
            row = conn.execute(
                "SELECT * FROM idempotency ORDER BY last_access_mono_ns, idem_key LIMIT 1"
            ).fetchone()
            if row is None:
                break
            evicted.append(
                self._evict_one(
                    conn,
                    row,
                    "capacity_exceeded",
                    {"window_entries": self.config.idem_max_entries, "observed_entries": count},
                )
            )
            count -= 1
        return evicted

    def _evict_one(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        reason: str,
        detail: dict[str, Any],
    ) -> dict[str, Any]:
        now = _now()
        conn.execute("DELETE FROM idempotency WHERE idem_key=?", (row["idem_key"],))
        self._runtime_idempotency.pop(row["idem_key"], None)
        conn.execute(
            """INSERT INTO idempotency_evictions(idem_key,txn_id,effect_id,reason,detail_json,evicted_mono_ns)
               VALUES(?,?,?,?,?,?)""",
            (row["idem_key"], row["txn_id"], row["effect_id"], reason, canonical_json(detail), now),
        )
        txrow = conn.execute("SELECT * FROM txns WHERE txn_id=?", (row["txn_id"],)).fetchone()
        if txrow is not None:
            tx = _tx_from_row(txrow)
            self._append(
                conn,
                tx,
                "idempotency_evict",
                {
                    "idem_key": row["idem_key"],
                    "txn_id": row["txn_id"],
                    "effect_id": row["effect_id"],
                    "reason": reason,
                    "detail": detail,
                },
            )
        return {"txn_id": row["txn_id"], "reason": reason, "detail": detail}

    # ------------------------------------------------------------------
    # crash recovery from complete SQLite rows and the event ledger
    # ------------------------------------------------------------------
    def recover(self) -> dict[str, Any]:
        """Rebuild runtime state from valid framed events.

        This method is intentionally called before background query workers are
        started.  Incomplete queries are changed to ``unknown`` and made due
        again.  Incomplete pulls are marked ``re_pull_required``; no result is
        fabricated for a request whose response was never observed.
        """
        recovered_at = _now()
        tail_quarantine: list[dict[str, Any]] = []
        other_quarantine: list[dict[str, Any]] = []
        valid_payloads: list[dict[str, Any]] = []
        with self.store.transaction() as conn:
            stored_digest = str(self.store.get_meta("content_digest", "") or "")
            stored_version = int(self.store.get_meta("format_version", str(FORMAT_VERSION)) or FORMAT_VERSION)
            rows = conn.execute("SELECT * FROM events ORDER BY event_seq, row_id").fetchall()
            max_all_seq = max((int(row["event_seq"]) for row in rows), default=0)
            previous_global = 0
            per_txn_seq: dict[str, int] = {}
            max_seq = 0
            def has_later_decodable(start_index: int) -> bool:
                for later in rows[start_index + 1 :]:
                    try:
                        decode_record(bytes(later["record"]))
                        return True
                    except MWError:
                        continue
                return False

            for index, row in enumerate(rows):
                if not bool(row["valid"]):
                    details = {
                        "row_id": row["row_id"],
                        "event_seq": row["event_seq"],
                        "txn_id": row["txn_id"],
                        "error_code": "E_LOG",
                        "path": "$.record",
                        "error_message": "record was quarantined during an earlier recovery",
                        "index": index,
                    }
                    if has_later_decodable(index):
                        other_quarantine.append(details)
                    else:
                        tail_quarantine.append(details)
                    continue
                try:
                    payload = decode_record(bytes(row["record"]))
                    seq = int(payload["event_seq"])
                    if seq <= previous_global:
                        raise ProtocolError("global event_seq is not strictly increasing", "$.record.event_seq")
                    previous = per_txn_seq.get(payload["txn_id"])
                    if previous is not None and seq <= previous:
                        raise ProtocolError(
                            "per-transaction event_seq is not strictly increasing",
                            "$.record.event_seq",
                        )
                except MWError as exc:
                    details = {
                        "row_id": row["row_id"],
                        "event_seq": row["event_seq"],
                        "txn_id": row["txn_id"],
                        "error_code": exc.code,
                        "path": exc.path,
                        "error_message": exc.message,
                        "index": index,
                    }
                    is_tail = not has_later_decodable(index)
                    if is_tail:
                        tail_quarantine.append(details)
                    else:
                        other_quarantine.append(details)
                    self.store.quarantine_row(conn, row, exc.message, exc.path)
                    self.logger.error(
                        "E_LOG",
                        "invalid or incomplete event-log record was quarantined",
                        **details,
                    )
                    continue
                valid_payloads.append(payload)
                seq = int(payload["event_seq"])
                previous_global = seq
                max_seq = max(max_seq, seq)
                per_txn_seq[payload["txn_id"]] = seq

            try:
                validate_sequence(valid_payloads)
            except MWError as exc:
                self.logger.error(
                    "E_LOG",
                    "event sequence validation failed",
                    path=exc.path,
                    sequence_details=exc.details,
                )

            observed_digest = self.store.digest_payload(self.store.ledger_snapshot(conn))
            digest_mismatch = bool(stored_digest) and stored_digest != observed_digest
            if digest_mismatch:
                self.logger.error(
                    "E_LOG",
                    "ledger content digest mismatch was repaired from valid committed records",
                    stored_content_digest=stored_digest,
                    observed_content_digest=observed_digest,
                )

            states, idems = self._replay_valid_events(valid_payloads)
            now = _now()
            incomplete_queries: list[str] = []
            orphaned_unacked: list[str] = []
            for tx in states.values():
                if tx.query_inflight:
                    incomplete_queries.append(tx.txn_id)
                    tx.query_inflight = False
                    tx.last_query_result = "unknown"
                    tx.next_query_mono_ns = now
                if tx.state == "half" and tx.disposition == "pending" and not tx.ack_sent:
                    tx.next_query_mono_ns = None
                    orphaned_unacked.append(tx.txn_id)

            conn.execute("DELETE FROM txns")
            for tx in states.values():
                self._save_tx(conn, tx)

            conn.execute("DELETE FROM idempotency")
            for idem in idems.values():
                conn.execute(
                    """INSERT INTO idempotency(idem_key,txn_id,effect_id,consumer_id,created_mono_ns,last_access_mono_ns,consume_count,result_json)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    (
                        idem["idem_key"],
                        idem["txn_id"],
                        idem["effect_id"],
                        idem["consumer_id"],
                        int(idem["created_mono_ns"]),
                        int(idem["last_access_mono_ns"]),
                        int(idem["consume_count"]),
                        idem["result_json"],
                    ),
                )

            self.store.set_meta(conn, "next_event_seq", str(max(max_seq, max_all_seq) + 1))
            for txn_id in incomplete_queries:
                tx = states[txn_id]
                self._append(
                    conn,
                    tx,
                    "recovery_query_unknown",
                    {"reason": "query was in flight at process termination", "requeued": True},
                )
                self._save_tx(conn, tx)

            pending_pulls = conn.execute(
                "SELECT * FROM pulls WHERE status='pending' ORDER BY started_mono_ns"
            ).fetchall()
            incomplete_pulls: list[dict[str, Any]] = []
            for pull in pending_pulls:
                conn.execute(
                    "UPDATE pulls SET status='re_pull_required', result_json=? WHERE request_id=?",
                    (canonical_json({"ok": False, "code": "E_RECOVERY", "re_pull_required": True}), pull["request_id"]),
                )
                incomplete_pulls.append(
                    {
                        "request_id": pull["request_id"],
                        "txn_id": pull["txn_id"],
                        "consumer_id": pull["consumer_id"],
                        "status": "re_pull_required",
                    }
                )
                tx = states.get(pull["txn_id"])
                if tx is not None:
                    self._append(
                        conn,
                        tx,
                        "recovery_pull_required",
                        {"request_id": pull["request_id"], "consumer_id": pull["consumer_id"]},
                    )

            if stored_version != FORMAT_VERSION:
                self.logger.error(
                    "E_REPLACE",
                    "ledger format version mismatch; state was rebuilt wholesale from valid records",
                    stored_format_version=stored_version,
                    expected_format_version=FORMAT_VERSION,
                )
            rebuilt_digest = self.store.refresh_digest(conn)

        report = {
            "ok": True,
            "recovery_report": True,
            "recovered_at_mono_ns": recovered_at,
            "shutdown_report_available": False,
            "termination_report_note": (
                "A forced process termination cannot produce an orderly shutdown report; "
                "this report is reconstructed only from complete SQLite rows and valid event frames."
            ),
            "format_version": FORMAT_VERSION,
            "recovered_transactions": len(states),
            "pending_query_transactions": sorted(
                tx.txn_id
                for tx in states.values()
                if tx.state == "half" and tx.disposition == "pending" and tx.ack_sent
            ),
            "incomplete_queries_marked_unknown_and_requeued": incomplete_queries,
            "orphaned_unacked_half_messages_not_queued": orphaned_unacked,
            "incomplete_pulls_requiring_re_pull": incomplete_pulls,
            "tail_incomplete_records": tail_quarantine,
            "other_quarantined_records": other_quarantine,
            "ledger_digest": {
                "stored": stored_digest,
                "rebuilt": rebuilt_digest,
                "observed_before_replay": observed_digest,
                "mismatch_repaired": digest_mismatch,
            },
        }
        return report

    def _replay_valid_events(
        self, payloads: Iterable[dict[str, Any]]
    ) -> tuple[dict[str, TxState], dict[str, dict[str, Any]]]:
        states: dict[str, TxState] = {}
        idems: dict[str, dict[str, Any]] = {}
        evicted: set[str] = set()
        for payload in payloads:
            txn_id = payload["txn_id"]
            event_type = payload["type"]
            data = payload.get("data", {})
            if event_type == "half":
                body = data.get("body")
                body_json = canonical_json(body)
                states[txn_id] = TxState(
                    txn_id=txn_id,
                    version=int(payload["version"]),
                    state="half",
                    disposition="pending",
                    body=body,
                    body_json=body_json,
                    body_digest=payload["body_digest"],
                    half_at_wall_ms=int(data.get("half_at_wall_ms", 0)),
                    deadline_mono_ns=int(data.get("deadline_mono_ns", 0)),
                    query_host=str(data.get("query_host", "127.0.0.1")),
                    query_port=int(data.get("query_port", 1)),
                    query_max=int(data.get("query_max", self.config.query_default_max)),
                    next_query_mono_ns=int(data.get("deadline_mono_ns", 0)),
                    updated_mono_ns=_now(),
                )
                continue
            tx = states.get(txn_id)
            if tx is None:
                # An event without its durable half cannot define a transaction.
                continue
            if event_type == "half_ack":
                tx.ack_sent = True
            elif event_type == "command":
                command = data.get("command")
                already_terminal = tx.state in {"committable", "consumed"} or tx.disposition in {
                    "rolled_back",
                    "manual_review",
                }
                if command in {"commit", "rollback"} and not already_terminal:
                    tx.pending_command = self._merge_pending_command(tx.pending_command, command)
            elif event_type == "query_started":
                attempt = int(data.get("attempt", tx.query_count + 1))
                tx.query_count = max(tx.query_count, attempt)
                if tx.first_query_mono_ns is None:
                    tx.first_query_mono_ns = int(data.get("first_query_mono_ns", data.get("started_mono_ns", 0)))
                tx.last_query_mono_ns = int(data.get("started_mono_ns", 0))
                tx.query_inflight = True
            elif event_type in {"query_result", "late_query_result"}:
                tx.query_inflight = False
                tx.last_query_result = data.get("result", "unknown")
                tx.last_query_mono_ns = int(data.get("ended_mono_ns", 0))
            elif event_type == "recovery_query_unknown":
                tx.query_inflight = False
                tx.last_query_result = "unknown"
            elif event_type == "decision":
                target = data.get("target_state", payload["state"])
                tx.state = target if target in {"half", "committable", "consumed"} else "half"
                tx.disposition = data.get("disposition", tx.disposition)
                tx.terminal_reason = data.get("reason", tx.terminal_reason)
                tx.query_inflight = False
                tx.pending_command = None
                tx.pending_command_seq = None
            elif event_type == "manual_review":
                tx.state = "half"
                tx.disposition = "manual_review"
                tx.terminal_reason = data.get("reason", "manual_review")
                tx.query_inflight = False
            elif event_type == "query_exhausted":
                tx.query_inflight = False
            elif event_type == "consume":
                tx.state = "consumed"
                tx.disposition = "consumed"
                tx.effect_id = data.get("effect_id")
                tx.consumed_at_mono_ns = int(data.get("result", {}).get("created_mono_ns", _now()))
                result = data.get("result", {})
                if tx.effect_id:
                    idems[txn_id] = {
                        "idem_key": txn_id,
                        "txn_id": txn_id,
                        "effect_id": tx.effect_id,
                        "consumer_id": data.get("consumer_id", ""),
                        "created_mono_ns": int(result.get("created_mono_ns", _now())),
                        "last_access_mono_ns": int(result.get("created_mono_ns", _now())),
                        "consume_count": 1,
                        "result_json": canonical_json(result or {"effect_id": tx.effect_id}),
                    }
                    evicted.discard(txn_id)
            elif event_type == "pull_duplicate":
                if txn_id in idems:
                    idems[txn_id]["last_access_mono_ns"] = int(
                        data.get("accessed_mono_ns", idems[txn_id]["last_access_mono_ns"])
                    )
            elif event_type == "idempotency_evict":
                idems.pop(txn_id, None)
                evicted.add(txn_id)
            elif event_type == "conflict":
                tx.conflict_reason = data.get("winner")
            # pull_started / recovery_pull_required / query_exhausted do not
            # alter durable transaction state beyond the surrounding events.
        return states, idems

    # ------------------------------------------------------------------
    # observations
    # ------------------------------------------------------------------
    @staticmethod
    def _aggregate(values: list[int], budget_us: int | None = None) -> dict[str, Any]:
        if not values:
            return {
                "sample_count": 0,
                "mean_us": 0,
                "p95_us": 0,
                "max_us": 0,
                "mean_ms_x1000": 0,
                "p95_ms_x1000": 0,
                "max_ms_x1000": 0,
                "budget_exceeded_count": 0,
                "quantization_error_upper_bound_us": 2,
            }
        ordered = sorted(values)
        count = len(ordered)
        p95_index = max(0, math.ceil(0.95 * count) - 1)
        mean_us = (sum(ordered) + count // 2) // count
        p95_us = ordered[p95_index]
        max_us = ordered[-1]
        return {
            "sample_count": count,
            "mean_us": mean_us,
            "p95_us": p95_us,
            "max_us": max_us,
            "mean_ms_x1000": mean_us,
            "p95_ms_x1000": p95_us,
            "max_ms_x1000": max_us,
            "budget_exceeded_count": sum(1 for value in ordered if budget_us is not None and value > budget_us),
            "quantization_error_upper_bound_us": 2,
        }

    def latency_stats(self, *, txn_id: str | None = None, operation: str | None = None) -> dict[str, Any]:
        if operation not in {None, "query", "pull"}:
            raise ProtocolError("operation must be query or pull", "$.operation")
        with self.store.read() as conn:
            clauses: list[str] = []
            params: list[Any] = []
            if txn_id is not None:
                clauses.append("txn_id=?")
                params.append(txn_id)
            if operation is not None:
                clauses.append("operation=?")
                params.append(operation)
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            rows = conn.execute(f"SELECT * FROM latency{where} ORDER BY id", params).fetchall()
        grouped: dict[str, dict[str, list[int]]] = {}
        overall: dict[str, list[int]] = {"query": [], "pull": []}
        budget = {"query": self.config.query_budget_ms, "pull": self.config.pull_budget_ms}
        for row in rows:
            op = row["operation"]
            grouped.setdefault(row["txn_id"], {"query": [], "pull": []})[op].append(int(row["duration_us"]))
            overall[op].append(int(row["duration_us"]))
        result = {
            "ok": True,
            "unit": "microseconds_integer",
            "millisecond_fixed_point": "duration_us / 1000 with exactly three decimal ms places",
            "quantization": {
                "clock": "time.monotonic_ns",
                "storage_unit_us": 1,
                "sample_truncation_upper_bound_us": 1,
                "mean_rounding_upper_bound_us": 1,
                "combined_upper_bound_us": 2,
            },
            "truth_source": "sqlite latency rows; one row per observed interaction",
            "records": [
                {
                    "operation": row["operation"],
                    "txn_id": row["txn_id"],
                    "request_id": row["request_id"],
                    "start_mono_ns": row["start_mono_ns"],
                    "end_mono_ns": row["end_mono_ns"],
                    "duration_us": row["duration_us"],
                    "duration_ms_x1000": row["duration_us"],
                    "budget_us": row["budget_us"],
                    "exceeded": bool(row["exceeded"]),
                    "truth_source": row["truth_source"],
                }
                for row in rows
            ],
            "budgets_ms": budget,
            "overall": {
                op: self._aggregate(overall[op], budget[op] * 1000)
                for op in ("query", "pull")
            },
            "per_txn": {
                key: {
                    op: self._aggregate(values[op], budget[op] * 1000)
                    for op in ("query", "pull")
                }
                for key, values in sorted(grouped.items())
            },
        }
        return result

    def transaction_trace(self, txn_id: str) -> dict[str, Any]:
        with self.store.read() as conn:
            row = conn.execute("SELECT * FROM txns WHERE txn_id=?", (txn_id,)).fetchone()
            if row is None:
                raise ProtocolError("transaction is not registered", "$.txn_id")
            tx = _tx_from_row(row)
            event_rows = conn.execute(
                "SELECT * FROM events WHERE txn_id=? ORDER BY event_seq", (txn_id,)
            ).fetchall()
            events = []
            for event in event_rows:
                try:
                    decoded = decode_record(event["record"])
                    events.append(
                        {
                            "event_seq": decoded["event_seq"],
                            "type": decoded["type"],
                            "state": decoded["state"],
                            "version": decoded["version"],
                            "body_digest": decoded["body_digest"],
                            "data": decoded["data"],
                            "valid": bool(event["valid"]),
                        }
                    )
                except MWError as exc:
                    events.append({"event_seq": event["event_seq"], "valid": False, "error": exc.as_dict()})
            conflicts = []
            for item in conn.execute(
                "SELECT * FROM conflicts WHERE txn_id=? ORDER BY id", (txn_id,)
            ).fetchall():
                conflict = dict(item)
                conflict["evidence"] = json.loads(conflict["evidence_json"])
                conflicts.append(conflict)
            evictions = []
            for item in conn.execute(
                "SELECT * FROM idempotency_evictions WHERE txn_id=? ORDER BY id", (txn_id,)
            ).fetchall():
                eviction = dict(item)
                eviction["detail"] = json.loads(eviction["detail_json"])
                evictions.append(eviction)
        return {
            "ok": True,
            "txn_id": txn_id,
            "version": tx.version,
            "state": tx.state,
            "disposition": tx.disposition,
            "terminal": tx.terminal,
            "terminal_reason": tx.terminal_reason,
            "conflict_reason": tx.conflict_reason,
            "error_code": (
                "E_QUERY"
                if tx.disposition == "manual_review"
                else "E_CONFLICT" if tx.conflict_reason else None
            ),
            "query_count": tx.query_count,
            "query_max": tx.query_max,
            "first_query_mono_ns": tx.first_query_mono_ns,
            "query_inflight": tx.query_inflight,
            "ack_sent": tx.ack_sent,
            "pending_command": tx.pending_command,
            "last_query_result": tx.last_query_result,
            "effect_id": tx.effect_id,
            "events": events,
            "conflicts": conflicts,
            "idempotency_evictions": evictions,
            "latency": self.latency_stats(txn_id=txn_id),
        }

    def inventory(self) -> dict[str, Any]:
        with self.store.read() as conn:
            counts = {
                "transactions": conn.execute("SELECT COUNT(*) AS n FROM txns").fetchone()["n"],
                "events": conn.execute("SELECT COUNT(*) AS n FROM events WHERE valid=1").fetchone()["n"],
                "idempotency_keys": conn.execute("SELECT COUNT(*) AS n FROM idempotency").fetchone()["n"],
                "pending_pulls": conn.execute("SELECT COUNT(*) AS n FROM pulls WHERE status='pending'").fetchone()["n"],
                "consumed_transactions": conn.execute("SELECT COUNT(*) AS n FROM txns WHERE state='consumed'").fetchone()["n"],
                "idempotency_evictions": conn.execute("SELECT COUNT(*) AS n FROM idempotency_evictions").fetchone()["n"],
            }
            tx_rows = conn.execute("SELECT * FROM txns ORDER BY txn_id").fetchall()
            transactions = [
                {
                    "txn_id": row["txn_id"],
                    "version": row["version"],
                    "state": row["state"],
                    "disposition": row["disposition"],
                    "query_count": row["query_count"],
                    "query_max": row["query_max"],
                    "effect_id": row["effect_id"],
                }
                for row in tx_rows
            ]
            digest = self.store.refresh_digest(conn)
        return {
            "ok": True,
            "counts": counts,
            "transactions": transactions,
            "latency": self.latency_stats(),
            "ledger": {
                "format_version": int(self.store.get_meta("format_version", str(FORMAT_VERSION)) or FORMAT_VERSION),
                "content_digest": digest,
            },
            "idempotency_window": {
                "max_entries": self.config.idem_max_entries,
                "ttl_ms": self.config.idem_ttl_ms,
            },
        }
