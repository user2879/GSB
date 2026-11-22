"""SQLite-backed transaction-message ledger.

The event stream and the transaction catalog are authoritative.  The
partition-prefix table is a cache refreshed in a *separate* transaction so a
forced termination can leave it stale; recovery always rebuilds it from the
durable event stream.
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

from .canonical import canonical_bytes, digest_bytes, digest_value, message_digest
from .content import (
    ledger_content,
    ledger_digest,
    normalise_event,
    normalise_visibility,
)
from .errors import BusinessError, OrderError, RecordError, SnapshotError, SpecError
from .models import (
    ABORTED,
    ALLOWED_TRANSITIONS,
    COMMITTED,
    COMMITTING,
    FORMAT_VERSION,
    PREPARED,
    STATES,
    LedgerConclusion,
    Message,
    Partition,
    PrefixReport,
    RecoveryReport,
    Transaction,
    TransitionResult,
    UnknownTransaction,
    merge_intervals,
)
from .observability import StructuredLogger, default_log_path, utc_now_text
from .replay import ReplayMismatch, derive_prefix_reports, replay_events, unknown_transactions


SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS partitions (
    partition_id TEXT PRIMARY KEY,
    partition_version INTEGER NOT NULL,
    max_position INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id TEXT PRIMARY KEY,
    txn_seq INTEGER NOT NULL UNIQUE,
    partition_id TEXT NOT NULL REFERENCES partitions(partition_id),
    local_txn_version INTEGER NOT NULL,
    commit_time_ms INTEGER NOT NULL,
    spec_json TEXT NOT NULL,
    spec_length INTEGER NOT NULL,
    spec_digest TEXT NOT NULL,
    current_state TEXT,
    last_event_seq INTEGER
);
CREATE TABLE IF NOT EXISTS messages (
    transaction_id TEXT NOT NULL REFERENCES transactions(transaction_id),
    partition_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    payload_json TEXT NOT NULL,
    payload_length INTEGER NOT NULL,
    message_digest TEXT NOT NULL,
    PRIMARY KEY (transaction_id, position),
    UNIQUE (partition_id, position)
);
CREATE TABLE IF NOT EXISTS events (
    event_seq INTEGER PRIMARY KEY,
    transaction_id TEXT NOT NULL REFERENCES transactions(transaction_id),
    state TEXT NOT NULL,
    local_txn_version INTEGER NOT NULL,
    partition_id TEXT NOT NULL,
    body_json TEXT NOT NULL,
    body_length INTEGER NOT NULL,
    body_digest TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS visibility (
    transaction_id TEXT NOT NULL REFERENCES transactions(transaction_id),
    partition_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    message_digest TEXT NOT NULL,
    effective_event_seq INTEGER NOT NULL,
    body_json TEXT NOT NULL,
    body_length INTEGER NOT NULL,
    body_digest TEXT NOT NULL,
    PRIMARY KEY (transaction_id, position),
    UNIQUE (partition_id, position)
);
CREATE TABLE IF NOT EXISTS partition_prefix (
    partition_id TEXT PRIMARY KEY REFERENCES partitions(partition_id),
    prefix INTEGER NOT NULL,
    truncated_position INTEGER,
    truncated_reason TEXT,
    truncated_transaction_id TEXT,
    unknown_intervals_json TEXT NOT NULL,
    computed_event_seq INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS recovery_marks (
    transaction_id TEXT PRIMARY KEY REFERENCES transactions(transaction_id),
    partition_id TEXT NOT NULL,
    positions_json TEXT NOT NULL,
    reason TEXT NOT NULL,
    detected_at TEXT NOT NULL,
    requires_rerun INTEGER NOT NULL
);
"""


def _encoded_record(value: dict[str, Any]) -> tuple[str, int, str]:
    body = canonical_bytes(value)
    return body.decode("utf-8"), len(body), digest_bytes(body)


def _payload_record(payload: Any) -> tuple[str, int]:
    body = canonical_bytes(payload)
    return body.decode("utf-8"), len(body)


class Ledger:
    """A local ledger with atomic event writes and recoverable prefix cache."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        create: bool = True,
        auto_recover: bool = True,
        readonly: bool = False,
        logger: StructuredLogger | None = None,
    ) -> None:
        self.path = Path(path)
        self.readonly = readonly
        self._logger = logger or StructuredLogger(default_log_path(self.path))
        self.last_recovery: RecoveryReport | None = None

        if readonly:
            if not self.path.exists():
                self._fail(RecordError("只读账本文件不存在", field="path", details={"path": str(self.path)}))
            uri = self.path.resolve().as_uri() + "?mode=ro"
            try:
                self._conn = sqlite3.connect(uri, uri=True)
            except sqlite3.DatabaseError as exc:
                self._fail(RecordError("无法以只读方式打开账本", field="path", details={"error": str(exc)}))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA query_only=ON")
        else:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.execute("PRAGMA journal_mode=DELETE")
            self._conn.execute("PRAGMA synchronous=FULL")
            if create:
                self._initialise_schema()
            else:
                self._require_schema()

        self.verify_records()
        if not readonly and auto_recover:
            self.last_recovery = self.recover()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

    def close(self) -> None:
        connection = getattr(self, "_conn", None)
        if connection is not None:
            connection.close()
            self._conn = None  # type: ignore[assignment]

    @property
    def connection(self) -> sqlite3.Connection:
        if getattr(self, "_conn", None) is None:
            raise RecordError("账本已关闭", field="connection")
        return self._conn

    # ------------------------------------------------------------------
    # Schema and transactions
    # ------------------------------------------------------------------
    def _initialise_schema(self) -> None:
        self._conn.executescript(SCHEMA)
        self._conn.commit()
        if self._meta_value("ledger_format_version") is None:
            initial_digest = ledger_digest(
                (), (), (), (), partition_set_version=0, transaction_set_version=0
            )
            initial = {
                "ledger_format_version": str(FORMAT_VERSION),
                "partition_set_version": "0",
                "transaction_set_version": "0",
                "ledger_digest": initial_digest,
                "created_at": utc_now_text(),
            }
            self._conn.executemany(
                "INSERT INTO meta(key, value) VALUES(?, ?)",
                [(key, value) for key, value in initial.items()],
            )
            self._conn.commit()

    def _require_schema(self) -> None:
        required = {"meta", "partitions", "transactions", "messages", "events", "visibility", "partition_prefix", "recovery_marks"}
        rows = self._conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        names = {row["name"] for row in rows}
        if not required.issubset(names):
            self._fail(
                RecordError(
                    "账本 schema 不完整",
                    field="schema",
                    details={"missing": sorted(required - names)},
                )
            )

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        if self.readonly:
            self._fail(RecordError("只读账本不允许写入", field="readonly"))
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield self.connection
        except BaseException:
            self.connection.rollback()
            raise
        else:
            self.connection.commit()

    def _fail(self, error: BusinessError) -> None:
        self._logger.write_error(error)
        raise error

    def _meta_value(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _meta_int(self, key: str) -> int:
        value = self._meta_value(key)
        if value is None:
            self._fail(RecordError("账本元数据缺失", field=f"meta.{key}"))
        try:
            return int(value)
        except ValueError:
            self._fail(RecordError("账本元数据不是整数", field=f"meta.{key}", details={"value": value}))
        raise AssertionError("unreachable")

    def _event_cursor(self, conn: sqlite3.Connection | None = None) -> int:
        row = (conn or self.connection).execute("SELECT COALESCE(MAX(event_seq), 0) AS cursor FROM events").fetchone()
        return int(row["cursor"])

    # ------------------------------------------------------------------
    # Read helpers
    # ------------------------------------------------------------------
    def partitions(self, conn: sqlite3.Connection | None = None) -> tuple[Partition, ...]:
        rows = (conn or self.connection).execute(
            "SELECT partition_id, partition_version, max_position FROM partitions ORDER BY partition_id"
        ).fetchall()
        return tuple(
            Partition(
                partition_id=row["partition_id"],
                partition_version=int(row["partition_version"]),
                max_position=int(row["max_position"]),
            )
            for row in rows
        )

    def transactions(self, conn: sqlite3.Connection | None = None) -> tuple[Transaction, ...]:
        connection = conn or self.connection
        tx_rows = connection.execute(
            "SELECT * FROM transactions ORDER BY txn_seq, transaction_id"
        ).fetchall()
        message_rows = connection.execute(
            "SELECT * FROM messages ORDER BY transaction_id, position"
        ).fetchall()
        messages_by_transaction: dict[str, list[Message]] = {}
        for row in message_rows:
            try:
                payload = json.loads(row["payload_json"])
            except (TypeError, ValueError) as exc:
                self._fail(
                    RecordError(
                        "消息载荷 JSON 损坏",
                        transaction_id=row["transaction_id"],
                        partition_id=row["partition_id"],
                        field="payload_json",
                        details={"error": str(exc)},
                    )
                )
            messages_by_transaction.setdefault(row["transaction_id"], []).append(
                Message(
                    position=int(row["position"]),
                    payload=payload,
                    message_digest=row["message_digest"],
                )
            )
        result: list[Transaction] = []
        for row in tx_rows:
            result.append(
                Transaction(
                    transaction_id=row["transaction_id"],
                    txn_seq=int(row["txn_seq"]),
                    partition_id=row["partition_id"],
                    local_txn_version=int(row["local_txn_version"]),
                    messages=tuple(messages_by_transaction.get(row["transaction_id"], ())),
                    commit_time_ms=int(row["commit_time_ms"]),
                )
            )
        return tuple(result)

    def events(self, conn: sqlite3.Connection | None = None) -> tuple[dict[str, Any], ...]:
        rows = (conn or self.connection).execute(
            "SELECT * FROM events ORDER BY event_seq"
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            try:
                result.append(normalise_event(json.loads(row["body_json"])))
            except (TypeError, ValueError) as exc:
                self._fail(
                    RecordError(
                        "事务状态事件 JSON 损坏",
                        transaction_id=row["transaction_id"],
                        partition_id=row["partition_id"],
                        event_seq=int(row["event_seq"]),
                        record_seq=int(row["event_seq"]),
                        field="body_json",
                        details={"error": str(exc)},
                    )
                )
        return tuple(result)

    def visibility(self, conn: sqlite3.Connection | None = None) -> tuple[dict[str, Any], ...]:
        rows = (conn or self.connection).execute(
            "SELECT * FROM visibility ORDER BY effective_event_seq, partition_id, position, transaction_id"
        ).fetchall()
        result: list[dict[str, Any]] = []
        for row in rows:
            try:
                body = json.loads(row["body_json"])
            except (TypeError, ValueError) as exc:
                self._fail(
                    RecordError(
                        "消息可见性记录 JSON 损坏",
                        transaction_id=row["transaction_id"],
                        partition_id=row["partition_id"],
                        event_seq=int(row["effective_event_seq"]),
                        record_seq=int(row["effective_event_seq"]),
                        field="body_json",
                        details={"error": str(exc)},
                    )
                )
            result.append(normalise_visibility(body))
        return tuple(result)

    def latest_states(self, conn: sqlite3.Connection | None = None) -> dict[str, str]:
        states: dict[str, str] = {}
        rows = (conn or self.connection).execute(
            "SELECT transaction_id, state FROM events ORDER BY event_seq"
        ).fetchall()
        for row in rows:
            states[row["transaction_id"]] = row["state"]
        return states

    def persisted_prefix_rows(self) -> dict[str, dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM partition_prefix ORDER BY partition_id"
        ).fetchall()
        result: dict[str, dict[str, Any]] = {}
        for row in rows:
            try:
                unknown = json.loads(row["unknown_intervals_json"])
            except (TypeError, ValueError):
                unknown = []
            result[row["partition_id"]] = {
                "prefix": int(row["prefix"]),
                "truncated_position": row["truncated_position"],
                "truncated_reason": row["truncated_reason"],
                "truncated_transaction_id": row["truncated_transaction_id"],
                "unknown_intervals": tuple(tuple(item) for item in unknown),
                "computed_event_seq": int(row["computed_event_seq"]),
            }
        return result

    # ------------------------------------------------------------------
    # Partition and transaction ingestion
    # ------------------------------------------------------------------
    def initialise_partitions(self, partitions: Iterable[Partition]) -> None:
        supplied = tuple(partitions)
        if not supplied:
            self._fail(SpecError("至少需要一个分区", field="partitions"))
        ids = [partition.partition_id for partition in supplied]
        if len(ids) != len(set(ids)):
            self._fail(SpecError("分区标识必须唯一", field="partitions.partition_id"))
        try:
            with self._write() as connection:
                existing = {partition.partition_id: partition for partition in self.partitions(connection)}
                missing_existing = sorted(set(existing) - set(ids))
                if missing_existing:
                    self._fail(
                        SpecError(
                            "账本中已有分区不能通过声明文件删除",
                            field="partitions",
                            details={"missing_partition_ids": missing_existing},
                        )
                    )
                changed = False
                for partition in supplied:
                    old = existing.get(partition.partition_id)
                    if old is None:
                        connection.execute(
                            "INSERT INTO partitions(partition_id, partition_version, max_position) VALUES(?,?,?)",
                            (partition.partition_id, partition.partition_version, partition.max_position),
                        )
                        changed = True
                    elif old != partition:
                        row = connection.execute(
                            "SELECT MAX(position) AS maximum FROM messages WHERE partition_id=?",
                            (partition.partition_id,),
                        ).fetchone()
                        maximum_used = row["maximum"]
                        if maximum_used is not None and partition.max_position < int(maximum_used):
                            self._fail(
                                SpecError(
                                    "新分区位点上限小于已有消息位点",
                                    partition_id=partition.partition_id,
                                    field="max_position",
                                    details={"maximum_used": int(maximum_used)},
                                )
                            )
                        connection.execute(
                            "UPDATE partitions SET partition_version=?, max_position=? WHERE partition_id=?",
                            (partition.partition_version, partition.max_position, partition.partition_id),
                        )
                        changed = True
                if changed:
                    self._bump_meta(connection, "partition_set_version")
                self._recompute_cache(connection)
                self._update_ledger_digest(connection)
        except BusinessError:
            raise
        except sqlite3.DatabaseError as exc:
            self._fail(RecordError("分区声明写入失败", field="partitions", details={"error": str(exc)}))

    def ingest(self, transactions: Iterable[Transaction]) -> int:
        incoming = tuple(transactions)
        if not incoming:
            self._fail(SpecError("事务批次不能为空", field="transactions"))
        incoming_ids = [transaction.transaction_id for transaction in incoming]
        if len(incoming_ids) != len(set(incoming_ids)):
            self._fail(SpecError("transaction_id 必须唯一", field="transaction_id"))
        inserted = 0
        try:
            with self._write() as connection:
                partition_ids = {partition.partition_id for partition in self.partitions(connection)}
                max_seq_row = connection.execute("SELECT COALESCE(MAX(txn_seq),0) AS maximum FROM transactions").fetchone()
                maximum_seq = int(max_seq_row["maximum"])
                previous_seq = maximum_seq
                for transaction in incoming:
                    if transaction.partition_id not in partition_ids:
                        self._fail(
                            SpecError(
                                "目标分区不存在",
                                transaction_id=transaction.transaction_id,
                                partition_id=transaction.partition_id,
                                field="partition_id",
                            )
                        )
                    existing = connection.execute(
                        "SELECT spec_digest FROM transactions WHERE transaction_id=?",
                        (transaction.transaction_id,),
                    ).fetchone()
                    if existing is not None:
                        if existing["spec_digest"] != transaction.spec_digest:
                            self._fail(
                                SpecError(
                                    "已有事务标识与批次内容冲突",
                                    transaction_id=transaction.transaction_id,
                                    partition_id=transaction.partition_id,
                                    field="transaction_id",
                                )
                            )
                        continue
                    if transaction.txn_seq <= previous_seq:
                        self._fail(
                            SpecError(
                                "事务序号必须严格递增并大于账本已有最大序号",
                                transaction_id=transaction.transaction_id,
                                partition_id=transaction.partition_id,
                                field="txn_seq",
                                details={"previous_txn_seq": previous_seq, "txn_seq": transaction.txn_seq},
                            )
                        )
                    previous_seq = transaction.txn_seq
                    previous_position: int | None = None
                    for message in transaction.messages:
                        if previous_position is not None and message.position <= previous_position:
                            self._fail(
                                SpecError(
                                    "同一事务的消息位点必须严格递增",
                                    transaction_id=transaction.transaction_id,
                                    partition_id=transaction.partition_id,
                                    field="messages.position",
                                )
                            )
                        previous_position = message.position
                        conflict = connection.execute(
                            "SELECT transaction_id FROM messages WHERE partition_id=? AND position=?",
                            (transaction.partition_id, message.position),
                        ).fetchone()
                        if conflict is not None:
                            self._fail(
                                OrderError(
                                    "同一分区内消息位点重复",
                                    transaction_id=transaction.transaction_id,
                                    partition_id=transaction.partition_id,
                                    field="messages.position",
                                    details={
                                        "position": message.position,
                                        "first_transaction_id": conflict["transaction_id"],
                                    },
                                )
                            )
                    spec_text, spec_length, spec_digest = _encoded_record(
                        transaction.as_spec_dict(include_digest=False)
                    )
                    connection.execute(
                        """
                        INSERT INTO transactions(
                            transaction_id, txn_seq, partition_id, local_txn_version,
                            commit_time_ms, spec_json, spec_length, spec_digest,
                            current_state, last_event_seq
                        ) VALUES(?,?,?,?,?,?,?,?,NULL,NULL)
                        """,
                        (
                            transaction.transaction_id,
                            transaction.txn_seq,
                            transaction.partition_id,
                            transaction.local_txn_version,
                            transaction.commit_time_ms,
                            spec_text,
                            spec_length,
                            spec_digest,
                        ),
                    )
                    for message in transaction.messages:
                        payload_text, payload_length = _payload_record(message.payload)
                        connection.execute(
                            "INSERT INTO messages(transaction_id, partition_id, position, payload_json, payload_length, message_digest) VALUES(?,?,?,?,?,?)",
                            (
                                transaction.transaction_id,
                                transaction.partition_id,
                                message.position,
                                payload_text,
                                payload_length,
                                message.message_digest,
                            ),
                        )
                    inserted += 1
                if inserted:
                    self._bump_meta(connection, "transaction_set_version")
                    self._recompute_cache(connection)
                    self._update_ledger_digest(connection)
        except BusinessError:
            raise
        except sqlite3.IntegrityError as exc:
            self._fail(
                RecordError(
                    "事务批次写入违反唯一性或外键约束",
                    field="transactions",
                    details={"error": str(exc)},
                )
            )
        except sqlite3.DatabaseError as exc:
            self._fail(RecordError("事务批次写入失败", field="transactions", details={"error": str(exc)}))
        return inserted

    def _bump_meta(self, connection: sqlite3.Connection, key: str) -> int:
        current = self._meta_int(key)
        connection.execute("UPDATE meta SET value=? WHERE key=?", (str(current + 1), key))
        return current + 1

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------
    def advance_state(
        self,
        transaction_id: str,
        new_state: str,
        *,
        crash_hook: Callable[[], Any] | None = None,
    ) -> TransitionResult:
        if new_state not in STATES:
            self._fail(
                SpecError(
                    "未知事务状态",
                    transaction_id=transaction_id,
                    field="state",
                    details={"state": new_state, "allowed": list(STATES)},
                )
            )
        try:
            with self._write() as connection:
                transaction_row = connection.execute(
                    "SELECT * FROM transactions WHERE transaction_id=?",
                    (transaction_id,),
                ).fetchone()
                if transaction_row is None:
                    self._fail(
                        SpecError(
                            "事务标识不存在",
                            transaction_id=transaction_id,
                            field="transaction_id",
                        )
                    )
                row = connection.execute(
                    "SELECT state FROM events WHERE transaction_id=? ORDER BY event_seq DESC LIMIT 1",
                    (transaction_id,),
                ).fetchone()
                previous_state = None if row is None else row["state"]
                if previous_state is None:
                    if new_state != PREPARED:
                        self._fail(
                            SpecError(
                                "事务首次状态必须为 prepared",
                                transaction_id=transaction_id,
                                partition_id=transaction_row["partition_id"],
                                field="state",
                                details={"state": new_state},
                            )
                        )
                elif new_state not in ALLOWED_TRANSITIONS[previous_state]:
                    self._fail(
                        SpecError(
                            "非法事务状态迁移",
                            transaction_id=transaction_id,
                            partition_id=transaction_row["partition_id"],
                            field="state",
                            details={
                                "previous_state": previous_state,
                                "state": new_state,
                                "allowed": sorted(ALLOWED_TRANSITIONS[previous_state]),
                            },
                        )
                    )

                event_seq = self._event_cursor(connection) + 1
                event = {
                    "event_seq": event_seq,
                    "transaction_id": transaction_id,
                    "state": new_state,
                    "local_txn_version": int(transaction_row["local_txn_version"]),
                    "partition_id": transaction_row["partition_id"],
                }
                event_text, event_length, event_digest = _encoded_record(event)
                connection.execute(
                    "INSERT INTO events(event_seq, transaction_id, state, local_txn_version, partition_id, body_json, body_length, body_digest) VALUES(?,?,?,?,?,?,?,?)",
                    (
                        event_seq,
                        transaction_id,
                        new_state,
                        int(transaction_row["local_txn_version"]),
                        transaction_row["partition_id"],
                        event_text,
                        event_length,
                        event_digest,
                    ),
                )

                if new_state == COMMITTED:
                    message_rows = connection.execute(
                        "SELECT * FROM messages WHERE transaction_id=? ORDER BY position",
                        (transaction_id,),
                    ).fetchall()
                    for message_row in message_rows:
                        payload = json.loads(message_row["payload_json"])
                        body = {
                            "transaction_id": transaction_id,
                            "partition_id": transaction_row["partition_id"],
                            "position": int(message_row["position"]),
                            "message_digest": message_row["message_digest"],
                            "effective_event_seq": event_seq,
                            "payload": payload,
                        }
                        body_text, body_length, body_digest = _encoded_record(body)
                        connection.execute(
                            "INSERT INTO visibility(transaction_id, partition_id, position, message_digest, effective_event_seq, body_json, body_length, body_digest) VALUES(?,?,?,?,?,?,?,?)",
                            (
                                transaction_id,
                                transaction_row["partition_id"],
                                int(message_row["position"]),
                                message_row["message_digest"],
                                event_seq,
                                body_text,
                                body_length,
                                body_digest,
                            ),
                        )

                connection.execute(
                    "UPDATE transactions SET current_state=?, last_event_seq=? WHERE transaction_id=?",
                    (new_state, event_seq, transaction_id),
                )
                if new_state in (COMMITTED, ABORTED):
                    connection.execute(
                        "DELETE FROM recovery_marks WHERE transaction_id=?",
                        (transaction_id,),
                    )

                self._validate_event_row(
                    connection.execute("SELECT * FROM events WHERE event_seq=?", (event_seq,)).fetchone()
                )
                if new_state == COMMITTED:
                    for visibility_row in connection.execute(
                        "SELECT * FROM visibility WHERE effective_event_seq=? ORDER BY position",
                        (event_seq,),
                    ).fetchall():
                        self._validate_visibility_row(connection, visibility_row)
                self._update_ledger_digest(connection)
        except BusinessError:
            raise
        except sqlite3.DatabaseError as exc:
            self._fail(
                RecordError(
                    "事务状态事件原子写入失败",
                    transaction_id=transaction_id,
                    field="events",
                    details={"error": str(exc)},
                )
            )

        if crash_hook is not None:
            crash_hook()

        reports = self._recompute_cache()
        affected = next(report for report in reports if report.partition_id == transaction_row["partition_id"])
        return TransitionResult(
            transaction_id=transaction_id,
            event_seq=event_seq,
            previous_state=previous_state,
            state=new_state,
            prefix=affected,
        )

    def advance_many(
        self,
        transitions: Sequence[tuple[str, str]],
        *,
        crash_hook: Callable[[], Any] | None = None,
    ) -> tuple[TransitionResult, ...]:
        results: list[TransitionResult] = []
        for transaction_id, state in transitions:
            results.append(
                self.advance_state(transaction_id, state, crash_hook=crash_hook)
            )
        return tuple(results)

    def _recompute_cache(
        self,
        connection: sqlite3.Connection | None = None,
    ) -> tuple[PrefixReport, ...]:
        own_transaction = connection is None
        if own_transaction:
            if self.readonly:
                return self._derive_current()
            with self._write() as write_connection:
                reports = self._derive_current(write_connection)
                self._store_cache(write_connection, reports)
                return reports
        reports = self._derive_current(connection)
        self._store_cache(connection, reports)
        return reports

    def _derive_current(
        self,
        connection: sqlite3.Connection | None = None,
    ) -> tuple[PrefixReport, ...]:
        connection = connection or self.connection
        states = {transaction_id: state for transaction_id, state in self.latest_states(connection).items()}
        transactions = self.transactions(connection)
        for transaction in transactions:
            states.setdefault(transaction.transaction_id, None)  # type: ignore[arg-type]
        return derive_prefix_reports(self.partitions(connection), transactions, states)

    def _store_cache(
        self,
        connection: sqlite3.Connection,
        reports: Sequence[PrefixReport],
    ) -> None:
        cursor = self._event_cursor(connection)
        for report in reports:
            connection.execute(
                """
                INSERT INTO partition_prefix(
                    partition_id, prefix, truncated_position, truncated_reason,
                    truncated_transaction_id, unknown_intervals_json, computed_event_seq
                ) VALUES(?,?,?,?,?,?,?)
                ON CONFLICT(partition_id) DO UPDATE SET
                    prefix=excluded.prefix,
                    truncated_position=excluded.truncated_position,
                    truncated_reason=excluded.truncated_reason,
                    truncated_transaction_id=excluded.truncated_transaction_id,
                    unknown_intervals_json=excluded.unknown_intervals_json,
                    computed_event_seq=excluded.computed_event_seq
                """,
                (
                    report.partition_id,
                    report.prefix,
                    report.truncated_position,
                    report.truncated_reason,
                    report.truncated_transaction_id,
                    canonical_bytes(
                        [[start, end] for start, end in report.unknown_intervals]
                    ).decode("utf-8"),
                    cursor,
                ),
            )

    # ------------------------------------------------------------------
    # Recovery and queries
    # ------------------------------------------------------------------
    def recover(self) -> RecoveryReport:
        if self.readonly:
            self._fail(RecordError("只读账本不能执行恢复写入", field="readonly"))
        self.verify_records()
        cursor = self._event_cursor()
        persisted = self.persisted_prefix_rows()
        derived = self.query_prefixes()
        repaired = self._cache_differs(persisted, derived, cursor)
        transactions = self.transactions()
        states: dict[str, str | None] = {transaction.transaction_id: None for transaction in transactions}
        states.update(self.latest_states())
        unknown = unknown_transactions(transactions, states)
        with self._write() as connection:
            connection.execute("DELETE FROM recovery_marks")
            for item in unknown:
                connection.execute(
                    "INSERT INTO recovery_marks(transaction_id, partition_id, positions_json, reason, detected_at, requires_rerun) VALUES(?,?,?,?,?,1)",
                    (
                        item.transaction_id,
                        item.partition_id,
                        canonical_bytes([[start, end] for start, end in item.intervals]).decode("utf-8"),
                        item.reason,
                        utc_now_text(),
                    ),
                )
            self._store_cache(connection, derived)
        return RecoveryReport(
            recovered=True,
            ledger_format_version=self._meta_int("ledger_format_version"),
            event_cursor=cursor,
            persisted_prefix_repaired=repaired,
            unknown_transactions=unknown,
            partitions=derived,
        )

    def _cache_differs(
        self,
        persisted: dict[str, dict[str, Any]],
        derived: Sequence[PrefixReport],
        cursor: int,
    ) -> bool:
        if set(persisted) != {report.partition_id for report in derived}:
            return True
        for report in derived:
            row = persisted[report.partition_id]
            if row["prefix"] != report.prefix:
                return True
            if row["truncated_position"] != report.truncated_position:
                return True
            if row["truncated_reason"] != report.truncated_reason:
                return True
            if row["truncated_transaction_id"] != report.truncated_transaction_id:
                return True
            if row["unknown_intervals"] != report.unknown_intervals:
                return True
            if row["computed_event_seq"] != cursor:
                return True
        return False

    def query_prefixes(self) -> tuple[PrefixReport, ...]:
        return self._derive_current()

    def unknown_transactions(self) -> tuple[UnknownTransaction, ...]:
        transactions = self.transactions()
        states: dict[str, str | None] = {transaction.transaction_id: None for transaction in transactions}
        states.update(self.latest_states())
        return unknown_transactions(transactions, states)

    def conclusion(self) -> LedgerConclusion:
        return LedgerConclusion(
            ledger_format_version=self._meta_int("ledger_format_version"),
            partition_set_version=self._meta_int("partition_set_version"),
            transaction_set_version=self._meta_int("transaction_set_version"),
            event_cursor=self._event_cursor(),
            ledger_digest=str(self._meta_value("ledger_digest")),
            partitions=self.query_prefixes(),
            unknown_transactions=self.unknown_transactions(),
        )

    def content(self) -> dict[str, Any]:
        return ledger_content(
            self.partitions(),
            self.transactions(),
            self.events(),
            self.visibility(),
            partition_set_version=self._meta_int("partition_set_version"),
            transaction_set_version=self._meta_int("transaction_set_version"),
            ledger_format_version=self._meta_int("ledger_format_version"),
        )

    def computed_ledger_digest(self) -> str:
        return digest_value(self.content())

    def _update_ledger_digest(self, connection: sqlite3.Connection) -> None:
        content = ledger_content(
            self.partitions(connection),
            self.transactions(connection),
            self.events(connection),
            self.visibility(connection),
            partition_set_version=self._meta_int("partition_set_version"),
            transaction_set_version=self._meta_int("transaction_set_version"),
            ledger_format_version=self._meta_int("ledger_format_version"),
        )
        connection.execute(
            "UPDATE meta SET value=? WHERE key='ledger_digest'",
            (digest_value(content),),
        )

    # ------------------------------------------------------------------
    # Record-level validation
    # ------------------------------------------------------------------
    def _validate_event_row(self, row: sqlite3.Row | None) -> dict[str, Any]:
        if row is None:
            self._fail(RecordError("状态事件记录不存在", field="events"))
        event_seq = int(row["event_seq"])
        raw = row["body_json"]
        if not isinstance(raw, str):
            self._fail(RecordError("事件体不是文本", event_seq=event_seq, record_seq=event_seq, field="body_json"))
        encoded = raw.encode("utf-8")
        if int(row["body_length"]) != len(encoded):
            self._fail(
                RecordError(
                    "事件长度字段不自治",
                    transaction_id=row["transaction_id"],
                    partition_id=row["partition_id"],
                    event_seq=event_seq,
                    record_seq=event_seq,
                    field="body_length",
                    details={"stored": int(row["body_length"]), "actual": len(encoded)},
                )
            )
        actual_digest = digest_bytes(encoded)
        if row["body_digest"] != actual_digest:
            self._fail(
                RecordError(
                    "事件摘要不匹配",
                    transaction_id=row["transaction_id"],
                    partition_id=row["partition_id"],
                    event_seq=event_seq,
                    record_seq=event_seq,
                    field="body_digest",
                    details={"stored": row["body_digest"], "actual": actual_digest},
                )
            )
        try:
            body = json.loads(raw)
        except (TypeError, ValueError) as exc:
            self._fail(
                RecordError(
                    "事件体 JSON 损坏",
                    transaction_id=row["transaction_id"],
                    partition_id=row["partition_id"],
                    event_seq=event_seq,
                    record_seq=event_seq,
                    field="body_json",
                    details={"error": str(exc)},
                )
            )
        if not isinstance(body, dict):
            self._fail(RecordError("事件体必须是对象", event_seq=event_seq, record_seq=event_seq, field="body_json"))
        if canonical_bytes(body).decode("utf-8") != raw:
            self._fail(
                RecordError(
                    "事件体不是规范化 JSON",
                    transaction_id=row["transaction_id"],
                    event_seq=event_seq,
                    record_seq=event_seq,
                    field="body_json",
                )
            )
        expected = normalise_event(body)
        columns = {
            "event_seq": event_seq,
            "transaction_id": row["transaction_id"],
            "state": row["state"],
            "local_txn_version": int(row["local_txn_version"]),
            "partition_id": row["partition_id"],
        }
        for key, value in expected.items():
            if value != columns[key]:
                self._fail(
                    RecordError(
                        "事件字段与事件体不一致",
                        transaction_id=row["transaction_id"],
                        partition_id=row["partition_id"],
                        event_seq=event_seq,
                        record_seq=event_seq,
                        field=key,
                        details={"column": columns[key], "body": value},
                    )
                )
        if columns["state"] not in STATES:
            self._fail(
                RecordError(
                    "事件状态非法",
                    transaction_id=row["transaction_id"],
                    event_seq=event_seq,
                    record_seq=event_seq,
                    field="state",
                )
            )
        return expected

    def _validate_visibility_row(
        self,
        connection: sqlite3.Connection,
        row: sqlite3.Row | None,
    ) -> dict[str, Any]:
        if row is None:
            self._fail(RecordError("可见性记录不存在", field="visibility"))
        transaction_id = row["transaction_id"]
        event_seq = int(row["effective_event_seq"])
        raw = row["body_json"]
        encoded = raw.encode("utf-8") if isinstance(raw, str) else b""
        if not isinstance(raw, str):
            self._fail(
                RecordError(
                    "可见性记录体不是文本",
                    transaction_id=transaction_id,
                    partition_id=row["partition_id"],
                    event_seq=event_seq,
                    record_seq=event_seq,
                    field="body_json",
                )
            )
        if int(row["body_length"]) != len(encoded):
            self._fail(
                RecordError(
                    "可见性记录长度字段不自治",
                    transaction_id=transaction_id,
                    partition_id=row["partition_id"],
                    event_seq=event_seq,
                    record_seq=event_seq,
                    field="body_length",
                    details={"stored": int(row["body_length"]), "actual": len(encoded)},
                )
            )
        digest = digest_bytes(encoded)
        if row["body_digest"] != digest:
            self._fail(
                RecordError(
                    "可见性记录体摘要不匹配",
                    transaction_id=transaction_id,
                    partition_id=row["partition_id"],
                    event_seq=event_seq,
                    record_seq=event_seq,
                    field="body_digest",
                    details={"stored": row["body_digest"], "actual": digest},
                )
            )
        try:
            body = json.loads(raw)
        except (TypeError, ValueError) as exc:
            self._fail(
                RecordError(
                    "可见性记录 JSON 损坏",
                    transaction_id=transaction_id,
                    partition_id=row["partition_id"],
                    event_seq=event_seq,
                    record_seq=event_seq,
                    field="body_json",
                    details={"error": str(exc)},
                )
            )
        if not isinstance(body, dict):
            self._fail(
                RecordError(
                    "可见性记录体必须是对象",
                    transaction_id=transaction_id,
                    event_seq=event_seq,
                    record_seq=event_seq,
                    field="body_json",
                )
            )
        if canonical_bytes(body).decode("utf-8") != raw:
            self._fail(
                RecordError(
                    "可见性记录不是规范化 JSON",
                    transaction_id=transaction_id,
                    event_seq=event_seq,
                    record_seq=event_seq,
                    field="body_json",
                )
            )
        expected = normalise_visibility(body)
        columns = {
            "effective_event_seq": event_seq,
            "message_digest": row["message_digest"],
            "partition_id": row["partition_id"],
            "position": int(row["position"]),
            "transaction_id": transaction_id,
        }
        for key, value in expected.items():
            if value != columns[key]:
                self._fail(
                    RecordError(
                        "可见性字段与记录体不一致",
                        transaction_id=transaction_id,
                        partition_id=row["partition_id"],
                        event_seq=event_seq,
                        record_seq=event_seq,
                        field=key,
                        details={"column": columns[key], "body": value},
                    )
                )
        payload = body.get("payload")
        computed_message_digest = message_digest(
            row["partition_id"], int(row["position"]), payload
        )
        if computed_message_digest != row["message_digest"]:
            self._fail(
                RecordError(
                    "可见性记录消息摘要不匹配",
                    transaction_id=transaction_id,
                    partition_id=row["partition_id"],
                    event_seq=event_seq,
                    record_seq=event_seq,
                    field="message_digest",
                    details={"stored": row["message_digest"], "actual": computed_message_digest},
                )
            )
        message_row = connection.execute(
            "SELECT * FROM messages WHERE transaction_id=? AND position=?",
            (transaction_id, int(row["position"])),
        ).fetchone()
        if message_row is None:
            self._fail(
                RecordError(
                    "可见性记录找不到消息源",
                    transaction_id=transaction_id,
                    partition_id=row["partition_id"],
                    event_seq=event_seq,
                    record_seq=event_seq,
                    field="position",
                )
            )
        if json.loads(message_row["payload_json"]) != payload:
            self._fail(
                RecordError(
                    "可见性载荷与消息源不一致",
                    transaction_id=transaction_id,
                    partition_id=row["partition_id"],
                    event_seq=event_seq,
                    record_seq=event_seq,
                    field="payload",
                )
            )
        return expected

    def verify_records(self) -> None:
        try:
            self._verify_records_impl()
        except BusinessError as error:
            self._logger.write_error(error)
            raise
        except sqlite3.DatabaseError as exc:
            error = RecordError("账本数据库读取失败", field="database", details={"error": str(exc)})
            self._logger.write_error(error)
            raise error from exc

    def _verify_records_impl(self) -> None:
        format_version = self._meta_int("ledger_format_version")
        if format_version != FORMAT_VERSION:
            self._fail(
                SnapshotError(
                    "账本格式版本不受支持",
                    field="ledger_format_version",
                    details={"supported": FORMAT_VERSION, "stored": format_version},
                )
            )
        partitions = self.partitions()
        transactions = self.transactions()
        transaction_map = {transaction.transaction_id: transaction for transaction in transactions}

        event_rows = self.connection.execute("SELECT * FROM events ORDER BY event_seq").fetchall()
        events: list[dict[str, Any]] = []
        for index, row in enumerate(event_rows, start=1):
            if int(row["event_seq"]) != index:
                self._fail(
                    RecordError(
                        "事件序号不连续",
                        transaction_id=row["transaction_id"],
                        partition_id=row["partition_id"],
                        event_seq=int(row["event_seq"]),
                        record_seq=int(row["event_seq"]),
                        field="event_seq",
                        details={"expected": index, "actual": int(row["event_seq"])},
                    )
                )
            events.append(self._validate_event_row(row))
            transaction = transaction_map.get(row["transaction_id"])
            if transaction is None:
                self._fail(
                    RecordError(
                        "事件引用不存在的事务",
                        transaction_id=row["transaction_id"],
                        event_seq=int(row["event_seq"]),
                        record_seq=int(row["event_seq"]),
                        field="transaction_id",
                    )
                )
            if row["partition_id"] != transaction.partition_id:
                self._fail(
                    RecordError(
                        "事件分区归属与事务不一致",
                        transaction_id=row["transaction_id"],
                        partition_id=row["partition_id"],
                        event_seq=int(row["event_seq"]),
                        record_seq=int(row["event_seq"]),
                        field="partition_id",
                    )
                )
            if int(row["local_txn_version"]) != transaction.local_txn_version:
                self._fail(
                    RecordError(
                        "事件本地事务版本与事务不一致",
                        transaction_id=row["transaction_id"],
                        event_seq=int(row["event_seq"]),
                        record_seq=int(row["event_seq"]),
                        field="local_txn_version",
                    )
                )

        visibility_rows = self.connection.execute(
            "SELECT * FROM visibility ORDER BY effective_event_seq, partition_id, position, transaction_id"
        ).fetchall()
        visibility: list[dict[str, Any]] = []
        for row in visibility_rows:
            visibility.append(self._validate_visibility_row(self.connection, row))

        self._verify_messages_and_specs(transaction_map)
        try:
            outcome = replay_events(
                partitions,
                transactions,
                events,
                visibility=visibility,
                expected_cursor=len(event_rows),
            )
        except ReplayMismatch as mismatch:
            self._fail(
                RecordError(
                    "事件回放校验失败",
                    transaction_id=mismatch.transaction_id,
                    event_seq=mismatch.event_seq,
                    record_seq=mismatch.event_seq,
                    field=mismatch.field,
                    details={"expected": mismatch.expected, "actual": mismatch.actual},
                )
            )

        state_rows = self.connection.execute(
            "SELECT transaction_id, current_state, last_event_seq FROM transactions ORDER BY transaction_id"
        ).fetchall()
        for row in state_rows:
            expected_state = outcome.states.get(row["transaction_id"])
            expected_seq = outcome.final_event_seq.get(row["transaction_id"])
            if row["current_state"] != expected_state:
                self._fail(
                    RecordError(
                        "事务物化状态与事件流不一致",
                        transaction_id=row["transaction_id"],
                        event_seq=expected_seq,
                        record_seq=expected_seq,
                        field="current_state",
                        details={"stored": row["current_state"], "derived": expected_state},
                    )
                )
            if row["last_event_seq"] != expected_seq:
                self._fail(
                    RecordError(
                        "事务最后事件游标与事件流不一致",
                        transaction_id=row["transaction_id"],
                        event_seq=expected_seq,
                        record_seq=expected_seq,
                        field="last_event_seq",
                        details={"stored": row["last_event_seq"], "derived": expected_seq},
                    )
                )

        stored_digest = self._meta_value("ledger_digest")
        actual_digest = self.computed_ledger_digest()
        if stored_digest != actual_digest:
            self._fail(
                SnapshotError(
                    "账本内容摘要不符",
                    field="ledger_digest",
                    details={"stored": stored_digest, "actual": actual_digest},
                )
            )

    def _verify_messages_and_specs(self, transaction_map: dict[str, Transaction]) -> None:
        state_rows = {
            row["transaction_id"]: row
            for row in self.connection.execute(
                "SELECT transaction_id, last_event_seq FROM transactions"
            ).fetchall()
        }
        message_rows = self.connection.execute(
            "SELECT * FROM messages ORDER BY transaction_id, position"
        ).fetchall()
        for row in message_rows:
            transaction = transaction_map.get(row["transaction_id"])
            last_event_seq = state_rows.get(row["transaction_id"], {})["last_event_seq"] if row["transaction_id"] in state_rows else None
            if transaction is None:
                self._fail(
                    RecordError(
                        "消息引用不存在的事务",
                        transaction_id=row["transaction_id"],
                        partition_id=row["partition_id"],
                        field="transaction_id",
                        event_seq=last_event_seq,
                        record_seq=last_event_seq,
                    )
                )
            raw = row["payload_json"]
            encoded = raw.encode("utf-8") if isinstance(raw, str) else b""
            if int(row["payload_length"]) != len(encoded):
                self._fail(
                    RecordError(
                        "消息载荷长度字段不自治",
                        transaction_id=row["transaction_id"],
                        partition_id=row["partition_id"],
                        field="payload_length",
                        event_seq=last_event_seq,
                        record_seq=last_event_seq,
                        details={"stored": int(row["payload_length"]), "actual": len(encoded)},
                    )
                )
            try:
                payload = json.loads(raw)
            except (TypeError, ValueError) as exc:
                self._fail(
                    RecordError(
                        "消息载荷 JSON 损坏",
                        transaction_id=row["transaction_id"],
                        partition_id=row["partition_id"],
                        field="payload_json",
                        event_seq=last_event_seq,
                        record_seq=last_event_seq,
                        details={"error": str(exc)},
                    )
                )
            if canonical_bytes(payload).decode("utf-8") != raw:
                self._fail(
                    RecordError(
                        "消息载荷不是规范化 JSON",
                        transaction_id=row["transaction_id"],
                        partition_id=row["partition_id"],
                        field="payload_json",
                        event_seq=last_event_seq,
                        record_seq=last_event_seq,
                    )
                )
            digest = message_digest(row["partition_id"], int(row["position"]), payload)
            if digest != row["message_digest"]:
                self._fail(
                    RecordError(
                        "消息摘要不匹配",
                        transaction_id=row["transaction_id"],
                        partition_id=row["partition_id"],
                        field="message_digest",
                        event_seq=last_event_seq,
                        record_seq=last_event_seq,
                        details={"stored": row["message_digest"], "actual": digest},
                    )
                )

        tx_rows = self.connection.execute("SELECT * FROM transactions ORDER BY txn_seq").fetchall()
        for row in tx_rows:
            transaction = transaction_map.get(row["transaction_id"])
            last_event_seq = state_rows.get(row["transaction_id"], {})["last_event_seq"] if row["transaction_id"] in state_rows else None
            if transaction is None:
                self._fail(
                    RecordError(
                        "事务目录记录缺失",
                        transaction_id=row["transaction_id"],
                        field="transaction_id",
                        event_seq=last_event_seq,
                        record_seq=last_event_seq,
                    )
                )
            raw = row["spec_json"]
            encoded = raw.encode("utf-8") if isinstance(raw, str) else b""
            if int(row["spec_length"]) != len(encoded):
                self._fail(
                    RecordError(
                        "事务规格长度字段不自治",
                        transaction_id=row["transaction_id"],
                        event_seq=last_event_seq,
                        record_seq=last_event_seq,
                        field="spec_length",
                    )
                )
            digest = digest_bytes(encoded)
            if row["spec_digest"] != digest:
                self._fail(
                    RecordError(
                        "事务规格摘要不匹配",
                        transaction_id=row["transaction_id"],
                        event_seq=last_event_seq,
                        record_seq=last_event_seq,
                        field="spec_digest",
                    )
                )
            if transaction.spec_digest != row["spec_digest"]:
                self._fail(
                    RecordError(
                        "事务规格摘要与消息目录不一致",
                        transaction_id=row["transaction_id"],
                        event_seq=last_event_seq,
                        record_seq=last_event_seq,
                        field="spec_digest",
                    )
                )

    # ------------------------------------------------------------------
    # Snapshot support primitives
    # ------------------------------------------------------------------
    def backup_to(self, destination: str | os.PathLike[str]) -> str:
        if self.readonly:
            self._fail(RecordError("只读账本不能创建一致性备份", field="readonly"))
        destination_path = Path(destination)
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_path.exists():
            destination_path.unlink()
        destination_connection = sqlite3.connect(str(destination_path))
        try:
            self.connection.backup(destination_connection)
            destination_connection.commit()
        finally:
            destination_connection.close()
        try:
            os.chmod(destination_path, 0o444)
        except OSError:
            pass
        return str(destination_path)

    def replace_all_state(
        self,
        partitions: Sequence[Partition],
        transactions: Sequence[Transaction],
        events: Sequence[dict[str, Any]],
        visibility: Sequence[dict[str, Any]],
        *,
        partition_set_version: int,
        transaction_set_version: int,
        ledger_format_version: int = FORMAT_VERSION,
    ) -> None:
        """Wholesale replacement used by snapshot restore (never incremental)."""

        if self.readonly:
            self._fail(RecordError("只读账本不能整体替换", field="readonly"))
        transaction_map = {transaction.transaction_id: transaction for transaction in transactions}
        try:
            with self._write() as connection:
                connection.execute("DELETE FROM recovery_marks")
                connection.execute("DELETE FROM visibility")
                connection.execute("DELETE FROM events")
                connection.execute("DELETE FROM partition_prefix")
                connection.execute("DELETE FROM messages")
                connection.execute("DELETE FROM transactions")
                connection.execute("DELETE FROM partitions")

                for partition in sorted(partitions, key=lambda item: item.partition_id):
                    connection.execute(
                        "INSERT INTO partitions(partition_id, partition_version, max_position) VALUES(?,?,?)",
                        (partition.partition_id, partition.partition_version, partition.max_position),
                    )
                for transaction in sorted(transactions, key=lambda item: (item.txn_seq, item.transaction_id)):
                    spec_text, spec_length, spec_digest = _encoded_record(
                        transaction.as_spec_dict(include_digest=False)
                    )
                    connection.execute(
                        """
                        INSERT INTO transactions(
                            transaction_id, txn_seq, partition_id, local_txn_version,
                            commit_time_ms, spec_json, spec_length, spec_digest,
                            current_state, last_event_seq
                        ) VALUES(?,?,?,?,?,?,?,?,NULL,NULL)
                        """,
                        (
                            transaction.transaction_id,
                            transaction.txn_seq,
                            transaction.partition_id,
                            transaction.local_txn_version,
                            transaction.commit_time_ms,
                            spec_text,
                            spec_length,
                            spec_digest,
                        ),
                    )
                    for message in transaction.messages:
                        payload_text, payload_length = _payload_record(message.payload)
                        connection.execute(
                            "INSERT INTO messages(transaction_id, partition_id, position, payload_json, payload_length, message_digest) VALUES(?,?,?,?,?,?)",
                            (
                                transaction.transaction_id,
                                transaction.partition_id,
                                message.position,
                                payload_text,
                                payload_length,
                                message.message_digest,
                            ),
                        )

                normalised_events = [normalise_event(event) for event in events]
                normalised_events.sort(key=lambda event: event["event_seq"])
                for event in normalised_events:
                    event_text, event_length, event_digest = _encoded_record(event)
                    connection.execute(
                        "INSERT INTO events(event_seq, transaction_id, state, local_txn_version, partition_id, body_json, body_length, body_digest) VALUES(?,?,?,?,?,?,?,?)",
                        (
                            event["event_seq"],
                            event["transaction_id"],
                            event["state"],
                            event["local_txn_version"],
                            event["partition_id"],
                            event_text,
                            event_length,
                            event_digest,
                        ),
                    )

                for row in sorted(
                    (normalise_visibility(item) for item in visibility),
                    key=lambda item: (item["effective_event_seq"], item["partition_id"], item["position"]),
                ):
                    transaction = transaction_map.get(row["transaction_id"])
                    if transaction is None:
                        self._fail(
                            RecordError(
                                "可见性记录引用不存在的事务",
                                transaction_id=row["transaction_id"],
                                partition_id=row["partition_id"],
                                event_seq=row["effective_event_seq"],
                                field="transaction_id",
                            )
                        )
                    message = transaction.message_at(int(row["position"]))
                    if message is None:
                        self._fail(
                            RecordError(
                                "可见性记录引用不存在的消息",
                                transaction_id=row["transaction_id"],
                                partition_id=row["partition_id"],
                                event_seq=row["effective_event_seq"],
                                field="position",
                            )
                        )
                    body = {
                        "transaction_id": row["transaction_id"],
                        "partition_id": row["partition_id"],
                        "position": row["position"],
                        "message_digest": row["message_digest"],
                        "effective_event_seq": row["effective_event_seq"],
                        "payload": message.payload,
                    }
                    body_text, body_length, body_digest = _encoded_record(body)
                    connection.execute(
                        "INSERT INTO visibility(transaction_id, partition_id, position, message_digest, effective_event_seq, body_json, body_length, body_digest) VALUES(?,?,?,?,?,?,?,?)",
                        (
                            row["transaction_id"],
                            row["partition_id"],
                            row["position"],
                            row["message_digest"],
                            row["effective_event_seq"],
                            body_text,
                            body_length,
                            body_digest,
                        ),
                    )

                for event_row in connection.execute(
                    "SELECT * FROM events ORDER BY event_seq"
                ).fetchall():
                    self._validate_event_row(event_row)
                for visibility_row in connection.execute(
                    "SELECT * FROM visibility ORDER BY effective_event_seq, partition_id, position"
                ).fetchall():
                    self._validate_visibility_row(connection, visibility_row)

                try:
                    outcome = replay_events(
                        partitions,
                        transactions,
                        normalised_events,
                        visibility=visibility,
                        expected_cursor=len(normalised_events),
                    )
                except ReplayMismatch as mismatch:
                    self._fail(
                        SnapshotError(
                            "快照事件流无法重放",
                            transaction_id=mismatch.transaction_id,
                            event_seq=mismatch.event_seq,
                            field=mismatch.field,
                            details={"expected": mismatch.expected, "actual": mismatch.actual},
                        )
                    )
                for transaction in transactions:
                    connection.execute(
                        "UPDATE transactions SET current_state=?, last_event_seq=? WHERE transaction_id=?",
                        (
                            outcome.states.get(transaction.transaction_id),
                            outcome.final_event_seq.get(transaction.transaction_id),
                            transaction.transaction_id,
                        ),
                    )
                meta_values = {
                    "ledger_format_version": str(ledger_format_version),
                    "partition_set_version": str(partition_set_version),
                    "transaction_set_version": str(transaction_set_version),
                }
                connection.executemany(
                    "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    list(meta_values.items()),
                )
                self._recompute_cache(connection)
                self._update_ledger_digest(connection)
        except BusinessError:
            raise
        except sqlite3.DatabaseError as exc:
            self._fail(
                SnapshotError(
                    "快照整体替换失败",
                    field="database",
                    details={"error": str(exc)},
                )
            )
