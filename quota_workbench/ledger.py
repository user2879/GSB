"""SQLite ledger persistence, batch transactions and crash recovery.

Every probe-transition commit and every arbitration batch commit happens
in a single SQLite transaction that writes request outcomes, basis
sequences, tenant quota balances, in-flight counts, instance health
state and the progress cursor together. The ledger carries an integer
format version and a content digest over its logical content.

After a forced kill, recovery trusts only fully committed records:
requests still marked in_flight in committed batches were granted but
never finalized, so they are marked 'unknown' and must be re-run.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3

from . import FORMAT_VERSION
from .arbiter import RunResult
from .probe import Transition
from .slog import StructuredLogger, NULL_LOGGER
from .spec import Spec

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS transitions (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    instance_id TEXT NOT NULL,
    from_state TEXT NOT NULL,
    to_state TEXT NOT NULL,
    probe_round INTEGER NOT NULL,
    consec_failures INTEGER NOT NULL,
    mono_ms INTEGER NOT NULL,
    virt_ms INTEGER NOT NULL,
    reason TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS requests (
    request_id TEXT PRIMARY KEY,
    seq INTEGER NOT NULL,
    tenant_id TEXT NOT NULL,
    demand INTEGER NOT NULL,
    arrival_ms INTEGER NOT NULL,
    state TEXT NOT NULL,
    outcome TEXT,
    basis TEXT,
    wait_start_ms INTEGER,
    wait_end_ms INTEGER,
    grant_ms INTEGER,
    done_ms INTEGER
);
CREATE TABLE IF NOT EXISTS tenants (
    tenant_id TEXT PRIMARY KEY,
    share INTEGER NOT NULL,
    consumed INTEGER NOT NULL,
    inflight INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS cursor (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS propagation (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    change TEXT NOT NULL,
    affected TEXT NOT NULL,
    basis TEXT NOT NULL
);
"""


def _canonical_digest(rows_requests, rows_transitions, rows_tenants) -> str:
    payload = {
        "requests": [
            [r[0], r[1], r[2]] for r in rows_requests
        ],
        "transitions": [list(r) for r in rows_transitions],
        "tenants": [list(r) for r in rows_tenants],
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def compute_content_digest(conn: sqlite3.Connection) -> str:
    rows_requests = conn.execute(
        "SELECT request_id, COALESCE(outcome, state), COALESCE(basis, '') FROM requests ORDER BY request_id"
    ).fetchall()
    rows_transitions = conn.execute(
        "SELECT instance_id, from_state, to_state, probe_round, consec_failures, virt_ms, reason "
        "FROM transitions ORDER BY seq"
    ).fetchall()
    rows_tenants = conn.execute(
        "SELECT tenant_id, consumed, inflight FROM tenants ORDER BY tenant_id"
    ).fetchall()
    return _canonical_digest(rows_requests, rows_transitions, rows_tenants)


class LedgerSession:
    """Persistence sink used during a run. Commits every `batch_size`
    finalized requests in one transaction. If `crash_after_batch` is set,
    the process hard-exits (os._exit, simulating a forced kill) right
    after that batch commits."""

    def __init__(self, db_path: str, spec: Spec, batch_size: int = 25,
                 crash_after_batch: int | None = None,
                 logger: StructuredLogger = NULL_LOGGER) -> None:
        self.db_path = db_path
        self.spec = spec
        self.batch_size = max(1, batch_size)
        self.crash_after_batch = crash_after_batch
        self.logger = logger
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        if os.path.exists(db_path):
            os.remove(db_path)
        self.conn = sqlite3.connect(db_path)
        self.conn.isolation_level = None
        self.conn.executescript(SCHEMA)
        self._pending_finals = 0
        self._batch_no = 0
        self._finalized_total = 0
        self._inflight = {tid: 0 for tid in spec.tenants}
        self._consumed = {tid: 0 for tid in spec.tenants}
        self._request_seq = 0
        self._txn_open = False
        self._begin_run()

    def _begin_run(self) -> None:
        self.conn.execute("BEGIN")
        meta = {
            "format_version": str(FORMAT_VERSION),
            "spec_digest": self.spec.digest(),
            "tenant_set_version": str(self.spec.tenant_set_version),
            "instance_set_version": str(self.spec.instance_set_version),
            "probe_policy_version": str(self.spec.probe_policy.policy_version),
            "rule_version": str(self.spec.rule_version),
        }
        for key, value in meta.items():
            self.conn.execute("INSERT INTO meta(key, value) VALUES(?, ?)", (key, value))
        for tid in sorted(self.spec.tenants):
            self.conn.execute(
                "INSERT INTO tenants(tenant_id, share, consumed, inflight) VALUES(?, ?, 0, 0)",
                (tid, self.spec.tenants[tid].share),
            )
        self.conn.execute("INSERT INTO cursor(key, value) VALUES('finalized_requests', 0)")
        self.conn.execute("COMMIT")

    def start_arbitration(self) -> None:
        """Open the transaction that accumulates grant/finalize writes until
        the next batch commit."""
        if not self._txn_open:
            self.conn.execute("BEGIN")
            self._txn_open = True

    def commit_probe_transitions(self, transitions: list) -> None:
        """Commit all probe transitions plus health state in one transaction."""
        self.conn.execute("BEGIN")
        for t in transitions:
            self.conn.execute(
                "INSERT INTO transitions(instance_id, from_state, to_state, probe_round,"
                " consec_failures, mono_ms, virt_ms, reason) VALUES(?,?,?,?,?,?,?,?)",
                (t.instance_id, t.from_state, t.to_state, t.probe_round,
                 t.consec_failures, t.mono_ms, t.virt_ms, t.reason),
            )
        self._store_digest_locked()
        self.conn.execute("COMMIT")
        self.logger.log("ledger_commit", kind="probe", transitions=len(transitions))

    def record_propagation(self, change: str, affected: list, basis: list) -> None:
        self.conn.execute("BEGIN")
        self.conn.execute(
            "INSERT INTO propagation(change, affected, basis) VALUES(?,?,?)",
            (change, json.dumps(affected), json.dumps(basis)),
        )
        self.conn.execute("COMMIT")

    # --- arbitration sink interface -------------------------------------

    def on_grant(self, outcome, now_ms: int) -> None:
        self._request_seq += 1
        self._inflight[outcome.tenant_id] += 1
        self._consumed[outcome.tenant_id] += outcome.demand
        self.conn.execute(
            "INSERT OR REPLACE INTO requests(request_id, seq, tenant_id, demand, arrival_ms,"
            " state, outcome, basis, wait_start_ms, wait_end_ms, grant_ms, done_ms)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (outcome.request_id, self._request_seq, outcome.tenant_id, outcome.demand,
             outcome.arrival_ms, "in_flight", None, None,
             _none_if_neg(outcome.wait_start_ms), None, now_ms, None),
        )

    def on_finalize(self, outcome) -> None:
        self._inflight[outcome.tenant_id] = max(0, self._inflight[outcome.tenant_id] - 1) \
            if outcome.outcome == "granted" else self._inflight[outcome.tenant_id]
        self._request_seq += 1
        self.conn.execute(
            "INSERT OR REPLACE INTO requests(request_id, seq, tenant_id, demand, arrival_ms,"
            " state, outcome, basis, wait_start_ms, wait_end_ms, grant_ms, done_ms)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (outcome.request_id, self._request_seq, outcome.tenant_id, outcome.demand,
             outcome.arrival_ms, "final", outcome.outcome, outcome.basis_blob(),
             _none_if_neg(outcome.wait_start_ms), _none_if_neg(outcome.wait_end_ms),
             _none_if_neg(outcome.grant_ms), _none_if_neg(outcome.done_ms)),
        )
        self._pending_finals += 1
        self._finalized_total += 1
        if self._pending_finals >= self.batch_size:
            self._commit_batch()

    # --------------------------------------------------------------------

    def _commit_batch(self) -> None:
        self._write_state_locked()
        self.conn.execute("COMMIT")
        self._txn_open = False
        self._batch_no += 1
        self._pending_finals = 0
        self.logger.log("ledger_commit", kind="arbitration", batch=self._batch_no,
                        finalized=self._finalized_total)
        if self.crash_after_batch is not None and self._batch_no == self.crash_after_batch:
            self.logger.log("crash_injected", batch=self._batch_no)
            self.conn.close()
            os._exit(3)
        self.start_arbitration()

    def _write_state_locked(self) -> None:
        for tid in sorted(self._consumed):
            self.conn.execute(
                "UPDATE tenants SET consumed=?, inflight=? WHERE tenant_id=?",
                (self._consumed[tid], self._inflight[tid], tid),
            )
        self.conn.execute(
            "UPDATE cursor SET value=? WHERE key='finalized_requests'", (self._finalized_total,))
        self._store_digest_locked()

    def _store_digest_locked(self) -> None:
        digest = compute_content_digest(self.conn)
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES('content_digest', ?)"
            " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (digest,),
        )

    def finalize_run(self) -> None:
        """Flush any remaining pending writes in a final transaction."""
        if not self._txn_open:
            self.conn.execute("BEGIN")
            self._txn_open = True
        self._write_state_locked()
        self.conn.execute("COMMIT")
        self._txn_open = False

    def close(self) -> None:
        self.conn.close()


def _none_if_neg(value: int):
    return None if value is None or value < 0 else value


def recover(db_path: str, logger: StructuredLogger = NULL_LOGGER) -> dict:
    """Recover after a forced kill. Only fully committed records are
    trusted; committed-but-still-in_flight requests are marked 'unknown'
    and must be re-run. No exit report for the killed run is produced."""
    conn = sqlite3.connect(db_path)
    conn.isolation_level = None
    meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    inflight_rows = conn.execute(
        "SELECT request_id, tenant_id FROM requests WHERE state='in_flight' ORDER BY request_id"
    ).fetchall()
    conn.execute("BEGIN")
    conn.execute("UPDATE requests SET state='unknown', outcome=NULL WHERE state='in_flight'")
    digest = compute_content_digest(conn)
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('content_digest', ?)"
        " ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (digest,),
    )
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('recovered', '1')"
        " ON CONFLICT(key) DO UPDATE SET value=excluded.value"
    )
    conn.execute("COMMIT")
    cursor_row = conn.execute(
        "SELECT value FROM cursor WHERE key='finalized_requests'").fetchone()
    report = {
        "format_version": int(meta.get("format_version", "0")),
        "unknown_requests": [r[0] for r in inflight_rows],
        "finalized_committed": cursor_row[0] if cursor_row else 0,
        "content_digest": digest,
    }
    conn.close()
    logger.log("ledger_recover", unknown=len(inflight_rows),
               finalized_committed=report["finalized_committed"])
    return report


def load_committed(db_path: str) -> dict:
    """Read the committed ledger content for replay comparison / GUI."""
    conn = sqlite3.connect(db_path)
    meta = dict(conn.execute("SELECT key, value FROM meta").fetchall())
    requests = conn.execute(
        "SELECT request_id, tenant_id, demand, arrival_ms, state, outcome, basis,"
        " wait_start_ms, wait_end_ms, grant_ms, done_ms FROM requests ORDER BY request_id"
    ).fetchall()
    transitions = conn.execute(
        "SELECT instance_id, from_state, to_state, probe_round, consec_failures, virt_ms, reason"
        " FROM transitions ORDER BY seq"
    ).fetchall()
    tenants = conn.execute(
        "SELECT tenant_id, share, consumed, inflight FROM tenants ORDER BY tenant_id"
    ).fetchall()
    propagation = conn.execute(
        "SELECT change, affected, basis FROM propagation ORDER BY seq"
    ).fetchall()
    conn.close()
    return {
        "meta": meta,
        "requests": requests,
        "transitions": transitions,
        "tenants": tenants,
        "propagation": propagation,
    }
