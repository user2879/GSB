"""SQLite ledger, preview cache snapshots, version replacement, and recovery state."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from .errors import GatewayError, ReplayError, SpecError
from .model import (
    RequestSample,
    Spec,
    canonical_json,
    parse_spec_obj,
    request_stream_digest,
)
from .routing import BatchPlan, RoutingPlan
from .scheduling import BatchExecution

LEDGER_FORMAT_VERSION = 1
SCHEMA_VERSION = 1


def _wall_ms() -> int:
    return time.time_ns() // 1_000_000


def _content_hash(prefix: bytes, value: Any) -> str:
    return hashlib.sha256(prefix + canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ActivationResult:
    revision: int
    fingerprint: str
    request_stream_digest: str
    changed: bool
    diff: Mapping[str, Any]
    diff_json: str


@dataclass(frozen=True)
class PreviewEntry:
    request_id: int
    client_key: str
    arrival_ms: int
    rule_id: str
    target: str
    backend_id: str
    bucket: int
    weight_bp: int
    weight_version: int
    routing_version: int
    batch_id: str
    output_digest: str
    wait_ms: int
    exec_ms: int
    parameter_digest: str
    receipt_digest: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "client_key": self.client_key,
            "arrival_ms": self.arrival_ms,
            "rule_id": self.rule_id,
            "target": self.target,
            "backend_id": self.backend_id,
            "bucket": self.bucket,
            "weight_bp": self.weight_bp,
            "weight_version": self.weight_version,
            "routing_version": self.routing_version,
            "batch_id": self.batch_id,
            "output_digest": self.output_digest,
            "wait_ms": self.wait_ms,
            "exec_ms": self.exec_ms,
            "parameter_digest": self.parameter_digest,
            "receipt_digest": self.receipt_digest,
        }


@dataclass(frozen=True)
class PreviewSnapshot:
    revision: int
    entries: tuple[PreviewEntry, ...]
    stats: Mapping[str, Any]

    def query(self, request_id: int) -> PreviewEntry | None:
        for entry in self.entries:
            if entry.request_id == request_id:
                return entry
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "entries": [entry.to_dict() for entry in self.entries],
            "stats": self.stats,
        }

    def canonical_json(self) -> str:
        return canonical_json(self.to_dict())

    def canonical_bytes(self) -> bytes:
        return self.canonical_json().encode("utf-8")


class PreviewCache:
    """In-memory read model populated by exactly one joined ledger query."""

    def __init__(self, snapshot: PreviewSnapshot) -> None:
        self.snapshot = snapshot
        self._entries = {entry.request_id: entry for entry in snapshot.entries}

    @property
    def revision(self) -> int:
        return self.snapshot.revision

    @property
    def stats(self) -> Mapping[str, Any]:
        return self.snapshot.stats

    @property
    def entries(self) -> tuple[PreviewEntry, ...]:
        return self.snapshot.entries

    def get(self, request_id: int) -> PreviewEntry | None:
        return self._entries.get(request_id)

    def canonical_bytes(self) -> bytes:
        return self.snapshot.canonical_bytes()


@dataclass(frozen=True)
class RecoveryReport:
    revision: int
    ledger_integrity_ok: bool
    ledger_content_digest: str
    expected_batches: int
    committed_batches: int
    expected_requests: int
    committed_requests: int
    missing_batches: tuple[str, ...]
    missing_request_ids: tuple[int, ...]
    extra_batches: tuple[str, ...]
    extra_request_ids: tuple[int, ...]
    unknown_batch_ids: tuple[str, ...]
    needs_rerun: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "revision": self.revision,
            "ledger_integrity_ok": self.ledger_integrity_ok,
            "ledger_content_digest": self.ledger_content_digest,
            "expected_batches": self.expected_batches,
            "committed_batches": self.committed_batches,
            "expected_requests": self.expected_requests,
            "committed_requests": self.committed_requests,
            "missing_batches": list(self.missing_batches),
            "missing_request_ids": list(self.missing_request_ids),
            "extra_batches": list(self.extra_batches),
            "extra_request_ids": list(self.extra_request_ids),
            "unknown_batch_ids": list(self.unknown_batch_ids),
            "needs_rerun": self.needs_rerun,
        }


def _inventory_diff(previous: Spec | None, current: Spec,
                    previous_revision: int | None) -> dict[str, Any]:
    old_inventory = previous.inventory() if previous is not None else {"backends": {}, "rules": {}}
    new_inventory = current.inventory()
    old_backends = old_inventory["backends"]
    new_backends = new_inventory["backends"]
    old_rules = old_inventory["rules"]
    new_rules = new_inventory["rules"]
    old_backend_objects = (
        previous.backends_by_id if previous is not None else {}
    )
    new_backend_objects = current.backends_by_id
    changed_backends = [
        {
            "id": backend_id,
            "old_digest": old_backends[backend_id],
            "new_digest": new_backends[backend_id],
            "changed_fields": [
                field
                for field in (
                    "host",
                    "port",
                    "model_dim",
                    "hidden_dim",
                    "seed",
                    "quantization",
                )
                if getattr(old_backend_objects[backend_id], field)
                != getattr(new_backend_objects[backend_id], field)
            ],
        }
        for backend_id in sorted(set(old_backends) & set(new_backends))
        if old_backends[backend_id] != new_backends[backend_id]
    ]
    changed_rules = [
        {
            "id": rule_id,
            "old_version": int(old_rules[rule_id]),
            "new_version": int(new_rules[rule_id]),
        }
        for rule_id in sorted(set(old_rules) & set(new_rules))
        if old_rules[rule_id] != new_rules[rule_id]
    ]
    return {
        "from_revision": previous_revision,
        "to_revision": current.revision,
        "replaced_whole_view": True,
        "backends": {
            "added": sorted(set(new_backends) - set(old_backends)),
            "removed": sorted(set(old_backends) - set(new_backends)),
            "changed": changed_backends,
        },
        "rules": {
            "added": sorted(set(new_rules) - set(old_rules)),
            "removed": sorted(set(old_rules) - set(new_rules)),
            "version_changed": changed_rules,
        },
    }


class LedgerStore:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).resolve().parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(
            self.path,
            timeout=30.0,
            isolation_level=None,
            check_same_thread=False,
        )
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        with self.lock:
            self.connection.execute("PRAGMA foreign_keys = ON")
            self.connection.execute("PRAGMA busy_timeout = 30000")
            self.connection.execute("PRAGMA synchronous = FULL")
            if self.path != ":memory:":
                self.connection.execute("PRAGMA journal_mode = WAL")
            self._init_schema()

    def _init_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS ledger_header (
                singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                format_version INTEGER NOT NULL,
                content_digest TEXT NOT NULL,
                active_revision INTEGER
            );
            CREATE TABLE IF NOT EXISTS spec_versions (
                revision INTEGER PRIMARY KEY,
                fingerprint TEXT NOT NULL,
                spec_json TEXT NOT NULL,
                request_stream_digest TEXT NOT NULL,
                format_version INTEGER NOT NULL,
                content_digest TEXT NOT NULL,
                activated_at_ms INTEGER NOT NULL,
                diff_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS batches (
                revision INTEGER NOT NULL,
                batch_id TEXT NOT NULL,
                rule_id TEXT NOT NULL,
                backend_id TEXT NOT NULL,
                target TEXT NOT NULL,
                batch_size INTEGER NOT NULL,
                request_ids_json TEXT NOT NULL,
                parameter_digest TEXT NOT NULL,
                receipt_digest TEXT NOT NULL,
                receipt_json TEXT NOT NULL,
                output_digest TEXT NOT NULL,
                wait_start_ms INTEGER NOT NULL,
                wait_end_ms INTEGER NOT NULL,
                exec_start_ms INTEGER NOT NULL,
                exec_end_ms INTEGER NOT NULL,
                elapsed_ms INTEGER NOT NULL,
                worker_elapsed_ms INTEGER NOT NULL,
                worker_pid INTEGER NOT NULL,
                global_limit INTEGER NOT NULL,
                backend_quota INTEGER NOT NULL,
                thread_name TEXT NOT NULL,
                stale INTEGER NOT NULL DEFAULT 0,
                content_digest TEXT NOT NULL,
                PRIMARY KEY (revision, batch_id),
                FOREIGN KEY (revision) REFERENCES spec_versions(revision)
            );
            CREATE TABLE IF NOT EXISTS ledger_entries (
                revision INTEGER NOT NULL,
                request_id INTEGER NOT NULL,
                client_key TEXT NOT NULL,
                arrival_ms INTEGER NOT NULL,
                rule_id TEXT NOT NULL,
                target TEXT NOT NULL,
                backend_id TEXT NOT NULL,
                bucket INTEGER NOT NULL,
                weight_bp INTEGER NOT NULL,
                weight_version INTEGER NOT NULL,
                routing_version INTEGER NOT NULL,
                batch_id TEXT NOT NULL,
                output_digest TEXT NOT NULL,
                wait_ms INTEGER NOT NULL,
                exec_ms INTEGER NOT NULL,
                parameter_digest TEXT NOT NULL,
                receipt_digest TEXT NOT NULL,
                stale INTEGER NOT NULL DEFAULT 0,
                content_digest TEXT NOT NULL,
                PRIMARY KEY (revision, request_id),
                FOREIGN KEY (revision, batch_id) REFERENCES batches(revision, batch_id)
            );
            CREATE TABLE IF NOT EXISTS quota_usage (
                revision INTEGER NOT NULL,
                batch_id TEXT NOT NULL,
                backend_id TEXT NOT NULL,
                global_slots_used INTEGER NOT NULL,
                backend_slots_used INTEGER NOT NULL,
                global_limit INTEGER NOT NULL,
                backend_quota INTEGER NOT NULL,
                wait_ms INTEGER NOT NULL,
                exec_ms INTEGER NOT NULL,
                PRIMARY KEY (revision, batch_id),
                FOREIGN KEY (revision, batch_id) REFERENCES batches(revision, batch_id)
            );
            CREATE TABLE IF NOT EXISTS run_progress (
                revision INTEGER PRIMARY KEY,
                request_stream_digest TEXT NOT NULL,
                expected_batches INTEGER NOT NULL,
                expected_requests INTEGER NOT NULL,
                committed_batches INTEGER NOT NULL DEFAULT 0,
                committed_requests INTEGER NOT NULL DEFAULT 0,
                last_batch_id TEXT,
                last_request_id INTEGER,
                updated_at_ms INTEGER NOT NULL,
                FOREIGN KEY (revision) REFERENCES spec_versions(revision)
            );
            CREATE TABLE IF NOT EXISTS recovery_unknowns (
                revision INTEGER NOT NULL,
                batch_id TEXT NOT NULL,
                request_ids_json TEXT NOT NULL,
                status TEXT NOT NULL,
                reason TEXT NOT NULL,
                detected_at_ms INTEGER NOT NULL,
                resolved_at_ms INTEGER,
                PRIMARY KEY (revision, batch_id),
                FOREIGN KEY (revision) REFERENCES spec_versions(revision)
            );
            """
        )
        meta = {
            "schema_version": str(SCHEMA_VERSION),
            "ledger_format_version": str(LEDGER_FORMAT_VERSION),
            "ledger_content_digest": "",
        }
        for key, value in meta.items():
            self.connection.execute(
                "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO NOTHING",
                (key, value),
            )
        self.connection.execute(
            "INSERT INTO ledger_header(singleton, format_version, content_digest, active_revision) "
            "VALUES(1, ?, '', NULL) ON CONFLICT(singleton) DO NOTHING",
            (LEDGER_FORMAT_VERSION,),
        )
        legacy_digest = self._meta("ledger_content_digest")
        legacy_revision = self._meta("current_revision")
        if legacy_digest:
            self.connection.execute(
                "UPDATE ledger_header SET content_digest = ?, active_revision = ? "
                "WHERE singleton = 1 AND content_digest = ''",
                (
                    legacy_digest,
                    int(legacy_revision) if legacy_revision else None,
                ),
            )

    @contextmanager
    def _transaction(self) -> Iterator[None]:
        with self.lock:
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                yield
            except Exception:
                self.connection.execute("ROLLBACK")
                raise
            else:
                self.connection.execute("COMMIT")

    def close(self) -> None:
        with self.lock:
            self.connection.close()

    def __enter__(self) -> "LedgerStore":
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def _meta(self, key: str) -> str:
        row = self.connection.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return "" if row is None else str(row[0])

    def _set_meta(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, value),
        )

    def latest_revision(self) -> int | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT revision FROM spec_versions ORDER BY revision DESC LIMIT 1"
            ).fetchone()
            return None if row is None else int(row[0])

    def latest_spec_row(self) -> sqlite3.Row | None:
        with self.lock:
            return self.connection.execute(
                "SELECT * FROM spec_versions ORDER BY revision DESC LIMIT 1"
            ).fetchone()

    def activate_spec(
        self,
        spec: Spec,
        plan: RoutingPlan,
        requests: Sequence[RequestSample],
    ) -> ActivationResult:
        digest = request_stream_digest(requests)
        latest = self.latest_spec_row()
        with self.lock:
            if latest is not None:
                if spec.revision < int(latest["revision"]):
                    raise SpecError(
                        f"revision {spec.revision} is older than active revision {latest['revision']}",
                        entry="spec",
                        field="revision",
                    )
                if spec.revision == int(latest["revision"]):
                    if spec.fingerprint != latest["fingerprint"]:
                        raise SpecError(
                            "spec content changed without a monotonic revision increment",
                            entry="spec",
                            field="revision",
                        )
                    if digest != latest["request_stream_digest"]:
                        raise SpecError(
                            "request stream changed without a spec revision increment",
                            entry="requests",
                            field="request_stream_digest",
                        )
                    return ActivationResult(
                        revision=spec.revision,
                        fingerprint=spec.fingerprint,
                        request_stream_digest=digest,
                        changed=False,
                        diff=json.loads(latest["diff_json"]),
                        diff_json=latest["diff_json"],
                    )

                previous = parse_spec_obj(json.loads(latest["spec_json"]))
                previous_rules = previous.rules_by_id
                current_rules = spec.rules_by_id
                for index, rule in enumerate(spec.rules):
                    old_rule = previous_rules.get(rule.id)
                    if old_rule is None:
                        continue
                    if rule.version < old_rule.version:
                        raise SpecError(
                            f"rule version for {rule.id!r} must be monotonically increasing",
                            entry=f"rules[{index}]",
                            field="version",
                        )
                    if rule.content_digest != old_rule.content_digest and rule.version <= old_rule.version:
                        raise SpecError(
                            f"modified rule {rule.id!r} requires a higher version",
                            entry=f"rules[{index}]",
                            field="version",
                        )
                diff = _inventory_diff(previous, spec, int(latest["revision"]))
            else:
                diff = _inventory_diff(None, spec, None)
            diff_json = canonical_json(diff)
            spec_content_digest = _content_hash(
                b"gateway-spec-v1\0",
                {
                    "format_version": SCHEMA_VERSION,
                    "spec": spec.to_dict(),
                    "request_stream_digest": digest,
                },
            )
            with self._transaction():
                self.connection.execute(
                    "UPDATE batches SET stale = 1 WHERE revision <> ?", (spec.revision,)
                )
                self.connection.execute(
                    "UPDATE ledger_entries SET stale = 1 WHERE revision <> ?",
                    (spec.revision,),
                )
                self.connection.execute(
                    "INSERT INTO spec_versions("
                    "revision, fingerprint, spec_json, request_stream_digest, format_version, "
                    "content_digest, activated_at_ms, diff_json"
                    ") VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        spec.revision,
                        spec.fingerprint,
                        canonical_json(spec.to_dict()),
                        digest,
                        SCHEMA_VERSION,
                        spec_content_digest,
                        _wall_ms(),
                        diff_json,
                    ),
                )
                self.connection.execute(
                    "INSERT INTO run_progress("
                    "revision, request_stream_digest, expected_batches, expected_requests, "
                    "committed_batches, committed_requests, last_batch_id, last_request_id, updated_at_ms"
                    ") VALUES(?, ?, ?, ?, 0, 0, NULL, NULL, ?)",
                    (
                        spec.revision,
                        digest,
                        len(plan.batches),
                        len(requests),
                        _wall_ms(),
                    ),
                )
                self._set_meta("current_revision", str(spec.revision))
                self._set_meta("ledger_format_version", str(LEDGER_FORMAT_VERSION))
                ledger_digest = self._compute_ledger_digest(spec.revision)
                self._set_meta("ledger_content_digest", ledger_digest)
                self.connection.execute(
                    "UPDATE ledger_header SET format_version = ?, content_digest = ?, "
                    "active_revision = ? WHERE singleton = 1",
                    (LEDGER_FORMAT_VERSION, ledger_digest, spec.revision),
                )
            return ActivationResult(
                revision=spec.revision,
                fingerprint=spec.fingerprint,
                request_stream_digest=digest,
                changed=True,
                diff=diff,
                diff_json=diff_json,
            )

    def _compute_ledger_digest(self, revision: int) -> str:
        batches = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM batches WHERE revision = ? ORDER BY batch_id", (revision,)
            )
        ]
        entries = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM ledger_entries WHERE revision = ? "
                "ORDER BY arrival_ms, request_id",
                (revision,),
            )
        ]
        quota = [
            dict(row)
            for row in self.connection.execute(
                "SELECT * FROM quota_usage WHERE revision = ? ORDER BY batch_id",
                (revision,),
            )
        ]
        return _content_hash(
            b"gateway-ledger-v1\0",
            {
                "format_version": LEDGER_FORMAT_VERSION,
                "revision": revision,
                "batches": batches,
                "entries": entries,
                "quota_usage": quota,
            },
        )

    def verify_ledger(self, revision: int | None = None) -> tuple[bool, str]:
        with self.lock:
            revision = self.latest_revision() if revision is None else revision
            if revision is None:
                return True, ""
            calculated = self._compute_ledger_digest(revision)
            header = self.ledger_header()
            return (
                calculated == self._meta("ledger_content_digest")
                and header["format_version"] == LEDGER_FORMAT_VERSION
                and header["content_digest"] == calculated,
                calculated,
            )

    def commit_batch(self, execution: BatchExecution) -> bool:
        plan = execution.plan
        revision = plan.revision
        request_digests = execution.output_digests
        receipt_json = canonical_json(dict(execution.receipt.raw))
        batch_core = {
            "revision": revision,
            "batch_id": plan.batch_id,
            "rule_id": plan.rule_id,
            "backend_id": plan.backend_id,
            "target": plan.target,
            "request_ids": list(plan.request_ids),
            "parameter_digest": execution.receipt.parameter_digest,
            "receipt_digest": execution.receipt.receipt_digest,
            "output_digest": execution.receipt.output_digest,
            "wait_start_ms": execution.wait_start_ms,
            "wait_end_ms": execution.wait_end_ms,
            "exec_start_ms": execution.exec_start_ms,
            "exec_end_ms": execution.exec_end_ms,
            "elapsed_ms": execution.elapsed_ms,
            "worker_elapsed_ms": execution.receipt.elapsed_ms,
            "worker_pid": execution.receipt.worker_pid,
            "global_limit": execution.global_limit,
            "backend_quota": execution.backend_quota,
            "thread_name": execution.thread_name,
        }
        batch_digest = _content_hash(b"gateway-batch-v1\0", batch_core)
        with self.lock:
            existing = self.connection.execute(
                "SELECT content_digest FROM batches WHERE revision = ? AND batch_id = ?",
                (revision, plan.batch_id),
            ).fetchone()
            if existing is not None:
                if existing["content_digest"] == batch_digest:
                    return False
                raise GatewayError(
                    "batch already committed with different content",
                    code="E_LEDGER",
                    backend_id=plan.backend_id,
                    batch_id=plan.batch_id,
                )
            with self._transaction():
                self.connection.execute(
                    "INSERT INTO batches("
                    "revision, batch_id, rule_id, backend_id, target, batch_size, request_ids_json, "
                    "parameter_digest, receipt_digest, receipt_json, output_digest, wait_start_ms, "
                    "wait_end_ms, exec_start_ms, exec_end_ms, elapsed_ms, worker_elapsed_ms, worker_pid, "
                    "global_limit, backend_quota, thread_name, stale, content_digest"
                    ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
                    (
                        revision,
                        plan.batch_id,
                        plan.rule_id,
                        plan.backend_id,
                        plan.target,
                        plan.batch_size,
                        canonical_json(list(plan.request_ids)),
                        execution.receipt.parameter_digest,
                        execution.receipt.receipt_digest,
                        receipt_json,
                        execution.receipt.output_digest,
                        execution.wait_start_ms,
                        execution.wait_end_ms,
                        execution.exec_start_ms,
                        execution.exec_end_ms,
                        execution.elapsed_ms,
                        execution.receipt.elapsed_ms,
                        execution.receipt.worker_pid,
                        execution.global_limit,
                        execution.backend_quota,
                        execution.thread_name,
                        batch_digest,
                    ),
                )
                self.connection.execute(
                    "INSERT INTO quota_usage("
                    "revision, batch_id, backend_id, global_slots_used, backend_slots_used, "
                    "global_limit, backend_quota, wait_ms, exec_ms"
                    ") VALUES(?, ?, ?, 1, 1, ?, ?, ?, ?)",
                    (
                        revision,
                        plan.batch_id,
                        plan.backend_id,
                        execution.global_limit,
                        execution.backend_quota,
                        execution.wait_ms,
                        execution.elapsed_ms,
                    ),
                )
                for item in plan.items:
                    request_id = item.sample.request_id
                    entry_payload = {
                        "revision": revision,
                        "request_id": request_id,
                        "rule_id": item.decision.rule_id,
                        "target": item.decision.target,
                        "backend_id": item.decision.backend_id,
                        "bucket": item.decision.bucket,
                        "weight_bp": item.decision.weight_bp,
                        "weight_version": item.decision.weight_version,
                        "batch_id": plan.batch_id,
                        "output_digest": request_digests[request_id],
                        "wait_ms": execution.wait_ms,
                        "exec_ms": execution.elapsed_ms,
                    }
                    self.connection.execute(
                        "INSERT INTO ledger_entries("
                        "revision, request_id, client_key, arrival_ms, rule_id, target, backend_id, "
                        "bucket, weight_bp, weight_version, routing_version, batch_id, output_digest, "
                        "wait_ms, exec_ms, parameter_digest, receipt_digest, stale, content_digest"
                        ") VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
                        (
                            revision,
                            request_id,
                            item.sample.client_key,
                            item.sample.arrival_ms,
                            item.decision.rule_id,
                            item.decision.target,
                            item.decision.backend_id,
                            item.decision.bucket,
                            item.decision.weight_bp,
                            item.decision.weight_version,
                            item.decision.routing_version,
                            plan.batch_id,
                            request_digests[request_id],
                            execution.wait_ms,
                            execution.elapsed_ms,
                            execution.receipt.parameter_digest,
                            execution.receipt.receipt_digest,
                            _content_hash(b"gateway-entry-v1\0", entry_payload),
                        ),
                    )
                committed_batches = int(
                    self.connection.execute(
                        "SELECT COUNT(*) FROM batches WHERE revision = ?", (revision,)
                    ).fetchone()[0]
                )
                committed_requests = int(
                    self.connection.execute(
                        "SELECT COUNT(*) FROM ledger_entries WHERE revision = ?", (revision,)
                    ).fetchone()[0]
                )
                self.connection.execute(
                    "UPDATE run_progress SET committed_batches = ?, committed_requests = ?, "
                    "last_batch_id = ?, last_request_id = ?, updated_at_ms = ? WHERE revision = ?",
                    (
                        committed_batches,
                        committed_requests,
                        plan.batch_id,
                        max(plan.request_ids),
                        _wall_ms(),
                        revision,
                    ),
                )
                self.connection.execute(
                    "UPDATE recovery_unknowns SET status = 'RESOLVED', resolved_at_ms = ? "
                    "WHERE revision = ? AND batch_id = ?",
                    (_wall_ms(), revision, plan.batch_id),
                )
                self._set_meta("ledger_content_digest", self._compute_ledger_digest(revision))
                self.connection.execute(
                    "UPDATE ledger_header SET format_version = ?, content_digest = ?, "
                    "active_revision = ? WHERE singleton = 1",
                    (LEDGER_FORMAT_VERSION, self._meta("ledger_content_digest"), revision),
                )
        return True

    def ledger_header(self) -> dict[str, Any]:
        with self.lock:
            row = self.connection.execute(
                "SELECT singleton, format_version, content_digest, active_revision "
                "FROM ledger_header WHERE singleton = 1"
            ).fetchone()
            if row is None:
                raise ReplayError("ledger header is missing", field="ledger_header")
            return dict(row)

    def committed_batch_ids(self, revision: int) -> tuple[str, ...]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT batch_id FROM batches WHERE revision = ? ORDER BY batch_id",
                (revision,),
            )
            return tuple(str(row[0]) for row in rows)

    def committed_request_ids(self, revision: int) -> tuple[int, ...]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT request_id FROM ledger_entries WHERE revision = ? ORDER BY request_id",
                (revision,),
            )
            return tuple(int(row[0]) for row in rows)

    def _load_preview_in_one_read(self, revision: int) -> tuple[PreviewEntry, ...]:
        rows = self.connection.execute(
            "SELECT e.request_id, e.client_key, e.arrival_ms, e.rule_id, e.target, "
            "e.backend_id, e.bucket, e.weight_bp, e.weight_version, e.routing_version, "
            "e.batch_id, e.output_digest, e.wait_ms, e.exec_ms, e.parameter_digest, "
            "e.receipt_digest FROM ledger_entries e "
            "JOIN batches b ON b.revision = e.revision AND b.batch_id = e.batch_id "
            "WHERE e.revision = ? AND e.stale = 0 AND b.stale = 0 "
            "ORDER BY e.arrival_ms, e.request_id",
            (revision,),
        ).fetchall()
        return tuple(
            PreviewEntry(
                request_id=int(row["request_id"]),
                client_key=str(row["client_key"]),
                arrival_ms=int(row["arrival_ms"]),
                rule_id=str(row["rule_id"]),
                target=str(row["target"]),
                backend_id=str(row["backend_id"]),
                bucket=int(row["bucket"]),
                weight_bp=int(row["weight_bp"]),
                weight_version=int(row["weight_version"]),
                routing_version=int(row["routing_version"]),
                batch_id=str(row["batch_id"]),
                output_digest=str(row["output_digest"]),
                wait_ms=int(row["wait_ms"]),
                exec_ms=int(row["exec_ms"]),
                parameter_digest=str(row["parameter_digest"]),
                receipt_digest=str(row["receipt_digest"]),
            )
            for row in rows
        )

    def load_preview_cache(self, revision: int | None = None) -> PreviewCache:
        with self.lock:
            revision = self.latest_revision() if revision is None else revision
            if revision is None:
                raise ReplayError("ledger has no active revision", field="revision")
            active = self.latest_revision()
            if active != revision:
                raise ReplayError(
                    f"revision {revision} is stale; active revision is {active}",
                    field="revision",
                )
            entries = self._load_preview_in_one_read(revision)
            stats = self._compute_stats(entries)
            return PreviewCache(PreviewSnapshot(revision, entries, stats))

    def snapshot_from_entries(
        self, revision: int, entries: Sequence[PreviewEntry]
    ) -> PreviewSnapshot:
        ordered = tuple(
            sorted(entries, key=lambda entry: (entry.arrival_ms, entry.request_id))
        )
        return PreviewSnapshot(revision, ordered, self._compute_stats(ordered))

    def _compute_stats(self, entries: Sequence[PreviewEntry]) -> dict[str, Any]:
        batches: dict[str, dict[str, Any]] = {}
        by_backend: dict[str, dict[str, Any]] = {}
        by_rule: dict[str, dict[str, Any]] = {}
        stable_count = 0
        canary_count = 0
        for entry in entries:
            if entry.target == "canary":
                canary_count += 1
            else:
                stable_count += 1
            batch = batches.setdefault(
                entry.batch_id,
                {
                    "batch_id": entry.batch_id,
                    "backend_id": entry.backend_id,
                    "wait_ms": entry.wait_ms,
                    "exec_ms": entry.exec_ms,
                    "request_ids": [],
                },
            )
            batch["request_ids"].append(entry.request_id)
            backend_stats = by_backend.setdefault(
                entry.backend_id,
                {
                    "batches": set(),
                    "requests": 0,
                    "total_wait_ms": 0,
                    "total_exec_ms": 0,
                },
            )
            backend_stats["batches"].add(entry.batch_id)
            backend_stats["requests"] += 1
            rule_stats = by_rule.setdefault(
                entry.rule_id,
                {"requests": 0, "stable": 0, "canary": 0},
            )
            rule_stats["requests"] += 1
            rule_stats[entry.target] += 1

        ordered_batches = [batches[key] for key in sorted(batches)]
        for batch in ordered_batches:
            batch["request_ids"].sort()
            backend_stats = by_backend[batch["backend_id"]]
            if batch["batch_id"] in backend_stats["batches"]:
                backend_stats["total_wait_ms"] += int(batch["wait_ms"])
                backend_stats["total_exec_ms"] += int(batch["exec_ms"])
                backend_stats["batches"].discard(batch["batch_id"])
        backend_result = {
            backend_id: {
                "batches": len(
                    {entry.batch_id for entry in entries if entry.backend_id == backend_id}
                ),
                "requests": by_backend[backend_id]["requests"],
                "total_wait_ms": by_backend[backend_id]["total_wait_ms"],
                "total_exec_ms": by_backend[backend_id]["total_exec_ms"],
            }
            for backend_id in sorted(by_backend)
        }
        return {
            "request_count": len(entries),
            "stable_count": stable_count,
            "canary_count": canary_count,
            "batch_count": len(ordered_batches),
            "total_wait_ms": sum(int(batch["wait_ms"]) for batch in ordered_batches),
            "total_exec_ms": sum(int(batch["exec_ms"]) for batch in ordered_batches),
            "max_wait_ms": max((int(batch["wait_ms"]) for batch in ordered_batches), default=0),
            "max_exec_ms": max((int(batch["exec_ms"]) for batch in ordered_batches), default=0),
            "by_backend": backend_result,
            "by_rule": {
                rule_id: by_rule[rule_id] for rule_id in sorted(by_rule)
            },
            "batches": ordered_batches,
        }

    def recovery_report(
        self,
        plan: RoutingPlan,
        requests: Sequence[RequestSample],
    ) -> RecoveryReport:
        digest = request_stream_digest(requests)
        expected_batches = {batch.batch_id: batch for batch in plan.batches}
        expected_request_ids = {
            item.sample.request_id for batch in plan.batches for item in batch.items
        }
        with self.lock:
            latest = self.latest_spec_row()
            if latest is None:
                raise ReplayError("ledger has no active spec", field="revision")
            if int(latest["revision"]) != plan.revision:
                raise ReplayError(
                    f"plan revision {plan.revision} is stale; active revision is {latest['revision']}",
                    field="revision",
                )
            if latest["request_stream_digest"] != digest:
                raise ReplayError(
                    "request stream differs from the stream bound to the ledger",
                    field="request_stream_digest",
                )
            committed_batches = set(self.committed_batch_ids(plan.revision))
            committed_request_ids = set(self.committed_request_ids(plan.revision))
            missing_batch_ids = tuple(
                sorted(set(expected_batches) - committed_batches)
            )
            extra_batch_ids = tuple(sorted(committed_batches - set(expected_batches)))
            missing_request_ids = tuple(sorted(expected_request_ids - committed_request_ids))
            extra_request_ids = tuple(sorted(committed_request_ids - expected_request_ids))
            unknown: list[str] = []
            now = _wall_ms()
            with self._transaction():
                for batch_id in missing_batch_ids:
                    batch = expected_batches[batch_id]
                    unknown.append(batch_id)
                    self.connection.execute(
                        "INSERT INTO recovery_unknowns("
                        "revision, batch_id, request_ids_json, status, reason, detected_at_ms, resolved_at_ms"
                        ") VALUES(?, ?, ?, 'UNKNOWN', 'NOT_COMMITTED', ?, NULL) "
                        "ON CONFLICT(revision, batch_id) DO UPDATE SET "
                        "status = 'UNKNOWN', reason = 'NOT_COMMITTED', detected_at_ms = excluded.detected_at_ms, "
                        "resolved_at_ms = NULL",
                        (
                            plan.revision,
                            batch_id,
                            canonical_json(list(batch.request_ids)),
                            now,
                        ),
                    )
            integrity_ok, computed_digest = self.verify_ledger(plan.revision)
            return RecoveryReport(
                revision=plan.revision,
                ledger_integrity_ok=integrity_ok,
                ledger_content_digest=computed_digest,
                expected_batches=len(expected_batches),
                committed_batches=len(committed_batches),
                expected_requests=len(expected_request_ids),
                committed_requests=len(committed_request_ids),
                missing_batches=missing_batch_ids,
                missing_request_ids=missing_request_ids,
                extra_batches=extra_batch_ids,
                extra_request_ids=extra_request_ids,
                unknown_batch_ids=tuple(unknown),
                needs_rerun=bool(missing_batch_ids or missing_request_ids or extra_batch_ids or extra_request_ids),
            )

    def version_history(self) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT revision, fingerprint, request_stream_digest, format_version, "
                "content_digest, activated_at_ms, diff_json FROM spec_versions ORDER BY revision"
            ).fetchall()
            return [dict(row) for row in rows]

    def activation_diff(self, revision: int | None = None) -> dict[str, Any]:
        with self.lock:
            if revision is None:
                row = self.latest_spec_row()
            else:
                row = self.connection.execute(
                    "SELECT * FROM spec_versions WHERE revision = ?", (revision,)
                ).fetchone()
            if row is None:
                raise ReplayError("spec revision not found", field="revision")
            return json.loads(row["diff_json"])

    def fetch_raw_batch(self, revision: int, batch_id: str) -> sqlite3.Row | None:
        with self.lock:
            return self.connection.execute(
                "SELECT * FROM batches WHERE revision = ? AND batch_id = ?",
                (revision, batch_id),
            ).fetchone()

    def fetch_raw_entries(self, revision: int) -> tuple[sqlite3.Row, ...]:
        with self.lock:
            return tuple(
                self.connection.execute(
                    "SELECT * FROM ledger_entries WHERE revision = ? "
                    "ORDER BY arrival_ms, request_id",
                    (revision,),
                ).fetchall()
            )

    def progress(self, revision: int) -> dict[str, Any] | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT * FROM run_progress WHERE revision = ?", (revision,)
            ).fetchone()
            return None if row is None else dict(row)

    def recovery_unknown_rows(self, revision: int) -> list[dict[str, Any]]:
        with self.lock:
            rows = self.connection.execute(
                "SELECT * FROM recovery_unknowns WHERE revision = ? ORDER BY batch_id",
                (revision,),
            ).fetchall()
            return [dict(row) for row in rows]

    def execute_for_test(self, sql: str, parameters: Iterable[Any] = ()) -> None:
        with self._transaction():
            self.connection.execute(sql, tuple(parameters))
