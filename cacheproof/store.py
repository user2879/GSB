"""Transactional cache-segment store, proofs, views and recovery."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any
import uuid

from .canonical import canonical_text, utf8_key
from .content import digest_file, entry_digest, persist_content_file
from .db import (
    LEDGER_FORMAT_VERSION,
    connect,
    ensure_schema,
    get_ledger_state,
    loads_json,
    refresh_ledger_digest,
    transaction,
    verify_ledger_digest,
)
from .errors import DiagnosticError, spec_error
from .eventlog import EventLogger
from .models import CacheSpec, OperationResult
from .paths import (
    WorkspacePaths,
    atomic_write_json,
    decode_component,
    encode_component,
)
from .proof import build_segment_proof, proof_json, verify_proof_digest
from .spec import latest_views, normalized_spec_payload, spec_digest
from .snapshot import (
    compare_replayed_state,
    compare_replayed_views,
    expected_replayed_views,
    replay_operations,
    seal_session_snapshot,
    snapshot_diff,
    verify_session_snapshot,
)
from .views import (
    build_view_snapshot,
    compute_view_diff,
    validate_view_snapshot,
    view_is_stale,
)


SESSION_SNAPSHOT_FORMAT = 1


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _elapsed_ns(start_ns: int) -> int:
    return max(0, time.monotonic_ns() - start_ns)


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return None if row is None else dict(row)


class CacheProofStore:
    """A local SQLite-backed cacheproof workspace.

    Instances are single-process coordinators; SQLite WAL still permits
    concurrent read snapshots from other connections/processes.
    """

    def __init__(
        self,
        workspace: str | Path,
        *,
        spec: CacheSpec | None = None,
        create: bool = False,
        auto_recover: bool = True,
        auto_snapshot: bool = True,
    ):
        self.paths = WorkspacePaths(workspace)
        self.paths.ensure()
        self.logger = EventLogger(self.paths.event_log)
        self.connection = connect(self.paths.ledger)
        self._lock = threading.RLock()
        self.auto_snapshot = auto_snapshot
        self._closed = False
        ensure_schema(self.connection)
        state = get_ledger_state(self.connection)
        if state is None:
            if spec is None or not create:
                raise DiagnosticError(
                    "E_NOT_FOUND",
                    "workspace ledger is not initialized; run init with a declaration",
                    field="ledger_state",
                )
            self._bootstrap(spec)
        else:
            verify_ledger_digest(self.connection)
            if spec is not None:
                self._reconcile_spec(spec)
            if auto_recover:
                self.recover_unknown_entries()

    def __enter__(self) -> "CacheProofStore":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        if not self._closed:
            self.connection.close()
            self._closed = True

    # ------------------------------------------------------------------
    # Bootstrap and declaration reconciliation
    # ------------------------------------------------------------------

    def _write_declaration_copy(self, spec: CacheSpec) -> None:
        payload = normalized_spec_payload(spec)
        atomic_write_json(self.paths.spec_copy, payload, self.paths.tmp)

    def _bootstrap(self, spec: CacheSpec) -> None:
        declaration_digest = spec_digest(spec)
        persisted: dict[tuple[str, str], tuple[str, str]] = {}
        for segment in sorted(spec.segments, key=lambda item: utf8_key(item.segment_id)):
            for entry in sorted(segment.entries, key=lambda item: utf8_key(item.key)):
                expected = entry_digest(entry.key, entry.size_bytes, entry.content_seed)
                relative = self.paths.relative_entry_path(
                    segment.segment_id, entry.key, expected
                )
                destination = self.paths.absolute_entry_path(relative)
                result = persist_content_file(
                    destination,
                    self.paths.tmp,
                    key=entry.key,
                    size_bytes=entry.size_bytes,
                    seed=entry.content_seed,
                    expected_digest=expected,
                )
                persisted[(segment.segment_id, entry.key)] = (
                    result["digest"],
                    relative,
                )

        with self._lock, transaction(self.connection):
            self.connection.execute(
                """
                INSERT INTO ledger_state
                    (singleton, format_version, batch_id, segment_set_version,
                     completed_cursor, spec_digest, content_digest)
                VALUES (1, ?, 0, 1, 0, ?, ?)
                """,
                (LEDGER_FORMAT_VERSION, declaration_digest, "sha256:" + "0" * 64),
            )
            self.connection.execute(
                "INSERT INTO spec_meta(key, value) VALUES ('spec_digest', ?)",
                (declaration_digest,),
            )
            for segment in sorted(
                spec.segments, key=lambda item: utf8_key(item.segment_id)
            ):
                self.connection.execute(
                    """
                    INSERT INTO segments
                        (segment_id, capacity_bytes, used_bytes, entry_count,
                         segment_version, proof_json, proof_digest)
                    VALUES (?, ?, 0, 0, 1, '', '')
                    """,
                    (segment.segment_id, segment.capacity_bytes),
                )
                for entry in sorted(
                    segment.entries, key=lambda item: utf8_key(item.key)
                ):
                    digest, relative = persisted[(segment.segment_id, entry.key)]
                    self.connection.execute(
                        """
                        INSERT INTO entries
                            (segment_id, entry_key, size_bytes, content_seed,
                             digest, file_relpath, committed_cursor)
                        VALUES (?, ?, ?, ?, ?, ?, 0)
                        """,
                        (
                            segment.segment_id,
                            entry.key,
                            entry.size_bytes,
                            entry.content_seed,
                            digest,
                            relative,
                        ),
                    )
                self._refresh_segment_proof(
                    self.connection, segment.segment_id, segment_set_version=1
                )

            for view in latest_views(spec):
                snapshot = build_view_snapshot(
                    view_id=view.view_id,
                    view_version=view.version,
                    segment_set_version=1,
                    segments=[
                        {
                            "segment_id": view_segment.segment_id,
                            "segment_version": 1,
                            "entries": [
                                {"key": entry.key, "digest": entry.digest}
                                for entry in view_segment.entries
                            ],
                        }
                        for view_segment in view.segments
                    ],
                )
                self.connection.execute(
                    """
                    INSERT INTO read_views
                        (view_id, view_version, segment_set_version, snapshot_json,
                         snapshot_digest, last_differences_json, updated_monotonic_ns)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        view.view_id,
                        view.version,
                        snapshot["segment_set_version"],
                        canonical_text(snapshot),
                        snapshot["content_digest"],
                        "{}",
                        time.monotonic_ns(),
                    ),
                )
            refresh_ledger_digest(self.connection, increment_batch=True)
        self._write_declaration_copy(spec)
        self.logger.emit(
            "workspace_initialized",
            business_code="OK",
            details={
                "segments": len(spec.segments),
                "views": len(latest_views(spec)),
                "spec_digest": declaration_digest,
            },
        )

    def _reconcile_spec(self, spec: CacheSpec) -> None:
        declaration_digest = spec_digest(spec)
        state = get_ledger_state(self.connection)
        assert state is not None
        segment_rows = {
            row["segment_id"]: row
            for row in self.connection.execute("SELECT * FROM segments").fetchall()
        }
        spec_ids = {segment.segment_id for segment in spec.segments}
        if spec_ids != set(segment_rows):
            missing = sorted(spec_ids - set(segment_rows), key=utf8_key)
            extra = sorted(set(segment_rows) - spec_ids, key=utf8_key)
            raise spec_error(
                "declaration segment set does not match the ledger",
                segment_id=(missing or extra or [None])[0],
                field="segments",
                details={"missing_segment_ids": missing, "ledger_only": extra},
            )

        capacity_changes: dict[str, int] = {}
        for segment in spec.segments:
            row = segment_rows[segment.segment_id]
            if segment.capacity_bytes < row["used_bytes"]:
                raise DiagnosticError(
                    "E_CAPACITY",
                    "declared segment capacity is smaller than committed used bytes",
                    segment_id=segment.segment_id,
                    field="capacity_bytes",
                    details={
                        "capacity_bytes": segment.capacity_bytes,
                        "used_bytes": row["used_bytes"],
                        "required_at_least_bytes": row["used_bytes"],
                    },
                )
            if segment.capacity_bytes != row["capacity_bytes"]:
                capacity_changes[segment.segment_id] = segment.capacity_bytes

            for declared_entry in segment.entries:
                committed = self.connection.execute(
                    "SELECT * FROM entries WHERE segment_id = ? AND entry_key = ?",
                    (segment.segment_id, declared_entry.key),
                ).fetchone()
                if committed is None:
                    continue
                expected_digest = entry_digest(
                    declared_entry.key,
                    declared_entry.size_bytes,
                    declared_entry.content_seed,
                )
                if (
                    committed["size_bytes"] != declared_entry.size_bytes
                    or committed["content_seed"] != declared_entry.content_seed
                    or committed["digest"] != expected_digest
                ):
                    raise DiagnosticError(
                        "E_REPLAY",
                        "declaration conflicts with a committed entry",
                        segment_id=segment.segment_id,
                        entry_key=declared_entry.key,
                        field="entry",
                        details={
                            "reason": "committed metadata and replayed declaration differ",
                            "expected_size_bytes": committed["size_bytes"],
                            "declared_size_bytes": declared_entry.size_bytes,
                            "expected_content_seed": committed["content_seed"],
                            "declared_content_seed": declared_entry.content_seed,
                        },
                    )

        new_views = [
            view
            for view in latest_views(spec)
            if self.connection.execute(
                "SELECT 1 FROM read_views WHERE view_id = ?", (view.view_id,)
            ).fetchone()
            is None
        ]
        metadata_changed = state["spec_digest"] != declaration_digest
        if capacity_changes or new_views or metadata_changed:
            with self._lock, transaction(self.connection):
                live_state = get_ledger_state(self.connection)
                assert live_state is not None
                global_version = live_state["segment_set_version"]
                for segment_id, capacity in sorted(
                    capacity_changes.items(), key=lambda item: utf8_key(item[0])
                ):
                    global_version += 1
                    self.connection.execute(
                        """
                        UPDATE segments
                        SET capacity_bytes = ?, segment_version = segment_version + 1
                        WHERE segment_id = ?
                        """,
                        (capacity, segment_id),
                    )
                if capacity_changes:
                    self.connection.execute(
                        "UPDATE ledger_state SET segment_set_version = ? WHERE singleton = 1",
                        (global_version,),
                    )
                    self._refresh_all_segment_proofs(
                        self.connection, segment_set_version=global_version
                    )
                self.connection.execute(
                    "UPDATE ledger_state SET spec_digest = ? WHERE singleton = 1",
                    (declaration_digest,),
                )
                self.connection.execute(
                    """
                    INSERT INTO spec_meta(key, value) VALUES ('spec_digest', ?)
                    ON CONFLICT(key) DO UPDATE SET value = excluded.value
                    """,
                    (declaration_digest,),
                )
                for view in new_views:
                    snapshot = build_view_snapshot(
                        view_id=view.view_id,
                        view_version=view.version,
                        segment_set_version=global_version,
                        segments=[
                            {
                                "segment_id": view_segment.segment_id,
                                "segment_version": self.connection.execute(
                                    "SELECT segment_version FROM segments WHERE segment_id = ?",
                                    (view_segment.segment_id,),
                                ).fetchone()["segment_version"],
                                "entries": [
                                    {"key": entry.key, "digest": entry.digest}
                                    for entry in view_segment.entries
                                ],
                            }
                            for view_segment in view.segments
                        ],
                    )
                    self.connection.execute(
                        """
                        INSERT INTO read_views
                            (view_id, view_version, segment_set_version, snapshot_json,
                             snapshot_digest, last_differences_json, updated_monotonic_ns)
                        VALUES (?, ?, ?, ?, ?, '{}', ?)
                        """,
                        (
                            view.view_id,
                            view.version,
                            global_version,
                            canonical_text(snapshot),
                            snapshot["content_digest"],
                            time.monotonic_ns(),
                        ),
                    )
                refresh_ledger_digest(self.connection, increment_batch=True)
            self.logger.emit(
                "declaration_reconciled",
                business_code="OK",
                details={
                    "capacity_changes": capacity_changes,
                    "new_views": [view.view_id for view in new_views],
                    "spec_digest": declaration_digest,
                },
            )
        self._write_declaration_copy(spec)

    # ------------------------------------------------------------------
    # Low-level deterministic state helpers
    # ------------------------------------------------------------------

    def _entries_for_segment(
        self, connection: sqlite3.Connection, segment_id: str
    ) -> list[dict[str, Any]]:
        rows = connection.execute(
            "SELECT * FROM entries WHERE segment_id = ?", (segment_id,)
        ).fetchall()
        result = [dict(row) for row in rows]
        result.sort(key=lambda item: utf8_key(item["entry_key"]))
        return result

    def _refresh_segment_proof(
        self,
        connection: sqlite3.Connection,
        segment_id: str,
        *,
        segment_set_version: int,
    ) -> dict[str, Any]:
        segment = connection.execute(
            "SELECT * FROM segments WHERE segment_id = ?", (segment_id,)
        ).fetchone()
        if segment is None:
            raise DiagnosticError(
                "E_NOT_FOUND",
                "segment does not exist",
                segment_id=segment_id,
                field="segment_id",
            )
        entries = self._entries_for_segment(connection, segment_id)
        proof = build_segment_proof(
            segment_id=segment_id,
            capacity_bytes=segment["capacity_bytes"],
            segment_set_version=segment_set_version,
            segment_version=segment["segment_version"],
            entries=entries,
        )
        connection.execute(
            """
            UPDATE segments
            SET used_bytes = ?, entry_count = ?, proof_json = ?, proof_digest = ?
            WHERE segment_id = ?
            """,
            (
                proof["used_bytes"],
                proof["entry_count"],
                proof_json(proof),
                proof["content_digest"],
                segment_id,
            ),
        )
        return proof

    def _refresh_all_segment_proofs(
        self,
        connection: sqlite3.Connection,
        *,
        segment_set_version: int,
    ) -> dict[str, dict[str, Any]]:
        segment_ids = [
            row["segment_id"]
            for row in connection.execute("SELECT segment_id FROM segments").fetchall()
        ]
        segment_ids.sort(key=utf8_key)
        return {
            segment_id: self._refresh_segment_proof(
                connection,
                segment_id,
                segment_set_version=segment_set_version,
            )
            for segment_id in segment_ids
        }

    def _operation_details(self, value: dict[str, Any] | None) -> str:
        return canonical_text(value or {})

    def _insert_operation(
        self,
        connection: sqlite3.Connection,
        *,
        operation_id: str,
        kind: str,
        status: str,
        result_code: str,
        segment_id: str,
        entry_key: str,
        size_bytes: int | None,
        content_seed: str | None,
        expected_digest: str | None,
        file_relpath: str | None,
        old_file_relpath: str | None,
        required_bytes: int | None,
        available_bytes: int | None,
        started_ns: int,
        details: dict[str, Any] | None = None,
        committed_cursor: int | None = None,
        finished_ns: int | None = None,
    ) -> int:
        elapsed = None if finished_ns is None else max(0, finished_ns - started_ns)
        cursor = connection.execute(
            """
            INSERT INTO operations
                (operation_id, kind, status, result_code, segment_id, entry_key,
                 size_bytes, content_seed, expected_digest, file_relpath,
                 old_file_relpath, required_bytes, available_bytes,
                 committed_cursor, started_monotonic_ns, finished_monotonic_ns,
                 elapsed_ns, details_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                operation_id,
                kind,
                status,
                result_code,
                segment_id,
                entry_key,
                size_bytes,
                content_seed,
                expected_digest,
                file_relpath,
                old_file_relpath,
                required_bytes,
                available_bytes,
                committed_cursor,
                started_ns,
                finished_ns,
                elapsed,
                self._operation_details(details),
            ),
        )
        return int(cursor.lastrowid)

    def _remove_file_if_unreferenced(self, relative_path: str | None) -> bool:
        if not relative_path:
            return False
        row = self.connection.execute(
            "SELECT 1 FROM entries WHERE file_relpath = ?", (relative_path,)
        ).fetchone()
        if row is not None:
            return False
        path = self.paths.absolute_entry_path(relative_path)
        try:
            path.unlink()
            return True
        except FileNotFoundError:
            return False
        except OSError as exc:
            self.logger.emit(
                "file_cleanup_deferred",
                business_code="E_IO",
                details={"file_relpath": relative_path, "error": str(exc)},
            )
            return False

    def _mark_unknown(
        self,
        *,
        operation_id: str,
        operation_seq: int | None,
        segment_id: str,
        entry_key: str,
        expected_digest: str | None,
        observed_digest: str | None,
        observed_size_bytes: int | None,
        observed_file_relpath: str | None,
        reason: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        now_ns = time.monotonic_ns()
        with self._lock, transaction(self.connection):
            if operation_seq is not None:
                self.connection.execute(
                    """
                    UPDATE operations
                    SET status = 'unknown', result_code = 'E_UNKNOWN',
                        finished_monotonic_ns = ?, elapsed_ns = ?,
                        details_json = ?
                    WHERE operation_seq = ?
                    """,
                    (
                        now_ns,
                        max(0, now_ns - self.connection.execute(
                            "SELECT started_monotonic_ns FROM operations WHERE operation_seq = ?",
                            (operation_seq,),
                        ).fetchone()[0]),
                        self._operation_details(details),
                        operation_seq,
                    ),
                )
            self.connection.execute(
                """
                INSERT INTO unknown_entries
                    (segment_id, entry_key, operation_id, expected_digest,
                     observed_digest, observed_size_bytes, observed_file_relpath,
                     reason, first_seen_monotonic_ns)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(segment_id, entry_key, operation_id, observed_file_relpath) DO UPDATE SET
                    expected_digest = excluded.expected_digest,
                    observed_digest = excluded.observed_digest,
                    observed_size_bytes = excluded.observed_size_bytes,
                    observed_file_relpath = excluded.observed_file_relpath,
                    reason = excluded.reason
                """,
                (
                    segment_id,
                    entry_key,
                    operation_id,
                    expected_digest,
                    observed_digest,
                    observed_size_bytes,
                    observed_file_relpath or "",
                    reason,
                    now_ns,
                ),
            )
            refresh_ledger_digest(self.connection, increment_batch=True)
        self.logger.emit(
            "unknown_entry",
            business_code="E_UNKNOWN",
            operation_seq=operation_seq,
            segment_id=segment_id,
            entry_key=entry_key,
            field="entries",
            details={
                "reason": reason,
                "expected_digest": expected_digest,
                "observed_digest": observed_digest,
                "observed_size_bytes": observed_size_bytes,
            },
        )

    # ------------------------------------------------------------------
    # Writes and deletes
    # ------------------------------------------------------------------

    def _validate_entry_args(
        self, segment_id: str, entry_key: str, size_bytes: int, content_seed: str
    ) -> None:
        if not isinstance(segment_id, str) or not segment_id:
            raise spec_error("segment_id must be a non-empty string", field="segment_id")
        if not isinstance(entry_key, str) or not entry_key:
            raise spec_error(
                "entry key must be a non-empty string",
                segment_id=segment_id,
                field="entry_key",
            )
        if not _is_int(size_bytes) or size_bytes < 0:
            raise spec_error(
                "size_bytes must be a non-negative integer",
                segment_id=segment_id,
                entry_key=entry_key,
                field="size_bytes",
            )
        if not isinstance(content_seed, str):
            raise spec_error(
                "content_seed must be a string",
                segment_id=segment_id,
                entry_key=entry_key,
                field="content_seed",
            )

    def put_entry(
        self,
        segment_id: str,
        entry_key: str,
        size_bytes: int,
        content_seed: str,
        *,
        failpoint: str | None = None,
    ) -> OperationResult:
        started_ns = time.monotonic_ns()
        self._validate_entry_args(segment_id, entry_key, size_bytes, content_seed)
        expected_digest = entry_digest(entry_key, size_bytes, content_seed)
        file_relpath = self.paths.relative_entry_path(
            segment_id, entry_key, expected_digest
        )
        operation_id = "op-" + uuid.uuid4().hex
        result: OperationResult | None = None
        committed_result: OperationResult | None = None
        idempotent_result: OperationResult | None = None
        conflict: DiagnosticError | None = None
        capacity_error: DiagnosticError | None = None

        with self._lock:
            with transaction(self.connection):
                segment = self.connection.execute(
                    "SELECT * FROM segments WHERE segment_id = ?", (segment_id,)
                ).fetchone()
                if segment is None:
                    raise DiagnosticError(
                        "E_NOT_FOUND",
                        "segment does not exist",
                        segment_id=segment_id,
                        field="segment_id",
                    )
                existing = self.connection.execute(
                    "SELECT * FROM entries WHERE segment_id = ? AND entry_key = ?",
                    (segment_id, entry_key),
                ).fetchone()
                if existing is not None:
                    if (
                        existing["size_bytes"] == size_bytes
                        and existing["content_seed"] == content_seed
                        and existing["digest"] == expected_digest
                    ):
                        actual_path = self.paths.absolute_entry_path(
                            existing["file_relpath"]
                        )
                        actual_digest = digest_file(
                            actual_path, entry_key, size_bytes
                        )
                        if actual_digest != expected_digest:
                            raise DiagnosticError(
                                "E_INTEGRITY",
                                "committed entry file does not match its digest",
                                segment_id=segment_id,
                                entry_key=entry_key,
                                field="digest",
                                details={
                                    "expected_digest": expected_digest,
                                    "actual_digest": actual_digest,
                                },
                            )
                        result = OperationResult(
                            ok=True,
                            code="OK",
                            segment_id=segment_id,
                            entry_key=entry_key,
                            elapsed_ns=_elapsed_ns(started_ns),
                            idempotent=True,
                            details={
                                "digest": expected_digest,
                                "file_relpath": existing["file_relpath"],
                            },
                        )
                        operation_seq = None
                        available = segment["capacity_bytes"] - segment["used_bytes"]
                    else:
                        raise DiagnosticError(
                            "E_SPEC",
                            "entry key already exists with different metadata",
                            segment_id=segment_id,
                            entry_key=entry_key,
                            field="entry_key",
                            details={
                                "existing_size_bytes": existing["size_bytes"],
                                "requested_size_bytes": size_bytes,
                            },
                        )
                else:
                    available = segment["capacity_bytes"] - segment["used_bytes"]
                    if size_bytes > available:
                        finished_ns = time.monotonic_ns()
                        operation_seq = self._insert_operation(
                            self.connection,
                            operation_id=operation_id,
                            kind="put",
                            status="rejected",
                            result_code="E_CAPACITY",
                            segment_id=segment_id,
                            entry_key=entry_key,
                            size_bytes=size_bytes,
                            content_seed=content_seed,
                            expected_digest=expected_digest,
                            file_relpath=file_relpath,
                            old_file_relpath=None,
                            required_bytes=size_bytes,
                            available_bytes=available,
                            started_ns=started_ns,
                            finished_ns=finished_ns,
                            details={
                                "capacity_bytes": segment["capacity_bytes"],
                                "used_bytes": segment["used_bytes"],
                            },
                        )
                        capacity_error = DiagnosticError(
                            "E_CAPACITY",
                            "segment does not have enough remaining capacity",
                            segment_id=segment_id,
                            entry_key=entry_key,
                            field="capacity_bytes",
                            details={
                                "required_bytes": size_bytes,
                                "available_bytes": available,
                                "capacity_bytes": segment["capacity_bytes"],
                                "used_bytes": segment["used_bytes"],
                            },
                        )
                        result = None
                    else:
                        operation_seq = self._insert_operation(
                            self.connection,
                            operation_id=operation_id,
                            kind="put",
                            status="prepared",
                            result_code="PENDING",
                            segment_id=segment_id,
                            entry_key=entry_key,
                            size_bytes=size_bytes,
                            content_seed=content_seed,
                            expected_digest=expected_digest,
                            file_relpath=file_relpath,
                            old_file_relpath=None,
                            required_bytes=size_bytes,
                            available_bytes=available,
                            started_ns=started_ns,
                        )
                        capacity_error = None
                        result = None

            if operation_seq is not None and result is None and capacity_error is not None:
                self.logger.emit(
                    "entry_put_rejected",
                    business_code="E_CAPACITY",
                    operation_seq=operation_seq,
                    segment_id=segment_id,
                    entry_key=entry_key,
                    field="capacity_bytes",
                    details=capacity_error.details,
                    monotonic_ns=time.monotonic_ns(),
                )
                raise capacity_error
            if result is not None:
                self.logger.emit(
                    "entry_put_idempotent",
                    business_code="OK",
                    operation_seq=None,
                    segment_id=segment_id,
                    entry_key=entry_key,
                    details={
                        "digest": expected_digest,
                        "file_relpath": result.details["file_relpath"],
                    },
                )
                return result

            if failpoint == "after_prepare":
                os._exit(91)

            try:
                persist_content_file(
                    self.paths.absolute_entry_path(file_relpath),
                    self.paths.tmp,
                    key=entry_key,
                    size_bytes=size_bytes,
                    seed=content_seed,
                    expected_digest=expected_digest,
                )
            except BaseException as exc:
                observed_digest = None
                observed_size = None
                candidate = self.paths.absolute_entry_path(file_relpath)
                if candidate.exists():
                    observed_size = candidate.stat().st_size
                    try:
                        observed_digest = digest_file(candidate, entry_key, None)
                    except DiagnosticError:
                        observed_digest = None
                self._mark_unknown(
                    operation_id=operation_id,
                    operation_seq=operation_seq,
                    segment_id=segment_id,
                    entry_key=entry_key,
                    expected_digest=expected_digest,
                    observed_digest=observed_digest,
                    observed_size_bytes=observed_size,
                    observed_file_relpath=file_relpath if candidate.exists() else None,
                    reason="content_persist_failed",
                    details={
                        "error_code": getattr(exc, "code", "E_IO"),
                        "error": str(exc),
                    },
                )
                raise

            if failpoint == "after_file_replace":
                os._exit(91)

            try:
                with transaction(self.connection):
                    state = get_ledger_state(self.connection)
                    segment = self.connection.execute(
                        "SELECT * FROM segments WHERE segment_id = ?", (segment_id,)
                    ).fetchone()
                    assert state is not None and segment is not None
                    existing = self.connection.execute(
                        "SELECT * FROM entries WHERE segment_id = ? AND entry_key = ?",
                        (segment_id, entry_key),
                    ).fetchone()
                    if existing is not None:
                        if (
                            existing["digest"] != expected_digest
                            or existing["size_bytes"] != size_bytes
                            or existing["content_seed"] != content_seed
                        ):
                            finished_ns = time.monotonic_ns()
                            self.connection.execute(
                                """
                                UPDATE operations
                                SET status = 'rejected', result_code = 'E_SPEC',
                                    finished_monotonic_ns = ?, elapsed_ns = ?,
                                    details_json = ?
                                WHERE operation_seq = ?
                                """,
                                (
                                    finished_ns,
                                    max(0, finished_ns - started_ns),
                                    self._operation_details(
                                        {"reason": "concurrent key conflict"}
                                    ),
                                    operation_seq,
                                ),
                            )
                            conflict: DiagnosticError | None = DiagnosticError(
                                "E_SPEC",
                                "entry key was committed concurrently with different metadata",
                                segment_id=segment_id,
                                entry_key=entry_key,
                                field="entry_key",
                            )
                        else:
                            finished_ns = time.monotonic_ns()
                            self.connection.execute(
                                """
                                UPDATE operations
                                SET status = 'committed', result_code = 'OK',
                                    finished_monotonic_ns = ?, elapsed_ns = ?,
                                    details_json = ?
                                WHERE operation_seq = ?
                                """,
                                (
                                    finished_ns,
                                    max(0, finished_ns - started_ns),
                                    self._operation_details(
                                        {"reason": "concurrent idempotent commit"}
                                    ),
                                    operation_seq,
                                ),
                            )
                            conflict = None
                            idempotent_result = OperationResult(
                                ok=True,
                                code="OK",
                                operation_id=operation_id,
                                operation_seq=operation_seq,
                                segment_id=segment_id,
                                entry_key=entry_key,
                                elapsed_ns=max(0, finished_ns - started_ns),
                                idempotent=True,
                            )
                    else:
                        conflict = None
                        available = segment["capacity_bytes"] - segment["used_bytes"]
                        if size_bytes > available:
                            finished_ns = time.monotonic_ns()
                            self.connection.execute(
                                """
                                UPDATE operations
                                SET status = 'rejected', result_code = 'E_CAPACITY',
                                    finished_monotonic_ns = ?, elapsed_ns = ?,
                                    details_json = ?
                                WHERE operation_seq = ?
                                """,
                                (
                                    finished_ns,
                                    max(0, finished_ns - started_ns),
                                    self._operation_details(
                                        {
                                            "required_bytes": size_bytes,
                                            "available_bytes": available,
                                            "reason": "capacity changed before commit",
                                        }
                                    ),
                                    operation_seq,
                                ),
                            )
                            conflict = DiagnosticError(
                                "E_CAPACITY",
                                "segment capacity changed before the write commit",
                                segment_id=segment_id,
                                entry_key=entry_key,
                                field="capacity_bytes",
                                details={
                                    "required_bytes": size_bytes,
                                    "available_bytes": available,
                                },
                            )
                        else:
                            new_cursor = state["completed_cursor"] + 1
                            new_global_version = state["segment_set_version"] + 1
                            self.connection.execute(
                                """
                                INSERT INTO entries
                                    (segment_id, entry_key, size_bytes, content_seed,
                                     digest, file_relpath, committed_cursor)
                                VALUES (?, ?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    segment_id,
                                    entry_key,
                                    size_bytes,
                                    content_seed,
                                    expected_digest,
                                    file_relpath,
                                    new_cursor,
                                ),
                            )
                            self.connection.execute(
                                "UPDATE segments SET segment_version = segment_version + 1 WHERE segment_id = ?",
                                (segment_id,),
                            )
                            self.connection.execute(
                                """
                                UPDATE ledger_state
                                SET segment_set_version = ?, completed_cursor = ?
                                WHERE singleton = 1
                                """,
                                (new_global_version, new_cursor),
                            )
                            proofs = self._refresh_all_segment_proofs(
                                self.connection,
                                segment_set_version=new_global_version,
                            )
                            proof = proofs[segment_id]
                            finished_ns = time.monotonic_ns()
                            self.connection.execute(
                                """
                                UPDATE operations
                                SET status = 'committed', result_code = 'OK',
                                    committed_cursor = ?, finished_monotonic_ns = ?,
                                    elapsed_ns = ?, details_json = ?
                                WHERE operation_seq = ?
                                """,
                                (
                                    new_cursor,
                                    finished_ns,
                                    max(0, finished_ns - started_ns),
                                    self._operation_details(
                                        {
                                            "digest": expected_digest,
                                            "used_bytes": proof["used_bytes"],
                                            "remaining_bytes": proof["remaining_bytes"],
                                        }
                                    ),
                                    operation_seq,
                                ),
                            )
                            refresh_ledger_digest(
                                self.connection, increment_batch=True
                            )
                            committed_result = OperationResult(
                                ok=True,
                                code="OK",
                                operation_id=operation_id,
                                operation_seq=operation_seq,
                                committed_cursor=new_cursor,
                                segment_id=segment_id,
                                entry_key=entry_key,
                                required_bytes=size_bytes,
                                available_bytes=available,
                                elapsed_ns=max(0, finished_ns - started_ns),
                                details={
                                    "digest": expected_digest,
                                    "used_bytes": proof["used_bytes"],
                                    "remaining_bytes": proof["remaining_bytes"],
                                    "file_written": True,
                                },
                            )
            except BaseException as exc:
                candidate = self.paths.absolute_entry_path(file_relpath)
                observed_digest = None
                observed_size = None
                if candidate.exists():
                    try:
                        observed_digest = digest_file(candidate, entry_key, None)
                    except DiagnosticError:
                        observed_digest = None
                    observed_size = candidate.stat().st_size
                self._mark_unknown(
                    operation_id=operation_id,
                    operation_seq=operation_seq,
                    segment_id=segment_id,
                    entry_key=entry_key,
                    expected_digest=expected_digest,
                    observed_digest=observed_digest,
                    observed_size_bytes=observed_size,
                    observed_file_relpath=file_relpath if candidate.exists() else None,
                    reason="commit_transaction_failed",
                    details={"error": str(exc)},
                )
                raise

            if conflict is not None:
                self._remove_file_if_unreferenced(file_relpath)
                self.logger.emit(
                    "entry_put_rejected",
                    business_code=conflict.code,
                    operation_seq=operation_seq,
                    segment_id=segment_id,
                    entry_key=entry_key,
                    field=conflict.field,
                    details=conflict.details,
                )
                raise conflict
            if failpoint == "after_commit":
                os._exit(91)
            if committed_result is not None:
                result = committed_result
            else:
                assert idempotent_result is not None
                result = idempotent_result
            self.logger.emit(
                "entry_put",
                business_code=result.code,
                operation_seq=result.operation_seq,
                segment_id=segment_id,
                entry_key=entry_key,
                details=result.details or {},
            )
            if self.auto_snapshot:
                self._save_snapshot_best_effort()
            return result

    def delete_entry(
        self,
        segment_id: str,
        entry_key: str,
        *,
        failpoint: str | None = None,
    ) -> OperationResult:
        started_ns = time.monotonic_ns()
        self._validate_entry_args(segment_id, entry_key, 0, "")
        operation_id = "op-" + uuid.uuid4().hex
        with self._lock:
            with transaction(self.connection):
                segment = self.connection.execute(
                    "SELECT * FROM segments WHERE segment_id = ?", (segment_id,)
                ).fetchone()
                if segment is None:
                    raise DiagnosticError(
                        "E_NOT_FOUND",
                        "segment does not exist",
                        segment_id=segment_id,
                        field="segment_id",
                    )
                entry = self.connection.execute(
                    "SELECT * FROM entries WHERE segment_id = ? AND entry_key = ?",
                    (segment_id, entry_key),
                ).fetchone()
                if entry is None:
                    raise DiagnosticError(
                        "E_NOT_FOUND",
                        "entry does not exist",
                        segment_id=segment_id,
                        entry_key=entry_key,
                        field="entry_key",
                    )
                digest_file(
                    self.paths.absolute_entry_path(entry["file_relpath"]),
                    entry_key,
                    entry["size_bytes"],
                )
                operation_seq = self._insert_operation(
                    self.connection,
                    operation_id=operation_id,
                    kind="delete",
                    status="prepared",
                    result_code="PENDING",
                    segment_id=segment_id,
                    entry_key=entry_key,
                    size_bytes=entry["size_bytes"],
                    content_seed=entry["content_seed"],
                    expected_digest=entry["digest"],
                    file_relpath=None,
                    old_file_relpath=entry["file_relpath"],
                    required_bytes=0,
                    available_bytes=segment["capacity_bytes"] - segment["used_bytes"],
                    started_ns=started_ns,
                )

            if failpoint == "after_prepare":
                os._exit(91)

            try:
                with transaction(self.connection):
                    state = get_ledger_state(self.connection)
                    entry = self.connection.execute(
                        "SELECT * FROM entries WHERE segment_id = ? AND entry_key = ?",
                        (segment_id, entry_key),
                    ).fetchone()
                    assert state is not None
                    if entry is None:
                        finished_ns = time.monotonic_ns()
                        self.connection.execute(
                            """
                            UPDATE operations
                            SET status = 'rejected', result_code = 'E_NOT_FOUND',
                                finished_monotonic_ns = ?, elapsed_ns = ?,
                                details_json = ?
                            WHERE operation_seq = ?
                            """,
                            (
                                finished_ns,
                                max(0, finished_ns - started_ns),
                                self._operation_details(
                                    {"reason": "entry was deleted concurrently"}
                                ),
                                operation_seq,
                            ),
                        )
                        missing: DiagnosticError | None = DiagnosticError(
                            "E_NOT_FOUND",
                            "entry was deleted concurrently",
                            segment_id=segment_id,
                            entry_key=entry_key,
                            field="entry_key",
                        )
                        deleted_result = None
                    else:
                        missing = None
                        new_cursor = state["completed_cursor"] + 1
                        new_global_version = state["segment_set_version"] + 1
                        self.connection.execute(
                            "DELETE FROM entries WHERE segment_id = ? AND entry_key = ?",
                            (segment_id, entry_key),
                        )
                        self.connection.execute(
                            "UPDATE segments SET segment_version = segment_version + 1 WHERE segment_id = ?",
                            (segment_id,),
                        )
                        self.connection.execute(
                            """
                            UPDATE ledger_state
                            SET segment_set_version = ?, completed_cursor = ?
                            WHERE singleton = 1
                            """,
                            (new_global_version, new_cursor),
                        )
                        proofs = self._refresh_all_segment_proofs(
                            self.connection,
                            segment_set_version=new_global_version,
                        )
                        proof = proofs[segment_id]
                        finished_ns = time.monotonic_ns()
                        self.connection.execute(
                            """
                            UPDATE operations
                            SET status = 'committed', result_code = 'OK',
                                committed_cursor = ?, finished_monotonic_ns = ?,
                                elapsed_ns = ?, details_json = ?
                            WHERE operation_seq = ?
                            """,
                            (
                                new_cursor,
                                finished_ns,
                                max(0, finished_ns - started_ns),
                                self._operation_details(
                                    {
                                        "old_digest": entry["digest"],
                                        "used_bytes": proof["used_bytes"],
                                        "remaining_bytes": proof["remaining_bytes"],
                                    }
                                ),
                                operation_seq,
                            ),
                        )
                        refresh_ledger_digest(self.connection, increment_batch=True)
                        deleted_result = OperationResult(
                            ok=True,
                            code="OK",
                            operation_id=operation_id,
                            operation_seq=operation_seq,
                            committed_cursor=new_cursor,
                            segment_id=segment_id,
                            entry_key=entry_key,
                            required_bytes=0,
                            available_bytes=proof["remaining_bytes"],
                            elapsed_ns=max(0, finished_ns - started_ns),
                            details={
                                "old_digest": entry["digest"],
                                "used_bytes": proof["used_bytes"],
                                "remaining_bytes": proof["remaining_bytes"],
                            },
                        )
            except BaseException as exc:
                self._mark_unknown(
                    operation_id=operation_id,
                    operation_seq=operation_seq,
                    segment_id=segment_id,
                    entry_key=entry_key,
                    expected_digest=None,
                    observed_digest=None,
                    observed_size_bytes=None,
                    observed_file_relpath=None,
                    reason="delete_commit_failed",
                    details={"error": str(exc)},
                )
                raise

            if missing is not None:
                self.logger.emit(
                    "entry_delete_rejected",
                    business_code="E_NOT_FOUND",
                    operation_seq=operation_seq,
                    segment_id=segment_id,
                    entry_key=entry_key,
                    field="entry_key",
                    details=missing.details,
                )
                raise missing
            assert deleted_result is not None
            removed = self._remove_file_for_deleted_operation(operation_seq)
            if failpoint == "after_commit":
                os._exit(91)
            self.logger.emit(
                "entry_delete",
                business_code="OK",
                operation_seq=operation_seq,
                segment_id=segment_id,
                entry_key=entry_key,
                details={
                    **(deleted_result.details or {}),
                    "file_removed": removed,
                },
            )
            if self.auto_snapshot:
                self._save_snapshot_best_effort()
            return deleted_result

    def _remove_file_for_deleted_operation(self, operation_seq: int) -> bool:
        row = self.connection.execute(
            "SELECT old_file_relpath FROM operations WHERE operation_seq = ?",
            (operation_seq,),
        ).fetchone()
        if row is None:
            return False
        return self._remove_file_if_unreferenced(row["old_file_relpath"])

    # ------------------------------------------------------------------
    # Recovery and unknown entries
    # ------------------------------------------------------------------

    def _parse_observed_file(self, path: Path, fallback_key: str) -> tuple[str | None, int | None]:
        if not path.exists():
            return None, None
        size = path.stat().st_size
        try:
            observed = digest_file(path, fallback_key, None)
        except DiagnosticError:
            observed = None
        return observed, size

    def recover_unknown_entries(self) -> dict[str, Any]:
        """Recover only committed ledger state and annotate interrupted entries.

        No exit report is synthesized for the killed process. The report says so
        explicitly and is derived from committed journal rows plus observed
        filesystem state.
        """

        with self._lock:
            verify_ledger_digest(self.connection)
            prepared = self.connection.execute(
                "SELECT * FROM operations WHERE status = 'prepared' ORDER BY operation_seq"
            ).fetchall()
            prepared_paths: set[str] = set()
            new_unknown_count = 0
            for operation in prepared:
                relative = operation["file_relpath"] or operation["old_file_relpath"]
                if relative:
                    prepared_paths.add(relative)
                observed_digest = None
                observed_size = None
                if relative:
                    candidate = self.paths.absolute_entry_path(relative)
                    observed_digest, observed_size = self._parse_observed_file(
                        candidate, operation["entry_key"] or ""
                    )
                reason = (
                    "interrupted_after_file_replace"
                    if observed_digest is not None
                    else "interrupted_no_observed_file"
                )
                self._mark_unknown(
                    operation_id=operation["operation_id"],
                    operation_seq=operation["operation_seq"],
                    segment_id=operation["segment_id"] or "",
                    entry_key=operation["entry_key"] or "",
                    expected_digest=operation["expected_digest"],
                    observed_digest=observed_digest,
                    observed_size_bytes=observed_size,
                    observed_file_relpath=relative,
                    reason=reason,
                    details={
                        "operation_kind": operation["kind"],
                        "recovery_basis": "committed_prepare_journal_and_observed_file",
                    },
                )
                new_unknown_count += 1

            referenced = {
                row["file_relpath"]
                for row in self.connection.execute(
                    "SELECT file_relpath FROM entries"
                ).fetchall()
            }
            excluded = set(prepared_paths)
            for row in self.connection.execute(
                "SELECT old_file_relpath FROM operations WHERE status = 'committed' AND kind = 'delete'"
            ).fetchall():
                if row["old_file_relpath"]:
                    excluded.add(row["old_file_relpath"])
                    self._remove_file_if_unreferenced(row["old_file_relpath"])
            for row in self.connection.execute(
                "SELECT observed_file_relpath FROM unknown_entries WHERE observed_file_relpath <> ''"
            ).fetchall():
                excluded.add(row["observed_file_relpath"])

            orphan_candidates: list[tuple[str, str, Path, str]] = []
            for path in self.paths.entries.rglob("*.bin"):
                if not path.is_file():
                    continue
                relative = path.relative_to(self.paths.root).as_posix()
                if relative in referenced or relative in excluded:
                    continue
                parts = path.relative_to(self.paths.entries).parts
                if len(parts) == 3:
                    try:
                        segment_id = decode_component(parts[0])
                        entry_key = decode_component(parts[1])
                    except Exception:
                        continue
                else:
                    segment_id = ""
                    entry_key = path.stem
                orphan_candidates.append((segment_id, entry_key, path, relative))
            orphan_candidates.sort(key=lambda item: (utf8_key(item[0]), utf8_key(item[1]), item[3]))
            for segment_id, entry_key, path, relative in orphan_candidates:
                observed_digest, observed_size = self._parse_observed_file(
                    path, entry_key
                )
                self._mark_unknown(
                    operation_id="",
                    operation_seq=None,
                    segment_id=segment_id,
                    entry_key=entry_key,
                    expected_digest=None,
                    observed_digest=observed_digest,
                    observed_size_bytes=observed_size,
                    observed_file_relpath=relative,
                    reason="orphan_content_file",
                    details={"recovery_basis": "filesystem_observation"},
                )
                new_unknown_count += 1

            for temp_file in sorted(self.paths.tmp.glob(".cacheproof-*"), key=lambda item: item.name):
                try:
                    temp_file.unlink()
                except OSError:
                    pass

        unknown_rows = self.list_unknown_entries()
        result = {
            "ok": True,
            "code": "OK",
            "recovery_kind": "post_crash_recovery" if prepared else "clean_open",
            "exit_report_available": False,
            "basis": "fully committed SQLite rows plus filesystem observations",
            "newly_annotated": new_unknown_count,
            "prepared_operations_seen": len(prepared),
            "unknown_entries": unknown_rows,
        }
        if prepared or new_unknown_count:
            self.logger.emit(
                "recovery_completed",
                business_code="OK",
                details={
                    "prepared_operations_seen": len(prepared),
                    "newly_annotated": new_unknown_count,
                    "exit_report_available": False,
                },
            )
        return result

    def list_unknown_entries(self) -> list[dict[str, Any]]:
        rows = self.connection.execute("SELECT * FROM unknown_entries").fetchall()
        result = []
        for row in rows:
            result.append(
                {
                    "segment_id": row["segment_id"],
                    "entry_key": row["entry_key"],
                    "operation_id": row["operation_id"],
                    "expected_digest": row["expected_digest"],
                    "observed_digest": row["observed_digest"],
                    "observed_size_bytes": row["observed_size_bytes"],
                    "observed_file_relpath": row["observed_file_relpath"] or None,
                    "reason": row["reason"],
                }
            )
        result.sort(
            key=lambda item: (
                utf8_key(item["segment_id"]),
                utf8_key(item["entry_key"]),
                item["operation_id"],
            )
        )
        return result

    # ------------------------------------------------------------------
    # Proofs and independent verification
    # ------------------------------------------------------------------

    def build_proof(self, segment_id: str) -> dict[str, Any]:
        state = get_ledger_state(self.connection)
        assert state is not None
        segment = self.connection.execute(
            "SELECT * FROM segments WHERE segment_id = ?", (segment_id,)
        ).fetchone()
        if segment is None:
            raise DiagnosticError(
                "E_NOT_FOUND",
                "segment does not exist",
                segment_id=segment_id,
                field="segment_id",
            )
        return build_segment_proof(
            segment_id=segment_id,
            capacity_bytes=segment["capacity_bytes"],
            segment_set_version=state["segment_set_version"],
            segment_version=segment["segment_version"],
            entries=self._entries_for_segment(self.connection, segment_id),
        )

    def _segment_file_set(self, segment_id: str) -> set[str]:
        segment_directory = self.paths.entries / encode_component(segment_id)
        if not segment_directory.exists():
            return set()
        return {
            path.relative_to(self.paths.root).as_posix()
            for path in segment_directory.rglob("*.bin")
            if path.is_file()
        }

    def _verify_segment(self, segment_id: str) -> dict[str, Any]:
        state = get_ledger_state(self.connection)
        assert state is not None
        row = self.connection.execute(
            "SELECT * FROM segments WHERE segment_id = ?", (segment_id,)
        ).fetchone()
        if row is None:
            raise DiagnosticError(
                "E_NOT_FOUND",
                "segment does not exist",
                segment_id=segment_id,
                field="segment_id",
            )
        entries = self._entries_for_segment(self.connection, segment_id)
        current = build_segment_proof(
            segment_id=segment_id,
            capacity_bytes=row["capacity_bytes"],
            segment_set_version=state["segment_set_version"],
            segment_version=row["segment_version"],
            entries=entries,
        )
        stored = loads_json(row["proof_json"], field=f"segments[{segment_id}].proof_json")
        try:
            verify_proof_digest(stored)
        except DiagnosticError as exc:
            exc.segment_id = segment_id
            raise
        if canonical_text(stored) != canonical_text(current):
            raise DiagnosticError(
                "E_INTEGRITY",
                "stored capacity proof differs from the independently rebuilt proof",
                segment_id=segment_id,
                field="proof",
                details={
                    "stored_digest": stored.get("content_digest"),
                    "rebuilt_digest": current.get("content_digest"),
                },
            )
        if current["used_bytes"] + current["remaining_bytes"] != current["capacity_bytes"]:
            raise DiagnosticError(
                "E_INTEGRITY",
                "used bytes and remaining bytes do not sum to declared capacity",
                segment_id=segment_id,
                field="used_bytes",
                details=current,
            )
        if current["used_bytes"] > current["capacity_bytes"]:
            raise DiagnosticError(
                "E_CAPACITY",
                "used bytes exceed declared capacity",
                segment_id=segment_id,
                field="capacity_bytes",
                details={
                    "used_bytes": current["used_bytes"],
                    "capacity_bytes": current["capacity_bytes"],
                },
            )
        if row["used_bytes"] != current["used_bytes"] or row["entry_count"] != current["entry_count"]:
            raise DiagnosticError(
                "E_INTEGRITY",
                "segment metadata does not match the capacity proof",
                segment_id=segment_id,
                field="used_bytes",
                details={
                    "stored_used_bytes": row["used_bytes"],
                    "proof_used_bytes": current["used_bytes"],
                    "stored_entry_count": row["entry_count"],
                    "proof_entry_count": current["entry_count"],
                },
            )

        entry_reports = []
        expected_files: set[str] = set()
        for entry in entries:
            relative = entry["file_relpath"]
            expected_files.add(relative)
            path = self.paths.absolute_entry_path(relative)
            if not path.exists():
                raise DiagnosticError(
                    "E_INTEGRITY",
                    "entry content file is missing",
                    segment_id=segment_id,
                    entry_key=entry["entry_key"],
                    field="content_file",
                    details={"file_relpath": relative},
                )
            try:
                actual_digest = digest_file(
                    path, entry["entry_key"], entry["size_bytes"]
                )
            except DiagnosticError as exc:
                exc.segment_id = segment_id
                if exc.entry_key is None:
                    exc.entry_key = entry["entry_key"]
                if exc.field is None:
                    exc.field = "digest"
                raise
            if actual_digest != entry["digest"]:
                raise DiagnosticError(
                    "E_INTEGRITY",
                    "entry content digest does not match the ledger",
                    segment_id=segment_id,
                    entry_key=entry["entry_key"],
                    field="digest",
                    details={
                        "expected_digest": entry["digest"],
                        "actual_digest": actual_digest,
                        "file_relpath": relative,
                    },
                )
            entry_reports.append(
                {
                    "key": entry["entry_key"],
                    "size_bytes": entry["size_bytes"],
                    "digest": entry["digest"],
                    "actual_digest": actual_digest,
                    "digest_ok": True,
                }
            )
        actual_files = self._segment_file_set(segment_id)
        known_unknown_files = {
            row["observed_file_relpath"]
            for row in self.connection.execute(
                "SELECT observed_file_relpath FROM unknown_entries WHERE observed_file_relpath <> ''"
            ).fetchall()
        }
        extra_files = sorted(
            actual_files - expected_files - known_unknown_files, key=utf8_key
        )
        if extra_files:
            raise DiagnosticError(
                "E_INTEGRITY",
                "segment directory contains unreferenced content files",
                segment_id=segment_id,
                field="content_directory",
                details={"extra_files": extra_files},
            )
        entry_reports.sort(key=lambda item: utf8_key(item["key"]))
        return {
            "segment_id": segment_id,
            "capacity_bytes": current["capacity_bytes"],
            "used_bytes": current["used_bytes"],
            "remaining_bytes": current["remaining_bytes"],
            "entry_count": current["entry_count"],
            "segment_version": current["segment_version"],
            "segment_set_version": current["segment_set_version"],
            "proof_content_digest": current["content_digest"],
            "proof_ok": True,
            "entries": entry_reports,
        }

    def verify_all(
        self,
        *,
        capacity_override: dict[str, int] | None = None,
    ) -> dict[str, Any]:
        state = verify_ledger_digest(self.connection)
        overrides = capacity_override or {}
        for segment_id, capacity in overrides.items():
            if not _is_int(capacity) or capacity < 0:
                raise spec_error(
                    "capacity override must be a non-negative integer",
                    segment_id=segment_id,
                    field="capacity_bytes",
                )
        segments = [
            row["segment_id"]
            for row in self.connection.execute(
                "SELECT segment_id, used_bytes, capacity_bytes FROM segments"
            ).fetchall()
        ]
        segments.sort(key=utf8_key)
        for segment_id in segments:
            row = self.connection.execute(
                "SELECT * FROM segments WHERE segment_id = ?", (segment_id,)
            ).fetchone()
            override = overrides.get(segment_id)
            if override is not None and row["used_bytes"] > override:
                raise DiagnosticError(
                    "E_CAPACITY",
                    "capacity is smaller than committed used bytes",
                    segment_id=segment_id,
                    field="capacity_bytes",
                    details={
                        "required_bytes": row["used_bytes"],
                        "available_bytes": max(0, override),
                        "capacity_bytes": override,
                        "used_bytes": row["used_bytes"],
                    },
                )
        segment_reports = [self._verify_segment(segment_id) for segment_id in segments]

        views = []
        rows = self.connection.execute("SELECT * FROM read_views").fetchall()
        sorted_rows = sorted(rows, key=lambda row: utf8_key(row["view_id"]))
        for row in sorted_rows:
            snapshot = loads_json(
                row["snapshot_json"], field=f"read_views[{row['view_id']}].snapshot"
            )
            validate_view_snapshot(snapshot)
            if row["snapshot_digest"] != snapshot["content_digest"]:
                raise DiagnosticError(
                    "E_SNAPSHOT",
                    "read-view stored digest differs from its snapshot",
                    field=f"read_views[{row['view_id']}].snapshot_digest",
                    details={
                        "stored_digest": row["snapshot_digest"],
                        "computed_digest": snapshot["content_digest"],
                    },
                )
            views.append(
                {
                    "view_id": row["view_id"],
                    "view_version": row["view_version"],
                    "snapshot_digest": row["snapshot_digest"],
                    "segment_set_version": row["segment_set_version"],
                }
            )
        unknown = self.list_unknown_entries()
        return {
            "ok": True,
            "code": "OK",
            "ledger": state,
            "segments": segment_reports,
            "views": views,
            "unknown_entries": unknown,
        }

    # ------------------------------------------------------------------
    # Read views: atomic, whole-row rebuilds
    # ------------------------------------------------------------------

    def _view_snapshot_from_row(
        self,
        row: sqlite3.Row,
        *,
        version: int | None = None,
        segment_set_version: int | None = None,
    ) -> dict[str, Any]:
        old = validate_view_snapshot(
            loads_json(row["snapshot_json"], field=f"read_views[{row['view_id']}].snapshot")
        )
        state = get_ledger_state(self.connection)
        assert state is not None
        segments = []
        for old_segment in old.get("segments", []):
            segment = self.connection.execute(
                "SELECT * FROM segments WHERE segment_id = ?",
                (old_segment["segment_id"],),
            ).fetchone()
            if segment is None:
                raise DiagnosticError(
                    "E_SNAPSHOT",
                    "read view references a missing segment",
                    segment_id=old_segment["segment_id"],
                    field="snapshot.segments",
                )
            entries = self._entries_for_segment(
                self.connection, old_segment["segment_id"]
            )
            segments.append(
                {
                    "segment_id": old_segment["segment_id"],
                    "segment_version": segment["segment_version"],
                    "entries": [
                        {"key": entry["entry_key"], "digest": entry["digest"]}
                        for entry in entries
                    ],
                }
            )
        return build_view_snapshot(
            view_id=row["view_id"],
            view_version=row["view_version"] if version is None else version,
            segment_set_version=(
                state["segment_set_version"]
                if segment_set_version is None
                else segment_set_version
            ),
            segments=segments,
        )

    def _load_view_row(self, view_id: str) -> sqlite3.Row:
        row = self.connection.execute(
            "SELECT * FROM read_views WHERE view_id = ?", (view_id,)
        ).fetchone()
        if row is None:
            raise DiagnosticError(
                "E_NOT_FOUND", "read view does not exist", field="view_id"
            )
        return row

    def preview_view(self, view_id: str) -> dict[str, Any]:
        """Return stale status and pending differences without changing SQLite."""

        row = self._load_view_row(view_id)
        old = validate_view_snapshot(
            loads_json(row["snapshot_json"], field=f"read_views[{view_id}].snapshot")
        )
        current = self._view_snapshot_from_row(row)
        stale = view_is_stale(old, current)
        expected_rebuild = build_view_snapshot(
            view_id=view_id,
            view_version=row["view_version"] + 1,
            segment_set_version=current["segment_set_version"],
            segments=current["segments"],
        )
        diff = compute_view_diff(old, expected_rebuild) if stale else {}
        last_differences = loads_json(
            row["last_differences_json"],
            field=f"read_views[{view_id}].last_differences_json",
        )
        return {
            "ok": True,
            "code": "OK",
            "view_id": view_id,
            "view_version": row["view_version"],
            "snapshot_digest": row["snapshot_digest"],
            "stale": stale,
            "pending_differences": diff,
            "last_rebuild_differences": last_differences,
        }

    def get_view(self, view_id: str, *, rebuild: bool = True) -> dict[str, Any]:
        with self._lock:
            if not rebuild:
                return self.preview_view(view_id)
            with transaction(self.connection):
                row = self._load_view_row(view_id)
                old = validate_view_snapshot(
                    loads_json(
                        row["snapshot_json"],
                        field=f"read_views[{view_id}].snapshot",
                    )
                )
                current_at_old_version = self._view_snapshot_from_row(row)
                if not view_is_stale(old, current_at_old_version):
                    return {
                        "ok": True,
                        "code": "OK",
                        "view_id": view_id,
                        "view_version": row["view_version"],
                        "snapshot": old,
                        "snapshot_digest": row["snapshot_digest"],
                        "stale": False,
                        "rebuilt": False,
                        "differences": loads_json(
                            row["last_differences_json"],
                            field=f"read_views[{view_id}].last_differences_json",
                        ),
                    }
                new_version = row["view_version"] + 1
                rebuilt_snapshot = build_view_snapshot(
                    view_id=view_id,
                    view_version=new_version,
                    segment_set_version=current_at_old_version["segment_set_version"],
                    segments=current_at_old_version["segments"],
                )
                differences = compute_view_diff(old, rebuilt_snapshot)
                self.connection.execute(
                    """
                    UPDATE read_views
                    SET view_version = ?, segment_set_version = ?,
                        snapshot_json = ?, snapshot_digest = ?,
                        last_differences_json = ?, updated_monotonic_ns = ?
                    WHERE view_id = ?
                    """,
                    (
                        new_version,
                        rebuilt_snapshot["segment_set_version"],
                        canonical_text(rebuilt_snapshot),
                        rebuilt_snapshot["content_digest"],
                        canonical_text(differences),
                        time.monotonic_ns(),
                        view_id,
                    ),
                )
                refresh_ledger_digest(self.connection, increment_batch=True)
            self.logger.emit(
                "read_view_rebuilt",
                business_code="OK",
                details={
                    "view_id": view_id,
                    "old_view_version": old["view_version"],
                    "new_view_version": new_version,
                    "differences": differences,
                },
            )
            if self.auto_snapshot:
                self._save_snapshot_best_effort()
            return {
                "ok": True,
                "code": "OK",
                "view_id": view_id,
                "view_version": new_version,
                "snapshot": rebuilt_snapshot,
                "snapshot_digest": rebuilt_snapshot["content_digest"],
                "stale": True,
                "rebuilt": True,
                "differences": differences,
            }

    def ensure_all_views(self) -> list[dict[str, Any]]:
        view_ids = [
            row["view_id"]
            for row in self.connection.execute(
                "SELECT view_id FROM read_views"
            ).fetchall()
        ]
        view_ids.sort(key=utf8_key)
        return [self.get_view(view_id, rebuild=True) for view_id in view_ids]

    # ------------------------------------------------------------------
    # Session snapshots and post-snapshot replay
    # ------------------------------------------------------------------

    def _collect_session_payload(self) -> dict[str, Any]:
        state = get_ledger_state(self.connection)
        if state is None:
            raise DiagnosticError(
                "E_NOT_FOUND", "ledger state is missing", field="ledger_state"
            )
        segment_rows = self.connection.execute(
            "SELECT * FROM segments"
        ).fetchall()
        segment_rows = sorted(
            segment_rows, key=lambda row: utf8_key(row["segment_id"])
        )
        proofs = []
        for row in segment_rows:
            proofs.append(
                build_segment_proof(
                    segment_id=row["segment_id"],
                    capacity_bytes=row["capacity_bytes"],
                    segment_set_version=state["segment_set_version"],
                    segment_version=row["segment_version"],
                    entries=self._entries_for_segment(
                        self.connection, row["segment_id"]
                    ),
                )
            )

        view_rows = self.connection.execute("SELECT * FROM read_views").fetchall()
        view_rows = sorted(view_rows, key=lambda row: utf8_key(row["view_id"]))
        views = []
        for row in view_rows:
            view_snapshot = validate_view_snapshot(
                loads_json(
                    row["snapshot_json"],
                    field=f"read_views[{row['view_id']}].snapshot_json",
                )
            )
            views.append(
                {
                    "view_id": row["view_id"],
                    "view_version": row["view_version"],
                    "segment_set_version": row["segment_set_version"],
                    "snapshot": view_snapshot,
                    "snapshot_digest": row["snapshot_digest"],
                    "last_differences": loads_json(
                        row["last_differences_json"],
                        field=f"read_views[{row['view_id']}].last_differences_json",
                    ),
                }
            )
        last_operation_seq_row = self.connection.execute(
            "SELECT COALESCE(MAX(operation_seq), 0) AS value FROM operations"
        ).fetchone()
        payload = {
            "format_version": SESSION_SNAPSHOT_FORMAT,
            "segment_set_version": state["segment_set_version"],
            "ledger_batch": state["batch_id"],
            "ledger_content_digest": state["content_digest"],
            "completed_cursor": state["completed_cursor"],
            "last_operation_seq": last_operation_seq_row["value"],
            "segments": proofs,
            "views": views,
            "unknown_entries": self.list_unknown_entries(),
        }
        return payload

    def save_snapshot(self) -> dict[str, Any]:
        with self._lock:
            verify_ledger_digest(self.connection)
            payload = self._collect_session_payload()
            sealed = seal_session_snapshot(payload)
            atomic_write_json(self.paths.session_snapshot, sealed, self.paths.tmp)
            self.logger.emit(
                "session_snapshot_saved",
                business_code="OK",
                details={
                    "content_digest": sealed["content_digest"],
                    "completed_cursor": sealed["completed_cursor"],
                    "ledger_batch": sealed["ledger_batch"],
                    "last_operation_seq": sealed["last_operation_seq"],
                },
            )
            return {
                "ok": True,
                "code": "OK",
                "path": str(self.paths.session_snapshot),
                "content_digest": sealed["content_digest"],
                "completed_cursor": sealed["completed_cursor"],
                "ledger_batch": sealed["ledger_batch"],
                "last_operation_seq": sealed["last_operation_seq"],
            }

    def _save_snapshot_best_effort(self) -> None:
        try:
            self.save_snapshot()
        except Exception as exc:
            self.logger.emit(
                "session_snapshot_save_failed",
                business_code="E_SNAPSHOT",
                field="snapshot",
                details={"error": str(exc)},
            )

    def _load_session_snapshot(self) -> dict[str, Any]:
        if not self.paths.session_snapshot.exists():
            raise DiagnosticError(
                "E_NOT_FOUND",
                "session snapshot does not exist",
                field="snapshot",
                details={"path": str(self.paths.session_snapshot)},
            )
        try:
            value = json.loads(
                self.paths.session_snapshot.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            raise DiagnosticError(
                "E_SNAPSHOT",
                "session snapshot is not readable JSON",
                field="snapshot",
                details={"error": str(exc)},
            ) from exc
        return verify_session_snapshot(value)

    def restore_snapshot(self) -> dict[str, Any]:
        """Compare a checkpoint with the ledger, replay remaining journal rows.

        A stale but replay-consistent checkpoint is replaced wholesale by the
        current committed state. A replay mismatch raises E_REPLAY with the
        first segment/key/field discrepancy.
        """

        with self._lock:
            verify_ledger_digest(self.connection)
            snapshot = self._load_session_snapshot()
            current_before = self._collect_session_payload()
            old_auto_snapshot = self.auto_snapshot
            operation_rows = self.connection.execute(
                """
                SELECT * FROM operations
                WHERE operation_seq > ?
                ORDER BY operation_seq
                """,
                (snapshot["last_operation_seq"],),
            ).fetchall()
            operations = [dict(row) for row in operation_rows]
            snapshot_matches = canonical_text(
                {key: value for key, value in snapshot.items() if key != "content_digest"}
            ) == canonical_text(current_before)
            if snapshot_matches:
                return {
                    "ok": True,
                    "code": "OK",
                    "snapshot_replaced": False,
                    "replayed_operations": 0,
                    "operation_codes": [],
                    "differences": {"changed": False, "changes": []},
                    "content_digest": snapshot["content_digest"],
                }

            try:
                self.auto_snapshot = False
                self.ensure_all_views()
            finally:
                self.auto_snapshot = old_auto_snapshot
            live = self._collect_session_payload()
            replay = replay_operations(snapshot, operations)
            compare_replayed_state(replay, live)
            expected_views = expected_replayed_views(snapshot, replay)
            compare_replayed_views(expected_views, live["views"])
            differences = snapshot_diff(snapshot, live)
            sealed_live = seal_session_snapshot(live)
            atomic_write_json(self.paths.session_snapshot, sealed_live, self.paths.tmp)
            self.logger.emit(
                "session_snapshot_restored",
                business_code="OK",
                details={
                    "snapshot_replaced": True,
                    "replayed_operations": len(operations),
                    "changed_fields": len(differences["changes"]),
                    "old_digest": snapshot["content_digest"],
                    "new_digest": sealed_live["content_digest"],
                },
            )
            return {
                "ok": True,
                "code": "OK",
                "snapshot_replaced": True,
                "replayed_operations": len(operations),
                "operation_codes": replay["operation_codes"],
                "differences": differences,
                "content_digest": sealed_live["content_digest"],
            }
