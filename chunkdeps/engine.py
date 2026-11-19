"""Business engine for deterministic chunk uploads and dependency caches."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from .errors import (
    E_AUDIT,
    E_CHUNK,
    E_CONFLICT,
    E_INTEGRITY,
    E_INTERNAL,
    E_INPUT,
    E_NOT_FOUND,
    E_STATE,
    E_VERSION,
    BusinessError,
)
from .ledger import Ledger
from .logging import JsonlLogger
from .util import (
    SUPPORTED_DIGESTS,
    atomic_write_bytes,
    atomic_write_json,
    b64decode,
    bitmap_from_indices,
    canonical_text,
    chunk_count,
    digest_bytes,
    digest_file,
    digest_json,
    expected_chunk_size,
    indices_from_bitmap,
    missing_indices,
    mono_ms,
    prefixes_for_key,
    tags_canonical,
    validate_chunk_size,
    validate_object_key,
    validate_total_size,
    view_prefix_matches,
)


TERMINAL_STATES = frozenset({"committed"})
RESUMABLE_STATES = frozenset({"created", "uploading", "aborted"})


class Workbench:
    """The local storage-service state machine and dependency workbench."""

    def __init__(
        self,
        root: str | Path,
        *,
        log_path: str | Path | None = None,
        logger: JsonlLogger | None = None,
        merge_pause_ms: int | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self.ledger = Ledger(root)
        if logger is not None:
            self.logger = logger
        else:
            logger_path = (
                Path(log_path) if log_path else self.ledger.logs_dir / "events.jsonl"
            )
            self.logger = JsonlLogger(logger_path)
        self.merge_pause_ms = (
            merge_pause_ms
            if merge_pause_ms is not None
            else int(os.environ.get("CHUNKDEPS_MERGE_PAUSE_MS", "0") or "0")
        )
        self.recovered_on_start = self._recover_after_restart()
        validation = self.ledger.graph.validate()
        self.logger.event(
            "workbench_opened",
            root=str(self.ledger.root),
            graph_validation=validation,
            recovered_sessions=len(self.recovered_on_start),
        )

    @property
    def graph(self):
        return self.ledger.graph

    @property
    def conn(self) -> sqlite3.Connection:
        return self.ledger.conn

    def close(self) -> None:
        with self._lock:
            self.ledger.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        self.conn.execute("BEGIN IMMEDIATE")
        try:
            yield self.conn
        except Exception:
            self.conn.rollback()
            raise
        else:
            self.conn.commit()

    def _recover_after_restart(self) -> list[dict[str, Any]]:
        now = mono_ms()
        rows = list(
            self.conn.execute(
                "SELECT * FROM sessions WHERE state='verifying' AND completed=0 ORDER BY session_id"
            )
        )
        recovered: list[dict[str, Any]] = []
        for row in rows:
            with self._transaction() as conn:
                conn.execute(
                    """
                    UPDATE sessions
                       SET commit_unknown=1,
                           recovery_status='unknown_needs_rerun',
                           updated_mono_ms=?
                     WHERE session_id=?
                    """,
                    (now, row["session_id"]),
                )
                conn.execute(
                    """
                    INSERT INTO session_events(
                        session_id, from_state, to_state, mono_ms, bitmap,
                        reason, commit_unknown
                    ) VALUES(?,?,?,?,?,?,1)
                    """,
                    (
                        row["session_id"],
                        "verifying",
                        "verifying",
                        now,
                        row["bitmap"],
                        "process restart while merge outcome was unknown; rerun required",
                    ),
                )
            recovered.append(
                {
                    "session_id": row["session_id"],
                    "object_key": row["object_key"],
                    "state": "verifying",
                    "recovery_status": "unknown_needs_rerun",
                    "commit_unknown": True,
                    "message": "merge was interrupted; no visible object was inferred",
                }
            )
            self.logger.event(
                "recovery_unknown_merge",
                session_id=row["session_id"],
                object_key=row["object_key"],
                exit_report_available=False,
            )

        for path in sorted(self.ledger.temp_dir.glob("*")):
            if path.is_file():
                path.unlink(missing_ok=True)
        return recovered

    # ------------------------------------------------------------------
    # Serialization helpers
    # ------------------------------------------------------------------
    def _session_payload(self, row: sqlite3.Row | None, *, include_chunks: bool = False) -> dict[str, Any]:
        if row is None:
            raise BusinessError(E_NOT_FOUND, "upload session not found")
        bitmap = row["bitmap"]
        payload: dict[str, Any] = {
            "session_id": row["session_id"],
            "object_key": row["object_key"],
            "chunk_size": row["chunk_size"],
            "total_size": row["total_size"],
            "chunk_count": row["chunk_count"],
            "digest_algorithm": row["digest_algorithm"],
            "tags": json.loads(row["tags_json"]),
            "state": row["state"],
            "bitmap": bitmap,
            "registered_count": row["registered_count"],
            "registered_indices": indices_from_bitmap(bitmap),
            "missing_indices": missing_indices(bitmap),
            "created_mono_ms": row["created_mono_ms"],
            "updated_mono_ms": row["updated_mono_ms"],
            "completed": bool(row["completed"]),
            "commit_unknown": bool(row["commit_unknown"]),
            "recovery_status": row["recovery_status"],
            "expected_base_version": row["expected_base_version"],
            "version": row["version"],
            "content_digest": row["content_digest"],
            "object_path": row["object_path"],
            "abort_reason": row["abort_reason"],
        }
        if include_chunks:
            payload["chunks"] = [
                self._chunk_payload(chunk) for chunk in self.ledger.chunk_rows(row["session_id"])
            ]
        return payload

    def _chunk_payload(self, row: sqlite3.Row) -> dict[str, Any]:
        return {
            "session_id": row["session_id"],
            "chunk_index": row["chunk_index"],
            "digest": row["digest"],
            "size": row["size"],
            "file_path": row["file_path"],
            "registered_mono_ms": row["registered_mono_ms"],
        }

    def _version_payload(self, row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        return {
            "object_key": row["object_key"],
            "version": row["version"],
            "session_id": row["session_id"],
            "metadata_digest": row["metadata_digest"],
            "manifest_digest": row["manifest_digest"],
            "content_digest": row["content_digest"],
            "chunk_size": row["chunk_size"],
            "chunk_count": row["chunk_count"],
            "chunk_set_digest": row["chunk_set_digest"],
            "digest_algorithm": row["digest_algorithm"],
            "tags": json.loads(row["tags_json"]),
            "status": row["status"],
            "visible": bool(row["visible"]),
            "object_path": row["object_path"],
            "created_mono_ms": row["created_mono_ms"],
            "replaced_mono_ms": row["replaced_mono_ms"],
        }

    def _get_session(self, session_id: str) -> sqlite3.Row:
        row = self.ledger.session_row(session_id)
        if row is None:
            raise BusinessError(E_NOT_FOUND, "upload session not found", {"session_id": session_id})
        return row

    def _transition(
        self,
        session_id: str,
        from_state: str,
        to_state: str,
        *,
        bitmap: str,
        reason: str,
        commit_unknown: bool = False,
        extra_updates: dict[str, Any] | None = None,
    ) -> None:
        now = mono_ms()
        assignments = [
            "state=?",
            "updated_mono_ms=?",
            "commit_unknown=?",
        ]
        values: list[Any] = [to_state, now, 1 if commit_unknown else 0]
        if extra_updates:
            for key, value in extra_updates.items():
                assignments.append(f"{key}=?")
                values.append(value)
        values.append(session_id)
        self.conn.execute(
            f"UPDATE sessions SET {', '.join(assignments)} WHERE session_id=?",
            values,
        )
        self.conn.execute(
            """
            INSERT INTO session_events(
                session_id, from_state, to_state, mono_ms, bitmap, reason, commit_unknown
            ) VALUES(?,?,?,?,?,?,?)
            """,
            (
                session_id,
                from_state,
                to_state,
                now,
                bitmap,
                reason,
                1 if commit_unknown else 0,
            ),
        )
        self.logger.event(
            "session_transition",
            session_id=session_id,
            from_state=from_state,
            to_state=to_state,
            bitmap=bitmap,
            reason=reason,
            commit_unknown=bool(commit_unknown),
        )

    # ------------------------------------------------------------------
    # Dependencies and derived values
    # ------------------------------------------------------------------
    @staticmethod
    def _chunk_node_id(session_id: str, chunk_index: int) -> str:
        return f"chunk:{session_id}:{chunk_index:08d}"

    @staticmethod
    def _manifest_node_id(session_id: str) -> str:
        return f"manifest:{session_id}"

    @staticmethod
    def _stage_metadata_node_id(session_id: str) -> str:
        return f"stage-metadata:{session_id}"

    @staticmethod
    def _session_state_node_id(session_id: str) -> str:
        return f"session-state:{session_id}"

    @staticmethod
    def _session_summary_node_id(session_id: str) -> str:
        return f"session-summary:{session_id}"

    @staticmethod
    def _key_node_id(object_key: str) -> str:
        return f"object-key:{object_key}"

    @staticmethod
    def _metadata_node_id(object_key: str, version: int) -> str:
        return f"metadata:{object_key}:v{version}"

    @staticmethod
    def _view_node_id(prefix: str) -> str:
        return f"prefix-view:{prefix}"

    def _manifest_digest(self, session: sqlite3.Row, chunks: list[sqlite3.Row]) -> str:
        manifest = {
            "schema": "chunk-manifest/v1",
            "object_key": session["object_key"],
            "session_id": session["session_id"],
            "chunk_size": session["chunk_size"],
            "chunk_count": session["chunk_count"],
            "digest_algorithm": session["digest_algorithm"],
            "chunks": [
                {
                    "chunk_index": row["chunk_index"],
                    "digest": row["digest"],
                    "size": row["size"],
                }
                for row in chunks
            ],
        }
        return digest_json(manifest, session["digest_algorithm"])

    def _chunk_set_digest(self, session: sqlite3.Row, chunks: list[sqlite3.Row]) -> str:
        payload = {
            "schema": "chunk-set/v1",
            "chunk_count": session["chunk_count"],
            "chunks": [
                [row["chunk_index"], row["digest"], row["size"]] for row in chunks
            ],
        }
        return digest_json(payload, session["digest_algorithm"])

    def _metadata_fields_from_session(
        self,
        session: sqlite3.Row,
        *,
        version: int,
        content_digest: str,
        manifest_digest: str,
        chunk_set_digest: str,
    ) -> dict[str, Any]:
        return {
            "schema": "object-metadata/v1",
            "object_key": session["object_key"],
            "version": version,
            "session_id": session["session_id"],
            "chunk_size": session["chunk_size"],
            "chunk_count": session["chunk_count"],
            "digest_algorithm": session["digest_algorithm"],
            "tags": json.loads(session["tags_json"]),
            "content_digest": content_digest,
            "manifest_digest": manifest_digest,
            "chunk_set_digest": chunk_set_digest,
        }

    def _metadata_digest_from_version(self, row: sqlite3.Row) -> str:
        fields = {
            "schema": "object-metadata/v1",
            "object_key": row["object_key"],
            "version": row["version"],
            "session_id": row["session_id"],
            "chunk_size": row["chunk_size"],
            "chunk_count": row["chunk_count"],
            "digest_algorithm": row["digest_algorithm"],
            "tags": json.loads(row["tags_json"]),
            "content_digest": row["content_digest"],
            "manifest_digest": row["manifest_digest"],
            "chunk_set_digest": row["chunk_set_digest"],
        }
        return digest_json(fields, row["digest_algorithm"])

    def _key_state(self, object_key: str) -> tuple[int, str | None, str]:
        active = self.ledger.active_version_row(object_key)
        if active is not None:
            return active["version"], active["metadata_digest"], "active"
        row = self.conn.execute(
            "SELECT * FROM object_versions WHERE object_key=? ORDER BY version DESC LIMIT 1",
            (object_key,),
        ).fetchone()
        if row is not None and row["status"] == "deleted":
            return row["version"], row["metadata_digest"], "deleted"
        return 0, None, "none"

    def _key_digest(self, object_key: str) -> tuple[str, dict[str, Any]]:
        version, metadata_digest, status = self._key_state(object_key)
        digest = self.graph.node_digest_for_key(
            object_key, version, metadata_digest, status
        )
        return digest, {
            "object_key": object_key,
            "active_version": version if status == "active" else 0,
            "latest_version": version,
            "metadata_digest": metadata_digest,
            "status": status,
        }

    def _ensure_key_node(self, object_key: str) -> dict[str, Any]:
        digest, payload = self._key_digest(object_key)
        node = self.graph.ensure_node(
            self._key_node_id(object_key), "object_key", object_key, digest, payload
        )
        active = self.ledger.active_version_row(object_key)
        if active is not None:
            metadata_node = self._ensure_version_node(active)
            self.graph.add_edge(metadata_node["node_id"], node["node_id"])
        return node

    def _ensure_version_node(self, version_row: sqlite3.Row) -> dict[str, Any]:
        session = self._get_session(version_row["session_id"])
        chunks = self.ledger.chunk_rows(session["session_id"])
        manifest_id = self._manifest_node_id(session["session_id"])
        manifest_digest = version_row["manifest_digest"]
        if self.graph.node(manifest_id) is None:
            manifest_payload = {
                "session_id": session["session_id"],
                "object_key": session["object_key"],
                "digest": manifest_digest,
                "chunk_count": len(chunks),
            }
            self.graph.ensure_node(
                manifest_id, "manifest", session["session_id"], manifest_digest, manifest_payload
            )
        for chunk in chunks:
            chunk_id = self._chunk_node_id(session["session_id"], chunk["chunk_index"])
            if self.graph.node(chunk_id) is None:
                self.graph.ensure_node(
                    chunk_id,
                    "chunk",
                    f"{session['session_id']}:{chunk['chunk_index']}",
                    chunk["digest"],
                    {"digest": chunk["digest"], "size": chunk["size"]},
                )
            self.graph.add_edge(chunk_id, manifest_id)

        metadata_id = self._metadata_node_id(version_row["object_key"], version_row["version"])
        metadata_payload = self._version_payload(version_row)
        assert metadata_payload is not None
        metadata_node = self.graph.ensure_node(
            metadata_id,
            "metadata",
            f"{version_row['object_key']}@v{version_row['version']}",
            version_row["metadata_digest"],
            metadata_payload,
        )
        self.graph.add_edge(manifest_id, metadata_id)
        key_node = self.graph.node(self._key_node_id(version_row["object_key"]))
        if key_node is None:
            key_digest, key_payload = self._key_digest(version_row["object_key"])
            key_node = self.graph.ensure_node(
                self._key_node_id(version_row["object_key"]),
                "object_key",
                version_row["object_key"],
                key_digest,
                key_payload,
            )
        self.graph.add_edge(metadata_id, key_node["node_id"])
        return metadata_node

    def _session_state_digest(self, session: sqlite3.Row) -> str:
        return digest_json(
            {
                "session_id": session["session_id"],
                "state": session["state"],
                "bitmap": session["bitmap"],
                "completed": bool(session["completed"]),
                "commit_unknown": bool(session["commit_unknown"]),
            }
        )

    def _ensure_session_graph(self, session: sqlite3.Row) -> dict[str, str]:
        session_id = session["session_id"]
        chunks = self.ledger.chunk_rows(session_id)
        key_node = self._ensure_key_node(session["object_key"])
        state_id = self._session_state_node_id(session_id)
        state_node = self.graph.ensure_node(
            state_id,
            "session_state",
            session_id,
            self._session_state_digest(session),
            {
                "session_id": session_id,
                "state": session["state"],
                "bitmap": session["bitmap"],
                "commit_unknown": bool(session["commit_unknown"]),
            },
        )

        manifest_id = self._manifest_node_id(session_id)
        manifest_digest = self._manifest_digest(session, chunks) if chunks else None
        manifest_node = self.graph.ensure_node(
            manifest_id,
            "manifest",
            session_id,
            manifest_digest,
            {
                "session_id": session_id,
                "object_key": session["object_key"],
                "chunk_count": len(chunks),
                "digest": manifest_digest,
            },
            valid=manifest_digest is not None,
        )
        self.graph.add_edge(state_id, manifest_id)
        for chunk in chunks:
            chunk_id = self._chunk_node_id(session_id, chunk["chunk_index"])
            if self.graph.node(chunk_id) is None:
                self.graph.ensure_node(
                    chunk_id,
                    "chunk",
                    f"{session_id}:{chunk['chunk_index']}",
                    chunk["digest"],
                    {
                        "session_id": session_id,
                        "chunk_index": chunk["chunk_index"],
                        "digest": chunk["digest"],
                        "size": chunk["size"],
                    },
                )
            self.graph.add_edge(chunk_id, manifest_id)

        stage_id = self._stage_metadata_node_id(session_id)
        stage_digest = (
            digest_json(
                {
                    "session_id": session_id,
                    "object_key": session["object_key"],
                    "manifest_digest": manifest_digest,
                    "tags": json.loads(session["tags_json"]),
                    "chunk_size": session["chunk_size"],
                    "digest_algorithm": session["digest_algorithm"],
                }
            )
            if manifest_digest
            else None
        )
        stage_node = self.graph.ensure_node(
            stage_id,
            "metadata_stage",
            session_id,
            stage_digest,
            {
                "session_id": session_id,
                "object_key": session["object_key"],
                "manifest_digest": manifest_digest,
            },
            valid=stage_digest is not None,
        )
        self.graph.add_edge(manifest_id, stage_id)
        self.graph.add_edge(key_node["node_id"], stage_id)

        summary_id = self._session_summary_node_id(session_id)
        summary_digest = digest_json(
            {
                "session_id": session_id,
                "state": session["state"],
                "bitmap": session["bitmap"],
                "stage_metadata_digest": stage_digest,
            }
        )
        summary_node = self.graph.ensure_node(
            summary_id,
            "session_summary",
            session_id,
            summary_digest,
            {
                "session_id": session_id,
                "state": session["state"],
                "bitmap": session["bitmap"],
            },
        )
        self.graph.add_edge(stage_id, summary_id)
        return {
            "key": key_node["node_id"],
            "state": state_node["node_id"],
            "manifest": manifest_node["node_id"],
            "stage_metadata": stage_node["node_id"],
            "session_summary": summary_node["node_id"],
        }

    # ------------------------------------------------------------------
    # Session and chunk operations
    # ------------------------------------------------------------------
    def create_session(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            object_key = validate_object_key(params.get("object_key"))
            chunk_size = validate_chunk_size(params.get("chunk_size"))
            total_size = validate_total_size(params.get("total_size"))
            algorithm = params.get("digest_algorithm", "sha256")
            if algorithm not in SUPPORTED_DIGESTS:
                raise BusinessError(
                    E_INPUT,
                    "unsupported digest algorithm",
                    {"algorithm": algorithm, "supported": sorted(SUPPORTED_DIGESTS)},
                )
            tags, tags_json = tags_canonical(params.get("tags"))
            expected_base = params.get("expected_base_version")
            if expected_base is not None:
                if isinstance(expected_base, bool) or not isinstance(expected_base, int) or expected_base < 0:
                    raise BusinessError(E_VERSION, "expected base version must be a non-negative integer")
            count = chunk_count(total_size, chunk_size)
            session_id = "s-" + uuid.uuid4().hex[:24]
            now = mono_ms()
            bitmap = "0" * count
            with self._transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO sessions(
                        session_id, object_key, chunk_size, total_size, chunk_count,
                        digest_algorithm, tags_json, state, bitmap, registered_count,
                        created_mono_ms, updated_mono_ms, completed, commit_unknown,
                        recovery_status, expected_base_version
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        session_id,
                        object_key,
                        chunk_size,
                        total_size,
                        count,
                        algorithm,
                        tags_json,
                        "created",
                        bitmap,
                        0,
                        now,
                        now,
                        0,
                        0,
                        "none",
                        expected_base,
                    ),
                )
                conn.execute(
                    """
                    INSERT INTO session_events(
                        session_id, from_state, to_state, mono_ms, bitmap,
                        reason, commit_unknown
                    ) VALUES(?,?,?,?,?,?,0)
                    """,
                    (session_id, "", "created", now, bitmap, "session created"),
                )
            session = self._get_session(session_id)
            graph_ids = self._ensure_session_graph(session)
            self.logger.event(
                "session_created",
                session_id=session_id,
                object_key=object_key,
                chunk_size=chunk_size,
                total_size=total_size,
                chunk_count=count,
                digest_algorithm=algorithm,
                tags=tags,
                graph_ids=graph_ids,
            )
            payload = self._session_payload(session)
            payload["skipped_indices"] = []
            payload["graph_ids"] = graph_ids
            return payload

    def get_session(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            session = self._get_session(str(params.get("session_id", "")))
            result = self._session_payload(session, include_chunks=True)
            result["events"] = [
                dict(row)
                for row in self.conn.execute(
                    "SELECT * FROM session_events WHERE session_id=? ORDER BY id",
                    (session["session_id"],),
                )
            ]
            result["schedules"] = self.ledger.schedule_events(session["session_id"])
            return result

    def upload_chunk(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            session_id = str(params.get("session_id", ""))
            session = self._get_session(session_id)
            if session["state"] == "committed":
                raise BusinessError(
                    E_STATE,
                    "a committed session cannot be uploaded, resumed, or aborted",
                    {"session_id": session_id, "state": session["state"]},
                )
            if session["state"] == "verifying":
                raise BusinessError(
                    E_STATE,
                    "session is in merge verification; rerun commit instead of uploading",
                    {
                        "session_id": session_id,
                        "state": session["state"],
                        "commit_unknown": bool(session["commit_unknown"]),
                    },
                )

            index = params.get("chunk_index")
            if isinstance(index, bool) or not isinstance(index, int):
                raise BusinessError(E_CHUNK, "chunk index must be an integer")
            if index < 0 or index >= session["chunk_count"]:
                raise BusinessError(
                    E_CHUNK,
                    "chunk index is outside the session range",
                    {"chunk_index": index, "chunk_count": session["chunk_count"]},
                )
            payload = b64decode(str(params.get("data_b64", "")))
            injection = params.get("inject")
            if injection == "truncate-chunk":
                truncated = payload[:-1] if payload else payload
                self.logger.event(
                    "fault_injection",
                    injection=injection,
                    result_code=E_CHUNK,
                    session_id=session_id,
                    chunk_index=index,
                    original_size=len(payload),
                    injected_size=len(truncated),
                )
                payload = truncated

            expected_size = expected_chunk_size(
                session["total_size"], session["chunk_size"], index, session["chunk_count"]
            )
            if len(payload) != expected_size:
                error = BusinessError(
                    E_CHUNK,
                    "chunk size does not match the declared deterministic split",
                    {
                        "session_id": session_id,
                        "chunk_index": index,
                        "expected_size": expected_size,
                        "actual_size": len(payload),
                        "injection": injection,
                    },
                )
                self.logger.event(
                    "business_error",
                    op="upload_chunk",
                    code=error.code,
                    message=error.message,
                    details=error.details,
                    injection=injection,
                )
                raise error

            algorithm = session["digest_algorithm"]
            digest = digest_bytes(payload, algorithm)
            provided_digest = params.get("digest")
            if provided_digest is not None and provided_digest != digest:
                error = BusinessError(
                    E_CHUNK,
                    "client chunk digest disagrees with the persisted payload",
                    {
                        "session_id": session_id,
                        "chunk_index": index,
                        "expected_digest": digest,
                        "provided_digest": provided_digest,
                    },
                )
                self.logger.event(
                    "business_error",
                    op="upload_chunk",
                    code=error.code,
                    message=error.message,
                    details=error.details,
                )
                raise error

            old_chunk = self.ledger.chunk_row(session_id, index)
            overwrite = old_chunk is not None
            old_bitmap = session["bitmap"]
            old_state = session["state"]
            rel_path = f"data/chunks/{session_id}/{index:08d}.part"
            chunk_path = self.ledger.resolve(rel_path)
            atomic_write_bytes(chunk_path, payload)
            now = mono_ms()
            registered = sorted(set(indices_from_bitmap(session["bitmap"])) | {index})
            bitmap = bitmap_from_indices(registered, session["chunk_count"])
            from_state = session["state"]
            to_state = "uploading" if from_state in {"created", "aborted"} else from_state
            reason = "first chunk registered" if from_state == "created" else (
                "resume and register chunk" if from_state == "aborted" else "chunk registered"
            )
            with self._transaction() as conn:
                conn.execute(
                    """
                    INSERT INTO chunks(
                        session_id, chunk_index, digest, size, file_path, registered_mono_ms
                    ) VALUES(?,?,?,?,?,?)
                    ON CONFLICT(session_id, chunk_index) DO UPDATE SET
                        digest=excluded.digest,
                        size=excluded.size,
                        file_path=excluded.file_path,
                        registered_mono_ms=excluded.registered_mono_ms
                    """,
                    (session_id, index, digest, len(payload), rel_path, now),
                )
                conn.execute(
                    """
                    UPDATE sessions
                       SET state=?, bitmap=?, registered_count=?, updated_mono_ms=?,
                           commit_unknown=0, recovery_status='none', abort_reason=NULL
                     WHERE session_id=?
                    """,
                    (to_state, bitmap, len(registered), now, session_id),
                )
                if from_state != to_state:
                    conn.execute(
                        """
                        INSERT INTO session_events(
                            session_id, from_state, to_state, mono_ms, bitmap,
                            reason, commit_unknown
                        ) VALUES(?,?,?,?,?,?,0)
                        """,
                        (session_id, from_state, to_state, now, bitmap, reason),
                    )
            if from_state != to_state:
                self.logger.event(
                    "session_transition",
                    session_id=session_id,
                    from_state=from_state,
                    to_state=to_state,
                    bitmap=bitmap,
                    reason=reason,
                    commit_unknown=False,
                )

            session = self._get_session(session_id)
            graph_ids = self._ensure_session_graph(session)
            trigger = self._chunk_node_id(session_id, index)
            triggers = [trigger]
            if session["bitmap"] != old_bitmap or session["state"] != old_state:
                triggers.append(graph_ids["state"])
            invalidation = self.graph.invalidate(
                triggers,
                trigger_type="chunk_overwrite" if overwrite else "chunk_registration",
                reason=(
                    "chunk retransmitted and overwritten"
                    if overwrite
                    else "chunk registered"
                ),
                context={
                    "session_id": session_id,
                    "chunk_index": index,
                    "old_digest": old_chunk["digest"] if old_chunk else None,
                    "new_digest": digest,
                    "overwrite": overwrite,
                },
            )
            if session["bitmap"] != old_bitmap or session["state"] != old_state:
                self.graph.refresh_node(
                    graph_ids["state"],
                    self._session_state_digest(session),
                    payload={
                        "session_id": session_id,
                        "state": session["state"],
                        "bitmap": session["bitmap"],
                        "commit_unknown": bool(session["commit_unknown"]),
                    },
                )
            self.logger.event(
                "chunk_registered",
                session_id=session_id,
                chunk_index=index,
                digest=digest,
                size=len(payload),
                overwrite=overwrite,
                bitmap=bitmap,
                invalidation_event=invalidation["event_id"],
                affected_count=invalidation["affected_count"],
                injection=injection,
            )
            return {
                "chunk": {
                    "session_id": session_id,
                    "chunk_index": index,
                    "digest": digest,
                    "size": len(payload),
                    "file_path": rel_path,
                    "overwrite": overwrite,
                },
                "session": self._session_payload(session),
                "invalidation": invalidation,
            }

    def resume_session(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            session_id = str(params.get("session_id", ""))
            session = self._get_session(session_id)
            if session["state"] == "committed":
                raise BusinessError(
                    E_STATE,
                    "a committed session cannot be resumed",
                    {"session_id": session_id, "state": session["state"]},
                )
            if session["state"] == "verifying":
                raise BusinessError(
                    E_STATE,
                    "an unknown merge must be rerun with the commit operation",
                    {
                        "session_id": session_id,
                        "state": session["state"],
                        "commit_unknown": bool(session["commit_unknown"]),
                    },
                )
            transition = None
            if session["state"] == "aborted":
                with self._transaction() as conn:
                    now = mono_ms()
                    conn.execute(
                        """
                        UPDATE sessions SET state='uploading', updated_mono_ms=?
                         WHERE session_id=?
                        """,
                        (now, session_id),
                    )
                    conn.execute(
                        """
                        INSERT INTO session_events(
                            session_id, from_state, to_state, mono_ms, bitmap,
                            reason, commit_unknown
                        ) VALUES(?,?,?,?,?,?,0)
                        """,
                        (
                            session_id,
                            "aborted",
                            "uploading",
                            now,
                            session["bitmap"],
                            "resume aborted session from bitmap",
                        ),
                    )
                self.logger.event(
                    "session_transition",
                    session_id=session_id,
                    from_state="aborted",
                    to_state="uploading",
                    bitmap=session["bitmap"],
                    reason="resume aborted session from bitmap",
                    commit_unknown=False,
                )
                session = self._get_session(session_id)
                transition = {"from_state": "aborted", "to_state": "uploading"}
            graph_ids = self._ensure_session_graph(session)
            if transition is not None:
                invalidation = self.graph.invalidate(
                    [graph_ids["state"]],
                    trigger_type="session_resumed",
                    reason="aborted session resumed; staged state caches invalidated",
                    context={"session_id": session_id, "bitmap": session["bitmap"]},
                )
                self.graph.refresh_node(
                    graph_ids["state"],
                    self._session_state_digest(session),
                    payload={
                        "session_id": session_id,
                        "state": session["state"],
                        "bitmap": session["bitmap"],
                        "commit_unknown": False,
                    },
                )
            else:
                invalidation = None
            payload = self._session_payload(session)
            payload["skipped_indices"] = indices_from_bitmap(session["bitmap"])
            payload["missing_indices"] = missing_indices(session["bitmap"])
            payload["transition"] = transition
            payload["invalidation"] = invalidation
            self.logger.event(
                "session_resumed",
                session_id=session_id,
                state=session["state"],
                skipped_indices=payload["skipped_indices"],
                missing_indices=payload["missing_indices"],
            )
            return payload

    def abort_session(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            session_id = str(params.get("session_id", ""))
            session = self._get_session(session_id)
            if session["state"] == "committed":
                raise BusinessError(
                    E_STATE,
                    "a committed session cannot be aborted",
                    {"session_id": session_id, "state": session["state"]},
                )
            if session["state"] == "verifying":
                raise BusinessError(
                    E_STATE,
                    "a merge in progress cannot be aborted; restart detection requires rerun",
                    {"session_id": session_id, "state": session["state"]},
                )
            if session["state"] == "aborted":
                return {
                    "session": self._session_payload(session),
                    "invalidation": None,
                    "idempotent": True,
                    "chunks_retained": len(indices_from_bitmap(session["bitmap"])),
                }
            reason = str(params.get("reason", "operator aborted session"))
            with self._transaction() as conn:
                now = mono_ms()
                conn.execute(
                    """
                    UPDATE sessions SET state='aborted', abort_reason=?, updated_mono_ms=?
                     WHERE session_id=?
                    """,
                    (reason, now, session_id),
                )
                conn.execute(
                    """
                    INSERT INTO session_events(
                        session_id, from_state, to_state, mono_ms, bitmap,
                        reason, commit_unknown
                    ) VALUES(?,?,?,?,?,?,0)
                    """,
                    (session_id, session["state"], "aborted", now, session["bitmap"], reason),
                )
            self.logger.event(
                "session_transition",
                session_id=session_id,
                from_state=session["state"],
                to_state="aborted",
                bitmap=session["bitmap"],
                reason=reason,
                commit_unknown=False,
            )
            session = self._get_session(session_id)
            graph_ids = self._ensure_session_graph(session)
            invalidation = self.graph.invalidate(
                [graph_ids["state"]],
                trigger_type="session_aborted",
                reason="upload session aborted; staged metadata caches invalidated",
                context={
                    "session_id": session_id,
                    "chunks_retained": len(indices_from_bitmap(session["bitmap"])),
                },
            )
            new_digest = self._session_state_digest(session)
            self.graph.refresh_node(
                graph_ids["state"],
                new_digest,
                payload={
                    "session_id": session_id,
                    "state": session["state"],
                    "bitmap": session["bitmap"],
                    "commit_unknown": False,
                },
            )
            return {
                "session": self._session_payload(session),
                "invalidation": invalidation,
                "chunks_retained": len(indices_from_bitmap(session["bitmap"])),
                "idempotent": False,
            }

    # ------------------------------------------------------------------
    # Views and graph-derived caches
    # ------------------------------------------------------------------
    def _compute_view(self, prefix: str) -> tuple[str, dict[str, Any]]:
        rows = list(
            self.conn.execute(
                """
                SELECT object_key, version, metadata_digest, chunk_set_digest
                  FROM object_versions
                 WHERE status='active'
                 ORDER BY object_key
                """
            )
        )
        entries = [
            {
                "object_key": row["object_key"],
                "version": row["version"],
                "metadata_digest": row["metadata_digest"],
                "chunk_set_digest": row["chunk_set_digest"],
            }
            for row in rows
            if view_prefix_matches(row["object_key"], prefix)
        ]
        payload = {
            "schema": "prefix-count-view/v1",
            "prefix": prefix,
            "count": len(entries),
            "objects": entries,
        }
        return digest_json(payload), payload

    def _ensure_view_dependencies(self, prefix: str, view_node_id: str) -> None:
        keys = [
            row["object_key"]
            for row in self.conn.execute(
                "SELECT DISTINCT object_key FROM object_versions ORDER BY object_key"
            )
            if view_prefix_matches(row["object_key"], prefix)
        ]
        for object_key in keys:
            key_node = self._ensure_key_node(object_key)
            self.graph.add_edge(key_node["node_id"], view_node_id)

    def _refresh_view(self, prefix: str) -> dict[str, Any]:
        digest, payload = self._compute_view(prefix)
        node_id = self._view_node_id(prefix)
        if self.graph.node(node_id) is None:
            node = self.graph.ensure_node(
                node_id, "prefix_view", prefix, digest, payload
            )
            node = {**node, "changed": True}
        else:
            node = self.graph.refresh_node(node_id, digest, payload=payload)
        self._ensure_view_dependencies(prefix, node_id)
        return {
            "prefix": prefix,
            "node_id": node_id,
            "count": payload["count"],
            "digest": digest,
            "entry_version": node["entry_version"],
            "changed": bool(node.get("changed", True)),
            "objects": payload["objects"],
        }

    def _capture_views(self, prefixes: list[str]) -> list[dict[str, Any]]:
        captures: list[dict[str, Any]] = []
        for prefix in prefixes:
            node_id = self._view_node_id(prefix)
            node = self.graph.node(node_id)
            captures.append(
                {
                    "prefix": prefix,
                    "node_id": node_id,
                    "exists": node is not None,
                    "entry_version": node["entry_version"] if node else 0,
                    "digest": node["cached_digest"] if node else None,
                    "valid": bool(node["valid"]) if node else False,
                }
            )
        return captures

    def refresh_views_for_key(self, object_key: str) -> dict[str, Any]:
        prefixes = prefixes_for_key(object_key)
        before = self._capture_views(prefixes)
        after = [self._refresh_view(prefix) for prefix in prefixes]
        return {
            "object_key": object_key,
            "views_before": before,
            "views_after": after,
        }

    def list_views(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            params = params or {}
            prefix = params.get("prefix")
            if prefix is not None:
                if not isinstance(prefix, str) or not prefix:
                    raise BusinessError(E_INPUT, "prefix must be a non-empty string")
                if prefix != "*" and (prefix.startswith("/") or prefix.endswith("/")):
                    raise BusinessError(E_INPUT, "prefix must not start or end with '/'")
                return {"view": self._refresh_view(prefix), "prefix": prefix}

            prefixes = [
                row["ref_id"]
                for row in self.conn.execute(
                    "SELECT ref_id FROM graph_nodes WHERE kind='prefix_view' ORDER BY ref_id"
                )
            ]
            views = [self._refresh_view(item) for item in prefixes]
            views.sort(key=lambda item: (item["prefix"] != "*", item["prefix"]))
            return {"views": views, "count": len(views)}

    # ------------------------------------------------------------------
    # Atomic commit and version replacement
    # ------------------------------------------------------------------
    def _commit_failure(
        self,
        session: sqlite3.Row,
        error: BusinessError,
        *,
        reason: str,
    ) -> None:
        self._transition(
            session["session_id"],
            "verifying",
            "uploading",
            bitmap=session["bitmap"],
            reason=reason,
            commit_unknown=False,
            extra_updates={"recovery_status": "none"},
        )
        self.logger.event(
            "business_error",
            op="commit_session",
            code=error.code,
            message=error.message,
            details=error.details,
            session_id=session["session_id"],
            object_key=session["object_key"],
        )

    def _verify_chunks(self, session: sqlite3.Row) -> dict[str, Any]:
        session_id = session["session_id"]
        chunks = self.ledger.chunk_rows(session_id)
        count = session["chunk_count"]
        expected_indices = list(range(count))
        if (
            session["registered_count"] != count
            or session["bitmap"] != "1" * count
            or len(chunks) != count
            or [row["chunk_index"] for row in chunks] != expected_indices
        ):
            raise BusinessError(
                E_INTEGRITY,
                "session bitmap and chunk registry are not complete",
                {
                    "session_id": session_id,
                    "expected_chunk_count": count,
                    "registered_count": session["registered_count"],
                    "bitmap": session["bitmap"],
                    "registry_indices": [row["chunk_index"] for row in chunks],
                },
            )

        algorithm = session["digest_algorithm"]
        content_hasher = hashlib.new(algorithm)
        for row in chunks:
            index = row["chunk_index"]
            expected_size = expected_chunk_size(
                session["total_size"], session["chunk_size"], index, count
            )
            path = self.ledger.resolve(row["file_path"])
            if not path.is_file():
                raise BusinessError(
                    E_INTEGRITY,
                    "persisted chunk file is missing",
                    {
                        "session_id": session_id,
                        "chunk_index": index,
                        "path": row["file_path"],
                    },
                )
            actual_size = path.stat().st_size
            if row["size"] != expected_size or actual_size != expected_size:
                raise BusinessError(
                    E_INTEGRITY,
                    "chunk size disagrees with data or ledger",
                    {
                        "session_id": session_id,
                        "chunk_index": index,
                        "expected_size": expected_size,
                        "ledger_size": row["size"],
                        "file_size": actual_size,
                    },
                )
            actual_digest = digest_file(path, algorithm)
            if actual_digest != row["digest"]:
                raise BusinessError(
                    E_INTEGRITY,
                    "persisted chunk digest does not match the registration",
                    {
                        "session_id": session_id,
                        "chunk_index": index,
                        "registered_digest": row["digest"],
                        "actual_digest": actual_digest,
                    },
                )
            with path.open("rb") as stream:
                while True:
                    block = stream.read(1024 * 1024)
                    if not block:
                        break
                    content_hasher.update(block)
        content_digest = content_hasher.hexdigest()
        manifest_digest = self._manifest_digest(session, chunks)
        chunk_set_digest = self._chunk_set_digest(session, chunks)
        return {
            "chunks": chunks,
            "content_digest": content_digest,
            "manifest_digest": manifest_digest,
            "chunk_set_digest": chunk_set_digest,
        }

    def _corrupt_registered_digest(self, session: sqlite3.Row) -> dict[str, Any] | None:
        chunks = self.ledger.chunk_rows(session["session_id"])
        if not chunks:
            return None
        chunk = chunks[0]
        self._ensure_session_graph(session)
        replacement = "0" * len(chunk["digest"])
        self.conn.execute(
            "UPDATE chunks SET digest=? WHERE session_id=? AND chunk_index=?",
            (replacement, session["session_id"], chunk["chunk_index"]),
        )
        invalidation = self.graph.invalidate(
            [self._chunk_node_id(session["session_id"], chunk["chunk_index"])],
            trigger_type="fault_injection",
            reason="registered chunk digest corrupted by explicit metadata injection",
            context={
                "injection": "corrupt-metadata",
                "session_id": session["session_id"],
                "chunk_index": chunk["chunk_index"],
                "old_digest": chunk["digest"],
                "new_digest": replacement,
            },
        )
        self.logger.event(
            "fault_injection",
            injection="corrupt-metadata",
            result_code=E_INTEGRITY,
            session_id=session["session_id"],
            chunk_index=chunk["chunk_index"],
            old_digest=chunk["digest"],
            new_digest=replacement,
            invalidation_event=invalidation["event_id"],
        )
        return {
            "injection": "corrupt-metadata",
            "chunk_index": chunk["chunk_index"],
            "old_digest": chunk["digest"],
            "new_digest": replacement,
        }

    def _write_object_temp(self, session: sqlite3.Row, chunks: list[sqlite3.Row]) -> tuple[Path, str]:
        temp_path = self.ledger.temp_dir / f"merge-{session['session_id']}-{uuid.uuid4().hex}.tmp"
        hasher = hashlib.new(session["digest_algorithm"])
        try:
            with temp_path.open("wb") as output:
                for chunk in chunks:
                    payload = self.ledger.resolve(chunk["file_path"]).read_bytes()
                    if digest_bytes(payload, session["digest_algorithm"]) != chunk["digest"]:
                        raise BusinessError(
                            E_INTEGRITY,
                            "chunk changed during merge",
                            {
                                "session_id": session["session_id"],
                                "chunk_index": chunk["chunk_index"],
                            },
                        )
                    output.write(payload)
                    hasher.update(payload)
                output.flush()
                os.fsync(output.fileno())
        except Exception:
            temp_path.unlink(missing_ok=True)
            raise
        return temp_path, hasher.hexdigest()

    def _version_change_fields(
        self,
        previous: sqlite3.Row | None,
        current: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if previous is None:
            return []
        prior = self._version_payload(previous)
        assert prior is not None
        fields = [
            "session_id",
            "chunk_size",
            "chunk_count",
            "digest_algorithm",
            "tags",
            "content_digest",
            "manifest_digest",
            "chunk_set_digest",
            "metadata_digest",
            "object_path",
            "status",
            "visible",
            "version",
        ]
        changes: list[dict[str, Any]] = []
        for field in sorted(fields):
            before = prior.get(field)
            after = current.get(field)
            if canonical_text(before) != canonical_text(after):
                changes.append({"field": field, "before": before, "after": after})
        return changes

    def commit_session(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            session_id = str(params.get("session_id", ""))
            session = self._get_session(session_id)
            if session["state"] == "committed" or bool(session["completed"]):
                raise BusinessError(
                    E_STATE,
                    "a committed session cannot be committed again",
                    {"session_id": session_id, "state": session["state"]},
                )
            if session["state"] == "aborted":
                raise BusinessError(
                    E_STATE,
                    "resume an aborted session before committing",
                    {"session_id": session_id, "state": session["state"]},
                )
            if session["state"] == "verifying" and not session["commit_unknown"]:
                raise BusinessError(
                    E_STATE,
                    "session is already in merge verification",
                    {"session_id": session_id, "state": session["state"]},
                )
            if session["state"] not in {"created", "uploading", "verifying"}:
                raise BusinessError(
                    E_STATE,
                    "session state cannot enter commit verification",
                    {"session_id": session_id, "state": session["state"]},
                )

            retry_unknown = session["state"] == "verifying" and bool(session["commit_unknown"])
            if retry_unknown:
                with self._transaction() as conn:
                    now = mono_ms()
                    conn.execute(
                        """
                        UPDATE sessions
                           SET commit_unknown=0, recovery_status='rerunning',
                               updated_mono_ms=?
                         WHERE session_id=?
                        """,
                        (now, session_id),
                    )
                    conn.execute(
                        """
                        INSERT INTO session_events(
                            session_id, from_state, to_state, mono_ms, bitmap,
                            reason, commit_unknown
                        ) VALUES(?,?,?,?,?,?,0)
                        """,
                        (
                            session_id,
                            "verifying",
                            "verifying",
                            now,
                            session["bitmap"],
                            "rerun merge after unknown forced-termination outcome",
                        ),
                    )
                self.logger.event(
                    "merge_rerun",
                    session_id=session_id,
                    object_key=session["object_key"],
                )
            else:
                self._transition(
                    session_id,
                    session["state"],
                    "verifying",
                    bitmap=session["bitmap"],
                    reason="begin deterministic verification before merge",
                    commit_unknown=False,
                    extra_updates={"recovery_status": "running"},
                )
            session = self._get_session(session_id)

            injection = params.get("inject")
            injection_result = None
            if injection == "corrupt-metadata":
                injection_result = self._corrupt_registered_digest(session)
                session = self._get_session(session_id)

            try:
                verified = self._verify_chunks(session)
            except BusinessError as error:
                if injection_result is not None:
                    error.details["injection"] = injection_result
                self._commit_failure(session, error, reason="verification failed; session retained for retry")
                raise

            active = self.ledger.active_version_row(session["object_key"])
            current_version = active["version"] if active else 0
            expected_base = params.get("expected_base_version", session["expected_base_version"])
            if params.get("inject") == "version-conflict":
                expected_base = current_version - 1 if current_version > 0 else 1
            if expected_base is not None:
                try:
                    expected_base = int(expected_base)
                except (TypeError, ValueError) as exc:
                    error = BusinessError(E_VERSION, "expected base version is not an integer")
                    self._commit_failure(session, error, reason="invalid version expectation")
                    raise error from exc
                if expected_base != current_version:
                    error = BusinessError(
                        E_VERSION,
                        "object metadata version changed before commit",
                        {
                            "object_key": session["object_key"],
                            "expected_base_version": expected_base,
                            "actual_base_version": current_version,
                            "session_id": session_id,
                            "injection": params.get("inject"),
                        },
                    )
                    self._commit_failure(session, error, reason="metadata version conflict")
                    raise error

            new_version_number = self.ledger.next_object_version(session["object_key"])
            object_hash = hashlib.sha256(session["object_key"].encode("utf-8")).hexdigest()[:24]
            object_rel = (
                f"data/objects/{object_hash}/v{new_version_number}.obj"
            )
            object_path = self.ledger.resolve(object_rel)
            object_path.parent.mkdir(parents=True, exist_ok=True)
            temp_path: Path | None = None
            try:
                temp_path, actual_content_digest = self._write_object_temp(
                    session, verified["chunks"]
                )
                if actual_content_digest != verified["content_digest"]:
                    raise BusinessError(
                        E_INTEGRITY,
                        "object content digest changed during merge",
                        {
                            "session_id": session_id,
                            "expected_digest": verified["content_digest"],
                            "actual_digest": actual_content_digest,
                        },
                    )
                if self.merge_pause_ms > 0:
                    self.logger.event(
                        "merge_pause_for_forced_termination_test",
                        session_id=session_id,
                        pause_ms=self.merge_pause_ms,
                    )
                    time.sleep(self.merge_pause_ms / 1000.0)

                metadata_fields = self._metadata_fields_from_session(
                    session,
                    version=new_version_number,
                    content_digest=verified["content_digest"],
                    manifest_digest=verified["manifest_digest"],
                    chunk_set_digest=verified["chunk_set_digest"],
                )
                metadata_digest = digest_json(metadata_fields, session["digest_algorithm"])
                created_mono_ms = mono_ms()

                if active is not None:
                    self._ensure_version_node(active)
                before_views = self._capture_views(prefixes_for_key(session["object_key"]))
                prior_payload = self._version_payload(active)
                if prior_payload is not None:
                    prior_payload["status"] = "expired"
                    prior_payload["visible"] = False
                    prior_payload["replaced_mono_ms"] = created_mono_ms
                current_record = {
                    "object_key": session["object_key"],
                    "version": new_version_number,
                    "session_id": session_id,
                    "metadata_digest": metadata_digest,
                    "manifest_digest": verified["manifest_digest"],
                    "content_digest": verified["content_digest"],
                    "chunk_size": session["chunk_size"],
                    "chunk_count": session["chunk_count"],
                    "chunk_set_digest": verified["chunk_set_digest"],
                    "digest_algorithm": session["digest_algorithm"],
                    "tags": json.loads(session["tags_json"]),
                    "status": "active",
                    "visible": True,
                    "object_path": object_rel,
                }
                field_changes = self._version_change_fields(active, current_record)
                with self._transaction() as conn:
                    if active is not None:
                        conn.execute(
                            """
                            UPDATE object_versions
                               SET status='expired', visible=0, replaced_mono_ms=?
                             WHERE object_key=? AND version=?
                            """,
                            (created_mono_ms, session["object_key"], active["version"]),
                        )
                    conn.execute(
                        """
                        INSERT INTO object_versions(
                            object_key, version, session_id, metadata_digest,
                            manifest_digest, content_digest, chunk_size, chunk_count,
                            chunk_set_digest, digest_algorithm, tags_json, status,
                            visible, object_path, created_mono_ms, replaced_mono_ms
                        ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,NULL)
                        """,
                        (
                            session["object_key"],
                            new_version_number,
                            session_id,
                            metadata_digest,
                            verified["manifest_digest"],
                            verified["content_digest"],
                            session["chunk_size"],
                            session["chunk_count"],
                            verified["chunk_set_digest"],
                            session["digest_algorithm"],
                            session["tags_json"],
                            "active",
                            1,
                            object_rel,
                            created_mono_ms,
                        ),
                    )
                    conn.execute(
                        """
                        UPDATE sessions
                           SET state='committed', completed=1, version=?,
                               content_digest=?, object_path=?, updated_mono_ms=?,
                               commit_unknown=0, recovery_status='none'
                         WHERE session_id=?
                        """,
                        (
                            new_version_number,
                            verified["content_digest"],
                            object_rel,
                            created_mono_ms,
                            session_id,
                        ),
                    )
                    conn.execute(
                        """
                        INSERT INTO session_events(
                            session_id, from_state, to_state, mono_ms, bitmap,
                            reason, commit_unknown
                        ) VALUES(?,?,?,?,?,?,0)
                        """,
                        (
                            session_id,
                            "verifying",
                            "committed",
                            created_mono_ms,
                            session["bitmap"],
                            "all chunks verified; metadata committed atomically",
                        ),
                    )
                    os.replace(temp_path, object_path)
                    temp_path = None
                self.logger.event(
                    "session_transition",
                    session_id=session_id,
                    from_state="verifying",
                    to_state="committed",
                    bitmap=session["bitmap"],
                    reason="all chunks verified; metadata committed atomically",
                    commit_unknown=False,
                )
            except BusinessError as error:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
                self._commit_failure(session, error, reason="merge failed; session retained for retry")
                raise
            except Exception:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)
                raise

            # Graph work is deliberately after the atomic metadata commit. The graph is
            # a cache; every entry is derived and can be rebuilt from SQLite.
            if active is not None:
                old_metadata_node = self._metadata_node_id(
                    session["object_key"], active["version"]
                )
                invalidation = self.graph.invalidate(
                    [old_metadata_node],
                    trigger_type="object_version_replaced",
                    reason="new object version replaces the active metadata view",
                    context={
                        "object_key": session["object_key"],
                        "old_version": active["version"],
                        "new_version": new_version_number,
                        "old_metadata_digest": active["metadata_digest"],
                        "new_metadata_digest": metadata_digest,
                    },
                )
            else:
                invalidation = self.graph.invalidate(
                    [self._key_node_id(session["object_key"])],
                    trigger_type="object_created",
                    reason="first active object version changes key and prefix views",
                    context={
                        "object_key": session["object_key"],
                        "new_version": new_version_number,
                        "new_metadata_digest": metadata_digest,
                    },
                )

            new_version_row = self.ledger.object_version_row(
                session["object_key"], new_version_number
            )
            assert new_version_row is not None
            new_metadata_node = self._ensure_version_node(new_version_row)
            key_digest, key_payload = self._key_digest(session["object_key"])
            key_node_id = self._key_node_id(session["object_key"])
            key_node = self.graph.refresh_node(
                key_node_id, key_digest, payload=key_payload
            )
            if active is not None:
                self.graph.remove_edge(
                    self._metadata_node_id(session["object_key"], active["version"]),
                    key_node_id,
                )
            self.graph.add_edge(new_metadata_node["node_id"], key_node_id)
            view_refresh = self.refresh_views_for_key(session["object_key"])
            new_payload = self._version_payload(new_version_row)
            assert new_payload is not None
            self.logger.event(
                "object_version_committed",
                object_key=session["object_key"],
                version=new_version_number,
                previous_version=active["version"] if active else None,
                metadata_digest=metadata_digest,
                manifest_digest=verified["manifest_digest"],
                content_digest=verified["content_digest"],
                replacement="whole",
                changed_fields=[entry["field"] for entry in field_changes],
                invalidation_event=invalidation["event_id"],
            )
            return {
                "session": self._session_payload(self._get_session(session_id)),
                "version": new_payload,
                "previous_version": prior_payload,
                "replacement": "whole",
                "field_changes": field_changes,
                "summary": {
                    "metadata_digest_before": active["metadata_digest"] if active else None,
                    "metadata_digest_after": metadata_digest,
                    "manifest_digest_before": active["manifest_digest"] if active else None,
                    "manifest_digest_after": verified["manifest_digest"],
                    "content_digest_before": active["content_digest"] if active else None,
                    "content_digest_after": verified["content_digest"],
                    "old_version_status": "expired" if active else None,
                    "view_entries_before": before_views,
                    "view_entries_after": view_refresh["views_after"],
                    "key_node": key_node,
                    "injection": injection_result,
                },
                "verified_chunks": len(verified["chunks"]),
                "object_path": object_rel,
                "invalidation": invalidation,
            }

    def delete_object(self, params: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            object_key = validate_object_key(params.get("object_key"))
            active = self.ledger.active_version_row(object_key)
            if active is None:
                raise BusinessError(
                    E_NOT_FOUND,
                    "object has no active metadata version",
                    {"object_key": object_key},
                )
            self._ensure_key_node(object_key)
            self._ensure_version_node(active)
            prefixes = prefixes_for_key(object_key)
            views_before = self._capture_views(prefixes)
            now = mono_ms()
            with self._transaction() as conn:
                conn.execute(
                    """
                    UPDATE object_versions
                       SET status='deleted', visible=0, replaced_mono_ms=?
                     WHERE object_key=? AND version=? AND status='active'
                    """,
                    (now, object_key, active["version"]),
                )
            metadata_node = self._metadata_node_id(object_key, active["version"])
            invalidation = self.graph.invalidate(
                [metadata_node],
                trigger_type="object_deleted",
                reason="active object deleted; metadata, key and prefix views invalidated",
                context={
                    "object_key": object_key,
                    "deleted_version": active["version"],
                    "metadata_digest": active["metadata_digest"],
                },
            )
            key_digest, key_payload = self._key_digest(object_key)
            key_node = self.graph.refresh_node(
                self._key_node_id(object_key), key_digest, payload=key_payload
            )
            view_refresh = self.refresh_views_for_key(object_key)
            self.logger.event(
                "object_deleted",
                object_key=object_key,
                version=active["version"],
                invalidation_event=invalidation["event_id"],
                affected_count=invalidation["affected_count"],
            )
            return {
                "object_key": object_key,
                "deleted_version": active["version"],
                "metadata_digest": active["metadata_digest"],
                "key_node": key_node,
                "views_before": views_before,
                "views_after": view_refresh["views_after"],
                "invalidation": invalidation,
            }

    # ------------------------------------------------------------------
    # Audit, snapshot and diagnostics
    # ------------------------------------------------------------------
    def _save_audit_run(
        self,
        *,
        passed: bool,
        summary: dict[str, Any],
        conclusion: dict[str, Any],
    ) -> str:
        run_id = "audit-" + uuid.uuid4().hex[:20]
        now = mono_ms()
        with self._transaction() as conn:
            conn.execute(
                """
                INSERT INTO audit_runs(run_id, mono_ms, passed, summary_json, conclusion_json)
                VALUES(?,?,?,?,?)
                """,
                (
                    run_id,
                    now,
                    int(passed),
                    json.dumps(summary, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                    json.dumps(conclusion, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                ),
            )
        conclusion["run_id"] = run_id
        return run_id

    def _audit_fail(
        self,
        *,
        object_key: str,
        chunk_index: int | None,
        kind: str,
        message: str,
        details: dict[str, Any],
        checked_objects: list[str],
    ) -> None:
        deviation = {
            "object_key": object_key,
            "chunk_index": chunk_index,
            "kind": kind,
            "message": message,
            **details,
        }
        summary = {
            "passed": False,
            "objects_checked": checked_objects,
            "first_deviation": deviation,
        }
        conclusion = {
            "passed": False,
            "verdict": "E_AUDIT",
            "first_deviation": deviation,
            "checked_object_count": len(checked_objects),
        }
        run_id = self._save_audit_run(passed=False, summary=summary, conclusion=conclusion)
        error = BusinessError(
            E_AUDIT,
            "consistency audit failed",
            {"run_id": run_id, "first_deviation": deviation},
        )
        self.logger.event(
            "business_error",
            op="audit",
            code=E_AUDIT,
            message=error.message,
            details=error.details,
        )
        raise error

    def audit(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            active_versions = list(
                self.conn.execute(
                    "SELECT * FROM object_versions WHERE status='active' ORDER BY object_key"
                )
            )
            checked_objects: list[str] = []
            per_object: list[dict[str, Any]] = []
            checked_chunks = 0
            try:
                for version in active_versions:
                    object_key = version["object_key"]
                    checked_objects.append(object_key)
                    session = self.ledger.session_row(version["session_id"])
                    if session is None:
                        self._audit_fail(
                            object_key=object_key,
                            chunk_index=None,
                            kind="session_missing",
                            message="metadata references a missing upload session",
                            details={"session_id": version["session_id"]},
                            checked_objects=checked_objects,
                        )
                    assert session is not None
                    if session["state"] != "committed" or not session["completed"]:
                        self._audit_fail(
                            object_key=object_key,
                            chunk_index=None,
                            kind="completion_marker",
                            message="active metadata is not backed by a completed session",
                            details={
                                "session_id": session["session_id"],
                                "state": session["state"],
                                "completed": bool(session["completed"]),
                            },
                            checked_objects=checked_objects,
                        )
                    if (
                        session["version"] != version["version"]
                        or session["content_digest"] != version["content_digest"]
                        or session["object_path"] != version["object_path"]
                    ):
                        self._audit_fail(
                            object_key=object_key,
                            chunk_index=None,
                            kind="session_version_link",
                            message="session completion marker disagrees with object metadata",
                            details={
                                "session_version": session["version"],
                                "metadata_version": version["version"],
                            },
                            checked_objects=checked_objects,
                        )
                    if (
                        version["chunk_size"] != session["chunk_size"]
                        or version["chunk_count"] != session["chunk_count"]
                        or version["digest_algorithm"] != session["digest_algorithm"]
                    ):
                        self._audit_fail(
                            object_key=object_key,
                            chunk_index=None,
                            kind="manifest_shape",
                            message="object metadata split shape differs from session",
                            details={
                                "metadata": {
                                    "chunk_size": version["chunk_size"],
                                    "chunk_count": version["chunk_count"],
                                    "digest_algorithm": version["digest_algorithm"],
                                },
                                "session": {
                                    "chunk_size": session["chunk_size"],
                                    "chunk_count": session["chunk_count"],
                                    "digest_algorithm": session["digest_algorithm"],
                                },
                            },
                            checked_objects=checked_objects,
                        )

                    chunks = self.ledger.chunk_rows(session["session_id"])
                    expected_indices = list(range(session["chunk_count"]))
                    actual_indices = [row["chunk_index"] for row in chunks]
                    if actual_indices != expected_indices:
                        missing = sorted(set(expected_indices) - set(actual_indices))
                        extra = sorted(set(actual_indices) - set(expected_indices))
                        first = (missing + extra + [None])[0]
                        self._audit_fail(
                            object_key=object_key,
                            chunk_index=first,
                            kind="chunk_registry",
                            message="registered chunk sequence is not contiguous",
                            details={"missing": missing, "extra": extra},
                            checked_objects=checked_objects,
                        )

                    chunk_dir = self.ledger.chunks_dir / session["session_id"]
                    expected_files = {
                        f"{index:08d}.part": index for index in expected_indices
                    }
                    actual_files = {
                        path.name: path for path in chunk_dir.iterdir()
                    } if chunk_dir.is_dir() else {}
                    if set(actual_files) != set(expected_files):
                        missing_names = sorted(set(expected_files) - set(actual_files))
                        extra_names = sorted(set(actual_files) - set(expected_files))
                        candidate_indices: list[int] = []
                        for name in missing_names:
                            candidate_indices.append(expected_files[name])
                        for name in extra_names:
                            try:
                                candidate_indices.append(int(name.split(".")[0]))
                            except ValueError:
                                candidate_indices.append(-1)
                        first = sorted(candidate_indices)[0] if candidate_indices else None
                        self._audit_fail(
                            object_key=object_key,
                            chunk_index=first,
                            kind="chunk_file_set",
                            message="disk chunk file set differs from the current metadata reference",
                            details={
                                "missing_files": missing_names,
                                "extra_files": extra_names,
                            },
                            checked_objects=checked_objects,
                        )

                    for chunk in chunks:
                        index = chunk["chunk_index"]
                        chunk_path = self.ledger.resolve(chunk["file_path"])
                        expected_size = expected_chunk_size(
                            session["total_size"],
                            session["chunk_size"],
                            index,
                            session["chunk_count"],
                        )
                        actual_size = chunk_path.stat().st_size
                        if chunk["size"] != expected_size or actual_size != expected_size:
                            self._audit_fail(
                                object_key=object_key,
                                chunk_index=index,
                                kind="chunk_size",
                                message="chunk file size is inconsistent",
                                details={
                                    "expected_size": expected_size,
                                    "ledger_size": chunk["size"],
                                    "file_size": actual_size,
                                },
                                checked_objects=checked_objects,
                            )
                        actual_digest = digest_file(
                            chunk_path, session["digest_algorithm"]
                        )
                        if actual_digest != chunk["digest"]:
                            self._audit_fail(
                                object_key=object_key,
                                chunk_index=index,
                                kind="chunk_digest",
                                message="chunk file digest differs from registration",
                                details={
                                    "registered_digest": chunk["digest"],
                                    "actual_digest": actual_digest,
                                },
                                checked_objects=checked_objects,
                            )
                        checked_chunks += 1

                    chunk_set_digest = self._chunk_set_digest(session, chunks)
                    if chunk_set_digest != version["chunk_set_digest"]:
                        self._audit_fail(
                            object_key=object_key,
                            chunk_index=None,
                            kind="chunk_set_digest",
                            message="chunk set digest differs from object metadata",
                            details={
                                "metadata_digest": version["chunk_set_digest"],
                                "computed_digest": chunk_set_digest,
                            },
                            checked_objects=checked_objects,
                        )
                    manifest_digest = self._manifest_digest(session, chunks)
                    if manifest_digest != version["manifest_digest"]:
                        self._audit_fail(
                            object_key=object_key,
                            chunk_index=None,
                            kind="manifest_digest",
                            message="manifest digest differs from object metadata",
                            details={
                                "metadata_digest": version["manifest_digest"],
                                "computed_digest": manifest_digest,
                            },
                            checked_objects=checked_objects,
                        )
                    recomputed_metadata_digest = self._metadata_digest_from_version(version)
                    if recomputed_metadata_digest != version["metadata_digest"]:
                        self._audit_fail(
                            object_key=object_key,
                            chunk_index=None,
                            kind="metadata_digest",
                            message="metadata digest does not match the metadata fields",
                            details={
                                "registered_digest": version["metadata_digest"],
                                "computed_digest": recomputed_metadata_digest,
                            },
                            checked_objects=checked_objects,
                        )

                    object_path = self.ledger.resolve(version["object_path"])
                    if not object_path.is_file():
                        self._audit_fail(
                            object_key=object_key,
                            chunk_index=None,
                            kind="object_file_missing",
                            message="visible object data file is missing",
                            details={"object_path": version["object_path"]},
                            checked_objects=checked_objects,
                        )
                    content_digest = digest_file(
                        object_path, session["digest_algorithm"]
                    )
                    if content_digest != version["content_digest"]:
                        self._audit_fail(
                            object_key=object_key,
                            chunk_index=None,
                            kind="content_digest",
                            message="visible object content digest differs from metadata",
                            details={
                                "metadata_digest": version["content_digest"],
                                "actual_digest": content_digest,
                            },
                            checked_objects=checked_objects,
                        )
                    per_object.append(
                        {
                            "object_key": object_key,
                            "version": version["version"],
                            "status": version["status"],
                            "visible": bool(version["visible"]),
                            "chunk_count": len(chunks),
                            "chunk_set_digest": chunk_set_digest,
                            "manifest_digest": manifest_digest,
                            "metadata_digest": recomputed_metadata_digest,
                            "content_digest": content_digest,
                        }
                    )
            except BusinessError:
                raise

            conclusion = {
                "passed": True,
                "verdict": "OK",
                "first_deviation": None,
                "checked_object_count": len(per_object),
                "checked_chunk_count": checked_chunks,
            }
            summary = {
                "passed": True,
                "objects": per_object,
                "object_count": len(per_object),
                "chunk_count": checked_chunks,
                "objects_checked": checked_objects,
            }
            summary["audit_digest"] = digest_json(summary)
            conclusion["audit_digest"] = summary["audit_digest"]
            run_id = self._save_audit_run(
                passed=True, summary=summary, conclusion=conclusion
            )
            conclusion["run_id"] = run_id
            result = {
                "passed": True,
                "run_id": run_id,
                "mono_ms": mono_ms(),
                "object_count": len(per_object),
                "chunk_count": checked_chunks,
                "objects": per_object,
                "audit_digest": summary["audit_digest"],
                "conclusion": conclusion,
            }
            self.logger.event(
                "audit_passed",
                run_id=run_id,
                object_count=len(per_object),
                chunk_count=checked_chunks,
                audit_digest=summary["audit_digest"],
            )
            return result

    def _object_signatures(self) -> dict[str, Any]:
        signatures: dict[str, Any] = {}
        keys = [
            row["object_key"]
            for row in self.conn.execute(
                "SELECT DISTINCT object_key FROM object_versions ORDER BY object_key"
            )
        ]
        for object_key in keys:
            versions = list(
                self.conn.execute(
                    "SELECT * FROM object_versions WHERE object_key=? ORDER BY version",
                    (object_key,),
                )
            )
            active = next((row for row in versions if row["status"] == "active"), None)
            signatures[object_key] = {
                "active_version": active["version"] if active else 0,
                "versions": [
                    {
                        "version": row["version"],
                        "status": row["status"],
                        "metadata_digest": row["metadata_digest"],
                        "chunk_set_digest": row["chunk_set_digest"],
                    }
                    for row in versions
                ],
                "active_chunk_set_digest": active["chunk_set_digest"] if active else None,
            }
        return signatures

    def _file_manifest(self) -> list[dict[str, Any]]:
        paths: list[str] = []
        for row in self.conn.execute(
            "SELECT file_path FROM chunks ORDER BY session_id, chunk_index"
        ):
            paths.append(row["file_path"])
        for row in self.conn.execute(
            "SELECT object_path FROM object_versions ORDER BY object_key, version"
        ):
            paths.append(row["object_path"])
        result: list[dict[str, Any]] = []
        seen: set[str] = set()
        for relative in paths:
            if relative in seen:
                continue
            seen.add(relative)
            path = self.ledger.resolve(relative)
            exists = path.is_file()
            actual_digest = digest_file(path, "sha256") if exists else None
            result.append(
                {
                    "path": relative,
                    "exists": exists,
                    "size": path.stat().st_size if exists else None,
                    "digest": actual_digest,
                }
            )
        return sorted(result, key=lambda item: item["path"])

    def _signature_differences(
        self,
        snapshot_signatures: dict[str, Any],
        current_signatures: dict[str, Any],
    ) -> dict[str, Any]:
        snapshot_keys = set(snapshot_signatures)
        current_keys = set(current_signatures)
        changed: list[str] = []
        chunk_set_changes: list[str] = []
        for object_key in sorted(snapshot_keys & current_keys):
            if canonical_text(snapshot_signatures[object_key]) != canonical_text(current_signatures[object_key]):
                changed.append(object_key)
            if (
                snapshot_signatures[object_key].get("active_chunk_set_digest")
                != current_signatures[object_key].get("active_chunk_set_digest")
                or snapshot_signatures[object_key].get("versions")
                != current_signatures[object_key].get("versions")
            ):
                chunk_set_changes.append(object_key)
        return {
            "metadata_version_changes": changed,
            "chunk_set_changes": chunk_set_changes,
            "missing_current": sorted(snapshot_keys - current_keys),
            "extra_current": sorted(current_keys - snapshot_keys),
            "changed_count": len(changed)
            + len(snapshot_keys - current_keys)
            + len(current_keys - snapshot_keys),
        }

    def snapshot_save(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            params = params or {}
            path_value = params.get("path")
            if path_value:
                snapshot_path = Path(str(path_value))
                if not snapshot_path.is_absolute():
                    snapshot_path = self.ledger.root / snapshot_path
            else:
                snapshot_path = self.ledger.snapshot_path
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            state = self.ledger.all_tables_snapshot()
            signatures = self._object_signatures()
            file_manifest = self._file_manifest()
            audit = self.ledger.latest_audit()
            payload = {
                "schema": "chunkdeps-snapshot/v1",
                "created_mono_ms": mono_ms(),
                "root": str(self.ledger.root),
                "graph_version": self.graph.graph_version(),
                "state_digest": digest_json(state),
                "object_signatures": signatures,
                "file_manifest": file_manifest,
                "audit_conclusion": audit,
                "state": state,
            }
            atomic_write_json(snapshot_path, payload)
            self.logger.event(
                "snapshot_saved",
                path=str(snapshot_path),
                state_digest=payload["state_digest"],
                graph_version=payload["graph_version"],
                object_count=len(signatures),
            )
            return {
                "path": str(snapshot_path),
                "state_digest": payload["state_digest"],
                "graph_version": payload["graph_version"],
                "object_count": len(signatures),
                "session_count": len(state["sessions"]),
                "audit_conclusion": audit,
                "file_count": len(file_manifest),
            }

    def _replace_state_from_snapshot(self, state: dict[str, Any]) -> None:
        with self._transaction() as conn:
            conn.execute("DELETE FROM scheduler_events")
            conn.execute("DELETE FROM audit_runs")
            conn.execute("DELETE FROM graph_edges")
            conn.execute("DELETE FROM graph_nodes")
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM session_events")
            conn.execute("DELETE FROM sessions")
            conn.execute("DELETE FROM object_versions")

            session_columns = [
                "session_id", "object_key", "chunk_size", "total_size", "chunk_count",
                "digest_algorithm", "tags_json", "state", "bitmap", "registered_count",
                "created_mono_ms", "updated_mono_ms", "completed", "commit_unknown",
                "recovery_status", "expected_base_version", "version", "content_digest",
                "object_path", "abort_reason",
            ]
            event_columns = [
                "id", "session_id", "from_state", "to_state", "mono_ms", "bitmap",
                "reason", "commit_unknown",
            ]
            chunk_columns = [
                "session_id", "chunk_index", "digest", "size", "file_path",
                "registered_mono_ms",
            ]
            version_columns = [
                "object_key", "version", "session_id", "metadata_digest",
                "manifest_digest", "content_digest", "chunk_size", "chunk_count",
                "chunk_set_digest", "digest_algorithm", "tags_json", "status",
                "visible", "object_path", "created_mono_ms", "replaced_mono_ms",
            ]
            audit_columns = [
                "id", "run_id", "mono_ms", "passed", "summary_json", "conclusion_json",
            ]
            schedule_columns = [
                "id", "session_id", "chunk_index", "queue_ms", "start_mono_ms",
                "end_mono_ms", "global_limit", "session_limit", "injection",
                "created_mono_ms",
            ]
            node_columns = [
                "node_id", "kind", "ref_id", "entry_version", "cached_digest",
                "valid", "updated_mono_ms", "payload_json",
            ]

            def insert_rows(table: str, columns: list[str], rows: list[dict[str, Any]]) -> None:
                if not rows:
                    return
                placeholders = ",".join("?" for _ in columns)
                sql = f"INSERT INTO {table}({','.join(columns)}) VALUES({placeholders})"
                for row in rows:
                    conn.execute(sql, tuple(row.get(column) for column in columns))

            insert_rows("sessions", session_columns, state.get("sessions", []))
            insert_rows("session_events", event_columns, state.get("session_events", []))
            insert_rows("chunks", chunk_columns, state.get("chunks", []))
            insert_rows("object_versions", version_columns, state.get("object_versions", []))
            insert_rows("audit_runs", audit_columns, state.get("audit_runs", []))
            insert_rows("scheduler_events", schedule_columns, state.get("scheduler_events", []))
            insert_rows("graph_nodes", node_columns, state.get("graph_nodes", []))
            for edge in state.get("graph_edges", []):
                conn.execute(
                    "INSERT INTO graph_edges(dependency_id, dependent_id) VALUES(?,?)",
                    (edge["dependency_id"], edge["dependent_id"]),
                )
            conn.execute(
                "UPDATE graph_meta SET graph_version=? WHERE id=1",
                (int(state.get("graph_version", 1)),),
            )

    def snapshot_restore(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            params = params or {}
            path_value = params.get("path")
            if path_value:
                snapshot_path = Path(str(path_value))
                if not snapshot_path.is_absolute():
                    snapshot_path = self.ledger.root / snapshot_path
            else:
                snapshot_path = self.ledger.snapshot_path
            if not snapshot_path.is_file():
                raise BusinessError(
                    E_NOT_FOUND,
                    "snapshot file not found",
                    {"path": str(snapshot_path)},
                )
            try:
                payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise BusinessError(
                    E_INPUT,
                    "snapshot is not readable JSON",
                    {"path": str(snapshot_path)},
                ) from exc
            if payload.get("schema") != "chunkdeps-snapshot/v1":
                raise BusinessError(
                    E_INPUT,
                    "unsupported snapshot schema",
                    {"schema": payload.get("schema")},
                )
            state = payload.get("state")
            if not isinstance(state, dict):
                raise BusinessError(E_INPUT, "snapshot is missing its state payload")
            snapshot_signatures = payload.get("object_signatures", {})
            current_signatures = self._object_signatures()
            differences = self._signature_differences(
                snapshot_signatures, current_signatures
            )
            current_files = {item["path"]: item for item in self._file_manifest()}
            snapshot_files = {
                item["path"]: item for item in payload.get("file_manifest", [])
            }
            file_mismatches = []
            for path in sorted(set(snapshot_files) | set(current_files)):
                left = snapshot_files.get(path)
                right = current_files.get(path)
                if canonical_text(left) != canonical_text(right):
                    file_mismatches.append(
                        {
                            "path": path,
                            "snapshot": left,
                            "current": right,
                        }
                    )
            self._replace_state_from_snapshot(state)
            validation = self.graph.validate()
            self.logger.event(
                "snapshot_restored",
                path=str(snapshot_path),
                mode="whole-state-replacement",
                differences=differences,
                graph_validation=validation,
                file_mismatch_count=len(file_mismatches),
            )
            return {
                "path": str(snapshot_path),
                "mode": "whole-state-replacement",
                "incremental_merge": False,
                "comparison": differences,
                "metadata_version_mismatch": differences["metadata_version_changes"],
                "chunk_set_mismatch": differences["chunk_set_changes"],
                "missing_current_objects": differences["missing_current"],
                "extra_current_objects": differences["extra_current"],
                "file_mismatches": file_mismatches,
                "requires_audit": bool(file_mismatches),
                "snapshot_state_digest": payload.get("state_digest"),
                "restored_graph_version": self.graph.graph_version(),
                "graph_validation": validation,
                "snapshot_audit_conclusion": payload.get("audit_conclusion"),
                "restored_counts": {
                    "sessions": len(state.get("sessions", [])),
                    "object_versions": len(state.get("object_versions", [])),
                    "chunks": len(state.get("chunks", [])),
                    "graph_nodes": len(state.get("graph_nodes", [])),
                    "graph_edges": len(state.get("graph_edges", [])),
                },
            }

    def list_sessions(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            params = params or {}
            state = params.get("state")
            object_key = params.get("object_key")
            clauses: list[str] = []
            values: list[Any] = []
            if state:
                clauses.append("state=?")
                values.append(state)
            if object_key:
                clauses.append("object_key=?")
                values.append(object_key)
            where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = self.conn.execute(
                f"SELECT * FROM sessions{where} ORDER BY created_mono_ms, session_id",
                values,
            ).fetchall()
            return {
                "sessions": [self._session_payload(row) for row in rows],
                "count": len(rows),
            }

    def list_objects(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            params = params or {}
            object_key = params.get("object_key")
            if object_key:
                validate_object_key(object_key)
                rows = self.conn.execute(
                    "SELECT * FROM object_versions WHERE object_key=? ORDER BY version",
                    (object_key,),
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT * FROM object_versions ORDER BY object_key, version"
                ).fetchall()
            versions = [self._version_payload(row) for row in rows]
            active = [item for item in versions if item and item["status"] == "active"]
            return {
                "objects": active,
                "active_count": len(active),
                "versions": versions,
                "version_count": len(versions),
            }

    def get_graph(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            params = params or {}
            return self.graph.graph_payload(
                include_events=bool(params.get("include_events", False))
            )

    def invalidations(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            params = params or {}
            limit = int(params.get("limit", 20))
            return {"events": self.graph.latest_events(limit), "count": len(self.graph.latest_events(limit))}

    def status(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            counts = {
                "sessions": self.conn.execute("SELECT COUNT(*) AS n FROM sessions").fetchone()["n"],
                "chunks": self.conn.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"],
                "object_versions": self.conn.execute("SELECT COUNT(*) AS n FROM object_versions").fetchone()["n"],
                "active_objects": self.conn.execute("SELECT COUNT(*) AS n FROM object_versions WHERE status='active'").fetchone()["n"],
                "graph_nodes": self.conn.execute("SELECT COUNT(*) AS n FROM graph_nodes").fetchone()["n"],
                "graph_edges": self.conn.execute("SELECT COUNT(*) AS n FROM graph_edges").fetchone()["n"],
            }
            return {
                "ok": True,
                "root": str(self.ledger.root),
                "database": str(self.ledger.db_path),
                "snapshot": str(self.ledger.snapshot_path),
                "graph_version": self.graph.graph_version(),
                "graph_validation": self.graph.validate(),
                "counts": counts,
                "recovered_on_start": self.recovered_on_start,
                "recent_invalidations": self.graph.latest_events(5),
            }

    def recovery_status(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            unknown = list(
                self.conn.execute(
                    "SELECT * FROM sessions WHERE recovery_status='unknown_needs_rerun' ORDER BY session_id"
                )
            )
            committed = list(
                self.conn.execute(
                    "SELECT * FROM sessions WHERE state='committed' AND completed=1 ORDER BY session_id"
                )
            )
            return {
                "recovered_on_start": self.recovered_on_start,
                "unknown_merge_sessions": [self._session_payload(row) for row in unknown],
                "committed_sessions": [self._session_payload(row) for row in committed],
                "exit_report_available": False,
                "rule": "only completed sessions and committed metadata are restored",
            }

    def schedules(self, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with self._lock:
            params = params or {}
            session_id = params.get("session_id")
            events = self.ledger.schedule_events(str(session_id) if session_id else None)
            return {"schedules": events, "count": len(events)}
