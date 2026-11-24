"""SQLite session ledger with atomic batches, snapshot replacement and replay."""

from __future__ import annotations

import base64
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .errors import E_FIELD, E_REPLAY, E_SNAPSHOT, E_SPEC, fail
from .fragmenter import context_for_plan, fragment_message
from .samples import generate_messages
from .spec import TunnelSpec
from .util import canonical_json, monotonic_ms, sha256_hex, sha256_json, wall_time_ms
from .wire import decode_fragment, encode_fragment, with_retry_count


LEDGER_FORMAT_VERSION = 1


@dataclass
class OpenResult:
    initialized: bool
    generation: int
    replaced: bool
    replacement_diff: dict[str, Any] | None
    unknown_batches: tuple[dict[str, Any], ...]
    content_digest: str

    @property
    def can_claim_exit_report(self) -> bool:
        return not self.unknown_batches and not self.replaced

    @property
    def exit_report(self) -> str | None:
        return "available_after_replay" if self.can_claim_exit_report else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "initialized": self.initialized,
            "generation": self.generation,
            "replaced": self.replaced,
            "replacement_diff": self.replacement_diff,
            "unknown_batches": list(self.unknown_batches),
            "content_digest": self.content_digest,
            "can_claim_exit_report": self.can_claim_exit_report,
            "exit_report": self.exit_report,
        }


@dataclass
class ReplayResult:
    checked_batches: int
    checked_fragments: int
    checked_reassemblies: int
    first_mismatch: None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "checked_batches": self.checked_batches,
            "checked_fragments": self.checked_fragments,
            "checked_reassemblies": self.checked_reassemblies,
            "first_mismatch": self.first_mismatch,
        }


class SessionLedger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=10, isolation_level=None)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "SessionLedger":
        return self

    def __exit__(self, _exc_type, _exc, _tb) -> None:
        self.close()

    def _create_schema(self) -> None:
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS snapshot_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                old_generation INTEGER NOT NULL,
                new_generation INTEGER NOT NULL,
                replaced_at_ms INTEGER NOT NULL,
                old_snapshot_json TEXT NOT NULL,
                diff_json TEXT NOT NULL,
                old_content_digest TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS batches (
                generation INTEGER NOT NULL,
                batch_id TEXT NOT NULL,
                message_key TEXT NOT NULL,
                status TEXT NOT NULL,
                target_format TEXT NOT NULL,
                batch_context_json TEXT NOT NULL,
                expected_fragments INTEGER NOT NULL,
                commit_digest TEXT,
                error_json TEXT,
                updated_monotonic_ms REAL NOT NULL,
                PRIMARY KEY (generation, batch_id)
            );
            CREATE TABLE IF NOT EXISTS planned_fragments (
                generation INTEGER NOT NULL,
                batch_id TEXT NOT NULL,
                fragment_index INTEGER NOT NULL,
                message_key TEXT NOT NULL,
                status TEXT NOT NULL,
                PRIMARY KEY (generation, batch_id, fragment_index)
            );
            CREATE TABLE IF NOT EXISTS fragments (
                generation INTEGER NOT NULL,
                batch_id TEXT NOT NULL,
                fragment_index INTEGER NOT NULL,
                message_key TEXT NOT NULL,
                fragment_offset INTEGER NOT NULL,
                fragment_length INTEGER NOT NULL,
                total_fragments INTEGER NOT NULL,
                message_id TEXT NOT NULL,
                tunnel_id TEXT NOT NULL,
                target_format TEXT NOT NULL,
                retry_count INTEGER NOT NULL,
                fragment_version INTEGER NOT NULL,
                wire_b64 TEXT NOT NULL,
                wire_sha256 TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                header_json TEXT NOT NULL,
                PRIMARY KEY (generation, batch_id, fragment_index)
            );
            CREATE TABLE IF NOT EXISTS scheduling (
                generation INTEGER NOT NULL,
                batch_id TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                event_json TEXT NOT NULL,
                PRIMARY KEY (generation, batch_id, ordinal)
            );
            CREATE TABLE IF NOT EXISTS reassembly (
                generation INTEGER NOT NULL,
                batch_id TEXT NOT NULL,
                message_key TEXT NOT NULL,
                payload_sha256 TEXT NOT NULL,
                payload_b64 TEXT NOT NULL,
                reassembly_version INTEGER NOT NULL,
                complete INTEGER NOT NULL,
                status TEXT NOT NULL,
                PRIMARY KEY (generation, batch_id)
            );
            CREATE TABLE IF NOT EXISTS progress (
                generation INTEGER NOT NULL,
                batch_id TEXT NOT NULL,
                cursor_json TEXT NOT NULL,
                updated_monotonic_ms REAL NOT NULL,
                PRIMARY KEY (generation, batch_id)
            );
            """
        )

    def _get_meta(self, key: str) -> str | None:
        row = self.connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return None if row is None else str(row["value"])

    def _set_meta(self, key: str, value: str) -> None:
        self.connection.execute(
            "INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def _snapshot_for(self, spec: TunnelSpec) -> dict[str, Any]:
        return {
            "versions": spec.versions.canonical(),
            "endpoint_fingerprint": spec.endpoint_fingerprint(),
            "route_fingerprint": spec.route_fingerprint(),
            "fragment_fingerprint": spec.fragment_fingerprint(),
            "declaration_fingerprint": spec.declaration_fingerprint(),
            "endpoints": {
                item.endpoint_id: item.canonical()
                for item in sorted(spec.endpoints, key=lambda endpoint: endpoint.endpoint_id)
            },
            "routes": {
                item.tunnel_id: item.canonical()
                for item in sorted(spec.routes, key=lambda route: route.tunnel_id)
            },
        }

    def _empty_content_digest(self, generation: int) -> str:
        return sha256_json({"ledger_format_version": LEDGER_FORMAT_VERSION, "generation": generation, "batches": []})

    def _initialize(self, spec: TunnelSpec) -> int:
        generation = 1
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self._set_meta("ledger_format_version", str(LEDGER_FORMAT_VERSION))
            self._set_meta("active_generation", str(generation))
            self._set_meta("snapshot_json", canonical_json(self._snapshot_for(spec)).decode("utf-8"))
            self._set_meta("ledger_content_digest", self._empty_content_digest(generation))
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        return generation

    def _verify_content_digest(self, generation: int) -> str:
        recorded = self._get_meta("ledger_content_digest")
        computed = self._compute_content_digest(generation)
        if recorded is None:
            fail(E_SNAPSHOT, "ledger.meta.content_digest", "ledger content digest is missing")
        if recorded != computed:
            fail(
                E_SNAPSHOT,
                "ledger.meta.content_digest",
                "ledger content digest does not match the committed rows",
                expected=computed,
                actual=recorded,
            )
        return computed

    def _compute_content_digest(self, generation: int) -> str:
        batches = []
        for row in self.connection.execute(
            "SELECT * FROM batches WHERE generation=? AND status='committed' ORDER BY batch_id",
            (generation,),
        ):
            batch = dict(row)
            batch_id = batch["batch_id"]
            batch["fragments"] = [
                dict(item)
                for item in self.connection.execute(
                    "SELECT * FROM fragments WHERE generation=? AND batch_id=? ORDER BY fragment_index",
                    (generation, batch_id),
                )
            ]
            batch["scheduling"] = [
                dict(item)
                for item in self.connection.execute(
                    "SELECT ordinal,event_json FROM scheduling WHERE generation=? AND batch_id=? ORDER BY ordinal",
                    (generation, batch_id),
                )
            ]
            batch["reassembly"] = [
                dict(item)
                for item in self.connection.execute(
                    "SELECT * FROM reassembly WHERE generation=? AND batch_id=?",
                    (generation, batch_id),
                )
            ]
            batch["progress"] = [
                dict(item)
                for item in self.connection.execute(
                    "SELECT * FROM progress WHERE generation=? AND batch_id=?",
                    (generation, batch_id),
                )
            ]
            batches.append(batch)
        return sha256_json(
            {
                "ledger_format_version": LEDGER_FORMAT_VERSION,
                "generation": generation,
                "batches": batches,
            }
        )

    @staticmethod
    def _diff_snapshot(old: Mapping[str, Any], new: Mapping[str, Any]) -> dict[str, Any]:
        def diff_entries(
            old_entries: Mapping[str, Mapping[str, Any]],
            new_entries: Mapping[str, Mapping[str, Any]],
        ) -> dict[str, Any]:
            added = sorted(set(new_entries) - set(old_entries))
            removed = sorted(set(old_entries) - set(new_entries))
            changed = []
            for key in sorted(set(old_entries) & set(new_entries)):
                fields = sorted(
                    field
                    for field in set(old_entries[key]) | set(new_entries[key])
                    if old_entries[key].get(field) != new_entries[key].get(field)
                )
                if fields:
                    changed.append({"id": key, "fields": fields})
            return {"added": added, "removed": removed, "changed": changed}

        return {
            "endpoints": diff_entries(old.get("endpoints", {}), new.get("endpoints", {})),
            "routes": diff_entries(old.get("routes", {}), new.get("routes", {})),
            "versions": {"old": old.get("versions"), "new": new.get("versions")},
        }

    def open(self, spec: TunnelSpec) -> OpenResult:
        format_text = self._get_meta("ledger_format_version")
        if format_text is None:
            generation = self._initialize(spec)
            return OpenResult(
                initialized=True,
                generation=generation,
                replaced=False,
                replacement_diff=None,
                unknown_batches=(),
                content_digest=self._verify_content_digest(generation),
            )
        try:
            format_version = int(format_text)
        except ValueError:
            fail(E_SNAPSHOT, "ledger.meta.ledger_format_version", "ledger format version is not an integer")
        if format_version != LEDGER_FORMAT_VERSION:
            fail(
                E_SNAPSHOT,
                "ledger.meta.ledger_format_version",
                "unsupported ledger format version",
                expected=LEDGER_FORMAT_VERSION,
                actual=format_version,
            )
        generation_text = self._get_meta("active_generation")
        if generation_text is None:
            fail(E_SNAPSHOT, "ledger.meta.active_generation", "active generation is missing")
        try:
            generation = int(generation_text)
        except ValueError:
            fail(E_SNAPSHOT, "ledger.meta.active_generation", "active generation is not an integer")
        content_digest = self._verify_content_digest(generation)
        snapshot_text = self._get_meta("snapshot_json")
        if snapshot_text is None:
            fail(E_SNAPSHOT, "ledger.meta.snapshot_json", "snapshot metadata is missing")
        try:
            old_snapshot = json.loads(snapshot_text)
        except json.JSONDecodeError as exc:
            fail(E_SNAPSHOT, "ledger.meta.snapshot_json", "snapshot metadata is not valid JSON", error=str(exc))
        new_snapshot = self._snapshot_for(spec)
        old_versions = old_snapshot.get("versions")
        if not isinstance(old_versions, dict):
            fail(E_SNAPSHOT, "ledger.meta.snapshot_json.versions", "snapshot versions are missing or invalid")
        version_names = ("declaration", "endpoint_set", "route_table", "fragment_policy")
        content_fingerprints = {
            "endpoint_set": "endpoint_fingerprint",
            "route_table": "route_fingerprint",
            "fragment_policy": "fragment_fingerprint",
        }
        for version_name in version_names:
            old_value = old_versions.get(version_name)
            new_value = new_snapshot["versions"].get(version_name)
            if not isinstance(old_value, int) or isinstance(old_value, bool):
                fail(
                    E_SNAPSHOT,
                    f"ledger.meta.snapshot_json.versions.{version_name}",
                    "snapshot version must be an integer",
                )
            if new_value < old_value:
                fail(
                    E_SPEC,
                    f"$.versions.{version_name}",
                    "declaration version must be monotonic across ledger snapshots",
                    old=old_value,
                    new=new_value,
                )
            fingerprint_name = content_fingerprints.get(version_name)
            if (
                fingerprint_name is not None
                and old_snapshot.get(fingerprint_name) != new_snapshot.get(fingerprint_name)
                and new_value == old_value
            ):
                fail(
                    E_SPEC,
                    f"$.versions.{version_name}",
                    "declaration content changed without a version increment",
                    old=old_value,
                    new=new_value,
                )
        if (
            old_snapshot.get("declaration_fingerprint") != new_snapshot.get("declaration_fingerprint")
            and new_snapshot["versions"]["declaration"] == old_versions.get("declaration")
        ):
            fail(
                E_SPEC,
                "$.versions.declaration",
                "declaration content changed without a declaration version increment",
            )
        mismatch_fields = [
            field
            for field in (
                "versions",
                "endpoint_fingerprint",
                "route_fingerprint",
                "fragment_fingerprint",
                "declaration_fingerprint",
            )
            if old_snapshot.get(field) != new_snapshot.get(field)
        ]
        replaced = bool(mismatch_fields)
        replacement_diff = None
        if replaced:
            replacement_diff = self._diff_snapshot(old_snapshot, new_snapshot)
            replacement_diff["causes"] = mismatch_fields
            new_generation = generation + 1
            self.connection.execute("BEGIN IMMEDIATE")
            try:
                self.connection.execute(
                    "INSERT INTO snapshot_history(old_generation,new_generation,replaced_at_ms,old_snapshot_json,diff_json,old_content_digest) VALUES(?,?,?,?,?,?)",
                    (
                        generation,
                        new_generation,
                        wall_time_ms(),
                        canonical_json(old_snapshot).decode("utf-8"),
                        canonical_json(replacement_diff).decode("utf-8"),
                        content_digest,
                    ),
                )
                self._set_meta("active_generation", str(new_generation))
                self._set_meta("snapshot_json", canonical_json(new_snapshot).decode("utf-8"))
                self._set_meta("ledger_content_digest", self._empty_content_digest(new_generation))
                self.connection.execute("COMMIT")
            except Exception:
                self.connection.execute("ROLLBACK")
                raise
            generation = new_generation
            content_digest = self._verify_content_digest(generation)

        unknown = self._mark_intents_unknown(generation)
        return OpenResult(
            initialized=False,
            generation=generation,
            replaced=replaced,
            replacement_diff=replacement_diff,
            unknown_batches=tuple(unknown),
            content_digest=content_digest,
        )

    def _active_generation(self) -> int:
        value = self._get_meta("active_generation")
        if value is None:
            fail(E_SNAPSHOT, "ledger.meta.active_generation", "active generation is missing")
        return int(value)

    def _mark_intents_unknown(self, generation: int) -> list[dict[str, Any]]:
        unknown_rows = list(
            self.connection.execute(
                "SELECT batch_id,message_key,expected_fragments,status FROM batches WHERE generation=? AND status IN ('intent','failed','unknown') ORDER BY batch_id",
                (generation,),
            )
        )
        if not unknown_rows:
            return []
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.execute(
                "UPDATE batches SET status='unknown',updated_monotonic_ms=? WHERE generation=? AND status IN ('intent','failed')",
                (monotonic_ms(), generation),
            )
            self.connection.execute(
                "UPDATE planned_fragments SET status='unknown' WHERE generation=? AND batch_id IN (SELECT batch_id FROM batches WHERE generation=? AND status='unknown')",
                (generation, generation),
            )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        result = []
        for row in unknown_rows:
            planned = [
                int(item["fragment_index"])
                for item in self.connection.execute(
                    "SELECT fragment_index FROM planned_fragments WHERE generation=? AND batch_id=? ORDER BY fragment_index",
                    (generation, row["batch_id"]),
                )
            ]
            result.append(
                {
                    "batch_id": row["batch_id"],
                    "message_key": row["message_key"],
                    "expected_fragments": row["expected_fragments"],
                    "unknown_fragment_indices": planned,
                    "rerun_required": True,
                    "previous_status": row["status"],
                }
            )
        return result

    def begin_batch(
        self,
        *,
        batch_id: str,
        message_key: str,
        context: Mapping[str, Any],
        fragment_indices: Iterable[int],
        target_format: str,
    ) -> dict[str, Any]:
        generation = self._active_generation()
        indices = sorted(set(int(index) for index in fragment_indices))
        if not indices:
            fail(E_FIELD, "batch.fragment_indices", "a batch must contain at least one fragment")
        existing = self.connection.execute(
            "SELECT status FROM batches WHERE generation=? AND batch_id=?",
            (generation, batch_id),
        ).fetchone()
        if existing is not None and existing["status"] == "committed":
            return {"generation": generation, "batch_id": batch_id, "status": "already_committed"}
        now = monotonic_ms()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.execute(
                "DELETE FROM planned_fragments WHERE generation=? AND batch_id=?",
                (generation, batch_id),
            )
            self.connection.execute(
                """
                INSERT INTO batches(generation,batch_id,message_key,status,target_format,batch_context_json,expected_fragments,commit_digest,error_json,updated_monotonic_ms)
                VALUES(?,?,?,'intent',?,?,?,NULL,NULL,?)
                ON CONFLICT(generation,batch_id) DO UPDATE SET
                    message_key=excluded.message_key,
                    status='intent',
                    target_format=excluded.target_format,
                    batch_context_json=excluded.batch_context_json,
                    expected_fragments=excluded.expected_fragments,
                    commit_digest=NULL,
                    error_json=NULL,
                    updated_monotonic_ms=excluded.updated_monotonic_ms
                """,
                (
                    generation,
                    batch_id,
                    message_key,
                    target_format,
                    canonical_json(dict(context)).decode("utf-8"),
                    len(indices),
                    now,
                ),
            )
            self.connection.executemany(
                "INSERT INTO planned_fragments(generation,batch_id,fragment_index,message_key,status) VALUES(?,?,?,?,'planned')",
                [(generation, batch_id, index, message_key) for index in indices],
            )
            self.connection.execute(
                "INSERT INTO progress(generation,batch_id,cursor_json,updated_monotonic_ms) VALUES(?,?,?,?) ON CONFLICT(generation,batch_id) DO UPDATE SET cursor_json=excluded.cursor_json,updated_monotonic_ms=excluded.updated_monotonic_ms",
                (generation, batch_id, canonical_json({"state": "intent", "next_fragment_index": indices[0]}).decode("utf-8"), now),
            )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        return {"generation": generation, "batch_id": batch_id, "status": "intent"}

    def mark_batch_unknown(self, batch_id: str, error: Mapping[str, Any] | None = None) -> None:
        generation = self._active_generation()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.execute(
                "UPDATE batches SET status='failed',error_json=?,updated_monotonic_ms=? WHERE generation=? AND batch_id=? AND status!='committed'",
                (
                    None if error is None else canonical_json(dict(error)).decode("utf-8"),
                    monotonic_ms(),
                    generation,
                    batch_id,
                ),
            )
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise

    def commit_batch(
        self,
        *,
        batch_id: str,
        fragments: Iterable[Mapping[str, Any]],
        reassembly: Mapping[str, Any],
        scheduling_statistics: Mapping[str, Any],
        scheduling_events: Iterable[Mapping[str, Any]],
        cursor: Mapping[str, Any],
    ) -> str:
        generation = self._active_generation()
        fragment_rows = sorted((dict(item) for item in fragments), key=lambda item: int(item["fragment_index"]))
        if not fragment_rows:
            fail(E_FIELD, "batch.fragments", "cannot commit an empty fragment batch")
        batch_row = self.connection.execute(
            "SELECT * FROM batches WHERE generation=? AND batch_id=?",
            (generation, batch_id),
        ).fetchone()
        if batch_row is None:
            fail(E_FIELD, "batch.batch_id", "batch must be started before commit", batch_id=batch_id)
        required_fragment_fields = {
            "fragment_index",
            "message_key",
            "fragment_offset",
            "fragment_length",
            "total_fragments",
            "message_id",
            "tunnel_id",
            "target_format",
            "retry_count",
            "fragment_version",
            "wire_b64",
            "wire_sha256",
            "payload_sha256",
            "header_json",
        }
        for row in fragment_rows:
            missing = sorted(required_fragment_fields - set(row))
            if missing:
                fail(
                    E_FIELD,
                    "batch.fragments",
                    "fragment record is missing required fields",
                    fragment_index=row.get("fragment_index"),
                    missing=missing,
                )
        expected = int(batch_row["expected_fragments"])
        if len(fragment_rows) != expected:
            fail(
                E_FIELD,
                "batch.fragments",
                "committed fragment count differs from the planned batch",
                expected=expected,
                actual=len(fragment_rows),
            )
        commit_payload = {
            "batch_id": batch_id,
            "fragments": fragment_rows,
            "reassembly": dict(reassembly),
            "scheduling_statistics": dict(scheduling_statistics),
            "scheduling_events": [dict(item) for item in scheduling_events],
            "cursor": dict(cursor),
        }
        commit_digest = sha256_json(commit_payload)
        now = monotonic_ms()
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            self.connection.execute(
                "DELETE FROM fragments WHERE generation=? AND batch_id=?",
                (generation, batch_id),
            )
            self.connection.executemany(
                """
                INSERT INTO fragments(generation,batch_id,fragment_index,message_key,fragment_offset,fragment_length,total_fragments,message_id,tunnel_id,target_format,retry_count,fragment_version,wire_b64,wire_sha256,payload_sha256,header_json)
                VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        generation,
                        batch_id,
                        int(row["fragment_index"]),
                        str(row["message_key"]),
                        int(row["fragment_offset"]),
                        int(row["fragment_length"]),
                        int(row["total_fragments"]),
                        str(row["message_id"]),
                        str(row["tunnel_id"]),
                        str(row["target_format"]),
                        int(row["retry_count"]),
                        int(row["fragment_version"]),
                        str(row["wire_b64"]),
                        str(row["wire_sha256"]),
                        str(row["payload_sha256"]),
                        str(row["header_json"]),
                    )
                    for row in fragment_rows
                ],
            )
            self.connection.execute(
                "DELETE FROM scheduling WHERE generation=? AND batch_id=?",
                (generation, batch_id),
            )
            scheduling_rows = [
                (generation, batch_id, ordinal, canonical_json(dict(event)).decode("utf-8"))
                for ordinal, event in enumerate(scheduling_events)
            ]
            scheduling_rows.insert(
                0,
                (
                    generation,
                    batch_id,
                    0,
                    canonical_json({"kind": "deterministic_statistics", "value": dict(scheduling_statistics)}).decode("utf-8"),
                ),
            )
            scheduling_rows = [
                (generation, batch_id, ordinal, row[3])
                for ordinal, row in enumerate(scheduling_rows)
            ]
            self.connection.executemany(
                "INSERT INTO scheduling(generation,batch_id,ordinal,event_json) VALUES(?,?,?,?)",
                scheduling_rows,
            )
            self.connection.execute(
                "DELETE FROM reassembly WHERE generation=? AND batch_id=?",
                (generation, batch_id),
            )
            self.connection.execute(
                "INSERT INTO reassembly(generation,batch_id,message_key,payload_sha256,payload_b64,reassembly_version,complete,status) VALUES(?,?,?,?,?,?,?,?)",
                (
                    generation,
                    batch_id,
                    str(reassembly["message_key"]),
                    str(reassembly["payload_sha256"]),
                    str(reassembly["payload_b64"]),
                    int(reassembly["reassembly_version"]),
                    1 if reassembly.get("complete", True) else 0,
                    "complete" if reassembly.get("complete", True) else "incomplete",
                ),
            )
            self.connection.execute(
                "INSERT INTO progress(generation,batch_id,cursor_json,updated_monotonic_ms) VALUES(?,?,?,?) ON CONFLICT(generation,batch_id) DO UPDATE SET cursor_json=excluded.cursor_json,updated_monotonic_ms=excluded.updated_monotonic_ms",
                (generation, batch_id, canonical_json(dict(cursor)).decode("utf-8"), now),
            )
            self.connection.execute(
                "UPDATE batches SET status='committed',commit_digest=?,error_json=NULL,updated_monotonic_ms=? WHERE generation=? AND batch_id=?",
                (commit_digest, now, generation, batch_id),
            )
            self.connection.execute(
                "UPDATE planned_fragments SET status='committed' WHERE generation=? AND batch_id=?",
                (generation, batch_id),
            )
            self._set_meta("ledger_content_digest", self._compute_content_digest(generation))
            self.connection.execute("COMMIT")
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        return commit_digest

    def committed_batches(self) -> list[sqlite3.Row]:
        generation = self._active_generation()
        return list(
            self.connection.execute(
                "SELECT * FROM batches WHERE generation=? AND status='committed' ORDER BY batch_id",
                (generation,),
            )
        )

    def replay_committed(self, spec: TunnelSpec) -> ReplayResult:
        generation = self._active_generation()
        self._verify_content_digest(generation)
        plans = {plan.message_key: plan for plan in generate_messages(spec)}
        checked_fragments = 0
        checked_reassemblies = 0
        batches = self.committed_batches()
        for batch in batches:
            batch_id = str(batch["batch_id"])
            message_key = str(batch["message_key"])
            try:
                context = json.loads(batch["batch_context_json"])
            except json.JSONDecodeError as exc:
                fail(E_REPLAY, f"batches[{batch_id}].batch_context_json", "batch context is not valid JSON", error=str(exc))
            if message_key not in plans:
                fail(
                    E_REPLAY,
                    f"batches[{batch_id}].message_key",
                    "committed batch no longer has a message plan in the current declaration",
                    message_key=message_key,
                )
            plan = plans[message_key]
            if context.get("payload_sha256") != plan.payload_sha256:
                fail(
                    E_REPLAY,
                    f"batches[{batch_id}].sample_payload_sha256",
                    "regenerated deterministic sample digest differs from the committed batch",
                    expected=plan.payload_sha256,
                    actual=context.get("payload_sha256"),
                    batch_id=batch_id,
                )
            target_format = str(context.get("target_format", batch["target_format"]))
            fresh = fragment_message(plan, target_format=target_format)
            rows = list(
                self.connection.execute(
                    "SELECT * FROM fragments WHERE generation=? AND batch_id=? ORDER BY fragment_index",
                    (generation, batch_id),
                )
            )
            if len(rows) != len(fresh.fragments):
                fail(
                    E_REPLAY,
                    f"batches[{batch_id}].fragment_count",
                    "committed fragment count differs from regenerated fragmentation",
                    expected=len(fresh.fragments),
                    actual=len(rows),
                )
            for row in rows:
                index = int(row["fragment_index"])
                if index >= len(fresh.fragments):
                    fail(
                        E_REPLAY,
                        f"batches[{batch_id}].fragments[{index}].fragment_index",
                        "committed fragment index is outside the regenerated sequence",
                        fragment_index=index,
                    )
                regenerated = fresh.fragments[index]
                retry_count = int(row["retry_count"])
                attempted = (
                    with_retry_count(regenerated.normalized, retry_count)
                    if target_format == "f3"
                    else regenerated.normalized
                )
                regenerated_wire = encode_fragment(attempted, target_format)
                comparisons = (
                    ("target_format", str(row["target_format"]), target_format),
                    ("fragment_offset", int(row["fragment_offset"]), regenerated.fragment_offset),
                    ("fragment_length", int(row["fragment_length"]), regenerated.fragment_length),
                    ("total_fragments", int(row["total_fragments"]), regenerated.total_fragments),
                    ("message_id", str(row["message_id"]), regenerated.message_id),
                    ("tunnel_id", str(row["tunnel_id"]), regenerated.tunnel_id),
                    ("retry_count", int(row["retry_count"]), attempted.retry_count),
                    ("wire_sha256", str(row["wire_sha256"]), sha256_hex(regenerated_wire)),
                    (
                        "wire_b64",
                        str(row["wire_b64"]),
                        base64.b64encode(regenerated_wire).decode("ascii"),
                    ),
                )
                for field_name, recorded, expected in comparisons:
                    if recorded != expected:
                        fail(
                            E_REPLAY,
                            f"batches[{batch_id}].fragments[{index}].{field_name}",
                            "first replay mismatch while regenerating a committed fragment",
                            batch_id=batch_id,
                            fragment_index=index,
                            field=field_name,
                            recorded=recorded,
                            expected=expected,
                        )
                decoded = decode_fragment(regenerated_wire, context_for_plan(plan, len(fresh.fragments)))
                if decoded.payload_sha256 != row["payload_sha256"]:
                    fail(
                        E_REPLAY,
                        f"batches[{batch_id}].fragments[{index}].payload_sha256",
                        "first replay mismatch in fragment payload digest",
                        batch_id=batch_id,
                        fragment_index=index,
                        field="payload_sha256",
                        recorded=row["payload_sha256"],
                        expected=decoded.payload_sha256,
                    )
                checked_fragments += 1
            reassembly_row = self.connection.execute(
                "SELECT * FROM reassembly WHERE generation=? AND batch_id=?",
                (generation, batch_id),
            ).fetchone()
            if reassembly_row is None:
                fail(
                    E_REPLAY,
                    f"batches[{batch_id}].reassembly",
                    "committed batch has no reassembly result",
                )
            if reassembly_row["payload_sha256"] != fresh.reassembly.payload_sha256:
                fail(
                    E_REPLAY,
                    f"batches[{batch_id}].reassembly.payload_sha256",
                    "first replay mismatch in reassembled payload digest",
                    batch_id=batch_id,
                    field="payload_sha256",
                    recorded=reassembly_row["payload_sha256"],
                    expected=fresh.reassembly.payload_sha256,
                )
            recorded_payload = base64.b64decode(reassembly_row["payload_b64"].encode("ascii"), validate=True)
            if recorded_payload != fresh.reassembly.payload:
                fail(
                    E_REPLAY,
                    f"batches[{batch_id}].reassembly.payload_b64",
                    "first replay mismatch in reassembled payload bytes",
                    batch_id=batch_id,
                    field="payload_b64",
                )
            checked_reassemblies += 1
        return ReplayResult(
            checked_batches=len(batches),
            checked_fragments=checked_fragments,
            checked_reassemblies=checked_reassemblies,
        )
