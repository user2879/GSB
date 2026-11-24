from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote

from . import LEDGER_FORMAT_VERSION
from .digest import generate_artifact
from .errors import LeaseError
from .spec import PipelineSpec, load_spec
from .util import canonical_sha256, is_int, mono_ms


SCHEMA_VERSION = 1
KNOWN_EVENT_KINDS = {
    "SESSION_STARTED",
    "SESSION_CLOSED",
    "NODE_SPEC_SYNC",
    "SYNC_DIGESTS",
    "LEASE_REQUESTED",
    "LEASE_GRANTED",
    "LEASE_RENEWED",
    "LEASE_EXPIRED",
    "QUOTA_RELEASED",
    "LEASE_CANCELLED",
    "LEASE_UNKNOWN",
    "LEASE_INVALIDATED",
    "DECISIONS_RECORDED",
    "SPEC_RECONCILED",
}


def _verify_error(
    message: str,
    *,
    field: str,
    expected: Any = None,
    actual: Any = None,
    artifact_key: str | None = None,
    node_id: str | None = None,
) -> None:
    raise LeaseError(
        "E_VERIFY",
        message,
        field=field,
        expected=expected,
        actual=actual,
        artifact_key=artifact_key,
        node_id=node_id,
    )


def _open_readonly(path: Path) -> sqlite3.Connection:
    if not path.exists():
        _verify_error(
            "ledger does not exist",
            field="ledger.path",
            expected="existing SQLite ledger",
            actual=str(path),
        )
    uri = "file:" + quote(str(path.resolve())) + "?mode=ro"
    connection = sqlite3.connect(uri, uri=True, timeout=10)
    connection.row_factory = sqlite3.Row
    return connection


def _table_rows(
    connection: sqlite3.Connection, sql: str, columns: tuple[str, ...]
) -> list[dict[str, Any]]:
    return [
        {column: row[column] for column in columns}
        for row in connection.execute(sql).fetchall()
    ]


def _recompute_content_digest(connection: sqlite3.Connection) -> str:
    content = {
        "format_version": LEDGER_FORMAT_VERSION,
        "schema_version": SCHEMA_VERSION,
        "events": _table_rows(
            connection,
            "SELECT seq,event_uid,kind,mono_ms,node_id,artifact_key,lease_id,"
            "request_seq,rationale,payload_json FROM events ORDER BY seq",
            (
                "seq",
                "event_uid",
                "kind",
                "mono_ms",
                "node_id",
                "artifact_key",
                "lease_id",
                "request_seq",
                "rationale",
                "payload_json",
            ),
        ),
        "nodes": _table_rows(
            connection,
            "SELECT node_id,host,port,capacity_slots,lease_limit,used_slots,"
            "active_leases,updated_seq FROM nodes ORDER BY node_id",
            (
                "node_id",
                "host",
                "port",
                "capacity_slots",
                "lease_limit",
                "used_slots",
                "active_leases",
                "updated_seq",
            ),
        ),
        "leases": _table_rows(
            connection,
            "SELECT lease_id,node_id,request_seq,artifact_keys_json,status,"
            "granted_mono_ms,expires_mono_ms,closed_mono_ms,terminal_reason,updated_seq "
            "FROM leases ORDER BY lease_id",
            (
                "lease_id",
                "node_id",
                "request_seq",
                "artifact_keys_json",
                "status",
                "granted_mono_ms",
                "expires_mono_ms",
                "closed_mono_ms",
                "terminal_reason",
                "updated_seq",
            ),
        ),
        "claims": _table_rows(
            connection,
            "SELECT artifact_key,lease_id,node_id FROM claims ORDER BY artifact_key",
            ("artifact_key", "lease_id", "node_id"),
        ),
        "summaries": _table_rows(
            connection,
            "SELECT node_id,artifact_key,digest,refs_json,generation,updated_seq "
            "FROM summaries ORDER BY artifact_key,node_id",
            ("node_id", "artifact_key", "digest", "refs_json", "generation", "updated_seq"),
        ),
        "sources": _table_rows(
            connection,
            "SELECT node_id,artifact_key,digest,refs_json,valid,generation,updated_seq "
            "FROM sources ORDER BY artifact_key,node_id",
            ("node_id", "artifact_key", "digest", "refs_json", "valid", "generation", "updated_seq"),
        ),
        "decisions": _table_rows(
            connection,
            "SELECT artifact_key,canonical_digest,refs_json,source_nodes_json,reusable,"
            "selected_node,verdicts_json,updated_seq FROM decisions ORDER BY artifact_key",
            (
                "artifact_key",
                "canonical_digest",
                "refs_json",
                "source_nodes_json",
                "reusable",
                "selected_node",
                "verdicts_json",
                "updated_seq",
            ),
        ),
        "progress": _table_rows(
            connection,
            "SELECT edit_hash,total_events,phase,completed_keys_json,updated_seq "
            "FROM progress WHERE id=1",
            ("edit_hash", "total_events", "phase", "completed_keys_json", "updated_seq"),
        ),
    }
    return canonical_sha256(content)


def _refs_json(references: list[str]) -> str:
    return json.dumps(
        sorted(references), ensure_ascii=False, separators=(",", ":")
    )


def _canonical_from_generated(
    spec: PipelineSpec,
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for artifact in sorted(spec.artifacts, key=lambda item: item.key):
        first_node = spec.sorted_node_ids[0]
        digest = generate_artifact(spec.node_by_id[first_node], artifact.key).digest
        result[artifact.key] = {
            "digest": digest,
            "references": list(artifact.references),
            "canonical_node": first_node,
        }
    return result


def _select_node(
    spec: PipelineSpec,
    key: str,
    canonical_digest: str,
    node_state: dict[str, dict[str, int]],
) -> tuple[str | None, list[dict[str, Any]]]:
    verdicts: list[dict[str, Any]] = []
    eligible: list[tuple[int, int, str]] = []
    for node in sorted(spec.nodes, key=lambda item: item.node_id):
        digest = generate_artifact(node, key).digest
        state = node_state[node.node_id]
        digest_match = digest == canonical_digest
        free = node.capacity_slots - state["used_slots"]
        active = state["active_leases"]
        veto: str | None = None
        detail = ""
        is_eligible = True
        if not digest_match:
            is_eligible = False
            veto = "DIGEST_MISMATCH"
            detail = "node content digest differs from canonical digest"
        elif active >= node.lease_limit:
            is_eligible = False
            veto = "E_LEASE_LIMIT"
            detail = f"active={active} limit={node.lease_limit}"
        elif free < 1:
            is_eligible = False
            veto = "E_CAPACITY"
            detail = f"free={free}"
        if is_eligible:
            eligible.append((-1, -free, node.node_id))
        verdicts.append(
            {
                "node_id": node.node_id,
                "digest": digest,
                "digest_match": digest_match,
                "free_slots": free,
                "active_leases": active,
                "lease_limit": node.lease_limit,
                "eligible": is_eligible,
                "veto_reason": veto,
                "veto_detail": detail,
            }
        )
    eligible.sort()
    selected = eligible[0][2] if eligible else None
    selected_free = -eligible[0][1] if eligible else 0
    for verdict in verdicts:
        verdict["selected"] = verdict["node_id"] == selected
        if not verdict["selected"] and verdict["eligible"]:
            if verdict["free_slots"] < selected_free:
                verdict["veto_reason"] = "FEWER_FREE_SLOTS"
                verdict["veto_detail"] = (
                    f"free slots {verdict['free_slots']} < selected {selected_free}"
                )
            else:
                verdict["veto_reason"] = "NODE_ID_ORDER"
                verdict["veto_detail"] = f"node id sorts after {selected}"
    return selected, verdicts


def verify_ledger(
    spec_path: str | Path, db_path: str | Path
) -> dict[str, Any]:
    started = mono_ms()
    spec = load_spec(spec_path)
    connection = _open_readonly(Path(db_path))
    try:
        meta = {
            row["key"]: row["value"]
            for row in connection.execute("SELECT * FROM meta ORDER BY key").fetchall()
        }
        if int(meta.get("schema_version", "0")) != SCHEMA_VERSION:
            _verify_error(
                "ledger schema version mismatch",
                field="meta.schema_version",
                expected=SCHEMA_VERSION,
                actual=meta.get("schema_version"),
            )
        if int(meta.get("format_version", "0")) != LEDGER_FORMAT_VERSION:
            _verify_error(
                "ledger format version mismatch",
                field="meta.format_version",
                expected=LEDGER_FORMAT_VERSION,
                actual=meta.get("format_version"),
            )

        canonical = _canonical_from_generated(spec)

        # 1. Independently regenerate every node/key digest and compare the
        # persisted summary.  Ordering is artifact key then node id.
        summary_rows = {
            (row["artifact_key"], row["node_id"]): row
            for row in connection.execute("SELECT * FROM summaries").fetchall()
        }
        for artifact in sorted(spec.artifacts, key=lambda item: item.key):
            for node in sorted(spec.nodes, key=lambda item: item.node_id):
                expected_digest = generate_artifact(node, artifact.key).digest
                row = summary_rows.get((artifact.key, node.node_id))
                if row is None:
                    _verify_error(
                        "missing node digest summary",
                        field="summaries.digest",
                        expected=expected_digest,
                        actual="<missing>",
                        artifact_key=artifact.key,
                        node_id=node.node_id,
                    )
                if row["digest"] != expected_digest:
                    _verify_error(
                        "regenerated artifact digest differs from ledger",
                        field="summaries.digest",
                        expected=expected_digest,
                        actual=row["digest"],
                        artifact_key=artifact.key,
                        node_id=node.node_id,
                    )
                expected_refs = _refs_json(list(artifact.references))
                if row["refs_json"] != expected_refs:
                    _verify_error(
                        "artifact reference list differs from specification",
                        field="summaries.references",
                        expected=expected_refs,
                        actual=row["refs_json"],
                        artifact_key=artifact.key,
                        node_id=node.node_id,
                    )
        extra_summaries = sorted(
            set(summary_rows)
            - {
                (artifact.key, node.node_id)
                for artifact in spec.artifacts
                for node in spec.nodes
            }
        )
        if extra_summaries:
            key, node_id = extra_summaries[0]
            _verify_error(
                "ledger contains a summary outside the specification",
                field="summaries.artifact_key",
                expected="only declared artifact/node pairs",
                actual=f"{node_id}/{key}",
                artifact_key=key,
                node_id=node_id,
            )

        # 2. Replay every committed event into an independent ownership model.
        events = connection.execute("SELECT * FROM events ORDER BY seq").fetchall()
        expected_sequences = list(range(1, len(events) + 1))
        actual_sequences = [int(row["seq"]) for row in events]
        if actual_sequences != expected_sequences:
            _verify_error(
                "event sequence is not contiguous",
                field="events.seq",
                expected=expected_sequences,
                actual=actual_sequences,
            )
        node_state = {
            node.node_id: {"used_slots": 0, "active_leases": 0}
            for node in spec.nodes
        }
        leases: dict[str, dict[str, Any]] = {}
        claims: dict[str, str] = {}
        produced: set[str] = set()
        for event in events:
            kind = event["kind"]
            if kind not in KNOWN_EVENT_KINDS:
                _verify_error(
                    "unknown event kind in ledger",
                    field="events.kind",
                    expected="known event kind",
                    actual=kind,
                    artifact_key=event["artifact_key"],
                    node_id=event["node_id"],
                )
            payload = json.loads(event["payload_json"])
            lease_id = event["lease_id"]
            event_node = event["node_id"]
            if kind == "LEASE_REQUESTED":
                if not lease_id or lease_id in leases:
                    _verify_error(
                        "duplicate or missing requested lease id",
                        field="events.lease_id",
                        expected="new unique lease id",
                        actual=lease_id,
                        artifact_key=event["artifact_key"],
                        node_id=event_node,
                    )
                key = event["artifact_key"]
                leases[lease_id] = {
                    "lease_id": lease_id,
                    "node_id": event_node,
                    "request_seq": int(event["request_seq"]),
                    "artifact_keys": [key],
                    "status": "PENDING",
                    "granted_mono_ms": None,
                    "expires_mono_ms": None,
                    "closed_mono_ms": None,
                    "terminal_reason": None,
                }
            elif kind == "LEASE_CANCELLED":
                lease = leases.get(lease_id)
                if lease is None or lease["status"] != "PENDING":
                    _verify_error(
                        "cancellation requires a pending lease",
                        field="leases.status",
                        expected="PENDING",
                        actual=None if lease is None else lease["status"],
                        artifact_key=event["artifact_key"],
                        node_id=event_node,
                    )
                lease["status"] = "CANCELLED"
                lease["closed_mono_ms"] = int(event["mono_ms"])
                lease["terminal_reason"] = event["rationale"]
            elif kind == "LEASE_GRANTED":
                lease = leases.get(lease_id)
                if lease is None or lease["status"] != "PENDING":
                    _verify_error(
                        "grant requires a pending lease",
                        field="leases.status",
                        expected="PENDING",
                        actual=None if lease is None else lease["status"],
                        artifact_key=event["artifact_key"],
                        node_id=event_node,
                    )
                keys = sorted(payload.get("artifact_keys", []))
                lease["artifact_keys"] = keys
                node_spec = spec.node_by_id.get(event_node)
                if event_node not in node_state:
                    if "node_capacity_slots" not in payload or "node_lease_limit" not in payload:
                        _verify_error(
                            "retired node grant lacks historical limits",
                            field="events.payload.node_lease_limit",
                            expected="grant-time capacity and lease limit",
                            actual="<missing>",
                            artifact_key=keys[0],
                            node_id=event_node,
                        )
                    node_state[event_node] = {"used_slots": 0, "active_leases": 0}
                capacity_slots = int(
                    payload.get(
                        "node_capacity_slots",
                        node_spec.capacity_slots if node_spec is not None else 0,
                    )
                )
                lease_limit = int(
                    payload.get(
                        "node_lease_limit",
                        node_spec.lease_limit if node_spec is not None else 0,
                    )
                )
                state = node_state[event_node]
                if state["active_leases"] + 1 > lease_limit:
                    _verify_error(
                        "grant exceeds the node lease limit",
                        field="nodes.active_leases",
                        expected=f"<= {lease_limit}",
                        actual=state["active_leases"] + 1,
                        artifact_key=keys[0],
                        node_id=event_node,
                    )
                if state["used_slots"] + len(keys) > capacity_slots:
                    _verify_error(
                        "grant exceeds node capacity",
                        field="nodes.used_slots",
                        expected=f"<= {capacity_slots}",
                        actual=state["used_slots"] + len(keys),
                        artifact_key=keys[0],
                        node_id=event_node,
                    )
                for key in keys:
                    if key in claims:
                        _verify_error(
                            "artifact has two simultaneous capacity owners",
                            field="claims.artifact_key",
                            expected="one owner",
                            actual=claims[key],
                            artifact_key=key,
                            node_id=event_node,
                        )
                    claims[key] = lease_id
                    produced.add(key)
                state["used_slots"] += len(keys)
                state["active_leases"] += 1
                lease["status"] = "ACTIVE"
                lease["granted_mono_ms"] = int(event["mono_ms"])
                lease["expires_mono_ms"] = int(payload["expires_mono_ms"])
            elif kind == "LEASE_RENEWED":
                lease = leases.get(lease_id)
                if lease is None or lease["status"] != "ACTIVE":
                    _verify_error(
                        "renewal requires an active lease",
                        field="leases.status",
                        expected="ACTIVE",
                        actual=None if lease is None else lease["status"],
                        artifact_key=event["artifact_key"],
                        node_id=event_node,
                    )
                new_expiry = int(payload["expires_mono_ms"])
                if new_expiry <= int(lease["expires_mono_ms"]):
                    _verify_error(
                        "renewal did not extend the monotonic expiry",
                        field="leases.expires_mono_ms",
                        expected=f"> {lease['expires_mono_ms']}",
                        actual=new_expiry,
                        artifact_key=event["artifact_key"],
                        node_id=event_node,
                    )
                lease["expires_mono_ms"] = new_expiry
            elif kind in {"LEASE_EXPIRED", "QUOTA_RELEASED", "LEASE_UNKNOWN", "LEASE_INVALIDATED"}:
                lease = leases.get(lease_id)
                if lease is None or lease["status"] not in {"PENDING", "ACTIVE"}:
                    _verify_error(
                        "terminal ownership event requires an open lease",
                        field="leases.status",
                        expected="PENDING or ACTIVE",
                        actual=None if lease is None else lease["status"],
                        artifact_key=event["artifact_key"],
                        node_id=event_node,
                    )
                if kind == "LEASE_EXPIRED":
                    if int(event["mono_ms"]) < int(lease["expires_mono_ms"]):
                        _verify_error(
                            "lease expired before its local monotonic deadline",
                            field="leases.expires_mono_ms",
                            expected=f"event mono_ms >= {lease['expires_mono_ms']}",
                            actual=event["mono_ms"],
                            artifact_key=event["artifact_key"],
                            node_id=event_node,
                        )
                    status = "EXPIRED"
                elif kind == "QUOTA_RELEASED":
                    status = "RELEASED"
                elif kind == "LEASE_UNKNOWN":
                    status = "UNKNOWN"
                else:
                    status = "INVALIDATED"
                if lease["status"] == "ACTIVE":
                    keys = lease["artifact_keys"]
                    state = node_state[event_node]
                    state["used_slots"] -= len(keys)
                    state["active_leases"] -= 1
                    if state["used_slots"] < 0 or state["active_leases"] < 0:
                        _verify_error(
                            "capacity ownership was released more than once",
                            field="nodes.used_slots",
                            expected="non-negative capacity projection",
                            actual=state,
                            artifact_key=event["artifact_key"],
                            node_id=event_node,
                        )
                    for key in keys:
                        if claims.get(key) != lease_id:
                            _verify_error(
                                "claim owner differs from the closing lease",
                                field="claims.lease_id",
                                expected=lease_id,
                                actual=claims.get(key),
                                artifact_key=key,
                                node_id=event_node,
                            )
                        claims.pop(key, None)
                lease["status"] = status
                lease["closed_mono_ms"] = int(event["mono_ms"])
                lease["terminal_reason"] = event["rationale"]
                for key in payload.get("invalidated_keys", []):
                    produced.discard(key)
                if kind == "LEASE_INVALIDATED":
                    for key in payload.get("affected_keys", []):
                        produced.discard(key)
            elif kind == "SYNC_DIGESTS":
                for key in payload.get("reuse_invalidated_keys", []):
                    produced.discard(key)
            # SESSION_STARTED, SESSION_CLOSED, NODE_SPEC_SYNC and
            # DECISIONS_RECORDED do not alter lease ownership.

        # Compare replay against persisted projection.
        db_nodes = {
            row["node_id"]: row
            for row in connection.execute("SELECT * FROM nodes ORDER BY node_id").fetchall()
        }
        for node in sorted(spec.nodes, key=lambda item: item.node_id):
            if node.node_id not in db_nodes:
                _verify_error(
                    "declared node is absent from ledger",
                    field="nodes.node_id",
                    expected=node.node_id,
                    actual="<missing>",
                    node_id=node.node_id,
                )
            state = node_state[node.node_id]
            row = db_nodes[node.node_id]
            expected_node_fields = {
                "host": node.host,
                "port": node.port,
                "capacity_slots": node.capacity_slots,
                "lease_limit": node.lease_limit,
            }
            for field, expected_value in expected_node_fields.items():
                if row[field] != expected_value:
                    _verify_error(
                        "node projection differs from current specification",
                        field=f"nodes.{field}",
                        expected=expected_value,
                        actual=row[field],
                        node_id=node.node_id,
                    )
            if int(row["used_slots"]) != state["used_slots"]:
                _verify_error(
                    "node used-slot projection differs from event replay",
                    field="nodes.used_slots",
                    expected=state["used_slots"],
                    actual=row["used_slots"],
                    node_id=node.node_id,
                )
            if int(row["active_leases"]) != state["active_leases"]:
                _verify_error(
                    "node active-lease projection differs from event replay",
                    field="nodes.active_leases",
                    expected=state["active_leases"],
                    actual=row["active_leases"],
                    node_id=node.node_id,
                )
        extra_nodes = sorted(set(db_nodes) - set(spec.node_by_id))
        if extra_nodes:
            node_id = extra_nodes[0]
            _verify_error(
                "ledger contains an undeclared node",
                field="nodes.node_id",
                expected="only declared nodes",
                actual=node_id,
                node_id=node_id,
            )
        for node_id, state in sorted(node_state.items()):
            if node_id not in spec.node_by_id and (
                state["used_slots"] != 0 or state["active_leases"] != 0
            ):
                _verify_error(
                    "retired node still owns capacity after replay",
                    field="nodes.used_slots",
                    expected=0,
                    actual=state,
                    node_id=node_id,
                )

        db_leases = {
            row["lease_id"]: row
            for row in connection.execute("SELECT * FROM leases ORDER BY lease_id").fetchall()
        }
        for lease_id in sorted(leases):
            lease = leases[lease_id]
            row = db_leases.get(lease_id)
            if row is None:
                _verify_error(
                    "lease missing from persisted projection",
                    field="leases.lease_id",
                    expected=lease_id,
                    actual="<missing>",
                    artifact_key=lease["artifact_keys"][0],
                    node_id=lease["node_id"],
                )
            for field in (
                "node_id",
                "request_seq",
                "status",
                "granted_mono_ms",
                "expires_mono_ms",
                "closed_mono_ms",
            ):
                if row[field] != lease[field]:
                    _verify_error(
                        "lease projection differs from event replay",
                        field=f"leases.{field}",
                        expected=lease[field],
                        actual=row[field],
                        artifact_key=lease["artifact_keys"][0] if lease["artifact_keys"] else None,
                        node_id=lease["node_id"],
                    )
            if sorted(json.loads(row["artifact_keys_json"])) != sorted(lease["artifact_keys"]):
                _verify_error(
                    "lease artifact set differs from event replay",
                    field="leases.artifact_keys",
                    expected=sorted(lease["artifact_keys"]),
                    actual=sorted(json.loads(row["artifact_keys_json"])),
                    artifact_key=lease["artifact_keys"][0],
                    node_id=lease["node_id"],
                )
        extra_leases = sorted(set(db_leases) - set(leases))
        if extra_leases:
            lease_id = extra_leases[0]
            row = db_leases[lease_id]
            _verify_error(
                "lease exists without a committed request event",
                field="leases.lease_id",
                expected="event-backed lease",
                actual=lease_id,
                artifact_key=json.loads(row["artifact_keys_json"])[0],
                node_id=row["node_id"],
            )
        db_claims = {
            row["artifact_key"]: row["lease_id"]
            for row in connection.execute("SELECT * FROM claims ORDER BY artifact_key").fetchall()
        }
        if db_claims != claims:
            keys = sorted(set(db_claims) | set(claims))
            key = keys[0]
            _verify_error(
                "active claim projection differs from event replay",
                field="claims.lease_id",
                expected=claims.get(key),
                actual=db_claims.get(key),
                artifact_key=key,
            )

        # 3. Reconstruct reusable source sets from produced keys and current
        # canonical digests.  This is independent of the ledger's source flags.
        source_rows = {
            (row["artifact_key"], row["node_id"]): row
            for row in connection.execute("SELECT * FROM sources ORDER BY artifact_key,node_id").fetchall()
        }
        expected_valid_by_key: dict[str, list[str]] = {}
        for artifact in sorted(spec.artifacts, key=lambda item: item.key):
            key = artifact.key
            canonical_info = canonical[key]
            expected_nodes: list[str] = []
            if key in produced:
                for node in sorted(spec.nodes, key=lambda item: item.node_id):
                    if generate_artifact(node, key).digest == canonical_info["digest"]:
                        expected_nodes.append(node.node_id)
            expected_valid_by_key[key] = expected_nodes
            actual_nodes = sorted(
                node_id
                for (row_key, node_id), row in source_rows.items()
                if row_key == key and int(row["valid"]) == 1
            )
            if actual_nodes != expected_nodes:
                symmetric = sorted(set(actual_nodes) ^ set(expected_nodes))
                _verify_error(
                    "valid reusable source nodes differ from independent derivation",
                    field="sources.valid_nodes",
                    expected=expected_nodes,
                    actual=actual_nodes,
                    artifact_key=key,
                    node_id=symmetric[0] if symmetric else None,
                )
            for node_id in expected_nodes:
                row = source_rows.get((key, node_id))
                if row is None:
                    _verify_error(
                        "expected reusable source row is missing",
                        field="sources.artifact_key",
                        expected=key,
                        actual="<missing>",
                        artifact_key=key,
                        node_id=node_id,
                    )
                if row["digest"] != canonical_info["digest"]:
                    _verify_error(
                        "source digest differs from canonical digest",
                        field="sources.digest",
                        expected=canonical_info["digest"],
                        actual=row["digest"],
                        artifact_key=key,
                        node_id=node_id,
                    )
                if row["refs_json"] != _refs_json(canonical_info["references"]):
                    _verify_error(
                        "source references differ from canonical references",
                        field="sources.references",
                        expected=_refs_json(canonical_info["references"]),
                        actual=row["refs_json"],
                        artifact_key=key,
                        node_id=node_id,
                    )

        # 4. Independently derive reuse closure and final arbitration.
        decisions = {
            row["artifact_key"]: row
            for row in connection.execute("SELECT * FROM decisions ORDER BY artifact_key").fetchall()
        }
        derived_reusable: dict[str, bool] = {}
        for layer in spec.topological_layers():
            for key in sorted(layer):
                artifact = spec.artifact_by_key[key]
                deps_ok = all(derived_reusable[ref] for ref in artifact.references)
                sources = expected_valid_by_key[key]
                reusable = bool(sources) and deps_ok
                derived_reusable[key] = reusable
                canonical_info = canonical[key]
                selected, verdicts = _select_node(
                    spec, key, canonical_info["digest"], node_state
                )
                row = decisions.get(key)
                if row is None:
                    _verify_error(
                        "artifact decision is missing",
                        field="decisions.artifact_key",
                        expected=key,
                        actual="<missing>",
                        artifact_key=key,
                    )
                checks = (
                    ("decisions.canonical_digest", canonical_info["digest"], row["canonical_digest"]),
                    (
                        "decisions.references",
                        _refs_json(artifact.references),
                        row["refs_json"],
                    ),
                    (
                        "decisions.source_nodes",
                        json.dumps(sources, separators=(",", ":")),
                        row["source_nodes_json"],
                    ),
                    ("decisions.reusable", 1 if reusable else 0, int(row["reusable"])),
                    ("decisions.selected_node", selected, row["selected_node"]),
                )
                for field, expected_value, actual_value in checks:
                    if expected_value != actual_value:
                        _verify_error(
                            "artifact decision differs from independent derivation",
                            field=field,
                            expected=expected_value,
                            actual=actual_value,
                            artifact_key=key,
                            node_id=selected,
                        )
                actual_verdicts = json.loads(row["verdicts_json"])
                for expected_verdict, actual_verdict in zip(verdicts, actual_verdicts):
                    for field in ("selected", "eligible", "veto_reason"):
                        if expected_verdict.get(field) != actual_verdict.get(field):
                            _verify_error(
                                "arbitration verdict differs from independent derivation",
                                field=f"decisions.verdicts.{field}",
                                expected=expected_verdict.get(field),
                                actual=actual_verdict.get(field),
                                artifact_key=key,
                                node_id=expected_verdict["node_id"],
                            )
        extra_decisions = sorted(set(decisions) - set(spec.artifact_by_key))
        if extra_decisions:
            key = extra_decisions[0]
            _verify_error(
                "decision exists for an undeclared artifact",
                field="decisions.artifact_key",
                expected="declared artifact key",
                actual=key,
                artifact_key=key,
            )

        # 5. Progress cursor and ledger content digest.
        progress = connection.execute("SELECT * FROM progress WHERE id=1").fetchone()
        max_seq = len(events)
        if progress is None:
            _verify_error(
                "progress cursor is missing",
                field="progress",
                expected="single row id=1",
                actual="<missing>",
            )
        if int(progress["total_events"]) != max_seq or int(progress["updated_seq"]) != max_seq:
            _verify_error(
                "progress cursor does not point at the last committed event",
                field="progress.total_events",
                expected=max_seq,
                actual={
                    "total_events": progress["total_events"],
                    "updated_seq": progress["updated_seq"],
                },
            )
        completed = sorted(
            {
                row["artifact_key"]
                for row in source_rows.values()
                if int(row["valid"]) == 1
            }
        )
        if sorted(json.loads(progress["completed_keys_json"])) != completed:
            _verify_error(
                "progress completed keys differ from valid reusable sources",
                field="progress.completed_keys",
                expected=completed,
                actual=sorted(json.loads(progress["completed_keys_json"])),
            )
        actual_digest = _recompute_content_digest(connection)
        if meta.get("content_digest") != actual_digest:
            _verify_error(
                "ledger content digest does not match committed records",
                field="meta.content_digest",
                expected=meta.get("content_digest"),
                actual=actual_digest,
            )
        return {
            "status": "ok",
            "format_version": LEDGER_FORMAT_VERSION,
            "spec_sha256": spec.content_sha256,
            "ledger_path": str(Path(db_path).resolve()),
            "ledger_content_digest": actual_digest,
            "event_count": max_seq,
            "artifact_count": len(spec.artifacts),
            "node_count": len(spec.nodes),
            "lease_count": len(leases),
            "reusable_artifacts": sorted(
                key for key, value in derived_reusable.items() if value
            ),
            "elapsed_mono_ms": mono_ms() - started,
        }
    finally:
        connection.close()
