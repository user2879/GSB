"""Independent, read-only ledger replay and consistency checker."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .content import derive_shard, manifest_digest
from .errors import ReplayError
from .ledger import SCHEMA_VERSION
from .model import Manifest


def _readonly_connection(path: Path) -> sqlite3.Connection:
    uri = path.resolve().as_uri() + "?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=5.0)
    conn.row_factory = sqlite3.Row
    return conn


def _capacity(manifest: Manifest, leases: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cfg = manifest.config
    by_node: dict[str, dict[str, Any]] = {
        node.node_id: {
            "slots_declared": node.slots,
            "lease_limit": node.lease_limit,
            "active_leases": 0,
            "active_bytes": 0,
            "used_slots": [],
            "remaining_slots": node.slots,
            "remaining_leases": node.lease_limit,
        }
        for node in manifest.nodes
    }
    global_active = 0
    global_bytes = 0
    for lease in leases.values():
        node_data = by_node.get(lease["node_id"])
        if node_data is None:
            continue
        global_active += 1
        global_bytes += int(lease["bytes"])
        node_data["active_leases"] += 1
        node_data["active_bytes"] += int(lease["bytes"])
        node_data["used_slots"].append(int(lease["slot_index"]))
    for node_data in by_node.values():
        node_data["used_slots"] = sorted(set(node_data["used_slots"]))
        node_data["remaining_slots"] = max(0, node_data["slots_declared"] - len(node_data["used_slots"]))
        node_data["remaining_leases"] = max(0, node_data["lease_limit"] - node_data["active_leases"])
    return {
        "global_active_leases": global_active,
        "global_active_bytes": global_bytes,
        "global_lease_limit": cfg.global_lease_limit,
        "global_remaining_leases": max(0, cfg.global_lease_limit - global_active),
        "global_byte_budget": cfg.global_byte_budget,
        "global_remaining_bytes": max(0, cfg.global_byte_budget - global_bytes),
        "nodes": {key: by_node[key] for key in sorted(by_node)},
    }


def _node_leases(manifest: Manifest, leases: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {node.node_id: [] for node in manifest.nodes}
    for lease in sorted(leases.values(), key=lambda item: (item["node_id"], item["slot_index"], item["lease_id"])):
        result.setdefault(lease["node_id"], []).append(
            {
                "lease_id": lease["lease_id"],
                "content_key": lease["content_key"],
                "shard_index": int(lease["shard_index"]),
                "slot_index": int(lease["slot_index"]),
                "bytes": int(lease["bytes"]),
                "expires_ms": int(lease["expires_ms"]),
                "lease_version": int(lease["lease_version"]),
                "quota_version": int(lease["quota_version"]),
                "digest": lease["digest"],
                "status": lease["status"],
            }
        )
    return result


def _first_diff(actual: Any, expected: Any, prefix: str = "") -> tuple[str, Any, Any] | None:
    if type(actual) is not type(expected):
        return prefix or "<root>", actual, expected
    if isinstance(actual, dict):
        actual_keys = set(actual)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            key = sorted(actual_keys ^ expected_keys)[0]
            path = f"{prefix}.{key}" if prefix else key
            return path, actual.get(key), expected.get(key)
        for key in sorted(actual):
            path = f"{prefix}.{key}" if prefix else key
            diff = _first_diff(actual[key], expected[key], path)
            if diff is not None:
                return diff
        return None
    if isinstance(actual, list):
        if len(actual) != len(expected):
            return prefix, actual, expected
        for index, (left, right) in enumerate(zip(actual, expected)):
            diff = _first_diff(left, right, f"{prefix}[{index}]")
            if diff is not None:
                return diff
        return None
    if actual != expected:
        return prefix or "<root>", actual, expected
    return None


def _node_from_path(path: str) -> str | None:
    parts = path.split(".")
    if len(parts) >= 2 and parts[0] == "nodes":
        return parts[1]
    return None


def replay_manifest(db_path: str | Path, manifest: Manifest) -> dict[str, Any]:
    path = Path(db_path)
    if not path.exists():
        raise ReplayError("ledger does not exist", field="<ledger>")
    try:
        conn = _readonly_connection(path)
    except sqlite3.DatabaseError as exc:
        raise ReplayError("ledger cannot be opened read-only", field="<ledger>") from exc
    try:
        meta_rows = conn.execute("SELECT key,value FROM meta").fetchall()
        meta = {row["key"]: row["value"] for row in meta_rows}
        expected_digest = manifest_digest(manifest)
        if meta.get("format_version") != str(SCHEMA_VERSION):
            raise ReplayError("ledger format_version does not match replay format", field="format_version")
        if meta.get("manifest_digest") != expected_digest:
            raise ReplayError("ledger manifest_digest does not match declarations", field="manifest_digest")
        journal = conn.execute("SELECT * FROM journal ORDER BY ledger_seq").fetchall()
        claims_db = {
            row["lease_id"]: dict(row)
            for row in conn.execute("SELECT * FROM slot_claims").fetchall()
        }
        state_row = conn.execute("SELECT * FROM state WHERE id=1").fetchone()
        if state_row is None:
            raise ReplayError("ledger state row is missing", field="state")
        current_view = {
            row["lease_id"]: dict(row)
            for row in conn.execute("SELECT * FROM current_view").fetchall()
        }
        leases: dict[str, dict[str, Any]] = {}
        claims: dict[str, dict[str, Any]] = {}
        batches: dict[str, str] = {}
        current_quota = manifest.initial_quota_version
        active_nodes = {node.node_id for node in manifest.nodes}
        requires_rerun = False
        last_event_seq = 0
        digest_by_lease: dict[str, str] = {}
        grant_event_seq: dict[str, int] = {}
        records = 0

        for row in journal:
            records += 1
            event_seq = int(row["event_seq"])
            last_event_seq = event_seq
            kind = row["kind"]
            payload = json.loads(row["payload_json"])
            node_hint = row["node_id"]
            if kind == "BATCH_OPEN":
                batches[payload["batch_id"]] = "OPEN"
            elif kind == "LEASE_GRANT":
                lease = {
                    "lease_id": payload["lease_id"],
                    "queue_id": payload.get("queue_id"),
                    "batch_id": payload.get("batch_id"),
                    "node_id": payload["node_id"],
                    "content_key": payload["content_key"],
                    "shard_index": int(payload["shard_index"]),
                    "slot_index": int(payload["slot_index"]),
                    "bytes": int(payload["bytes"]),
                    "granted_ms": int(payload["granted_ms"]),
                    "expires_ms": int(payload["expires_ms"]),
                    "lease_version": int(payload["lease_version"]),
                    "quota_version": int(payload["quota_version"]),
                    "digest": payload["digest"],
                    "checksum": int(payload["checksum"]),
                    "status": "RESERVED",
                    "attempt": int(payload.get("attempt", 0)),
                }
                lease_id = lease["lease_id"]
                node_hint = lease["node_id"]
                if lease_id in leases:
                    raise ReplayError("duplicate lease grant", event_seq=event_seq, node_id=node_hint, field="lease_id")
                if lease["node_id"] not in manifest.node_map or lease["content_key"] not in manifest.content_map:
                    raise ReplayError("grant references unknown declaration", event_seq=event_seq, node_id=node_hint, field="lease_id")
                for other in leases.values():
                    if other["node_id"] == lease["node_id"] and other["slot_index"] == lease["slot_index"]:
                        raise ReplayError(
                            "capacity slot is already owned",
                            event_seq=event_seq,
                            node_id=node_hint,
                            field="slot_index",
                        )
                computed = derive_shard(
                    manifest.node_map[lease["node_id"]],
                    manifest.content_map[lease["content_key"]],
                    lease["shard_index"],
                    manifest.config.sample_scale,
                )
                if computed.digest != lease["digest"]:
                    raise ReplayError(
                        "replayed shard digest differs",
                        event_seq=event_seq,
                        node_id=node_hint,
                        field="digest",
                    )
                if computed.checksum != lease["checksum"]:
                    raise ReplayError(
                        "replayed fixed-point checksum differs",
                        event_seq=event_seq,
                        node_id=node_hint,
                        field="checksum",
                    )
                leases[lease_id] = lease
                digest_by_lease[lease_id] = lease["digest"]
                grant_event_seq[lease_id] = event_seq
            elif kind == "BATCH_COMMIT":
                for lease_id in payload["lease_ids"]:
                    if lease_id not in leases:
                        raise ReplayError("commit references unknown lease", event_seq=event_seq, field="lease_ids")
                    leases[lease_id]["status"] = "ACTIVE"
                batches[payload["batch_id"]] = "COMMITTED"
            elif kind == "LEASE_RENEW":
                lease_id = payload["lease_id"]
                lease = leases.get(lease_id)
                if lease is None:
                    raise ReplayError("renew references inactive lease", event_seq=event_seq, node_id=node_hint, field="lease_id")
                if int(payload["old_version"]) != lease["lease_version"]:
                    raise ReplayError("renew old_version mismatch", event_seq=event_seq, node_id=node_hint, field="old_version")
                lease["lease_version"] = int(payload["new_version"])
                lease["expires_ms"] = int(payload["new_expires_ms"])
            elif kind == "LEASE_CLAIM":
                lease_id = payload["lease_id"]
                if lease_id in claims:
                    raise ReplayError("duplicate claim for lease", event_seq=event_seq, node_id=node_hint, field="claim_kind")
                if lease_id not in leases:
                    raise ReplayError("claim references inactive lease", event_seq=event_seq, node_id=node_hint, field="lease_id")
                claims[lease_id] = {
                    "lease_id": lease_id,
                    "node_id": leases[lease_id]["node_id"],
                    "slot_index": int(leases[lease_id]["slot_index"]),
                    "claim_kind": payload["claim_kind"],
                    "basis": payload["basis"],
                    "decision_event_seq": event_seq,
                    "quota_version": int(payload.get("claim_quota_version", leases[lease_id]["quota_version"])),
                }
                leases.pop(lease_id, None)
            elif kind == "LEASE_UNKNOWN":
                batch_id = payload["batch_id"]
                batches[batch_id] = "UNKNOWN"
                for lease_info in payload.get("leases", []):
                    lease_id = lease_info["lease_id"]
                    if lease_id in claims:
                        raise ReplayError("duplicate unknown claim", event_seq=event_seq, node_id=node_hint, field="claim_kind")
                    lease = leases.get(lease_id)
                    if lease is None:
                        raise ReplayError("unknown lease is not reserved", event_seq=event_seq, node_id=node_hint, field="lease_id")
                    claims[lease_id] = {
                        "lease_id": lease_id,
                        "node_id": lease["node_id"],
                        "slot_index": int(lease["slot_index"]),
                        "claim_kind": "UNKNOWN",
                        "basis": payload["basis"],
                        "decision_event_seq": event_seq,
                        "quota_version": int(lease["quota_version"]),
                    }
                    leases.pop(lease_id, None)
                requires_rerun = True
            elif kind == "QUOTA_CHANGE":
                previous = current_quota
                current_quota = int(payload["to_quota_version"])
                active_nodes = set(payload["node_ids"])
                stale = [lease for lease in leases.values() if lease["quota_version"] < current_quota]
                if stale:
                    raise ReplayError(
                        "quota replacement retained old-version leases",
                        event_seq=event_seq,
                        node_id=stale[0]["node_id"],
                        field="quota_version",
                    )
                if payload.get("replacement") != "whole_view":
                    raise ReplayError("quota replacement was not a whole-view replacement", event_seq=event_seq, field="replacement")
                if payload.get("from_quota_version") != previous:
                    raise ReplayError("quota replacement from_version mismatch", event_seq=event_seq, field="from_quota_version")
            elif kind == "RERUN_RESOLVED":
                requires_rerun = False
            # LEASE_REJECT, QUEUE_START and QUEUE_CANCEL leave lease/capacity state unchanged.

            expected_capacity = _capacity(manifest, leases)
            expected_node_leases = _node_leases(manifest, leases)
            checkpoint = conn.execute("SELECT * FROM checkpoints WHERE ledger_seq=?", (row["ledger_seq"],)).fetchone()
            if checkpoint is None:
                raise ReplayError("missing checkpoint", event_seq=event_seq, node_id=node_hint, field="checkpoint")
            if int(checkpoint["format_version"]) != SCHEMA_VERSION:
                raise ReplayError("checkpoint format_version mismatch", event_seq=event_seq, node_id=node_hint, field="format_version")
            checkpoint_capacity = json.loads(checkpoint["capacity_json"])
            diff = _first_diff(checkpoint_capacity, expected_capacity)
            if diff is not None:
                path_text, actual, expected = diff
                raise ReplayError(
                    f"capacity checkpoint mismatch: actual={actual!r} expected={expected!r}",
                    event_seq=event_seq,
                    node_id=_node_from_path(path_text) or node_hint,
                    field=path_text,
                )
            checkpoint_leases = json.loads(checkpoint["node_leases_json"])
            diff = _first_diff(checkpoint_leases, expected_node_leases)
            if diff is not None:
                path_text, actual, expected = diff
                raise ReplayError(
                    f"node lease checkpoint mismatch: actual={actual!r} expected={expected!r}",
                    event_seq=event_seq,
                    node_id=_node_from_path(path_text) or node_hint,
                    field=path_text,
                )
            if int(checkpoint["progress_cursor"]) < event_seq:
                raise ReplayError("checkpoint progress_cursor regressed", event_seq=event_seq, field="progress_cursor")
            if kind in {"LEASE_GRANT", "LEASE_RENEW", "LEASE_CLAIM"}:
                expected_digest = digest_by_lease.get(payload.get("lease_id"))
                if expected_digest is not None and checkpoint["content_digest"] != expected_digest:
                    raise ReplayError("checkpoint content_digest mismatch", event_seq=event_seq, node_id=node_hint, field="content_digest")

        view_fields = (
            "lease_id",
            "node_id",
            "content_key",
            "shard_index",
            "slot_index",
            "bytes",
            "expires_ms",
            "lease_version",
            "quota_version",
            "digest",
            "status",
        )
        reconstructed_view = {
            lease_id: {field: lease[field] for field in view_fields} for lease_id, lease in leases.items()
        }
        current_sorted = sorted(
            ({field: row[field] for field in view_fields} for row in current_view.values()),
            key=lambda item: item["lease_id"],
        )
        current_by_id = {row["lease_id"]: row for row in current_sorted}
        diff = _first_diff(current_by_id, reconstructed_view)
        if diff is not None:
            path_text, actual, expected = diff
            lease_id = path_text.split(".")[0]
            node_id = (current_by_id.get(lease_id) or reconstructed_view.get(lease_id) or {}).get("node_id")
            field = path_text.split(".", 1)[1] if "." in path_text else path_text
            raise ReplayError(
                f"final current view mismatch: actual={actual!r} expected={expected!r}",
                event_seq=grant_event_seq.get(lease_id, last_event_seq),
                node_id=node_id,
                field=field,
            )
        normalized_claims = {}
        for lease_id, claim in claims.items():
            normalized_claims[lease_id] = {
                "lease_id": lease_id,
                "node_id": claim["node_id"],
                "slot_index": int(claim["slot_index"]),
                "claim_kind": claim["claim_kind"],
                "basis": claim["basis"],
                "decision_event_seq": int(claim["decision_event_seq"]),
                "quota_version": int(claim["quota_version"]),
            }
        db_claims = {
            lease_id: {
                "lease_id": claim["lease_id"],
                "node_id": claim["node_id"],
                "slot_index": int(claim["slot_index"]),
                "claim_kind": claim["claim_kind"],
                "basis": claim["basis"],
                "decision_event_seq": int(claim["decision_event_seq"]),
                "quota_version": int(claim["quota_version"]),
            }
            for lease_id, claim in claims_db.items()
        }
        claim_mismatches: list[tuple[int, str, str, Any, Any]] = []
        for lease_id in sorted(set(db_claims) | set(normalized_claims)):
            actual_claim = db_claims.get(lease_id)
            expected_claim = normalized_claims.get(lease_id)
            diff = _first_diff(actual_claim, expected_claim, lease_id)
            if diff is not None:
                path_text, actual, expected = diff
                event_seq = int(
                    (actual_claim or expected_claim or {}).get("decision_event_seq", last_event_seq)
                )
                claim_mismatches.append((event_seq, lease_id, path_text, actual, expected))
        if claim_mismatches:
            event_seq, lease_id, path_text, actual, expected = min(
                claim_mismatches, key=lambda item: (item[0], item[1], item[2])
            )
            node_id = (claims_db.get(lease_id) or {}).get("node_id") or (normalized_claims.get(lease_id) or {}).get("node_id")
            field = path_text.split(".", 1)[1] if "." in path_text else path_text
            raise ReplayError(
                f"claim adjudication mismatch: actual={actual!r} expected={expected!r}",
                event_seq=event_seq,
                node_id=node_id,
                field=field,
            )
        if int(state_row["current_quota_version"]) != current_quota:
            raise ReplayError("final quota version mismatch", event_seq=last_event_seq, field="current_quota_version")
        if set(json.loads(state_row["active_nodes_json"])) != active_nodes:
            raise ReplayError("final active node set mismatch", event_seq=last_event_seq, field="active_nodes")
        if bool(state_row["requires_rerun"]) != requires_rerun:
            raise ReplayError("requires_rerun flag mismatch", event_seq=last_event_seq, field="requires_rerun")
        return {
            "ok": True,
            "code": "OK",
            "format_version": SCHEMA_VERSION,
            "manifest_digest": expected_digest,
            "journal_records": records,
            "last_event_seq": last_event_seq,
            "current_quota_version": current_quota,
            "active_nodes": sorted(active_nodes),
            "active_lease_count": len(reconstructed_view),
            "claim_count": len(normalized_claims),
            "capacity": _capacity(manifest, reconstructed_view),
        }
    except ReplayError:
        raise
    except (sqlite3.DatabaseError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise ReplayError(f"ledger replay failed: {exc}", field="<ledger>") from exc
    finally:
        conn.close()
