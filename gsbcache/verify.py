"""Independent ledger replay verifier.

The verifier does not import the runtime arbiter.  It reconstructs the initial
capacity/quota state, re-generates every node/key digest from the explicit
content rule, re-runs candidate ranking, and re-applies custody transitions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .errors import LedgerError, VerifyError
from .ledger import GENESIS_HASH, LEDGER_FORMAT_VERSION, Ledger
from .spec import Spec, load_spec
from .util import canonical_json, sha256_value


def _independent_digest(node, artifact_key: str) -> str:
    """Independent copy of the documented generation rule.

    This deliberately does not call the runtime NodeStore/content helper.
    """
    payload = {
        "artifact_key": artifact_key,
        "content_version": 1,
        "node_seed": node.seed,
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_fail(
    message: str,
    *,
    artifact_key: str | None = None,
    node_id: str | None = None,
    field: str,
    expected: Any = None,
    actual: Any = None,
    event_seq: int | None = None,
    **details: Any,
) -> None:
    payload = {
        "artifact_key": artifact_key,
        "node_id": node_id,
        "field": field,
        "expected": expected,
        "actual": actual,
        "event_seq": event_seq,
    }
    payload.update(details)
    raise VerifyError(message, **payload)


def _parse_row(item: dict[str, Any]) -> dict[str, Any]:
    result = dict(item)
    result["artifact_keys"] = json.loads(result["artifact_keys"]) if isinstance(result.get("artifact_keys"), str) else list(result.get("artifact_keys", []))
    result["digests"] = json.loads(result["digests"]) if isinstance(result.get("digests"), str) else dict(result.get("digests", {}))
    result["details"] = json.loads(result["details"]) if isinstance(result.get("details"), str) else dict(result.get("details", {}))
    return result


def verify_ledger(spec_path: str | Path, ledger_path: str | Path) -> dict[str, Any]:
    spec = load_spec(spec_path)
    try:
        ledger = Ledger(ledger_path, spec, read_only=True)
    except LedgerError as exc:
        raise VerifyError(
            "ledger integrity check failed before replay",
            field=exc.details.get("field", "ledger"),
            expected=exc.details.get("expected"),
            actual=exc.details.get("actual"),
            differences=exc.details.get("differences", []),
        ) from exc
    try:
        return _verify_open_ledger(spec, ledger, ledger_path)
    finally:
        ledger.close()


def _verify_open_ledger(spec: Spec, ledger: Ledger, ledger_path: str | Path) -> dict[str, Any]:
    meta = {
        "format_version": ledger._meta_int("format_version"),
        "cursor": ledger._meta_int("cursor"),
        "last_event_hash": ledger._meta_get("last_event_hash"),
        "spec_digest": ledger._meta_get("spec_digest"),
        "content_digest": ledger._meta_get("content_digest"),
        "pipeline_id": ledger._meta_get("pipeline_id"),
    }
    if meta["format_version"] != LEDGER_FORMAT_VERSION:
        _verify_fail(
            "ledger format version mismatch",
            field="format_version",
            expected=LEDGER_FORMAT_VERSION,
            actual=meta["format_version"],
        )
    if meta["spec_digest"] != spec.digest:
        _verify_fail(
            "ledger spec digest does not match the original declaration",
            field="spec_digest",
            expected=spec.digest,
            actual=meta["spec_digest"],
        )
    if meta["pipeline_id"] != spec.pipeline.pipeline_id:
        _verify_fail(
            "ledger pipeline id does not match the original declaration",
            field="pipeline_id",
            expected=spec.pipeline.pipeline_id,
            actual=meta["pipeline_id"],
        )

    rows = ledger.conn.execute("SELECT * FROM events ORDER BY seq").fetchall()
    events = [_parse_row(dict(row)) for row in rows]
    node_state: dict[str, dict[str, int]] = {
        node.node_id: {"capacity_used": 0, "active_leases": 0} for node in spec.nodes
    }
    active: dict[str, dict[str, Any]] = {}
    active_keys: dict[str, str] = {}
    previous_hash = GENESIS_HASH
    expected_seq = 0
    last_arbitration: int | None = None
    replayed: list[dict[str, Any]] = []

    for event in events:
        seq = event.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool) or seq != expected_seq + 1:
            _verify_fail(
                "event sequence is not contiguous",
                field="seq",
                expected=expected_seq + 1,
                actual=seq,
                event_seq=seq if isinstance(seq, int) else None,
            )
        expected_seq = seq
        if event.get("prev_hash") != previous_hash:
            _verify_fail(
                "event hash chain prev_hash mismatch",
                field="prev_hash",
                expected=previous_hash,
                actual=event.get("prev_hash"),
                event_seq=seq,
            )
        record = {key: value for key, value in event.items() if key != "event_hash"}
        computed_hash = sha256_value(record)
        if event.get("event_hash") != computed_hash:
            _verify_fail(
                "event hash mismatch",
                field="event_hash",
                expected=computed_hash,
                actual=event.get("event_hash"),
                event_seq=seq,
            )
        previous_hash = computed_hash
        event_type = event["event_type"]
        keys = list(event.get("artifact_keys") or [])
        digests = dict(event.get("digests") or {})
        first_key = keys[0] if keys else None

        if event_type == "lease_granted":
            arbitration = event.get("arbitration_index")
            if not isinstance(arbitration, int) or isinstance(arbitration, bool):
                _verify_fail(
                    "grant is missing integer arbitration_index",
                    artifact_key=first_key,
                    node_id=event.get("node_id"),
                    field="arbitration_index",
                    actual=arbitration,
                    event_seq=seq,
                )
            if last_arbitration is not None and arbitration <= last_arbitration:
                _verify_fail(
                    "grant arbitration indices are not strictly increasing",
                    artifact_key=first_key,
                    node_id=event.get("node_id"),
                    field="arbitration_index",
                    expected=f">{last_arbitration}",
                    actual=arbitration,
                    event_seq=seq,
                )
            last_arbitration = arbitration
            node_id = event.get("node_id")
            if node_id not in spec.node_by_id:
                _verify_fail(
                    "grant references an unknown node",
                    artifact_key=first_key,
                    node_id=node_id,
                    field="node_id",
                    actual=node_id,
                    event_seq=seq,
                )
            node = spec.node_by_id[node_id]
            if not keys or len(set(keys)) != len(keys):
                _verify_fail(
                    "grant key set is empty or contains duplicates",
                    artifact_key=first_key,
                    node_id=node_id,
                    field="artifact_keys",
                    actual=keys,
                    event_seq=seq,
                )
            for key in keys:
                if key not in spec.artifacts:
                    _verify_fail(
                        "grant references an unknown artifact",
                        artifact_key=key,
                        node_id=node_id,
                        field="artifact_key",
                        actual=key,
                        event_seq=seq,
                    )
                if key in active_keys:
                    _verify_fail(
                        "artifact was granted while already owned by another lease",
                        artifact_key=key,
                        node_id=node_id,
                        field="ownership",
                        expected=f"single owner, got {active_keys[key]}",
                        actual=event.get("lease_id"),
                        event_seq=seq,
                    )
                generated = _independent_digest(node, key)
                if generated != spec.artifacts[key].expected_digest:
                    _verify_fail(
                        "ledger granted a node whose generated digest does not match the manifest",
                        artifact_key=key,
                        node_id=node_id,
                        field="expected_digest",
                        expected=spec.artifacts[key].expected_digest,
                        actual=generated,
                        event_seq=seq,
                    )
                if digests.get(key) != generated:
                    _verify_fail(
                        "ledger node digest differs from independently generated digest",
                        artifact_key=key,
                        node_id=node_id,
                        field="digests",
                        expected=generated,
                        actual=digests.get(key),
                        event_seq=seq,
                    )
                for ref in spec.artifacts[key].refs:
                    if _independent_digest(node, ref) != spec.artifacts[ref].expected_digest:
                        _verify_fail(
                            "granted replica has a non-reusable reference",
                            artifact_key=key,
                            node_id=node_id,
                            field="refs",
                            expected=spec.artifacts[ref].expected_digest,
                            actual=_independent_digest(node, ref),
                            reference=ref,
                            event_seq=seq,
                        )

            # Independently derive the deterministic winner at this point.
            candidates: list[tuple[int, str, int]] = []
            key = keys[0]
            artifact = spec.artifacts[key]
            for candidate_node in spec.nodes:
                generated = _independent_digest(candidate_node, key)
                digest_match = generated == artifact.expected_digest
                refs_ok = all(
                    _independent_digest(candidate_node, ref) == spec.artifacts[ref].expected_digest
                    for ref in artifact.refs
                )
                state = node_state[candidate_node.node_id]
                cap_remaining = candidate_node.capacity_slots - state["capacity_used"]
                quota_remaining = candidate_node.lease_limit - state["active_leases"]
                if (
                    digest_match
                    and refs_ok
                    and cap_remaining > 0
                    and quota_remaining > 0
                    and key not in active_keys
                ):
                    candidates.append((-cap_remaining, candidate_node.node_id, quota_remaining))
            candidates.sort(key=lambda item: (item[0], item[1]))
            expected_winner = candidates[0][1] if candidates else None
            if node_id != expected_winner:
                _verify_fail(
                    "ledger owner differs from independent arbitration winner",
                    artifact_key=key,
                    node_id=node_id,
                    field="node_id",
                    expected=expected_winner,
                    actual=node_id,
                    event_seq=seq,
                    candidates=[
                        {"node_id": n.node_id, "capacity_remaining": n.capacity_slots - node_state[n.node_id]["capacity_used"]}
                        for n in spec.nodes
                    ],
                )
            state = node_state[node_id]
            if state["capacity_used"] + len(keys) > node.capacity_slots:
                _verify_fail(
                    "node capacity slots exceeded",
                    artifact_key=key,
                    node_id=node_id,
                    field="capacity_slots",
                    expected=f"<= {node.capacity_slots}",
                    actual=state["capacity_used"] + len(keys),
                    event_seq=seq,
                )
            if state["active_leases"] + 1 > node.lease_limit:
                _verify_fail(
                    "node lease limit exceeded",
                    artifact_key=key,
                    node_id=node_id,
                    field="lease_limit",
                    expected=f"<= {node.lease_limit}",
                    actual=state["active_leases"] + 1,
                    event_seq=seq,
                )
            lease_id = event.get("lease_id")
            if not isinstance(lease_id, str) or not lease_id or lease_id in active:
                _verify_fail(
                    "lease id is missing, empty or duplicated",
                    artifact_key=key,
                    node_id=node_id,
                    field="lease_id",
                    actual=lease_id,
                    event_seq=seq,
                )
            if not isinstance(event.get("expires_at_ms"), int) or isinstance(event.get("expires_at_ms"), bool):
                _verify_fail(
                    "lease expiry must be an integer millisecond timestamp",
                    artifact_key=key,
                    node_id=node_id,
                    field="expires_at_ms",
                    actual=event.get("expires_at_ms"),
                    event_seq=seq,
                )
            state["capacity_used"] += len(keys)
            state["active_leases"] += 1
            active[lease_id] = {
                "lease_id": lease_id,
                "node_id": node_id,
                "artifact_keys": keys,
                "digests": digests,
                "expires_at_ms": event["expires_at_ms"],
                "state": "active",
            }
            for key in keys:
                active_keys[key] = lease_id
            replayed.append({"seq": seq, "event_type": event_type, "lease_id": lease_id})

        elif event_type == "lease_renewed":
            lease_id = event.get("lease_id")
            lease = active.get(lease_id)
            if lease is None:
                _verify_fail(
                    "renewal references a non-active lease",
                    artifact_key=event.get("artifact_keys", [None])[0] if event.get("artifact_keys") else None,
                    node_id=event.get("node_id"),
                    field="lease_id",
                    actual=lease_id,
                    event_seq=seq,
                )
            if not isinstance(event.get("expires_at_ms"), int) or isinstance(event.get("expires_at_ms"), bool):
                _verify_fail(
                    "renewal expiry must be an integer",
                    artifact_key=lease["artifact_keys"][0],
                    node_id=lease["node_id"],
                    field="expires_at_ms",
                    actual=event.get("expires_at_ms"),
                    event_seq=seq,
                )
            if int(event.get("details", {}).get("capacity_slots_changed", 0)) != 0:
                _verify_fail(
                    "renewal must not change capacity",
                    artifact_key=lease["artifact_keys"][0],
                    node_id=lease["node_id"],
                    field="capacity_slots_changed",
                    expected=0,
                    actual=event.get("details", {}).get("capacity_slots_changed"),
                    event_seq=seq,
                )
            lease["expires_at_ms"] = event["expires_at_ms"]
            replayed.append({"seq": seq, "event_type": event_type, "lease_id": lease_id})

        elif event_type in {"lease_expired", "quota_recycled", "lease_unknown", "lease_released"}:
            lease_id = event.get("lease_id")
            lease = active.get(lease_id)
            if lease is None:
                _verify_fail(
                    "custody release references a non-active or already released lease",
                    artifact_key=first_key,
                    node_id=event.get("node_id"),
                    field="lease_state",
                    expected="active",
                    actual="missing_or_closed",
                    event_seq=seq,
                )
            parent_key = lease["artifact_keys"][0] if lease["artifact_keys"] else None
            node_id = lease["node_id"]
            details = event.get("details", {})
            expected_slots = len(lease["artifact_keys"])
            if int(details.get("capacity_slots_released", -1)) != expected_slots:
                _verify_fail(
                    "custody release capacity count is not exactly-once",
                    artifact_key=parent_key,
                    node_id=node_id,
                    field="capacity_slots_released",
                    expected=expected_slots,
                    actual=details.get("capacity_slots_released"),
                    event_seq=seq,
                )
            if int(details.get("quota_units_released", -1)) != 1:
                _verify_fail(
                    "custody release quota count is not exactly-once",
                    artifact_key=parent_key,
                    node_id=node_id,
                    field="quota_units_released",
                    expected=1,
                    actual=details.get("quota_units_released"),
                    event_seq=seq,
                )
            if details.get("custody_token") != lease_id:
                _verify_fail(
                    "custody token does not identify the released lease",
                    artifact_key=parent_key,
                    node_id=node_id,
                    field="custody_token",
                    expected=lease_id,
                    actual=details.get("custody_token"),
                    event_seq=seq,
                )
            if event_type == "lease_expired" and event.get("release_kind") != "expiry":
                _verify_fail(
                    "lease_expired event has the wrong release kind",
                    artifact_key=parent_key,
                    node_id=node_id,
                    field="release_kind",
                    expected="expiry",
                    actual=event.get("release_kind"),
                    event_seq=seq,
                )
            if event_type == "quota_recycled" and event.get("release_kind") != "quota_recycle":
                _verify_fail(
                    "quota_recycled event has the wrong release kind",
                    artifact_key=parent_key,
                    node_id=node_id,
                    field="release_kind",
                    expected="quota_recycle",
                    actual=event.get("release_kind"),
                    event_seq=seq,
                )
            state = node_state[node_id]
            state["capacity_used"] = max(0, state["capacity_used"] - expected_slots)
            state["active_leases"] = max(0, state["active_leases"] - 1)
            for key in lease["artifact_keys"]:
                if active_keys.get(key) == lease_id:
                    active_keys.pop(key, None)
            active.pop(lease_id, None)
            replayed.append({"seq": seq, "event_type": event_type, "lease_id": lease_id})
        elif event_type == "attempt_unknown":
            if not event.get("details", {}).get("requires_reapply"):
                _verify_fail(
                    "unknown attempt does not require reapplication",
                    artifact_key=first_key,
                    node_id=event.get("node_id"),
                    field="requires_reapply",
                    expected=True,
                    actual=False,
                    event_seq=seq,
                )
            replayed.append({"seq": seq, "event_type": event_type})
        else:
            _verify_fail(
                "unknown ledger event type",
                artifact_key=first_key,
                node_id=event.get("node_id"),
                field="event_type",
                actual=event_type,
                event_seq=seq,
            )

    if meta["cursor"] != expected_seq:
        _verify_fail(
            "ledger cursor does not match replayed event count",
            field="cursor",
            expected=expected_seq,
            actual=meta["cursor"],
        )
    if meta["last_event_hash"] != previous_hash:
        _verify_fail(
            "ledger final event hash does not match replayed hash chain",
            field="last_event_hash",
            expected=previous_hash,
            actual=meta["last_event_hash"],
        )

    db_nodes = {row["node_id"]: row for row in ledger.get_nodes()}
    for node in spec.nodes:
        db_row = db_nodes.get(node.node_id)
        expected_state = node_state[node.node_id]
        if db_row is None:
            _verify_fail(
                "ledger node row is missing",
                node_id=node.node_id,
                field="node_row",
                expected="present",
                actual="missing",
            )
        if int(db_row["capacity_used"]) != expected_state["capacity_used"]:
            _verify_fail(
                "ledger capacity remaining disagrees with replay",
                node_id=node.node_id,
                field="capacity_used",
                expected=expected_state["capacity_used"],
                actual=db_row["capacity_used"],
            )
        if int(db_row["capacity_slots"]) != node.capacity_slots:
            _verify_fail(
                "ledger capacity_slots disagrees with the declaration",
                node_id=node.node_id,
                field="capacity_slots",
                expected=node.capacity_slots,
                actual=db_row["capacity_slots"],
            )
        if int(db_row["lease_limit"]) != node.lease_limit:
            _verify_fail(
                "ledger lease_limit disagrees with the declaration",
                node_id=node.node_id,
                field="lease_limit",
                expected=node.lease_limit,
                actual=db_row["lease_limit"],
            )
        if "capacity_remaining" in db_row and int(db_row["capacity_remaining"]) != node.capacity_slots - expected_state["capacity_used"]:
            _verify_fail(
                "ledger capacity_remaining disagrees with replay",
                node_id=node.node_id,
                field="capacity_remaining",
                expected=node.capacity_slots - expected_state["capacity_used"],
                actual=db_row["capacity_remaining"],
            )
        if int(db_row["active_leases"]) != expected_state["active_leases"]:
            _verify_fail(
                "ledger active lease count disagrees with replay",
                node_id=node.node_id,
                field="active_leases",
                expected=expected_state["active_leases"],
                actual=db_row["active_leases"],
            )
        if "quota_remaining" in db_row and int(db_row["quota_remaining"]) != node.lease_limit - expected_state["active_leases"]:
            _verify_fail(
                "ledger quota_remaining disagrees with replay",
                node_id=node.node_id,
                field="quota_remaining",
                expected=node.lease_limit - expected_state["active_leases"],
                actual=db_row["quota_remaining"],
            )

    db_leases = {
        row["lease_id"]: _parse_row(dict(row))
        for row in ledger.conn.execute("SELECT * FROM leases ORDER BY lease_id").fetchall()
    }
    expected_ids = set(active)
    db_active_ids = {
        lease_id for lease_id, row in db_leases.items() if row.get("state") == "active"
    }
    if db_active_ids != expected_ids:
        missing = sorted(expected_ids - db_active_ids)
        extra = sorted(db_active_ids - expected_ids)
        focus = (missing or extra)[0] if (missing or extra) else None
        _verify_fail(
            "active lease set disagrees with replay",
            artifact_key=db_leases.get(focus, {}).get("artifact_keys", [None])[0] if focus else None,
            node_id=db_leases.get(focus, {}).get("node_id"),
            field="state",
            expected=sorted(expected_ids),
            actual=sorted(db_active_ids),
        )
    for lease_id, expected_lease in active.items():
        db_lease = db_leases.get(lease_id)
        if db_lease is None:
            _verify_fail(
                "replayed lease is absent from the ledger",
                artifact_key=expected_lease["artifact_keys"][0],
                node_id=expected_lease["node_id"],
                field="lease_id",
                expected=lease_id,
                actual=None,
            )
        if int(db_lease["expires_at_ms"]) != int(expected_lease["expires_at_ms"]):
            _verify_fail(
                "lease expiry disagrees with replay",
                artifact_key=expected_lease["artifact_keys"][0],
                node_id=expected_lease["node_id"],
                field="expires_at_ms",
                expected=expected_lease["expires_at_ms"],
                actual=db_lease["expires_at_ms"],
            )
        if db_lease["node_id"] != expected_lease["node_id"]:
            _verify_fail(
                "lease owner disagrees with replay",
                artifact_key=expected_lease["artifact_keys"][0],
                node_id=expected_lease["node_id"],
                field="node_id",
                expected=expected_lease["node_id"],
                actual=db_lease["node_id"],
            )
        if list(db_lease["artifact_keys"]) != list(expected_lease["artifact_keys"]):
            _verify_fail(
                "lease key set disagrees with replay",
                artifact_key=expected_lease["artifact_keys"][0],
                node_id=expected_lease["node_id"],
                field="artifact_keys",
                expected=expected_lease["artifact_keys"],
                actual=db_lease["artifact_keys"],
            )
        if dict(db_lease["digests"]) != dict(expected_lease["digests"]):
            _verify_fail(
                "lease digests disagree with replay",
                artifact_key=expected_lease["artifact_keys"][0],
                node_id=expected_lease["node_id"],
                field="digests",
                expected=expected_lease["digests"],
                actual=db_lease["digests"],
            )

    return {
        "ok": True,
        "code": "OK",
        "ledger": str(ledger_path),
        "format_version": meta["format_version"],
        "content_digest": meta["content_digest"],
        "events_replayed": len(events),
        "cursor": meta["cursor"],
        "active_leases": len(active),
        "final_capacity_used": {
            node_id: values["capacity_used"] for node_id, values in sorted(node_state.items())
        },
        "final_active_leases": {
            node_id: values["active_leases"] for node_id, values in sorted(node_state.items())
        },
        "conclusion": "ledger replay matches independently derived digests, ownership and custody",
    }
