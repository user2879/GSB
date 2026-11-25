from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from . import LEDGER_FORMAT_VERSION
from .canonical import canonical_dumps
from .errors import ReplayError
from .integrity import compute_shard_proof
from .spec import SpecBundle, load_spec_dir


TERMINAL_EVENTS = {
    "shard_verified": "completed",
    "lease_expired": "expiry_reaper",
    "lease_quota_revoked": "quota_revision",
    "recovery_reclaimed": "startup_recovery",
}


@dataclass
class ReplayReport:
    ok: bool
    ledger_path: str
    event_count: int
    lease_count: int
    active_leases: list[dict[str, Any]]
    verified_shards: list[str]
    unknown_shards: list[str]
    adjudications: int
    receipts_adopted: int
    receipts_rejected: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "ledger_path": self.ledger_path,
            "event_count": self.event_count,
            "lease_count": self.lease_count,
            "active_leases": self.active_leases,
            "verified_shards": self.verified_shards,
            "unknown_shards": self.unknown_shards,
            "adjudications": self.adjudications,
            "receipts_adopted": self.receipts_adopted,
            "receipts_rejected": self.receipts_rejected,
        }


def _ro_conn(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path.resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def verify_ledger(spec_dir: str | Path, ledger_path: str | Path) -> ReplayReport:
    spec = load_spec_dir(spec_dir)
    path = Path(ledger_path)
    if not path.exists():
        raise ReplayError(f"ledger not found: {path}", field="ledger_path")
    conn = _ro_conn(path)
    try:
        meta = {row["key"]: row["value"] for row in conn.execute("SELECT key,value FROM meta")}
        if meta.get("ledger_format_version") != str(LEDGER_FORMAT_VERSION):
            raise ReplayError(
                f"ledger format version {meta.get('ledger_format_version')!r} != {LEDGER_FORMAT_VERSION}",
                field="meta.ledger_format_version",
            )
        if meta.get("content_digest") != spec.content_digest:
            raise ReplayError(
                "ledger content_digest does not match original declaration",
                field="meta.content_digest",
            )

        events = [dict(row) for row in conn.execute("SELECT * FROM events ORDER BY event_seq")]
        leases_db = {row["lease_id"]: dict(row) for row in conn.execute("SELECT * FROM leases")}
        margins = [dict(row) for row in conn.execute("SELECT * FROM capacity_margins ORDER BY margin_id")]
        results = [dict(row) for row in conn.execute("SELECT * FROM shard_results ORDER BY shard_id,node_id,quota_version")]
        adjudications = [dict(row) for row in conn.execute("SELECT * FROM adjudications ORDER BY event_seq")]
        receipts = [dict(row) for row in conn.execute("SELECT * FROM receipts ORDER BY receipt_id")]

        active: dict[str, dict[str, Any]] = {}
        occupied_slots: set[tuple[str, int, int]] = set()
        held_shards: set[str] = set()
        margin_by_event_node: dict[tuple[int, str], dict[str, Any]] = {}
        for margin in margins:
            margin_by_event_node[(margin["event_seq"], margin["scope_node"])] = margin

        for event in events:
            etype = event["event_type"]
            payload = json.loads(event["payload_json"] or "{}")
            lease_id = event["lease_id"]
            node_id = event["node_id"]

            if etype == "lease_granted":
                if not lease_id or lease_id in active:
                    raise ReplayError("duplicate or invalid grant", field="lease_id", basis=event.get("decision_basis"))
                slot = int(payload["slot_index"])
                quota = int(payload["quota_version"])
                slot_key = (quota, node_id, slot)
                shard_id = event["shard_id"]
                if slot_key in occupied_slots:
                    raise ReplayError("slot already occupied in folded history", field="slot_index", shard_id=shard_id)
                if shard_id in held_shards:
                    raise ReplayError("shard held by two active leases", field="shard_id", shard_id=shard_id)
                occupied_slots.add(slot_key)
                held_shards.add(shard_id)
                active[lease_id] = dict(payload)

            elif etype == "lease_renewed":
                lease = active.get(lease_id)
                if lease is None:
                    raise ReplayError("renewal for non-active folded lease", field="lease_version")
                if int(payload["expected_version"]) != int(lease["lease_version"]):
                    raise ReplayError(
                        "renewal expected version does not fold",
                        field="lease_version",
                        basis=event.get("decision_basis"),
                    )
                lease["lease_version"] = int(payload["new_version"])
                lease["renewal_count"] = int(payload["renewal_count"])
                lease["expires_ms"] = int(payload["expires_ms"])

            elif etype in TERMINAL_EVENTS:
                lease = active.pop(lease_id, None)
                if lease is None and etype != "shard_verified":
                    raise ReplayError(
                        f"{etype} for missing folded active lease",
                        field="state",
                        shard_id=event.get("shard_id"),
                        basis=event.get("decision_basis"),
                    )
                if lease is not None:
                    occupied_slots.discard((int(lease["quota_version"]), lease["node_id"], int(lease["slot_index"])))
                    held_shards.discard(lease["shard_id"])

            if etype in {"lease_granted", "lease_renewed", "shard_verified", "lease_expired", "lease_quota_revoked", "recovery_reclaimed"} and node_id:
                margin = margin_by_event_node.get((event["event_seq"], node_id))
                if margin is None:
                    raise ReplayError("missing capacity margin for event", field="capacity_margin", basis=event.get("decision_basis"))
                node_active = sum(1 for lease in active.values() if lease["node_id"] == node_id)
                if margin["active_leases"] != node_active:
                    raise ReplayError(
                        f"margin active_leases {margin['active_leases']} != folded {node_active}",
                        field="capacity_margin.active_leases",
                    )
                try:
                    node = spec.node(node_id)
                    capacity = node.capacity_slots
                except Exception:
                    capacity = margin["capacity_slots"]
                if margin["capacity_slots"] != capacity:
                    raise ReplayError("capacity slots differ from declaration", field="capacity_slots")
                if margin["available_slots"] != capacity - node_active:
                    raise ReplayError("available capacity margin is inconsistent", field="available_slots")

        active_db = {lid: row for lid, row in leases_db.items() if row["state"] == "active"}
        if set(active_db) != set(active):
            raise ReplayError(
                f"active lease set differs: folded={sorted(active)} ledger={sorted(active_db)}",
                field="active_leases",
            )
        for lease_id in sorted(active, key=lambda lid: active[lid]["shard_id"]):
            folded = active[lease_id]
            row = active_db[lease_id]
            for field_name in ("node_id", "shard_id", "slot_index", "lease_version", "expires_ms", "quota_version"):
                if folded[field_name] != row[field_name]:
                    raise ReplayError(
                        f"lease field {field_name} differs",
                        field=f"lease.{field_name}",
                        shard_id=folded["shard_id"],
                    )

        latest_cursor = conn.execute("SELECT * FROM progress_cursor WHERE cursor_id=1").fetchone()
        if latest_cursor is None or int(latest_cursor["last_event_seq"]) != len(events):
            raise ReplayError("progress cursor does not match event count", field="progress_cursor.last_event_seq")

        for result in sorted(results, key=lambda r: (r["shard_id"], r["node_id"])):
            shard_id = result["shard_id"]
            if "stale" in result.keys() and int(result["quota_version"]) < spec.quota_version and not result["stale"]:
                raise ReplayError(
                    "old quota result was not marked stale",
                    field="shard_results.stale",
                    shard_id=shard_id,
                    basis="quota_version_evolution",
                )
            if int(result["unknown"]) == 1 or result["status"] == "unknown":
                if result["digest"] is not None:
                    raise ReplayError("unknown shard must not carry a digest", field="unknown", shard_id=shard_id)
                continue
            try:
                seq = int(shard_id.rsplit("-", 1)[1])
                content = spec.content_for(spec.content.content_key)
                node = spec.node(result["node_id"])
            except Exception as exc:
                raise ReplayError("result references unknown node/content", field="node_id", shard_id=shard_id) from exc
            expected = compute_shard_proof(node, content, seq)
            if result["digest"] != expected.digest:
                raise ReplayError(
                    "shard digest does not match deterministic recomputation",
                    field="digest",
                    shard_id=shard_id,
                    basis="integrity.compute_shard_proof",
                )
            if result["content_bytes_digest"] != expected.content_bytes_digest:
                raise ReplayError(
                    "content byte digest does not match",
                    field="content_bytes_digest",
                    shard_id=shard_id,
                    basis="integrity.generate_shard_bytes",
                )
            proof_json = result["proof_json"]
            if proof_json:
                proof = json.loads(proof_json)
                if proof.get("digest") != expected.digest:
                    raise ReplayError("stored proof JSON digest differs", field="proof_json.digest", shard_id=shard_id)

        adj_by_lease: dict[str, list[dict[str, Any]]] = {}
        for adjudication in adjudications:
            adj_by_lease.setdefault(adjudication["lease_id"], []).append(adjudication)
            event = next((e for e in events if e["event_seq"] == adjudication["event_seq"]), None)
            if event is None:
                raise ReplayError("adjudication has no event", field="adjudications.event_seq")
            if event["decision_basis"] != adjudication["basis"]:
                raise ReplayError(
                    "adjudication basis differs from event basis",
                    field="decision_basis",
                    shard_id=event.get("shard_id"),
                    basis=adjudication["basis"],
                )
            lease = leases_db.get(adjudication["lease_id"])
            if lease and lease["decision_basis"] != adjudication["basis"]:
                raise ReplayError(
                    "lease terminal basis differs from adjudication basis",
                    field="leases.decision_basis",
                    shard_id=event.get("shard_id"),
                    basis=adjudication["basis"],
                )
            winner = adjudication["winner"]
            if winner == "expiry_reaper" and event["event_type"] != "lease_expired":
                raise ReplayError("expiry winner is not uniquely represented", field="winner", basis=adjudication["basis"])
            if winner == "quota_revision":
                if event["event_type"] != "lease_quota_revoked":
                    raise ReplayError("quota winner event type mismatch", field="winner", basis=adjudication["basis"])
                payload = json.loads(event["payload_json"])
                if not (payload.get("basis") or "").startswith(("lease_quota_version", "node_absent")):
                    raise ReplayError("quota adjudication lacks version/node basis", field="basis", basis=adjudication["basis"])
        for lease_id, items in adj_by_lease.items():
            terminal_winners = [item for item in items if item["winner"] in {"expiry_reaper", "quota_revision", "startup_recovery", "completed"}]
            if len(terminal_winners) > 1:
                raise ReplayError("lease has multiple terminal attribution winners", field="winner", basis=canonical_dumps([i["winner"] for i in items]))

        adopted = sum(1 for r in receipts if r["adopted"] == 1)
        rejected = len(receipts) - adopted
        verified = sorted(r["shard_id"] for r in results if r["status"] == "verified")
        unknown = sorted(r["shard_id"] for r in results if int(r["unknown"]) == 1)
        active_summary = [
            {
                "lease_id": lid,
                "node_id": row["node_id"],
                "shard_id": row["shard_id"],
                "slot_index": row["slot_index"],
                "lease_version": row["lease_version"],
                "expires_ms": row["expires_ms"],
            }
            for lid, row in sorted(active_db.items())
        ]
        return ReplayReport(
            ok=True,
            ledger_path=str(path),
            event_count=len(events),
            lease_count=len(leases_db),
            active_leases=active_summary,
            verified_shards=verified,
            unknown_shards=unknown,
            adjudications=len(adjudications),
            receipts_adopted=adopted,
            receipts_rejected=rejected,
        )
    finally:
        conn.close()
