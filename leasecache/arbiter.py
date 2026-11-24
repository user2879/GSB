from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateVerdict:
    node_id: str
    digest: str
    digest_match: bool
    free_slots: int
    active_leases: int
    lease_limit: int
    eligible: bool
    selected: bool
    veto_reason: str | None
    veto_detail: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "digest": self.digest,
            "digest_match": self.digest_match,
            "free_slots": self.free_slots,
            "active_leases": self.active_leases,
            "lease_limit": self.lease_limit,
            "eligible": self.eligible,
            "selected": self.selected,
            "veto_reason": self.veto_reason,
            "veto_detail": self.veto_detail,
        }


@dataclass(frozen=True)
class ArbitrationResult:
    artifact_key: str
    expected_digest: str
    selected_node: str | None
    verdicts: tuple[CandidateVerdict, ...]

    @property
    def queue_required(self) -> bool:
        return self.selected_node is None

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_key": self.artifact_key,
            "expected_digest": self.expected_digest,
            "selected_node": self.selected_node,
            "queue_required": self.queue_required,
            "verdicts": [verdict.to_dict() for verdict in self.verdicts],
        }


def arbitrate(
    artifact_key: str,
    expected_digest: str,
    node_rows: list[dict[str, Any]],
    per_node_digests: dict[str, str],
    *,
    needed_slots: int = 1,
) -> ArbitrationResult:
    """Choose one service node with a total, stable ordering.

    Rules, in order: exact content digest match, more free capacity slots,
    lexicographically smaller node id.  Nodes that cannot take a new lease are
    never bypassed through another ordering key.
    """

    rows = sorted(node_rows, key=lambda row: row["node_id"])
    scored: list[tuple[tuple[int, int, str], dict[str, Any], bool, str | None, str]] = []
    for row in rows:
        node_id = row["node_id"]
        digest = per_node_digests.get(node_id, "")
        digest_match = digest == expected_digest
        free_slots = int(row["free_slots"])
        active_leases = int(row["active_leases"])
        lease_limit = int(row["lease_limit"])
        eligible = True
        veto_reason: str | None = None
        detail = ""
        if not digest_match:
            eligible = False
            veto_reason = "DIGEST_MISMATCH"
            detail = f"node digest {digest[:12]} != expected {expected_digest[:12]}"
        elif active_leases >= lease_limit:
            eligible = False
            veto_reason = "E_LEASE_LIMIT"
            detail = f"active leases {active_leases} >= limit {lease_limit}"
        elif free_slots < needed_slots:
            eligible = False
            veto_reason = "E_CAPACITY"
            detail = f"free slots {free_slots} < required {needed_slots}"
        sort_key = (-int(digest_match), -free_slots, node_id)
        scored.append((sort_key, row, eligible, veto_reason, detail))

    eligible_rows = sorted(
        [item for item in scored if item[2]], key=lambda item: item[0]
    )
    selected_node = eligible_rows[0][1]["node_id"] if eligible_rows else None
    selected_sort = eligible_rows[0][0] if eligible_rows else None
    verdicts: list[CandidateVerdict] = []
    for sort_key, row, eligible, veto_reason, detail in scored:
        node_id = row["node_id"]
        selected = node_id == selected_node
        if not selected and eligible:
            if selected_sort is not None and sort_key[1] > selected_sort[1]:
                veto_reason = "FEWER_FREE_SLOTS"
                detail = f"free slots {row['free_slots']} < selected {eligible_rows[0][1]['free_slots']}"
            else:
                veto_reason = "NODE_ID_ORDER"
                detail = f"node id {node_id} sorts after {selected_node}"
        verdicts.append(
            CandidateVerdict(
                node_id=node_id,
                digest=per_node_digests.get(node_id, ""),
                digest_match=per_node_digests.get(node_id, "") == expected_digest,
                free_slots=int(row["free_slots"]),
                active_leases=int(row["active_leases"]),
                lease_limit=int(row["lease_limit"]),
                eligible=eligible,
                selected=selected,
                veto_reason=veto_reason,
                veto_detail=detail,
            )
        )
    return ArbitrationResult(
        artifact_key=artifact_key,
        expected_digest=expected_digest,
        selected_node=selected_node,
        verdicts=tuple(verdicts),
    )
