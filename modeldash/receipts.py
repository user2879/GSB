"""Receipt trust decay and per-node untrust scoring.

All scores are pure functions of persisted receipts and the current config
version.  Dictionary iteration is always replaced by explicit sorting so the
result is independent of task execution order.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from .util import digest_json


CURRENT = "CURRENT"
VERIFIED = "VERIFIED_UNDER_CURRENT"
MISMATCH = "UNTRUSTED_MISMATCH"
DEPENDENCY = "UNTRUSTED_DEPENDENCY"
UNVERIFIED = "UNTRUSTED_UNVERIFIED"
INCONSISTENT = "UNTRUSTED_INCONSISTENT"
UNTRUSTED_STATUSES = {MISMATCH, DEPENDENCY, UNVERIFIED, INCONSISTENT}


@dataclass
class Receipt:
    receipt_key: str
    node_id: str
    task_id: str
    config_version: int
    shard_index: int
    shard_count: int
    item_keys: tuple[str, ...]
    input_digest: str
    output_summary: str
    generated_version: int
    attempt: int = 1
    decay_count: int = 0
    untrust_score: int = 0
    status: str = CURRENT
    reason: str = ""
    dependency_refs: tuple[str, ...] = ()
    is_stale: bool = False

    def comparison_key(self) -> tuple[str, int, str, int]:
        return (self.node_id, self.shard_index, self.input_digest, self.shard_count)

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_key": self.receipt_key,
            "node_id": self.node_id,
            "task_id": self.task_id,
            "config_version": self.config_version,
            "shard_index": self.shard_index,
            "shard_count": self.shard_count,
            "item_keys": list(self.item_keys),
            "input_digest": self.input_digest,
            "output_summary": self.output_summary,
            "generated_version": self.generated_version,
            "attempt": self.attempt,
            "decay_count": self.decay_count,
            "untrust_score": self.untrust_score,
            "status": self.status,
            "reason": self.reason,
            "dependency_refs": list(self.dependency_refs),
            "is_stale": self.is_stale,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "Receipt":
        return cls(
            receipt_key=str(value["receipt_key"]),
            node_id=str(value["node_id"]),
            task_id=str(value["task_id"]),
            config_version=int(value["config_version"]),
            shard_index=int(value["shard_index"]),
            shard_count=int(value["shard_count"]),
            item_keys=tuple(str(item) for item in value.get("item_keys", [])),
            input_digest=str(value["input_digest"]),
            output_summary=str(value["output_summary"]),
            generated_version=int(value["generated_version"]),
            attempt=int(value.get("attempt", 1)),
            decay_count=int(value.get("decay_count", 0)),
            untrust_score=int(value.get("untrust_score", 0)),
            status=str(value.get("status", CURRENT)),
            reason=str(value.get("reason", "")),
            dependency_refs=tuple(str(item) for item in value.get("dependency_refs", [])),
            is_stale=bool(value.get("is_stale", False)),
        )


def receipt_key(node_id: str, task_id: str, config_version: int, shard_index: int, input_digest: str) -> str:
    return f"{node_id}|{task_id}|v{config_version}|s{shard_index}|{input_digest[:16]}"


@dataclass
class ReceiptBook:
    receipts: dict[str, Receipt] = field(default_factory=dict)
    conflicts: list[dict[str, Any]] = field(default_factory=list)

    def add(self, receipt: Receipt) -> None:
        previous = self.receipts.get(receipt.receipt_key)
        if previous is not None:
            if previous.output_summary != receipt.output_summary:
                self.conflicts.append(
                    {
                        "receipt_key": receipt.receipt_key,
                        "first_output": previous.output_summary,
                        "second_output": receipt.output_summary,
                    }
                )
            if receipt.attempt < previous.attempt:
                return
        self.receipts[receipt.receipt_key] = receipt

    def values(self) -> list[Receipt]:
        return [self.receipts[key] for key in sorted(self.receipts)]

    def add_result(
        self,
        *,
        node_id: str,
        task_id: str,
        config_version: int,
        shard_index: int,
        shard_count: int,
        item_keys: Iterable[str],
        input_digest: str,
        output_summary: str,
        attempt: int,
    ) -> Receipt:
        key = receipt_key(node_id, task_id, config_version, shard_index, input_digest)
        receipt = Receipt(
            receipt_key=key,
            node_id=node_id,
            task_id=task_id,
            config_version=config_version,
            shard_index=shard_index,
            shard_count=shard_count,
            item_keys=tuple(sorted(item_keys)),
            input_digest=input_digest,
            output_summary=output_summary,
            generated_version=config_version,
            attempt=attempt,
        )
        self.add(receipt)
        return self.receipts[key]

    def audit(self, current_version: int) -> list[Receipt]:
        """Return audited receipt copies for the current configuration view.

        Rule R5 (explicit, integer-only):
        * a pre-current receipt is a direct mismatch when a current-version
          receipt for the same node/shard/input exists and its output differs;
        * mismatch references node + item-key dependencies.  Older receipts on
          the same node that share an item key become dependency-untrusted;
        * a pre-current receipt without a current recomputation is conservatively
          unverified;
        * decay_count = version_gap + 2*impact_class and
          untrust_score = 100*impact_class + 10*version_gap + 3*decay_count.
        """

        values = self.values()
        by_comparison: dict[tuple[str, int, str, int], list[Receipt]] = {}
        current_values: list[Receipt] = []
        for receipt in values:
            by_comparison.setdefault(receipt.comparison_key(), []).append(receipt)
            if receipt.generated_version == current_version:
                current_values.append(receipt)

        current_outputs: dict[tuple[str, int, str, int], dict[str, list[str]]] = {}
        for receipt in current_values:
            current_outputs.setdefault(receipt.comparison_key(), {}).setdefault(receipt.output_summary, []).append(
                receipt.receipt_key
            )

        direct_mismatches: dict[str, Receipt] = {}
        inconsistent_current: set[str] = set()
        for comparison, outputs in current_outputs.items():
            if len(outputs) > 1:
                for keys in outputs.values():
                    inconsistent_current.update(keys)

        for receipt in values:
            if receipt.generated_version >= current_version:
                continue
            outputs = current_outputs.get(receipt.comparison_key(), {})
            if not outputs:
                continue
            if receipt.output_summary not in outputs:
                direct_mismatches[receipt.receipt_key] = receipt

        mismatch_node_items: dict[str, set[str]] = {}
        mismatch_by_node: dict[str, set[str]] = {}
        for mismatch in direct_mismatches.values():
            mismatch_by_node.setdefault(mismatch.node_id, set()).add(mismatch.receipt_key)
            mismatch_node_items.setdefault(mismatch.node_id, set()).update(mismatch.item_keys)

        audited: list[Receipt] = []
        for receipt in values:
            if receipt.generated_version == current_version:
                if receipt.receipt_key in inconsistent_current:
                    audited.append(
                        replace(
                            receipt,
                            status=INCONSISTENT,
                            reason="current_version_receipts_disagree",
                            decay_count=2,
                            untrust_score=220,
                        )
                    )
                else:
                    audited.append(replace(receipt, status=CURRENT, reason="generated_for_current_view", decay_count=0, untrust_score=0))
                continue

            version_gap = max(0, current_version - receipt.generated_version)
            outputs = current_outputs.get(receipt.comparison_key(), {})
            if receipt.receipt_key in direct_mismatches:
                status = MISMATCH
                reason = "new_version_output_digest_differs"
                impact = 2
                refs = (f"node:{receipt.node_id}", *(f"item:{key}" for key in receipt.item_keys))
            elif receipt.node_id in mismatch_node_items and set(receipt.item_keys) & mismatch_node_items[receipt.node_id]:
                status = DEPENDENCY
                reason = "shares_node_item_dependency_with_mismatch"
                impact = 1
                shared = sorted(set(receipt.item_keys) & mismatch_node_items[receipt.node_id])
                refs = tuple(sorted(mismatch_by_node[receipt.node_id])) + tuple(f"item:{key}" for key in shared)
            elif not outputs:
                status = UNVERIFIED
                reason = "no_current_version_recomputation"
                impact = 1
                refs = ()
            else:
                status = VERIFIED
                reason = "current_version_output_digest_matches"
                impact = 0
                refs = tuple(sorted(outputs[next(iter(outputs))]))

            decay_count = version_gap + 2 * impact
            untrust_score = 100 * impact + 10 * version_gap + 3 * decay_count
            audited.append(
                replace(
                    receipt,
                    status=status,
                    reason=reason,
                    decay_count=decay_count,
                    untrust_score=untrust_score,
                    dependency_refs=tuple(sorted(refs)),
                )
            )
        audited.sort(key=lambda item: item.receipt_key)
        return audited

    def replace_with_audited(self, current_version: int) -> list[Receipt]:
        audited = self.audit(current_version)
        self.receipts = {item.receipt_key: item for item in audited}
        return audited


def build_node_untrust(
    audited_receipts: Iterable[Receipt], node_ids: Iterable[str], current_version: int
) -> list[dict[str, Any]]:
    node_set = sorted(set(node_ids))
    grouped: dict[str, list[Receipt]] = {node_id: [] for node_id in node_set}
    for receipt in audited_receipts:
        grouped.setdefault(receipt.node_id, []).append(receipt)

    rows: list[dict[str, Any]] = []
    for node_id in sorted(grouped):
        receipts = sorted(grouped[node_id], key=lambda item: item.receipt_key)
        untrusted = [receipt for receipt in receipts if receipt.status in UNTRUSTED_STATUSES]
        score = sum(receipt.untrust_score for receipt in untrusted)
        max_decay = max((receipt.decay_count for receipt in receipts), default=0)
        status_counts: dict[str, int] = {}
        for receipt in receipts:
            status_counts[receipt.status] = status_counts.get(receipt.status, 0) + 1
        rows.append(
            {
                "node_id": node_id,
                "score": int(score),
                "rank": 0,
                "receipt_count": len(receipts),
                "untrusted_receipts": len(untrusted),
                "max_decay": int(max_decay),
                "status_counts": dict(sorted(status_counts.items())),
                "untrusted_receipt_keys": [receipt.receipt_key for receipt in untrusted],
                "current_version": int(current_version),
                "digest": digest_json(
                    {
                        "node_id": node_id,
                        "score": int(score),
                        "untrusted": [receipt.receipt_key for receipt in untrusted],
                    }
                ),
            }
        )
    rows.sort(key=lambda item: (-int(item["score"]), str(item["node_id"])))
    for index, row in enumerate(rows, start=1):
        row["rank"] = index
    return rows


def receipt_digest(receipts: Iterable[Receipt]) -> str:
    return digest_json([receipt.to_dict() for receipt in sorted(receipts, key=lambda item: item.receipt_key)])
