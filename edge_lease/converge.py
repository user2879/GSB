"""Receipt convergence and byte-exact expected digest reconciliation.

Expected digests are recomputed from the manifest version and derivation graph
(the same fixed-point contract the nodes use) and compared byte-for-byte with
the final successful receipt of every plan item.  Missing receipts, digest
mismatches and quantization-grid version mismatches all raise
:class:`ConvergeError` with a deterministically ordered deviation list.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .errors import ConvergeError
from .models import PlanItem, Receipt


@dataclass
class ConvergenceRow:
    item_id: str
    node_id: str
    version: int
    path: str
    expected_digest: int
    actual_digest: Optional[int]
    expected_shard_digest: int
    actual_shard_digest: Optional[int]
    quant_version: str
    actual_quant_version: Optional[str]
    error_bound: Dict[str, int]
    ok: bool
    diff_fields: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "item_id": self.item_id,
            "node_id": self.node_id,
            "version": self.version,
            "path": self.path,
            "expected_digest": self.expected_digest,
            "actual_digest": self.actual_digest,
            "expected_shard_digest": self.expected_shard_digest,
            "actual_shard_digest": self.actual_shard_digest,
            "quant_version": self.quant_version,
            "actual_quant_version": self.actual_quant_version,
            "error_bound": self.error_bound,
            "ok": self.ok,
            "diff_fields": self.diff_fields,
        }


def _final_ok_receipt(receipts: List[Receipt]) -> Optional[Receipt]:
    ok = [receipt for receipt in receipts if receipt.status == "ok"]
    return ok[-1] if ok else None


def reconcile(
    items: List[PlanItem],
    receipts_by_item: Dict[str, List[Receipt]],
) -> Tuple[List[ConvergenceRow], List[Dict[str, Any]], Dict[str, Dict[str, int]]]:
    """Return ``(rows, deviations, per-node error summary)`` or raise.

    Raises :class:`ConvergeError` when any item lacks an OK receipt, any digest
    differs or the quantization grid version embedded in a receipt differs.
    """
    rows: List[ConvergenceRow] = []
    deviations: List[Dict[str, Any]] = []
    node_errors: Dict[str, Dict[str, int]] = {}
    for item in sorted(items, key=lambda plan: plan.item_id):
        receipt = _final_ok_receipt(receipts_by_item.get(item.item_id, []))
        diff_fields: List[str] = []
        actual_digest = receipt.derived_digest if receipt else None
        actual_shard = receipt.shard_hit_digest if receipt else None
        actual_quant = receipt.quant_version if receipt else None
        if receipt is None:
            diff_fields.append("receipt_missing")
        else:
            if str(actual_quant) != item.quant_version:
                diff_fields.append("quant_version")
            if int(actual_digest) != int(item.expected_digest):
                diff_fields.append("derived_digest")
            if int(actual_shard) != int(item.shard_hit_digest):
                diff_fields.append("shard_hit_digest")
        ok = not diff_fields
        row = ConvergenceRow(
            item_id=item.item_id,
            node_id=item.node_id,
            version=item.version,
            path=item.path,
            expected_digest=item.expected_digest,
            actual_digest=actual_digest,
            expected_shard_digest=item.shard_hit_digest,
            actual_shard_digest=actual_shard,
            quant_version=item.quant_version,
            actual_quant_version=actual_quant,
            error_bound=dict(item.expected_error),
            ok=ok,
            diff_fields=diff_fields,
        )
        rows.append(row)
        node_errors.setdefault(
            item.node_id, {"single": 0, "propagated": 0, "bound": 0}
        )
        for key in ("single", "propagated", "bound"):
            node_errors[item.node_id][key] += int(item.expected_error[key])
        if not ok:
            deviations.append(
                {
                    "item_id": item.item_id,
                    "node_id": item.node_id,
                    "version": item.version,
                    "path": item.path,
                    "expected_digest": item.expected_digest,
                    "actual_digest": actual_digest,
                    "expected_shard_digest": item.shard_hit_digest,
                    "actual_shard_digest": actual_shard,
                    "expected_quant_version": item.quant_version,
                    "actual_quant_version": actual_quant,
                    "diff_fields": diff_fields,
                }
            )
    if deviations:
        raise ConvergeError(
            f"{len(deviations)} plan item(s) failed convergence",
            "converge.reconcile",
            {"deviations": deviations},
        )
    return rows, deviations, node_errors


def authoritative_recompute(
    items: List[PlanItem], receipts: List[Receipt]
) -> Tuple[List[ConvergenceRow], List[Dict[str, Any]]]:
    """Ledger-authority variant used by replay; same ordering and field rules."""
    grouped: Dict[str, List[Receipt]] = {}
    for receipt in receipts:
        grouped.setdefault(receipt.item_id, []).append(receipt)
    rows, deviations, _summary = reconcile(items, grouped)
    return rows, deviations
