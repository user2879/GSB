from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .canonical import sha256_value
from .errors import LabError
from .receipts import Receipt, decode_receipts


def _as_receipts(items: Iterable[Any]) -> list[Receipt]:
    values = list(items)
    if not values:
        raise LabError("E_FORMAT", "at least one receipt is required for merge")
    if all(isinstance(item, Receipt) for item in values):
        return list(values)  # type: ignore[arg-type]
    return decode_receipts(values)


@dataclass(frozen=True)
class Candidate:
    receipt: Receipt
    receipt_identity: str
    merge_seq: int
    param_version: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_id": self.receipt_identity,
            "format": self.receipt.format,
            "merge_seq": self.merge_seq,
            "param_version": self.param_version,
            "destination": self.receipt.destination,
            "output_digest": self.receipt.output_digest,
            "sample_ids": list(self.receipt.sample_ids),
            "receipt": self.receipt.to_dict(),
        }


def _normalize_samples(receipt: Receipt, expected_sample_ids: Iterable[str] | None) -> tuple[str, ...]:
    if receipt.sample_ids:
        return tuple(sorted(receipt.sample_ids))
    if expected_sample_ids is None:
        return ()
    return tuple(sorted(set(expected_sample_ids)))


def merge_receipts(
    receipts: Iterable[Any],
    *,
    expected_sample_ids: Iterable[str] | None = None,
    param_versions: Mapping[str, int] | None = None,
    rule_id: str | None = None,
) -> dict[str, Any]:
    """Merge historical and hot-updated receipts for one batch.

    The candidate list is displayed using (merge_seq ascending, parameter
    version descending, receipt digest ascending). Adoption takes the highest
    merge sequence, then the highest parameter version in that sequence, then
    the lexicographically smallest receipt digest as a final stable tie-break.
    No input iteration or arrival order can affect the result.
    """
    items = _as_receipts(receipts)
    batch_ids = {item.batch_id for item in items}
    if len(batch_ids) != 1:
        raise LabError("E_FORMAT", "all receipts in a merge must share one batch_id", field="batch_id")
    batch_id = items[0].batch_id
    seq_keys: set[tuple[str, int]] = set()
    for item in items:
        key = (item.batch_id, item.effective_merge_seq)
        if key in seq_keys:
            raise LabError(
                "E_FORMAT",
                "duplicate (batch_id, merge_seq) in merge input",
                field="merge_seq",
                batch_id=batch_id,
                details={"merge_seq": item.effective_merge_seq},
            )
        seq_keys.add(key)

    version_map = dict(param_versions or {})
    candidates: list[Candidate] = []
    for item in items:
        version = item.backend_param_version
        if version is None and item.backend_param_digest is not None:
            version = version_map.get(item.backend_param_digest, 0)
        candidates.append(
            Candidate(
                receipt=item,
                receipt_identity=item.identity_digest(),
                merge_seq=item.effective_merge_seq,
                param_version=int(version or 0),
            )
        )

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            candidate.merge_seq,
            -candidate.param_version,
            candidate.receipt_identity,
        ),
    )
    highest_seq = max(candidate.merge_seq for candidate in ordered)
    seq_group = [candidate for candidate in ordered if candidate.merge_seq == highest_seq]
    highest_version = max(candidate.param_version for candidate in seq_group)
    version_group = [candidate for candidate in seq_group if candidate.param_version == highest_version]
    adopted = min(version_group, key=lambda candidate: candidate.receipt_identity)

    adopted_samples = set(_normalize_samples(adopted.receipt, expected_sample_ids))
    rejected: list[dict[str, Any]] = []
    for candidate in ordered:
        if candidate is adopted:
            continue
        if candidate.merge_seq < adopted.merge_seq:
            reason = "superseded_by_higher_merge_seq"
        elif candidate.param_version < adopted.param_version:
            reason = "lower_param_version"
        else:
            reason = "stable_digest_tie_break"
        candidate_samples = set(_normalize_samples(candidate.receipt, expected_sample_ids))
        discrepancies = []
        for sample_id in sorted(candidate_samples ^ adopted_samples):
            discrepancies.append(
                {
                    "sample_id": sample_id,
                    "in_rejected": sample_id in candidate_samples,
                    "in_adopted": sample_id in adopted_samples,
                    "kind": "rejected_only" if sample_id in candidate_samples else "adopted_only",
                }
            )
        rejected.append(
            {
                **candidate.to_dict(),
                "rejection_reason": reason,
                "destination_mismatch": candidate.receipt.destination != adopted.receipt.destination,
                "output_digest_mismatch": candidate.receipt.output_digest != adopted.receipt.output_digest,
                "sample_discrepancies": discrepancies,
                "unknown_fields": sorted(candidate.receipt.extras),
            }
        )

    conclusion = {
        "algorithm": "gray-merge-lab-merge-v1",
        "batch_id": batch_id,
        "rule_id": rule_id,
        "destination": adopted.receipt.destination,
        "sample_ids": sorted(adopted_samples),
        "output_digest": adopted.receipt.output_digest,
        "backend_param_digest": adopted.receipt.backend_param_digest,
        "backend_param_version": adopted.param_version,
        "merge_seq": adopted.merge_seq,
        "adopted_receipt": {
            "receipt_id": adopted.receipt_identity,
            "format": adopted.receipt.format,
            "unknown_fields": sorted(adopted.receipt.extras),
        },
        "comparison_order": [candidate.to_dict() for candidate in ordered],
        "rejected": rejected,
        "arrival_order_independent": True,
        "decision_rule": (
            "highest_merge_seq; then highest_backend_param_version; "
            "then lexicographically_smallest_receipt_digest"
        ),
    }
    conclusion["conclusion_digest"] = sha256_value(conclusion)
    return conclusion


def receipt_discrepancy_report(conclusion: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rejected in conclusion.get("rejected", []):
        for item in rejected.get("sample_discrepancies", []):
            rows.append(
                {
                    "rejected_receipt_id": rejected["receipt_id"],
                    "rejection_reason": rejected["rejection_reason"],
                    **item,
                }
            )
    return sorted(
        rows,
        key=lambda row: (row["sample_id"], row["rejected_receipt_id"], row["kind"]),
    )

