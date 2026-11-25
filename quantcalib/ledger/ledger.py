"""Authoritative receipt ledger adjudication.

Multiple receipt versions may describe the same layer. The authority decision
is driven by the parameter version sequence and the covered duplicate
references, while the layer invalidation boundary (handled by the cache layer)
depends on the group scale digest. Budget deduction order must follow the
order produced here, which is stable: layer name lexicographic, then the
adjudicated winner.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

from quantcalib.models import UnifiedReceipt

_VERSION_RANK = {"q1": 1, "q2": 2, "q3": 3}


@dataclass
class LedgerEntry:
    layer: str
    authority_source: str
    coverage_permille: int
    param_version: int | None
    group_scale_digest: str | None
    duplicate_refs: list[str]
    weight_snapshot: str | None
    rationale: str
    candidates: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _winner_key(receipt: UnifiedReceipt) -> tuple:
    # Higher content/param version wins first; q3 coverage references beat q2;
    # richer format, then higher coverage, then lexicographic source are stable
    # tie breakers that never depend on insertion/thread order.
    return (
        receipt.param_version if receipt.param_version is not None else -1,
        len(receipt.duplicate_refs),
        _VERSION_RANK[receipt.source_version],
        receipt.coverage_permille,
    )


def adjudicate(receipts: list[UnifiedReceipt]) -> list[LedgerEntry]:
    grouped: dict[str, list[UnifiedReceipt]] = {}
    for receipt in receipts:
        grouped.setdefault(receipt.layer, []).append(receipt)

    entries: list[LedgerEntry] = []
    for layer in sorted(grouped):
        candidates = sorted(
            grouped[layer],
            key=lambda r: (_winner_key(r),),
            reverse=True,
        )
        winner = candidates[0]
        runner = candidates[1] if len(candidates) > 1 else None
        if runner is None:
            rationale = "single candidate"
        elif (winner.param_version or -1) > (runner.param_version or -1):
            rationale = "higher param_version"
        elif len(winner.duplicate_refs) > len(runner.duplicate_refs):
            rationale = "more covered duplicate calibration references"
        elif _VERSION_RANK[winner.source_version] > _VERSION_RANK[runner.source_version]:
            rationale = "richer receipt format on tie"
        else:
            rationale = "higher coverage permille on full tie"
        entries.append(LedgerEntry(
            layer=layer,
            authority_source=winner.source_version,
            coverage_permille=winner.coverage_permille,
            param_version=winner.param_version,
            group_scale_digest=winner.group_scale_digest,
            duplicate_refs=list(winner.duplicate_refs),
            weight_snapshot=winner.weight_snapshot,
            rationale=rationale,
            candidates=[f"{r.source_version}(pv={r.param_version})" for r in candidates],
        ))
    return entries


def ledger_fingerprint(entries: list[LedgerEntry]) -> str:
    from quantcalib.fingerprint import fingerprint_json
    return fingerprint_json([e.to_dict() for e in entries])
