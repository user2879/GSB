"""Cache key matching, dependency propagation and degraded receipt policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from .models import Artifact, ReceiptRecord


@dataclass(frozen=True)
class ReuseDecision:
    reuse: bool
    reason: str
    digest_match: Optional[bool]
    degraded: bool
    degradation_basis: Optional[str]
    receipt_version: Optional[int]


class CacheIndex:
    """Baseline cache receipts keyed by artifact id."""

    def __init__(self, receipts: Tuple[ReceiptRecord, ...], degraded_policy: str):
        self.receipts = {record.artifact_id: record for record in receipts}
        self.degraded_policy = degraded_policy

    def receipt_for(self, artifact_id: str) -> Optional[ReceiptRecord]:
        return self.receipts.get(artifact_id)

    def decide(
        self,
        artifact: Artifact,
        expected_param_version: int,
        sample_fingerprint_changed: bool,
        upstream_reused: bool,
    ) -> ReuseDecision:
        receipt = self.receipts.get(artifact.artifact_id)
        if receipt is None:
            return ReuseDecision(False, "no baseline receipt", None, False, None, None)
        if not upstream_reused:
            receipt_version = receipt.source_version if receipt is not None else None
            degraded = receipt is not None and receipt.source_version < 3
            basis = (
                f"source=v{receipt.source_version}; digest absent; upstream recompute prevents reuse"
                if degraded
                else None
            )
            return ReuseDecision(
                False,
                "upstream artifact was recomputed",
                None,
                degraded,
                basis,
                receipt_version,
            )
        if receipt.source_version == 3:
            digest_match = receipt.digest == artifact.digest
            if digest_match and receipt.param_version == expected_param_version:
                return ReuseDecision(
                    True,
                    "v3 digest and param version match",
                    True,
                    False,
                    None,
                    3,
                )
            reason = "v3 digest mismatch" if not digest_match else "v3 param version mismatch"
            return ReuseDecision(False, reason, digest_match, False, None, 3)
        degraded = True
        basis = (
            f"source=v{receipt.source_version}; digest absent; policy={self.degraded_policy}; "
            f"receipt_param_version={receipt.param_version}; expected={expected_param_version}; "
            f"sample_fingerprint_changed={sample_fingerprint_changed}"
        )
        if self.degraded_policy == "strict":
            return ReuseDecision(
                False,
                "degraded receipt cannot prove content identity under strict policy",
                None,
                True,
                basis,
                receipt.source_version,
            )
        if sample_fingerprint_changed:
            return ReuseDecision(
                False,
                "degraded receipt cannot prove sample-derived content identity",
                None,
                True,
                basis,
                receipt.source_version,
            )
        if receipt.param_version is None or receipt.param_version != expected_param_version:
            return ReuseDecision(
                False,
                "degraded param version differs",
                None,
                True,
                basis,
                receipt.source_version,
            )
        return ReuseDecision(
            True,
            "degraded conservative param-version comparison allows reuse",
            None,
            True,
            basis,
            receipt.source_version,
        )


def downstream_closure(artifacts: Tuple[Artifact, ...], roots: Set[str]) -> Set[str]:
    """Return roots plus every artifact that transitively depends on them."""

    consumers: Dict[str, List[str]] = {}
    for artifact in artifacts:
        for upstream_id in artifact.upstream:
            consumers.setdefault(upstream_id, []).append(artifact.artifact_id)
    dirty = set(roots)
    stack = list(roots)
    while stack:
        current = stack.pop()
        for consumer in consumers.get(current, ()):
            if consumer not in dirty:
                dirty.add(consumer)
                stack.append(consumer)
    return dirty
