"""Receipt convergence against an authoritative recomputation."""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

from ..config.compiler import CompiledVersion
from ..config.manifest import ManifestVersion, Registry
from ..config.planner import DistributionPlan
from ..errors import E_CONVERGE, ErrorDetail, PlatformError
from ..util.canonical import canonical_bytes
from .scheduler import SchedulerResult
from .summary import node_summary, summary_bytes


@dataclass
class NodeConvergence:
    node_id: str
    request_id: str
    expected_version: int
    observed_version: int | None
    match: bool
    reason: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "request_id": self.request_id,
            "expected_version": self.expected_version,
            "observed_version": self.observed_version,
            "match": self.match,
            "reason": self.reason,
        }


@dataclass
class ConvergenceReport:
    version: int
    converged: bool
    nodes: list[NodeConvergence]
    expected_summaries: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "converged": self.converged,
            "nodes": [n.to_dict() for n in self.nodes],
        }


def expected_node_summary(
    node_id: str,
    request_keys: list[str],
    manifest_version: ManifestVersion,
    compiled: CompiledVersion,
    registry: Registry,
) -> dict[str, Any]:
    spec = registry.nodes[node_id]
    item_reals = {key: compiled.values[key].real for key in request_keys}
    return node_summary(
        node_id=node_id,
        version=manifest_version.version,
        aggregate=spec.aggregate,
        sample=list(spec.sample),
        item_reals=item_reals,
        ordered_keys=sorted(request_keys),
    )


def evaluate_convergence(
    plan: DistributionPlan,
    manifest_version: ManifestVersion,
    compiled: CompiledVersion,
    registry: Registry,
    scheduler_result: SchedulerResult,
) -> ConvergenceReport:
    receipts_by_node = scheduler_result.receipts
    nodes: list[NodeConvergence] = []
    expected_summaries: dict[str, dict[str, Any]] = {}
    details: list[ErrorDetail] = []

    for request in plan.requests:
        node_id = request.node_id
        expected = expected_node_summary(
            node_id=node_id,
            request_keys=list(request.keys),
            manifest_version=manifest_version,
            compiled=compiled,
            registry=registry,
        )
        expected_summaries[node_id] = expected
        receipt = receipts_by_node.get(node_id)

        if receipt is None:
            reason = "no receipt received (all attempts exhausted)"
            nodes.append(
                NodeConvergence(node_id, request.request_id, request.version, None, False, reason)
            )
            details.append(ErrorDetail(reason=reason, key=node_id, source="convergence"))
            continue

        observed_version = receipt.get("version")
        if observed_version != request.version:
            reason = f"stale receipt version {observed_version} != expected {request.version}"
            nodes.append(
                NodeConvergence(node_id, request.request_id, request.version, observed_version, False, reason)
            )
            details.append(ErrorDetail(reason=reason, key=node_id, source="convergence"))
            continue

        observed = receipt.get("summary", {})
        if canonical_bytes(observed) != canonical_bytes(expected):
            reason = "summary bytes differ from authoritative recomputation"
            nodes.append(
                NodeConvergence(node_id, request.request_id, request.version, observed_version, False, reason)
            )
            details.append(ErrorDetail(reason=reason, key=node_id, source="convergence"))
            continue

        # Explicit byte-level check of the node-provided encoding too.
        if receipt.get("summary_bytes") and receipt["summary_bytes"].encode("utf-8") != summary_bytes(expected):
            reason = "summary_bytes encoding differs"
            nodes.append(
                NodeConvergence(node_id, request.request_id, request.version, observed_version, False, reason)
            )
            details.append(ErrorDetail(reason=reason, key=node_id, source="convergence"))
            continue

        nodes.append(
            NodeConvergence(node_id, request.request_id, request.version, observed_version, True, None)
        )

    converged = all(node.match for node in nodes)
    report = ConvergenceReport(
        version=plan.version,
        converged=converged,
        nodes=sorted(nodes, key=lambda n: n.node_id),
        expected_summaries=expected_summaries,
    )
    if not converged:
        raise PlatformError(
            E_CONVERGE,
            "distribution did not converge on the authoritative summaries",
            details,
        )
    return report
