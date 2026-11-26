from __future__ import annotations

from typing import Any, Mapping, Sequence


def receipt_node(request_id: str, node_id: str) -> str:
    return f"receipt:{request_id}:{node_id}"


def metric_node(request_id: str, node_id: str) -> str:
    return f"metric:{request_id}:{node_id}"


def accum_node(request_id: str, node_id: str) -> str:
    return f"accum:{request_id}:{node_id}"


def e2e_node(request_id: str) -> str:
    return f"e2e:{request_id}"


def attribution_node(request_id: str) -> str:
    return f"attribution:{request_id}"


def plan_invalidation(
    *,
    request_ids: Sequence[str],
    chain_node_ids: Sequence[str],
    stored_receipt_versions: Mapping[tuple[str, str], Mapping[str, Any]],
    current_config_versions: Mapping[str, int],
    current_scale_factors: Mapping[str, int],
    scale_caliber_changed: bool = False,
) -> dict[str, Any]:
    """Compute direct and transitive invalid nodes from receipts/version refs.

    ``chain_node_ids`` includes the client root and then every receipt-bearing
    node.  Output ordering is request id, chain position, then dependency kind,
    which makes plans stable regardless of thread completion order.
    """

    receipt_nodes = list(chain_node_ids[1:])
    invalid: set[str] = set()
    redispatch: set[str] = set()
    steps: list[dict[str, Any]] = []
    affected_requests: set[str] = set()

    def add(node: str, *, request_id: str, node_id: str | None, kind: str,
            reason: str, by: str | None) -> None:
        if node not in invalid:
            invalid.add(node)
            steps.append(
                {
                    "node": node,
                    "request_id": request_id,
                    "chain_node_id": node_id,
                    "kind": kind,
                    "reason": reason,
                    "propagated_from": by,
                }
            )
            affected_requests.add(request_id)

    for request_id in sorted(request_ids):
        invalid_accum_indices: set[int] = set()

        for index, node_id in enumerate(receipt_nodes):
            stored = stored_receipt_versions.get((request_id, node_id), {})
            current_version = current_config_versions.get(node_id)
            current_scale = current_scale_factors.get(node_id)
            version_changed = (
                current_version is not None
                and stored.get("config_version") != current_version
            )
            scale_changed = (
                current_scale is not None
                and stored.get("scale_factor") != current_scale
            )

            if version_changed or scale_changed:
                reasons = []
                if version_changed:
                    reasons.append("config_version_changed")
                if scale_changed:
                    reasons.append("scale_factor_changed")
                reason = "+".join(reasons)
                rnode = receipt_node(request_id, node_id)
                add(rnode, request_id=request_id, node_id=node_id, kind="receipt",
                    reason=reason, by=None)
                redispatch.add(rnode)

                mnode = metric_node(request_id, node_id)
                add(mnode, request_id=request_id, node_id=node_id, kind="metric",
                    reason="receipt_changed", by=rnode)
                anode = accum_node(request_id, node_id)
                add(anode, request_id=request_id, node_id=node_id, kind="accumulation",
                    reason="metric_changed", by=mnode)
                invalid_accum_indices.add(index)
            elif scale_caliber_changed:
                mnode = metric_node(request_id, node_id)
                add(mnode, request_id=request_id, node_id=node_id, kind="metric",
                    reason="scale_caliber_version_changed", by=None)
                anode = accum_node(request_id, node_id)
                add(anode, request_id=request_id, node_id=node_id, kind="accumulation",
                    reason="metric_recalibrated", by=mnode)
                invalid_accum_indices.add(index)

        # A changed cumulative value at index i changes every later chain sum.
        for index, node_id in enumerate(receipt_nodes):
            prior_invalid = any(bad_index < index for bad_index in invalid_accum_indices)
            current_invalid = index in invalid_accum_indices
            if prior_invalid and not current_invalid:
                source_index = max(bad for bad in invalid_accum_indices if bad < index)
                source_node = receipt_nodes[source_index]
                anode = accum_node(request_id, node_id)
                add(
                    anode,
                    request_id=request_id,
                    node_id=node_id,
                    kind="accumulation",
                    reason="downstream_chain_sum_depends_on_prior_accumulation",
                    by=accum_node(request_id, source_node),
                )
                invalid_accum_indices.add(index)

        if invalid_accum_indices:
            enode = e2e_node(request_id)
            last_invalid_index = max(invalid_accum_indices)
            add(enode, request_id=request_id, node_id=None, kind="e2e",
                reason="final_sum_contains_invalid_accumulation",
                by=accum_node(request_id, receipt_nodes[last_invalid_index]))
            attr = attribution_node(request_id)
            add(attr, request_id=request_id, node_id=None, kind="attribution",
                reason="error_ranking_depends_on_all_metrics", by=enode)

    ordered_steps = sorted(
        steps,
        key=lambda step: (
            step["request_id"],
            0 if step["chain_node_id"] is None
            else receipt_nodes.index(step["chain_node_id"]),
            step["kind"],
            step["node"],
        ),
    )
    return {
        "schema": "hopledger.invalidation/1",
        "invalid_nodes": sorted(invalid),
        "redispatch_nodes": sorted(redispatch),
        "recompute_only_nodes": sorted(invalid - redispatch),
        "affected_request_ids": sorted(affected_requests),
        "steps": ordered_steps,
    }


def verify_digests_unchanged(
    before: Mapping[str, str],
    after: Mapping[str, str],
    *,
    allowed_changed: Sequence[str],
) -> list[str]:
    """Return violations for non-allowed keys whose digest changed."""

    allowed = set(allowed_changed)
    violations: list[str] = []
    for key in sorted(set(before) | set(after)):
        if key in allowed:
            continue
        if before.get(key) != after.get(key):
            violations.append(key)
    return violations
