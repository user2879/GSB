from __future__ import annotations

from typing import Any, Mapping, Sequence

from .canonical import canonical_json
from .fixedpoint import (
    OVERFLOW_RULE,
    TRUNCATION_DIRECTION,
    apportion_basis_points,
    quantize_component,
    quantization_tick_bound_units,
    saturated_add,
    saturate_i64,
)


METRIC_SCHEMA = "hopledger.conclusion/1"
ATTRIBUTION_SCHEMA = "hopledger.attribution/1"


def _receipt_value(receipt: Mapping[str, Any], key: str, default: int = 0) -> int:
    value = receipt.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool):
        return default
    return value


def build_conclusion(
    *,
    request_id: str,
    node_ids: Sequence[str],
    receipts: Sequence[Mapping[str, Any]],
    session_scale: int,
    scale_caliber_version: int,
    hop_set_version: int,
) -> dict[str, Any]:
    """Build deterministic fixed-point hop metrics and ordered end-to-end sum."""

    receipts_by_node = {receipt["node_id"]: receipt for receipt in receipts}
    metric_node_ids = list(node_ids[1:])  # node_ids[0] is the client root.
    missing = [node_id for node_id in metric_node_ids if node_id not in receipts_by_node]
    if missing:
        raise KeyError(f"missing receipts for {request_id}: {','.join(missing)}")

    nodes: list[dict[str, Any]] = []
    operations: list[dict[str, Any]] = [
        {"seq": 0, "node_id": node_ids[0], "op": "client_root", "cumulative_units": 0}
    ]
    cumulative = 0
    seq = 1
    config_versions: dict[str, int] = {node_ids[0]: 0}
    used_scale_factors: dict[str, int] = {node_ids[0]: session_scale}

    for chain_index, node_id in enumerate(metric_node_ids, start=1):
        receipt = receipts_by_node[node_id]
        if receipt.get("request_id") != request_id:
            raise ValueError(f"receipt request mismatch at {node_id}")
        scale_factor = int(receipt.get("scale_factor", session_scale))
        config_version = int(receipt.get("config_version", 0))
        config_versions[node_id] = config_version
        used_scale_factors[node_id] = scale_factor

        components = []
        node_total = 0
        for name in ("queue_ms", "process_ms", "forward_ms"):
            ms = _receipt_value(receipt, name)
            component_name = name.removesuffix("_ms")
            scaled = quantize_component(component_name, ms, scale_factor, session_scale)
            component = scaled.to_dict()
            components.append(component)
            node_total = saturated_add(node_total, scaled.units)
            operations.append(
                {
                    "seq": seq,
                    "node_id": node_id,
                    "op": f"add_{component_name}",
                    "value_units": scaled.units,
                    "cumulative_units": saturated_add(cumulative, node_total)
                    if name == "forward_ms"
                    else None,
                }
            )
            seq += 1

        before_cumulative = cumulative
        cumulative = saturated_add(cumulative, node_total)
        saturated = cumulative >= (1 << 63) - 1
        nodes.append(
            {
                "node_id": node_id,
                "chain_index": chain_index,
                "kind": receipt.get("kind", "hop"),
                "config_version": config_version,
                "scale_factor": scale_factor,
                "components": components,
                "node_total_units": node_total,
                "before_cumulative_units": before_cumulative,
                "cumulative_units": cumulative,
                "saturated": saturated,
            }
        )
        operations.append(
            {
                "seq": seq,
                "node_id": node_id,
                "op": "node_subtotal",
                "value_units": node_total,
                "cumulative_units": cumulative,
            }
        )
        seq += 1

    return {
        "schema": METRIC_SCHEMA,
        "request_id": request_id,
        "hop_set_version": hop_set_version,
        "scale_caliber_version": scale_caliber_version,
        "session_scale": session_scale,
        "truncation_direction": TRUNCATION_DIRECTION,
        "overflow_rule": OVERFLOW_RULE,
        "chain_order": list(node_ids),
        "accumulation_order": [operation["node_id"] for operation in operations
                               if operation["op"] == "node_subtotal"],
        "operations": operations,
        "config_versions": config_versions,
        "used_scale_factors": used_scale_factors,
        "nodes": nodes,
        "e2e_total_units": cumulative,
    }


def conclusion_bytes(conclusion: Mapping[str, Any]) -> bytes:
    return canonical_json(conclusion)


def attribute_errors(
    conclusion: Mapping[str, Any],
    error_config: Any,
    *,
    failure_ratio_bp_by_node: Mapping[str, int] | None = None,
    expected_config_versions: Mapping[str, int] | None = None,
    expected_scale_factors: Mapping[str, int] | None = None,
    expected_scale_caliber_version: int | None = None,
    observed_failure_ratio_bp_by_node: Mapping[str, int] | None = None,
    timeout_bound_units_by_node: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    """Derive integer upper bounds and deterministic dominant-hop attribution."""

    failure_ratios = failure_ratio_bp_by_node or {}
    observed_failures = observed_failure_ratio_bp_by_node or {}
    timeout_bounds = timeout_bound_units_by_node or {}

    version_conflicts: list[dict[str, Any]] = []
    if (
        expected_scale_caliber_version is not None
        and conclusion.get("scale_caliber_version") != expected_scale_caliber_version
    ):
        version_conflicts.append(
            {
                "node_id": "<session>",
                "field": "scale_caliber_version",
                "used": conclusion.get("scale_caliber_version"),
                "expected": expected_scale_caliber_version,
            }
        )

    config_versions = conclusion.get("config_versions", {})
    scale_factors = conclusion.get("used_scale_factors", {})
    for node in conclusion.get("nodes", []):
        node_id = node["node_id"]
        if expected_config_versions and node_id in expected_config_versions:
            used = config_versions.get(node_id)
            expected = expected_config_versions[node_id]
            if used != expected:
                version_conflicts.append(
                    {"node_id": node_id, "field": "config_version",
                     "used": used, "expected": expected}
                )
        if expected_scale_factors and node_id in expected_scale_factors:
            used = scale_factors.get(node_id)
            expected = expected_scale_factors[node_id]
            if used != expected:
                version_conflicts.append(
                    {"node_id": node_id, "field": "scale_factor",
                     "used": used, "expected": expected}
                )

    node_rows: list[dict[str, Any]] = []
    weights: list[int] = []
    high_failure_nodes: list[str] = []
    high_failure_threshold = int(getattr(error_config, "high_failure_ratio_bp", 5_000))

    for node in conclusion.get("nodes", []):
        node_id = node["node_id"]
        scale_factor = int(node.get("scale_factor", conclusion["session_scale"]))
        session_scale = int(conclusion["session_scale"])
        component_count = len(node.get("components", [])) or 3
        tick_bound = quantization_tick_bound_units(scale_factor, session_scale)

        quantization_bound = saturate_i64(component_count * tick_bound)
        # One floor conversion per component can lose less than one session ULP.
        # Those losses are carried in chain order, so this is the chain-
        # accumulation/truncation segment of the conservative upper bound.
        chain_accumulation_bound = saturate_i64(component_count)

        forward_ms = 0
        for component in node.get("components", []):
            if component.get("name") == "forward":
                forward_ms = int(component.get("milliseconds", 0))
        if node.get("kind") == "origin":
            timeout_estimation_bound = int(timeout_bounds.get(node_id, 0))
        else:
            estimate_bound = session_scale * max(1, forward_ms)
            timeout_estimation_bound = int(
                timeout_bounds.get(node_id, saturate_i64(estimate_bound))
            )
        timeout_estimation_bound = saturate_i64(max(0, timeout_estimation_bound))

        total_bound = saturate_i64(
            quantization_bound + chain_accumulation_bound + timeout_estimation_bound
        )
        configured_failure_bp = int(failure_ratios.get(node_id, 0))
        observed_failure_bp = int(observed_failures.get(node_id, 0))
        if max(configured_failure_bp, observed_failure_bp) >= high_failure_threshold:
            high_failure_nodes.append(node_id)

        row = {
            "node_id": node_id,
            "chain_index": node.get("chain_index"),
            "sources": {
                "quantization_truncation_bound_units": quantization_bound,
                "chain_accumulation_bound_units": chain_accumulation_bound,
                "timeout_estimation_bound_units": timeout_estimation_bound,
            },
            "error_upper_bound_units": total_bound,
            "configured_failure_ratio_bp": configured_failure_bp,
            "observed_failure_ratio_bp": observed_failure_bp,
        }
        node_rows.append(row)
        weights.append(total_bound)

    contributions = apportion_basis_points(weights)
    for row, share in zip(node_rows, contributions):
        row["contribution_bp"] = share

    ranked = sorted(
        node_rows,
        key=lambda row: (-int(row["contribution_bp"]), int(row["chain_index"])),
    )
    dominant = ranked[0]["node_id"] if ranked else None
    top_share = int(ranked[0]["contribution_bp"]) if ranked else 0
    second_share = int(ranked[1]["contribution_bp"]) if len(ranked) > 1 else 0
    lead_bp = top_share - second_share

    threshold = int(getattr(error_config, "dominance_threshold_bp", 6_000))
    margin = int(getattr(error_config, "indistinguishable_margin_bp", 500))
    is_dominant = top_share >= threshold and lead_bp >= margin

    causes: list[str] = []
    if version_conflicts:
        causes.append("config_version_conflict")
    if high_failure_nodes:
        causes.append("failure_injection_ratio_high")
    if not is_dominant:
        causes.append("error_indistinguishable")
    if not causes:
        causes.append("dominant_hop_identified")

    if version_conflicts:
        primary_cause = "config_version_conflict"
    elif high_failure_nodes:
        primary_cause = "failure_injection_ratio_high"
    elif not is_dominant:
        primary_cause = "error_indistinguishable"
    else:
        primary_cause = "dominant_hop_identified"

    total_bound = saturate_i64(sum(weight for weight in weights))
    source_totals = {
        "quantization_truncation_bound_units": saturate_i64(
            sum(row["sources"]["quantization_truncation_bound_units"] for row in node_rows)
        ),
        "chain_accumulation_bound_units": saturate_i64(
            sum(row["sources"]["chain_accumulation_bound_units"] for row in node_rows)
        ),
        "timeout_estimation_bound_units": saturate_i64(
            sum(row["sources"]["timeout_estimation_bound_units"] for row in node_rows)
        ),
    }

    return {
        "schema": ATTRIBUTION_SCHEMA,
        "request_id": conclusion["request_id"],
        "dominance_threshold_bp": threshold,
        "indistinguishable_margin_bp": margin,
        "dominant_hop": dominant,
        "declared_dominant_hop": dominant if is_dominant else None,
        "is_dominant": is_dominant,
        "top_contribution_bp": top_share,
        "second_contribution_bp": second_share,
        "lead_bp": lead_bp,
        "primary_cause": primary_cause,
        "cause_codes": causes,
        "high_failure_nodes": high_failure_nodes,
        "version_conflicts": version_conflicts,
        "end_to_end_error_upper_bound_units": total_bound,
        "source_totals": source_totals,
        "nodes": node_rows,
        "ranking": [row["node_id"] for row in ranked],
    }


def attribution_bytes(attribution: Mapping[str, Any]) -> bytes:
    return canonical_json(attribution)
