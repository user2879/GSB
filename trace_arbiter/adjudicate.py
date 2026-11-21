from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from .canonical import canonical_text, digest_text, stable_bucket
from .errors import ArbiterError


@dataclass(frozen=True)
class NodeRule:
    node_id: str
    propagation_rule_version: int
    sample_ratio_bps: int
    random_seed: int


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _decode_node(raw: dict[str, Any]) -> NodeRule:
    try:
        node_id = raw["node_id"]
        version = raw["propagation_rule_version"]
        ratio = raw["sample_ratio_bps"]
        seed = raw["random_seed"]
    except (KeyError, TypeError) as exc:
        raise ArbiterError("E_SPEC", "worker request contains an incomplete node") from exc
    if not isinstance(node_id, str) or not node_id:
        raise ArbiterError("E_SPEC", "worker node_id must be a non-empty string")
    if not (_is_int(version) and _is_int(ratio) and _is_int(seed)):
        raise ArbiterError("E_SPEC", "worker node rule fields must be integers")
    if not (1 <= version and 0 <= ratio <= 10_000):
        raise ArbiterError("E_SPEC", "worker node rule fields are out of range")
    return NodeRule(node_id, version, ratio, seed)


def _decode_span(raw: dict[str, Any], node_ids: set[str]) -> dict[str, Any]:
    required = (
        "trace_id",
        "span_id",
        "parent_span_id",
        "node_id",
        "timestamp_ms",
        "duration_us",
        "sampled",
    )
    if not isinstance(raw, dict) or any(key not in raw for key in required):
        raise ArbiterError("E_SPEC", "worker request contains an incomplete span")
    for key in ("trace_id", "span_id", "node_id"):
        if not isinstance(raw[key], str) or not raw[key]:
            raise ArbiterError("E_SPEC", f"worker span field {key} must be a non-empty string")
    parent = raw["parent_span_id"]
    if parent is not None and not isinstance(parent, str):
        raise ArbiterError("E_SPEC", "worker span parent_span_id must be string or null")
    if raw["node_id"] not in node_ids:
        raise ArbiterError("E_CHAIN", "span references an undeclared node", field="node_id")
    if not (
        _is_int(raw["timestamp_ms"])
        and _is_int(raw["duration_us"])
        and raw["timestamp_ms"] >= 0
        and raw["duration_us"] >= 0
    ):
        raise ArbiterError("E_SPEC", "worker span timing values must be non-negative integers")
    if not isinstance(raw["sampled"], bool):
        raise ArbiterError("E_SPEC", "worker span sampled flag must be boolean")
    return {
        "trace_id": raw["trace_id"],
        "span_id": raw["span_id"],
        "parent_span_id": parent or None,
        "node_id": raw["node_id"],
        "timestamp_ms": raw["timestamp_ms"],
        "duration_us": raw["duration_us"],
        "sampled": raw["sampled"],
    }


def _rules_by_version(nodes: Iterable[NodeRule]) -> dict[int, tuple[int, int]]:
    rules: dict[int, tuple[int, int]] = {}
    for node in nodes:
        current = (node.sample_ratio_bps, node.random_seed)
        previous = rules.get(node.propagation_rule_version)
        if previous is not None and previous != current:
            raise ArbiterError(
                "E_SPEC",
                "one propagation rule version has inconsistent ratio/seed",
                field="sample_ratio_bps",
            )
        rules[node.propagation_rule_version] = current
    return rules


def _validate_trace(span_list: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for span in span_list:
        if span["span_id"] in by_id:
            raise ArbiterError("E_SPEC", "duplicate span_id in trace", field="span_id")
        by_id[span["span_id"]] = span

    roots = []
    for span in span_list:
        parent = span["parent_span_id"]
        if parent is None:
            roots.append(span)
        elif parent not in by_id:
            raise ArbiterError(
                "E_CHAIN",
                "parent_span_id does not exist in trace",
                trace_id=span["trace_id"],
                field="parent_span_id",
            )
    if len(roots) != 1:
        raise ArbiterError(
            "E_SPEC",
            "trace must have exactly one root span",
            field="root_span_id",
        )

    state: dict[str, int] = {}
    stack: list[str] = []

    def visit(span_id: str) -> None:
        status = state.get(span_id, 0)
        if status == 1:
            raise ArbiterError(
                "E_SPEC",
                "parent_span_id cycle detected",
                field="parent_span_id",
            )
        if status == 2:
            return
        state[span_id] = 1
        stack.append(span_id)
        parent = by_id[span_id]["parent_span_id"]
        if parent is not None:
            visit(parent)
        stack.pop()
        state[span_id] = 2

    for span_id in sorted(by_id):
        visit(span_id)

    for span in span_list:
        parent = span["parent_span_id"]
        if parent is not None and span["timestamp_ms"] < by_id[parent]["timestamp_ms"]:
            raise ArbiterError(
                "E_SPEC",
                "timestamp_ms is earlier than parent timestamp",
                trace_id=span["trace_id"],
                field="timestamp_ms",
            )
    return by_id, roots[0]


def build_trace_decisions(
    nodes: Iterable[dict[str, Any]],
    spans: Iterable[dict[str, Any]],
    *,
    ruleset_version: int,
) -> list[dict[str, Any]]:
    """Derive the canonical, order-independent decision for every trace."""

    node_list = sorted((_decode_node(node) for node in nodes), key=lambda node: node.node_id)
    node_by_id = {node.node_id: node for node in node_list}
    if len(node_by_id) != len(node_list):
        raise ArbiterError("E_SPEC", "duplicate worker node_id", field="node_id")
    version_rules = _rules_by_version(node_list)
    span_list = sorted(
        (_decode_span(span, set(node_by_id)) for span in spans),
        key=lambda span: (span["trace_id"], span["timestamp_ms"], span["span_id"]),
    )
    if not span_list:
        raise ArbiterError("E_SPEC", "worker request has no spans")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for span in span_list:
        grouped.setdefault(span["trace_id"], []).append(span)

    trace_decisions: list[dict[str, Any]] = []
    for trace_id in sorted(grouped):
        trace_spans = grouped[trace_id]
        by_id, root = _validate_trace(trace_spans)
        participants = sorted({span["node_id"] for span in trace_spans})
        trace_versions = sorted(
            {node_by_id[node_id].propagation_rule_version for node_id in participants}
        )
        decisions: list[dict[str, Any]] = []
        sensitive_span_ids: set[str] = set()
        for span in trace_spans:
            node = node_by_id[span["node_id"]]
            ratio, seed = version_rules[node.propagation_rule_version]
            bucket = stable_bucket(
                seed,
                node.propagation_rule_version,
                trace_id,
                span["span_id"],
                node.node_id,
            )
            sampled = bucket < ratio
            version_decisions = []
            outcomes: set[bool] = set()
            for version in trace_versions:
                version_ratio, version_seed = version_rules[version]
                version_bucket = stable_bucket(
                    version_seed,
                    version,
                    trace_id,
                    span["span_id"],
                    node.node_id,
                )
                version_sampled = version_bucket < version_ratio
                outcomes.add(version_sampled)
                version_decisions.append(
                    {
                        "bucket": version_bucket,
                        "rule_version": version,
                        "sampled": version_sampled,
                    }
                )
            if len(outcomes) > 1:
                sensitive_span_ids.add(span["span_id"])
            decisions.append(
                {
                    "bucket": bucket,
                    "downstream_visible": sampled,
                    "duration_us": span["duration_us"],
                    "node_id": span["node_id"],
                    "parent_span_id": span["parent_span_id"],
                    "reported_sampled": span["sampled"],
                    "reported_sampled_matches": span["sampled"] == sampled,
                    "rule_version": node.propagation_rule_version,
                    "sampled": sampled,
                    "span_id": span["span_id"],
                    "timestamp_ms": span["timestamp_ms"],
                    "version_decisions": version_decisions,
                }
            )

        sampled_decisions = [decision for decision in decisions if decision["sampled"]]
        sampled_nodes = sorted({decision["node_id"] for decision in sampled_decisions})
        sampled_span_ids = sorted(decision["span_id"] for decision in sampled_decisions)
        version_groups: list[dict[str, Any]] = []
        for version in trace_versions:
            version_span_ids = sorted(
                decision["span_id"]
                for decision in decisions
                if next(
                    item["sampled"]
                    for item in decision["version_decisions"]
                    if item["rule_version"] == version
                )
            )
            version_groups.append(
                {
                    "node_ids": sorted(
                        node.node_id
                        for node in node_list
                        if node.node_id in participants
                        and node.propagation_rule_version == version
                    ),
                    "rule_version": version,
                    "sampled_node_ids": sorted(
                        {
                            decision["node_id"]
                            for decision in decisions
                            if decision["span_id"] in set(version_span_ids)
                        }
                    ),
                    "sampled_span_ids": version_span_ids,
                }
            )
        affected_node_ids = sorted(
            {
                decision["node_id"]
                for decision in decisions
                if decision["span_id"] in sensitive_span_ids
            }
        )
        trace_decisions.append(
            {
                "format_version": 1,
                "participating_nodes": participants,
                "root_span_id": root["span_id"],
                "rule_difference": {
                    "affected_node_ids": affected_node_ids,
                    "detected": bool(sensitive_span_ids),
                    "span_ids": sorted(sensitive_span_ids),
                    "versions": version_groups,
                },
                "rule_versions": trace_versions,
                "sampled_node_ids": sampled_nodes,
                "sampled_span_ids": sampled_span_ids,
                "span_decisions": decisions,
                "trace_id": trace_id,
            }
        )
    return trace_decisions


def authoritative_payload(
    nodes: Iterable[dict[str, Any]],
    spans: Iterable[dict[str, Any]],
    *,
    ruleset_version: int,
) -> dict[str, Any]:
    trace_decisions = build_trace_decisions(
        nodes,
        spans,
        ruleset_version=ruleset_version,
    )
    return {
        "format_version": 1,
        "ruleset_version": ruleset_version,
        "trace_decisions": trace_decisions,
        "trace_count": len(trace_decisions),
    }


def authoritative_summary(
    nodes: Iterable[dict[str, Any]],
    spans: Iterable[dict[str, Any]],
    *,
    ruleset_version: int,
) -> tuple[str, str, dict[str, Any]]:
    payload = authoritative_payload(nodes, spans, ruleset_version=ruleset_version)
    summary = canonical_text(payload)
    return summary, digest_text(summary), payload


def batch_decision_summary(
    nodes: Iterable[dict[str, Any]],
    spans: Iterable[dict[str, Any]],
    *,
    ruleset_version: int,
    target_node_id: str,
    target_span_ids: Iterable[str] | None = None,
    target_spans: Iterable[dict[str, Any]] | None = None,
) -> tuple[str, str, dict[str, Any]]:
    span_records = list(spans)
    node_records = list(nodes)
    refs: set[tuple[str, str]] = set()
    if target_spans is not None:
        for item in target_spans:
            if not isinstance(item, dict) or not isinstance(item.get("trace_id"), str) or not isinstance(
                item.get("span_id"), str
            ):
                raise ArbiterError("E_PROTOCOL", "target_spans entries must contain trace_id and span_id")
            refs.add((item["trace_id"], item["span_id"]))
    if target_span_ids is not None:
        for span_id in target_span_ids:
            if not isinstance(span_id, str):
                raise ArbiterError("E_PROTOCOL", "target_span_ids entries must be strings")
            matches = [
                (span["trace_id"], span["span_id"])
                for span in span_records
                if span.get("span_id") == span_id
            ]
            if len(matches) != 1:
                raise ArbiterError(
                    "E_PROTOCOL",
                    "target_span_ids is ambiguous; use target_spans for duplicate span ids",
                    field="target_span_ids",
                )
            refs.add(matches[0])
    if not refs:
        raise ArbiterError("E_PROTOCOL", "target_spans must not be empty")
    all_decisions = build_trace_decisions(
        node_records,
        span_records,
        ruleset_version=ruleset_version,
    )
    selected: list[dict[str, Any]] = []
    trace_context: list[dict[str, Any]] = []
    target_trace_ids = {trace_id for trace_id, _span_id in refs}
    for trace in all_decisions:
        selected.extend(
            decision
            for decision in trace["span_decisions"]
            if (trace["trace_id"], decision["span_id"]) in refs
        )
        if trace["trace_id"] in target_trace_ids and any(
            decision["node_id"] == target_node_id for decision in trace["span_decisions"]
        ):
            trace_context.append(
                {
                    "participating_nodes": trace["participating_nodes"],
                    "root_span_id": trace["root_span_id"],
                    "trace_id": trace["trace_id"],
                }
            )
    if len(selected) != len(refs):
        found = {(trace["trace_id"], decision["span_id"]) for trace in all_decisions for decision in trace["span_decisions"]}
        missing = sorted([{"trace_id": trace_id, "span_id": span_id} for trace_id, span_id in refs - found], key=lambda item: (item["trace_id"], item["span_id"]))
        raise ArbiterError(
            "E_CHAIN",
            "target_span_ids contains an unknown span",
            field="target_span_ids",
            details={"missing": missing},
        )
    payload = {
        "format_version": 1,
        "mode": "batch",
        "span_decisions": sorted(
            selected,
            key=lambda decision: (decision["timestamp_ms"], decision["span_id"]),
        ),
        "target_node_id": target_node_id,
        "target_spans": [
            {"span_id": span_id, "trace_id": trace_id}
            for trace_id, span_id in sorted(refs)
        ],
        "trace_context": sorted(trace_context, key=lambda item: item["trace_id"]),
    }
    summary = canonical_text(payload)
    return summary, digest_text(summary), payload
