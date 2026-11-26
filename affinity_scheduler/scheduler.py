from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .errors import SpecError
from .model import (
    DEFAULT_TIE_BREAK,
    FILTER_ORDER,
    PLUGIN_NAMES,
    VETO_BY_PLUGIN,
    LabelRequirement,
    Node,
    Pod,
    PodSelectorRequirement,
    Spec,
    Toleration,
)
from .scoring import ScoringManager
from .util import canonical_json, digest_value


CONCLUSION_SCHEMA = "pod-schedule-conclusion-v1"


def label_requirement_matches(
    labels: Mapping[str, str], requirement: LabelRequirement
) -> bool:
    if requirement.operator == "In":
        return requirement.key in labels and labels[requirement.key] in requirement.values
    if requirement.operator == "NotIn":
        return (
            requirement.key in labels
            and labels[requirement.key] not in requirement.values
        )
    if requirement.operator == "Exists":
        return requirement.key in labels
    if requirement.operator == "DoesNotExist":
        return requirement.key not in labels
    return False


def pod_selector_matches(
    labels: Mapping[str, str], requirement: PodSelectorRequirement
) -> bool:
    if requirement.operator == "In":
        return requirement.key in labels and labels[requirement.key] in requirement.values
    if requirement.operator == "NotIn":
        return (
            requirement.key in labels
            and labels[requirement.key] not in requirement.values
        )
    if requirement.operator == "Exists":
        return requirement.key in labels
    if requirement.operator == "DoesNotExist":
        return requirement.key not in labels
    return False


def tolerates(taint: Any, tolerations: Iterable[Toleration]) -> bool:
    for toleration in tolerations:
        if toleration.effect != -1 and toleration.effect != taint.effect:
            continue
        if toleration.operator == "Exists":
            if toleration.key == "" or toleration.key == taint.key:
                return True
        elif toleration.key == taint.key and toleration.value == taint.value:
            return True
    return False


def _anti_conflicts(
    pod: Pod,
    node: Node,
    requirements: Iterable[PodSelectorRequirement],
    bindings: Mapping[str, str],
    spec: Spec,
) -> list[tuple[str, PodSelectorRequirement]]:
    conflicts: list[tuple[str, PodSelectorRequirement]] = []
    number_of_nodes = 0
    for other_id, other_node_name in sorted(bindings.items()):
        if other_id == pod.pod_id:
            continue
        other = spec.pod_by_id.get(other_id)
        if other is None:
            continue
        other_node = spec.node_by_name.get(other_node_name)
        if other_node is None:
            continue
        for requirement in requirements:
            if not pod_selector_matches(other.labels, requirement):
                continue
            candidate_topology = node.labels.get(requirement.topology_key)
            other_topology = other_node.labels.get(requirement.topology_key)
            if (
                candidate_topology is not None
                and other_topology is not None
                and candidate_topology == other_topology
            ):
                conflicts.append((other_id, requirement))
                number_of_nodes += 1
    return conflicts


def _taint_filter(
    pod: Pod, node: Node
) -> tuple[bool, int, str]:
    untolerated_prefer = 0
    for taint in node.taints:
        tolerated = tolerates(taint, pod.tolerations)
        if not tolerated and taint.effect in {0, 2}:
            return (
                False,
                untolerated_prefer,
                f"untolerated taint {taint.key}={taint.value} effect={taint.effect_name}",
            )
        if not tolerated and taint.effect == 1:
            untolerated_prefer += 1
    return True, untolerated_prefer, "all blocking taints are tolerated"


def _node_affinity_filter(pod: Pod, node: Node) -> tuple[bool, str]:
    for requirement in pod.node_hard:
        if not label_requirement_matches(node.labels, requirement):
            return (
                False,
                f"hard node affinity {requirement.key} {requirement.operator} {list(requirement.values)} unmet",
            )
    return True, "hard node affinity terms satisfied"


def _resource_filter(pod: Pod, node: Node) -> tuple[bool, str]:
    if pod.request_cpu > node.cpu:
        return False, f"cpu request {pod.request_cpu} exceeds allocatable {node.cpu}"
    if pod.request_memory > node.memory:
        return (
            False,
            f"memory request {pod.request_memory} exceeds allocatable {node.memory}",
        )
    return True, "request fits allocatable resources"


def _step(order: int, plugin: str, passed: bool, reason: str) -> dict[str, Any]:
    return {
        "order": order,
        "plugin": plugin,
        "result": "pass" if passed else "reject",
        "reason": reason,
    }


def filter_node(
    pod: Pod,
    node: Node,
    bindings: Mapping[str, str],
    spec: Spec,
) -> tuple[bool, list[dict[str, Any]], str | None, int]:
    trace: list[dict[str, Any]] = []

    passed, untolerated_prefer, reason = _taint_filter(pod, node)
    trace.append(_step(1, "taint_toleration", passed, reason))
    if not passed:
        return False, trace, "taint_toleration", untolerated_prefer

    passed, reason = _node_affinity_filter(pod, node)
    trace.append(_step(2, "node_affinity", passed, reason))
    if not passed:
        return False, trace, "node_affinity", untolerated_prefer

    hard_conflicts = _anti_conflicts(pod, node, pod.anti_hard, bindings, spec)
    if hard_conflicts:
        reason = "hard pod anti-affinity conflict with " + ",".join(
            sorted({other_id for other_id, _ in hard_conflicts})
        )
        trace.append(_step(3, "pod_anti_affinity", False, reason))
        return False, trace, "pod_anti_affinity", untolerated_prefer
    trace.append(_step(3, "pod_anti_affinity", True, "no hard anti-affinity conflict"))

    passed, reason = _resource_filter(pod, node)
    trace.append(_step(4, "resource_fit", passed, reason))
    if not passed:
        return False, trace, "resource_fit", untolerated_prefer
    return True, trace, None, untolerated_prefer


def build_score_features(
    pod: Pod,
    node: Node,
    bindings: Mapping[str, str],
    spec: Spec,
    untolerated_prefer: int,
) -> dict[str, Any]:
    soft_match_weight = sum(
        requirement.weight
        for requirement in pod.node_soft
        if label_requirement_matches(node.labels, requirement)
    )
    soft_conflicts = _anti_conflicts(
        pod, node, pod.anti_soft, bindings, spec
    )
    anti_weight = sum(requirement.weight for _, requirement in soft_conflicts)
    return {
        "has_taints": bool(node.taints),
        "untolerated_prefer_taints": untolerated_prefer,
        "has_node_affinity": bool(pod.node_hard or pod.node_soft),
        "node_soft_match_weight": soft_match_weight,
        "has_pod_anti_affinity": bool(pod.anti_hard or pod.anti_soft),
        "anti_soft_conflicts": anti_weight,
        "remaining_cpu": node.cpu - pod.request_cpu,
        "remaining_memory": node.memory - pod.request_memory,
        "allocatable_cpu": node.cpu,
        "allocatable_memory": node.memory,
    }


def _veto_for_rejections(rejected_by: list[str]) -> int:
    unique = set(rejected_by)
    if len(unique) == 1:
        return VETO_BY_PLUGIN[next(iter(unique))]
    return 50


def _zero_plugin_map() -> dict[str, int]:
    return {name: 0 for name in PLUGIN_NAMES}


def _candidate_sort_key(reply: Mapping[str, Any]) -> tuple[Any, ...]:
    scores = reply["plugin_scores"]
    return (
        -int(reply["total_fixed"]),
        *(-int(scores[name]) for name in PLUGIN_NAMES),
        str(reply["node_name"]),
    )


def stable_conclusion(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": CONCLUSION_SCHEMA,
        "pod_id": result["pod_id"],
        "node_name": result["node_name"],
        "status": result["status"],
        "veto_code": result["veto_code"],
        "veto_name": result["veto_name"],
        "total_fixed": result["total_fixed"],
        "total_score": result["total_score"],
        "plugin_scores": {
            name: int(result["plugin_scores"][name]) for name in PLUGIN_NAMES
        },
        "plugin_contributions": {
            name: int(result["plugin_contributions"][name]) for name in PLUGIN_NAMES
        },
        "error_bound": result["error_bound"],
        "tie_break": list(result["tie_break"]),
        "filter_trace": result["filter_trace"],
    }


def conclusion_bytes(result: Mapping[str, Any]) -> bytes:
    return canonical_json(stable_conclusion(result)).encode("utf-8")


def conclusion_digest(result: Mapping[str, Any]) -> str:
    return digest_value(stable_conclusion(result))


def _schedule_pod(
    spec: Spec,
    pod: Pod,
    manager: ScoringManager,
    bindings: Mapping[str, str],
) -> dict[str, Any]:
    filter_trace: list[dict[str, Any]] = []
    rejected: list[str] = []
    rejected_by_node: dict[str, str] = {}
    passed_nodes: list[tuple[Node, int]] = []
    for node_name in sorted(spec.node_by_name):
        node = spec.node_by_name[node_name]
        passed, trace, rejected_by, untolerated_prefer = filter_node(
            pod, node, bindings, spec
        )
        filter_trace.append({"node_name": node_name, "steps": trace})
        if not passed:
            rejected.append(rejected_by or "no_feasible_node")
            rejected_by_node[node_name] = rejected_by or "no_feasible_node"
        else:
            passed_nodes.append((node, untolerated_prefer))

    receipts: dict[str, Any] = {}
    for node, untolerated_prefer in passed_nodes:
        features = build_score_features(
            pod, node, bindings, spec, untolerated_prefer
        )
        reply = manager.score(
            pod_id=pod.pod_id,
            node_name=node.name,
            parameter_version=spec.parameter_version,
            scale_exponent=spec.scale_exponent,
            weights_myriad=spec.weights_myriad,
            features=features,
        )
        receipts[node.name] = reply

    if not receipts:
        veto_code = _veto_for_rejections(rejected)
        result = {
            "schema": CONCLUSION_SCHEMA,
            "pod_id": pod.pod_id,
            "node_name": "-",
            "status": "unbound",
            "veto_code": veto_code,
            "veto_name": _veto_name(veto_code),
            "total_fixed": 0,
            "total_score": 0,
            "plugin_scores": _zero_plugin_map(),
            "plugin_contributions": _zero_plugin_map(),
            "error_bound": 0,
            "tie_break": list(DEFAULT_TIE_BREAK),
            "filter_trace": filter_trace,
            "receipts": {},
            "parameter_version": spec.parameter_version,
            "scale_exponent": spec.scale_exponent,
            "scale_factor": spec.scale_factor,
        }
    else:
        best = min(receipts.values(), key=_candidate_sort_key)
        result = {
            "schema": CONCLUSION_SCHEMA,
            "pod_id": pod.pod_id,
            "node_name": best["node_name"],
            "status": "bound",
            "veto_code": 0,
            "veto_name": _veto_name(0),
            "total_fixed": best["total_fixed"],
            "total_score": best["total_score"],
            "plugin_scores": {
                name: int(best["plugin_scores"][name]) for name in PLUGIN_NAMES
            },
            "plugin_contributions": {
                name: int(best["plugin_contributions"][name])
                for name in PLUGIN_NAMES
            },
            "error_bound": best["error_bound"],
            "tie_break": list(DEFAULT_TIE_BREAK),
            "filter_trace": filter_trace,
            "receipts": {name: receipts[name] for name in sorted(receipts)},
            "parameter_version": spec.parameter_version,
            "scale_exponent": spec.scale_exponent,
            "scale_factor": spec.scale_factor,
        }
    result["conclusion_digest"] = conclusion_digest(result)
    return result


def _veto_name(code: int) -> str:
    from .model import VETO_CODES

    return VETO_CODES[code]


def schedule_full(spec: Spec, manager: ScoringManager) -> dict[str, Any]:
    results: dict[str, Any] = {}
    bindings: dict[str, str] = {}
    for pod_id in sorted(spec.pod_by_id):
        pod = spec.pod_by_id[pod_id]
        result = _schedule_pod(spec, pod, manager, bindings)
        results[pod_id] = result
        if result["status"] == "bound":
            bindings[pod_id] = result["node_name"]
    return _state_payload(spec, results)


def schedule_incremental(
    spec: Spec,
    manager: ScoringManager,
    previous_state: Mapping[str, Any],
    affected_pod_ids: Iterable[str],
) -> dict[str, Any]:
    previous_results = previous_state.get("results", {})
    if not isinstance(previous_results, dict):
        previous_results = {}
    affected = set(affected_pod_ids)
    results: dict[str, Any] = {}
    bindings: dict[str, str] = {}
    for pod_id, result in sorted(previous_results.items()):
        if pod_id in affected or pod_id not in spec.pod_by_id:
            continue
        results[pod_id] = copy.deepcopy(result)
        reused_versions = sorted(
            {
                int(reply.get("parameter_version", 0))
                for reply in results[pod_id].get("receipts", {}).values()
                if isinstance(reply, dict)
            }
        )
        results[pod_id]["reused_receipts"] = True
        results[pod_id]["receipt_parameter_versions"] = reused_versions
        if result.get("status") == "bound":
            bindings[pod_id] = str(result["node_name"])
    for pod_id in sorted(affected):
        pod = spec.pod_by_id[pod_id]
        result = _schedule_pod(spec, pod, manager, bindings)
        results[pod_id] = result
        if result["status"] == "bound":
            bindings[pod_id] = result["node_name"]
    for pod_id in sorted(spec.pod_by_id):
        if pod_id not in results:
            pod = spec.pod_by_id[pod_id]
            result = _schedule_pod(spec, pod, manager, bindings)
            results[pod_id] = result
            if result["status"] == "bound":
                bindings[pod_id] = result["node_name"]
    return _state_payload(spec, results)


def _state_payload(spec: Spec, results: Mapping[str, Any]) -> dict[str, Any]:
    from .invalidation import spec_metadata

    payload = spec_metadata(spec)
    payload["results"] = {
        pod_id: results[pod_id] for pod_id in sorted(results)
    }
    payload["receipt_catalog"] = {
        f"{reply.get('parameter_version')}:{pod_id}:{node_name}": reply
        for pod_id in sorted(results)
        for node_name, reply in sorted(results[pod_id].get("receipts", {}).items())
    }
    payload["conclusion_digests"] = {
        pod_id: results[pod_id].get("conclusion_digest", conclusion_digest(results[pod_id]))
        for pod_id in sorted(results)
    }
    return payload


def compare_results_stable(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> tuple[str, str, str] | None:
    pod_ids = sorted(set(left.get("results", {})) | set(right.get("results", {})))
    for pod_id in pod_ids:
        left_result = left.get("results", {}).get(pod_id)
        right_result = right.get("results", {}).get(pod_id)
        if left_result is None or right_result is None:
            return pod_id, "-", "missing_result"
        if conclusion_bytes(left_result) == conclusion_bytes(right_result):
            continue
        left_stable = stable_conclusion(left_result)
        right_stable = stable_conclusion(right_result)
        for field in left_stable:
            if left_stable[field] != right_stable[field]:
                return pod_id, str(left_result.get("node_name")), field
    return None
