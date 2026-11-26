from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .db import connect_readonly
from .errors import VerifyError
from .ledger import load_entries_readonly
from .model import PLUGIN_NAMES, VETO_CODES, Node, Pod, Spec
from .records import load_records
from .scheduler import stable_conclusion
from .scoring import ScoringManager


def _label_match(labels: Mapping[str, str], requirement: Any) -> bool:
    values = set(requirement.values)
    if requirement.operator == "In":
        return requirement.key in labels and labels[requirement.key] in values
    if requirement.operator == "NotIn":
        return requirement.key in labels and labels[requirement.key] not in values
    if requirement.operator == "Exists":
        return requirement.key in labels
    return requirement.key not in labels


def _pod_label_match(labels: Mapping[str, str], requirement: Any) -> bool:
    return _label_match(labels, requirement)


def _toleration_allows(taint: Any, pod: Pod) -> bool:
    for toleration in pod.tolerations:
        if toleration.effect not in {-1, taint.effect}:
            continue
        if toleration.operator == "Exists" and toleration.key in {"", taint.key}:
            return True
        if (
            toleration.operator == "Equal"
            and toleration.key == taint.key
            and toleration.value == taint.value
        ):
            return True
    return False


def _anti_conflicts(
    pod: Pod,
    node: Node,
    requirements: Any,
    bindings: Mapping[str, str],
    spec: Spec,
) -> list[tuple[Pod, Any]]:
    result: list[tuple[Pod, Any]] = []
    for other_id in sorted(bindings):
        if other_id == pod.pod_id:
            continue
        other = spec.pod_by_id.get(other_id)
        other_node = spec.node_by_name.get(bindings[other_id])
        if other is None or other_node is None:
            continue
        for requirement in requirements:
            if not _pod_label_match(other.labels, requirement):
                continue
            candidate_topology = node.labels.get(requirement.topology_key)
            other_topology = other_node.labels.get(requirement.topology_key)
            if (
                candidate_topology is not None
                and other_topology is not None
                and candidate_topology == other_topology
            ):
                result.append((other, requirement))
    return result


def _independently_score_pod(
    spec: Spec,
    pod: Pod,
    manager: ScoringManager,
    bindings: Mapping[str, str],
) -> dict[str, Any]:
    feasible: list[tuple[Node, int]] = []
    rejected_codes: list[int] = []
    filter_trace: list[dict[str, Any]] = []
    for node_name in sorted(spec.node_by_name):
        node = spec.node_by_name[node_name]
        steps: list[dict[str, Any]] = []
        decision = None
        prefer_penalty = 0
        taint_reason = "all blocking taints are tolerated"
        for taint in node.taints:
            tolerated = _toleration_allows(taint, pod)
            if not tolerated and taint.effect in {0, 2}:
                decision = 10
                taint_reason = (
                    f"untolerated taint {taint.key}={taint.value} "
                    f"effect={taint.effect_name}"
                )
                break
            if not tolerated and taint.effect == 1:
                prefer_penalty += 1
        steps.append(
            {
                "order": 1,
                "plugin": "taint_toleration",
                "result": "reject" if decision is not None else "pass",
                "reason": taint_reason,
            }
        )
        if decision is not None:
            rejected_codes.append(decision)
            filter_trace.append({"node_name": node_name, "steps": steps})
            continue
        failed_requirement = next(
            (
                term
                for term in pod.node_hard
                if not _label_match(node.labels, term)
            ),
            None,
        )
        if failed_requirement is not None:
            rejected_codes.append(20)
            steps.append(
                {
                    "order": 2,
                    "plugin": "node_affinity",
                    "result": "reject",
                    "reason": (
                        f"hard node affinity {failed_requirement.key} "
                        f"{failed_requirement.operator} "
                        f"{list(failed_requirement.values)} unmet"
                    ),
                }
            )
            filter_trace.append({"node_name": node_name, "steps": steps})
            continue
        steps.append(
            {
                "order": 2,
                "plugin": "node_affinity",
                "result": "pass",
                "reason": "hard node affinity terms satisfied",
            }
        )
        hard_conflicts = _anti_conflicts(pod, node, pod.anti_hard, bindings, spec)
        if hard_conflicts:
            rejected_codes.append(30)
            steps.append(
                {
                    "order": 3,
                    "plugin": "pod_anti_affinity",
                    "result": "reject",
                    "reason": "hard pod anti-affinity conflict with "
                    + ",".join(sorted({other.pod_id for other, _ in hard_conflicts})),
                }
            )
            filter_trace.append({"node_name": node_name, "steps": steps})
            continue
        steps.append(
            {
                "order": 3,
                "plugin": "pod_anti_affinity",
                "result": "pass",
                "reason": "no hard anti-affinity conflict",
            }
        )
        if pod.request_cpu > node.cpu or pod.request_memory > node.memory:
            rejected_codes.append(40)
            if pod.request_cpu > node.cpu:
                reason = (
                    f"cpu request {pod.request_cpu} exceeds allocatable {node.cpu}"
                )
            else:
                reason = (
                    f"memory request {pod.request_memory} exceeds allocatable {node.memory}"
                )
            steps.append(
                {
                    "order": 4,
                    "plugin": "resource_fit",
                    "result": "reject",
                    "reason": reason,
                }
            )
            filter_trace.append({"node_name": node_name, "steps": steps})
            continue
        steps.append(
            {
                "order": 4,
                "plugin": "resource_fit",
                "result": "pass",
                "reason": "request fits allocatable resources",
            }
        )
        filter_trace.append({"node_name": node_name, "steps": steps})
        feasible.append((node, prefer_penalty))

    receipts: dict[str, dict[str, Any]] = {}
    for node, prefer_penalty in feasible:
        soft_match = sum(
            term.weight
            for term in pod.node_soft
            if _label_match(node.labels, term)
        )
        soft_conflicts = _anti_conflicts(
            pod, node, pod.anti_soft, bindings, spec
        )
        features = {
            "has_taints": bool(node.taints),
            "untolerated_prefer_taints": prefer_penalty,
            "has_node_affinity": bool(pod.node_hard or pod.node_soft),
            "node_soft_match_weight": soft_match,
            "has_pod_anti_affinity": bool(pod.anti_hard or pod.anti_soft),
            "anti_soft_conflicts": sum(
                requirement.weight for _, requirement in soft_conflicts
            ),
            "remaining_cpu": node.cpu - pod.request_cpu,
            "remaining_memory": node.memory - pod.request_memory,
            "allocatable_cpu": node.cpu,
            "allocatable_memory": node.memory,
        }
        receipts[node.name] = manager.score(
            pod_id=pod.pod_id,
            node_name=node.name,
            parameter_version=spec.parameter_version,
            scale_exponent=spec.scale_exponent,
            weights_myriad=spec.weights_myriad,
            features=features,
        )

    if not receipts:
        veto_code = (
            50 if len(set(rejected_codes)) != 1 else rejected_codes[0]
        )
        return {
            "pod_id": pod.pod_id,
            "node_name": "-",
            "status": "unbound",
            "veto_code": veto_code,
            "veto_name": VETO_CODES[veto_code],
            "total_score": 0,
            "total_fixed": 0,
            "plugin_scores": {name: 0 for name in PLUGIN_NAMES},
            "plugin_contributions": {name: 0 for name in PLUGIN_NAMES},
            "error_bound": 0,
            "tie_break": [*PLUGIN_NAMES, "node_name"],
            "filter_trace": filter_trace,
        }
    best = min(
        receipts.values(),
        key=lambda reply: (
            -int(reply["total_fixed"]),
            *(-int(reply["plugin_scores"][name]) for name in PLUGIN_NAMES),
            str(reply["node_name"]),
        ),
    )
    return {
        "pod_id": pod.pod_id,
        "node_name": best["node_name"],
        "status": "bound",
        "veto_code": 0,
        "veto_name": VETO_CODES[0],
        "total_score": best["total_score"],
        "total_fixed": best["total_fixed"],
        "plugin_scores": {
            name: int(best["plugin_scores"][name]) for name in PLUGIN_NAMES
        },
        "plugin_contributions": {
            name: int(best["plugin_contributions"][name]) for name in PLUGIN_NAMES
        },
        "error_bound": best["error_bound"],
        "tie_break": [*PLUGIN_NAMES, "node_name"],
        "filter_trace": filter_trace,
    }


def _ledger_meta(db_path: str | Path) -> dict[str, str]:
    connection = connect_readonly(db_path)
    try:
        return {
            str(row["key"]): str(row["value"])
            for row in connection.execute("SELECT key,value FROM ledger_meta")
        }
    finally:
        connection.close()


def verify_ledger(
    ledger_path: str | Path,
    spec: Spec,
    records_path: str | Path,
) -> dict[str, Any]:
    record_batch = load_records(records_path)
    entries = load_entries_readonly(ledger_path)
    meta = _ledger_meta(ledger_path)
    ledger_versions = tuple(json.loads(meta.get("snapshot.format_version_set", "[]")))
    expected_versions = record_batch.version_set
    if tuple(ledger_versions) != tuple(expected_versions):
        raise VerifyError(
            "ledger format version set differs from the records declaration",
            node_name="-",
            details={
                "field": "format_versions",
                "ledger": list(ledger_versions),
                "disk": expected_versions,
            },
        )

    recomputed: dict[str, dict[str, Any]] = {}
    bindings: dict[str, str] = {}
    with ScoringManager(spec) as manager:
        for pod_id in sorted(spec.pod_by_id):
            pod = spec.pod_by_id[pod_id]
            result = _independently_score_pod(spec, pod, manager, bindings)
            recomputed[pod_id] = result
            if result["status"] == "bound":
                bindings[pod_id] = str(result["node_name"])

    for pod_id in sorted(set(entries) | set(recomputed)):
        entry = entries.get(pod_id)
        expected = recomputed.get(pod_id)
        if entry is None or expected is None:
            raise VerifyError(
                "ledger and independent recomputation have different pod sets",
                pod_id=pod_id,
                node_name=(expected or {}).get("node_name", "-"),
                details={
                    "field": "pod_set",
                    "missing_from": "ledger" if entry is None else "recomputation",
                },
            )
        if entry.get("status") != "committed" or not isinstance(
            entry.get("conclusion"), dict
        ):
            raise VerifyError(
                "ledger contains an uncommitted pod conclusion",
                pod_id=pod_id,
                node_name=str(entry.get("node_name", "-")),
                details={"field": "status", "actual": entry.get("status")},
            )
        expected_conclusion = stable_conclusion(expected)
        ledger_conclusion = entry["conclusion"]
        for field in (
            "node_name",
            "status",
            "veto_code",
            "total_fixed",
            "total_score",
            "plugin_scores",
            "plugin_contributions",
            "error_bound",
            "tie_break",
            "filter_trace",
        ):
            if ledger_conclusion.get(field) != expected_conclusion.get(field):
                raise VerifyError(
                    "independent verification found a differing conclusion field",
                    pod_id=pod_id,
                    node_name=str(expected_conclusion.get("node_name", "-")),
                    details={
                        "field": field,
                        "ledger": ledger_conclusion.get(field),
                        "recomputed": expected_conclusion.get(field),
                    },
                )

    return {
        "ok": True,
        "verified_pods": sorted(recomputed),
        "format_version_set": list(expected_versions),
        "snapshot_content_digest": meta.get("snapshot.content_digest"),
        "mode": "readonly-ledger+full-independent-recompute",
    }
