from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .errors import SpecError
from .model import PLUGIN_NAMES, Pod, Spec
from .util import canonical_json, digest_value


STATE_SCHEMA = "affinity-scheduler-state-v1"


def pod_fingerprint(pod: Pod) -> str:
    return digest_value(
        {
            "id": pod.pod_id,
            "labels": dict(pod.labels),
            "requests": {"cpu": pod.request_cpu, "memory": pod.request_memory},
            "node_hard": [term.__dict__ for term in pod.node_hard],
            "node_soft": [term.__dict__ for term in pod.node_soft],
            "anti_hard": [term.__dict__ for term in pod.anti_hard],
            "anti_soft": [term.__dict__ for term in pod.anti_soft],
            "tolerations": [term.__dict__ for term in pod.tolerations],
        }
    )


def node_fingerprint(node: Any) -> str:
    return digest_value(
        {
            "name": node.name,
            "labels": dict(node.labels),
            "cpu": node.cpu,
            "memory": node.memory,
            "taints": [taint.__dict__ for taint in node.taints],
        }
    )


def spec_metadata(spec: Spec) -> dict[str, Any]:
    return {
        "schema": STATE_SCHEMA,
        "parameter_version": spec.parameter_version,
        "scale_exponent": spec.scale_exponent,
        "scale_factor": spec.scale_factor,
        "weights_myriad": {name: spec.weights_myriad[name] for name in PLUGIN_NAMES},
        "nodes": {
            node.name: node_fingerprint(node) for node in sorted(spec.nodes, key=lambda item: item.name)
        },
        "pods": {
            pod.pod_id: pod_fingerprint(pod) for pod in sorted(spec.pods, key=lambda item: item.pod_id)
        },
    }


@dataclass(frozen=True)
class InvalidationReport:
    affected_pods: tuple[str, ...]
    affected_nodes: tuple[str, ...]
    steps: tuple[Mapping[str, Any], ...]
    parameter_version_changed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "affected_pods": list(self.affected_pods),
            "affected_nodes": list(self.affected_nodes),
            "steps": [dict(step) for step in self.steps],
            "parameter_version_changed": self.parameter_version_changed,
        }


def _receipt_nodes(result: Mapping[str, Any]) -> list[str]:
    receipts = result.get("receipts", {})
    if not isinstance(receipts, dict):
        return []
    return sorted(str(node) for node in receipts)


def _plugin_was_observable(result: Mapping[str, Any], plugin: str) -> bool:
    receipts = result.get("receipts", {})
    if not isinstance(receipts, dict):
        return True
    for reply in receipts.values():
        if not isinstance(reply, dict):
            continue
        scores = reply.get("plugin_scores", {})
        if isinstance(scores, dict) and int(scores.get(plugin, 0)) > 0:
            return True
    return False


def _label_change_may_affect_others(pod: Pod) -> bool:
    return bool(pod.anti_hard or pod.anti_soft)


def compute_invalidations(
    previous: Mapping[str, Any] | None,
    spec: Spec,
) -> InvalidationReport:
    """Derive the affected pod/node set from prior state and current declarations."""

    current = spec_metadata(spec)
    all_pods = tuple(sorted(spec.pod_by_id))
    if previous is None:
        steps = [
            {
                "step": 1,
                "pod_id": pod_id,
                "source": "state:absent",
                "target": f"pod:{pod_id}",
                "basis": "initial state has no committed receipt for this pod",
            }
            for pod_id in all_pods
        ]
        return InvalidationReport(
            affected_pods=all_pods,
            affected_nodes=tuple(sorted(spec.node_by_name)),
            steps=tuple(steps),
            parameter_version_changed=True,
        )

    previous_version = int(previous.get("parameter_version", 0))
    if spec.parameter_version < previous_version:
        raise SpecError(
            f"parameter_version must be monotonic non-decreasing: {spec.parameter_version} < {previous_version}",
            field_path="parameter_version",
            source_line=1,
            source_path=spec.source_path,
        )
    version_changed = spec.parameter_version != previous_version

    affected: set[str] = set()
    step_rows: list[dict[str, Any]] = []
    results = previous.get("results", {})
    if not isinstance(results, dict):
        results = {}

    def add(pod_id: str, source: str, basis: str) -> None:
        if pod_id not in spec.pod_by_id:
            return
        if pod_id not in affected:
            affected.add(pod_id)
        step_rows.append(
            {
                "step": len(step_rows) + 1,
                "pod_id": pod_id,
                "source": source,
                "target": f"pod:{pod_id}",
                "basis": basis,
            }
        )

    previous_nodes = previous.get("nodes", {})
    current_nodes = current["nodes"]
    if not isinstance(previous_nodes, dict):
        previous_nodes = {}
    changed_node_set = sorted(set(previous_nodes) ^ set(current_nodes))
    changed_node_content = sorted(
        name
        for name in set(previous_nodes) & set(current_nodes)
        if previous_nodes[name] != current_nodes[name]
    )
    if changed_node_set or changed_node_content:
        reason = (
            f"node declaration changed: added/removed={changed_node_set}, "
            f"content={changed_node_content}"
        )
        for pod_id in all_pods:
            add(pod_id, "node:declaration", reason)

    previous_scale = int(previous.get("scale_exponent", -1))
    if previous_scale != spec.scale_exponent:
        reason = (
            f"scale_exponent changed {previous_scale}->{spec.scale_exponent}; "
            "every fixed-point contribution denominator changes"
        )
        for pod_id in all_pods:
            add(pod_id, "parameter:scale_exponent", reason)

    previous_weights_raw = previous.get("weights_myriad", {})
    if not isinstance(previous_weights_raw, dict):
        previous_weights_raw = {}
    for plugin in PLUGIN_NAMES:
        old_weight = previous_weights_raw.get(plugin)
        new_weight = spec.weights_myriad[plugin]
        if old_weight == new_weight:
            continue
        for pod_id in all_pods:
            result = results.get(pod_id, {})
            if not isinstance(result, dict):
                add(
                    pod_id,
                    f"plugin_weight:{plugin}",
                    f"weight {old_weight}->{new_weight}; no reusable receipt is available",
                )
                continue
            if _plugin_was_observable(result, plugin):
                add(
                    pod_id,
                    f"plugin_weight:{plugin}",
                    f"weight {old_weight}->{new_weight}; candidate receipts expose a non-zero {plugin} score",
                )

    previous_pods = previous.get("pods", {})
    current_pods = current["pods"]
    if not isinstance(previous_pods, dict):
        previous_pods = {}
    changed_pod_ids = sorted(
        pod_id
        for pod_id in set(previous_pods) | set(current_pods)
        if previous_pods.get(pod_id) != current_pods.get(pod_id)
    )
    for pod_id in changed_pod_ids:
        if pod_id in spec.pod_by_id:
            add(
                pod_id,
                "pod:declaration",
                "pod declaration changed or has no previous declaration",
            )
    anti_pods = [
        pod for pod in spec.pods if pod.anti_hard or pod.anti_soft
    ]
    if changed_pod_ids and anti_pods:
        for pod in anti_pods:
            add(
                pod.pod_id,
                "pod_anti_affinity:closure",
                f"pod declaration set changed: {changed_pod_ids}; anti-affinity outcomes may move",
            )

    if version_changed and not affected:
        step_rows.append(
            {
                "step": len(step_rows) + 1,
                "pod_id": None,
                "source": "parameter_version",
                "target": "metadata:only",
                "basis": (
                    f"parameter_version {previous_version}->{spec.parameter_version}; "
                    "effective weights, scale and node/pod declarations are unchanged"
                ),
            }
        )

    affected_nodes: set[str] = set()
    for pod_id in affected:
        result = results.get(pod_id, {})
        if isinstance(result, dict):
            affected_nodes.update(_receipt_nodes(result))
    for node_name in changed_node_set + changed_node_content:
        affected_nodes.add(node_name)

    ordered_steps = sorted(
        step_rows,
        key=lambda row: (
            str(row.get("pod_id") or ""),
            int(row.get("step", 0)),
            str(row.get("source", "")),
        ),
    )
    for index, row in enumerate(ordered_steps, start=1):
        row["step"] = index
    steps_sorted = tuple(ordered_steps)
    return InvalidationReport(
        affected_pods=tuple(sorted(affected)),
        affected_nodes=tuple(sorted(affected_nodes)),
        steps=steps_sorted,
        parameter_version_changed=version_changed,
    )


def state_metadata_json(spec: Spec) -> str:
    return canonical_json(spec_metadata(spec))
