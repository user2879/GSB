"""Deterministic Filter/Score scheduling with dependency-driven recomputation."""
from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Callable, Iterable

from .deps import (
    DependencyGraph,
    build_graph,
    fact_node_capacity,
    fact_node_images,
    fact_node_label,
    fact_node_taints,
    fact_pod_affinity,
    fact_pod_image,
    fact_pod_labels,
    fact_pod_resources,
    fact_pod_tolerations,
    fact_plugin_weights,
    result_pod,
    stat_anti_count,
    stat_image_locality,
    stat_label_match,
    stat_resource_remaining,
    stat_taint_gate,
)
from .errors import BusinessError, E_INCR, E_REF
from .model import (
    PLUGIN_NAMES,
    AntiRule,
    Node,
    PluginConfig,
    Pod,
    Selector,
    Taint,
    Toleration,
    valid_label_key,
    valid_label_value,
)
from .util import canonical_json, first_deep_difference, sha256_digest


SCORE_SCALE = 1000
TRUNCATION = "floor_toward_zero_non_negative"
FILTER_ORDER = (
    "node_affinity_hard",
    "anti_affinity_hard",
    "taint_toleration",
    "allocatable_resources",
)


def _selector_mismatches(selector: Selector, labels: dict[str, str]) -> list[str]:
    mismatches: list[str] = []
    for key, expected in sorted(selector.match_labels.items()):
        actual = labels.get(key)
        if actual != expected:
            mismatches.append(f"matchLabels[{key}] expected {expected!r}, actual {actual!r}")
    for expression in selector.match_expressions:
        present = expression.key in labels
        actual = labels.get(expression.key)
        if expression.operator == "In":
            if not present or actual not in expression.values:
                mismatches.append(
                    f"{expression.key} In {list(expression.values)!r}, actual {actual!r}"
                )
        elif expression.operator == "NotIn":
            if present and actual in expression.values:
                mismatches.append(
                    f"{expression.key} NotIn {list(expression.values)!r}, actual {actual!r}"
                )
        elif expression.operator == "Exists" and not present:
            mismatches.append(f"{expression.key} Exists, actual absent")
        elif expression.operator == "DoesNotExist" and present:
            mismatches.append(f"{expression.key} DoesNotExist, actual {actual!r}")
    return mismatches


def _labels_for_selector(selector: Selector, labels: dict[str, str]) -> dict[str, str | None]:
    return {key: labels.get(key) for key in sorted(selector.referenced_keys())}


class WorkbenchSession:
    """Mutable session with cached statistics and persistent-ready results."""

    def __init__(
        self,
        nodes: Iterable[Node],
        pods: Iterable[Pod],
        plugins: PluginConfig = PluginConfig(weights={name: 1 for name in PLUGIN_NAMES}, version=1),
    ) -> None:
        self.nodes: dict[str, Node] = {node.name: node for node in nodes}
        self.pods: list[Pod] = list(pods)
        self.pods_by_id: dict[str, Pod] = {pod.pod_id: pod for pod in self.pods}
        self.plugins = plugins.normalized()
        self.node_version = 1
        self.pod_version = 1
        self.plugin_version = self.plugins.version
        self.graph_version = 0
        self.fact_revisions: dict[str, int] = {}
        self.stats: dict[str, dict[str, Any]] = {}
        self.results: dict[str, dict[str, Any]] = {}
        self.graph: DependencyGraph = build_graph(self._ordered_nodes(), self.pods, self.plugins)
        self._initialize_fact_revisions()
        self.last_invalidation: dict[str, Any] | None = None
        self.last_change: dict[str, Any] | None = None
        self.last_recomputed_pods: list[str] = []
        self.last_recomputed_stats: list[str] = []
        self._stat_revision = 0
        self._unaffected_check: dict[str, Any] | None = None

    @classmethod
    def from_data(cls, nodes: Iterable[Node], pods: Iterable[Pod], plugins: PluginConfig) -> "WorkbenchSession":
        return cls(nodes, pods, plugins)

    def _ordered_nodes(self) -> list[Node]:
        return [self.nodes[name] for name in sorted(self.nodes)]

    def _initialize_fact_revisions(self) -> None:
        for node_id, kind in self.graph.nodes.items():
            if kind == "fact" and node_id not in self.fact_revisions:
                self.fact_revisions[node_id] = 1

    def _bump_fact(self, node_id: str) -> None:
        self.fact_revisions[node_id] = self.fact_revisions.get(node_id, 0) + 1

    def _rebuild_graph(self) -> None:
        self.graph = build_graph(self._ordered_nodes(), self.pods, self.plugins)
        self._initialize_fact_revisions()
        self.graph_version += 1

    def content_payload(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self._ordered_nodes()],
            "pods": [pod.to_dict() for pod in self.pods],
            "plugins": self.plugins.to_dict(),
        }

    def content_digest(self) -> str:
        return sha256_digest(self.content_payload())

    def versions(self) -> dict[str, int]:
        return {
            "node": self.node_version,
            "pod": self.pod_version,
            "plugin": self.plugin_version,
            "graph": self.graph_version,
        }

    def _stat_value(self, stat_id: str, producer: Callable[[], dict[str, Any]]) -> dict[str, Any]:
        cached = self.stats.get(stat_id)
        if cached is not None:
            return cached["value"]
        value = producer()
        self._stat_revision += 1
        self.stats[stat_id] = {"value": value, "revision": self._stat_revision}
        return value

    def _label_match_value(self, pod: Pod, node: Node) -> dict[str, Any]:
        hard = pod.node_affinity_hard
        hard_mismatches = [] if hard is None else _selector_mismatches(hard, node.labels)
        soft: list[dict[str, Any]] = []
        for index, rule in enumerate(pod.node_affinity_soft):
            mismatches = _selector_mismatches(rule.selector, node.labels)
            soft.append(
                {
                    "rule_index": index,
                    "weight": rule.weight,
                    "matched": not mismatches,
                    "mismatches": mismatches,
                }
            )
        referenced: dict[str, str | None] = {}
        if hard is not None:
            referenced.update(_labels_for_selector(hard, node.labels))
        for rule in pod.node_affinity_soft:
            referenced.update(_labels_for_selector(rule.selector, node.labels))
        return {
            "node": node.name,
            "hard_matched": not hard_mismatches,
            "hard_mismatches": hard_mismatches,
            "soft": soft,
            "referenced_labels": dict(sorted(referenced.items())),
        }

    def _taint_gate_value(self, pod: Pod, node: Node) -> dict[str, Any]:
        taints: list[dict[str, Any]] = []
        blocking: list[dict[str, Any]] = []
        for taint in node.taints:
            tolerated = any(toleration.tolerates(taint) for toleration in pod.tolerations)
            item = {**taint.to_dict(), "tolerated": tolerated}
            taints.append(item)
            if taint.effect in {"NoSchedule", "NoExecute"} and not tolerated:
                blocking.append(item)
        return {"node": node.name, "accepted": not blocking, "taints": taints, "blocking": blocking}

    def _resource_remaining_value(self, pod: Pod, node: Node) -> dict[str, Any]:
        cpu = node.cpu
        memory = node.memory
        consumers: list[dict[str, Any]] = []
        for prior in self.pods:
            if prior.pod_id == pod.pod_id:
                break
            result = self.results.get(prior.pod_id)
            if result is not None and result.get("selected") == node.name:
                cpu -= prior.cpu
                memory -= prior.memory
                consumers.append({"pod": prior.pod_id, "cpu": prior.cpu, "memory": prior.memory})
        return {
            "node": node.name,
            "capacity": {"cpu": node.cpu, "memory": node.memory},
            "remaining": {"cpu": cpu, "memory": memory},
            "consumers": consumers,
        }

    def _anti_count_value(self, pod: Pod, node: Node) -> dict[str, Any]:
        earlier = self.pods[: self._pod_index(pod.pod_id)]
        matching_pods: set[str] = set()

        def evaluate(rule: AntiRule, hard: bool, index: int) -> dict[str, Any]:
            topology_value = node.labels.get(rule.topology_key)
            matching: list[str] = []
            if topology_value is not None:
                for prior in earlier:
                    result = self.results.get(prior.pod_id)
                    if result is None or result.get("selected") is None:
                        continue
                    selected_node = self.nodes[result["selected"]]
                    if selected_node.labels.get(rule.topology_key) != topology_value:
                        continue
                    if rule.selector.matches(prior.labels):
                        matching.append(prior.pod_id)
            matching_pods.update(matching)
            return {
                "rule_index": index,
                "hard": hard,
                "topology_key": rule.topology_key,
                "topology_value": topology_value,
                "topology_missing": topology_value is None,
                "count": len(matching),
                "matching_pods": sorted(matching),
            }

        hard_rules = [evaluate(rule, True, index) for index, rule in enumerate(pod.anti_affinity_hard)]
        soft_rules = [evaluate(rule, False, index) for index, rule in enumerate(pod.anti_affinity_soft)]
        return {
            "node": node.name,
            "hard": hard_rules,
            "soft": soft_rules,
            "matching_pods": sorted(matching_pods),
        }

    def _image_locality_value(self, pod: Pod, node: Node) -> dict[str, Any]:
        local = pod.image is not None and pod.image in node.images
        return {"node": node.name, "image": pod.image, "local": local, "node_images": list(node.images)}

    def _pod_index(self, pod_id: str) -> int:
        for index, pod in enumerate(self.pods):
            if pod.pod_id == pod_id:
                return index
        raise KeyError(pod_id)

    def _stats_for(self, pod: Pod, node: Node, *, only: set[str] | None = None) -> dict[str, dict[str, Any]]:
        values = {
            "label_match": self._stat_value(
                stat_label_match(pod.pod_id, node.name), lambda: self._label_match_value(pod, node)
            ),
            "taint_gate": self._stat_value(
                stat_taint_gate(pod.pod_id, node.name), lambda: self._taint_gate_value(pod, node)
            ),
            "resource_remaining": self._stat_value(
                stat_resource_remaining(pod.pod_id, node.name),
                lambda: self._resource_remaining_value(pod, node),
            ),
            "anti_count": self._stat_value(
                stat_anti_count(pod.pod_id, node.name), lambda: self._anti_count_value(pod, node)
            ),
            "image_locality": self._stat_value(
                stat_image_locality(pod.pod_id, node.name), lambda: self._image_locality_value(pod, node)
            ),
        }
        if only is not None:
            values = {name: value for name, value in values.items() if name in only}
        return values

    def _filter(self, pod: Pod, node: Node, stats: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
        label_match = stats["label_match"]
        if not label_match["hard_matched"]:
            return {
                "stage": 1,
                "stage_name": FILTER_ORDER[0],
                "reason": "node affinity hard constraint failed: " + "; ".join(label_match["hard_mismatches"]),
            }

        anti = stats["anti_count"]
        for rule in anti["hard"]:
            if rule["topology_missing"]:
                return {
                    "stage": 2,
                    "stage_name": FILTER_ORDER[1],
                    "reason": f"anti-affinity hard rule {rule['rule_index']} topologyKey {rule['topology_key']!r} is missing on node",
                }
            if rule["count"] > 0:
                return {
                    "stage": 2,
                    "stage_name": FILTER_ORDER[1],
                    "reason": (
                        f"anti-affinity hard rule {rule['rule_index']} found {rule['count']} matching pod(s) "
                        f"in topology {rule['topology_key']}={rule['topology_value']!r}: "
                        + ",".join(rule["matching_pods"])
                    ),
                }

        taint_gate = stats["taint_gate"]
        if not taint_gate["accepted"]:
            descriptions = [
                f"{item['key']}={item['value']}:{item['effect']}" for item in taint_gate["blocking"]
            ]
            return {
                "stage": 3,
                "stage_name": FILTER_ORDER[2],
                "reason": "untolerated taint(s): " + ",".join(descriptions),
            }

        remaining = stats["resource_remaining"]["remaining"]
        if (
            remaining["cpu"] < pod.cpu
            or remaining["memory"] < pod.memory
            or remaining["cpu"] < 0
            or remaining["memory"] < 0
        ):
            return {
                "stage": 4,
                "stage_name": FILTER_ORDER[3],
                "reason": (
                    f"insufficient resources: request cpu={pod.cpu},memory={pod.memory}; "
                    f"remaining cpu={remaining['cpu']},memory={remaining['memory']}"
                ),
            }
        return None

    def _raw_scores(self, pod: Pod, node: Node, stats: dict[str, dict[str, Any]]) -> dict[str, int]:
        soft_affinity = stats["label_match"]["soft"]
        node_affinity = 0
        if soft_affinity:
            total_weight = sum(item["weight"] for item in soft_affinity)
            matched_weight = sum(item["weight"] for item in soft_affinity if item["matched"])
            node_affinity = (100 * matched_weight) // total_weight

        anti_affinity = 0
        anti_rules = stats["anti_count"]["soft"]
        if anti_rules:
            total_weight = sum(
                pod.anti_affinity_soft[item["rule_index"]].weight or 0 for item in anti_rules
            )
            weighted = 0
            for item in anti_rules:
                rule_weight = pod.anti_affinity_soft[item["rule_index"]].weight or 0
                rule_score = 0 if item["topology_missing"] else max(0, 100 - 20 * item["count"])
                weighted += rule_score * rule_weight
            anti_affinity = weighted // total_weight if total_weight else 0

        resource = stats["resource_remaining"]
        cpu_after = resource["remaining"]["cpu"] - pod.cpu
        memory_after = resource["remaining"]["memory"] - pod.memory
        cpu_percent = max(0, cpu_after * 100 // resource["capacity"]["cpu"])
        memory_percent = max(0, memory_after * 100 // resource["capacity"]["memory"])
        resource_balance = max(0, 100 - abs(cpu_percent - memory_percent))

        image_locality = 100 if stats["image_locality"]["local"] else 0
        return {
            "node_affinity": node_affinity,
            "anti_affinity": anti_affinity,
            "resource_balance": resource_balance,
            "image_locality": image_locality,
        }

    def _score_node(self, pod: Pod, node: Node, stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
        raw = self._raw_scores(pod, node, stats)
        total_weight = self.plugins.total_weight()
        contributions = {
            name: (raw[name] * self.plugins.weights[name] * SCORE_SCALE) // total_weight
            for name in PLUGIN_NAMES
        }
        remaining = stats["resource_remaining"]["remaining"]
        cpu_after = remaining["cpu"] - pod.cpu
        memory_after = remaining["memory"] - pod.memory
        capacity = stats["resource_remaining"]["capacity"]
        return {
            "raw": raw,
            "weighted_contrib_milli": contributions,
            "total_milli": sum(contributions.values()),
            "remaining_after": {"cpu": cpu_after, "memory": memory_after},
            "min_free_ratio_milli": min(
                max(0, cpu_after * SCORE_SCALE // capacity["cpu"]),
                max(0, memory_after * SCORE_SCALE // capacity["memory"]),
            ),
            "anti_count_total": sum(item["count"] for item in stats["anti_count"]["hard"])
            + sum(item["count"] for item in stats["anti_count"]["soft"]),
        }

    def _score_and_select(self, pod: Pod, candidates: list[Node]) -> tuple[dict[str, Any], str | None]:
        active_plugins = sum(1 for name in PLUGIN_NAMES if self.plugins.weights[name] > 0)
        error_bound = active_plugins
        scores: dict[str, dict[str, Any]] = {}
        for node in candidates:
            stats = self._stats_for(pod, node)
            scores[node.name] = self._score_node(pod, node, stats)
        if not scores:
            return {
                "plugin_weights": dict(sorted(self.plugins.weights.items())),
                "score_scale": SCORE_SCALE,
                "truncation": TRUNCATION,
                "error_bound_milli": error_bound,
                "scores": {},
                "selected": None,
                "selected_score_milli": None,
                "tie_break": {"rule": "no_candidate"},
            }, None

        node_names = sorted(scores)
        best_total = max(scores[name]["total_milli"] for name in node_names)
        band = [name for name in node_names if best_total - scores[name]["total_milli"] < error_bound]
        if len(node_names) == 1:
            selected = node_names[0]
            tie_break = {
                "rule": "unique_candidate",
                "band": band,
                "margin_milli": None,
            }
        elif len(band) == 1:
            selected = band[0]
            runner_up = max(
                (scores[name]["total_milli"] for name in node_names if name != selected), default=best_total
            )
            tie_break = {
                "rule": "unique_score_outside_error_band",
                "band": band,
                "margin_milli": best_total - runner_up,
            }
        else:
            ranked = sorted(
                band,
                key=lambda name: (
                    scores[name]["anti_count_total"],
                    -scores[name]["min_free_ratio_milli"],
                    name,
                ),
            )
            selected = ranked[0]
            tie_break = {
                "rule": "error_band_then[anti_count asc, min_free_ratio desc, node_name asc]",
                "band": band,
                "band_deltas_milli": {name: best_total - scores[name]["total_milli"] for name in band},
                "selection_key": {
                    "anti_count": scores[selected]["anti_count_total"],
                    "min_free_ratio_milli": scores[selected]["min_free_ratio_milli"],
                    "node": selected,
                },
            }
        return {
            "plugin_weights": dict(sorted(self.plugins.weights.items())),
            "score_scale": SCORE_SCALE,
            "truncation": TRUNCATION,
            "error_bound_milli": error_bound,
            "scores": {name: scores[name] for name in node_names},
            "selected": selected,
            "selected_score_milli": scores[selected]["total_milli"],
            "tie_break": tie_break,
        }, selected

    def _compute_pod_result(self, pod: Pod, version: int) -> dict[str, Any]:
        candidates: list[Node] = []
        filtered: list[dict[str, Any]] = []
        for node in self._ordered_nodes():
            stats = self._stats_for(pod, node)
            rejection = self._filter(pod, node, stats)
            if rejection is None:
                candidates.append(node)
            else:
                filtered.append({"node": node.name, **rejection})
        scored, selected = self._score_and_select(pod, candidates)
        result: dict[str, Any] = {
            "pod": pod.pod_id,
            "queue_index": self._pod_index(pod.pod_id),
            "version": version,
            "filter_order": list(FILTER_ORDER),
            "candidates": [node.name for node in candidates],
            "filtered": filtered,
            "selected": selected,
            "unschedulable_reason": (
                None if selected is not None else f"no candidate node passed Filter ({len(filtered)} rejected)"
            ),
            **scored,
        }
        return result

    def _recompute_pod(self, pod: Pod) -> dict[str, Any]:
        old = self.results.get(pod.pod_id)
        version = int(old.get("version", 0)) + 1 if old is not None else 1
        new_result = self._compute_pod_result(pod, version)
        if old is not None:
            old_content = {key: value for key, value in old.items() if key != "version"}
            new_content = {key: value for key, value in new_result.items() if key != "version"}
            if canonical_json(old_content) == canonical_json(new_content):
                new_result = {**new_result, "version": old["version"]}
        self.results[pod.pod_id] = new_result
        return new_result

    def _topological_fact_ids(self) -> list[str]:
        return [node_id for node_id in self.graph.topological_order() if self.graph.nodes[node_id] == "fact"]

    def full_schedule(self) -> dict[str, Any]:
        """Recompute every statistic and every pod from scratch."""
        self._rebuild_graph()
        self.stats.clear()
        self.results.clear()
        roots = self._topological_fact_ids()
        report = self.graph.invalidation_report(roots, "initial-full-schedule")
        self.last_invalidation = report
        self.last_change = {"source": "initial-full-schedule", "roots": roots}
        self.last_recomputed_pods = [pod.pod_id for pod in self.pods]
        self.last_recomputed_stats = list(report["stats"])
        self._unaffected_check = None
        for pod in self.pods:
            self._recompute_pod(pod)
        return report

    def _incremental(self, roots: list[str], source: str) -> dict[str, Any]:
        self._rebuild_graph()
        report = self.graph.invalidation_report(roots, source)
        affected_pods = set(report["pods"])
        before = {
            pod_id: canonical_json(self.results[pod_id])
            for pod_id in self.results
            if pod_id not in affected_pods
        }
        invalidated_stats = set(report["stats"])
        for stat_id in invalidated_stats:
            self.stats.pop(stat_id, None)

        recomputed: list[str] = []
        for pod in self.pods:
            if pod.pod_id in affected_pods:
                self._recompute_pod(pod)
                recomputed.append(pod.pod_id)

        after = {
            pod_id: canonical_json(self.results[pod_id])
            for pod_id in self.results
            if pod_id not in affected_pods
        }
        for pod_id in sorted(before):
            if before[pod_id] != after.get(pod_id):
                raise BusinessError(
                    E_INCR,
                    f"unaffected pod changed during incremental recomputation: {pod_id}",
                    details={"pod": pod_id},
                )
        self._unaffected_check = {
            "count": len(before),
            "pod_ids": sorted(before),
            "byte_identical": before == after,
            "versions_unchanged": True,
        }
        self.last_invalidation = report
        self.last_change = {"source": source, "roots": roots}
        self.last_recomputed_pods = recomputed
        self.last_recomputed_stats = [stat_id for stat_id in report["stats"] if stat_id in invalidated_stats]
        return report

    def _commit_mutation(
        self,
        mutate: Callable[[], list[str]],
        source: str,
        *,
        version_bumps: dict[str, int],
    ) -> dict[str, Any]:
        old_nodes = dict(self.nodes)
        old_pods = list(self.pods)
        old_pods_by_id = dict(self.pods_by_id)
        old_plugins = self.plugins
        old_versions = (self.node_version, self.pod_version, self.plugin_version, self.graph_version)
        old_fact_revisions = dict(self.fact_revisions)
        old_stats = dict(self.stats)
        old_results = dict(self.results)
        old_last_invalidation = self.last_invalidation
        old_last_change = self.last_change
        old_last_recomputed_pods = self.last_recomputed_pods
        old_last_recomputed_stats = self.last_recomputed_stats
        old_unaffected_check = self._unaffected_check
        try:
            roots = mutate()
            self.node_version += version_bumps.get("node", 0)
            self.pod_version += version_bumps.get("pod", 0)
            self.plugin_version += version_bumps.get("plugin", 0)
            for root in roots:
                self._bump_fact(root)
            return self._incremental(roots, source)
        except Exception:
            self.nodes = old_nodes
            self.pods = old_pods
            self.pods_by_id = old_pods_by_id
            self.plugins = old_plugins
            self.node_version, self.pod_version, self.plugin_version, self.graph_version = old_versions
            self.fact_revisions = old_fact_revisions
            self.stats = old_stats
            self.results = old_results
            self.last_invalidation = old_last_invalidation
            self.last_change = old_last_change
            self.last_recomputed_pods = old_last_recomputed_pods
            self.last_recomputed_stats = old_last_recomputed_stats
            self._unaffected_check = old_unaffected_check
            self.graph = build_graph(self._ordered_nodes(), self.pods, self.plugins)
            raise

    def edit_node_label(self, node_name: str, key: str, value: str | None) -> dict[str, Any]:
        if node_name not in self.nodes:
            raise BusinessError(E_REF, f"unknown node '{node_name}'", details={"node": node_name})
        if not valid_label_key(key):
            raise BusinessError(E_REF, f"invalid label key '{key}'", details={"key": key})
        if value is not None and not valid_label_value(value):
            raise BusinessError(E_REF, f"invalid label value '{value}'", details={"key": key, "value": value})
        if self.nodes[node_name].labels.get(key) == value:
            return {
                "source": f"node-label:{node_name}:{key}",
                "noop": True,
                "roots": [],
                "steps": [],
                "stats": [],
                "pods": [],
            }

        def mutate() -> list[str]:
            node = self.nodes[node_name]
            labels = dict(node.labels)
            if value is None:
                labels.pop(key, None)
            else:
                labels[key] = value
            self.nodes[node_name] = replace(node, labels=labels)
            self._validate_runtime_references()
            return [fact_node_label(node_name, key)]

        return self._commit_mutation(
            mutate,
            f"node-label:{node_name}:{key}",
            version_bumps={"node": 1},
        )

    def edit_node_taints(self, node_name: str, taints: Iterable[Taint]) -> dict[str, Any]:
        if node_name not in self.nodes:
            raise BusinessError(E_REF, f"unknown node '{node_name}'", details={"node": node_name})

        def mutate() -> list[str]:
            self.nodes[node_name] = replace(self.nodes[node_name], taints=tuple(taints))
            return [fact_node_taints(node_name)]

        return self._commit_mutation(mutate, f"node-taints:{node_name}", version_bumps={"node": 1})

    def edit_node_capacity(self, node_name: str, cpu: int, memory: int) -> dict[str, Any]:
        if node_name not in self.nodes:
            raise BusinessError(E_REF, f"unknown node '{node_name}'", details={"node": node_name})
        if isinstance(cpu, bool) or not isinstance(cpu, int) or cpu <= 0:
            raise BusinessError(E_REF, "node CPU must be a positive integer")
        if isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0:
            raise BusinessError(E_REF, "node memory must be a positive integer")

        def mutate() -> list[str]:
            self.nodes[node_name] = replace(self.nodes[node_name], cpu=cpu, memory=memory)
            self._validate_runtime_references()
            return [fact_node_capacity(node_name)]

        return self._commit_mutation(mutate, f"node-capacity:{node_name}", version_bumps={"node": 1})

    def edit_node_images(self, node_name: str, images: Iterable[str]) -> dict[str, Any]:
        if node_name not in self.nodes:
            raise BusinessError(E_REF, f"unknown node '{node_name}'", details={"node": node_name})

        def mutate() -> list[str]:
            self.nodes[node_name] = replace(self.nodes[node_name], images=tuple(images))
            return [fact_node_images(node_name)]

        return self._commit_mutation(mutate, f"node-images:{node_name}", version_bumps={"node": 1})

    def edit_pod_labels(self, pod_id: str, labels: dict[str, str]) -> dict[str, Any]:
        pod = self._require_pod(pod_id)
        for key, value in labels.items():
            if not valid_label_key(key) or not valid_label_value(value):
                raise BusinessError(E_REF, f"invalid pod label {key!r}={value!r}")

        def mutate() -> list[str]:
            index = self._pod_index(pod_id)
            self.pods[index] = replace(self.pods[index], labels=dict(labels))
            self.pods_by_id[pod_id] = self.pods[index]
            return [fact_pod_labels(pod_id)]

        return self._commit_mutation(mutate, f"pod-labels:{pod_id}", version_bumps={"pod": 1})

    def edit_pod_resources(self, pod_id: str, cpu: int, memory: int) -> dict[str, Any]:
        self._require_pod(pod_id)
        if isinstance(cpu, bool) or not isinstance(cpu, int) or cpu <= 0:
            raise BusinessError(E_REF, "pod CPU must be a positive integer")
        if isinstance(memory, bool) or not isinstance(memory, int) or memory <= 0:
            raise BusinessError(E_REF, "pod memory must be a positive integer")

        def mutate() -> list[str]:
            self._validate_resource_limit(cpu, memory)
            index = self._pod_index(pod_id)
            self.pods[index] = replace(self.pods[index], cpu=cpu, memory=memory)
            self.pods_by_id[pod_id] = self.pods[index]
            return [fact_pod_resources(pod_id)]

        return self._commit_mutation(mutate, f"pod-resources:{pod_id}", version_bumps={"pod": 1})

    def edit_pod_constraints(self, pod_id: str, **changes: Any) -> dict[str, Any]:
        self._require_pod(pod_id)
        allowed = {
            "node_affinity_hard",
            "node_affinity_soft",
            "anti_affinity_hard",
            "anti_affinity_soft",
            "tolerations",
            "image",
        }
        unknown = sorted(set(changes) - allowed)
        if unknown:
            raise BusinessError(E_REF, "unknown pod constraint field(s)", details={"fields": unknown})

        root_map = {
            "image": fact_pod_image(pod_id),
            "tolerations": fact_pod_tolerations(pod_id),
        }

        def mutate() -> list[str]:
            index = self._pod_index(pod_id)
            self.pods[index] = replace(self.pods[index], **changes)
            self.pods_by_id[pod_id] = self.pods[index]
            self._validate_runtime_references()
            roots = [fact_pod_affinity(pod_id)]
            for field in changes:
                root = root_map.get(field)
                if root is not None:
                    roots.append(root)
            return sorted(set(roots))

        return self._commit_mutation(mutate, f"pod-constraints:{pod_id}", version_bumps={"pod": 1})

    def set_plugin_weights(self, weights: dict[str, int], *, version: int | None = None) -> dict[str, Any]:
        normalized = PluginConfig(weights=weights, version=version or self.plugin_version + 1).normalized()
        if set(normalized.weights) != set(PLUGIN_NAMES):
            raise BusinessError(E_REF, "plugin weight set is incomplete")
        if any(value < 0 for value in normalized.weights.values()) or normalized.total_weight() <= 0:
            raise BusinessError(E_REF, "plugin weights must be non-negative and have a positive total")

        def mutate() -> list[str]:
            self.plugins = normalized
            return [fact_plugin_weights()]

        return self._commit_mutation(mutate, "plugin-weights", version_bumps={"plugin": 1})

    def _require_pod(self, pod_id: str) -> Pod:
        if pod_id not in self.pods_by_id:
            raise BusinessError(E_REF, f"unknown pod '{pod_id}'", details={"pod": pod_id})
        return self.pods_by_id[pod_id]

    def _validate_resource_limit(self, cpu: int, memory: int) -> None:
        max_cpu = max(node.cpu for node in self.nodes.values())
        max_memory = max(node.memory for node in self.nodes.values())
        if cpu > max_cpu or memory > max_memory:
            raise BusinessError(
                E_INCR,
                "pod request exceeds maximum single-node allocatable resource",
                details={"cpu": cpu, "memory": memory, "maxCpu": max_cpu, "maxMemory": max_memory},
            )

    def _validate_runtime_references(self) -> None:
        node_label_keys = {key for node in self.nodes.values() for key in node.labels}
        pod_label_keys = {key for pod in self.pods for key in pod.labels}
        for pod in self.pods:
            for key in sorted(pod.node_selector_keys()):
                if key not in node_label_keys:
                    raise BusinessError(
                        E_REF,
                        f"label key '{key}' is referenced by pod '{pod.pod_id}' but does not occur on any node",
                        details={"pod": pod.pod_id, "key": key},
                    )
            for key in sorted(pod.anti_selector_keys()):
                if key not in pod_label_keys:
                    raise BusinessError(
                        E_REF,
                        f"label key '{key}' is used by pod '{pod.pod_id}' anti-affinity but does not occur in any pod labels",
                        details={"pod": pod.pod_id, "key": key},
                    )
            self._validate_resource_limit(pod.cpu, pod.memory)

    def result_content(self, pod_id: str) -> dict[str, Any]:
        result = self.results[pod_id]
        return {key: value for key, value in result.items() if key != "version"}

    def verify_full_consistency(self) -> dict[str, Any]:
        """Compare current incremental state with an independent full recomputation."""
        fresh = WorkbenchSession.from_data(self._ordered_nodes(), self.pods, self.plugins)
        fresh.full_schedule()
        pod_ids = [pod.pod_id for pod in self.pods]
        for pod_id in pod_ids:
            expected = self.result_content(pod_id)
            actual = fresh.result_content(pod_id)
            diff = first_deep_difference(expected, actual)
            if diff is not None:
                raise BusinessError(
                    E_INCR,
                    f"incremental result differs from full recomputation at pod {pod_id}",
                    details={"pod": pod_id, **diff},
                )

        current_stats = {
            stat_id: canonical_json(cache["value"]) for stat_id, cache in sorted(self.stats.items())
        }
        fresh_stats = {
            stat_id: canonical_json(cache["value"]) for stat_id, cache in sorted(fresh.stats.items())
        }
        if current_stats != fresh_stats:
            differing = sorted(
                stat_id
                for stat_id in set(current_stats) | set(fresh_stats)
                if current_stats.get(stat_id) != fresh_stats.get(stat_id)
            )
            first_pod = self._first_pod_for_stats(differing)
            raise BusinessError(
                E_INCR,
                f"incremental statistics differ from full recomputation at pod {first_pod}",
                details={
                    "pod": first_pod,
                    "field": "stats",
                    "differingStats": differing,
                },
            )
        return {
            "ok": True,
            "podCount": len(pod_ids),
            "statCount": len(current_stats),
            "unaffectedCheck": self._unaffected_check,
        }

    def _first_pod_for_stats(self, stat_ids: list[str]) -> str:
        stat_set = set(stat_ids)
        for pod in self.pods:
            for node_name in self.nodes:
                for stat_id in (
                    stat_label_match(pod.pod_id, node_name),
                    stat_taint_gate(pod.pod_id, node_name),
                    stat_resource_remaining(pod.pod_id, node_name),
                    stat_anti_count(pod.pod_id, node_name),
                    stat_image_locality(pod.pod_id, node_name),
                ):
                    if stat_id in stat_set:
                        return pod.pod_id
        return self.pods[0].pod_id

    def diagnostic_state(self) -> dict[str, Any]:
        """Stable, JSON-ready diagnostic view shared by CLI and Tkinter."""
        return {
            "versions": self.versions(),
            "digest": self.content_digest(),
            "graph": {
                "signature": self.graph.signature(),
                "nodes": len(self.graph.nodes),
                "edges": sum(len(value) for value in self.graph.edges.values()),
            },
            "pods": [
                {
                    "pod": pod.pod_id,
                    "queue_index": index,
                    "result": copy.deepcopy(self.results.get(pod.pod_id)),
                }
                for index, pod in enumerate(self.pods)
            ],
            "invalidation": copy.deepcopy(self.last_invalidation),
            "recomputed_pods": list(self.last_recomputed_pods),
            "recomputed_stats": list(self.last_recomputed_stats),
            "unaffected_check": copy.deepcopy(self._unaffected_check),
        }

    def stats_payload(self) -> dict[str, Any]:
        return {stat_id: copy.deepcopy(cache["value"]) for stat_id, cache in sorted(self.stats.items())}

    def results_payload(self) -> list[dict[str, Any]]:
        return [copy.deepcopy(self.results[pod.pod_id]) for pod in self.pods]
