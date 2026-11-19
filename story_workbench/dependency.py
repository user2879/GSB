"""Exact dependency-graph invalidation for timeline, conditions, and endings."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

from .canonical import double, edge_sequence_hash, replay_hash, sha256_json, state_hash
from .engine import (
    Arbitration,
    BranchEvaluation,
    CausalChain,
    CausalContribution,
    ConditionJudgment,
    apply_edge,
    project_node_outgoing_effects,
)
from .errors import E_DEPENDENCY_INVALID, WorkbenchError
from .expressions import render_expression, replace_coefficient
from .model import Edge, NarrativeSpec
from .timeline import make_snapshot


@dataclass(frozen=True)
class DependencyNode:
    id: str
    kind: str
    dependencies: tuple[str, ...]
    edge_id: str | None = None
    step_index: int | None = None
    condition_id: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "dependencies": list(self.dependencies),
            "edge_id": self.edge_id,
            "step_index": self.step_index,
            "condition_id": self.condition_id,
        }


@dataclass
class CachedNode:
    node: DependencyNode
    value: Any
    version: int
    content_hash: str


class IncrementalDependencyGraph:
    """A per-sequence graph that only executes builder nodes in an exact closure."""

    def __init__(self, spec: NarrativeSpec, sequence_id: str) -> None:
        self.spec = spec
        self.sequence = spec.sample_by_id(sequence_id)
        self.sequence_id = self.sequence.id
        self.topology: tuple[DependencyNode, ...] = ()
        self.topology_order: dict[str, int] = {}
        self.reverse_dependencies: dict[str, tuple[str, ...]] = {}
        self.key_owners: dict[str, tuple[str, ...]] = {}
        self.cache: dict[str, CachedNode] = {}
        self.last_recomputed: tuple[str, ...] = ()
        self._build_topology()
        self._build_all()

    def _build_topology(self) -> None:
        spec = self.spec
        sequence = self.sequence
        nodes: list[DependencyNode] = []
        key_owners: dict[str, list[str]] = {}

        initial_id = f"initial_state:{sequence.id}"
        nodes.append(DependencyNode(initial_id, "path_state", ()))
        for component in spec.components:
            key_owners[f"initial:{component}:value"] = [initial_id]

        for edge in spec.edges:
            edge_node_id = f"edge_effect:{edge.id}"
            nodes.append(DependencyNode(edge_node_id, "edge_effect", (), edge_id=edge.id))
            key_owners[f"effect_set:{edge.id}"] = [edge_node_id]
            for key in edge.coefficient_keys:
                key_owners[key] = [edge_node_id]

        step_nodes: list[str] = [spec.start]
        for edge_id in sequence.edges:
            step_nodes.append(spec.edge_by_id(edge_id).target)

        condition_ids_by_step: dict[int, list[str]] = {}
        for step_index, node_id in enumerate(step_nodes):
            raw_id = f"raw_state:{sequence.id}:{step_index}"
            if step_index == 0:
                raw_dependencies = (initial_id,)
            else:
                previous_raw = f"raw_state:{sequence.id}:{step_index - 1}"
                chosen_edge = spec.edge_by_id(sequence.edges[step_index - 1])
                chosen_conditions = tuple(
                    f"condition:{sequence.id}:{step_index - 1}:{chosen_edge.id}:{condition.id}"
                    for condition in chosen_edge.conditions
                )
                raw_dependencies = (
                    previous_raw,
                    f"edge_effect:{chosen_edge.id}",
                    *chosen_conditions,
                )
            nodes.append(
                DependencyNode(
                    raw_id,
                    "raw_state",
                    raw_dependencies,
                    step_index=step_index,
                )
            )

            condition_ids: list[str] = []
            for edge in spec.outgoing(node_id):
                for condition in edge.conditions:
                    condition_id = f"condition:{sequence.id}:{step_index}:{edge.id}:{condition.id}"
                    condition_ids.append(condition_id)
                    nodes.append(
                        DependencyNode(
                            condition_id,
                            "condition",
                            (raw_id,),
                            edge_id=edge.id,
                            step_index=step_index,
                            condition_id=condition.id,
                        )
                    )
                    key_owners.setdefault(condition.threshold_key, []).append(condition_id)
            condition_ids_by_step[step_index] = condition_ids

            outgoing_edge_rules = tuple(f"edge_effect:{edge.id}" for edge in spec.outgoing(node_id))
            path_node_id = f"path_state:{sequence.id}:{step_index}"
            nodes.append(
                DependencyNode(
                    path_node_id,
                    "path_state",
                    (raw_id, *condition_ids, *outgoing_edge_rules),
                    step_index=step_index,
                )
            )

        ending_id = f"ending:{sequence.id}"
        last_path = f"path_state:{sequence.id}:{len(sequence.edges)}"
        nodes.append(DependencyNode(ending_id, "ending", (last_path,)))

        self.topology = tuple(nodes)
        self.topology_order = {node.id: index for index, node in enumerate(nodes)}
        reverse: dict[str, list[str]] = {node.id: [] for node in nodes}
        for node in nodes:
            for dependency in node.dependencies:
                if dependency not in reverse:
                    raise WorkbenchError(
                        E_DEPENDENCY_INVALID,
                        f"dependency graph references unknown node {dependency!r}",
                    )
                reverse[dependency].append(node.id)
        self.reverse_dependencies = {
            node_id: tuple(sorted(dependents, key=lambda item: self.topology_order[item]))
            for node_id, dependents in reverse.items()
        }
        self.key_owners = {
            key: tuple(sorted(set(owners), key=lambda item: self.topology_order[item]))
            for key, owners in key_owners.items()
        }

    def _build_all(self) -> None:
        self.cache = {}
        for node in self.topology:
            self._build_node(node, version=1)
        self.last_recomputed = tuple(node.id for node in self.topology)

    def _build_node(self, node: DependencyNode, *, version: int) -> Any:
        value = self._execute_node(node)
        content_hash = sha256_json(value)
        self.cache[node.id] = CachedNode(node=node, value=value, version=version, content_hash=content_hash)
        return value

    def _execute_node(self, node: DependencyNode) -> Any:
        if node.kind == "path_state" and node.id.startswith("initial_state:"):
            return self._initial_value()
        if node.kind == "edge_effect":
            return self._edge_rule_value(node.edge_id or "")
        if node.kind == "raw_state":
            return self._raw_value(node.step_index or 0)
        if node.kind == "condition":
            return self._condition_value(node.step_index or 0, node.edge_id or "", node.condition_id or "")
        if node.kind == "path_state":
            return self._path_value(node.step_index or 0)
        if node.kind == "ending":
            return self._ending_value()
        raise WorkbenchError(E_DEPENDENCY_INVALID, f"unknown dependency node kind {node.kind!r}")

    def _initial_value(self) -> dict[str, Any]:
        state = tuple(double(value) for value in self.spec.initial_state)
        chains = [
            CausalChain(component=component, initial=state[index], value=state[index], contributions=())
            for index, component in enumerate(self.spec.components)
        ]
        return {
            "state_vector": list(state),
            "state": {component: state[index] for index, component in enumerate(self.spec.components)},
            "state_hash": state_hash(self.spec.components, state),
            "causal_chains": [chain.as_dict() for chain in chains],
        }

    def _edge_rule_value(self, edge_id: str) -> dict[str, Any]:
        edge = self.spec.edge_by_id(edge_id)
        return {
            "edge_id": edge.id,
            "edge_index": edge.index,
            "source": edge.source,
            "target": edge.target,
            "effects": [effect.as_dict() for effect in edge.effects],
            "coefficient_keys": list(edge.coefficient_keys),
            "trigger_keys": list(edge.trigger_keys),
            "effects_present": bool(edge.effects),
        }

    def _condition_value(self, step_index: int, edge_id: str, condition_id: str) -> dict[str, Any]:
        raw = self.cache[f"raw_state:{self.sequence_id}:{step_index}"].value
        edge = self.spec.edge_by_id(edge_id)
        condition = next(condition for condition in edge.conditions if condition.id == condition_id)
        state = raw["state"]
        from .engine import evaluate_condition

        judgment = evaluate_condition(condition, state[condition.component], self.spec.boundary_tolerance)
        return judgment.as_dict()

    def _condition_judgments_for_edge(
        self,
        step_index: int,
        edge: Edge,
    ) -> tuple[ConditionJudgment, ...]:
        judgments: list[ConditionJudgment] = []
        for condition in edge.conditions:
            node_id = f"condition:{self.sequence_id}:{step_index}:{edge.id}:{condition.id}"
            payload = self.cache[node_id].value
            judgments.append(ConditionJudgment(**payload))
        return tuple(judgments)

    def _branch_status(self, judgments: Sequence[ConditionJudgment]) -> tuple[str, str]:
        if not judgments:
            return "reachable", "no-conditions"
        if any(judgment.status == "unsatisfied" for judgment in judgments):
            return "unreachable", "hard-condition-failed"
        if any(judgment.status == "boundary" for judgment in judgments):
            return "boundary", "condition-on-boundary"
        return "reachable", "all-conditions-satisfied"

    def _raw_value(self, step_index: int) -> dict[str, Any]:
        sequence = self.sequence
        if step_index == 0:
            initial = self.cache[f"initial_state:{sequence.id}"].value
            node_id = self.spec.start
            node = self.spec.node_by_id(node_id)
            state = tuple(initial["state_vector"])
            chains = self._chains_from_payload(initial["causal_chains"])
            status = "ended" if node.ending is not None else "active"
            return self._raw_payload(
                step_index=0,
                node_id=node_id,
                state=state,
                chains=chains,
                history=(),
                incoming_edge_id=None,
                requested_edge_id=None,
                status=status,
                ending=node.ending,
                edge_application=None,
            )

        previous = self.cache[f"raw_state:{sequence.id}:{step_index - 1}"].value
        if previous["status"] in {"ended", "blocked", "inactive"}:
            chains = self._chains_from_payload(previous["causal_chains"])
            return self._raw_payload(
                step_index=step_index,
                node_id=previous["node_id"],
                state=tuple(previous["state_vector"]),
                chains=chains,
                history=tuple(previous["edge_history"]),
                incoming_edge_id=previous["incoming_edge_id"],
                requested_edge_id=None,
                status="inactive",
                ending=previous["ending"],
                edge_application=None,
            )

        chosen = self.spec.edge_by_id(sequence.edges[step_index - 1])
        judgments = self._condition_judgments_for_edge(step_index - 1, chosen)
        branch_status, _ = self._branch_status(judgments)
        if branch_status != "reachable":
            chains = self._chains_from_payload(previous["causal_chains"])
            return self._raw_payload(
                step_index=step_index,
                node_id=previous["node_id"],
                state=tuple(previous["state_vector"]),
                chains=chains,
                history=tuple(previous["edge_history"]),
                incoming_edge_id=previous["incoming_edge_id"],
                requested_edge_id=chosen.id,
                status="blocked",
                ending=f"blocked:{previous['node_id']}",
                edge_application=None,
            )

        chains = self._chains_from_payload(previous["causal_chains"])
        application = apply_edge(self.spec, chosen, previous["state_vector"], chains)
        history = (*previous["edge_history"], chosen.id)
        target = self.spec.node_by_id(chosen.target)
        status = "ended" if target.ending is not None else "active"
        return self._raw_payload(
            step_index=step_index,
            node_id=chosen.target,
            state=application.state,
            chains=application.causal_chains,
            history=history,
            incoming_edge_id=chosen.id,
            requested_edge_id=None,
            status=status,
            ending=target.ending,
            edge_application=application.as_dict(),
        )

    def _raw_payload(
        self,
        *,
        step_index: int,
        node_id: str,
        state: Sequence[float],
        chains: Sequence[CausalChain],
        history: Sequence[str],
        incoming_edge_id: str | None,
        requested_edge_id: str | None,
        status: str,
        ending: str | None,
        edge_application: dict[str, Any] | None,
    ) -> dict[str, Any]:
        return {
            "sequence_id": self.sequence_id,
            "sequence_step": step_index,
            "node_id": node_id,
            "state_vector": [double(value) for value in state],
            "state": {
                component: double(state[index])
                for index, component in enumerate(self.spec.components)
            },
            "state_hash": state_hash(self.spec.components, state),
            "causal_chains": [chain.as_dict() for chain in chains],
            "edge_history": list(history),
            "incoming_edge_id": incoming_edge_id,
            "requested_edge_id": requested_edge_id,
            "status": status,
            "active": status == "active",
            "ending": ending,
            "edge_application": edge_application,
        }

    def _chains_from_payload(self, payload: Sequence[Mapping[str, Any]]) -> tuple[CausalChain, ...]:
        chains: list[CausalChain] = []
        for chain_payload in payload:
            contributions = tuple(
                CausalContribution(
                    edge_id=contribution["edge_id"],
                    edge_index=contribution["edge_index"],
                    effect_id=contribution["effect_id"],
                    effect_index=contribution["effect_index"],
                    before=double(contribution["before"]),
                    delta=double(contribution["delta"]),
                    after=double(contribution["after"]),
                )
                for contribution in chain_payload["contributions"]
            )
            chains.append(
                CausalChain(
                    component=chain_payload["component"],
                    initial=double(chain_payload["initial"]),
                    value=double(chain_payload["value"]),
                    contributions=contributions,
                )
            )
        return tuple(chains)

    def _path_value(self, step_index: int) -> dict[str, Any]:
        raw = self.cache[f"raw_state:{self.sequence_id}:{step_index}"].value
        node_id = raw["node_id"]
        condition_step = step_index
        if raw["status"] in {"blocked", "inactive"} and step_index > 0:
            condition_step = step_index - 1
        branches: list[BranchEvaluation] = []
        for edge in self.spec.outgoing(node_id):
            judgments = self._condition_judgments_for_edge(condition_step, edge)
            status, reason = self._branch_status(judgments)
            branches.append(
                BranchEvaluation(
                    edge_id=edge.id,
                    edge_index=edge.index,
                    target_node=edge.target,
                    label=edge.label,
                    conditions=judgments,
                    status=status,
                    reason=reason,
                )
            )
        projection = project_node_outgoing_effects(self.spec, node_id, raw["state_vector"])
        chains = self._chains_from_payload(raw["causal_chains"])
        arbitrations: tuple[Arbitration, ...] = ()
        if raw["edge_application"] is not None:
            arbitrations = tuple(
                Arbitration(
                    scope=payload["scope"],
                    edge_id=payload["edge_id"],
                    target=payload["target"],
                    reason=payload["reason"],
                    winner_effect_id=payload["winner_effect_id"],
                    winner_effect_index=payload["winner_effect_index"],
                    overwritten_effect_id=payload["overwritten_effect_id"],
                    overwritten_effect_index=payload["overwritten_effect_index"],
                    overwritten_value=double(payload["overwritten_value"]),
                    effective_value=double(payload["effective_value"]),
                    declaration_order=tuple(payload["declaration_order"]),
                )
                for payload in raw["edge_application"]["arbitrations"]
            )
        snapshot = make_snapshot(
            self.spec,
            sequence_id=self.sequence_id,
            state_version=step_index + 1,
            step_index=step_index,
            node_id=node_id,
            status=raw["status"],
            state_values=raw["state_vector"],
            edge_history=raw["edge_history"],
            incoming_edge_id=raw["incoming_edge_id"],
            requested_edge_id=raw["requested_edge_id"],
            ending=raw["ending"],
            causal_chains=chains,
            arbitrations=arbitrations,
            edge_application=raw["edge_application"],
            branches=branches,
            outgoing_projection=projection,
        )
        return snapshot.as_dict()

    def _ending_value(self) -> dict[str, Any]:
        last_step = len(self.sequence.edges)
        snapshot = self.cache[f"path_state:{self.sequence_id}:{last_step}"].value
        raw = self.cache[f"raw_state:{self.sequence_id}:{last_step}"].value
        if snapshot["ending"] is not None:
            ending = snapshot["ending"]
            status = snapshot["status"]
        elif raw["active"]:
            ending = f"unresolved:{raw['node_id']}"
            status = "unresolved"
        else:
            ending = raw["ending"]
            status = raw["status"]
        state = tuple(raw["state_vector"])
        history = tuple(raw["edge_history"])
        return {
            "sequence_id": self.sequence_id,
            "ending": ending,
            "status": status,
            "final_state_vector": list(state),
            "final_state": {
                component: state[index]
                for index, component in enumerate(self.spec.components)
            },
            "state_hash": state_hash(self.spec.components, state),
            "edge_history": list(history),
            "edge_sequence_hash": edge_sequence_hash(history),
            "replay_hash": replay_hash(history, self.spec.components, state),
        }

    def closure(self, trigger_key: str) -> tuple[str, ...]:
        owners = self.key_owners.get(trigger_key)
        if not owners:
            raise WorkbenchError(
                E_DEPENDENCY_INVALID,
                f"unknown dependency trigger key {trigger_key!r}",
                details={"known_keys": sorted(self.key_owners)},
            )
        affected: set[str] = set(owners)
        queue = list(owners)
        while queue:
            node_id = queue.pop(0)
            for dependent in self.reverse_dependencies[node_id]:
                if dependent not in affected:
                    affected.add(dependent)
                    queue.append(dependent)
        return tuple(sorted(affected, key=lambda node_id: self.topology_order[node_id]))

    def _recompute(self, closure: Sequence[str], trigger_key: str) -> dict[str, Any]:
        closure_set = set(closure)
        old_hashes = {node_id: self.cache[node_id].content_hash for node_id in closure}
        recomputed: list[str] = []
        for node_id in closure:
            old = self.cache[node_id]
            self._build_node(old.node, version=old.version + 1)
            recomputed.append(node_id)
        self.last_recomputed = tuple(recomputed)
        if set(recomputed) != closure_set:  # pragma: no cover - internal invariant
            raise WorkbenchError(
                E_DEPENDENCY_INVALID,
                "recomputed node set differs from invalidation closure",
                details={"closure": list(closure), "recomputed": recomputed},
            )

        reused: dict[str, dict[str, Any]] = {}
        for node_id in recomputed:
            for dependency in self.cache[node_id].node.dependencies:
                if dependency in closure_set:
                    continue
                cached = self.cache[dependency]
                reused.setdefault(
                    dependency,
                    {
                        "node_id": dependency,
                        "version": cached.version,
                        "content_hash": cached.content_hash,
                        "used_by": [],
                    },
                )
                if node_id not in reused[dependency]["used_by"]:
                    reused[dependency]["used_by"].append(node_id)

        return {
            "trigger_key": trigger_key,
            "owners": list(self.key_owners[trigger_key]),
            "closure_nodes": sorted(closure),
            "recomputed_nodes": sorted(recomputed),
            "actual_recomputed_equals_closure": sorted(recomputed) == sorted(closure),
            "closure_size": len(closure),
            "reused_sources": [reused[key] for key in sorted(reused)],
            "node_versions": {node_id: self.cache[node_id].version for node_id in sorted(self.cache)},
            "node_hashes": {node_id: self.cache[node_id].content_hash for node_id in sorted(self.cache)},
            "changed_nodes": [
                {
                    "node_id": node_id,
                    "before_hash": old_hashes[node_id],
                    "after_hash": self.cache[node_id].content_hash,
                    "changed": old_hashes[node_id] != self.cache[node_id].content_hash,
                    "version": self.cache[node_id].version,
                }
                for node_id in recomputed
            ],
            "parameter_version_stamp": f"param-v{max(self.cache[node_id].version for node_id in self.cache):04d}",
        }

    def invalidate(self, trigger_key: str) -> dict[str, Any]:
        closure = self.closure(trigger_key)
        return self._recompute(closure, trigger_key)

    def set_threshold(self, trigger_key: str, value: float) -> dict[str, Any]:
        closure = self.closure(trigger_key)
        located = None
        for edge_index, edge in enumerate(self.spec.edges):
            for condition_index, condition in enumerate(edge.conditions):
                if condition.threshold_key == trigger_key:
                    located = (edge_index, edge, condition_index)
                    break
            if located is not None:
                break
        if located is None:
            raise WorkbenchError(E_DEPENDENCY_INVALID, f"not a threshold key: {trigger_key!r}")
        edge_index, edge, condition_index = located
        conditions = list(edge.conditions)
        conditions[condition_index] = replace(conditions[condition_index], threshold=double(value))
        edges = list(self.spec.edges)
        edges[edge_index] = replace(edge, conditions=tuple(conditions))
        self.spec = self.spec.with_edges(edges)
        return self._recompute(closure, trigger_key)

    def set_coefficient(self, trigger_key: str, value: float) -> dict[str, Any]:
        closure = self.closure(trigger_key)
        located = None
        for edge_index, edge in enumerate(self.spec.edges):
            for effect_index, effect in enumerate(edge.effects):
                for ordinal, key in enumerate(effect.coefficient_keys):
                    if key == trigger_key:
                        located = (edge_index, edge, effect_index, effect, ordinal)
                        break
                if located is not None:
                    break
            if located is not None:
                break
        if located is None:
            raise WorkbenchError(E_DEPENDENCY_INVALID, f"not a coefficient key: {trigger_key!r}")
        edge_index, edge, effect_index, effect, ordinal = located
        new_ast = replace_coefficient(effect.ast, ordinal, double(value))
        new_effect = replace(effect, ast=new_ast, expression=render_expression(new_ast))
        effects = list(edge.effects)
        effects[effect_index] = new_effect
        edges = list(self.spec.edges)
        edges[edge_index] = replace(edge, effects=tuple(effects))
        self.spec = self.spec.with_edges(edges)
        return self._recompute(closure, trigger_key)

    def snapshot_payloads(self) -> list[dict[str, Any]]:
        snapshots: list[dict[str, Any]] = []
        for step_index in range(len(self.sequence.edges) + 1):
            snapshots.append(self.cache[f"path_state:{self.sequence_id}:{step_index}"].value)
        return snapshots

    def node_manifest(self) -> dict[str, Any]:
        return {
            "schema": "narrative-workbench/dependency-manifest/v1",
            "spec_id": self.spec.id,
            "sequence_id": self.sequence_id,
            "nodes": [
                {
                    **node.as_dict(),
                    "version": self.cache[node.id].version,
                    "content_hash": self.cache[node.id].content_hash,
                }
                for node in self.topology
            ],
            "trigger_key_owners": {key: list(owners) for key, owners in sorted(self.key_owners.items())},
        }

    def report(self) -> dict[str, Any]:
        payload = {
            "schema": "narrative-workbench/dependency-report/v1",
            "spec_id": self.spec.id,
            "spec_hash": self.spec.spec_hash,
            "sequence_id": self.sequence_id,
            "graph": self.node_manifest(),
            "snapshots": self.snapshot_payloads(),
            "ending": self.cache[f"ending:{self.sequence_id}"].value,
        }
        payload["report_hash"] = sha256_json(payload)
        return payload
