"""Full re-evaluation, incremental migration and round-diff synthesis paths."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

from .canonical import canonical_bytes
from .models import (
    ROLE_PRIMARY,
    InstanceState,
    RuntimeInstance,
    clone_runtime,
    primary_of,
)
from .statemachine import apply_observation, initial_states
from .switchover import evaluate_service

PATH_FULL = "full_rebuild"
PATH_INCREMENTAL = "incremental_migration"
PATH_DIFF = "round_diff_synthesis"
PATHS = (PATH_FULL, PATH_INCREMENTAL, PATH_DIFF)


def initial_runtime(scene) -> dict[str, RuntimeInstance]:
    return {
        instance_id: RuntimeInstance(
            instance_id=instance_id,
            service_id=instance.service_id,
            role=instance.role,
            version=instance.version,
        )
        for instance_id, instance in sorted(scene.instances.items())
    }


@dataclass
class EngineState:
    scene: Any
    instance_states: dict[str, InstanceState]
    runtime: dict[str, RuntimeInstance]
    last_switch_at_ms: dict[str, int | None] = field(default_factory=dict)
    cursor_round: int | None = None

    @classmethod
    def fresh(cls, scene) -> "EngineState":
        return cls(
            scene=scene,
            instance_states=initial_states(scene),
            runtime=initial_runtime(scene),
            last_switch_at_ms={service_id: None for service_id in scene.services},
        )


def decision_timestamp(scene, round_index: int, service_id: str) -> int:
    members = scene.service_members[service_id]
    timestamps = [
        scene.by_instance_round[instance_id][round_index].observed_at_ms
        for instance_id in members
        if round_index in scene.by_instance_round.get(instance_id, {})
    ]
    if timestamps:
        return max(timestamps)
    return 0


def observed_services(scene, round_index: int) -> set[str]:
    affected: set[str] = set()
    for instance_id, table in scene.by_instance_round.items():
        if round_index in table:
            affected.add(scene.instances[instance_id].service_id)
    return affected


def build_round_conclusion(
    scene,
    round_index: int,
    engine: EngineState,
    observed_set: set[str],
    service_decisions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    instance_records = {}
    for instance_id in sorted(scene.instances):
        instance_records[instance_id] = engine.instance_states[instance_id].snapshot(
            round_index, instance_id in observed_set
        )
    available = {
        service_id: service_decisions[service_id]["available_instances"]
        for service_id in sorted(service_decisions)
    }
    targets = {
        service_id: (
            service_decisions[service_id]["switch"]["new_primary"]
            if service_decisions[service_id]["switch"]
            else None
        )
        for service_id in sorted(service_decisions)
    }
    codes = {
        service_id: service_decisions[service_id]["code"]
        for service_id in sorted(service_decisions)
    }
    return {
        "round_index": round_index,
        "instance_states": instance_records,
        "service_decisions": {
            service_id: service_decisions[service_id]
            for service_id in sorted(service_decisions)
        },
        "available_instances": available,
        "switch_targets": targets,
        "business_codes": codes,
        "primary_instances": {
            service_id: primary_of(engine.runtime, service_id)
            for service_id in sorted(scene.services)
        },
        "runtime_roles": {
            instance_id: engine.runtime[instance_id].role
            for instance_id in sorted(engine.runtime)
        },
        "runtime_versions": {
            instance_id: engine.runtime[instance_id].version
            for instance_id in sorted(engine.runtime)
        },
    }


def _carry_decision(previous: dict[str, Any] | None, service_id: str, round_index: int):
    if previous is None:
        return None
    carried = deepcopy(previous["service_decisions"][service_id])
    carried["round_index"] = round_index
    return carried


def initial_conclusion(scene, round_index: int, engine: EngineState) -> dict[str, Any]:
    """Round-zero baseline covering every service, including untouched ones."""

    decisions: dict[str, dict[str, Any]] = {}
    for service_id in sorted(scene.services):
        decisions[service_id] = {
            "service_id": service_id,
            "round_index": round_index,
            "decision_at_ms": 0,
            "primary_instance_id": primary_of(engine.runtime, service_id),
            "healthy_instances": [],
            "available_instances": [],
            "trigger_reasons": [],
            "code": "OK",
            "cooldown_remaining_ms": 0,
            "switch": None,
            "busy": [],
            "request_ids": [],
        }
    return build_round_conclusion(scene, round_index, engine, set(), decisions)


def advance_round(
    scene,
    engine: EngineState,
    round_index: int,
    *,
    observed_only: bool,
    previous_conclusion: dict[str, Any] | None = None,
    extra_requests: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    extra_requests = extra_requests or []
    observed_set: set[str] = set()
    affected_services: set[str] = set()
    for instance_id in sorted(scene.instances):
        observation = scene.by_instance_round.get(instance_id, {}).get(round_index)
        if observation is None:
            continue
        observed_set.add(instance_id)
        affected_services.add(scene.instances[instance_id].service_id)
        config = scene.services[observation.service_id]
        apply_observation(engine.instance_states[instance_id], config, observation)

    for request in extra_requests:
        if request["round_index"] == round_index:
            affected_services.add(request["service_id"])

    service_decisions: dict[str, dict[str, Any]] = {}
    for service_id in sorted(scene.services):
        if service_id in affected_services:
            timestamp = decision_timestamp(scene, round_index, service_id)
            decision = evaluate_service(
                scene,
                service_id,
                round_index,
                timestamp,
                engine.instance_states,
                engine.runtime,
                engine.last_switch_at_ms,
                extra_requests,
            )
            service_decisions[service_id] = decision
            continue
        carried = _carry_decision(previous_conclusion, service_id, round_index)
        if carried is not None:
            service_decisions[service_id] = carried

    engine.cursor_round = round_index
    return build_round_conclusion(
        scene, round_index, engine, observed_set, service_decisions
    )


def run_full(scene, up_to_round: int | None = None) -> list[dict[str, Any]]:
    """Reference path: every round scans every instance and every service.

    Services without an observation in the round keep their previous decision
    while their instance state records are still rebuilt and flagged as having
    no observation, so the state set stays complete on every round.
    """

    engine = EngineState.fresh(scene)
    conclusions: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for round_index in scene.rounds:
        conclusion = advance_round(
            scene,
            engine,
            round_index,
            observed_only=False,
            previous_conclusion=previous,
        )
        conclusions.append(conclusion)
        previous = conclusion
        if up_to_round is not None and round_index >= up_to_round:
            break
    return conclusions


def run_incremental(
    scene,
    up_to_round: int | None = None,
    extra_requests: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Only observed instances migrate; only affected services re-decide."""

    engine = EngineState.fresh(scene)
    conclusions: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for round_index in scene.rounds:
        if previous is None:
            previous = initial_conclusion(scene, round_index, engine)
        conclusion = advance_round(
            scene,
            engine,
            round_index,
            observed_only=True,
            previous_conclusion=previous,
            extra_requests=extra_requests,
        )
        conclusions.append(conclusion)
        previous = conclusion
        if up_to_round is not None and round_index >= up_to_round:
            break
    return conclusions


def _diff_round(
    previous: dict[str, Any] | None, current: dict[str, Any]
) -> dict[str, Any]:
    if previous is None:
        return {
            "round_index": current["round_index"],
            "changed_instances": dict(current["instance_states"]),
            "changed_services": dict(current["service_decisions"]),
            "full_replacement": True,
        }
    changed_instances: dict[str, Any] = {}
    for instance_id in sorted(current["instance_states"]):
        before = previous["instance_states"][instance_id]
        after = current["instance_states"][instance_id]
        comparable_before = dict(before)
        comparable_after = dict(after)
        if canonical_bytes(comparable_before) != canonical_bytes(comparable_after):
            changed_instances[instance_id] = comparable_after
    changed_services: dict[str, Any] = {}
    for service_id in sorted(current["service_decisions"]):
        before = previous["service_decisions"].get(service_id)
        after = current["service_decisions"][service_id]
        if before is None or canonical_bytes(before) != canonical_bytes(after):
            changed_services[service_id] = after
    return {
        "round_index": current["round_index"],
        "changed_instances": changed_instances,
        "changed_services": changed_services,
        "full_replacement": False,
    }


def _synthesize(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    diff: dict[str, Any],
) -> dict[str, Any]:
    if previous is None or diff["full_replacement"]:
        return deepcopy(current)
    synthesized_instances = deepcopy(previous["instance_states"])
    for instance_id, record in diff["changed_instances"].items():
        synthesized_instances[instance_id] = deepcopy(record)
    synthesized_decisions = deepcopy(previous["service_decisions"])
    for service_id, decision in diff["changed_services"].items():
        synthesized_decisions[service_id] = deepcopy(decision)
    synthesized = {
        "round_index": current["round_index"],
        "instance_states": synthesized_instances,
        "service_decisions": synthesized_decisions,
        "available_instances": deepcopy(current["available_instances"]),
        "switch_targets": deepcopy(current["switch_targets"]),
        "business_codes": deepcopy(current["business_codes"]),
        "primary_instances": deepcopy(current["primary_instances"]),
        "runtime_roles": deepcopy(current["runtime_roles"]),
        "runtime_versions": deepcopy(current["runtime_versions"]),
    }
    return synthesized


def run_diff_synthesis(
    scene,
    up_to_round: int | None = None,
    extra_requests: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Emit round diffs and synthesize full conclusions from the prior round."""

    engine = EngineState.fresh(scene)
    synthesized_rounds: list[dict[str, Any]] = []
    diffs: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for round_index in scene.rounds:
        if previous is None:
            previous = initial_conclusion(scene, round_index, engine)
        current = advance_round(
            scene,
            engine,
            round_index,
            observed_only=True,
            previous_conclusion=previous,
            extra_requests=extra_requests,
        )
        diff = _diff_round(previous, current)
        synthesized = _synthesize(previous, current, diff)
        diffs.append(diff)
        synthesized_rounds.append(synthesized)
        previous = synthesized
        if up_to_round is not None and round_index >= up_to_round:
            break
    return synthesized_rounds, diffs


PATH_RUNNERS = {
    PATH_FULL: lambda scene, extra=None: run_full(scene),
    PATH_INCREMENTAL: lambda scene, extra=None: run_incremental(
        scene, extra_requests=extra
    ),
    PATH_DIFF: lambda scene, extra=None: run_diff_synthesis(scene, extra_requests=extra)[0],
}


def run_path(
    path_name: str,
    scene,
    extra_requests: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    if path_name not in PATH_RUNNERS:
        raise KeyError(f"unknown computation path {path_name!r}")
    return PATH_RUNNERS[path_name](scene, extra_requests)
