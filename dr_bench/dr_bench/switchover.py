"""Deterministic switch condition evaluation and transaction arbitration."""

from __future__ import annotations

from typing import Any

from .models import (
    CODE_BUSY,
    CODE_COOLDOWN,
    CODE_MANUAL,
    CODE_NO_TARGET,
    CODE_OK,
    CODE_SWITCHED,
    ROLE_PRIMARY,
    ROLE_REPLICA,
    STATE_HEALTHY,
    STATE_UNHEALTHY,
    STRATEGY_MANUAL,
    InstanceState,
    RuntimeInstance,
    healthy_members,
    primary_of,
)


def health_degree(state: InstanceState) -> int:
    """Explicit, order-only health score used by target selection."""

    if state.state == STATE_HEALTHY:
        return 1_000_000 + state.success_streak
    if state.state == STATE_UNHEALTHY:
        return -1_000_000 + state.failure_streak
    return 0


def select_target(
    service_id: str,
    instance_states: dict[str, InstanceState],
    runtime: dict[str, RuntimeInstance],
) -> str | None:
    """Stable selection: health degree desc, version desc, instance id asc."""

    current_primary = primary_of(runtime, service_id)
    candidates = [
        instance_id
        for instance_id, item in runtime.items()
        if item.service_id == service_id
        and instance_id != current_primary
        and instance_states[instance_id].state == STATE_HEALTHY
    ]
    if not candidates:
        return None
    candidates.sort(
        key=lambda instance_id: (
            -health_degree(instance_states[instance_id]),
            -runtime[instance_id].version,
            instance_id,
        )
    )
    return candidates[0]


def evaluate_trigger(
    service_id: str,
    config,
    instance_states: dict[str, InstanceState],
    runtime: dict[str, RuntimeInstance],
) -> list[str]:
    reasons: list[str] = []
    primary_id = primary_of(runtime, service_id)
    if instance_states[primary_id].state == STATE_UNHEALTHY:
        reasons.append("primary_unhealthy")
    healthy = healthy_members(instance_states, runtime, service_id)
    if len(healthy) < config.min_healthy_replicas:
        reasons.append("below_min_healthy_replicas")
    return reasons


def _commit_switch(
    service_id: str,
    target_id: str,
    runtime: dict[str, RuntimeInstance],
    round_index: int,
    request_id: str,
    decision_at_ms: int,
) -> dict[str, Any]:
    primary_id = primary_of(runtime, service_id)
    member_ids = sorted(
        instance_id
        for instance_id, item in runtime.items()
        if item.service_id == service_id
    )
    old_roles = {instance_id: runtime[instance_id].role for instance_id in member_ids}
    old_versions = {instance_id: runtime[instance_id].version for instance_id in member_ids}
    next_version = max(old_versions.values()) + 1
    runtime[primary_id].role = ROLE_REPLICA
    runtime[target_id].role = ROLE_PRIMARY
    runtime[target_id].version = next_version
    new_roles = {instance_id: runtime[instance_id].role for instance_id in member_ids}
    new_versions = {instance_id: runtime[instance_id].version for instance_id in member_ids}
    return {
        "service_id": service_id,
        "round_index": round_index,
        "request_id": request_id,
        "old_primary": primary_id,
        "new_primary": target_id,
        "promoted_version": next_version,
        "old_roles": old_roles,
        "new_roles": new_roles,
        "old_versions": old_versions,
        "new_versions": new_versions,
        "committed_at_ms": decision_at_ms,
    }


def arbitrate_requests(request_ids: list[str]) -> str:
    """Deterministic winner: lowest request id; every order is explicit."""

    return sorted(request_ids)[0]


def evaluate_service(
    scene,
    service_id: str,
    round_index: int,
    decision_at_ms: int,
    instance_states: dict[str, InstanceState],
    runtime: dict[str, RuntimeInstance],
    last_switch_at_ms: dict[str, int | None],
    extra_requests: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate one service for one round and apply at most one transaction."""

    config = scene.services[service_id]
    trigger_reasons = evaluate_trigger(service_id, config, instance_states, runtime)
    healthy = healthy_members(instance_states, runtime, service_id)
    primary_id = primary_of(runtime, service_id)

    requests: list[str] = []
    if trigger_reasons:
        requests.append(f"auto:{service_id}")
    for request in sorted(extra_requests, key=lambda item: item["request_id"]):
        if request["service_id"] == service_id and request["round_index"] == round_index:
            requests.append(request["request_id"])

    busy: list[dict[str, str]] = []
    winner = arbitrate_requests(requests) if requests else None
    if winner is not None:
        for request_id in sorted(requests):
            if request_id != winner:
                busy.append(
                    {
                        "request_id": request_id,
                        "code": CODE_BUSY,
                        "reason": f"lost arbitration to {winner}",
                    }
                )

    code = CODE_OK
    switch_record = None
    cooldown_remaining_ms = 0
    previous_switch = last_switch_at_ms.get(service_id)
    if previous_switch is not None and config.cooldown_ms > 0:
        elapsed = decision_at_ms - previous_switch
        if elapsed < config.cooldown_ms:
            cooldown_remaining_ms = config.cooldown_ms - elapsed

    if winner is not None:
        if config.strategy == STRATEGY_MANUAL:
            code = CODE_MANUAL
        elif cooldown_remaining_ms > 0:
            code = CODE_COOLDOWN
        else:
            target_id = select_target(service_id, instance_states, runtime)
            if target_id is None:
                code = CODE_NO_TARGET
            else:
                switch_record = _commit_switch(
                    service_id,
                    target_id,
                    runtime,
                    round_index,
                    winner,
                    decision_at_ms,
                )
                code = CODE_SWITCHED
                last_switch_at_ms[service_id] = decision_at_ms

    return {
        "service_id": service_id,
        "round_index": round_index,
        "decision_at_ms": decision_at_ms,
        "primary_instance_id": primary_of(runtime, service_id),
        "healthy_instances": sorted(healthy),
        "available_instances": sorted(healthy),
        "trigger_reasons": trigger_reasons,
        "code": code,
        "cooldown_remaining_ms": cooldown_remaining_ms,
        "switch": switch_record,
        "busy": busy,
        "request_ids": sorted(requests),
    }
