"""Deterministic pressure application, preemption, and convergence engine."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .allocation import allocation_by_tenant, derive_entitlements
from .canonical import canonical_json, digest_value, sha256_bytes
from .spec import Declaration, TriggerEvent, Workload


SELECTION_RULE = (
    "overage tenants by priority ASC, tenant_id ASC; "
    "preemptible workloads only; startup_order DESC, workload_id ASC; "
    "one replica per convergence round; never reduce below tenant reserve"
)


@dataclass
class RuntimeState:
    running_replicas: dict[str, int]
    pressure_by_tenant: dict[str, int]

    @classmethod
    def initial(cls, declaration: Declaration) -> "RuntimeState":
        return cls(
            running_replicas={
                workload.workload_id: workload.requested_replicas
                for workload in declaration.sorted_workloads()
            },
            pressure_by_tenant={
                tenant.tenant_id: 0 for tenant in declaration.sorted_tenants()
            },
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any], declaration: Declaration) -> "RuntimeState":
        running = data.get("running_replicas", {})
        pressure = data.get("pressure_by_tenant", {})
        if not isinstance(running, dict) or not isinstance(pressure, dict):
            raise ValueError("runtime state maps are invalid")
        result_running: dict[str, int] = {}
        for workload in declaration.sorted_workloads():
            value = running.get(workload.workload_id, workload.requested_replicas)
            if not isinstance(value, int) or value < 0 or value > workload.requested_replicas:
                raise ValueError(f"invalid running replica count for {workload.workload_id}")
            result_running[workload.workload_id] = value
        result_pressure: dict[str, int] = {}
        for tenant in declaration.sorted_tenants():
            value = pressure.get(tenant.tenant_id, 0)
            if not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid pressure for {tenant.tenant_id}")
            result_pressure[tenant.tenant_id] = value
        return cls(result_running, result_pressure)

    def clone(self) -> "RuntimeState":
        return RuntimeState(dict(self.running_replicas), dict(self.pressure_by_tenant))

    def to_dict(self, declaration: Declaration) -> dict[str, Any]:
        return {
            "running_replicas": {
                workload.workload_id: self.running_replicas[workload.workload_id]
                for workload in declaration.sorted_workloads()
            },
            "pressure_by_tenant": {
                tenant.tenant_id: self.pressure_by_tenant.get(tenant.tenant_id, 0)
                for tenant in declaration.sorted_tenants()
            },
        }


@dataclass
class EventResult:
    event_id: int
    record: dict[str, Any]
    record_json: str
    record_digest: str
    state: RuntimeState
    terminal_code: str

    @property
    def unsatisfied(self) -> bool:
        return self.terminal_code == "E_UNSATISFIED"


def _workload_usage_by_tenant(
    declaration: Declaration, state: RuntimeState
) -> dict[str, int]:
    usage = {tenant.tenant_id: 0 for tenant in declaration.sorted_tenants()}
    for workload in declaration.sorted_workloads():
        usage[workload.tenant_id] += (
            state.running_replicas[workload.workload_id] * workload.units_per_replica
        )
    return usage


def build_matrix(declaration: Declaration, state: RuntimeState) -> dict[str, Any]:
    return derive_entitlements(
        declaration,
        workload_units=_workload_usage_by_tenant(declaration, state),
        pressure_units=state.pressure_by_tenant,
    )


def _has_excess(matrix: dict[str, Any]) -> bool:
    return matrix["cluster_excess_units"] > 0 or matrix["tenant_excess_total_units"] > 0


def _candidate(
    declaration: Declaration,
    state: RuntimeState,
    matrix: dict[str, Any],
) -> tuple[Workload, dict[str, Any]] | None:
    rows = allocation_by_tenant(matrix)
    overage_ids = sorted(
        (row["tenant_id"] for row in matrix["tenants"] if row["excess_units"] > 0),
        key=lambda tenant_id: (rows[tenant_id]["priority"], tenant_id),
    )
    workloads_by_tenant: dict[str, list[Workload]] = {
        tenant.tenant_id: [] for tenant in declaration.sorted_tenants()
    }
    for workload in declaration.sorted_workloads():
        workloads_by_tenant[workload.tenant_id].append(workload)
    for tenant_id in overage_ids:
        candidates = sorted(
            workloads_by_tenant[tenant_id],
            key=lambda item: (-item.startup_order, item.workload_id),
        )
        row = rows[tenant_id]
        for workload in candidates:
            running = state.running_replicas[workload.workload_id]
            if not workload.preemptible or running <= 0:
                continue
            usage_after = row["actual_units"] - workload.units_per_replica
            if usage_after < row["reserve_units"]:
                continue
            basis = {
                "rule": SELECTION_RULE,
                "tenant_priority": row["priority"],
                "tenant_id": tenant_id,
                "tenant_excess_before": row["excess_units"],
                "workload_id": workload.workload_id,
                "preemptible": workload.preemptible,
                "startup_order": workload.startup_order,
                "units_per_replica": workload.units_per_replica,
                "running_replicas_before": running,
                "reserve_guard": (
                    f"{row['actual_units']}-{workload.units_per_replica}>="
                    f"{row['reserve_units']}"
                ),
            }
            return workload, basis
    return None


def _blocked_requests(
    declaration: Declaration, state: RuntimeState, matrix: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = allocation_by_tenant(matrix)
    blocked: list[dict[str, Any]] = []
    for tenant in declaration.sorted_tenants():
        row = rows[tenant.tenant_id]
        if row["excess_units"] <= 0:
            continue
        workloads = sorted(
            (item for item in declaration.workloads if item.tenant_id == tenant.tenant_id),
            key=lambda item: (-item.startup_order, item.workload_id),
        )
        for workload in workloads:
            running = state.running_replicas[workload.workload_id]
            if running <= 0:
                continue
            if not workload.preemptible:
                reason = "NON_PREEMPTIBLE"
            elif row["actual_units"] - workload.units_per_replica < row["reserve_units"]:
                reason = "RESERVE_FLOOR"
            else:
                continue
            blocked.append(
                {
                    "tenant_id": tenant.tenant_id,
                    "workload_id": workload.workload_id,
                    "running_replicas": running,
                    "units_per_replica": workload.units_per_replica,
                    "reclaimable_units": running * workload.units_per_replica,
                    "reason": reason,
                }
            )
    return blocked


def _gap(matrix: dict[str, Any], blocked: list[dict[str, Any]]) -> dict[str, Any]:
    tenant_gaps = {
        row["tenant_id"]: row["excess_units"]
        for row in matrix["tenants"]
        if row["excess_units"] > 0
    }
    remaining = max(
        matrix["cluster_excess_units"],
        max(tenant_gaps.values(), default=0),
    )
    return {
        "remaining_gap_units": remaining,
        "cluster_gap_units": matrix["cluster_excess_units"],
        "tenant_gap_units": tenant_gaps,
        "affected_tenant_ids": sorted(tenant_gaps),
        "blocked_workloads": blocked,
        "rejected_preemption_requests": blocked,
    }


def _snapshot(
    iteration: int,
    matrix: dict[str, Any],
    terminal_reason: str | None,
    gap: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "iteration": iteration,
        "terminal_reason": terminal_reason,
        "matrix_digest": digest_value(matrix),
        "matrix": matrix,
        "gap": gap,
    }


def process_event(
    declaration: Declaration,
    state: RuntimeState,
    event: TriggerEvent,
) -> EventResult:
    workload_by_id = declaration.workload_by_id
    tenant_ids = set(declaration.tenant_by_id)
    next_state = state.clone()

    if event.trigger_type == "TENANT_PRESSURE":
        if event.target_tenant not in tenant_ids:
            raise ValueError(f"invalid event tenant: {event.target_tenant}")
        pressure_tenant = event.target_tenant
    elif event.trigger_type == "WORKLOAD_PRESSURE":
        workload = workload_by_id.get(event.target_workload or "")
        if workload is None:
            raise ValueError(f"invalid event workload: {event.target_workload}")
        pressure_tenant = workload.tenant_id
    else:
        raise ValueError(f"unsupported event type: {event.trigger_type}")

    next_state.pressure_by_tenant[pressure_tenant] = (
        next_state.pressure_by_tenant.get(pressure_tenant, 0) + event.pressure_units
    )

    iterations: list[dict[str, Any]] = []
    preemptions: list[dict[str, Any]] = []
    reclaimed_units = 0
    iteration = 0
    terminal_reason = ""

    while True:
        matrix = build_matrix(declaration, next_state)
        if not _has_excess(matrix):
            terminal_reason = "NO_EXCESS"
            iterations.append(_snapshot(iteration, matrix, terminal_reason, None))
            break
        gap = _gap(matrix, _blocked_requests(declaration, next_state, matrix))
        candidate = _candidate(declaration, next_state, matrix)
        if candidate is None:
            terminal_reason = "NO_PREEMPTIBLE_REPLICA"
            iterations.append(_snapshot(iteration, matrix, terminal_reason, gap))
            break

        iterations.append(_snapshot(iteration, matrix, None, gap))

        workload, basis = candidate
        rows = allocation_by_tenant(matrix)
        tenant_id = workload.tenant_id
        usage_before = rows[tenant_id]["actual_units"]
        total_before = matrix["actual_total_units"]
        next_state.running_replicas[workload.workload_id] -= 1
        after_matrix = build_matrix(declaration, next_state)
        after_rows = allocation_by_tenant(after_matrix)
        usage_after = after_rows[tenant_id]["actual_units"]
        total_after = after_matrix["actual_total_units"]
        reclaimed_units += workload.units_per_replica
        iteration += 1
        preemptions.append(
            {
                "sequence": len(preemptions) + 1,
                "iteration": iteration,
                "tenant_id": tenant_id,
                "workload_id": workload.workload_id,
                "replicas_preempted": 1,
                "units_per_replica": workload.units_per_replica,
                "reclaimed_units": workload.units_per_replica,
                "tenant_usage_before_units": usage_before,
                "tenant_usage_after_units": usage_after,
                "cluster_usage_before_units": total_before,
                "cluster_usage_after_units": total_after,
                "matrix_before_digest": iterations[-1]["matrix_digest"],
                "matrix_after_digest": digest_value(after_matrix),
                "selection_basis": basis,
            }
        )

    final_matrix = iterations[-1]["matrix"]
    unsatisfied = terminal_reason != "NO_EXCESS"
    terminal_code = "E_UNSATISFIED" if unsatisfied else "OK"
    final_blocked = (
        _blocked_requests(declaration, next_state, final_matrix) if unsatisfied else []
    )
    final_gap = _gap(final_matrix, final_blocked) if unsatisfied else None
    workload_state = next_state.to_dict(declaration)
    record: dict[str, Any] = {
        "format_version": 1,
        "event_id": event.event_id,
        "event": event.to_dict(),
        "terminal": {
            "code": terminal_code,
            "reason": terminal_reason,
            "iteration_count": len(iterations) - 1,
            "round_count": len(iterations) - 1,
            "preemption_count": len(preemptions),
            "reclaimed_units": reclaimed_units,
        },
        "iterations": iterations,
        "preemptions": preemptions,
        "final_matrix": final_matrix,
        "workload_state": workload_state,
        "gap": final_gap,
    }
    record_json = canonical_json(record)
    record_digest = sha256_bytes(record_json.encode("utf-8"))
    return EventResult(
        event_id=event.event_id,
        record=record,
        record_json=record_json,
        record_digest=record_digest,
        state=next_state,
        terminal_code=terminal_code,
    )


@dataclass(frozen=True)
class Engine:
    declaration: Declaration

    def apply(self, state: RuntimeState, event: TriggerEvent) -> EventResult:
        return process_event(self.declaration, state, event)
