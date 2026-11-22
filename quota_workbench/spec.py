"""Strict, local-only loading and validation of declarations and NDJSON events."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .canonical import digest_value
from .errors import WorkbenchError, spec_error


SUPPORTED_EVENT_TYPES = ("TENANT_PRESSURE", "WORKLOAD_PRESSURE")
SHARE_SCALE = 10_000


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_dict(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise spec_error(
            f"{path} must be an object",
            object_type=path,
            object_id=None,
            field=path,
        )
    return value


def _require_nonempty_string(value: Any, path: str, object_type: str, object_id: str | None) -> str:
    if not isinstance(value, str) or not value.strip():
        raise spec_error(
            f"{path} must be a non-empty string",
            object_type=object_type,
            object_id=object_id,
            field=path,
        )
    return value


@dataclass(frozen=True, order=True)
class Tenant:
    tenant_id: str
    quota_bp: int
    reserve_units: int
    priority: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "tenant_id": self.tenant_id,
            "quota_bp": self.quota_bp,
            "reserve_units": self.reserve_units,
            "priority": self.priority,
        }


@dataclass(frozen=True, order=True)
class Workload:
    workload_id: str
    tenant_id: str
    requested_replicas: int
    units_per_replica: int
    preemptible: bool
    startup_order: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id,
            "tenant_id": self.tenant_id,
            "requested_replicas": self.requested_replicas,
            "units_per_replica": self.units_per_replica,
            "preemptible": self.preemptible,
            "startup_order": self.startup_order,
        }


@dataclass(frozen=True, order=True)
class TriggerEvent:
    event_id: int
    time_ms: int
    trigger_type: str
    pressure_units: int
    target_tenant: str | None = None
    target_workload: str | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "event_id": self.event_id,
            "time_ms": self.time_ms,
            "trigger_type": self.trigger_type,
            "pressure_units": self.pressure_units,
        }
        if self.target_tenant is not None:
            value["target_tenant"] = self.target_tenant
        if self.target_workload is not None:
            value["target_workload"] = self.target_workload
        return value


@dataclass(frozen=True)
class Declaration:
    cluster_total_units: int
    tenants: tuple[Tenant, ...]
    workloads: tuple[Workload, ...]

    @property
    def tenant_by_id(self) -> dict[str, Tenant]:
        return {item.tenant_id: item for item in self.tenants}

    @property
    def workload_by_id(self) -> dict[str, Workload]:
        return {item.workload_id: item for item in self.workloads}

    def sorted_tenants(self) -> list[Tenant]:
        return sorted(self.tenants, key=lambda item: item.tenant_id)

    def sorted_workloads(self) -> list[Workload]:
        return sorted(self.workloads, key=lambda item: item.workload_id)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_total_units": self.cluster_total_units,
            "tenants": [item.to_dict() for item in self.sorted_tenants()],
            "workloads": [item.to_dict() for item in self.sorted_workloads()],
        }

    def tenant_set_version(self) -> str:
        return digest_value({"tenants": [item.to_dict() for item in self.sorted_tenants()]})

    def workload_declaration_version(self) -> str:
        return digest_value({"workloads": [item.to_dict() for item in self.sorted_workloads()]})

    def quota_context_version(self) -> str:
        return digest_value(
            {
                "cluster_total_units": self.cluster_total_units,
                "share_scale": SHARE_SCALE,
                "tenant_context": [item.to_dict() for item in self.sorted_tenants()],
            }
        )

    def spec_version(self) -> str:
        return digest_value(self.to_dict())

    def versions(self) -> dict[str, Any]:
        return {
            "format_version": 1,
            "tenant_set_version": self.tenant_set_version(),
            "workload_declaration_version": self.workload_declaration_version(),
            "quota_context_version": self.quota_context_version(),
            "spec_version": self.spec_version(),
            "cluster_total_units": self.cluster_total_units,
        }


def parse_declaration(data: Any) -> Declaration:
    root = _require_dict(data, "$")
    total = root.get("cluster_total_units")
    if not _is_int(total) or total <= 0:
        raise spec_error(
            "cluster_total_units must be a positive integer",
            object_type="cluster",
            object_id=None,
            field="cluster_total_units",
        )

    tenants_raw = root.get("tenants")
    if not isinstance(tenants_raw, list) or not tenants_raw:
        raise spec_error(
            "tenants must be a non-empty array",
            object_type="declaration",
            object_id=None,
            field="tenants",
        )

    tenants: list[Tenant] = []
    seen_tenants: set[str] = set()
    for index, item in enumerate(tenants_raw):
        path = f"tenants[{index}]"
        obj = _require_dict(item, path)
        tenant_id = _require_nonempty_string(obj.get("tenant_id"), f"{path}.tenant_id", "tenant", None)
        if tenant_id in seen_tenants:
            raise spec_error(
                f"duplicate tenant_id {tenant_id!r}",
                object_type="tenant",
                object_id=tenant_id,
                field="tenant_id",
            )
        seen_tenants.add(tenant_id)
        quota_bp = obj.get("quota_bp")
        if not _is_int(quota_bp) or quota_bp <= 0:
            raise spec_error(
                "quota_bp must be a positive integer",
                object_type="tenant",
                object_id=tenant_id,
                field="quota_bp",
            )
        reserve = obj.get("reserve_units")
        if not _is_int(reserve) or reserve < 0:
            raise spec_error(
                "reserve_units must be a non-negative integer",
                object_type="tenant",
                object_id=tenant_id,
                field="reserve_units",
            )
        priority = obj.get("priority")
        if not _is_int(priority):
            raise spec_error(
                "priority must be an integer",
                object_type="tenant",
                object_id=tenant_id,
                field="priority",
            )
        tenants.append(Tenant(tenant_id, quota_bp, reserve, priority))

    share_sum = sum(item.quota_bp for item in tenants)
    if share_sum > SHARE_SCALE:
        raise WorkbenchError(
            "E_QUOTA",
            f"sum of tenant quota_bp is {share_sum}, exceeding {SHARE_SCALE}",
            object_type="tenant_set",
            field="quota_bp_sum",
            details={
                "share_scale": SHARE_SCALE,
                "quota_bp_sum": share_sum,
                "over_by": share_sum - SHARE_SCALE,
                "parent_code": "E_SPEC",
            },
        )

    reserve_sum = sum(item.reserve_units for item in tenants)
    if reserve_sum > total:
        raise WorkbenchError(
            "E_RESERVE",
            f"sum of tenant reserve_units is {reserve_sum}, exceeding cluster_total_units {total}",
            object_type="tenant_set",
            field="reserve_units_sum",
            details={
                "cluster_total_units": total,
                "reserve_units_sum": reserve_sum,
                "over_by": reserve_sum - total,
                "parent_code": "E_SPEC",
            },
        )

    workloads_raw = root.get("workloads")
    if not isinstance(workloads_raw, list):
        raise spec_error(
            "workloads must be an array",
            object_type="declaration",
            object_id=None,
            field="workloads",
        )
    workloads: list[Workload] = []
    seen_workloads: set[str] = set()
    seen_startup_orders: set[int] = set()
    tenant_ids = seen_tenants
    for index, item in enumerate(workloads_raw):
        path = f"workloads[{index}]"
        obj = _require_dict(item, path)
        workload_id = _require_nonempty_string(
            obj.get("workload_id"), f"{path}.workload_id", "workload", None
        )
        if workload_id in seen_workloads:
            raise spec_error(
                f"duplicate workload_id {workload_id!r}",
                object_type="workload",
                object_id=workload_id,
                field="workload_id",
            )
        seen_workloads.add(workload_id)
        tenant_id = _require_nonempty_string(
            obj.get("tenant_id"), f"{path}.tenant_id", "workload", workload_id
        )
        if tenant_id not in tenant_ids:
            raise spec_error(
                f"workload references unknown tenant {tenant_id!r}",
                object_type="workload",
                object_id=workload_id,
                field="tenant_id",
            )
        replicas = obj.get("requested_replicas")
        if not _is_int(replicas) or replicas <= 0:
            raise spec_error(
                "requested_replicas must be a positive integer",
                object_type="workload",
                object_id=workload_id,
                field="requested_replicas",
            )
        units = obj.get("units_per_replica")
        if not _is_int(units) or units <= 0:
            raise spec_error(
                "units_per_replica must be a positive integer",
                object_type="workload",
                object_id=workload_id,
                field="units_per_replica",
            )
        preemptible = obj.get("preemptible")
        if not isinstance(preemptible, bool):
            raise spec_error(
                "preemptible must be a boolean",
                object_type="workload",
                object_id=workload_id,
                field="preemptible",
            )
        startup_order = obj.get("startup_order")
        if not _is_int(startup_order):
            raise spec_error(
                "startup_order must be an integer",
                object_type="workload",
                object_id=workload_id,
                field="startup_order",
            )
        if startup_order in seen_startup_orders:
            raise spec_error(
                f"duplicate startup_order {startup_order}",
                object_type="workload",
                object_id=workload_id,
                field="startup_order",
            )
        seen_startup_orders.add(startup_order)
        workloads.append(
            Workload(workload_id, tenant_id, replicas, units, preemptible, startup_order)
        )

    return Declaration(total, tuple(tenants), tuple(workloads))


def load_declaration(path: Path | str) -> Declaration:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise WorkbenchError(
            "E_IO", f"cannot read declaration {source}: {exc}", field="path"
        ) from exc
    except json.JSONDecodeError as exc:
        raise spec_error(
            f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}",
            object_type="declaration",
            object_id=str(source),
            field=f"$[{exc.lineno}:{exc.colno}]",
        ) from exc
    return parse_declaration(data)


def _find_target_event_error(
    event_id: int,
    event_type: str,
    target_tenant: Any,
    target_workload: Any,
    tenant_ids: set[str],
    workload_by_id: dict[str, Workload],
) -> None:
    if event_type == "TENANT_PRESSURE":
        if not isinstance(target_tenant, str) or not target_tenant:
            raise spec_error(
                "TENANT_PRESSURE requires target_tenant",
                object_type="event",
                object_id=str(event_id),
                field="target_tenant",
            )
        if target_tenant not in tenant_ids:
            raise spec_error(
                f"event references unknown tenant {target_tenant!r}",
                object_type="event",
                object_id=str(event_id),
                field="target_tenant",
            )
        if target_workload not in (None, ""):
            raise spec_error(
                "TENANT_PRESSURE must not carry target_workload",
                object_type="event",
                object_id=str(event_id),
                field="target_workload",
            )
    elif event_type == "WORKLOAD_PRESSURE":
        if not isinstance(target_workload, str) or not target_workload:
            raise spec_error(
                "WORKLOAD_PRESSURE requires target_workload",
                object_type="event",
                object_id=str(event_id),
                field="target_workload",
            )
        if target_workload not in workload_by_id:
            raise spec_error(
                f"event references unknown workload {target_workload!r}",
                object_type="event",
                object_id=str(event_id),
                field="target_workload",
            )
        if target_tenant not in (None, ""):
            raise spec_error(
                "WORKLOAD_PRESSURE must not carry target_tenant",
                object_type="event",
                object_id=str(event_id),
                field="target_tenant",
            )


def parse_events(lines: Iterable[str], declaration: Declaration) -> tuple[TriggerEvent, ...]:
    tenants = declaration.tenant_by_id
    workloads = declaration.workload_by_id
    events: list[TriggerEvent] = []
    previous_time: int | None = None
    expected_id = 1
    for line_number, raw in enumerate(lines, start=1):
        if not raw.strip():
            continue
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise spec_error(
                f"invalid NDJSON at line {line_number}: {exc.msg}",
                object_type="event_stream",
                object_id=str(line_number),
                field="ndjson",
            ) from exc
        if not isinstance(obj, dict):
            raise spec_error(
                "event line must be a JSON object",
                object_type="event_stream",
                object_id=str(line_number),
                field="ndjson",
            )
        event_id = obj.get("event_id")
        if not _is_int(event_id) or event_id <= 0:
            raise spec_error(
                "event_id must be a positive integer",
                object_type="event",
                object_id=str(line_number),
                field="event_id",
            )
        if event_id != expected_id:
            raise spec_error(
                f"event_id must be continuous: expected {expected_id}, got {event_id}",
                object_type="event",
                object_id=str(event_id),
                field="event_id",
            )
        expected_id += 1
        time_ms = obj.get("time_ms")
        if not _is_int(time_ms) or time_ms < 0:
            raise spec_error(
                "time_ms must be a non-negative integer",
                object_type="event",
                object_id=str(event_id),
                field="time_ms",
            )
        if previous_time is not None and time_ms < previous_time:
            raise spec_error(
                "time_ms must be monotonically non-decreasing",
                object_type="event",
                object_id=str(event_id),
                field="time_ms",
            )
        previous_time = time_ms
        event_type = obj.get("type")
        if event_type not in SUPPORTED_EVENT_TYPES:
            raise spec_error(
                f"type must be one of {list(SUPPORTED_EVENT_TYPES)}",
                object_type="event",
                object_id=str(event_id),
                field="type",
            )
        pressure = obj.get("pressure_units")
        if not _is_int(pressure) or pressure < 0:
            raise spec_error(
                "pressure_units must be a non-negative integer",
                object_type="event",
                object_id=str(event_id),
                field="pressure_units",
            )
        target_tenant = obj.get("target_tenant")
        target_workload = obj.get("target_workload")
        _find_target_event_error(
            event_id, event_type, target_tenant, target_workload, set(tenants), workloads
        )
        events.append(
            TriggerEvent(
                event_id=event_id,
                time_ms=time_ms,
                trigger_type=event_type,
                pressure_units=pressure,
                target_tenant=target_tenant if event_type == "TENANT_PRESSURE" else None,
                target_workload=target_workload if event_type == "WORKLOAD_PRESSURE" else None,
            )
        )
    return tuple(events)


def load_events(path: Path | str, declaration: Declaration) -> tuple[TriggerEvent, ...]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8-sig") as handle:
            return parse_events(handle, declaration)
    except OSError as exc:
        raise WorkbenchError("E_IO", f"cannot read events {source}: {exc}", field="path") from exc


def load_declaration_data(data: Any) -> Declaration:
    return parse_declaration(data)


def mutate_declaration_file(path: Path | str, operation: dict[str, Any]) -> tuple[Declaration, dict[str, Any]]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise WorkbenchError("E_IO", f"cannot read declaration {source}: {exc}", field="path") from exc
    except json.JSONDecodeError as exc:
        raise spec_error(
            f"invalid JSON: {exc.msg}", object_type="declaration", object_id=str(source), field="json"
        ) from exc
    op = operation.get("op")
    target = operation.get("path", "")
    if op == "quota_overflow":
        tenant = data["tenants"][int(operation.get("index", 0))]
        tenant["quota_bp"] = int(tenant["quota_bp"]) + int(operation.get("delta", 1))
        changed = {"field": f"tenants[{operation.get('index', 0)}].quota_bp", "tenant_id": tenant["tenant_id"]}
    elif op == "reserve_overflow":
        tenant = data["tenants"][int(operation.get("index", 0))]
        tenant["reserve_units"] = int(data["cluster_total_units"]) + int(operation.get("delta", 1))
        changed = {
            "field": f"tenants[{operation.get('index', 0)}].reserve_units",
            "tenant_id": tenant["tenant_id"],
        }
    else:
        raise WorkbenchError(
            "E_SPEC",
            f"unsupported declaration injection operation {op!r}",
            object_type="injection",
            field="op",
            details={"path": target},
        )
    return load_declaration_data(data), changed


def load_injection_catalog(path: Path | str) -> dict[str, Any]:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8-sig") as handle:
            data = json.load(handle)
    except OSError as exc:
        raise WorkbenchError("E_IO", f"cannot read injection catalog {source}: {exc}", field="path") from exc
    except json.JSONDecodeError as exc:
        raise spec_error(
            f"invalid injection catalog JSON: {exc.msg}",
            object_type="injection_catalog",
            object_id=str(source),
            field="json",
        ) from exc
    injections = data.get("injections") if isinstance(data, dict) else None
    if not isinstance(injections, list):
        raise spec_error(
            "injections must be an array",
            object_type="injection_catalog",
            object_id=str(source),
            field="injections",
        )
    result: dict[str, Any] = {}
    for index, item in enumerate(injections):
        if not isinstance(item, dict) or not isinstance(item.get("id"), str):
            raise spec_error(
                "each injection needs a string id",
                object_type="injection",
                object_id=str(index),
                field="id",
            )
        result[item["id"]] = item
    return result
