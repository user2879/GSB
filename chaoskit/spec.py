"""Experiment declaration loading and strong validation.

A declaration describes loopback target services, injection items, an
experiment budget and an observation policy. Validation is all-or-nothing:
every offending JSON path is collected and the declaration is rejected with
``E_SPEC`` as a whole. Nothing is partially applied.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .errors import SpecError

LOOPBACK = "127.0.0.1"
SUPPORTED_INJECTION_TYPES = ("network_delay", "error_rate", "suspend", "terminate")
CURRENT_SPEC_VERSION = 1


def _is_int(value: Any) -> bool:
    # bool is a subclass of int; an explicit true/false is not a number here.
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass(frozen=True)
class ServiceSpec:
    service_id: str
    host: str
    data_port: int
    proxy_port: int
    fixed_cost_ms: int
    base_error_bps: int


@dataclass(frozen=True)
class InjectionSpec:
    injection_id: str
    injection_type: str
    target_id: str
    duration_ms: int
    params: Dict[str, int] = field(default_factory=dict)

    @property
    def stable_key(self) -> Tuple[str, str, str]:
        # Order-independent identity: sorting on this key makes the timeline
        # independent of the declaration's textual order.
        return (self.target_id, self.injection_type, self.injection_id)


@dataclass(frozen=True)
class BudgetSpec:
    max_injections: int
    max_parallel: int


@dataclass(frozen=True)
class ObservationSpec:
    sample_interval_ms: int
    min_samples: int


@dataclass(frozen=True)
class ExperimentSpec:
    spec_version: int
    experiment_id: str
    seed: int
    budget: BudgetSpec
    observation: ObservationSpec
    services: Dict[str, ServiceSpec]
    injections: List[InjectionSpec]
    canonical_bytes: bytes

    @property
    def spec_digest(self) -> str:
        return hashlib.sha256(self.canonical_bytes).hexdigest()

    def sorted_injections(self) -> List[InjectionSpec]:
        return sorted(self.injections, key=lambda item: item.stable_key)


def canonical_json_bytes(data: Any) -> bytes:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def load_spec(path: str) -> ExperimentSpec:
    with open(path, "r", encoding="utf-8") as handle:
        try:
            raw = json.load(handle)
        except json.JSONDecodeError as exc:
            raise SpecError([f"$:{exc.msg}"], "declaration is not valid JSON") from exc
    return parse_spec(raw)


def parse_spec(raw: Any) -> ExperimentSpec:
    errors: List[str] = []

    if not isinstance(raw, dict):
        raise SpecError(["$"], "declaration root must be a JSON object")

    version = raw.get("version")
    if not _is_int(version) or version <= 0:
        errors.append("$.version")

    experiment_id = raw.get("experiment_id", "experiment")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        errors.append("$.experiment_id")

    seed = raw.get("seed")
    if not _is_int(seed) or seed < 0:
        errors.append("$.seed")

    budget = _parse_budget(raw.get("budget"), errors)
    observation = _parse_observation(raw.get("observation"), errors)
    services = _parse_services(raw.get("services"), errors)
    injections = _parse_injections(raw.get("injections"), errors, services)

    if budget is not None and injections is not None:
        if len(injections) > budget.max_injections:
            errors.append("$.budget.max_injections")

    if errors:
        raise SpecError(sorted(set(errors)))

    canonical = canonical_json_bytes(raw)
    return ExperimentSpec(
        spec_version=version,
        experiment_id=experiment_id,
        seed=seed,
        budget=budget,
        observation=observation,
        services=services,
        injections=injections,
        canonical_bytes=canonical,
    )


def _parse_budget(raw: Any, errors: List[str]) -> BudgetSpec | None:
    if not isinstance(raw, dict):
        errors.append("$.budget")
        return None
    ok = True
    max_injections = raw.get("max_injections")
    if not _is_int(max_injections) or max_injections <= 0:
        errors.append("$.budget.max_injections")
        ok = False
    max_parallel = raw.get("max_parallel")
    if not _is_int(max_parallel) or max_parallel <= 0:
        errors.append("$.budget.max_parallel")
        ok = False
    if ok and max_parallel > max_injections:
        errors.append("$.budget.max_parallel")
        ok = False
    return BudgetSpec(max_injections, max_parallel) if ok else None


def _parse_observation(raw: Any, errors: List[str]) -> ObservationSpec | None:
    if not isinstance(raw, dict):
        errors.append("$.observation")
        return None
    ok = True
    interval = raw.get("sample_interval_ms")
    if not _is_int(interval) or interval <= 0:
        errors.append("$.observation.sample_interval_ms")
        ok = False
    min_samples = raw.get("min_samples")
    if not _is_int(min_samples) or min_samples <= 0:
        errors.append("$.observation.min_samples")
        ok = False
    return ObservationSpec(interval, min_samples) if ok else None


def _parse_services(raw: Any, errors: List[str]) -> Dict[str, ServiceSpec] | None:
    if not isinstance(raw, list) or not raw:
        errors.append("$.services")
        return None

    services: Dict[str, ServiceSpec] = {}
    seen_ports: Dict[int, str] = {}
    for index, item in enumerate(raw):
        base = f"$.services[{index}]"
        if not isinstance(item, dict):
            errors.append(base)
            continue

        service_id = item.get("service_id")
        if not isinstance(service_id, str) or not service_id.strip():
            errors.append(f"{base}.service_id")
        elif service_id in services:
            errors.append(f"{base}.service_id")

        host = item.get("host", LOOPBACK)
        if host != LOOPBACK:
            errors.append(f"{base}.host")

        data_port = item.get("data_port")
        if not _is_int(data_port) or not 1 <= data_port <= 65535:
            errors.append(f"{base}.data_port")
        elif data_port in seen_ports:
            errors.append(f"{base}.data_port")
        else:
            seen_ports[data_port] = service_id

        proxy_port = item.get("proxy_port")
        if not _is_int(proxy_port) or not 1 <= proxy_port <= 65535:
            errors.append(f"{base}.proxy_port")
        elif proxy_port == data_port or proxy_port in seen_ports:
            errors.append(f"{base}.proxy_port")
        else:
            seen_ports[proxy_port] = service_id

        fixed_cost = item.get("fixed_cost_ms")
        if not _is_int(fixed_cost) or fixed_cost < 0:
            errors.append(f"{base}.fixed_cost_ms")

        base_error = item.get("base_error_bps", 0)
        if not _is_int(base_error) or not 0 <= base_error <= 10000:
            errors.append(f"{base}.base_error_bps")

        if all(isinstance(value, str) for value in (service_id,)) and service_id and service_id not in services:
            services[service_id] = ServiceSpec(
                service_id=service_id,
                host=host if isinstance(host, str) else LOOPBACK,
                data_port=data_port if _is_int(data_port) else -1,
                proxy_port=proxy_port if _is_int(proxy_port) else -1,
                fixed_cost_ms=fixed_cost if _is_int(fixed_cost) else 0,
                base_error_bps=base_error if _is_int(base_error) else 0,
            )

    return services if services else None


def _parse_injections(raw: Any, errors: List[str],
                      services: Dict[str, ServiceSpec] | None) -> List[InjectionSpec] | None:
    if not isinstance(raw, list) or not raw:
        errors.append("$.injections")
        return None

    injections: List[InjectionSpec] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(raw):
        base = f"$.injections[{index}]"
        if not isinstance(item, dict):
            errors.append(base)
            continue

        injection_id = item.get("injection_id")
        if not isinstance(injection_id, str) or not injection_id.strip() or injection_id in seen_ids:
            errors.append(f"{base}.injection_id")
        else:
            seen_ids.add(injection_id)

        injection_type = item.get("type")
        if injection_type not in SUPPORTED_INJECTION_TYPES:
            errors.append(f"{base}.type")

        target_id = item.get("target_id")
        if not isinstance(target_id, str) or services is None or target_id not in services:
            errors.append(f"{base}.target_id")

        duration = item.get("duration_ms")
        if not _is_int(duration) or duration <= 0:
            errors.append(f"{base}.duration_ms")

        params = item.get("params", {})
        if not isinstance(params, dict):
            errors.append(f"{base}.params")
        else:
            _validate_params(injection_type, params, f"{base}.params", errors)

        injections.append(
            InjectionSpec(
                injection_id=injection_id if isinstance(injection_id, str) else f"invalid-{index}",
                injection_type=injection_type if injection_type in SUPPORTED_INJECTION_TYPES else "invalid",
                target_id=target_id if isinstance(target_id, str) else "",
                duration_ms=duration if _is_int(duration) else 0,
                params=params if isinstance(params, dict) else {},
            )
        )

    return injections


def _validate_params(injection_type: Any, params: Dict[str, Any], base: str, errors: List[str]) -> None:
    if injection_type == "network_delay":
        delay = params.get("delay_ms")
        if not _is_int(delay) or delay <= 0:
            errors.append(f"{base}.delay_ms")
    elif injection_type == "error_rate":
        ratio = params.get("error_bps")
        if not _is_int(ratio) or not 0 <= ratio <= 10000:
            errors.append(f"{base}.error_bps")
    elif injection_type in ("suspend", "terminate"):
        for key in params:
            errors.append(f"{base}.{key}")
