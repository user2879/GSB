"""Loading and strict validation of tenant/instance declarations and NDJSON
request streams.

Any violation aborts the whole load with E_SPEC (or E_QUOTA for the
share-sum rule) and reports the first offending field, plus the 1-based
line number for request-stream violations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional

from .errors import QuotaError, SpecError
from . import SUPPORTED_RULE_VERSIONS

LOOPBACK_LITERAL = "127.0.0.1"
BEHAVIORS = ("normal", "refused", "delay", "error")
MIN_PORT = 1024
MAX_PORT = 65535


@dataclass(frozen=True)
class ProbePolicy:
    policy_version: int
    interval_ms: int
    timeout_ms: int
    failure_threshold: int
    eject_threshold: int


@dataclass(frozen=True)
class Tenant:
    tenant_id: str
    share: int
    max_concurrency: int
    burst: int


@dataclass(frozen=True)
class Instance:
    instance_id: str
    host: str
    port: int
    weight: int
    behavior: str
    delay_ms: int
    hosts: tuple


@dataclass(frozen=True)
class Request:
    request_id: str
    tenant_id: str
    demand: int
    arrival_ms: int


@dataclass(frozen=True)
class Spec:
    tenant_set_version: int
    instance_set_version: int
    rule_version: int
    global_concurrency: int
    total_share_cap: int
    duration_factor_ms: int
    probe_policy: ProbePolicy
    tenants: dict
    instances: dict
    tenant_instance: dict
    raw: dict

    def digest(self) -> str:
        blob = json.dumps(self.raw, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def instance_of(self, tenant_id: str) -> Instance:
        return self.instances[self.tenant_instance[tenant_id]]


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(obj: dict, key: str, field_path: str, minv: int, maxv: Optional[int] = None) -> int:
    value = obj.get(key)
    if not _is_int(value) or value < minv or (maxv is not None and value > maxv):
        raise SpecError(
            f"field {field_path} must be an integer in [{minv}, {maxv if maxv is not None else 'inf'}], got {value!r}",
            field=field_path,
        )
    return value


def _require_str(obj: dict, key: str, field_path: str) -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value:
        raise SpecError(f"field {field_path} must be a non-empty string, got {value!r}", field=field_path)
    return value


def load_spec(path: str, _raw_override: Optional[dict] = None) -> Spec:
    """Load and validate the JSON declaration. Raises WorkbenchError on the
    first violation, with a stable field path."""
    if _raw_override is not None:
        raw = _raw_override
    else:
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise SpecError(f"cannot read spec: {exc}", field="<file>")
    if not isinstance(raw, dict):
        raise SpecError("spec root must be a JSON object", field="<root>")

    tenant_set_version = _require_int(raw, "tenant_set_version", "tenant_set_version", 1)
    instance_set_version = _require_int(raw, "instance_set_version", "instance_set_version", 1)
    rule_version = _require_int(raw, "rule_version", "rule_version", 1)
    if rule_version not in SUPPORTED_RULE_VERSIONS:
        raise SpecError(
            f"unsupported rule_version {rule_version}; supported: {list(SUPPORTED_RULE_VERSIONS)}",
            field="rule_version",
        )
    global_concurrency = _require_int(raw, "global_concurrency", "global_concurrency", 1)
    total_share_cap = _require_int(raw, "total_share_cap", "total_share_cap", 0)
    duration_factor_ms = _require_int(raw, "duration_factor_ms", "duration_factor_ms", 1)

    policy_raw = raw.get("probe_policy")
    if not isinstance(policy_raw, dict):
        raise SpecError("probe_policy must be an object", field="probe_policy")
    policy = ProbePolicy(
        policy_version=_require_int(policy_raw, "policy_version", "probe_policy.policy_version", 1),
        interval_ms=_require_int(policy_raw, "interval_ms", "probe_policy.interval_ms", 1),
        timeout_ms=_require_int(policy_raw, "timeout_ms", "probe_policy.timeout_ms", 1),
        failure_threshold=_require_int(policy_raw, "failure_threshold", "probe_policy.failure_threshold", 1),
        eject_threshold=_require_int(policy_raw, "eject_threshold", "probe_policy.eject_threshold", 1),
    )
    if policy.eject_threshold < policy.failure_threshold:
        raise SpecError(
            "probe_policy.eject_threshold must be >= probe_policy.failure_threshold",
            field="probe_policy.eject_threshold",
        )

    tenants_raw = raw.get("tenants")
    if not isinstance(tenants_raw, list) or not tenants_raw:
        raise SpecError("tenants must be a non-empty array", field="tenants")
    tenants = {}
    for idx, item in enumerate(tenants_raw):
        fp = f"tenants[{idx}]"
        if not isinstance(item, dict):
            raise SpecError(f"{fp} must be an object", field=fp)
        tenant_id = _require_str(item, "tenant_id", f"{fp}.tenant_id")
        if tenant_id in tenants:
            raise SpecError(f"duplicate tenant_id {tenant_id!r}", field=f"{fp}.tenant_id")
        tenants[tenant_id] = Tenant(
            tenant_id=tenant_id,
            share=_require_int(item, "share", f"{fp}.share", 0),
            max_concurrency=_require_int(item, "max_concurrency", f"{fp}.max_concurrency", 0),
            burst=_require_int(item, "burst", f"{fp}.burst", 0),
        )

    share_sum = sum(t.share for t in tenants.values())
    if share_sum > total_share_cap:
        raise QuotaError(
            f"sum of tenant shares {share_sum} exceeds declared total_share_cap {total_share_cap}",
            field="tenants[].share",
            details={"share_sum": share_sum, "total_share_cap": total_share_cap},
        )

    instances_raw = raw.get("instances")
    if not isinstance(instances_raw, list) or not instances_raw:
        raise SpecError("instances must be a non-empty array", field="instances")
    instances = {}
    used_ports = {}
    tenant_instance = {}
    for idx, item in enumerate(instances_raw):
        fp = f"instances[{idx}]"
        if not isinstance(item, dict):
            raise SpecError(f"{fp} must be an object", field=fp)
        instance_id = _require_str(item, "instance_id", f"{fp}.instance_id")
        if instance_id in instances:
            raise SpecError(f"duplicate instance_id {instance_id!r}", field=f"{fp}.instance_id")
        host = _require_str(item, "host", f"{fp}.host")
        if host != LOOPBACK_LITERAL:
            raise SpecError(
                f"host must be the loopback literal {LOOPBACK_LITERAL!r}, got {host!r}",
                field=f"{fp}.host",
            )
        port = _require_int(item, "port", f"{fp}.port", MIN_PORT, MAX_PORT)
        if port in used_ports:
            raise SpecError(
                f"port {port} already used by instance {used_ports[port]!r}",
                field=f"{fp}.port",
            )
        used_ports[port] = instance_id
        weight = _require_int(item, "weight", f"{fp}.weight", 1)
        behavior = item.get("behavior", "normal")
        if behavior not in BEHAVIORS:
            raise SpecError(
                f"behavior must be one of {list(BEHAVIORS)}, got {behavior!r}",
                field=f"{fp}.behavior",
            )
        delay_ms = _require_int(item, "delay_ms", f"{fp}.delay_ms", 0) if "delay_ms" in item else 0
        hosts_raw = item.get("hosts", [])
        if not isinstance(hosts_raw, list) or any(not isinstance(h, str) for h in hosts_raw):
            raise SpecError(f"{fp}.hosts must be an array of tenant ids", field=f"{fp}.hosts")
        for h in hosts_raw:
            if h not in tenants:
                raise SpecError(f"hosts references unknown tenant {h!r}", field=f"{fp}.hosts")
            if h in tenant_instance:
                raise SpecError(
                    f"tenant {h!r} hosted by more than one instance",
                    field=f"{fp}.hosts",
                )
            tenant_instance[h] = instance_id
        instances[instance_id] = Instance(
            instance_id=instance_id,
            host=host,
            port=port,
            weight=weight,
            behavior=behavior,
            delay_ms=delay_ms,
            hosts=tuple(hosts_raw),
        )

    for tenant_id in tenants:
        if tenant_id not in tenant_instance:
            raise SpecError(
                f"tenant {tenant_id!r} is not hosted by any instance",
                field="instances[].hosts",
            )

    return Spec(
        tenant_set_version=tenant_set_version,
        instance_set_version=instance_set_version,
        rule_version=rule_version,
        global_concurrency=global_concurrency,
        total_share_cap=total_share_cap,
        duration_factor_ms=duration_factor_ms,
        probe_policy=policy,
        tenants=tenants,
        instances=instances,
        tenant_instance=tenant_instance,
        raw=raw,
    )


def load_requests(path: str, spec: Spec) -> list:
    """Load the NDJSON request stream with strict per-line validation."""
    requests = []
    seen_ids = set()
    last_arrival = 0
    try:
        fh = open(path, "r", encoding="utf-8")
    except OSError as exc:
        raise SpecError(f"cannot read request stream: {exc}", field="<file>")
    with fh:
        for lineno, line in enumerate(fh, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise SpecError(f"invalid JSON: {exc}", field="<line>", line=lineno)
            if not isinstance(obj, dict):
                raise SpecError("request line must be a JSON object", field="<line>", line=lineno)
            request_id = obj.get("request_id")
            if not isinstance(request_id, str) or not request_id:
                raise SpecError("request_id must be a non-empty string", field="request_id", line=lineno)
            if request_id in seen_ids:
                raise SpecError(f"duplicate request_id {request_id!r}", field="request_id", line=lineno)
            seen_ids.add(request_id)
            tenant_id = obj.get("tenant_id")
            if not isinstance(tenant_id, str) or tenant_id not in spec.tenants:
                raise SpecError(
                    f"request references unknown tenant {tenant_id!r}",
                    field="tenant_id",
                    line=lineno,
                )
            demand = obj.get("demand")
            if not _is_int(demand) or demand < 1:
                raise SpecError("demand must be a positive integer", field="demand", line=lineno)
            arrival_ms = obj.get("arrival_ms")
            if not _is_int(arrival_ms) or arrival_ms < 0:
                raise SpecError("arrival_ms must be a non-negative integer", field="arrival_ms", line=lineno)
            if arrival_ms < last_arrival:
                raise SpecError(
                    f"arrival_ms {arrival_ms} is before previous arrival {last_arrival} (must be monotonic non-decreasing)",
                    field="arrival_ms",
                    line=lineno,
                )
            last_arrival = arrival_ms
            requests.append(Request(request_id, tenant_id, demand, arrival_ms))
    return requests
