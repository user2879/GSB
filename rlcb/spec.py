"""Local JSON declaration loading with strict validation.

Two declaration files are supported, both parsed with line-number awareness:

* a services declaration: global limits plus a list of service objects;
* a request stream: a list of request objects ordered by arrival sequence.

The first offending field is reported as ``E_SPEC`` with its JSON pointer-ish
field path, the 1-based source line and the source file name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import linejson
from .codes import SpecError

SUPPORTED_DEGRADE_LEVELS = frozenset({0, 1, 2, 3})
LOOPBACK_HOST = "127.0.0.1"
MAX_FAILURE_BP = 10_000
DEFAULT_SEED = 20261126


@dataclass(frozen=True)
class ServiceSpec:
    service_id: str
    host: str
    port: int
    rps_limit: int
    concurrency_limit: int
    failure_threshold_bp: int
    window_length: int
    degrade_level: int
    recovery_windows: int
    config_version: int
    seed: int

    def param_digest_input(self) -> dict:
        return {
            "service_id": self.service_id,
            "rps_limit": self.rps_limit,
            "concurrency_limit": self.concurrency_limit,
            "failure_threshold_bp": self.failure_threshold_bp,
            "window_length": self.window_length,
            "degrade_level": self.degrade_level,
            "recovery_windows": self.recovery_windows,
        }


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    service_id: str
    batch_id: str
    arrival_seq: int
    failed: int

    def key(self) -> tuple:
        return (self.service_id, self.arrival_seq, self.request_id)


@dataclass(frozen=True)
class GlobalLimits:
    max_active_leases: int
    max_active_batches: int
    lease_ttl_ms: int

@dataclass
class SpecBundle:
    source_services: str
    source_requests: str
    global_limits: GlobalLimits
    services: list[ServiceSpec]
    requests: list[RequestSpec]
    service_by_id: dict[str, ServiceSpec] = field(default_factory=dict)


def _is_nonneg_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _is_pos_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _fail(source: str, field_path: str, line: int | None, message: str) -> None:
    raise SpecError(
        f"{message} (field '{field_path}', line {line}, source {source})",
        field=field_path, line=line, source=source,
    )


def load_bundle(services_path: str | Path, requests_path: str | Path) -> SpecBundle:
    services_path = Path(services_path)
    requests_path = Path(requests_path)
    raw_services = linejson.parse_with_lines(services_path.read_text("utf-8"))
    raw_requests = linejson.parse_with_lines(requests_path.read_text("utf-8"))
    source_s = services_path.name
    source_r = requests_path.name

    services = _validate_services(raw_services, source_s)
    limits = _validate_global(raw_services, source_s)
    requests = _validate_requests(raw_requests, source_r, services)
    by_id = {svc.service_id: svc for svc in services}
    return SpecBundle(
        source_services=source_s,
        source_requests=source_r,
        global_limits=limits,
        services=services,
        requests=requests,
        service_by_id=by_id,
    )


def _validate_global(root: Any, source: str) -> GlobalLimits:
    if not isinstance(root, linejson.Located):
        _fail(source, "$", None, "services declaration must be a JSON object")
    gl = root.get("global_limits")
    if not isinstance(gl, linejson.Located):
        _fail(source, "global_limits", linejson.value_line(root, "global_limits"),
              "global_limits must be an object")
    for name in ("max_active_leases", "max_active_batches", "lease_ttl_ms"):
        if name not in gl:
            _fail(source, f"global_limits.{name}", linejson.object_line(gl),
                  f"missing required field {name}")
        if not _is_pos_int(gl[name]):
            _fail(source, f"global_limits.{name}",
                  linejson.value_line(gl, name),
                  f"{name} must be a positive integer")
    return GlobalLimits(
        max_active_leases=int(gl["max_active_leases"]),
        max_active_batches=int(gl["max_active_batches"]),
        lease_ttl_ms=int(gl["lease_ttl_ms"]),
    )


def _validate_services(root: Any, source: str) -> list[ServiceSpec]:
    items = root.get("services")
    if not isinstance(items, list) or not items:
        _fail(source, "services", linejson.value_line(root, "services"),
              "services must be a non-empty array")
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    result: list[ServiceSpec] = []
    for index, item in enumerate(items):
        path = f"services[{index}]"
        if not isinstance(item, linejson.Located):
            _fail(source, path, None, "each service must be an object")
        def fval(name: str):
            return linejson.value_line(item, name)

        sid = item.get("service_id")
        if not isinstance(sid, str) or not sid:
            _fail(source, f"{path}.service_id", fval("service_id"),
                  "service_id must be a non-empty string")
        if sid in seen_ids:
            _fail(source, f"{path}.service_id", fval("service_id"),
                  f"duplicate service_id {sid!r}")
        seen_ids.add(sid)

        host = item.get("host", LOOPBACK_HOST)
        if host != LOOPBACK_HOST:
            _fail(source, f"{path}.host", fval("host"),
                  f"host must be the literal {LOOPBACK_HOST}")

        port = item.get("port")
        if not _is_pos_int(port) or not (1 <= port <= 65535):
            _fail(source, f"{path}.port", fval("port"),
                  "port must be an integer in [1,65535]")
        if port in seen_ports:
            _fail(source, f"{path}.port", fval("port"),
                  f"duplicate loopback port {port}")
        seen_ports.add(port)

        for name in ("rps_limit", "concurrency_limit"):
            if name not in item or not _is_nonneg_int(item[name]):
                _fail(source, f"{path}.{name}", fval(name),
                      f"{name} must be a non-negative integer")

        threshold = item.get("failure_threshold_bp")
        if not _is_nonneg_int(threshold) or threshold > MAX_FAILURE_BP:
            _fail(source, f"{path}.failure_threshold_bp",
                  fval("failure_threshold_bp"),
                  "failure_threshold_bp must be an integer in [0,10000]")

        window = item.get("window_length")
        if not _is_pos_int(window):
            _fail(source, f"{path}.window_length", fval("window_length"),
                  "window_length must be a positive integer")

        degrade = item.get("degrade_level")
        if not _is_nonneg_int(degrade) or degrade not in SUPPORTED_DEGRADE_LEVELS:
            _fail(source, f"{path}.degrade_level", fval("degrade_level"),
                  "degrade_level must be one of "
                  f"{sorted(SUPPORTED_DEGRADE_LEVELS)}")

        recovery = item.get("recovery_windows", 1)
        if not _is_pos_int(recovery):
            _fail(source, f"{path}.recovery_windows", fval("recovery_windows"),
                  "recovery_windows must be a positive integer")

        version = item.get("config_version", 1)
        if not _is_pos_int(version):
            _fail(source, f"{path}.config_version", fval("config_version"),
                  "config_version must be a positive integer")

        seed = item.get("seed", DEFAULT_SEED)
        if not _is_pos_int(seed):
            _fail(source, f"{path}.seed", fval("seed"),
                  "seed must be a positive integer")

        result.append(ServiceSpec(
            service_id=sid,
            host=LOOPBACK_HOST,
            port=int(port),
            rps_limit=int(item["rps_limit"]),
            concurrency_limit=int(item["concurrency_limit"]),
            failure_threshold_bp=int(threshold),
            window_length=int(window),
            degrade_level=int(degrade),
            recovery_windows=int(recovery),
            config_version=int(version),
            seed=int(seed),
        ))
    return result


def _validate_requests(
    root: Any, source: str, services: list[ServiceSpec]
) -> list[RequestSpec]:
    if isinstance(root, linejson.Located):
        items = root.get("requests")
        container_line = linejson.value_line(root, "requests")
    else:
        items = root if isinstance(root, list) else None
        container_line = None
    if not isinstance(items, list) or not items:
        _fail(source, "requests", container_line,
              "request stream must be a non-empty array under 'requests'")
    valid_services = {svc.service_id for svc in services}
    last_seq_by_service: dict[str, int] = {}
    seen_request_ids: set[str] = set()
    result: list[RequestSpec] = []
    for index, item in enumerate(items):
        path = f"requests[{index}]"
        if not isinstance(item, linejson.Located):
            _fail(source, path, None, "each request must be an object")

        def fval(name: str):
            return linejson.value_line(item, name)

        rid = item.get("request_id")
        if not isinstance(rid, str) or not rid:
            _fail(source, f"{path}.request_id", fval("request_id"),
                  "request_id must be a non-empty string")
        if rid in seen_request_ids:
            _fail(source, f"{path}.request_id", fval("request_id"),
                  f"duplicate request_id {rid!r}")
        seen_request_ids.add(rid)

        sid = item.get("service_id")
        if sid not in valid_services:
            _fail(source, f"{path}.service_id", fval("service_id"),
                  f"unknown service_id {sid!r}")

        batch_id = item.get("batch_id")
        if not isinstance(batch_id, str) or not batch_id:
            _fail(source, f"{path}.batch_id", fval("batch_id"),
                  "batch_id must be a non-empty string")

        seq = item.get("arrival_seq")
        if not _is_nonneg_int(seq):
            _fail(source, f"{path}.arrival_seq", fval("arrival_seq"),
                  "arrival_seq must be a non-negative integer")
        previous = last_seq_by_service.get(sid)
        if previous is not None and seq < previous:
            _fail(source, f"{path}.arrival_seq", fval("arrival_seq"),
                  f"arrival_seq for {sid!r} decreased ({previous} -> {seq})")
        last_seq_by_service[sid] = int(seq)

        failed = item.get("failed", 0)
        if failed not in (0, 1):
            _fail(source, f"{path}.failed", fval("failed"),
                  "failed must be 0 or 1")

        result.append(RequestSpec(
            request_id=rid,
            service_id=sid,
            batch_id=batch_id,
            arrival_seq=int(seq),
            failed=int(failed),
        ))
    return result
