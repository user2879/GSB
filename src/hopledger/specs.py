from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .errors import SpecError


CHAIN_FORMAT = "hopledger.chain/1"
ORIGIN_FORMAT = "hopledger.origin/1"
DIAGNOSTIC_FORMAT_VERSION = 1


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def is_power_of_ten(value: Any) -> bool:
    if not _is_int(value) or value <= 0:
        return False
    current = value
    while current % 10 == 0:
        current //= 10
    return current == 1


def _read_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except FileNotFoundError as exc:
        raise SpecError(f"spec file not found: {path}", field="$", path=str(path)) from exc
    except json.JSONDecodeError as exc:
        raise SpecError(
            f"invalid JSON in {path}: {exc.msg}",
            field="$",
            path=str(path),
            line=exc.lineno,
            column=exc.colno,
        ) from exc
    except OSError as exc:
        raise SpecError(f"cannot read spec {path}: {exc}", field="$", path=str(path)) from exc
    if not isinstance(value, dict):
        raise SpecError("top-level JSON declaration must be an object", field="$", path=str(path))
    return value


def _require(condition: bool, message: str, *, hop_id: str | None, field: str) -> None:
    if not condition:
        raise SpecError(message, hop_id=hop_id, field=field)


@dataclass(frozen=True)
class HopSpec:
    hop_id: str
    port: int
    role: str
    queue_ms: int
    process_ms: int
    failure_ratio: float
    scale_factor: int
    config_version: int = 1
    forward_estimate_ms: int = 1
    next_hop_id: str | None = None

    @property
    def failure_ratio_bp(self) -> int:
        # Four decimal places are the declaration resolution for basis points.
        return int(math.floor(self.failure_ratio * 10_000.0 + 0.5))

    def to_declaration(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OriginSample:
    request_id: str
    path: str
    seed_offset: int


@dataclass(frozen=True)
class OriginSpec:
    origin_id: str
    port: int
    seed: int
    body_bytes: int
    samples: tuple[OriginSample, ...]
    config_version: int = 1

    def sample_for(self, request_id: str) -> OriginSample:
        for sample in self.samples:
            if sample.request_id == request_id:
                return sample
        raise SpecError(f"no origin sample for request {request_id!r}", field="samples")


@dataclass(frozen=True)
class SchedulerConfig:
    max_in_flight: int = 2
    hop_quota: int = 1
    pool_workers: int = 4


@dataclass(frozen=True)
class ErrorModelConfig:
    dominance_threshold_bp: int = 6_000
    indistinguishable_margin_bp: int = 500
    high_failure_ratio_bp: int = 5_000


@dataclass(frozen=True)
class DiagnosticConfig:
    diagnostic_id: str
    format_version: int
    scale_caliber_version: int
    session_scale: int
    timeout_seconds: float
    handshake_timeout_seconds: float
    batch_size: int
    scheduler: SchedulerConfig
    error_model: ErrorModelConfig


@dataclass(frozen=True)
class ChainSpec:
    format: str
    hop_set_version: int
    client_id: str
    origin_id: str
    allowed_scale_factors: tuple[int, ...]
    hops: tuple[HopSpec, ...]
    source_path: str | None = None

    @property
    def hop_ids(self) -> list[str]:
        return [hop.hop_id for hop in self.hops]

    @property
    def node_ids(self) -> list[str]:
        return [self.client_id, *self.hop_ids, self.origin_id]

    def hop(self, hop_id: str) -> HopSpec:
        for hop in self.hops:
            if hop.hop_id == hop_id:
                return hop
        raise SpecError(f"unknown hop {hop_id!r}", hop_id=hop_id, field="hop_id")

    def config_versions(self) -> dict[str, int]:
        return {hop.hop_id: hop.config_version for hop in self.hops}

    def scale_factors(self) -> dict[str, int]:
        return {hop.hop_id: hop.scale_factor for hop in self.hops}


def parse_chain(data: dict[str, Any], *, source_path: str | Path | None = None) -> ChainSpec:
    if data.get("format") != CHAIN_FORMAT:
        raise SpecError(
            f"chain format must be {CHAIN_FORMAT!r}",
            hop_id=None,
            field="format",
        )

    hop_set_version = data.get("hop_set_version")
    _require(_is_int(hop_set_version) and hop_set_version > 0,
             "hop_set_version must be a positive integer", hop_id=None,
             field="hop_set_version")

    client_id = data.get("client_id")
    _require(isinstance(client_id, str) and client_id,
             "client_id must be a non-empty string", hop_id=None, field="client_id")

    origin_id = data.get("origin_id")
    _require(isinstance(origin_id, str) and origin_id,
             "origin_id must be a non-empty string", hop_id=None, field="origin_id")

    allowed = data.get("allowed_scale_factors")
    _require(isinstance(allowed, list) and allowed,
             "allowed_scale_factors must be a non-empty array", hop_id=None,
             field="allowed_scale_factors")
    allowed_tuple: list[int] = []
    for index, factor in enumerate(allowed):
        _require(is_power_of_ten(factor),
                 "allowed scale factor must be a positive integer power of 10",
                 hop_id=None, field=f"allowed_scale_factors[{index}]")
        allowed_tuple.append(factor)
    _require(len(set(allowed_tuple)) == len(allowed_tuple),
             "allowed_scale_factors must be unique", hop_id=None,
             field="allowed_scale_factors")

    raw_hops = data.get("hops")
    _require(isinstance(raw_hops, list) and raw_hops,
             "hops must be a non-empty array", hop_id=None, field="hops")

    hop_id_counts: dict[str, int] = {}
    port_counts: dict[int, int] = {}
    for raw_hop in raw_hops:
        if isinstance(raw_hop, dict):
            raw_hop_id = raw_hop.get("hop_id")
            if isinstance(raw_hop_id, str) and raw_hop_id:
                hop_id_counts[raw_hop_id] = hop_id_counts.get(raw_hop_id, 0) + 1
            raw_port = raw_hop.get("port")
            if _is_int(raw_port):
                port_counts[raw_port] = port_counts.get(raw_port, 0) + 1

    hops: list[HopSpec] = []
    seen_hop_ids: set[str] = set()
    seen_ports: set[int] = set()

    for index, raw_hop in enumerate(raw_hops):
        hop_ref = f"hops[{index}]"
        _require(isinstance(raw_hop, dict), "hop declaration must be an object",
                 hop_id=hop_ref, field=f"hops[{index}]")

        hop_id = raw_hop.get("hop_id")
        hop_ref = hop_id if isinstance(hop_id, str) and hop_id else hop_ref
        _require(isinstance(hop_id, str) and hop_id,
                 "hop_id must be a non-empty string", hop_id=hop_ref, field="hop_id")
        _require(hop_id_counts.get(hop_id, 0) == 1 and hop_id not in seen_hop_ids,
                 f"duplicate hop_id {hop_id!r}",
                 hop_id=hop_id, field="hop_id")

        port = raw_hop.get("port")
        _require(_is_int(port) and 1024 <= port <= 65535,
                 "port must be an integer in 1024..65535", hop_id=hop_id, field="port")
        _require(port_counts.get(port, 0) == 1 and port not in seen_ports,
                 f"duplicate port {port}",
                 hop_id=hop_id, field="port")

        role = raw_hop.get("role")
        _require(role in {"forward", "reverse"},
                 "role must be 'forward' or 'reverse'", hop_id=hop_id, field="role")

        queue_ms = raw_hop.get("queue_ms")
        _require(_is_int(queue_ms) and queue_ms >= 0,
                 "queue_ms must be a non-negative integer", hop_id=hop_id, field="queue_ms")

        process_ms = raw_hop.get("process_ms")
        _require(_is_int(process_ms) and process_ms >= 0,
                 "process_ms must be a non-negative integer", hop_id=hop_id,
                 field="process_ms")

        failure_ratio = raw_hop.get("failure_ratio")
        _require(_is_number(failure_ratio) and math.isfinite(float(failure_ratio))
                 and 0.0 <= float(failure_ratio) <= 1.0,
                 "failure_ratio must be a finite number in [0, 1]",
                 hop_id=hop_id, field="failure_ratio")

        scale_factor = raw_hop.get("scale_factor")
        _require(scale_factor in allowed_tuple,
                 f"scale_factor must be one of {allowed_tuple}",
                 hop_id=hop_id, field="scale_factor")

        config_version = raw_hop.get("config_version", 1)
        _require(_is_int(config_version) and config_version > 0,
                 "config_version must be a positive integer", hop_id=hop_id,
                 field="config_version")

        forward_estimate_ms = raw_hop.get("forward_estimate_ms", 1)
        _require(_is_int(forward_estimate_ms) and forward_estimate_ms >= 0,
                 "forward_estimate_ms must be a non-negative integer",
                 hop_id=hop_id, field="forward_estimate_ms")

        next_hop_id = raw_hop.get("next_hop_id")
        if next_hop_id is not None:
            _require(isinstance(next_hop_id, str) and next_hop_id,
                     "next_hop_id must be a non-empty string",
                     hop_id=hop_id, field="next_hop_id")

        seen_hop_ids.add(hop_id)
        seen_ports.add(port)
        hops.append(
            HopSpec(
                hop_id=hop_id,
                port=port,
                role=role,
                queue_ms=queue_ms,
                process_ms=process_ms,
                failure_ratio=float(failure_ratio),
                scale_factor=scale_factor,
                config_version=config_version,
                forward_estimate_ms=forward_estimate_ms,
                next_hop_id=next_hop_id,
            )
        )

    _require(hops[0].role == "forward",
             "first hop must be a forward proxy connected to the client",
             hop_id=hops[0].hop_id, field="role")
    _require(hops[-1].role == "reverse",
             "last hop must be a reverse proxy connected to the origin",
             hop_id=hops[-1].hop_id, field="role")

    # The declared array is the chain order. Running the edge set through a
    # topological check also rejects future explicit next_hop_id extensions.
    edges: dict[str, set[str]] = {hop.hop_id: set() for hop in hops}
    for index, hop in enumerate(hops):
        if hop.next_hop_id is None:
            if index + 1 < len(hops):
                edges[hop.hop_id].add(hops[index + 1].hop_id)
        else:
            _require(hop.next_hop_id in edges,
                     "next_hop_id must reference a declared hop", hop_id=hop.hop_id,
                     field="next_hop_id")
            edges[hop.hop_id].add(hop.next_hop_id)
    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(node_id: str) -> None:
        _require(node_id not in visiting, "proxy chain contains a cycle",
                 hop_id=node_id, field="next_hop_id")
        if node_id in visited:
            return
        visiting.add(node_id)
        for neighbor in sorted(edges[node_id]):
            visit(neighbor)
        visiting.remove(node_id)
        visited.add(node_id)

    for hop in hops:
        visit(hop.hop_id)

    return ChainSpec(
        format=data["format"],
        hop_set_version=hop_set_version,
        client_id=client_id,
        origin_id=origin_id,
        allowed_scale_factors=tuple(allowed_tuple),
        hops=tuple(hops),
        source_path=str(source_path) if source_path else None,
    )


def load_chain(path: str | Path) -> ChainSpec:
    return parse_chain(_read_json(path), source_path=path)


def parse_origin(data: dict[str, Any]) -> OriginSpec:
    if data.get("format") != ORIGIN_FORMAT:
        raise SpecError(f"origin format must be {ORIGIN_FORMAT!r}", field="format")
    origin_id = data.get("origin_id")
    _require(isinstance(origin_id, str) and origin_id,
             "origin_id must be a non-empty string", hop_id="origin", field="origin_id")
    port = data.get("port")
    _require(_is_int(port) and 1024 <= port <= 65535,
             "origin port must be an integer in 1024..65535",
             hop_id=origin_id, field="port")
    seed = data.get("seed")
    _require(_is_int(seed) and seed >= 0, "seed must be a non-negative integer",
             hop_id=origin_id, field="seed")
    body_bytes = data.get("body_bytes")
    _require(_is_int(body_bytes) and 1 <= body_bytes <= 1_048_576,
             "body_bytes must be an integer in 1..1048576",
             hop_id=origin_id, field="body_bytes")
    config_version = data.get("config_version", 1)
    _require(_is_int(config_version) and config_version > 0,
             "origin config_version must be a positive integer",
             hop_id=origin_id, field="config_version")
    raw_samples = data.get("samples")
    _require(isinstance(raw_samples, list) and raw_samples,
             "samples must be a non-empty array", hop_id=origin_id, field="samples")
    samples: list[OriginSample] = []
    seen_requests: set[str] = set()
    seen_paths: set[str] = set()
    for index, raw_sample in enumerate(raw_samples):
        _require(isinstance(raw_sample, dict), "sample must be an object",
                 hop_id=origin_id, field=f"samples[{index}]")
        request_id = raw_sample.get("request_id")
        _require(isinstance(request_id, str) and request_id,
                 "sample request_id must be a non-empty string",
                 hop_id=origin_id, field=f"samples[{index}].request_id")
        _require(request_id not in seen_requests, "sample request_id values must be unique",
                 hop_id=origin_id, field=f"samples[{index}].request_id")
        path = raw_sample.get("path")
        _require(isinstance(path, str) and path.startswith("/"),
                 "sample path must start with '/'", hop_id=origin_id,
                 field=f"samples[{index}].path")
        _require(path not in seen_paths, "sample path values must be unique",
                 hop_id=origin_id, field=f"samples[{index}].path")
        seed_offset = raw_sample.get("seed_offset")
        _require(_is_int(seed_offset) and seed_offset >= 0,
                 "seed_offset must be a non-negative integer", hop_id=origin_id,
                 field=f"samples[{index}].seed_offset")
        seen_requests.add(request_id)
        seen_paths.add(path)
        samples.append(OriginSample(request_id, path, seed_offset))
    return OriginSpec(origin_id, port, seed, body_bytes, tuple(samples), config_version)


def load_origin(path: str | Path) -> OriginSpec:
    return parse_origin(_read_json(path))


def parse_diagnostic(data: dict[str, Any]) -> DiagnosticConfig:
    diagnostic_id = data.get("diagnostic_id", "diagnostic")
    _require(isinstance(diagnostic_id, str) and diagnostic_id,
             "diagnostic_id must be a non-empty string", hop_id=None,
             field="diagnostic_id")
    format_version = data.get("format_version")
    _require(_is_int(format_version) and format_version == DIAGNOSTIC_FORMAT_VERSION,
             f"format_version must be {DIAGNOSTIC_FORMAT_VERSION}", hop_id=None,
             field="format_version")
    caliber = data.get("scale_caliber_version")
    _require(_is_int(caliber) and caliber > 0,
             "scale_caliber_version must be a positive integer", hop_id=None,
             field="scale_caliber_version")
    session_scale = data.get("session_scale")
    _require(is_power_of_ten(session_scale),
             "session_scale must be a positive integer power of 10", hop_id=None,
             field="session_scale")

    timeout_seconds = data.get("timeout_seconds")
    _require(_is_number(timeout_seconds) and 0 < float(timeout_seconds) <= 30,
             "timeout_seconds must be a number in (0, 30]", hop_id=None,
             field="timeout_seconds")
    handshake_timeout = data.get("handshake_timeout_seconds")
    _require(_is_number(handshake_timeout) and 0 < float(handshake_timeout) <= 10,
             "handshake_timeout_seconds must be a number in (0, 10]", hop_id=None,
             field="handshake_timeout_seconds")
    batch_size = data.get("batch_size")
    _require(_is_int(batch_size) and batch_size > 0,
             "batch_size must be a positive integer", hop_id=None, field="batch_size")

    raw_scheduler = data.get("scheduler", {})
    _require(isinstance(raw_scheduler, dict), "scheduler must be an object",
             hop_id=None, field="scheduler")
    max_in_flight = raw_scheduler.get("max_in_flight", 2)
    hop_quota = raw_scheduler.get("hop_quota", 1)
    pool_workers = raw_scheduler.get("pool_workers", 4)
    _require(_is_int(max_in_flight) and max_in_flight > 0,
             "scheduler.max_in_flight must be a positive integer", hop_id=None,
             field="scheduler.max_in_flight")
    _require(_is_int(hop_quota) and hop_quota > 0,
             "scheduler.hop_quota must be a positive integer", hop_id=None,
             field="scheduler.hop_quota")
    _require(_is_int(pool_workers) and pool_workers >= max_in_flight,
             "scheduler.pool_workers must be at least max_in_flight", hop_id=None,
             field="scheduler.pool_workers")

    raw_error = data.get("error_model", {})
    _require(isinstance(raw_error, dict), "error_model must be an object",
             hop_id=None, field="error_model")
    dominance = raw_error.get("dominance_threshold_bp", 6_000)
    margin = raw_error.get("indistinguishable_margin_bp", 500)
    high_failure = raw_error.get("high_failure_ratio_bp", 5_000)
    for name, value in (
        ("dominance_threshold_bp", dominance),
        ("indistinguishable_margin_bp", margin),
        ("high_failure_ratio_bp", high_failure),
    ):
        _require(_is_int(value) and 0 <= value <= 10_000,
                 f"error_model.{name} must be an integer in 0..10000",
                 hop_id=None, field=f"error_model.{name}")

    return DiagnosticConfig(
        diagnostic_id=diagnostic_id,
        format_version=format_version,
        scale_caliber_version=caliber,
        session_scale=session_scale,
        timeout_seconds=float(timeout_seconds),
        handshake_timeout_seconds=float(handshake_timeout),
        batch_size=batch_size,
        scheduler=SchedulerConfig(max_in_flight, hop_quota, pool_workers),
        error_model=ErrorModelConfig(dominance, margin, high_failure),
    )


def load_diagnostic(path: str | Path) -> DiagnosticConfig:
    return parse_diagnostic(_read_json(path))


@dataclass(frozen=True)
class SpecBundle:
    chain: ChainSpec
    origin: OriginSpec
    diagnostic: DiagnosticConfig
    config_dir: Path

    @property
    def request_ids(self) -> list[str]:
        return [sample.request_id for sample in self.origin.samples]

    @property
    def ports(self) -> dict[str, int]:
        result = {hop.hop_id: hop.port for hop in self.chain.hops}
        result[self.origin.origin_id] = self.origin.port
        return result


def validate_bundle(chain: ChainSpec, origin: OriginSpec, diagnostic: DiagnosticConfig) -> None:
    if chain.origin_id != origin.origin_id:
        raise SpecError(
            f"chain origin_id {chain.origin_id!r} does not match origin declaration {origin.origin_id!r}",
            hop_id=origin.origin_id,
            field="origin_id",
        )
    used_ports: dict[int, str] = {hop.port: hop.hop_id for hop in chain.hops}
    if origin.port in used_ports:
        raise SpecError(
            f"origin port {origin.port} duplicates hop {used_ports[origin.port]!r}",
            hop_id=origin.origin_id,
            field="port",
        )
    if diagnostic.session_scale not in chain.allowed_scale_factors:
        raise SpecError(
            "diagnostic.session_scale must be in chain.allowed_scale_factors",
            hop_id=None,
            field="diagnostic.session_scale",
        )
    chain_requests = None
    # Sample request IDs are the request universe and are validated for uniqueness above.
    if not origin.samples:
        raise SpecError("origin requires at least one sample", hop_id=origin.origin_id,
                        field="samples")


def load_bundle(config_dir: str | Path) -> SpecBundle:
    config_dir = Path(config_dir)
    chain = load_chain(config_dir / "chain.json")
    origin = load_origin(config_dir / "origin.json")
    diagnostic = load_diagnostic(config_dir / "diagnostic.json")
    validate_bundle(chain, origin, diagnostic)
    return SpecBundle(chain, origin, diagnostic, config_dir)


def iter_invalid_examples() -> Iterable[dict[str, Any]]:
    """Small helper used by documentation/tests; not part of validation flow."""

    return iter(())
