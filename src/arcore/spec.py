"""Local JSON declaration loading with strict validation."""

import json
from dataclasses import dataclass
from pathlib import Path

from .errors import SpecError
from .model import SourceSpec, Topology


def _is_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _require_int(value, field: str, *, source_id: str | None = None,
                 minimum: int = 0) -> int:
    if not _is_int(value):
        raise SpecError(f"{field} must be an integer, got {type(value).__name__}",
                        source_id=source_id, field=field)
    if value < minimum:
        raise SpecError(f"{field} must be >= {minimum}, got {value}",
                        source_id=source_id, field=field)
    return value


def _require_string_list(value, field: str, *, source_id: str | None,
                         nonempty: bool = True) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise SpecError(f"{field} must be a list of strings",
                        source_id=source_id, field=field)
    if nonempty and not value:
        raise SpecError(f"{field} must not be empty",
                        source_id=source_id, field=field)
    if len(set(value)) != len(value):
        raise SpecError(f"{field} contains duplicate entries",
                        source_id=source_id, field=field)
    if any(not x for x in value):
        raise SpecError(f"{field} contains empty strings",
                        source_id=source_id, field=field)
    return tuple(value)


@dataclass(frozen=True)
class Parameters:
    silence_window_ms: int
    correlation_window_ms: int
    global_concurrency: int
    service_concurrency: int
    lease_ttl_ms: int
    exact_alert_threshold: int
    fingerprint_cardinality_threshold: int
    layer_weights: tuple[int, ...]
    fixed_point_scale: int
    saturation_max: int


@dataclass(frozen=True)
class Declaration:
    sources: tuple[SourceSpec, ...]
    topology: Topology
    parameters: Parameters


def _build_source(raw: dict, index: int) -> SourceSpec:
    field_prefix = f"sources[{index}]"
    source_id = raw.get("source_id")
    if not isinstance(source_id, str) or not source_id:
        raise SpecError(f"{field_prefix}.source_id must be a non-empty string",
                        field=f"{field_prefix}.source_id")
    seed = _require_int(raw.get("seed"), f"{field_prefix}.seed",
                        source_id=source_id, minimum=0)
    record_count = _require_int(
        raw.get("record_count"), f"{field_prefix}.record_count",
        source_id=source_id, minimum=0)
    services = _require_string_list(raw.get("services"),
                                    f"{field_prefix}.services",
                                    source_id=source_id)
    levels = _require_string_list(raw.get("levels"),
                                  f"{field_prefix}.levels",
                                  source_id=source_id)
    start_ts = _require_int(raw.get("start_ts_ms", 0),
                            f"{field_prefix}.start_ts_ms",
                            source_id=source_id, minimum=0)
    max_gap = _require_int(raw.get("max_gap_ms", 1000),
                           f"{field_prefix}.max_gap_ms",
                           source_id=source_id, minimum=0)
    namespace = _require_int(raw.get("fingerprint_namespace", 16),
                             f"{field_prefix}.fingerprint_namespace",
                             source_id=source_id, minimum=1)
    vocabulary = _require_string_list(
        raw.get("label_vocabulary", []), f"{field_prefix}.label_vocabulary",
        source_id=source_id, nonempty=False)
    max_labels = _require_int(raw.get("max_labels", len(vocabulary)),
                              f"{field_prefix}.max_labels",
                              source_id=source_id, minimum=0)
    if max_labels > len(vocabulary):
        raise SpecError(
            f"{field_prefix}.max_labels exceeds label_vocabulary size",
            source_id=source_id, field=f"{field_prefix}.max_labels")
    probability = raw.get("closed_probability", 0.25)
    if not isinstance(probability, (int, float)) or isinstance(probability, bool):
        raise SpecError(f"{field_prefix}.closed_probability must be a number",
                        source_id=source_id,
                        field=f"{field_prefix}.closed_probability")
    if not 0.0 <= float(probability) <= 1.0:
        raise SpecError(
            f"{field_prefix}.closed_probability must be within [0, 1]",
            source_id=source_id, field=f"{field_prefix}.closed_probability")
    return SourceSpec(
        source_id=source_id, seed=seed, record_count=record_count,
        services=services, levels=levels, start_ts_ms=start_ts,
        max_gap_ms=max_gap, fingerprint_namespace=namespace,
        label_vocabulary=vocabulary, max_labels=max_labels,
        closed_probability=float(probability))


def _build_parameters(raw: dict) -> Parameters:
    if not isinstance(raw, dict):
        raise SpecError("parameters must be an object", field="parameters")
    silence = _require_int(raw.get("silence_window_ms"),
                           "parameters.silence_window_ms", minimum=1)
    window = _require_int(raw.get("correlation_window_ms"),
                          "parameters.correlation_window_ms", minimum=1)
    global_cap = _require_int(raw.get("global_concurrency", 4),
                              "parameters.global_concurrency", minimum=1)
    service_cap = _require_int(raw.get("service_concurrency", 2),
                               "parameters.service_concurrency", minimum=1)
    ttl = _require_int(raw.get("lease_ttl_ms"),
                       "parameters.lease_ttl_ms", minimum=1)
    exact_threshold = _require_int(raw.get("exact_alert_threshold", 500),
                                   "parameters.exact_alert_threshold",
                                   minimum=0)
    cardinality_threshold = _require_int(
        raw.get("fingerprint_cardinality_threshold", 64),
        "parameters.fingerprint_cardinality_threshold", minimum=0)
    weights = raw.get("layer_weights")
    if not isinstance(weights, list) or not weights or any(
            not _is_int(w) for w in weights):
        raise SpecError("parameters.layer_weights must be a non-empty list of integers",
                        field="parameters.layer_weights")
    scale = _require_int(raw.get("fixed_point_scale", 10000),
                         "parameters.fixed_point_scale", minimum=1)
    powers_of_ten = {10 ** k for k in range(1, 13)}
    if scale not in powers_of_ten:
        raise SpecError("parameters.fixed_point_scale must be a power of 10",
                        field="parameters.fixed_point_scale")
    saturation = _require_int(raw.get("saturation_max", 1_000_000),
                              "parameters.saturation_max", minimum=1)
    return Parameters(
        silence_window_ms=silence, correlation_window_ms=window,
        global_concurrency=global_cap, service_concurrency=service_cap,
        lease_ttl_ms=ttl, exact_alert_threshold=exact_threshold,
        fingerprint_cardinality_threshold=cardinality_threshold,
        layer_weights=tuple(weights), fixed_point_scale=scale,
        saturation_max=saturation)


def load_declaration(path: str | Path) -> Declaration:
    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise SpecError(f"declaration file not found: {path}", field="file")
    except json.JSONDecodeError as exc:
        raise SpecError(f"invalid JSON: {exc}", field="file") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("sources"), list):
        raise SpecError("declaration must contain a sources list",
                        field="sources")
    if not raw["sources"]:
        raise SpecError("at least one alert source is required",
                        field="sources")
    seen_ids: set[str] = set()
    sources: list[SourceSpec] = []
    declared_services: set[str] = set()
    for index, item in enumerate(raw["sources"]):
        if not isinstance(item, dict):
            raise SpecError(f"sources[{index}] must be an object",
                            field=f"sources[{index}]")
        source = _build_source(item, index)
        if source.source_id in seen_ids:
            raise SpecError(
                f"duplicate source identifier {source.source_id!r}",
                source_id=source.source_id,
                field=f"sources[{index}].source_id")
        seen_ids.add(source.source_id)
        sources.append(source)
        declared_services.update(source.services)
    topo_raw = raw.get("topology") or {}
    edges_raw = topo_raw.get("edges", []) if isinstance(topo_raw, dict) else None
    if not isinstance(edges_raw, list):
        raise SpecError("topology.edges must be a list of [from, to] pairs",
                        field="topology.edges")
    edges: list[tuple[str, str]] = []
    for index, edge in enumerate(edges_raw):
        if (not isinstance(edge, list) or len(edge) != 2
                or not all(isinstance(x, str) for x in edge)):
            raise SpecError(
                f"topology.edges[{index}] must be a [from, to] string pair",
                field=f"topology.edges[{index}]")
        edges.append((edge[0], edge[1]))
    topology = Topology.build(edges)
    referenced = set(topology.nodes)
    undeclared = sorted(referenced.difference(declared_services))
    missing = sorted(declared_services.difference(referenced))
    if undeclared:
        raise SpecError(
            f"topology references services not declared by any source: "
            f"{undeclared}", field="topology.edges")
    if missing:
        raise SpecError(
            f"services missing from topology: {missing}",
            field="topology.edges")
    parameters = _build_parameters(raw.get("parameters", {}))
    return Declaration(sources=tuple(sources), topology=topology,
                       parameters=parameters)
