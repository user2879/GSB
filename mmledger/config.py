"""Configuration, request-stream, and merge-rule validation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import (
    E_CONFIG_JSON,
    E_CONFIG_SCHEMA,
    E_LIMIT_INVALID,
    E_MERGE_PARAMETER_SET_UNKNOWN,
    E_MERGE_PARAMETER_UNKNOWN,
    E_MERGE_WORKER_SET_UNKNOWN,
    E_MERGE_WORKER_UNKNOWN,
    E_MODEL_DIMENSION,
    E_QUANTIZATION_BITS,
    E_REQUEST_ARRIVAL,
    E_REQUEST_BATCH_SIZE,
    E_REQUEST_KIND,
    E_REQUEST_SEQUENCE,
    E_REQUEST_WORKER_UNKNOWN,
    E_WORKER_DUPLICATE_ID,
    E_WORKER_HOST,
    E_WORKER_PARAMETER_UNKNOWN,
    E_WORKER_PORT_DUPLICATE,
    E_WORKER_PORT_RANGE,
    Failure,
    sort_failures,
)
from .jsonutil import hash_json
from .models import (
    REQUEST_KINDS,
    ConfigBundle,
    LimitsSpec,
    MergeRules,
    ModelSpec,
    ParameterSpec,
    RequestSpec,
    WorkerSpec,
)


@dataclass(frozen=True)
class ConfigLoadResult:
    bundle: ConfigBundle | None
    failures: tuple[Failure, ...]


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _nonempty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _read_json(path: Path, source: str, failures: list[Failure]) -> Any | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        failures.append(Failure(E_CONFIG_JSON, f"cannot read JSON: {exc}", source))
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        failures.append(
            Failure(
                E_CONFIG_JSON,
                f"invalid JSON: {exc.msg}",
                source,
                ordinal=int(exc.lineno),
                details={"line": exc.lineno, "column": exc.colno},
            )
        )
        return None


def _default_worker_quota(worker_raw: dict[str, Any], limits: LimitsSpec | None) -> int:
    value = worker_raw.get("quota_points")
    if _is_int(value) and value >= 0:
        return value
    return limits.worker_quota_points if limits is not None else 0


def load_config(config_path: str | Path) -> ConfigLoadResult:
    path = Path(config_path).resolve()
    failures: list[Failure] = []
    source = str(path)
    raw = _read_json(path, source, failures)
    if not isinstance(raw, dict):
        if raw is not None:
            failures.append(Failure(E_CONFIG_SCHEMA, "config root must be an object", source))
        return ConfigLoadResult(None, tuple(sort_failures(failures)))

    if raw.get("schema_version") not in (None, "mmledger-config-v1", 1, "1"):
        failures.append(Failure(E_CONFIG_SCHEMA, "unsupported config schema_version", source))

    root_value = raw.get("workspace_root", ".")
    if not isinstance(root_value, str):
        failures.append(Failure(E_CONFIG_SCHEMA, "workspace_root must be a string", source))
        root_value = "."
    root = (path.parent / root_value).resolve()

    seed = raw.get("seed")
    if not _is_int(seed):
        failures.append(Failure(E_CONFIG_SCHEMA, "seed must be an integer", source))
        seed = 0

    model_raw = raw.get("model")
    model: ModelSpec | None = None
    if not isinstance(model_raw, dict):
        failures.append(Failure(E_CONFIG_SCHEMA, "model must be an object", source))
    else:
        input_dim = model_raw.get("input_dim")
        hidden_dim = model_raw.get("hidden_dim")
        quantization_bits = model_raw.get("quantization_bits")
        if not (_is_int(input_dim) and input_dim > 0):
            failures.append(
                Failure(E_MODEL_DIMENSION, "model.input_dim must be a positive integer", source)
            )
        if not (_is_int(hidden_dim) and hidden_dim > 0):
            failures.append(
                Failure(E_MODEL_DIMENSION, "model.hidden_dim must be a positive integer", source)
            )
        if not (_is_int(quantization_bits) and 1 <= quantization_bits <= 16):
            failures.append(
                Failure(
                    E_QUANTIZATION_BITS,
                    "model.quantization_bits must be an integer in [1, 16]",
                    source,
                )
            )
        if (
            _is_int(input_dim)
            and input_dim > 0
            and _is_int(hidden_dim)
            and hidden_dim > 0
            and _is_int(quantization_bits)
            and 1 <= quantization_bits <= 16
        ):
            model = ModelSpec(input_dim, hidden_dim, quantization_bits)

    limits_raw = raw.get("limits")
    limits: LimitsSpec | None = None
    if not isinstance(limits_raw, dict):
        failures.append(Failure(E_LIMIT_INVALID, "limits must be an object", source))
    else:
        batch_size = limits_raw.get("batch_size")
        normalization_budget = limits_raw.get("normalization_budget")
        worker_quota_points = limits_raw.get("worker_quota_points")
        if not (_is_int(batch_size) and batch_size > 0):
            failures.append(Failure(E_LIMIT_INVALID, "limits.batch_size must be positive", source))
        if not (_is_int(normalization_budget) and normalization_budget >= 0):
            failures.append(
                Failure(E_LIMIT_INVALID, "limits.normalization_budget must be non-negative", source)
            )
        if not (_is_int(worker_quota_points) and worker_quota_points >= 0):
            failures.append(
                Failure(E_LIMIT_INVALID, "limits.worker_quota_points must be non-negative", source)
            )
        if (
            _is_int(batch_size)
            and batch_size > 0
            and _is_int(normalization_budget)
            and normalization_budget >= 0
            and _is_int(worker_quota_points)
            and worker_quota_points >= 0
        ):
            limits = LimitsSpec(batch_size, normalization_budget, worker_quota_points)

    parameters_raw = raw.get("parameters", [])
    parameters: list[ParameterSpec] = []
    versions_seen: set[int] = set()
    if not isinstance(parameters_raw, list):
        failures.append(Failure(E_CONFIG_SCHEMA, "parameters must be a list", source))
        parameters_raw = []
    for index, parameter_raw in enumerate(parameters_raw):
        if not isinstance(parameter_raw, dict):
            failures.append(Failure(E_CONFIG_SCHEMA, "parameter entry must be an object", source, index))
            continue
        version = parameter_raw.get("parameter_version", parameter_raw.get("version"))
        digest = parameter_raw.get("digest")
        if not (_is_int(version) and version > 0):
            failures.append(
                Failure(E_CONFIG_SCHEMA, "parameter_version must be a positive integer", source, index)
            )
            continue
        if version in versions_seen:
            failures.append(
                Failure(
                    E_CONFIG_SCHEMA,
                    f"duplicate parameter_version {version}",
                    source,
                    index,
                )
            )
            continue
        if not _nonempty_str(digest):
            failures.append(Failure(E_CONFIG_SCHEMA, "parameter digest must be non-empty", source, index))
            continue
        versions_seen.add(version)
        parameters.append(ParameterSpec(version, digest))

    port_range_raw = raw.get("port_range", {})
    min_port = 1024
    max_port = 65535
    if isinstance(port_range_raw, dict):
        min_port = port_range_raw.get("min", min_port)
        max_port = port_range_raw.get("max", max_port)
    if not (_is_int(min_port) and _is_int(max_port) and 1024 <= min_port <= max_port <= 65535):
        failures.append(
            Failure(E_WORKER_PORT_RANGE, "port_range must satisfy 1024 <= min <= max <= 65535", source)
        )
        min_port, max_port = 1024, 65535

    workers_raw = raw.get("workers")
    workers: list[WorkerSpec] = []
    worker_ids: set[str] = set()
    worker_ports: set[int] = set()
    if not isinstance(workers_raw, list):
        failures.append(Failure(E_CONFIG_SCHEMA, "workers must be a list", source))
        workers_raw = []
    for index, worker_raw in enumerate(workers_raw):
        if not isinstance(worker_raw, dict):
            failures.append(Failure(E_CONFIG_SCHEMA, "worker entry must be an object", source, index))
            continue
        worker_id = worker_raw.get("worker_id")
        host = worker_raw.get("host", "127.0.0.1")
        port = worker_raw.get("port")
        parameter_version = worker_raw.get("parameter_version")
        parameter_digest = worker_raw.get("parameter_digest")
        valid = True
        if not _nonempty_str(worker_id):
            failures.append(Failure(E_CONFIG_SCHEMA, "worker_id must be non-empty", source, index))
            valid = False
        elif worker_id in worker_ids:
            failures.append(
                Failure(E_WORKER_DUPLICATE_ID, f"duplicate worker_id {worker_id}", source, index)
            )
            valid = False
        if host != "127.0.0.1":
            failures.append(Failure(E_WORKER_HOST, "worker host must be 127.0.0.1", source, index))
            valid = False
        if not (_is_int(port) and min_port <= port <= max_port):
            failures.append(
                Failure(
                    E_WORKER_PORT_RANGE,
                    f"worker port must be an integer in [{min_port}, {max_port}]",
                    source,
                    index,
                )
            )
            valid = False
        elif port in worker_ports:
            failures.append(
                Failure(E_WORKER_PORT_DUPLICATE, f"duplicate worker port {port}", source, index)
            )
            valid = False
        if not (_is_int(parameter_version) and parameter_version > 0):
            failures.append(
                Failure(E_CONFIG_SCHEMA, "worker parameter_version must be positive", source, index)
            )
            valid = False
        elif parameter_version not in versions_seen:
            failures.append(
                Failure(
                    E_WORKER_PARAMETER_UNKNOWN,
                    f"unknown worker parameter_version {parameter_version}",
                    source,
                    index,
                )
            )
            valid = False
        if not _nonempty_str(parameter_digest):
            failures.append(
                Failure(E_CONFIG_SCHEMA, "worker parameter_digest must be non-empty", source, index)
            )
            valid = False
        quota_value = worker_raw.get("quota_points")
        if quota_value is not None and not (_is_int(quota_value) and quota_value >= 0):
            failures.append(
                Failure(E_LIMIT_INVALID, "worker quota_points must be a non-negative integer", source, index)
            )
            valid = False
        if valid:
            worker_ids.add(worker_id)
            worker_ports.add(port)
            workers.append(
                WorkerSpec(
                    worker_id=worker_id,
                    host=host,
                    port=port,
                    parameter_version=parameter_version,
                    parameter_digest=parameter_digest,
                    quota_points=_default_worker_quota(worker_raw, limits),
                )
            )

    request_stream_value = raw.get("request_stream", "requests.json")
    merge_rules_value = raw.get("merge_rules", "merge_rules.json")
    receipt_glob = raw.get("receipt_glob", "receipts/*.json")
    fault_injections = raw.get("fault_injections", "fault_injections.json")
    fault_receipt_glob = raw.get("fault_receipt_glob", "fault_receipts/*.json")
    for name, value in (
        ("request_stream", request_stream_value),
        ("merge_rules", merge_rules_value),
        ("receipt_glob", receipt_glob),
        ("fault_injections", fault_injections),
        ("fault_receipt_glob", fault_receipt_glob),
    ):
        if not _nonempty_str(value):
            failures.append(Failure(E_CONFIG_SCHEMA, f"{name} must be a non-empty string", source))

    requests_path = path.parent / str(request_stream_value)
    merge_path = path.parent / str(merge_rules_value)
    raw_requests = _read_json(requests_path, str(requests_path), failures)
    raw_merge = _read_json(merge_path, str(merge_path), failures)

    requests: list[RequestSpec] = []
    if not isinstance(raw_requests, dict):
        failures.append(Failure(E_CONFIG_SCHEMA, "request stream root must be an object", str(requests_path)))
        raw_requests = {}
    request_items = raw_requests.get("requests", [])
    if not isinstance(request_items, list):
        failures.append(Failure(E_CONFIG_SCHEMA, "request stream requests must be a list", str(requests_path)))
        request_items = []

    previous_sequence: int | None = None
    batch_counts: dict[str, int] = {}
    for index, request_raw in enumerate(request_items):
        request_source = str(requests_path)
        if not isinstance(request_raw, dict):
            failures.append(Failure(E_CONFIG_SCHEMA, "request entry must be an object", request_source, index))
            continue
        sequence = request_raw.get("sequence")
        request_id = request_raw.get("request_id")
        kind = request_raw.get("kind")
        arrival_ms = request_raw.get("arrival_ms")
        batch_id = request_raw.get("batch_id")
        worker_id = request_raw.get("worker_id")
        input_seed = request_raw.get("input_seed", seed + index)
        overwritten_ref = request_raw.get("overwritten_ref")
        valid = True
        if not (_is_int(sequence) and sequence >= 0):
            failures.append(Failure(E_REQUEST_SEQUENCE, "sequence must be a non-negative integer", request_source, index))
            valid = False
        elif previous_sequence is not None and sequence <= previous_sequence:
            failures.append(
                Failure(
                    E_REQUEST_SEQUENCE,
                    f"sequence {sequence} is not strictly greater than {previous_sequence}",
                    request_source,
                    index,
                )
            )
            valid = False
        else:
            previous_sequence = sequence
        if not (_is_int(arrival_ms) and arrival_ms >= 0):
            failures.append(Failure(E_REQUEST_ARRIVAL, "arrival_ms must be a non-negative integer", request_source, index))
            valid = False
        if not _nonempty_str(request_id):
            failures.append(Failure(E_CONFIG_SCHEMA, "request_id must be non-empty", request_source, index))
            valid = False
        if kind not in REQUEST_KINDS:
            failures.append(Failure(E_REQUEST_KIND, f"kind must be one of {REQUEST_KINDS}", request_source, index))
            valid = False
        if not _nonempty_str(batch_id):
            failures.append(Failure(E_CONFIG_SCHEMA, "batch_id must be non-empty", request_source, index))
            valid = False
        if worker_id not in worker_ids:
            failures.append(
                Failure(E_REQUEST_WORKER_UNKNOWN, f"unknown request worker_id {worker_id}", request_source, index)
            )
            valid = False
        if not _is_int(input_seed):
            failures.append(Failure(E_CONFIG_SCHEMA, "input_seed must be an integer", request_source, index))
            valid = False
        if overwritten_ref is not None and not _nonempty_str(overwritten_ref):
            failures.append(
                Failure(E_CONFIG_SCHEMA, "overwritten_ref must be null or a non-empty string", request_source, index)
            )
            valid = False
        if valid:
            request = RequestSpec(
                sequence=sequence,
                request_id=request_id,
                kind=kind,
                arrival_ms=arrival_ms,
                batch_id=batch_id,
                worker_id=worker_id,
                input_seed=input_seed,
                overwritten_ref=overwritten_ref,
            )
            requests.append(request)
            batch_counts[batch_id] = batch_counts.get(batch_id, 0) + 1

    if limits is not None:
        batch_size = limits.batch_size
        for batch_id, count in sorted(batch_counts.items()):
            if count > batch_size:
                failures.append(
                    Failure(
                        E_REQUEST_BATCH_SIZE,
                        f"batch {batch_id} has {count} requests, exceeds batch_size {batch_size}",
                        str(requests_path),
                        item=batch_id,
                    )
                )

    merge_rules: MergeRules | None = None
    if not isinstance(raw_merge, dict):
        failures.append(Failure(E_CONFIG_SCHEMA, "merge rules root must be an object", str(merge_path)))
        raw_merge = {}
    worker_sets_raw = raw_merge.get("worker_sets", {})
    parameter_sets_raw = raw_merge.get("parameter_version_sets", {})
    if not isinstance(worker_sets_raw, dict):
        failures.append(Failure(E_CONFIG_SCHEMA, "merge_rules.worker_sets must be an object", str(merge_path)))
        worker_sets_raw = {}
    if not isinstance(parameter_sets_raw, dict):
        failures.append(
            Failure(E_CONFIG_SCHEMA, "merge_rules.parameter_version_sets must be an object", str(merge_path))
        )
        parameter_sets_raw = {}

    worker_sets: dict[str, tuple[str, ...]] = {}
    for set_name, members in sorted(worker_sets_raw.items()):
        if not isinstance(set_name, str) or not set_name:
            failures.append(Failure(E_CONFIG_SCHEMA, "worker set name must be non-empty", str(merge_path)))
            continue
        if not isinstance(members, list):
            failures.append(Failure(E_CONFIG_SCHEMA, f"worker set {set_name} must be a list", str(merge_path)))
            continue
        normalized: list[str] = []
        for member in members:
            if member not in worker_ids:
                failures.append(
                    Failure(
                        E_MERGE_WORKER_UNKNOWN,
                        f"worker set {set_name} references unknown worker {member}",
                        str(merge_path),
                        item=set_name,
                    )
                )
            else:
                normalized.append(member)
        worker_sets[set_name] = tuple(normalized)

    parameter_sets: dict[str, tuple[int, ...]] = {}
    for set_name, members in sorted(parameter_sets_raw.items()):
        if not isinstance(set_name, str) or not set_name:
            failures.append(Failure(E_CONFIG_SCHEMA, "parameter set name must be non-empty", str(merge_path)))
            continue
        if not isinstance(members, list):
            failures.append(Failure(E_CONFIG_SCHEMA, f"parameter set {set_name} must be a list", str(merge_path)))
            continue
        normalized_versions: list[int] = []
        for member in members:
            if not (_is_int(member) and member in versions_seen):
                failures.append(
                    Failure(
                        E_MERGE_PARAMETER_UNKNOWN,
                        f"parameter set {set_name} references unknown version {member}",
                        str(merge_path),
                        item=set_name,
                    )
                )
            else:
                normalized_versions.append(member)
        parameter_sets[set_name] = tuple(normalized_versions)

    active_worker_set = raw_merge.get("active_worker_set")
    active_parameter_set = raw_merge.get("active_parameter_version_set")
    if active_worker_set not in worker_sets:
        failures.append(
            Failure(
                E_MERGE_WORKER_SET_UNKNOWN,
                f"active_worker_set {active_worker_set!r} does not exist",
                str(merge_path),
            )
        )
    if active_parameter_set not in parameter_sets:
        failures.append(
            Failure(
                E_MERGE_PARAMETER_SET_UNKNOWN,
                f"active_parameter_version_set {active_parameter_set!r} does not exist",
                str(merge_path),
            )
        )

    priority_raw = raw_merge.get("worker_priority", list(worker_sets.get(active_worker_set, ())))
    if not isinstance(priority_raw, list):
        failures.append(Failure(E_CONFIG_SCHEMA, "worker_priority must be a list", str(merge_path)))
        priority_raw = []
    priority: list[str] = []
    for worker_id in priority_raw:
        if worker_id not in worker_ids:
            failures.append(
                Failure(
                    E_MERGE_WORKER_UNKNOWN,
                    f"worker_priority references unknown worker {worker_id}",
                    str(merge_path),
                )
            )
        else:
            priority.append(worker_id)

    if worker_sets and active_worker_set not in worker_sets:
        merge_rules = None
    elif parameter_sets and active_parameter_set not in parameter_sets:
        merge_rules = None
    elif active_worker_set in worker_sets and active_parameter_set in parameter_sets:
        merge_rules = MergeRules(
            worker_sets=worker_sets,
            parameter_version_sets=parameter_sets,
            active_worker_set=active_worker_set,
            active_parameter_version_set=active_parameter_set,
            worker_priority=tuple(priority),
        )

    if failures or model is None or limits is None or merge_rules is None:
        return ConfigLoadResult(None, tuple(sort_failures(failures)))

    config_hash = hash_json(raw)
    requests_hash = hash_json(raw_requests)
    merge_rules_hash = hash_json(raw_merge)
    bundle = ConfigBundle(
        config_path=str(path),
        root=str(root),
        raw_config=raw,
        raw_requests=raw_requests,
        raw_merge_rules=raw_merge,
        seed=seed,
        model=model,
        limits=limits,
        workers=tuple(sorted(workers, key=lambda item: item.worker_id)),
        parameters=tuple(sorted(parameters, key=lambda item: item.parameter_version)),
        requests=tuple(requests),
        merge_rules=merge_rules,
        receipt_glob=str(receipt_glob),
        fault_injections_path=str(fault_injections),
        fault_receipt_glob=str(fault_receipt_glob),
        config_hash=config_hash,
        requests_hash=requests_hash,
        merge_rules_hash=merge_rules_hash,
    )
    return ConfigLoadResult(bundle, ())


def known_parameter_versions(bundle: ConfigBundle) -> frozenset[int]:
    return frozenset(bundle.parameters_by_version)


def receipt_paths(bundle: ConfigBundle, *, fault: bool = False) -> list[Path]:
    config_dir = Path(bundle.config_path).parent
    patterns = [bundle.receipt_glob]
    if fault:
        patterns.append(bundle.fault_receipt_glob)
    paths: dict[str, Path] = {}
    for pattern in patterns:
        for path in config_dir.glob(pattern):
            paths[path.resolve().as_posix()] = path
    return sorted(paths.values(), key=lambda path: path.as_posix())


def load_injections(path: str | Path) -> tuple[dict[str, Any], ...]:
    injection_path = Path(path)
    if not injection_path.exists():
        return ()
    try:
        raw = json.loads(injection_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ()
    items = raw.get("injections", []) if isinstance(raw, dict) else []
    if not isinstance(items, list):
        return ()
    return tuple(item for item in items if isinstance(item, dict))


def injection_map(path: str | Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in load_injections(path):
        injection_id = item.get("id")
        source_file = item.get("source_file")
        if isinstance(injection_id, str) and isinstance(source_file, str):
            mapping[source_file] = injection_id
    return mapping


def flatten_worker_sets(worker_sets: Iterable[tuple[str, tuple[str, ...]]]) -> list[str]:
    values: set[str] = set()
    for _, members in worker_sets:
        values.update(members)
    return sorted(values)
