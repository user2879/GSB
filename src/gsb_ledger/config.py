"""Local JSON configuration and NDJSON request-stream validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .errors import (
    E_CONFIG_INVALID,
    E_REQUEST_INVALID,
    Issue,
    IssueCollector,
)
from .model import (
    AppConfig,
    MergeEntry,
    ReceiptSource,
    RequestSpec,
    WorkerSpec,
)
from .util import digest_value, is_int, read_json, sha256_file


_MERGE_RULES = {
    "covered_ref",
    "param_version_desc",
    "duration_ms_asc",
    "batch_id_asc",
    "worker_id_asc",
    "source_order_asc",
}


class ConfigValidation(Exception):
    def __init__(self, issues: IssueCollector):
        super().__init__("configuration or request validation failed")
        self.issues = issues


def _issue(code: str, message: str, *, path: str = "", item: str = "") -> Issue:
    return Issue(code=code, message=message, path=path, item=item)


def _require_int(
    value: Any,
    *,
    name: str,
    path: str,
    issues: IssueCollector,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int | None:
    if not is_int(value):
        issues.add(_issue(E_CONFIG_INVALID, f"{name} must be an integer", path=path, item=name))
        return None
    if minimum is not None and value < minimum:
        issues.add(
            _issue(
                E_CONFIG_INVALID,
                f"{name} must be >= {minimum}",
                path=path,
                item=name,
            )
        )
        return None
    if maximum is not None and value > maximum:
        issues.add(
            _issue(
                E_CONFIG_INVALID,
                f"{name} must be <= {maximum}",
                path=path,
                item=name,
            )
        )
        return None
    return value


def _require_id(
    value: Any,
    *,
    name: str,
    path: str,
    issues: IssueCollector,
    maximum_width: int,
) -> str | None:
    if not isinstance(value, str) or not value or len(value) > maximum_width:
        issues.add(
            _issue(
                E_CONFIG_INVALID,
                f"{name} must be a non-empty string of width <= {maximum_width}",
                path=path,
                item=name,
            )
        )
        return None
    if any(character.isspace() or character in "/\\\x00" for character in value):
        issues.add(
            _issue(
                E_CONFIG_INVALID,
                f"{name} contains a forbidden character",
                path=path,
                item=name,
            )
        )
        return None
    return value


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    issue_path = str(config_path)
    issues = IssueCollector()
    try:
        raw = read_json(config_path)
    except (OSError, json.JSONDecodeError, UnicodeError) as exc:
        issues.add(_issue(E_CONFIG_INVALID, f"cannot read config: {exc}", path=issue_path))
        raise ConfigValidation(issues) from exc

    if not isinstance(raw, dict):
        issues.add(_issue(E_CONFIG_INVALID, "config root must be an object", path=issue_path))
        raise ConfigValidation(issues)

    schema_version = _require_int(
        raw.get("schema_version"),
        name="schema_version",
        path=issue_path,
        issues=issues,
        minimum=1,
    )
    if schema_version != 1:
        issues.add(
            _issue(
                E_CONFIG_INVALID,
                "unsupported schema_version; expected 1",
                path=issue_path,
                item="schema_version",
            )
        )

    port_range = raw.get("port_range")
    if not isinstance(port_range, dict):
        issues.add(
            _issue(E_CONFIG_INVALID, "port_range must be an object", path=issue_path, item="port_range")
        )
        port_min, port_max = 1, 0
    else:
        port_min = _require_int(
            port_range.get("min"),
            name="port_range.min",
            path=issue_path,
            issues=issues,
            minimum=1,
            maximum=65535,
        )
        port_max = _require_int(
            port_range.get("max"),
            name="port_range.max",
            path=issue_path,
            issues=issues,
            minimum=1,
            maximum=65535,
        )
        if port_min is None or port_max is None:
            port_min, port_max = 1, 0
        elif port_min > port_max:
            issues.add(
                _issue(
                    E_CONFIG_INVALID,
                    "port_range.min must not exceed port_range.max",
                    path=issue_path,
                    item="port_range",
                )
            )
            port_min, port_max = 1, 0

    workers_raw = raw.get("workers")
    workers: list[WorkerSpec] = []
    seen_worker_ids: set[str] = set()
    seen_ports: set[int] = set()
    if not isinstance(workers_raw, list) or not workers_raw:
        issues.add(
            _issue(E_CONFIG_INVALID, "workers must be a non-empty list", path=issue_path, item="workers")
        )
        workers_raw = []
    for index, item in enumerate(workers_raw):
        item_name = f"workers[{index}]"
        if not isinstance(item, dict):
            issues.add(
                _issue(
                    E_CONFIG_INVALID,
                    "worker must be an object",
                    path=issue_path,
                    item=item_name,
                )
            )
            continue
        worker_id = _require_id(
            item.get("worker_id"),
            name=f"{item_name}.worker_id",
            path=issue_path,
            issues=issues,
            maximum_width=32,
        )
        param_version = _require_int(
            item.get("param_version"),
            name=f"{item_name}.param_version",
            path=issue_path,
            issues=issues,
            minimum=0,
            maximum=999_999,
        )
        port = _require_int(
            item.get("port"),
            name=f"{item_name}.port",
            path=issue_path,
            issues=issues,
            minimum=1,
            maximum=65535,
        )
        model_dim = _require_int(
            item.get("model_dim"),
            name=f"{item_name}.model_dim",
            path=issue_path,
            issues=issues,
            minimum=1,
        )
        hidden_dim = _require_int(
            item.get("hidden_dim"),
            name=f"{item_name}.hidden_dim",
            path=issue_path,
            issues=issues,
            minimum=1,
        )
        seed = _require_int(
            item.get("seed"),
            name=f"{item_name}.seed",
            path=issue_path,
            issues=issues,
            minimum=0,
        )
        quant_scale = _require_int(
            item.get("quant_scale", 1000),
            name=f"{item_name}.quant_scale",
            path=issue_path,
            issues=issues,
            minimum=1,
        )
        known_versions_raw = item.get("known_param_versions", [param_version])
        known_param_versions: tuple[int, ...] | None = None
        if not isinstance(known_versions_raw, list) or not all(
            is_int(value) and value >= 0 for value in known_versions_raw
        ):
            issues.add(
                _issue(
                    E_CONFIG_INVALID,
                    "known_param_versions must be a list of non-negative integers",
                    path=issue_path,
                    item=f"{item_name}.known_param_versions",
                )
            )
        else:
            known_param_versions = tuple(sorted(set(int(value) for value in known_versions_raw)))
            if param_version is not None and param_version not in known_param_versions:
                issues.add(
                    _issue(
                        E_CONFIG_INVALID,
                        "known_param_versions must include the active param_version",
                        path=issue_path,
                        item=f"{item_name}.known_param_versions",
                    )
                )
        if worker_id is not None:
            if worker_id in seen_worker_ids:
                issues.add(
                    _issue(
                        E_CONFIG_INVALID,
                        f"duplicate worker_id {worker_id!r}",
                        path=issue_path,
                        item=item_name,
                    )
                )
            seen_worker_ids.add(worker_id)
        if port is not None:
            if not (port_min <= port <= port_max):
                issues.add(
                    _issue(
                        E_CONFIG_INVALID,
                        f"port {port} outside allowed range {port_min}..{port_max}",
                        path=issue_path,
                        item=item_name,
                    )
                )
            if port in seen_ports:
                issues.add(
                    _issue(
                        E_CONFIG_INVALID,
                        f"duplicate loopback port {port}",
                        path=issue_path,
                        item=item_name,
                    )
                )
            seen_ports.add(port)
        if all(
            value is not None
            for value in (
                worker_id,
                param_version,
                port,
                model_dim,
                hidden_dim,
                seed,
                quant_scale,
                known_param_versions,
            )
        ):
            workers.append(
                WorkerSpec(
                    worker_id=str(worker_id),
                    param_version=int(param_version),
                    port=int(port),
                    model_dim=int(model_dim),
                    hidden_dim=int(hidden_dim),
                    seed=int(seed),
                    quant_scale=int(quant_scale),
                    known_param_versions=tuple(known_param_versions or (int(param_version),)),
                )
            )

    batch_size = _require_int(
        raw.get("batch_size"),
        name="batch_size",
        path=issue_path,
        issues=issues,
        minimum=1,
    )
    fixed_point_bits = _require_int(
        raw.get("fixed_point_bits", 32),
        name="fixed_point_bits",
        path=issue_path,
        issues=issues,
        minimum=8,
        maximum=64,
    )
    throughput = _require_int(
        raw.get("throughput_ops_per_ms", 1),
        name="throughput_ops_per_ms",
        path=issue_path,
        issues=issues,
        minimum=1,
    )

    merge_raw = raw.get("merge_rule")
    merge_entries: list[MergeEntry] = []
    tie_break = (
        "covered_ref",
        "duration_ms_asc",
        "param_version_desc",
        "batch_id_asc",
        "worker_id_asc",
        "source_order_asc",
    )
    if not isinstance(merge_raw, dict):
        issues.add(
            _issue(E_CONFIG_INVALID, "merge_rule must be an object", path=issue_path, item="merge_rule")
        )
    else:
        raw_tie_break = merge_raw.get("tie_break", list(tie_break))
        if not isinstance(raw_tie_break, list) or not all(
            isinstance(value, str) and value in _MERGE_RULES for value in raw_tie_break
        ):
            issues.add(
                _issue(
                    E_CONFIG_INVALID,
                    "merge_rule.tie_break contains an unknown rule",
                    path=issue_path,
                    item="merge_rule.tie_break",
                )
            )
        else:
            tie_break = tuple(raw_tie_break)

        entries_raw = merge_raw.get("entries")
        if entries_raw is None:
            worker_ids = merge_raw.get("worker_ids")
            param_versions = merge_raw.get("param_versions")
            if isinstance(worker_ids, list) and isinstance(param_versions, list):
                if len(worker_ids) == len(param_versions):
                    entries_raw = [
                        {"worker_id": worker_id, "param_version": version}
                        for worker_id, version in zip(worker_ids, param_versions, strict=True)
                    ]
        if not isinstance(entries_raw, list) or not entries_raw:
            issues.add(
                _issue(
                    E_CONFIG_INVALID,
                    "merge_rule must contain entries, or paired worker_ids/param_versions",
                    path=issue_path,
                    item="merge_rule.entries",
                )
            )
            entries_raw = []

        worker_pairs = {
            (worker.worker_id, version)
            for worker in workers
            for version in (worker.known_param_versions or (worker.param_version,))
        }
        for index, item in enumerate(entries_raw):
            item_name = f"merge_rule.entries[{index}]"
            if not isinstance(item, dict):
                issues.add(
                    _issue(
                        E_CONFIG_INVALID,
                        "merge entry must be an object",
                        path=issue_path,
                        item=item_name,
                    )
                )
                continue
            worker_id = item.get("worker_id")
            version = item.get("param_version")
            if not isinstance(worker_id, str) or worker_id not in seen_worker_ids:
                issues.add(
                    _issue(
                        E_CONFIG_INVALID,
                        f"merge rule references unknown worker {worker_id!r}",
                        path=issue_path,
                        item=item_name,
                    )
                )
                continue
            if not is_int(version) or (worker_id, version) not in worker_pairs:
                issues.add(
                    _issue(
                        E_CONFIG_INVALID,
                        f"merge rule references unknown param version {version!r} for {worker_id}",
                        path=issue_path,
                        item=item_name,
                    )
                )
                continue
            merge_entries.append(MergeEntry(worker_id=worker_id, param_version=int(version)))

    sources_raw = raw.get("receipt_sources", [])
    receipt_sources: list[ReceiptSource] = []
    if not isinstance(sources_raw, list):
        issues.add(
            _issue(
                E_CONFIG_INVALID,
                "receipt_sources must be a list",
                path=issue_path,
                item="receipt_sources",
            )
        )
        sources_raw = []
    seen_source_paths: set[str] = set()
    for index, item in enumerate(sources_raw):
        item_name = f"receipt_sources[{index}]"
        if isinstance(item, str):
            item = {"path": item}
        if not isinstance(item, dict):
            issues.add(
                _issue(
                    E_CONFIG_INVALID,
                    "receipt source must be a string or object",
                    path=issue_path,
                    item=item_name,
                )
            )
            continue
        source_path = item.get("path")
        if not isinstance(source_path, str) or not source_path:
            issues.add(
                _issue(
                    E_CONFIG_INVALID,
                    "receipt source path must be a non-empty string",
                    path=issue_path,
                    item=item_name,
                )
            )
            continue
        if source_path in seen_source_paths:
            issues.add(
                _issue(
                    E_CONFIG_INVALID,
                    f"duplicate receipt source {source_path!r}",
                    path=issue_path,
                    item=item_name,
                )
            )
        seen_source_paths.add(source_path)
        default_worker_id = item.get("default_worker_id")
        if default_worker_id is not None:
            if not isinstance(default_worker_id, str) or default_worker_id not in seen_worker_ids:
                issues.add(
                    _issue(
                        E_CONFIG_INVALID,
                        f"receipt source references unknown worker {default_worker_id!r}",
                        path=issue_path,
                        item=item_name,
                    )
                )
                default_worker_id = None
        default_param_version = item.get("default_param_version")
        if default_param_version is not None and (
            not is_int(default_param_version) or default_param_version < 0
        ):
            issues.add(
                _issue(
                    E_CONFIG_INVALID,
                    "receipt source default_param_version must be a non-negative integer",
                    path=issue_path,
                    item=item_name,
                )
            )
            default_param_version = None
        source_format = item.get("format")
        if source_format is not None and source_format not in {"r1", "r2", "r3"}:
            issues.add(
                _issue(
                    E_CONFIG_INVALID,
                    "receipt source format must be r1, r2 or r3",
                    path=issue_path,
                    item=item_name,
                )
            )
            source_format = None
        receipt_sources.append(
            ReceiptSource(
                path=source_path,
                default_worker_id=default_worker_id,
                default_param_version=default_param_version,
                format=source_format,
            )
        )

    interaction_budget_ms = _require_int(
        raw.get("interaction_budget_ms", 50),
        name="interaction_budget_ms",
        path=issue_path,
        issues=issues,
        minimum=0,
    )
    fault_config = raw.get("fault_config")
    if fault_config is not None and not isinstance(fault_config, str):
        issues.add(
            _issue(
                E_CONFIG_INVALID,
                "fault_config must be a string path",
                path=issue_path,
                item="fault_config",
            )
        )
        fault_config = None

    if issues:
        raise ConfigValidation(issues)

    assert schema_version is not None
    assert port_min is not None and port_max is not None
    assert batch_size is not None
    assert fixed_point_bits is not None
    assert throughput is not None
    assert interaction_budget_ms is not None
    return AppConfig(
        schema_version=int(schema_version),
        workers=tuple(workers),
        port_min=int(port_min),
        port_max=int(port_max),
        batch_size=int(batch_size),
        fixed_point_bits=int(fixed_point_bits),
        throughput_ops_per_ms=int(throughput),
        merge_entries=tuple(merge_entries),
        tie_break=tie_break,
        receipt_sources=tuple(receipt_sources),
        interaction_budget_ms=int(interaction_budget_ms),
        fault_config=fault_config,
        source_path=issue_path,
        raw=raw,
    )


def load_requests(path: str | Path, *, batch_size: int | None = None) -> tuple[RequestSpec, ...]:
    request_path = Path(path)
    issue_path = str(request_path)
    issues = IssueCollector()
    requests: list[RequestSpec] = []
    previous_seq: int | None = None
    seen_ids: set[str] = set()
    try:
        handle = request_path.open("rb")
    except OSError as exc:
        issues.add(_issue(E_REQUEST_INVALID, f"cannot read request stream: {exc}", path=issue_path))
        raise ConfigValidation(issues) from exc

    with handle:
        for line_number, raw_line in enumerate(handle, start=1):
            try:
                decoded = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                issues.add(
                    _issue(
                        E_REQUEST_INVALID,
                        "request line is not valid UTF-8",
                        path=issue_path,
                        item=f"line {line_number}",
                    )
                )
                continue
            text = decoded.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                issues.add(
                    _issue(
                        E_REQUEST_INVALID,
                        f"invalid JSON: {exc.msg}",
                        path=issue_path,
                        item=f"line {line_number}",
                    )
                )
                continue
            if not isinstance(item, dict):
                issues.add(
                    _issue(
                        E_REQUEST_INVALID,
                        "request must be an object",
                        path=issue_path,
                        item=f"line {line_number}",
                    )
                )
                continue
            seq = item.get("seq")
            request_id = item.get("request_id")
            arrival_ms = item.get("arrival_ms")
            prompt = item.get("prompt", "")
            payload = item.get("payload", [])
            valid = True
            if not is_int(seq) or seq < 0:
                issues.add(
                    _issue(
                        E_REQUEST_INVALID,
                        "seq must be a non-negative integer",
                        path=issue_path,
                        item=f"line {line_number}",
                    )
                )
                valid = False
            elif previous_seq is not None and seq <= previous_seq:
                issues.add(
                    _issue(
                        E_REQUEST_INVALID,
                        f"seq must increase strictly; got {seq} after {previous_seq}",
                        path=issue_path,
                        item=f"line {line_number}",
                    )
                )
                valid = False
            if not isinstance(request_id, str) or not request_id or len(request_id) > 64:
                issues.add(
                    _issue(
                        E_REQUEST_INVALID,
                        "request_id must be a non-empty string of width <= 64",
                        path=issue_path,
                        item=f"line {line_number}",
                    )
                )
                valid = False
            elif any(character.isspace() or character in "/\\\x00" for character in request_id):
                issues.add(
                    _issue(
                        E_REQUEST_INVALID,
                        "request_id contains a forbidden character",
                        path=issue_path,
                        item=f"line {line_number}",
                    )
                )
                valid = False
            elif request_id in seen_ids:
                issues.add(
                    _issue(
                        E_REQUEST_INVALID,
                        f"duplicate request_id {request_id!r}",
                        path=issue_path,
                        item=f"line {line_number}",
                    )
                )
                valid = False
            if not is_int(arrival_ms) or arrival_ms < 0:
                issues.add(
                    _issue(
                        E_REQUEST_INVALID,
                        "arrival_ms must be a non-negative integer",
                        path=issue_path,
                        item=f"line {line_number}",
                    )
                )
                valid = False
            if not isinstance(prompt, str):
                issues.add(
                    _issue(
                        E_REQUEST_INVALID,
                        "prompt must be a string",
                        path=issue_path,
                        item=f"line {line_number}",
                    )
                )
                valid = False
            if not isinstance(payload, list) or not all(is_int(value) for value in payload):
                issues.add(
                    _issue(
                        E_REQUEST_INVALID,
                        "payload must be a list of integers",
                        path=issue_path,
                        item=f"line {line_number}",
                    )
                )
                valid = False
            if valid:
                assert is_int(seq) and is_int(arrival_ms)
                previous_seq = int(seq)
                seen_ids.add(str(request_id))
                requests.append(
                    RequestSpec(
                        seq=int(seq),
                        request_id=str(request_id),
                        arrival_ms=int(arrival_ms),
                        prompt=str(prompt),
                        payload=tuple(int(value) for value in payload),
                        source_path=issue_path,
                        source_line=line_number,
                    )
                )

    if batch_size is not None:
        if not is_int(batch_size) or batch_size < 1:
            issues.add(
                _issue(
                    E_CONFIG_INVALID,
                    "batch_size must be a positive integer",
                    path=issue_path,
                    item="batch_size",
                )
            )
        elif batch_size > len(requests):
            issues.add(
                _issue(
                    E_CONFIG_INVALID,
                    f"batch_size {batch_size} exceeds request count {len(requests)}",
                    path=issue_path,
                    item="batch_size",
                )
            )
    if issues:
        raise ConfigValidation(issues)
    return tuple(requests)


def config_hash(config: AppConfig) -> str:
    return digest_value(config.to_dict())


def requests_hash(requests: tuple[RequestSpec, ...]) -> str:
    return digest_value([request.to_dict() for request in requests])


def file_fingerprint(path: Path) -> str:
    return sha256_file(path) if path.exists() else "missing"


def validate_project(config_path: str | Path, requests_path: str | Path) -> tuple[AppConfig, tuple[RequestSpec, ...]]:
    config = load_config(config_path)
    requests = load_requests(requests_path, batch_size=config.batch_size)
    return config, requests
