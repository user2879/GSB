"""Strict local declarations for requests, workers, and merge rules."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .errors import DiagnosticError, is_int
from .jsonio import load_json_file


WORKER_KEYS = {
    "worker_id",
    "host",
    "port",
    "seed",
    "input_dim",
    "hidden_dim",
    "param_version_seq",
    "batch_size",
    "concurrency_quota",
}
REQUEST_KEYS = {"request_id", "arrival_seq", "worker_id", "input"}
FORMAT_VERSIONS = ("r1", "r2", "r3")
FILLABLE_FIELDS = {
    "r1": {
        "worker_id",
        "param_version_seq",
        "latency_ms",
        "param_digest",
        "rank_in_batch",
        "superseded_request_ids",
    },
    "r2": {"param_digest", "rank_in_batch", "superseded_request_ids"},
    "r3": set(),
}
ALLOWED_FILL_STRATEGIES = {
    "worker_id": {"batch.worker_id"},
    "param_version_seq": {"batch.param_version_seq"},
    "latency_ms": {"constant:0"},
    "param_digest": {"batch.param_digest"},
    "rank_in_batch": {"request.arrival_rank"},
    "superseded_request_ids": {"empty_list"},
}


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    host: str
    port: int
    seed: int
    input_dim: int
    hidden_dim: int
    param_version_seq: int
    batch_size: int
    concurrency_quota: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "host": self.host,
            "port": self.port,
            "seed": self.seed,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "param_version_seq": self.param_version_seq,
            "batch_size": self.batch_size,
            "concurrency_quota": self.concurrency_quota,
        }

    def parameter_signature(self) -> tuple[int, int, int, int]:
        return (
            self.seed,
            self.input_dim,
            self.hidden_dim,
            self.param_version_seq,
        )


@dataclass(frozen=True)
class WorkerSet:
    schema_version: int
    worker_set_version: int
    workers: tuple[WorkerSpec, ...]

    def by_id(self) -> dict[str, WorkerSpec]:
        return {worker.worker_id: worker for worker in self.workers}

    def param_versions(self) -> dict[str, int]:
        return {
            worker.worker_id: worker.param_version_seq
            for worker in sorted(self.workers, key=lambda item: item.worker_id)
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "worker_set_version": self.worker_set_version,
            "workers": [worker.to_dict() for worker in self.workers],
        }


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    arrival_seq: int
    worker_id: str
    input: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "arrival_seq": self.arrival_seq,
            "worker_id": self.worker_id,
            "input": list(self.input),
        }


@dataclass(frozen=True)
class RequestSet:
    schema_version: int
    requests: tuple[RequestSpec, ...]

    def by_id(self) -> dict[str, RequestSpec]:
        return {request.request_id: request for request in self.requests}

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "requests": [request.to_dict() for request in self.requests],
        }


@dataclass(frozen=True)
class MergeRules:
    schema_version: int
    global_concurrency: int
    formats: dict[str, dict[str, str]]

    def fill_for(self, format_version: str) -> dict[str, str]:
        return dict(self.formats[format_version])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "global_concurrency": self.global_concurrency,
            "formats": {
                version: {"fill": dict(self.formats[version])}
                for version in FORMAT_VERSIONS
            },
        }


def _object(value: Any, object_name: str, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DiagnosticError(
            code="E_SPEC",
            message="expected a JSON object",
            object=object_name,
            field=field,
        )
    return value


def _list(value: Any, object_name: str, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise DiagnosticError(
            code="E_SPEC",
            message="expected a JSON array",
            object=object_name,
            field=field,
        )
    return value


def _required(
    source: dict[str, Any], key: str, object_name: str, allowed: set[str] | None = None
) -> Any:
    if key not in source:
        raise DiagnosticError(
            code="E_SPEC",
            message=f"missing required field {key!r}",
            object=object_name,
            field=key,
        )
    if allowed is not None and key not in allowed:
        raise DiagnosticError(
            code="E_SPEC",
            message=f"unexpected field {key!r}",
            object=object_name,
            field=key,
        )
    return source[key]


def _check_keys(source: dict[str, Any], allowed: set[str], object_name: str) -> None:
    for key in source:
        if key not in allowed:
            raise DiagnosticError(
                code="E_SPEC",
                message=f"unexpected field {key!r}",
                object=object_name,
                field=key,
            )


def _string(value: Any, object_name: str, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise DiagnosticError(
            code="E_SPEC",
            message="expected a non-empty string",
            object=object_name,
            field=field,
        )
    return value


def _integer(
    value: Any,
    object_name: str,
    field: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if not is_int(value):
        raise DiagnosticError(
            code="E_SPEC",
            message="expected an integer",
            object=object_name,
            field=field,
        )
    if minimum is not None and value < minimum:
        raise DiagnosticError(
            code="E_SPEC",
            message=f"integer must be >= {minimum}",
            object=object_name,
            field=field,
        )
    if maximum is not None and value > maximum:
        raise DiagnosticError(
            code="E_SPEC",
            message=f"integer must be <= {maximum}",
            object=object_name,
            field=field,
        )
    return value


def _schema_version(document: dict[str, Any], object_name: str) -> int:
    value = _required(document, "schema_version", object_name)
    return _integer(value, object_name, "schema_version", minimum=1)


def parse_workers(document: Any, *, object_name: str = "workers.json") -> WorkerSet:
    root = _object(document, object_name, "<root>")
    _check_keys(root, {"schema_version", "worker_set_version", "workers"}, object_name)
    schema_version = _schema_version(root, object_name)
    worker_set_version = _integer(
        _required(root, "worker_set_version", object_name),
        object_name,
        "worker_set_version",
        minimum=1,
    )
    raw_workers = _list(_required(root, "workers", object_name), object_name, "workers")
    if not raw_workers:
        raise DiagnosticError(
            code="E_SPEC",
            message="at least one worker is required",
            object=object_name,
            field="workers",
        )

    workers: list[WorkerSpec] = []
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    for index, raw in enumerate(raw_workers):
        item_name = f"{object_name}:workers[{index}]"
        item = _object(raw, item_name, "<object>")
        _check_keys(item, WORKER_KEYS, item_name)
        worker_id = _string(
            _required(item, "worker_id", item_name), item_name, "worker_id"
        )
        if worker_id in seen_ids:
            raise DiagnosticError(
                code="E_SPEC",
                message=f"duplicate worker_id {worker_id!r}",
                object=item_name,
                field="worker_id",
            )
        seen_ids.add(worker_id)
        host = _string(_required(item, "host", item_name), item_name, "host")
        if host != "127.0.0.1":
            raise DiagnosticError(
                code="E_SPEC",
                message="host must be the literal loopback address 127.0.0.1",
                object=item_name,
                field="host",
            )
        port = _integer(
            _required(item, "port", item_name),
            item_name,
            "port",
            minimum=1024,
            maximum=65535,
        )
        if port in seen_ports:
            raise DiagnosticError(
                code="E_SPEC",
                message=f"duplicate worker port {port}",
                object=item_name,
                field="port",
            )
        seen_ports.add(port)
        seed = _integer(
            _required(item, "seed", item_name),
            item_name,
            "seed",
            minimum=0,
            maximum=2**63 - 1,
        )
        input_dim = _integer(
            _required(item, "input_dim", item_name),
            item_name,
            "input_dim",
            minimum=1,
        )
        hidden_dim = _integer(
            _required(item, "hidden_dim", item_name),
            item_name,
            "hidden_dim",
            minimum=1,
        )
        param_version_seq = _integer(
            _required(item, "param_version_seq", item_name),
            item_name,
            "param_version_seq",
            minimum=1,
        )
        batch_size = _integer(
            _required(item, "batch_size", item_name),
            item_name,
            "batch_size",
            minimum=1,
        )
        concurrency_quota = _integer(
            _required(item, "concurrency_quota", item_name),
            item_name,
            "concurrency_quota",
            minimum=1,
        )
        workers.append(
            WorkerSpec(
                worker_id=worker_id,
                host=host,
                port=port,
                seed=seed,
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                param_version_seq=param_version_seq,
                batch_size=batch_size,
                concurrency_quota=concurrency_quota,
            )
        )
    return WorkerSet(
        schema_version=schema_version,
        worker_set_version=worker_set_version,
        workers=tuple(workers),
    )


def load_workers(path: str | Path) -> WorkerSet:
    return parse_workers(
        load_json_file(path, object_name=str(path)), object_name=str(path)
    )


def _worker_map(workers: WorkerSet) -> dict[str, WorkerSpec]:
    return workers.by_id()


def parse_requests(
    document: Any,
    workers: WorkerSet,
    *,
    object_name: str = "requests.json",
) -> RequestSet:
    root = _object(document, object_name, "<root>")
    _check_keys(root, {"schema_version", "requests"}, object_name)
    schema_version = _schema_version(root, object_name)
    raw_requests = _list(
        _required(root, "requests", object_name), object_name, "requests"
    )
    if not raw_requests:
        raise DiagnosticError(
            code="E_SPEC",
            message="at least one request is required",
            object=object_name,
            field="requests",
        )
    worker_map = _worker_map(workers)
    requests: list[RequestSpec] = []
    seen_ids: set[str] = set()
    seen_arrivals: set[int] = set()
    for index, raw in enumerate(raw_requests):
        item_name = f"{object_name}:requests[{index}]"
        item = _object(raw, item_name, "<object>")
        _check_keys(item, REQUEST_KEYS, item_name)
        request_id = _string(
            _required(item, "request_id", item_name), item_name, "request_id"
        )
        if request_id in seen_ids:
            raise DiagnosticError(
                code="E_SPEC",
                message=f"duplicate request_id {request_id!r}",
                object=item_name,
                field="request_id",
            )
        seen_ids.add(request_id)
        arrival_seq = _integer(
            _required(item, "arrival_seq", item_name),
            item_name,
            "arrival_seq",
            minimum=0,
        )
        if arrival_seq in seen_arrivals:
            raise DiagnosticError(
                code="E_SPEC",
                message=f"duplicate arrival_seq {arrival_seq}",
                object=item_name,
                field="arrival_seq",
            )
        seen_arrivals.add(arrival_seq)
        worker_id = _string(
            _required(item, "worker_id", item_name), item_name, "worker_id"
        )
        worker = worker_map.get(worker_id)
        if worker is None:
            raise DiagnosticError(
                code="E_SPEC",
                message=f"unknown worker_id {worker_id!r}",
                object=item_name,
                field="worker_id",
            )
        raw_input = _list(
            _required(item, "input", item_name), item_name, "input"
        )
        if len(raw_input) != worker.input_dim:
            raise DiagnosticError(
                code="E_SPEC",
                message=(
                    f"input dimension {len(raw_input)} does not match worker "
                    f"input_dim {worker.input_dim}"
                ),
                object=item_name,
                field="input",
            )
        vector: list[float] = []
        for value_index, value in enumerate(raw_input):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise DiagnosticError(
                    code="E_SPEC",
                    message="input values must be finite JSON numbers",
                    object=item_name,
                    field=f"input[{value_index}]",
                )
            vector.append(float(value))
        requests.append(
            RequestSpec(
                request_id=request_id,
                arrival_seq=arrival_seq,
                worker_id=worker_id,
                input=tuple(vector),
            )
        )
    return RequestSet(schema_version=schema_version, requests=tuple(requests))


def load_requests(path: str | Path, workers: WorkerSet) -> RequestSet:
    return parse_requests(
        load_json_file(path, object_name=str(path)),
        workers,
        object_name=str(path),
    )


def parse_merge_rules(document: Any, *, object_name: str = "merge_rules.json") -> MergeRules:
    root = _object(document, object_name, "<root>")
    _check_keys(root, {"schema_version", "global_concurrency", "formats"}, object_name)
    schema_version = _schema_version(root, object_name)
    global_concurrency = _integer(
        _required(root, "global_concurrency", object_name),
        object_name,
        "global_concurrency",
        minimum=1,
    )
    raw_formats = _object(
        _required(root, "formats", object_name), object_name, "formats"
    )
    for version in raw_formats:
        if version not in FORMAT_VERSIONS:
            raise DiagnosticError(
                code="E_SPEC",
                message=f"unsupported receipt format {version!r}",
                object=object_name,
                field=f"formats.{version}",
            )
    parsed_formats: dict[str, dict[str, str]] = {}
    for version in FORMAT_VERSIONS:
        format_name = f"{object_name}:formats.{version}"
        if version not in raw_formats:
            raise DiagnosticError(
                code="E_SPEC",
                message=f"missing format rules for {version!r}",
                object=format_name,
                field="<object>",
            )
        raw_format = _object(raw_formats[version], format_name, "<object>")
        _check_keys(raw_format, {"fill"}, format_name)
        raw_fill = _object(
            _required(raw_format, "fill", format_name), format_name, "fill"
        )
        for field, strategy in raw_fill.items():
            if field not in FILLABLE_FIELDS[version]:
                raise DiagnosticError(
                    code="E_SPEC",
                    message=f"field {field!r} cannot be filled for {version}",
                    object=format_name,
                    field=f"fill.{field}",
                )
            if not isinstance(strategy, str) or strategy not in ALLOWED_FILL_STRATEGIES[field]:
                raise DiagnosticError(
                    code="E_SPEC",
                    message=f"unsupported fill strategy {strategy!r}",
                    object=format_name,
                    field=f"fill.{field}",
                )
        missing = FILLABLE_FIELDS[version] - set(raw_fill)
        if missing:
            first = sorted(missing)[0]
            raise DiagnosticError(
                code="E_SPEC",
                message=f"missing fill strategy for field {first!r}",
                object=format_name,
                field=f"fill.{first}",
            )
        parsed_formats[version] = {
            field: str(strategy) for field, strategy in raw_fill.items()
        }
    return MergeRules(
        schema_version=schema_version,
        global_concurrency=global_concurrency,
        formats=parsed_formats,
    )


def load_merge_rules(path: str | Path) -> MergeRules:
    return parse_merge_rules(
        load_json_file(path, object_name=str(path)), object_name=str(path)
    )


def first_duplicate(values: Iterable[Any]) -> Any | None:
    seen: set[Any] = set()
    for value in values:
        if value in seen:
            return value
        seen.add(value)
    return None
