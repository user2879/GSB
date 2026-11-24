"""Loading and strict validation of the three local JSON inputs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .canonical import canonical_json, digest_json
from .errors import DiagnosticError, fail


_RECEIPT_FORMATS = ("r1", "r2", "r3")


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    arrival_seq: int
    input_vector: tuple[float, ...]
    duplicate_refs: tuple[str, ...] = ()
    source_index: int = 0

    def dependency_payload(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "arrival_seq": self.arrival_seq,
            "input": list(self.input_vector),
            "duplicate_refs": list(self.duplicate_refs),
        }

    def signature(self) -> str:
        return digest_json(self.dependency_payload())


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    host: str
    port: int
    model_dim: int
    hidden_dim: int
    seed: int
    param_version: int
    batch_size: int
    concurrency: int = 1
    source_index: int = 0

    def computation_signature(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "model_dim": self.model_dim,
            "hidden_dim": self.hidden_dim,
            "seed": self.seed,
            "param_version": self.param_version,
            "batch_size": self.batch_size,
        }

    def digest(self) -> str:
        return digest_json(
            {
                "computation": self.computation_signature(),
                "host": self.host,
                "port": self.port,
                "concurrency": self.concurrency,
            }
        )


@dataclass(frozen=True)
class MergeRules:
    global_concurrency: int
    socket_timeout_ms: int
    worker_start_timeout_ms: int
    receipt_format: str
    assignments: tuple[tuple[str, str], ...]

    @property
    def assignment_map(self) -> dict[str, str]:
        return dict(self.assignments)


@dataclass(frozen=True)
class SampleSpec:
    requests: tuple[RequestSpec, ...]
    workers: tuple[WorkerSpec, ...]
    rules: MergeRules
    request_file: str = "request_stream.json"
    worker_file: str = "worker_spec.json"
    rules_file: str = "merge_rules.json"

    @property
    def request_by_id(self) -> dict[str, RequestSpec]:
        return {item.request_id: item for item in self.requests}

    @property
    def worker_by_id(self) -> dict[str, WorkerSpec]:
        return {item.worker_id: item for item in self.workers}

    @property
    def requests_by_arrival(self) -> tuple[RequestSpec, ...]:
        return tuple(sorted(self.requests, key=lambda item: item.arrival_seq))

    @property
    def workers_by_id(self) -> tuple[WorkerSpec, ...]:
        return tuple(sorted(self.workers, key=lambda item: item.worker_id))

    def assigned_worker(self, request_id: str) -> WorkerSpec:
        worker_id = self.rules.assignment_map[request_id]
        return self.worker_by_id[worker_id]

    def fingerprint(self) -> str:
        return digest_json(
            {
                "requests": [item.dependency_payload() for item in self.requests_by_arrival],
                "workers": [
                    {
                        **item.computation_signature(),
                        "host": item.host,
                        "port": item.port,
                        "concurrency": item.concurrency,
                    }
                    for item in self.workers_by_id
                ],
                "rules": {
                    "global_concurrency": self.rules.global_concurrency,
                    "socket_timeout_ms": self.rules.socket_timeout_ms,
                    "worker_start_timeout_ms": self.rules.worker_start_timeout_ms,
                    "receipt_format": self.rules.receipt_format,
                    "assignments": list(self.rules.assignments),
                },
            }
        )


def _load_json(path: Path, label: str) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        fail("E_IO", label, f"file not found: {path}")
    except json.JSONDecodeError as exc:
        fail(
            "E_SPEC",
            f"{label}:{exc.lineno}:{exc.colno}",
            f"invalid JSON: {exc.msg}",
        )
    except OSError as exc:
        fail("E_IO", label, f"cannot read {path}: {exc.strerror}")


def _object(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        fail("E_SPEC", path, "expected a JSON object")
    return value


def _sequence(value: Any, path: str) -> Sequence[Any]:
    if not isinstance(value, list):
        fail("E_SPEC", path, "expected a JSON array")
    return value


def _string(value: Any, path: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        fail("E_SPEC", path, "expected a non-empty string")
    return value


def _int(
    value: Any,
    path: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        fail("E_SPEC", path, "expected an integer (booleans are not integers)")
    if minimum is not None and value < minimum:
        fail("E_SPEC", path, f"must be >= {minimum}")
    if maximum is not None and value > maximum:
        fail("E_SPEC", path, f"must be <= {maximum}")
    return value


def _number(value: Any, path: str) -> float:
    if type(value) not in (int, float):
        fail("E_SPEC", path, "expected a finite number")
    number = float(value)
    if not math.isfinite(number):
        fail("E_SPEC", path, "expected a finite number")
    return number


def _optional_format_version(document: Mapping[str, Any], path: str) -> None:
    if "format_version" in document:
        version = document["format_version"]
        if type(version) is not int or version != 1:
            fail("E_SPEC", f"{path}.format_version", "supported value is integer 1")


def _parse_requests(document: Any) -> tuple[RequestSpec, ...]:
    root = _object(document, "request_stream")
    _optional_format_version(root, "request_stream")
    raw_requests = _sequence(root.get("requests"), "request_stream.requests")
    parsed: list[RequestSpec] = []
    request_ids: set[str] = set()
    arrivals: set[int] = set()
    for index, raw in enumerate(raw_requests):
        base = f"request_stream.requests[{index}]"
        item = _object(raw, base)
        request_id = _string(item.get("request_id"), f"{base}.request_id")
        if request_id in request_ids:
            fail("E_SPEC", f"{base}.request_id", f"duplicate request_id: {request_id}")
        arrival_seq = _int(item.get("arrival_seq"), f"{base}.arrival_seq", minimum=0)
        if arrival_seq in arrivals:
            fail(
                "E_SPEC",
                f"{base}.arrival_seq",
                f"duplicate arrival_seq: {arrival_seq}",
            )
        raw_input = _sequence(item.get("input"), f"{base}.input")
        input_vector = tuple(
            _number(value, f"{base}.input[{position}]")
            for position, value in enumerate(raw_input)
        )
        raw_duplicates = item.get("duplicate_refs", [])
        duplicate_values = _sequence(raw_duplicates, f"{base}.duplicate_refs")
        duplicate_refs: list[str] = []
        duplicate_seen: set[str] = set()
        for duplicate_index, value in enumerate(duplicate_values):
            reference = _string(
                value, f"{base}.duplicate_refs[{duplicate_index}]"
            )
            if reference in duplicate_seen:
                fail(
                    "E_SPEC",
                    f"{base}.duplicate_refs[{duplicate_index}]",
                    f"duplicate reference: {reference}",
                )
            if reference == request_id:
                fail(
                    "E_SPEC",
                    f"{base}.duplicate_refs[{duplicate_index}]",
                    "a request cannot reference itself",
                )
            duplicate_seen.add(reference)
            duplicate_refs.append(reference)
        request_ids.add(request_id)
        arrivals.add(arrival_seq)
        parsed.append(
            RequestSpec(
                request_id=request_id,
                arrival_seq=arrival_seq,
                input_vector=input_vector,
                duplicate_refs=tuple(sorted(duplicate_refs)),
                source_index=index,
            )
        )
    return tuple(parsed)


def _parse_workers(document: Any) -> tuple[WorkerSpec, ...]:
    root = _object(document, "worker_spec")
    _optional_format_version(root, "worker_spec")
    raw_workers = _sequence(root.get("workers"), "worker_spec.workers")
    parsed: list[WorkerSpec] = []
    worker_ids: set[str] = set()
    ports: set[int] = set()
    for index, raw in enumerate(raw_workers):
        base = f"worker_spec.workers[{index}]"
        item = _object(raw, base)
        worker_id = _string(item.get("worker_id"), f"{base}.worker_id")
        if worker_id in worker_ids:
            fail("E_SPEC", f"{base}.worker_id", f"duplicate worker_id: {worker_id}")
        host = item.get("host")
        if type(host) is not str or host != "127.0.0.1":
            fail("E_SPEC", f"{base}.host", "expected the literal string 127.0.0.1")
        port = _int(item.get("port"), f"{base}.port", minimum=1, maximum=65535)
        if port in ports:
            fail("E_SPEC", f"{base}.port", f"duplicate worker port: {port}")
        model_dim = _int(item.get("model_dim"), f"{base}.model_dim", minimum=1)
        hidden_dim = _int(item.get("hidden_dim"), f"{base}.hidden_dim", minimum=1)
        seed = _int(item.get("seed"), f"{base}.seed")
        param_version = _int(
            item.get("param_version"), f"{base}.param_version", minimum=0
        )
        batch_size = _int(item.get("batch_size"), f"{base}.batch_size", minimum=1)
        concurrency = _int(
            item.get("concurrency", 1), f"{base}.concurrency", minimum=1
        )
        worker_ids.add(worker_id)
        ports.add(port)
        parsed.append(
            WorkerSpec(
                worker_id=worker_id,
                host=host,
                port=port,
                model_dim=model_dim,
                hidden_dim=hidden_dim,
                seed=seed,
                param_version=param_version,
                batch_size=batch_size,
                concurrency=concurrency,
                source_index=index,
            )
        )
    return tuple(parsed)


def _parse_rules(document: Any, workers: tuple[WorkerSpec, ...]) -> MergeRules:
    root = _object(document, "merge_rules")
    _optional_format_version(root, "merge_rules")
    global_concurrency = _int(
        root.get("global_concurrency"),
        "merge_rules.global_concurrency",
        minimum=1,
    )
    socket_timeout_ms = _int(
        root.get("socket_timeout_ms", 5000),
        "merge_rules.socket_timeout_ms",
        minimum=1,
    )
    worker_start_timeout_ms = _int(
        root.get("worker_start_timeout_ms", 10000),
        "merge_rules.worker_start_timeout_ms",
        minimum=1,
    )
    receipt_format = _string(
        root.get("receipt_format", "r3"), "merge_rules.receipt_format"
    )
    if receipt_format not in _RECEIPT_FORMATS:
        fail(
            "E_SPEC",
            "merge_rules.receipt_format",
            f"expected one of {', '.join(_RECEIPT_FORMATS)}",
        )
    raw_routes = _sequence(root.get("routes"), "merge_rules.routes")
    worker_ids = {item.worker_id for item in workers}
    assignment_map: dict[str, str] = {}
    parsed_routes: list[tuple[str, str]] = []
    for index, raw in enumerate(raw_routes):
        base = f"merge_rules.routes[{index}]"
        item = _object(raw, base)
        worker_id = _string(item.get("worker_id"), f"{base}.worker_id")
        if worker_id not in worker_ids:
            fail("E_SPEC", f"{base}.worker_id", f"unknown worker_id: {worker_id}")
        raw_ids = _sequence(item.get("request_ids"), f"{base}.request_ids")
        for request_index, value in enumerate(raw_ids):
            request_id = _string(value, f"{base}.request_ids[{request_index}]")
            if request_id in assignment_map:
                fail(
                    "E_SPEC",
                    f"{base}.request_ids[{request_index}]",
                    f"request assigned more than once: {request_id}",
                )
            assignment_map[request_id] = worker_id
            parsed_routes.append((request_id, worker_id))
    return MergeRules(
        global_concurrency=global_concurrency,
        socket_timeout_ms=socket_timeout_ms,
        worker_start_timeout_ms=worker_start_timeout_ms,
        receipt_format=receipt_format,
        assignments=tuple(sorted(parsed_routes)),
    )


def load_specs(directory: str | Path) -> SampleSpec:
    base = Path(directory)
    request_doc = _load_json(base / "request_stream.json", "request_stream.json")
    workers_doc = _load_json(base / "worker_spec.json", "worker_spec.json")
    rules_doc = _load_json(base / "merge_rules.json", "merge_rules.json")

    requests = _parse_requests(request_doc)
    workers = _parse_workers(workers_doc)
    rules = _parse_rules(rules_doc, workers)

    request_by_id = {item.request_id: item for item in requests}
    worker_by_id = {item.worker_id: item for item in workers}

    for request in sorted(requests, key=lambda item: item.source_index):
        for index, reference in enumerate(request.duplicate_refs):
            if reference not in request_by_id:
                fail(
                    "E_SPEC",
                    f"request_stream.requests[{request.source_index}].duplicate_refs[{index}]",
                    f"unknown request reference: {reference}",
                )

    assignment_map = rules.assignment_map
    for request in sorted(requests, key=lambda item: item.source_index):
        if request.request_id not in assignment_map:
            fail(
                "E_SPEC",
                f"request_stream.requests[{request.source_index}].request_id",
                f"request has no worker assignment: {request.request_id}",
            )
        worker = worker_by_id[assignment_map[request.request_id]]
        if len(request.input_vector) != worker.model_dim:
            fail(
                "E_SPEC",
                f"request_stream.requests[{request.source_index}].input",
                (
                    f"dimension {len(request.input_vector)} does not match "
                    f"worker {worker.worker_id} model_dim {worker.model_dim}"
                ),
            )
    for request_id in sorted(assignment_map):
        if request_id not in request_by_id:
            fail(
                "E_SPEC",
                "merge_rules.routes",
                f"assignment references unknown request_id: {request_id}",
            )

    # Assignment order is canonicalized by request arrival, never by JSON key order.
    canonical_assignments = tuple(
        sorted(
            (
                (request_id, assignment_map[request_id])
                for request_id in request_by_id
            ),
            key=lambda pair: (
                request_by_id[pair[0]].arrival_seq,
                pair[0],
            ),
        )
    )
    rules = MergeRules(
        global_concurrency=rules.global_concurrency,
        socket_timeout_ms=rules.socket_timeout_ms,
        worker_start_timeout_ms=rules.worker_start_timeout_ms,
        receipt_format=rules.receipt_format,
        assignments=canonical_assignments,
    )
    return SampleSpec(
        requests=tuple(sorted(requests, key=lambda item: item.arrival_seq)),
        workers=tuple(sorted(workers, key=lambda item: item.worker_id)),
        rules=rules,
    )


def spec_as_json(spec: SampleSpec) -> str:
    return canonical_json(
        {
            "requests": [item.dependency_payload() for item in spec.requests],
            "workers": [
                {
                    **item.computation_signature(),
                    "host": item.host,
                    "port": item.port,
                    "concurrency": item.concurrency,
                }
                for item in spec.workers
            ],
            "rules": {
                "global_concurrency": spec.rules.global_concurrency,
                "socket_timeout_ms": spec.rules.socket_timeout_ms,
                "worker_start_timeout_ms": spec.rules.worker_start_timeout_ms,
                "receipt_format": spec.rules.receipt_format,
                "assignments": [list(item) for item in spec.rules.assignments],
            },
        }
    )
