"""Local JSON specifications: request stream, worker declarations, budget.

Loaders perform strict validation and raise :class:`SpecError` (``E_SPEC``)
pointing at the first illegal entry and its field. Deterministic fingerprints
derived here drive snapshot version comparison and invalidation propagation.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from .canonical import sha256_hex
from .errors import SpecError

SEED_MAX = 2**32 - 1
LOOPBACK_HOST = "127.0.0.1"
PORT_MIN, PORT_MAX = 1, 65535


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_int(
    value: Any,
    *,
    name: str,
    entry: str,
    minimum: int,
    maximum: int | None = None,
) -> int:
    if not _is_plain_int(value):
        raise SpecError(
            f"field {name!r} of {entry} must be an integer",
            entry=entry,
            field=name,
            value=repr(value),
        )
    if value < minimum or (maximum is not None and value > maximum):
        upper = "" if maximum is None else f"..{maximum}"
        raise SpecError(
            f"field {name!r} of {entry} out of range [{minimum}{upper}]",
            entry=entry,
            field=name,
            value=value,
        )
    return value


def _check_required(mapping: Any, key: str, entry: str) -> Any:
    if not isinstance(mapping, dict) or key not in mapping:
        raise SpecError(
            f"required field {key!r} missing on {entry}",
            entry=entry,
            field=key,
        )
    return mapping[key]


@dataclass(frozen=True)
class RequestSpec:
    request_id: str
    arrival_seq: int
    input_dim: int
    seed: int


@dataclass(frozen=True)
class WorkerSpec:
    worker_id: str
    host: str
    port: int
    seed: int
    param_version: int
    input_dim: int
    hidden_dim: int
    batch_size: int
    max_in_flight: int
    retries: int

    def parameter_fingerprint(self) -> str:
        """Fingerprint of the parameter-defining fields (version -> batch)."""

        return sha256_hex(
            [
                self.worker_id,
                self.seed,
                self.param_version,
                self.input_dim,
                self.hidden_dim,
            ]
        )


@dataclass(frozen=True)
class BudgetSpec:
    global_max_in_flight: int
    retry_limit: int
    cost_budget_ms: int


@dataclass(frozen=True)
class SystemSpec:
    requests: tuple[RequestSpec, ...]
    workers: tuple[WorkerSpec, ...]
    budget: BudgetSpec
    source_paths: dict[str, str] = field(default_factory=dict)

    def request_by_id(self) -> dict[str, RequestSpec]:
        return {request.request_id: request for request in self.requests}

    def worker_by_id(self) -> dict[str, WorkerSpec]:
        return {worker.worker_id: worker for worker in self.workers}

    def input_dim(self) -> int:
        return self.requests[0].input_dim

    def worker_set_fingerprint(self) -> str:
        """Version of the worker parameter set (host/port + parameter fields)."""

        payload = [
            [
                worker.worker_id,
                worker.host,
                worker.port,
                worker.seed,
                worker.param_version,
                worker.input_dim,
                worker.hidden_dim,
            ]
            for worker in sorted(self.workers, key=lambda worker: worker.worker_id)
        ]
        return sha256_hex(payload)

    def batch_size_fingerprint(self) -> str:
        payload = [
            [worker.worker_id, worker.batch_size]
            for worker in sorted(self.workers, key=lambda worker: worker.worker_id)
        ]
        return sha256_hex(payload)

    def content_fingerprint(self) -> str:
        """Overall digest covering requests, workers and budget."""

        return sha256_hex(
            [
                [
                    r.request_id,
                    r.arrival_seq,
                    r.input_dim,
                    r.seed,
                ]
                for r in sorted(self.requests, key=lambda item: item.arrival_seq)
            ]
            + [
                [
                    w.worker_id,
                    w.host,
                    w.port,
                    w.seed,
                    w.param_version,
                    w.input_dim,
                    w.hidden_dim,
                    w.batch_size,
                    w.max_in_flight,
                    w.retries,
                ]
                for w in sorted(self.workers, key=lambda item: item.worker_id)
            ]
            + [
                self.budget.global_max_in_flight,
                self.budget.retry_limit,
                self.budget.cost_budget_ms,
            ]
        )


def load_json_file(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        try:
            return json.load(handle)
        except json.JSONDecodeError as exc:
            raise SpecError(
                f"file is not valid JSON: {exc.msg}",
                entry=os.fspath(path),
                field="<document>",
            ) from exc


def _validate_requests(raw: Any) -> tuple[RequestSpec, ...]:
    entry = "requests"
    if not isinstance(raw, list) or not raw:
        raise SpecError("requests must be a non-empty list", entry=entry, field="<document>")
    seen_ids: set[str] = set()
    seen_seqs: set[int] = set()
    requests: list[RequestSpec] = []
    for index, item in enumerate(raw):
        entry_ref = f"requests[{index}]"
        if not isinstance(item, dict):
            raise SpecError("request entry must be an object", entry=entry_ref, field="<entry>")
        request_id = _check_required(item, "request_id", entry_ref)
        if not isinstance(request_id, str) or not request_id:
            raise SpecError(
                "request_id must be a non-empty string",
                entry=entry_ref,
                field="request_id",
                value=repr(request_id),
            )
        if request_id in seen_ids:
            raise SpecError(
                "duplicate request_id",
                entry=entry_ref,
                field="request_id",
                value=request_id,
            )
        seen_ids.add(request_id)
        arrival_seq = _check_int(
            _check_required(item, "arrival_seq", entry_ref),
            name="arrival_seq",
            entry=entry_ref,
            minimum=0,
        )
        if arrival_seq in seen_seqs:
            raise SpecError(
                "duplicate arrival_seq",
                entry=entry_ref,
                field="arrival_seq",
                value=arrival_seq,
            )
        seen_seqs.add(arrival_seq)
        input_dim = _check_int(
            _check_required(item, "input_dim", entry_ref),
            name="input_dim",
            entry=entry_ref,
            minimum=1,
        )
        seed = _check_int(
            _check_required(item, "seed", entry_ref),
            name="seed",
            entry=entry_ref,
            minimum=0,
            maximum=SEED_MAX,
        )
        requests.append(RequestSpec(request_id, arrival_seq, input_dim, seed))
    return tuple(requests)


def _validate_workers(raw: Any, stream_input_dim: int) -> tuple[WorkerSpec, ...]:
    entry = "workers"
    if not isinstance(raw, list) or not raw:
        raise SpecError("workers must be a non-empty list", entry=entry, field="<document>")
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    workers: list[WorkerSpec] = []
    for index, item in enumerate(raw):
        entry_ref = f"workers[{index}]"
        if not isinstance(item, dict):
            raise SpecError("worker entry must be an object", entry=entry_ref, field="<entry>")
        worker_id = _check_required(item, "worker_id", entry_ref)
        if not isinstance(worker_id, str) or not worker_id:
            raise SpecError(
                "worker_id must be a non-empty string",
                entry=entry_ref,
                field="worker_id",
                value=repr(worker_id),
            )
        if worker_id in seen_ids:
            raise SpecError(
                "duplicate worker_id",
                entry=entry_ref,
                field="worker_id",
                value=worker_id,
            )
        seen_ids.add(worker_id)
        host = _check_required(item, "host", entry_ref)
        if host != LOOPBACK_HOST:
            raise SpecError(
                "worker host must be the 127.0.0.1 loopback literal",
                entry=entry_ref,
                field="host",
                value=repr(host),
            )
        port = _check_int(
            _check_required(item, "port", entry_ref),
            name="port",
            entry=entry_ref,
            minimum=PORT_MIN,
            maximum=PORT_MAX,
        )
        if port in seen_ports:
            raise SpecError(
                "duplicate loopback port",
                entry=entry_ref,
                field="port",
                value=port,
            )
        seen_ports.add(port)
        seed = _check_int(
            _check_required(item, "seed", entry_ref),
            name="seed",
            entry=entry_ref,
            minimum=0,
            maximum=SEED_MAX,
        )
        param_version = _check_int(
            _check_required(item, "param_version", entry_ref),
            name="param_version",
            entry=entry_ref,
            minimum=1,
        )
        input_dim = _check_int(
            _check_required(item, "input_dim", entry_ref),
            name="input_dim",
            entry=entry_ref,
            minimum=1,
        )
        if input_dim != stream_input_dim:
            raise SpecError(
                "worker input_dim must match request stream input_dim",
                entry=entry_ref,
                field="input_dim",
                value=input_dim,
                expected=stream_input_dim,
            )
        hidden_dim = _check_int(
            _check_required(item, "hidden_dim", entry_ref),
            name="hidden_dim",
            entry=entry_ref,
            minimum=1,
        )
        batch_size = _check_int(
            _check_required(item, "batch_size", entry_ref),
            name="batch_size",
            entry=entry_ref,
            minimum=1,
        )
        max_in_flight = _check_int(
            _check_required(item, "max_in_flight", entry_ref),
            name="max_in_flight",
            entry=entry_ref,
            minimum=1,
        )
        retries = _check_int(
            _check_required(item, "retries", entry_ref),
            name="retries",
            entry=entry_ref,
            minimum=1,
        )
        workers.append(
            WorkerSpec(
                worker_id,
                host,
                port,
                seed,
                param_version,
                input_dim,
                hidden_dim,
                batch_size,
                max_in_flight,
                retries,
            )
        )
    return tuple(workers)


def _validate_budget(raw: Any) -> BudgetSpec:
    entry = "budget"
    if not isinstance(raw, dict):
        raise SpecError("budget must be an object", entry=entry, field="<document>")
    global_max_in_flight = _check_int(
        _check_required(raw, "global_max_in_flight", entry),
        name="global_max_in_flight",
        entry=entry,
        minimum=1,
    )
    retry_limit = _check_int(
        _check_required(raw, "retry_limit", entry),
        name="retry_limit",
        entry=entry,
        minimum=1,
    )
    cost_budget_ms = _check_int(
        _check_required(raw, "cost_budget_ms", entry),
        name="cost_budget_ms",
        entry=entry,
        minimum=1,
    )
    return BudgetSpec(global_max_in_flight, retry_limit, cost_budget_ms)


def load_spec(requests_path: str, workers_path: str, budget_path: str) -> SystemSpec:
    """Load and strongly validate the three local JSON declarations."""

    requests = _validate_requests(load_json_file(requests_path))
    workers = _validate_workers(load_json_file(workers_path), requests[0].input_dim)
    budget = _validate_budget(load_json_file(budget_path))
    return SystemSpec(
        requests=requests,
        workers=workers,
        budget=budget,
        source_paths={
            "requests": os.path.abspath(requests_path),
            "workers": os.path.abspath(workers_path),
            "budget": os.path.abspath(budget_path),
        },
    )
