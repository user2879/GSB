from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Iterable

from .codes import E_ARRIVAL_ORDER, E_BATCH_SIZE_INVALID, E_WORKER_INVALID
from .model import Failure

DEFAULT_CONFIG: dict[str, Any] = {
    "batch_size": 4,
    "merge_rule": "coverage_then_version_v1",
    "allowed_merge_rules": ["coverage_then_version_v1", "version_then_batch_v1"],
    "seed": 20261125,
    "model_dim": 16,
    "hidden_dim": 32,
    "output_dim": 16,
    "quant_bits": 8,
    "fixed_point_scale": 1_000_000,
    "budget_ms": 100,
    "worker_throughput_mac_per_ms_q": 2_000_000,
    "port_min": 20_000,
    "port_max": 39_999,
    "interactive_budget_ms": 250,
    "global_version": 1,
}


def deep_default_config() -> dict[str, Any]:
    return copy.deepcopy(DEFAULT_CONFIG)


def _is_plain_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


@dataclass
class ValidatedInputs:
    config: dict[str, Any]
    workers: list[dict[str, Any]]
    requests: list[dict[str, Any]]
    failures: list[Failure] = field(default_factory=list)
    batches: list[dict[str, Any]] = field(default_factory=list)


def validate_config(config: dict[str, Any]) -> list[Failure]:
    failures: list[Failure] = []
    batch_size = config.get("batch_size")
    if not _is_plain_int(batch_size) or batch_size <= 0:
        failures.append(
            Failure(
                E_BATCH_SIZE_INVALID,
                "batch_size must be an integer greater than zero",
                source="config.json",
                details={"batch_size": batch_size},
            )
        )
    for key in ("model_dim", "hidden_dim"):
        value = config.get(key)
        if not _is_plain_int(value) or value <= 0:
            failures.append(
                Failure(
                    E_WORKER_INVALID,
                    f"{key} must be a positive integer",
                    source="config.json",
                    details={key: value},
                )
            )
    port_min = config.get("port_min")
    port_max = config.get("port_max")
    if not _is_plain_int(port_min) or not _is_plain_int(port_max) or port_min <= 0 or port_max < port_min:
        failures.append(
            Failure(
                E_WORKER_INVALID,
                "port range must contain positive integer min <= max",
                source="config.json",
                details={"port_min": port_min, "port_max": port_max},
            )
        )
    return failures


def validate_workers(workers: Iterable[dict[str, Any]], config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[Failure]]:
    failures: list[Failure] = []
    valid: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_ports: set[int] = set()
    port_min_value = config.get("port_min", 20_000)
    port_max_value = config.get("port_max", 39_999)
    port_min = port_min_value if _is_plain_int(port_min_value) and port_min_value > 0 else 20_000
    port_max = port_max_value if _is_plain_int(port_max_value) and port_max_value >= port_min else 39_999

    for index, raw in enumerate(workers):
        worker = dict(raw)
        worker_id = worker.get("worker_id")
        source = str(worker.get("_source", "workers.json"))
        if not isinstance(worker_id, str) or not worker_id:
            failures.append(Failure(E_WORKER_INVALID, "worker_id must be a non-empty string", source, index, {"worker": raw}))
            continue
        if worker_id in seen_ids:
            failures.append(Failure(E_WORKER_INVALID, "duplicate worker_id", source, index, {"worker_id": worker_id}))
            continue

        port = worker.get("port")
        port_ok = _is_plain_int(port) and (port == 0 or port_min <= port <= port_max)
        if not port_ok:
            failures.append(
                Failure(
                    E_WORKER_INVALID,
                    f"port must be 0 (auto) or in [{port_min}, {port_max}]",
                    source,
                    index,
                    {"worker_id": worker_id, "port": port},
                )
            )
            continue
        if port != 0 and port in seen_ports:
            failures.append(Failure(E_WORKER_INVALID, "duplicate loopback port", source, index, {"worker_id": worker_id, "port": port}))
            continue

        for key in ("model_dim", "hidden_dim"):
            value = worker.get(key, config.get(key))
            if not _is_plain_int(value) or value <= 0:
                failures.append(
                    Failure(E_WORKER_INVALID, f"{key} must be a positive integer", source, index, {"worker_id": worker_id, key: value})
                )
                break
        else:
            seen_ids.add(worker_id)
            if port != 0:
                seen_ports.add(port)
            worker.setdefault("model_dim", config["model_dim"])
            worker.setdefault("hidden_dim", config["hidden_dim"])
            worker.setdefault("param_version", 1)
            if not _is_plain_int(worker["param_version"]) or worker["param_version"] < 0:
                failures.append(
                    Failure(
                        E_WORKER_INVALID,
                        "param_version must be a non-negative integer",
                        source,
                        index,
                        {"worker_id": worker_id, "param_version": worker["param_version"]},
                    )
                )
                continue
            valid.append(worker)

    valid.sort(key=lambda item: item["worker_id"])
    return valid, failures


def validate_and_split_requests(
    requests: Iterable[dict[str, Any]], batch_size: Any
) -> tuple[list[dict[str, Any]], list[Failure], list[dict[str, Any]]]:
    failures: list[Failure] = []
    valid: list[dict[str, Any]] = []
    previous_seq: int | None = None
    previous_arrival: int | None = None

    for index, raw in enumerate(requests):
        record = dict(raw)
        source = str(record.pop("_source", "requests.jsonl"))
        seq = record.get("seq", record.get("sequence"))
        arrival_ms = record.get("arrival_ms")
        request_id = record.get("request_id")
        ok = True

        if not _is_plain_int(seq):
            failures.append(Failure(E_ARRIVAL_ORDER, "seq must be an integer", source, index, {"seq": seq, "request_id": request_id}))
            ok = False
        elif previous_seq is not None and seq <= previous_seq:
            failures.append(
                Failure(
                    E_ARRIVAL_ORDER,
                    "request seq must be strictly increasing",
                    source,
                    index,
                    {"seq": seq, "previous_seq": previous_seq, "request_id": request_id},
                )
            )
            ok = False

        if not _is_plain_int(arrival_ms) or arrival_ms < 0:
            failures.append(
                Failure(E_ARRIVAL_ORDER, "arrival_ms must be a non-negative integer", source, index, {"arrival_ms": arrival_ms, "request_id": request_id})
            )
            ok = False
        elif previous_arrival is not None and arrival_ms < previous_arrival:
            failures.append(
                Failure(
                    E_ARRIVAL_ORDER,
                    "arrival_ms moved backwards",
                    source,
                    index,
                    {"arrival_ms": arrival_ms, "previous_arrival_ms": previous_arrival, "request_id": request_id},
                )
            )
            ok = False

        if not isinstance(request_id, str) or not request_id:
            failures.append(Failure(E_ARRIVAL_ORDER, "request_id must be a non-empty string", source, index, {"request_id": request_id}))
            ok = False

        if ok:
            record["seq"] = int(seq)
            record["arrival_ms"] = int(arrival_ms)
            valid.append(record)
            previous_seq = int(seq)
            previous_arrival = int(arrival_ms)
        else:
            # Keep ordering diagnostics anchored to the last values actually seen.
            if _is_plain_int(seq):
                previous_seq = int(seq)
            if _is_plain_int(arrival_ms) and arrival_ms >= 0:
                previous_arrival = int(arrival_ms)

    batches: list[dict[str, Any]] = []
    if _is_plain_int(batch_size) and batch_size > 0 and valid:
        worker_ids: list[str] = []
        # Assignment is supplied by the caller through request records or derived later.
        for start in range(0, len(valid), int(batch_size)):
            number = start // int(batch_size) + 1
            batch_id = f"B-{number:04d}"
            members = valid[start : start + int(batch_size)]
            for position, item in enumerate(members):
                item["planned_batch_id"] = batch_id
                item["planned_rank"] = position
            batches.append(
                {
                    "batch_id": batch_id,
                    "batch_index": number,
                    "request_ids": [item["request_id"] for item in members],
                    "seqs": [item["seq"] for item in members],
                    "size": len(members),
                }
            )
    return valid, failures, batches


def assign_batch_workers(batches: list[dict[str, Any]], workers: list[dict[str, Any]]) -> None:
    if not workers:
        return
    ordered = sorted(workers, key=lambda item: item["worker_id"])
    for batch in batches:
        worker = ordered[(int(batch["batch_index"]) - 1) % len(ordered)]
        batch["worker_id"] = worker["worker_id"]
        batch["param_version"] = worker["param_version"]


def normalize_failure_order(failures: Iterable[Failure]) -> list[dict[str, Any]]:
    unique: dict[tuple[Any, ...], Failure] = {}
    for failure in failures:
        unique[failure.sort_key + (failure.code,)] = failure
    return [item.to_dict() for _, item in sorted(unique.items(), key=lambda kv: kv[0])]
