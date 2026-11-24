"""Deterministic sample-workspace generation and demonstration helpers."""

from __future__ import annotations

import json
import shutil
import socket
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Mapping

from .conversion import (
    ConversionJournal,
    SidecarStore,
    convert_receipt,
    receipt_to_file_payload,
    write_conversion_bundle,
)
from .jsonutil import atomic_write_json, atomic_write_text, canonical_dumps, hash_json, pretty_dumps
from .models import R1, R2, R3, Receipt
from .receipts import parse_receipt_paths
from .supervisor import infer_batch, start_workers, stop_workers
from .worker import forward_summary


SEED = 20261125


def _free_ports(start: int, count: int) -> list[int]:
    for base in range(start, start + 500):
        ports = []
        sockets = []
        try:
            for port in range(base, base + count):
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(("127.0.0.1", port))
                sockets.append(sock)
                ports.append(port)
            return ports
        except OSError:
            continue
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError("could not find consecutive free loopback ports")


def _request(
    sequence: int,
    request_id: str,
    kind: str,
    batch_id: str,
    worker_id: str,
    seed: int,
    *,
    arrival_ms: int | None = None,
    overwritten_ref: str | None = None,
) -> dict[str, Any]:
    item = {
        "sequence": sequence,
        "request_id": request_id,
        "kind": kind,
        "arrival_ms": sequence * 7 if arrival_ms is None else arrival_ms,
        "batch_id": batch_id,
        "worker_id": worker_id,
        "input_seed": seed,
    }
    if overwritten_ref is not None:
        item["overwritten_ref"] = overwritten_ref
    return item


def _config(ports: list[int]) -> dict[str, Any]:
    return {
        "schema_version": "mmledger-config-v1",
        "workspace_root": "..",
        "seed": SEED,
        "model": {"input_dim": 16, "hidden_dim": 8, "quantization_bits": 8},
        "limits": {
            "batch_size": 3,
            "normalization_budget": 64,
            "worker_quota_points": 20,
        },
        "port_range": {"min": 18800, "max": 19400},
        "parameters": [
            {"parameter_version": 3, "digest": "sha256:pv3-alpha"},
            {"parameter_version": 4, "digest": "sha256:pv4-beta"},
        ],
        "workers": [
            {
                "worker_id": "w-alpha",
                "host": "127.0.0.1",
                "port": ports[0],
                "parameter_version": 3,
                "parameter_digest": "sha256:pv3-alpha",
                "quota_points": 20,
            },
            {
                "worker_id": "w-beta",
                "host": "127.0.0.1",
                "port": ports[1],
                "parameter_version": 4,
                "parameter_digest": "sha256:pv4-beta",
                "quota_points": 20,
            },
        ],
        "request_stream": "requests.json",
        "merge_rules": "merge_rules.json",
        "receipt_glob": "receipts/*.json",
        "fault_injections": "fault_injections.json",
        "fault_receipt_glob": "fault_receipts/*.json",
    }


def _requests() -> dict[str, Any]:
    return {
        "schema_version": "request-stream-v1",
        "requests": [
            _request(1, "req-001", "image_text", "batch-001", "w-alpha", 1001),
            _request(2, "req-002", "image_text", "batch-001", "w-alpha", 1002),
            _request(3, "req-003", "image_text", "batch-001", "w-alpha", 1003),
            _request(4, "req-004", "image_only", "batch-002", "w-beta", 1004),
            _request(5, "req-005", "image_only", "batch-002", "w-beta", 1005),
            _request(6, "req-006", "image_only", "batch-002", "w-beta", 1006),
            _request(7, "req-007", "image_text", "batch-003", "w-alpha", 1007),
            _request(8, "req-008", "image_text", "batch-003", "w-alpha", 1008),
            _request(9, "req-009", "image_text", "batch-003", "w-alpha", 1009),
            _request(10, "req-010", "image_only", "batch-004", "w-beta", 1010),
            _request(
                11,
                "req-009",
                "image_text",
                "batch-004",
                "w-beta",
                1009,
                overwritten_ref="batch-003:req-009:w-alpha:3",
            ),
            _request(12, "req-011", "image_text", "batch-005", "w-alpha", 1011),
            _request(13, "req-012", "image_text", "batch-005", "w-alpha", 1012),
        ],
    }


def _merge_rules() -> dict[str, Any]:
    return {
        "schema_version": "merge-rules-v1",
        "worker_sets": {
            "active": ["w-alpha", "w-beta"],
            "image_only": ["w-beta"],
        },
        "parameter_version_sets": {
            "production": [3, 4],
        },
        "active_worker_set": "active",
        "active_parameter_version_set": "production",
        "worker_priority": ["w-alpha", "w-beta"],
    }


def _receipt_file(version: str, records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    return {"version": version, "receipts": [dict(record) for record in records]}


def _record_from_response(
    response: Mapping[str, Any],
    request: Mapping[str, Any],
    *,
    version: str,
    stable_rank: int | None = None,
    worker_id: str | None = None,
    parameter_version: int | None = None,
    parameter_digest: str | None = None,
) -> dict[str, Any]:
    matching = next(
        item for item in response["requests"] if item["request_id"] == request["request_id"]
    )
    record: dict[str, Any] = {
        "batch_id": response["batch_id"],
        "request_id": request["request_id"],
        "output_digest": matching["output_digest"],
    }
    if version in (R2, R3):
        record.update(
            {
                "parameter_version": response["parameter_version"]
                if parameter_version is None
                else parameter_version,
                "duration_ms": response["duration_ms"],
                "worker_id": response["worker_id"] if worker_id is None else worker_id,
            }
        )
    if version == R3:
        record.update(
            {
                "stable_rank": matching["stable_rank"] if stable_rank is None else stable_rank,
                "overwritten_ref": request.get("overwritten_ref"),
                "parameter_digest": response["parameter_digest"]
                if parameter_digest is None
                else parameter_digest,
            }
        )
    return record


def _direct_response(
    bundle: Any,
    batch_id: str,
    requests: list[Mapping[str, Any]],
) -> dict[str, Any]:
    worker = bundle.workers_by_id[requests[0]["worker_id"]]
    ranked = {item["request_id"]: index + 1 for index, item in enumerate(sorted(requests, key=lambda item: item["request_id"]))}
    summaries = []
    for request in requests:
        summary = forward_summary(
            model=bundle.model,
            seed=bundle.seed,
            request_id=request["request_id"],
            sequence=request["sequence"],
            kind=request["kind"],
            worker_id=worker.worker_id,
            parameter_version=worker.parameter_version,
            parameter_digest=worker.parameter_digest,
        )
        summary["stable_rank"] = ranked[request["request_id"]]
        summary["overwritten_ref"] = request.get("overwritten_ref")
        summaries.append(summary)
    per_request_ms = max(1, (bundle.model.macs_per_request * bundle.model.quantization_bits // 8) // 64)
    return {
        "batch_id": batch_id,
        "worker_id": worker.worker_id,
        "parameter_version": worker.parameter_version,
        "parameter_digest": worker.parameter_digest,
        "duration_ms": max(1, len(summaries) * per_request_ms),
        "requests": sorted(summaries, key=lambda item: item["request_id"]),
        "macs_per_request": bundle.model.macs_per_request,
        "quantized_parameter_units": bundle.model.quantized_parameter_units,
    }


def generate_workspace(
    root: str | Path,
    *,
    use_subprocess_workers: bool = True,
    clean_outputs: bool = True,
    jobs: int = 4,
) -> dict[str, Any]:
    workspace = Path(root).resolve()
    samples = workspace / "samples"
    snapshots = workspace / "snapshots"
    reports = workspace / "reports"
    for directory in (samples, snapshots, reports):
        directory.mkdir(parents=True, exist_ok=True)
    if clean_outputs:
        for directory in (samples / "receipts", samples / "fault_receipts", samples / "conversion"):
            if directory.exists():
                shutil.rmtree(directory)
        for directory in (snapshots, reports):
            if directory.exists():
                shutil.rmtree(directory)
            directory.mkdir(parents=True, exist_ok=True)
        for path in (
            samples / "config.json",
            samples / "requests.json",
            samples / "merge_rules.json",
            samples / "fault_injections.json",
        ):
            if path.exists():
                path.unlink()
    (samples / "receipts").mkdir(parents=True, exist_ok=True)
    (samples / "fault_receipts").mkdir(parents=True, exist_ok=True)
    (samples / "conversion").mkdir(parents=True, exist_ok=True)

    ports = _free_ports(18900, 2)
    config = _config(ports)
    requests = _requests()
    rules = _merge_rules()
    atomic_write_json(samples / "config.json", config)
    atomic_write_json(samples / "requests.json", requests)
    atomic_write_json(samples / "merge_rules.json", rules)

    from .config import load_config

    config_result = load_config(samples / "config.json")
    if config_result.bundle is None:
        raise RuntimeError("; ".join(f"{item.code}: {item.message}" for item in config_result.failures))
    bundle = config_result.bundle
    response_by_batch: dict[str, dict[str, Any]] = {}
    request_by_batch: dict[str, list[Mapping[str, Any]]] = {}
    for request in requests["requests"]:
        request_by_batch.setdefault(request["batch_id"], []).append(request)

    processes = []
    try:
        if use_subprocess_workers:
            bundle, processes = start_workers(samples / "config.json")
            for batch_id in sorted(request_by_batch):
                worker_id = request_by_batch[batch_id][0]["worker_id"]
                worker = bundle.workers_by_id[worker_id]
                response_by_batch[batch_id] = dict(
                    infer_batch(
                        worker,
                        batch_id,
                        tuple(
                            request
                            for request in bundle.requests
                            if request.batch_id == batch_id
                        ),
                    )
                )
        else:
            for batch_id in sorted(request_by_batch):
                response_by_batch[batch_id] = _direct_response(bundle, batch_id, request_by_batch[batch_id])
    finally:
        stop_workers(processes)

    r1_requests = request_by_batch["batch-001"]
    atomic_write_json(
        samples / "receipts" / "r1_batch_001.json",
        _receipt_file(
            R1,
            (
                _record_from_response(response_by_batch["batch-001"], request, version=R1)
                for request in r1_requests
            ),
        ),
    )
    r2_requests = request_by_batch["batch-002"]
    atomic_write_json(
        samples / "receipts" / "r2_batch_002.json",
        _receipt_file(
            R2,
            (
                _record_from_response(response_by_batch["batch-002"], request, version=R2)
                for request in r2_requests
            ),
        ),
    )
    for batch_id, filename in (
        ("batch-003", "r3_batch_003.json"),
        ("batch-004", "r3_batch_004_overwrite.json"),
        ("batch-005", "r3_batch_005.json"),
    ):
        atomic_write_json(
            samples / "receipts" / filename,
            _receipt_file(
                R3,
                (
                    _record_from_response(response_by_batch[batch_id], request, version=R3)
                    for request in request_by_batch[batch_id]
                ),
            ),
        )

    conflict_request = request_by_batch["batch-005"][0]
    conflict_response = response_by_batch["batch-005"]
    conflict_record = _record_from_response(
        conflict_response,
        conflict_request,
        version=R3,
        stable_rank=2,
        worker_id="w-beta",
        parameter_version=4,
        parameter_digest="sha256:pv4-beta",
    )
    conflict_record["output_digest"] = hash_json(
        {"conflict": conflict_record["output_digest"], "worker_id": "w-beta"}
    )
    atomic_write_json(
        samples / "receipts" / "r3_batch_005_rank_conflict.json",
        _receipt_file(R3, [conflict_record]),
    )

    _write_faults(samples, response_by_batch, request_by_batch)
    _write_conversion_demo(samples, bundle)
    summary = {
        "workspace": str(workspace),
        "ports": ports,
        "workers": [worker.worker_id for worker in bundle.workers],
        "batches": sorted(request_by_batch),
        "receipts": sorted(path.name for path in (samples / "receipts").glob("*.json")),
        "faults": sorted(path.name for path in (samples / "fault_receipts").glob("*.json")),
    }
    atomic_write_json(reports / "generation_summary.json", summary)
    return summary


def _write_faults(
    samples: Path,
    responses: Mapping[str, Mapping[str, Any]],
    request_by_batch: Mapping[str, list[Mapping[str, Any]]],
) -> None:
    fault_dir = samples / "fault_receipts"
    atomic_write_text(fault_dir / "injected_truncated.json", '{"version":"r3","receipts":[')

    unknown = _record_from_response(
        responses["batch-002"], request_by_batch["batch-002"][0], version=R2
    )
    unknown["parameter_version"] = 99
    atomic_write_json(fault_dir / "injected_unknown_parameter.json", _receipt_file(R2, [unknown]))

    nonfinite = _record_from_response(
        responses["batch-002"], request_by_batch["batch-002"][1], version=R2
    )
    nonfinite["request_id"] = "fault-nonfinite"
    text = pretty_dumps(_receipt_file(R2, [nonfinite])).replace(
        f'"duration_ms": {nonfinite["duration_ms"]}', '"duration_ms": NaN'
    )
    atomic_write_text(fault_dir / "injected_nonfinite_duration.json", text)

    duplicate = _record_from_response(
        responses["batch-001"], request_by_batch["batch-001"][0], version=R2
    )
    atomic_write_json(
        fault_dir / "injected_duplicate_batch.json",
        _receipt_file(R2, [duplicate, dict(duplicate)]),
    )

    atomic_write_json(
        samples / "fault_injections.json",
        {
            "schema_version": "mmledger-fault-injections-v1",
            "injections": [
                {
                    "id": "INJ-TRUNCATED",
                    "kind": "truncated_receipt",
                    "source_file": "fault_receipts/injected_truncated.json",
                    "expected_code": "E_TRUNCATED_RECEIPT",
                },
                {
                    "id": "INJ-UNKNOWN-PARAM",
                    "kind": "unknown_parameter_version",
                    "source_file": "fault_receipts/injected_unknown_parameter.json",
                    "expected_code": "E_UNKNOWN_PARAM_VERSION",
                },
                {
                    "id": "INJ-NONFINITE-DURATION",
                    "kind": "nonfinite_duration",
                    "source_file": "fault_receipts/injected_nonfinite_duration.json",
                    "expected_code": "E_DURATION_NONFINITE",
                },
                {
                    "id": "INJ-DUPLICATE-BATCH",
                    "kind": "duplicate_batch_id",
                    "source_file": "fault_receipts/injected_duplicate_batch.json",
                    "expected_code": "E_DUPLICATE_BATCH_ID",
                },
            ],
        },
    )


def _write_conversion_demo(samples: Path, bundle: Any) -> None:
    parsed = parse_receipt_paths(
        [samples / "receipts" / "r3_batch_003.json"],
        base_dir=samples,
        known_versions=frozenset(bundle.parameters_by_version),
    )
    source = parsed.receipts[0]
    sidecar = SidecarStore()
    journal = ConversionJournal()
    down, event, failure = convert_receipt(source, R1, sidecar, sequence=1)
    if failure is not None or down is None or event is None:
        raise RuntimeError("failed to create downgrade demo")
    journal.add(event)
    conversion_dir = samples / "conversion"
    atomic_write_json(conversion_dir / "r1_from_r3.json", receipt_to_file_payload(down))
    atomic_write_json(conversion_dir / "sidecar.json", sidecar.as_dict())
    r1_parsed = parse_receipt_paths(
        [conversion_dir / "r1_from_r3.json"],
        base_dir=samples,
        known_versions=frozenset(bundle.parameters_by_version),
    )
    if not r1_parsed.receipts:
        raise RuntimeError("failed to parse downgraded demo receipt")
    restored, event2, failure2 = convert_receipt(r1_parsed.receipts[0], R3, sidecar, sequence=2)
    if failure2 is not None or restored is None or event2 is None:
        raise RuntimeError("failed to create restore demo")
    journal.add(event2)
    atomic_write_json(conversion_dir / "r3_restored.json", receipt_to_file_payload(restored))
    atomic_write_json(conversion_dir / "conversion_journal.json", journal.as_dict())
    atomic_write_json(conversion_dir / "sidecar.json", sidecar.as_dict())
    comparison = {
        "source_uid": source.uid,
        "restored_uid": restored.uid,
        "fields_equal": {
            name: getattr(source, name) == getattr(restored, name)
            for name in (
                "batch_id",
                "request_id",
                "output_digest",
                "parameter_version",
                "duration_ms",
                "worker_id",
                "stable_rank",
                "overwritten_ref",
                "parameter_digest",
            )
        },
        "completeness": restored.completeness,
    }
    atomic_write_json(conversion_dir / "roundtrip_check.json", comparison)
