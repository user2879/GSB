"""Deterministic receipt generation for scenarios and local tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .compute import WorkerRuntime, build_receipt_rows
from .config import load_config, load_requests
from .model import AppConfig, RequestSpec
from .util import atomic_write_text, canonical_json


def generate_worker_rows(
    config: AppConfig,
    requests: tuple[RequestSpec, ...],
    *,
    worker_id: str,
    version: str,
    param_version: int | None = None,
    batch_size: int | None = None,
    request_ids: Iterable[str] | None = None,
    batch_prefix: str | None = None,
    duration_ms: int | None = None,
) -> list[dict[str, Any]]:
    worker = config.worker_by_id.get(worker_id)
    if worker is None:
        raise ValueError(f"unknown worker {worker_id!r}")
    if version not in {"r1", "r2", "r3"}:
        raise ValueError("version must be r1, r2 or r3")
    version_number = worker.param_version if param_version is None else int(param_version)
    known = set(worker.known_param_versions or (worker.param_version,))
    if version_number not in known:
        raise ValueError(
            f"parameter version {version_number} is not known for {worker_id}; expected one of {sorted(known)}"
        )
    runtime = WorkerRuntime(worker, version_number, config.throughput_ops_per_ms)
    selected = list(requests)
    if request_ids is not None:
        requested = set(request_ids)
        selected = [request for request in requests if request.request_id in requested]
        missing = sorted(requested - {request.request_id for request in selected})
        if missing:
            raise ValueError(f"unknown request ids: {missing}")
    size = config.batch_size if batch_size is None else int(batch_size)
    if size < 1:
        raise ValueError("batch_size must be positive")
    prefix = batch_prefix or f"b-{worker_id}-{version}"
    rows: list[dict[str, Any]] = []
    for batch_index, start in enumerate(range(0, len(selected), size), start=1):
        batch_requests = selected[start : start + size]
        rows.extend(
            build_receipt_rows(
                runtime,
                batch_id=f"{prefix}-{batch_index:03d}",
                requests=batch_requests,
                version=version,
                duration_ms=duration_ms,
            )
        )
    return rows


def rows_to_ndjson(rows: Iterable[dict[str, Any]]) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


def generate_file(
    *,
    config_path: str | Path,
    requests_path: str | Path,
    output_path: str | Path,
    worker_id: str,
    version: str,
    param_version: int | None = None,
    batch_size: int | None = None,
    request_ids: Iterable[str] | None = None,
    batch_prefix: str | None = None,
    duration_ms: int | None = None,
) -> list[dict[str, Any]]:
    config = load_config(config_path)
    requests = load_requests(requests_path, batch_size=config.batch_size)
    rows = generate_worker_rows(
        config,
        requests,
        worker_id=worker_id,
        version=version,
        param_version=param_version,
        batch_size=batch_size,
        request_ids=request_ids,
        batch_prefix=batch_prefix,
        duration_ms=duration_ms,
    )
    atomic_write_text(Path(output_path), rows_to_ndjson(rows))
    return rows

