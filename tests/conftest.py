from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ossdiag.engine import Workbench
from ossdiag.models import SessionSpec, load_spec_data


KiB = 1024
MiB = 1024 * 1024


def object_doc(
    key: str = "object/a.bin",
    *,
    total_bytes: int = 128 * KiB,
    shard_size: int = 64 * KiB,
    batch_size: int = 64 * KiB,
    concurrency: int = 1,
    lease_ms: int = 500,
    renew_ms: int = 100,
    seed_base: int = 100,
) -> dict[str, Any]:
    shard_count = (total_bytes + shard_size - 1) // shard_size
    return {
        "object_key": key,
        "total_bytes": total_bytes,
        "shard_size": shard_size,
        "shard_count": shard_count,
        "shard_seeds": [seed_base + index for index in range(shard_count)],
        "concurrency_limit": concurrency,
        "batch_size": batch_size,
        "lease_policy": {
            "lease_ms": lease_ms,
            "renew_interval_ms": renew_ms,
            "max_concurrent_leases": concurrency,
        },
    }


def spec_doc(
    objects: list[dict[str, Any]] | None = None,
    *,
    global_lease_quota: int = 3,
    throughput_budget_bytes: int = 4 * MiB,
    scale_exponent: int = 6,
    tie_threshold_work: int = 32,
    worker_threads: int = 3,
    batch_size: int = 64 * KiB,
) -> dict[str, Any]:
    return {
        "session": {
            "global_lease_quota": global_lease_quota,
            "throughput_budget_bytes": throughput_budget_bytes,
            "scale_exponent": scale_exponent,
            "tie_threshold_work": tie_threshold_work,
            "worker_threads": worker_threads,
            "batch_size": batch_size,
        },
        "objects": objects or [object_doc()],
    }


def session_from(doc: dict[str, Any]) -> SessionSpec:
    return load_spec_data(doc)


def write_spec(path: Path, doc: dict[str, Any]) -> Path:
    path.write_text(json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def workbench(tmp_path: Path, session: SessionSpec | None = None, doc: dict[str, Any] | None = None) -> Workbench:
    if session is None:
        session = session_from(doc or spec_doc())
    return Workbench(
        session,
        ledger_path=tmp_path / "ledger.sqlite",
        data_dir=tmp_path / "data",
        log_path=tmp_path / "events.jsonl",
        backup_dir=tmp_path / "ledger_backups",
    )


@pytest.fixture
def small_doc() -> dict[str, Any]:
    return spec_doc()
