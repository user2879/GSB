from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import write_json_atomic


def sample_requests() -> dict[str, list[int]]:
    # Explicit deterministic traces; NumPy is used below to cross-check the
    # fixed-point vector estimate used by batch/sample tooling.
    alpha = [
        0, 4, 9, 10, 11, 18, 25, 31, 22, 15, 8, 3,
        2, 5, 12, 20, 28, 35, 24, 16, 9, 4, 2, 1,
    ]
    beta = [
        0, 2, 7, 9, 15, 17, 20, 16, 12, 7, 4, 2,
        1, 3, 8, 14, 19, 22, 18, 13, 8, 5, 2, 1,
    ]
    gamma = [
        1, 3, 6, 8, 13, 19, 25, 21, 15, 9, 4, 2,
        1, 4, 10, 17, 23, 20, 14, 8, 5, 3, 2, 1,
    ]
    return {"alpha": alpha, "beta": beta, "gamma": gamma}


def build_sample_spec(*, minute_count: int = 18) -> dict[str, Any]:
    requests = sample_requests()
    if not 1 <= minute_count <= 24:
        raise ValueError("minute_count must be between 1 and 24")
    return {
        "format": "scale-review-spec",
        "schema_version": 1,
        "scaling_factor_permille": 1000,
        "decision_rules": {
            "estimation_rule_version": "demand-ceil-v1",
            "scale_up_threshold": 1,
            "scale_down_threshold": 1,
        },
        "algorithm": {
            "window_length": 4,
            "path_switch_length": 16,
            "effort_tie_boundary": 2,
            "tie_break_path": "full",
        },
        "budget": {
            "global_replica_budget": 6,
            "service_concurrency_quotas": {
                "alpha": 30,
                "beta": 16,
                "gamma": 12,
            },
        },
        "lease_policy": {
            "lease_ttl_ms": 450,
            "renew_interval_ms": 120,
            "max_inflight_tasks": 2,
            "max_task_replicas": 2,
            "simulated_task_ms": 35,
        },
        "services": [
            {
                "service_id": "alpha",
                "requests_per_minute": requests["alpha"][:minute_count],
                "concurrency_per_replica": 10,
                "request_duration_ms": 42,
                "min_replicas": 1,
                "max_replicas": 5,
            },
            {
                "service_id": "beta",
                "requests_per_minute": requests["beta"][:minute_count],
                "concurrency_per_replica": 8,
                "request_duration_ms": 35,
                "min_replicas": 1,
                "max_replicas": 4,
            },
            {
                "service_id": "gamma",
                "requests_per_minute": requests["gamma"][:minute_count],
                "concurrency_per_replica": 6,
                "request_duration_ms": 28,
                "min_replicas": 1,
                "max_replicas": 4,
            },
        ],
    }


def build_invalid_spec() -> dict[str, Any]:
    spec = build_sample_spec(minute_count=4)
    spec["services"][1]["requests_per_minute"][2] = -3
    spec["lease_policy"]["renew_interval_ms"] = 900
    return spec


def write_sample_bundle(root: str | Path = ".") -> dict[str, Path]:
    root = Path(root)
    config_dir = root / "configs"
    batch_dir = root / "samples" / "request_batches"
    artifact_dir = root / "artifacts"
    config_dir.mkdir(parents=True, exist_ok=True)
    batch_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    base_path = config_dir / "spec.json"
    full_path = root / "samples" / "spec.full.json"
    invalid_path = config_dir / "invalid-spec.json"
    batch1_path = batch_dir / "batch-001.json"
    batch2_path = batch_dir / "batch-002.json"

    write_json_atomic(base_path, build_sample_spec(minute_count=18))
    write_json_atomic(full_path, build_sample_spec(minute_count=24))
    write_json_atomic(invalid_path, build_invalid_spec())

    requests = sample_requests()
    write_json_atomic(
        batch1_path,
        {
            "batch_id": "batch-001",
            "first_minute": 18,
            "minutes": [18, 19, 20],
            "requests": {
                service_id: values[18:21]
                for service_id, values in requests.items()
            },
        },
    )
    write_json_atomic(
        batch2_path,
        {
            "batch_id": "batch-002",
            "first_minute": 21,
            "minutes": [21, 22, 23],
            "requests": {
                service_id: values[21:24]
                for service_id, values in requests.items()
            },
        },
    )
    return {
        "base_spec": base_path,
        "full_spec": full_path,
        "invalid_spec": invalid_path,
        "batch_001": batch1_path,
        "batch_002": batch2_path,
    }
