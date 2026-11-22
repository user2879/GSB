"""Deterministic in-task sample declarations (no network, no RNG)."""

from __future__ import annotations

import json
import os

MINUTES = 24


def _series(pattern: str) -> list[int]:
    values: list[int] = []
    for minute in range(MINUTES):
        if pattern == "wave":
            value = 1200 + 900 * ((minute % 12) // 6) + minute * 17
        elif pattern == "rising":
            value = 200 + minute * minute * 9
        elif pattern == "idle":
            value = 40 + (minute % 4) * 15
        elif pattern == "spiky":
            value = 300 + (2400 if minute % 7 == 3 else 0) + minute * 11
        else:
            raise ValueError(pattern)
        values.append(int(value))
    return values


def build_cluster() -> dict:
    return {
        "global_replica_budget": 40,
        "fixed_point_scale": 1000,
        "latency_budget_ms": 50,
        "incremental_threshold_minutes": 12,
        "window_size": 8,
        "services": [
            {
                "service_id": "svc-embedding",
                "requests": _series("wave"),
                "concurrency_per_replica": 8,
                "latency_ms": 120,
                "scale_up_threshold": 7500,
                "scale_down_threshold": 2500,
                "min_replicas": 2,
                "max_replicas": 16,
                "concurrency_quota": 14,
            },
            {
                "service_id": "svc-ranker",
                "requests": _series("rising"),
                "concurrency_per_replica": 4,
                "latency_ms": 60,
                "scale_up_threshold": 8000,
                "scale_down_threshold": 3000,
                "min_replicas": 1,
                "max_replicas": 12,
                "concurrency_quota": 10,
            },
            {
                "service_id": "svc-router",
                "requests": _series("spiky"),
                "concurrency_per_replica": 16,
                "latency_ms": 25,
                "scale_up_threshold": 8500,
                "scale_down_threshold": 3500,
                "min_replicas": 1,
                "max_replicas": 20,
                "concurrency_quota": 18,
            },
            {
                "service_id": "svc-cache",
                "requests": _series("idle"),
                "concurrency_per_replica": 32,
                "latency_ms": 10,
                "scale_up_threshold": 9000,
                "scale_down_threshold": 2000,
                "min_replicas": 1,
                "max_replicas": 6,
                "concurrency_quota": 6,
            },
        ],
    }


def build_small_cluster() -> dict:
    cluster = build_cluster()
    cluster["incremental_threshold_minutes"] = 30
    for service in cluster["services"]:
        service["requests"] = service["requests"][:8]
    return cluster


def build_corrupt_cluster() -> dict:
    cluster = build_small_cluster()
    cluster["global_replica_budget"] = -1
    return cluster


def write_samples(directory: str) -> list[str]:
    os.makedirs(directory, exist_ok=True)
    payloads = {
        "cluster.json": build_cluster(),
        "cluster_small.json": build_small_cluster(),
        "cluster_corrupt_schema.json": build_corrupt_cluster(),
    }
    paths = []
    for name, payload in payloads.items():
        path = os.path.join(directory, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
        paths.append(path)
    return paths
