"""Deterministic sample scene generation used by demos and tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canonical import canonical_text
from .models import (
    OBS_ERROR,
    OBS_REFUSED,
    OBS_SUCCESS,
    OBS_TIMEOUT,
    ROLE_PRIMARY,
    ROLE_REPLICA,
)


def _obs(service_id, instance_id, round_index, timestamp, result):
    return {
        "service_id": service_id,
        "instance_id": instance_id,
        "round_index": round_index,
        "observed_at_ms": timestamp,
        "result": result,
    }


def build_demo_scene() -> dict[str, Any]:
    services = [
        {
            "service_id": "order-service",
            "strategy": "auto",
            "cooldown_ms": 1000,
            "min_healthy_replicas": 1,
            "success_threshold": 2,
            "failure_threshold": 2,
        },
        {
            "service_id": "payment-service",
            "strategy": "manual",
            "cooldown_ms": 500,
            "min_healthy_replicas": 2,
            "success_threshold": 1,
            "failure_threshold": 1,
        },
        {
            "service_id": "inventory-service",
            "strategy": "auto",
            "cooldown_ms": 0,
            "min_healthy_replicas": 1,
            "success_threshold": 3,
            "failure_threshold": 2,
        },
    ]
    instances = [
        {"service_id": "order-service", "instance_id": "ord-1", "role": ROLE_PRIMARY, "version": 1},
        {"service_id": "order-service", "instance_id": "ord-2", "role": ROLE_REPLICA, "version": 1},
        {"service_id": "order-service", "instance_id": "ord-3", "role": ROLE_REPLICA, "version": 2},
        {"service_id": "payment-service", "instance_id": "pay-1", "role": ROLE_PRIMARY, "version": 1},
        {"service_id": "payment-service", "instance_id": "pay-2", "role": ROLE_REPLICA, "version": 1},
        {"service_id": "payment-service", "instance_id": "pay-3", "role": ROLE_REPLICA, "version": 3},
        {"service_id": "inventory-service", "instance_id": "inv-1", "role": ROLE_PRIMARY, "version": 1},
        {"service_id": "inventory-service", "instance_id": "inv-2", "role": ROLE_REPLICA, "version": 1},
    ]
    observations = []
    # order-service: healthy start, primary fails twice -> switch, cooldown, re-check
    order_rows = [
        (0, 100, OBS_SUCCESS, OBS_SUCCESS, OBS_SUCCESS),
        (1, 200, OBS_TIMEOUT, OBS_SUCCESS, OBS_SUCCESS),
        (2, 300, OBS_REFUSED, OBS_SUCCESS, OBS_SUCCESS),
        (3, 400, OBS_ERROR, OBS_SUCCESS, OBS_TIMEOUT),
        (4, 500, OBS_TIMEOUT, OBS_SUCCESS, OBS_SUCCESS),
        (5, 1450, OBS_TIMEOUT, OBS_SUCCESS, OBS_SUCCESS),
        (6, 2450, OBS_SUCCESS, OBS_SUCCESS, OBS_SUCCESS),
        (7, 2550, OBS_SUCCESS, OBS_REFUSED, OBS_SUCCESS),
    ]
    for round_index, timestamp, r1, r2, r3 in order_rows:
        observations.append(_obs("order-service", "ord-1", round_index, timestamp, r1))
        observations.append(_obs("order-service", "ord-2", round_index, timestamp, r2))
        observations.append(_obs("order-service", "ord-3", round_index, timestamp, r3))

    # payment-service: manual strategy must refuse auto switch; min healthy 2
    payment_rows = [
        (0, 100, OBS_SUCCESS, OBS_SUCCESS, OBS_SUCCESS),
        (1, 200, OBS_REFUSED, OBS_SUCCESS, OBS_SUCCESS),
        (2, 300, OBS_ERROR, OBS_TIMEOUT, OBS_SUCCESS),
        (3, 400, OBS_ERROR, OBS_ERROR, OBS_SUCCESS),
        (5, 600, OBS_TIMEOUT, OBS_SUCCESS, OBS_SUCCESS),
    ]
    for round_index, timestamp, r1, r2, r3 in payment_rows:
        observations.append(_obs("payment-service", "pay-1", round_index, timestamp, r1))
        observations.append(_obs("payment-service", "pay-2", round_index, timestamp, r2))
        observations.append(_obs("payment-service", "pay-3", round_index, timestamp, r3))

    # inventory-service: sparse rounds and a no-observation round for inv-2
    inventory_rows = [
        (0, 100, OBS_SUCCESS, OBS_SUCCESS),
        (1, 200, OBS_SUCCESS, OBS_SUCCESS),
        (2, 300, OBS_SUCCESS, OBS_TIMEOUT),
        (4, 500, OBS_REFUSED, OBS_SUCCESS),
        (5, 600, OBS_ERROR, OBS_SUCCESS),
        (6, 700, OBS_SUCCESS, OBS_SUCCESS),
    ]
    for round_index, timestamp, r1, r2 in inventory_rows:
        observations.append(_obs("inventory-service", "inv-1", round_index, timestamp, r1))
        observations.append(_obs("inventory-service", "inv-2", round_index, timestamp, r2))

    return {
        "scene_id": "dr-demo-scene",
        "scene_version": 1,
        "services": services,
        "instances": instances,
        "observations": observations,
    }


def build_scale_scene(service_count: int = 12, instances_per_service: int = 4, rounds: int = 10) -> dict[str, Any]:
    services = []
    instances = []
    observations = []
    for service_index in range(service_count):
        service_id = f"svc-{service_index:03d}"
        services.append(
            {
                "service_id": service_id,
                "strategy": "auto" if service_index % 2 == 0 else "manual",
                "cooldown_ms": 800,
                "min_healthy_replicas": 1,
                "success_threshold": 2,
                "failure_threshold": 2,
            }
        )
        for instance_index in range(instances_per_service):
            instance_id = f"{service_id}-i{instance_index}"
            instances.append(
                {
                    "service_id": service_id,
                    "instance_id": instance_id,
                    "role": ROLE_PRIMARY if instance_index == 0 else ROLE_REPLICA,
                    "version": 1 + (instance_index % 3),
                }
            )
            for round_index in range(rounds):
                if (service_index + instance_index + round_index) % 7 == 0:
                    continue
                result = (
                    OBS_REFUSED
                    if (service_index * 3 + round_index) % 5 == 0 and instance_index == 0
                    else OBS_SUCCESS
                )
                observations.append(
                    _obs(
                        service_id,
                        instance_id,
                        round_index,
                        round_index * 100 + instance_index,
                        result,
                    )
                )
    return {
        "scene_id": f"scale-{service_count}x{instances_per_service}x{rounds}",
        "scene_version": 1,
        "services": services,
        "instances": instances,
        "observations": observations,
    }


def write_samples(output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, payload in (
        ("demo_scene.json", build_demo_scene()),
        ("scale_scene.json", build_scale_scene()),
    ):
        target = output / name
        target.write_text(canonical_text(payload) + "\n", encoding="utf-8")
        paths[name] = target
    return paths
