from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

from .model import PLUGIN_NAMES
from .records import NormalizationParams, NormalizedRecord, encode_v1, encode_v3
from .util import atomic_write_text, canonical_json


def find_free_port_base(group_count: int, *, start: int = 39100) -> int:
    if group_count < 1:
        raise ValueError("group_count must be positive")
    for base in range(start, 64000 - group_count):
        sockets: list[socket.socket] = []
        try:
            for offset in range(group_count):
                probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", base + offset))
                sockets.append(probe)
            return base
        except OSError:
            continue
        finally:
            for probe in sockets:
                probe.close()
    raise OSError("no consecutive free loopback ports found")


def sample_spec(parameter_version: int = 1, port_base: int | None = None) -> dict[str, Any]:
    if port_base is None:
        port_base = find_free_port_base(2)
    if parameter_version == 2:
        weights = {
            "taint_toleration": 2000,
            "node_affinity": 2500,
            "pod_anti_affinity": 2500,
            "resource_fit": 3000,
        }
    else:
        weights = {
            "taint_toleration": 2000,
            "node_affinity": 3500,
            "pod_anti_affinity": 1500,
            "resource_fit": 3000,
        }
    return {
        "parameter_version": parameter_version,
        "scale_exponent": 3,
        "plugins": {
            name: {"weight_myriad": weights[name]} for name in PLUGIN_NAMES
        },
        "nodes": [
            {
                "name": "node-a",
                "labels": {"zone": "east", "rack": "r1", "pool": "general"},
                "allocatable": {"cpu": 8, "memory": 16384},
                "taints": [],
            },
            {
                "name": "node-b",
                "labels": {"zone": "east", "rack": "r2", "pool": "general"},
                "allocatable": {"cpu": 8, "memory": 8192},
                "taints": [{"key": "dedicated", "value": "ml", "effect": 0}],
            },
            {
                "name": "node-c",
                "labels": {"zone": "west", "rack": "r3", "pool": "gpu"},
                "allocatable": {"cpu": 4, "memory": 4096},
                "taints": [{"key": "maintenance", "value": "ops", "effect": 1}],
            },
            {
                "name": "node-d",
                "labels": {"zone": "west", "rack": "r4", "pool": "gpu-expanded"},
                "allocatable": {"cpu": 2, "memory": 2048},
                "taints": [],
            },
        ],
        "pods": [
            {
                "id": "pod-0001",
                "labels": {"app": "web", "workload": "api"},
                "requests": {"cpu": 2, "memory": 1024},
                "node_affinity": {
                    "hard": [{"key": "zone", "operator": "In", "values": ["east"]}],
                    "soft": [{"key": "zone", "operator": "In", "values": ["east"], "weight": 2}],
                },
            },
            {
                "id": "pod-0002",
                "labels": {"app": "web", "workload": "worker"},
                "requests": {"cpu": 2, "memory": 2048},
                "tolerations": [
                    {"key": "dedicated", "operator": "Equal", "value": "ml", "effect": 0}
                ],
            },
            {
                "id": "pod-0003",
                "labels": {"app": "cache", "workload": "cache"},
                "requests": {"cpu": 1, "memory": 512},
                "pod_anti_affinity": {
                    "soft": [
                        {
                            "label_key": "app",
                            "operator": "In",
                            "values": ["web"],
                            "topology_key": "rack",
                            "weight": 1,
                        }
                    ]
                },
            },
            {
                "id": "pod-0004",
                "labels": {"app": "batch", "workload": "batch"},
                "requests": {"cpu": 1, "memory": 512},
                "node_affinity": {
                    "soft": [{"key": "zone", "operator": "In", "values": ["west"], "weight": 1}]
                },
            },
        ],
        "service": {
            "host": "127.0.0.1",
            "port_base": port_base,
            "timeout_ms": 2500,
            "groups": [
                {"name": "group-east", "nodes": ["node-a", "node-b"]},
                {"name": "group-west", "nodes": ["node-c", "node-d"]},
            ],
        },
    }


def _sample_records(port_base: int) -> list[NormalizedRecord]:
    params = NormalizationParams(
        parameter_version=1,
        scale_exponent=3,
        weights_myriad={
            "taint_toleration": 2000,
            "node_affinity": 3500,
            "pod_anti_affinity": 1500,
            "resource_fit": 3000,
        },
    )
    base = {
        "parameter_version": 1,
        "scale_exponent": 3,
        "scale_factor": 1000,
        "tie_break": ("taint_toleration", "node_affinity", "pod_anti_affinity", "resource_fit", "node_name"),
        "completion_basis": {},
        "source_total_score": 0,
    }
    del port_base
    return [
        NormalizedRecord(
            source_format_version=1,
            source_line=1,
            pod_id="pod-0001",
            node_name="node-a",
            total_score=84,
            veto_code=0,
            plugin_scores={name: 0 for name in PLUGIN_NAMES},
            plugin_contributions={name: 0 for name in PLUGIN_NAMES},
            plugin_weights_myriad=dict(params.weights_myriad),
            parameter_version=1,
            scale_exponent=3,
            scale_factor=1000,
            error_bound=0,
            tie_break=base["tie_break"],
            filled_fields=(),
            completion_basis={},
            source_total_score=84,
        ),
        NormalizedRecord(
            source_format_version=2,
            source_line=2,
            pod_id="pod-0002",
            node_name="node-b",
            total_score=77,
            veto_code=0,
            plugin_scores={
                "taint_toleration": 100,
                "node_affinity": 0,
                "pod_anti_affinity": 0,
                "resource_fit": 75,
            },
            plugin_contributions={
                name: 0 for name in PLUGIN_NAMES
            },
            plugin_weights_myriad=dict(params.weights_myriad),
            parameter_version=1,
            scale_exponent=3,
            scale_factor=1000,
            error_bound=2,
            tie_break=base["tie_break"],
            filled_fields=(),
            completion_basis={},
            source_total_score=77.25,
        ),
        NormalizedRecord(
            source_format_version=3,
            source_line=3,
            pod_id="pod-0003",
            node_name="node-c",
            total_score=73,
            veto_code=0,
            plugin_scores={
                "taint_toleration": 100,
                "node_affinity": 0,
                "pod_anti_affinity": 80,
                "resource_fit": 75,
            },
            plugin_contributions={
                "taint_toleration": 20000,
                "node_affinity": 0,
                "pod_anti_affinity": 12000,
                "resource_fit": 22500,
            },
            plugin_weights_myriad=dict(params.weights_myriad),
            parameter_version=1,
            scale_exponent=3,
            scale_factor=1000,
            error_bound=1,
            tie_break=base["tie_break"],
            filled_fields=(),
            completion_basis={},
            source_total_score=73,
        ),
    ]


def write_sample(root: str | Path, *, port_base: int | None = None) -> dict[str, Any]:
    root_path = Path(root)
    (root_path / "config").mkdir(parents=True, exist_ok=True)
    (root_path / "logs").mkdir(parents=True, exist_ok=True)
    (root_path / "data").mkdir(parents=True, exist_ok=True)
    if port_base is None:
        port_base = find_free_port_base(2)
    spec = sample_spec(1, port_base)
    spec_v2 = sample_spec(2, port_base)
    atomic_write_text(
        root_path / "config" / "spec.json",
        json.dumps(spec, indent=2, ensure_ascii=False) + "\n",
    )
    atomic_write_text(
        root_path / "config" / "spec-v2.json",
        json.dumps(spec_v2, indent=2, ensure_ascii=False) + "\n",
    )
    records = _sample_records(port_base)
    lines = [
        encode_v1(records[0]),
        json.dumps(records[1].to_v2_dict(), sort_keys=True, ensure_ascii=False),
        encode_v3(records[2]),
    ]
    atomic_write_text(
        root_path / "data" / "records.jsonl",
        "\n".join(lines) + "\n",
    )
    return {
        "root": str(root_path),
        "spec": str(root_path / "config" / "spec.json"),
        "spec_v2": str(root_path / "config" / "spec-v2.json"),
        "records": str(root_path / "data" / "records.jsonl"),
        "port_base": port_base,
    }
