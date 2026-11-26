from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest


def free_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])
    finally:
        sock.close()


def base_config(ports: list[int]) -> dict:
    return {
        "workers": [
            {
                "worker_id": "wa",
                "host": "127.0.0.1",
                "port": ports[0],
                "concurrency_quota": 2,
                "lease_limit": 2,
                "frac_bits": 8,
                "quant_level": 8,
            },
            {
                "worker_id": "wb",
                "host": "127.0.0.1",
                "port": ports[1],
                "concurrency_quota": 1,
                "lease_limit": 1,
                "frac_bits": 8,
                "quant_level": 8,
            },
        ],
        "operator": {
            "revision": 1,
            "frac_bits": 8,
            "quant_level": 8,
            "layers": [
                {"name": "l1", "rows": 3, "cols": 2, "group_size": 2, "weight_seed": 301},
                {"name": "l2", "rows": 2, "cols": 3, "group_size": 3, "weight_seed": 302},
            ],
        },
        "batches": [
            {"batch_id": "b1", "sample_seed": 401, "batch_size": 2},
            {"batch_id": "b2", "sample_seed": 402, "batch_size": 3},
        ],
    }


def commands() -> dict:
    return {
        "commands": [
            {"seq": 1, "type": "replace_weight", "layer": "l1", "weight_seed": 501, "revision": 2},
            {"seq": 2, "type": "set_quant_level", "value": 16, "revision": 3},
            {"seq": 3, "type": "set_frac_bits", "value": 10, "revision": 4},
            {"seq": 4, "type": "append_batch", "batch": {"batch_id": "b3", "sample_seed": 403, "batch_size": 2}},
        ]
    }


@pytest.fixture()
def local_fixture(tmp_path: Path) -> dict:
    ports = [free_port(), free_port()]
    while ports[1] == ports[0]:
        ports[1] = free_port()
    config_path = tmp_path / "base.json"
    commands_path = tmp_path / "commands.json"
    injections_path = tmp_path / "injections.json"
    config_path.write_text(json.dumps(base_config(ports)), encoding="utf-8")
    commands_path.write_text(json.dumps(commands()), encoding="utf-8")
    injections_path.write_text(
        json.dumps(
            {
                "injections": [
                    {"id": "T-INJ-001", "kind": "dimension_mismatch", "batch_id": "b1"},
                    {"id": "T-INJ-002", "kind": "level_out_of_range", "batch_id": "b1", "value": 99},
                    {"id": "T-INJ-003", "kind": "nonfinite_weight", "batch_id": "b2", "layer": "l1"},
                    {"id": "T-INJ-004", "kind": "stale_generation", "batch_id": "b2"},
                ]
            }
        ),
        encoding="utf-8",
    )
    return {
        "root": tmp_path,
        "config": config_path,
        "commands": commands_path,
        "injections": injections_path,
        "ports": ports,
    }

