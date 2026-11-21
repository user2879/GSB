from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SPANS = ROOT / "sample_data" / "spans.ndjson"


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def make_nodes(
    *,
    ruleset_version: int = 1,
    nodes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if nodes is None:
        ports: list[int] = []
        while len(ports) < 3:
            candidate = free_port()
            if candidate not in ports:
                ports.append(candidate)
        nodes = [
            {
                "node_id": "api",
                "host": "127.0.0.1",
                "port": ports[0],
                "propagation_rule_version": 1,
                "sample_ratio_bps": 10000,
                "random_seed": 101,
            },
            {
                "node_id": "worker",
                "host": "127.0.0.1",
                "port": ports[1],
                "propagation_rule_version": 2,
                "sample_ratio_bps": 0,
                "random_seed": 202,
            },
            {
                "node_id": "static",
                "host": "127.0.0.1",
                "port": ports[2],
                "propagation_rule_version": 5,
                "sample_ratio_bps": 5000,
                "random_seed": 505,
            },
        ]
    return {"format_version": 1, "ruleset_version": ruleset_version, "nodes": nodes}


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_spans(path: Path, spans: list[dict[str, Any]]) -> Path:
    path.write_text(
        "\n".join(json.dumps(span, ensure_ascii=False, separators=(",", ":")) for span in spans)
        + "\n",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def sample_dataset(tmp_path: Path):
    from trace_arbiter.spec import load_dataset

    nodes = write_json(tmp_path / "nodes.json", make_nodes())
    return load_dataset(str(nodes), str(SAMPLE_SPANS))


@pytest.fixture
def evolved_nodes(tmp_path: Path) -> Path:
    base = make_nodes()
    ports = {node["node_id"]: node["port"] for node in base["nodes"]}
    nodes = [
        {
            "node_id": "api",
            "host": "127.0.0.1",
            "port": ports["api"],
            "propagation_rule_version": 3,
            "sample_ratio_bps": 0,
            "random_seed": 303,
        },
        {
            "node_id": "worker",
            "host": "127.0.0.1",
            "port": ports["worker"],
            "propagation_rule_version": 4,
            "sample_ratio_bps": 10000,
            "random_seed": 404,
        },
        {
            "node_id": "static",
            "host": "127.0.0.1",
            "port": ports["static"],
            "propagation_rule_version": 5,
            "sample_ratio_bps": 5000,
            "random_seed": 505,
        },
    ]
    return write_json(tmp_path / "nodes_v2.json", make_nodes(ruleset_version=2, nodes=nodes))
