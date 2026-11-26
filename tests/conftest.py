from __future__ import annotations

import json
import socket
from dataclasses import dataclass
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "samples"


def free_ports(count: int = 2) -> list[int]:
    sockets: list[socket.socket] = []
    ports: list[int] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
            ports.append(sock.getsockname()[1])
    finally:
        for sock in sockets:
            sock.close()
    return ports


@dataclass
class SpecFiles:
    root: Path
    shards: Path
    layers: Path
    checkpoints: Path
    events: Path
    edited_layers: Path
    ports: list[int]


def _write(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


@pytest.fixture
def spec_files(tmp_path: Path) -> SpecFiles:
    ports = free_ports(2)
    shards = json.loads((SAMPLES / "shards.json").read_text(encoding="utf-8"))
    shards["shards"][0]["port"] = ports[0]
    shards["shards"][1]["port"] = ports[1]
    shard_path = tmp_path / "shards.json"
    _write(shard_path, shards)
    layers_path = tmp_path / "layers.json"
    _write(layers_path, json.loads((SAMPLES / "layers.json").read_text(encoding="utf-8")))
    checkpoints_path = tmp_path / "checkpoints.json"
    _write(
        checkpoints_path,
        json.loads((SAMPLES / "checkpoints.json").read_text(encoding="utf-8")),
    )
    events_path = tmp_path / "events.json"
    _write(events_path, json.loads((SAMPLES / "events.json").read_text(encoding="utf-8")))
    edited_path = tmp_path / "layers_edited.json"
    _write(
        edited_path,
        json.loads((SAMPLES / "layers_edited.json").read_text(encoding="utf-8")),
    )
    return SpecFiles(
        root=tmp_path,
        shards=shard_path,
        layers=layers_path,
        checkpoints=checkpoints_path,
        events=events_path,
        edited_layers=edited_path,
        ports=ports,
    )

