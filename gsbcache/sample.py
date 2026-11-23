"""Generate the self-contained local acceptance sample."""
from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

from .content import artifact_digest, artifact_content, spec_published_files
from .spec import NodeSpec, parse_spec
from .util import canonical_bytes, write_json_file

SAMPLE_SEED = 17
SAMPLE_ALT_SEED = 23


def _free_loopback_ports(count: int) -> list[int]:
    ports: list[int] = []
    held: list[socket.socket] = []
    try:
        while len(ports) < count:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
            if port not in ports:
                held.append(sock)
                ports.append(port)
            else:
                sock.close()
    finally:
        for sock in held:
            sock.close()
    return ports


def build_sample_spec(base_port: int | None = None) -> dict[str, Any]:
    if base_port is None:
        ports = _free_loopback_ports(3)
    else:
        if base_port <= 0 or base_port + 2 > 65535:
            raise ValueError("base_port must allow three valid ports")
        ports = [base_port, base_port + 1, base_port + 2]
    provisional_nodes = [
        NodeSpec("node-a", "127.0.0.1", ports[0], 2, 1, SAMPLE_SEED),
        NodeSpec("node-b", "127.0.0.1", ports[1], 1, 2, SAMPLE_SEED),
        NodeSpec("node-c", "127.0.0.1", ports[2], 4, 2, SAMPLE_ALT_SEED),
    ]
    by_id = {node.node_id: node for node in provisional_nodes}
    artifact_keys = ["src:main", "src:util", "app:bin", "test:report"]
    refs = {
        "src:main": [],
        "src:util": [],
        "app:bin": ["src:main", "src:util"],
        "test:report": ["app:bin"],
    }

    def artifact(key: str) -> dict[str, Any]:
        return {
            "key": key,
            "expected_digest": artifact_digest(by_id["node-a"], key),
            "refs": refs[key],
        }

    return {
        "format_version": 1,
        "nodes": [
            {
                "node_id": "node-a",
                "host": "127.0.0.1",
                "port": ports[0],
                "capacity_slots": 2,
                "lease_limit": 1,
                "seed": SAMPLE_SEED,
            },
            {
                "node_id": "node-b",
                "host": "127.0.0.1",
                "port": ports[1],
                "capacity_slots": 1,
                "lease_limit": 2,
                "seed": SAMPLE_SEED,
            },
            {
                "node_id": "node-c",
                "host": "127.0.0.1",
                "port": ports[2],
                "capacity_slots": 4,
                "lease_limit": 2,
                "seed": SAMPLE_ALT_SEED,
            },
        ],
        "pipeline": {
            "pipeline_id": "windows-local-cache-demo",
            "global_concurrency": 3,
            "stages": [
                {
                    "stage_id": "prepare",
                    "concurrency": 2,
                    "artifacts": [artifact("src:main"), artifact("src:util")],
                },
                {
                    "stage_id": "compile",
                    "concurrency": 2,
                    "artifacts": [artifact("app:bin")],
                },
                {
                    "stage_id": "test",
                    "concurrency": 1,
                    "artifacts": [artifact("test:report")],
                },
            ],
        },
    }


def create_sample(out_dir: str | Path, base_port: int | None = None) -> dict[str, Any]:
    root = Path(out_dir)
    root.mkdir(parents=True, exist_ok=True)
    raw = build_sample_spec(base_port)
    spec_path = root / "spec.json"
    write_json_file(spec_path, raw)
    spec = parse_spec(raw, source_path=str(spec_path))
    published = spec_published_files(spec, root / "published")
    reusability: list[dict[str, Any]] = []
    for node in spec.nodes:
        for key in spec.all_artifact_keys:
            generated = artifact_digest(node, key)
            reusability.append(
                {
                    "node_id": node.node_id,
                    "artifact_key": key,
                    "expected_digest": spec.artifacts[key].expected_digest,
                    "actual_digest": generated,
                    "reusable": generated == spec.artifacts[key].expected_digest,
                    "reason": (
                        "digest_match"
                        if generated == spec.artifacts[key].expected_digest
                        else "digest_mismatch"
                    ),
                }
            )
    return {
        "ok": True,
        "out_dir": str(root),
        "spec": str(spec_path),
        "spec_digest": spec.digest,
        "ports": [node.port for node in spec.nodes],
        "published": published,
        "reusability": reusability,
        "note": "node-c intentionally has a different seed and is rejected for every manifest artifact",
    }
