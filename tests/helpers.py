from __future__ import annotations

import json
import socket
import tempfile
from pathlib import Path
from typing import Any


def reserve_ports(count: int) -> list[int]:
    sockets: list[socket.socket] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
        return [int(sock.getsockname()[1]) for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def base_spec(ports: list[int]) -> dict[str, Any]:
    return {
        "revision": 1,
        "limits": {
            "global_concurrency": 2,
            "backend_concurrency": {
                "stable": 1,
                "canary": 1,
            },
        },
        "backends": [
            {
                "id": "stable",
                "host": "127.0.0.1",
                "port": ports[0],
                "model_dim": 4,
                "hidden_dim": 3,
                "seed": 11,
                "quantization": "int8",
            },
            {
                "id": "canary",
                "host": "127.0.0.1",
                "port": ports[1],
                "model_dim": 4,
                "hidden_dim": 5,
                "seed": 23,
                "quantization": "int4",
            },
        ],
        "rules": [
            {
                "id": "alpha",
                "prefix": "alpha:",
                "stable_backend": "stable",
                "canary_backend": "canary",
                "canary_weight_bp": 5000,
                "batch_size": 2,
                "version": 1,
            }
        ],
    }


def sample_requests(count: int = 6) -> list[dict[str, Any]]:
    return [
        {
            "request_id": index + 1,
            "client_key": f"alpha:client-{index + 1:03d}",
            "input": [0.1 * index, -0.2, 0.3, 0.4 + 0.01 * index],
            "arrival_ms": 1000 + index,
        }
        for index in range(count)
    ]


class Workspace:
    def __init__(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="gateway-diag-test-")
        self.root = Path(self.temporary.name)

    def write_spec(self, value: dict[str, Any], name: str = "spec.json") -> Path:
        path = self.root / name
        path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def write_requests(self, values: list[dict[str, Any]], name: str = "requests.ndjson") -> Path:
        path = self.root / name
        path.write_text(
            "\n".join(json.dumps(value, ensure_ascii=False) for value in values) + "\n",
            encoding="utf-8",
        )
        return path

    @property
    def db(self) -> Path:
        return self.root / "ledger.sqlite3"

    def close(self) -> None:
        self.temporary.cleanup()
