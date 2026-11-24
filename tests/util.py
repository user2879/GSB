from __future__ import annotations

import copy
import json
import socket
from decimal import Decimal
from pathlib import Path
from typing import Any

from config_arbiter.arbitration import Candidate
from config_arbiter.spec import SessionSpec

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SPEC = ROOT / "samples" / "sample_spec.json"


def sample_object() -> dict[str, Any]:
    return json.loads(SAMPLE_SPEC.read_text(encoding="utf-8"))


def write_temp_spec(directory: Path, payload: dict[str, Any], name: str = "spec.json") -> Path:
    path = directory / name
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def free_ports(count: int) -> list[int]:
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


def ported_sample(directory: Path) -> tuple[Path, dict[str, Any]]:
    payload = copy.deepcopy(sample_object())
    ports = free_ports(len(payload["environments"]))
    for environment, port in zip(payload["environments"], ports):
        environment["port"] = port
    payload["session"].update(
        {
            "sample_count": 16,
            "group_size": 4,
            "noise_bound": 3,
            "global_concurrency": 2,
            "per_environment_concurrency": 1,
            "handshake_timeout_ms": 1200,
        }
    )
    return write_temp_spec(directory, payload), payload


def candidate(
    env_id: str,
    aggregate: int,
    *,
    version: int = 1,
    weight: int = 1000,
    value_error: str = "0",
    weight_error: str = "0",
) -> Candidate:
    return Candidate(
        env_id=env_id,
        slice_version=version,
        aggregate_units=aggregate,
        weight_units=weight,
        value_quantization_error_units=Decimal(value_error),
        weight_quantization_error_units=Decimal(weight_error),
        receipt_digest=f"receipt-{env_id}",
        sample_digest=f"sample-{env_id}",
    )


def session_spec(
    *,
    global_concurrency: int = 2,
    per_environment_concurrency: int = 1,
    handshake_timeout_ms: int = 1000,
) -> SessionSpec:
    return SessionSpec(
        scale_exponent=3,
        seed=7,
        sample_count=16,
        group_size=4,
        noise_bound=3,
        global_concurrency=global_concurrency,
        per_environment_concurrency=per_environment_concurrency,
        handshake_timeout_ms=handshake_timeout_ms,
    )

