from __future__ import annotations

import json
import socket
from pathlib import Path
from typing import Any

from .specs import SpecBundle, load_bundle


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def make_sample_configs(
    config_dir: str | Path,
    *,
    request_count: int = 4,
    batch_size: int = 2,
    seed: int = 20261126,
    failure_ratio: float = 0.0,
    fwd_scale: int = 1,
    rev_scale: int = 1000,
    session_scale: int = 1000,
) -> SpecBundle:
    config_dir = Path(config_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    fwd_port = free_port()
    rev_port = free_port()
    origin_port = free_port()

    chain = {
        "format": "hopledger.chain/1",
        "hop_set_version": 1,
        "client_id": "client",
        "origin_id": "origin",
        "allowed_scale_factors": [1, 10, 100, 1000, 10000],
        "hops": [
            {
                "hop_id": "fwd1",
                "port": fwd_port,
                "role": "forward",
                "config_version": 1,
                "queue_ms": 1,
                "process_ms": 1,
                "forward_estimate_ms": 1,
                "failure_ratio": failure_ratio,
                "scale_factor": fwd_scale,
            },
            {
                "hop_id": "rev2",
                "port": rev_port,
                "role": "reverse",
                "config_version": 1,
                "queue_ms": 1,
                "process_ms": 1,
                "forward_estimate_ms": 1,
                "failure_ratio": failure_ratio,
                "scale_factor": rev_scale,
            },
        ],
    }
    origin = {
        "format": "hopledger.origin/1",
        "origin_id": "origin",
        "config_version": 1,
        "port": origin_port,
        "seed": seed,
        "body_bytes": 64,
        "samples": [
            {
                "request_id": f"req-{index:04d}",
                "path": f"/samples/{index - 1}",
                "seed_offset": index - 1,
            }
            for index in range(1, request_count + 1)
        ],
    }
    diagnostic = {
        "diagnostic_id": "local-loopback-deterministic",
        "format_version": 1,
        "scale_caliber_version": 1,
        "session_scale": session_scale,
        "timeout_seconds": 3.0,
        "handshake_timeout_seconds": 1.0,
        "batch_size": batch_size,
        "scheduler": {
            "max_in_flight": 2,
            "hop_quota": 1,
            "pool_workers": 4,
        },
        "error_model": {
            "dominance_threshold_bp": 6000,
            "indistinguishable_margin_bp": 500,
            "high_failure_ratio_bp": 5000,
        },
    }
    write_json(config_dir / "chain.json", chain)
    write_json(config_dir / "origin.json", origin)
    write_json(config_dir / "diagnostic.json", diagnostic)
    return load_bundle(config_dir)
