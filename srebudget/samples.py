"""Deterministic, offline sample chain and trace."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def sample_chain_data() -> dict[str, Any]:
    return {
        "version": "chain-2026.11.20-r1",
        "services": [
            {
                "id": "gateway",
                "upstream": [],
                "timeout_ms": 2500,
                "budget": {
                    "window_hours": 1,
                    "ratio_bp": 5000,
                    "min_samples": 3,
                },
                "circuit": {
                    "failure_threshold": 3,
                    "cooldown_ms": 2500,
                    "half_open_probes": 1,
                },
            },
            {
                "id": "api",
                "upstream": ["gateway"],
                "timeout_ms": 1800,
                "budget": {
                    "window_hours": 1,
                    "ratio_bp": 5000,
                    "min_samples": 3,
                },
                "circuit": {
                    "failure_threshold": 3,
                    "cooldown_ms": 2000,
                    "half_open_probes": 1,
                },
            },
            {
                "id": "orders",
                "upstream": ["api"],
                "timeout_ms": 1200,
                "budget": {
                    "window_hours": 1,
                    "ratio_bp": 10000,
                    "min_samples": 2,
                },
                "circuit": {
                    "failure_threshold": 2,
                    "cooldown_ms": 2000,
                    "half_open_probes": 1,
                },
            },
            {
                "id": "db",
                "upstream": ["orders"],
                "timeout_ms": 700,
                "budget": {
                    "window_hours": 1,
                    "ratio_bp": 5000,
                    "min_samples": 2,
                },
                "circuit": {
                    "failure_threshold": 4,
                    "cooldown_ms": 1000,
                    "half_open_probes": 1,
                },
            },
        ],
    }


def sample_trace_records() -> list[dict[str, Any]]:
    full = ["db", "orders", "api", "gateway"]
    middle = ["orders", "api", "gateway"]
    return [
        {"timestamp_ms": 1000, "path": full, "outcome": "failure", "duration_ms": 480},
        {"timestamp_ms": 2000, "path": full, "outcome": "failure", "duration_ms": 510},
        {"timestamp_ms": 3000, "path": full, "outcome": "success", "duration_ms": 240},
        {"timestamp_ms": 4000, "path": middle, "outcome": "failure", "duration_ms": 900},
        {"timestamp_ms": 5000, "path": middle, "outcome": "failure", "duration_ms": 940},
        {"timestamp_ms": 9000, "path": middle, "outcome": "success", "duration_ms": 210},
        {"timestamp_ms": 10000, "path": middle, "outcome": "failure", "duration_ms": 890},
        {"timestamp_ms": 11000, "path": middle, "outcome": "failure", "duration_ms": 920},
        {"timestamp_ms": 12000, "path": full, "outcome": "success", "duration_ms": 250},
        {"timestamp_ms": 15000, "path": middle, "outcome": "timeout", "duration_ms": 1200},
        {"timestamp_ms": 19000, "path": middle, "outcome": "success", "duration_ms": 230},
        {"timestamp_ms": 20000, "path": full, "outcome": "success", "duration_ms": 260},
    ]


def write_samples(data_dir: str | Path) -> tuple[Path, Path]:
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    chain_path = directory / "chain.json"
    trace_path = directory / "trace.ndjson"
    chain_text = json.dumps(
        sample_chain_data(), ensure_ascii=False, indent=2, sort_keys=True
    ) + "\n"
    trace_text = "".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for record in sample_trace_records()
    )
    chain_path.write_text(chain_text, encoding="utf-8", newline="\n")
    trace_path.write_text(trace_text, encoding="utf-8", newline="\n")
    return chain_path, trace_path


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
