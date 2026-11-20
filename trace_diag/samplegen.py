from __future__ import annotations

from pathlib import Path
from typing import Any

from .canonical import write_json_atomic


def build_default_services(record_count: int = 600) -> dict[str, Any]:
    fields = {
        "timestamp_ms": {"type": "int", "start": 1_735_689_600_000, "max_step_ms": 41},
        "magnitude": {"type": "int", "min": 1, "max": 5},
        "client_key": {"type": "string", "key_space": 173},
        "duration_us": {"type": "int", "min_us": 900, "max_us": 180_000},
        "trace_id": {"type": "string"},
        "sampled": {"type": "bool", "rate_bp": 2_000},
    }
    return {
        "version": 1,
        "description": "Deterministic local latency and trace sample declaration",
        "services": [
            {
                "id": "api-checkout",
                "description": "Checkout API near-tie benchmark A",
                "seed": 1103,
                "record_count": record_count,
                "fields": fields,
            },
            {
                "id": "api-catalog",
                "description": "Catalog API near-tie benchmark B",
                "seed": 2207,
                "record_count": record_count,
                "fields": {
                    **fields,
                    "timestamp_ms": {
                        "type": "int",
                        "start": 1_735_689_600_500,
                        "max_step_ms": 39,
                    },
                    "client_key": {"type": "string", "key_space": 181},
                    "duration_us": {"type": "int", "min_us": 1_000, "max_us": 182_000},
                },
            },
            {
                "id": "worker-ledger",
                "description": "Background ledger worker with a wider tail",
                "seed": 3301,
                "record_count": record_count,
                "fields": {
                    **fields,
                    "timestamp_ms": {
                        "type": "int",
                        "start": 1_735_689_600_900,
                        "max_step_ms": 53,
                    },
                    "magnitude": {"type": "int", "min": 2, "max": 7},
                    "client_key": {"type": "string", "key_space": 211},
                    "duration_us": {"type": "int", "min_us": 1_200, "max_us": 300_000},
                },
            },
        ],
    }


def build_default_policies() -> dict[str, Any]:
    return {
        "version": 1,
        "description": "Two deterministic head/tail candidate strategies",
        "policies": [
            {
                "id": "head-balanced",
                "head_ratio_bp": 4_200,
                "tail_duration_threshold_us": 50_000,
                "tail_ratio_bp": 1_800,
                "scale_power": 1,
                "bucket_count": 32,
                "description": "More head coverage, moderate tail catch-up",
            },
            {
                "id": "tail-first",
                "head_ratio_bp": 2_600,
                "tail_duration_threshold_us": 35_000,
                "tail_ratio_bp": 3_200,
                "scale_power": 1,
                "bucket_count": 32,
                "description": "Lean head sampling with stronger tail sampling",
            },
        ],
    }


def generate_sample_files(
    data_dir: str | Path,
    *,
    record_count: int = 600,
) -> dict[str, Path]:
    directory = Path(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    services_path = directory / "services.json"
    policies_path = directory / "policies.json"
    write_json_atomic(services_path, build_default_services(record_count))
    write_json_atomic(policies_path, build_default_policies())
    (directory / "ledger").mkdir(parents=True, exist_ok=True)
    (directory / "logs").mkdir(parents=True, exist_ok=True)
    return {"services": services_path, "policies": policies_path}
