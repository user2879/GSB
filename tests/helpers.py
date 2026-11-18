from __future__ import annotations

from typing import Any


def config_payload(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": 1,
        "memory_budget_mib": 1,
        "block_size_kb": 16,
        "precision": "fixed",
        "quant_scale_power": 4,
        "batch_size_limit": 4,
        "preallocation": "block_pool",
        "budget_rounding": "reject",
        "pressure_threshold_ms": 25.0,
        "ui_query_budget_ms": 16.0,
    }
    payload.update(overrides)
    return payload
