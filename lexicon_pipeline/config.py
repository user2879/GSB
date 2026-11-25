"""Local configuration loading and hashing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .hashing import fingerprint


DEFAULT_CONFIG: dict[str, Any] = {
    "concurrency": 4,
    "fixed_point": {
        "fraction_bits": 10,
        "quantization_tier": "standard",
        "tier_shifts": {"coarse": -2, "standard": 0, "fine": 2},
        "rounding_mode": "ROUND_HALF_UP",
        "boundary_threshold": 0.0,
        "score_temperature": 1.0,
    },
    "match_budget_points": 9000,
    "doc_quota_points": 2600,
    "jv1_widths": {"doc_seq": 4, "fragment_seq": 4, "score_permille": 5, "pos_code": 4},
    "snapshot_dir": "artifacts/snapshot",
    "cache_path": "artifacts/pipeline-state/cache.json",
    "ledger_path": "artifacts/ledger.jsonl",
    "report_path": "artifacts/diagnostic-report.json",
}


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config = json.loads(json.dumps(DEFAULT_CONFIG))
    if path is not None and Path(path).exists():
        loaded = json.loads(Path(path).read_text(encoding="utf-8"))
        _deep_update(config, loaded)
    if not isinstance(config["concurrency"], int) or config["concurrency"] < 1:
        raise ValueError("concurrency must be a positive integer")
    fp = config["fixed_point"]
    bits = fp["fraction_bits"] + fp["tier_shifts"][fp["quantization_tier"]]
    if bits < 0:
        raise ValueError("effective fixed-point width must be non-negative")
    config["effective_fraction_bits"] = bits
    config["fixed_point_scale"] = 1 << bits
    config["high_precision_bits"] = bits + 1
    config["high_precision_scale"] = 1 << (bits + 1)
    config["config_hash"] = fingerprint(
        {
            "concurrency": config["concurrency"],
            "fixed_point": fp,
            "match_budget_points": config["match_budget_points"],
            "doc_quota_points": config["doc_quota_points"],
            "jv1_widths": config["jv1_widths"],
        },
        20,
    )
    return config


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> None:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
