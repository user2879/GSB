"""Deterministic, self-contained sample workspace generation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


REGIONS = ["north", "south", "east", "west", "central", "unmapped"]
CITIES_BY_REGION = {
    "north": ["hill", "river"],
    "south": ["plain", "delta"],
    "east": ["harbor", "metro"],
    "west": ["river", "harbor"],
    "central": ["metro", "plain"],
}
PLANS = ["basic", "plus", "pro"]


def _bucket(*parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _is_missing(seed: int, row_id: str, column: str, rate: float) -> bool:
    return (_bucket(seed, "missing", row_id, column) % 10_000) < int(rate * 10_000)


def _cell(value: Any, missing: bool) -> Any:
    return {"missing": True} if missing else value


def _base_row(seed: int, index: int) -> dict[str, Any]:
    row_id = f"r{index:04d}"
    region = REGIONS[(index * 7 + seed) % 4]  # base rows never use central/unmapped
    city_options = CITIES_BY_REGION[region]
    city = city_options[(index + seed) % len(city_options)]
    age = round(22.0 + ((index * 13 + seed) % 37) + ((index % 4) * 0.25), 2)
    income = round(38.0 + ((index * 19 + seed) % 104) + ((index % 7) * 0.125), 3)
    score = round(0.55 * age + 0.32 * income + ((index * 11) % 13) * 0.7 - 18.0, 3)
    plan = PLANS[(index * 5 + seed) % 3]
    logit = -5.4 + 0.055 * age + 0.033 * income + 0.028 * score + (0.55 if plan == "pro" else 0.0)
    churn = 1 if logit > 0 else 0
    return {
        "row_id": row_id,
        "age": _cell(age, _is_missing(seed, row_id, "age", 0.18)),
        "income": _cell(income, _is_missing(seed, row_id, "income", 0.22)),
        "score": _cell(score, _is_missing(seed, row_id, "score", 0.20)),
        "city": _cell(city, _is_missing(seed, row_id, "city", 0.15)),
        "region": region,
        "plan": _cell(plan, _is_missing(seed, row_id, "plan", 0.10)),
        "churn": churn,
    }


def _config(seed: int) -> dict[str, Any]:
    return {
        "format": 1,
        "seed": seed,
        "id_column": "row_id",
        "target_column": "churn",
        "decimal_places": 2,
        "quantization_level": 5,
        "rounding": "half_even",
        "thread_pool_workers": 4,
        "batch_cell_budget": 12,
        "snapshot_interval_batches": 1,
        "validation_fraction": 0.25,
        "model": {
            "learning_rate": 0.12,
            "epochs": 36,
            "batch_size": 4,
            "l2": 0.002,
        },
        "columns": {
            "age": {
                "type": "numeric",
                "imputer": "median",
                "min_valid_samples": 3,
                "fallback": 33.0,
            },
            "income": {
                "type": "numeric",
                "imputer": "mean_noise",
                "noise_coefficient": 0.22,
                "min_valid_samples": 3,
                "fallback": 60.0,
            },
            "score": {
                "type": "numeric",
                "imputer": "grouped_median",
                "group_key": "region",
                "min_valid_samples": 3,
                "fallback": 55.0,
            },
            "city": {
                "type": "category",
                "imputer": "mode",
                "domain": ["delta", "harbor", "hill", "metro", "plain", "river"],
                "min_valid_samples": 3,
                "fallback": "river",
            },
            "region": {
                "type": "category",
                "imputer": "constant",
                "domain": REGIONS,
                "min_valid_samples": 1,
                "constant": "north",
                "fallback": "north",
            },
            "plan": {
                "type": "category",
                "imputer": "constant",
                "domain": PLANS,
                "min_valid_samples": 1,
                "constant": "basic",
                "fallback": "basic",
            },
            "churn": {
                "type": "numeric",
                "imputer": "constant",
                "min_valid_samples": 1,
                "constant": 0,
                "fallback": 0,
            },
        },
    }


def _group_map() -> dict[str, Any]:
    return {
        "format": 1,
        "key_column": "region",
        "mapping": {
            "north": "inland",
            "south": "inland",
            "east": "coastal",
            "west": "coastal",
        },
        "groups": ["inland", "coastal"],
    }


def _edit_plan() -> dict[str, Any]:
    return {
        "format": 1,
        "batches": [
            {
                "batch_id": "b1-append",
                "operations": [
                    {
                        "op": "append_row",
                        "row": {
                            "row_id": "r0025",
                            "age": {"missing": True},
                            "income": 77.625,
                            "score": {"missing": True},
                            "city": "plain",
                            "region": "north",
                            "plan": "plus",
                            "churn": 1,
                        },
                    }
                ],
            },
            {
                "batch_id": "b2-correct",
                "operations": [
                    {
                        "op": "correct_row",
                        "row_id": "r0003",
                        "changes": {"age": 48.5, "income": {"missing": True}},
                    }
                ],
            },
            {
                "batch_id": "b3-imputer-change",
                "operations": [
                    {
                        "op": "change_imputer",
                        "column": "income",
                        "changes": {"imputer": "median"},
                    }
                ],
            },
            {
                "batch_id": "b4-group-add",
                "operations": [
                    {
                        "op": "change_group_mapping",
                        "key_column": "region",
                        "changes": {"central": "highland"},
                    },
                    {
                        "op": "append_row",
                        "row": {
                            "row_id": "r0026",
                            "age": 36.25,
                            "income": {"missing": True},
                            "score": {"missing": True},
                            "city": "delta",
                            "region": "central",
                            "plan": "basic",
                            "churn": 0,
                        },
                    },
                ],
            },
        ],
    }


def _injections() -> dict[str, Any]:
    return {
        "format": 1,
        "items": [
            {
                "id": "INJ-001",
                "stage": "load",
                "kind": "non_finite_value",
                "row_id": "r0007",
                "column": "age",
                "business_code": "E_DATA_ROW_INVALID",
            },
            {
                "id": "INJ-002",
                "stage": "group_mapping",
                "kind": "unmapped_group_value",
                "row_id": "r0014",
                "column": "region",
                "value": "unmapped",
                "business_code": "E_GROUP_MAP_INVALID",
            },
            {
                "id": "INJ-003",
                "stage": "config",
                "kind": "decimal_width_out_of_range",
                "target": "columns.income.decimal_places",
                "value": 99,
                "business_code": "E_CONFIG_INVALID",
            },
            {
                "id": "INJ-004",
                "stage": "config",
                "kind": "min_valid_samples_out_of_range",
                "target": "columns.age.min_valid_samples",
                "value": 0,
                "business_code": "E_CONFIG_INVALID",
            },
        ],
    }


def generate_sample(out_dir: Path, seed: int = 20261121, rows: int = 24) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    config = _config(seed)
    with (out_dir / "imputation.yaml").open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)
    with (out_dir / "raw_data.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(1, rows + 1):
            handle.write(json.dumps(_base_row(seed, index), ensure_ascii=False, sort_keys=True) + "\n")
    for filename, payload in (
        ("group_map.json", _group_map()),
        ("edit_plan.json", _edit_plan()),
        ("injections.json", _injections()),
    ):
        with (out_dir / filename).open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.write("\n")
