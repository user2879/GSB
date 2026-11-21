"""Deterministic demo dataset and pipeline declarations."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np

DEMO_SEED = 20261122
DATASET_FORMAT = "mlpipe.dataset.v1"
PIPELINE_FORMAT = "mlpipe.pipeline.v1"


def _float_text(value: float, places: int = 6) -> str:
    text = f"{value:.{places}f}".rstrip("0").rstrip(".")
    return text if text not in ("", "-0") else "0"


def _nulls(length: int, period: int, offset: int = 0) -> list[int]:
    return [1 if (index + offset) % period == 0 else 0 for index in range(length)]


def build_dataset(seed: int = DEMO_SEED) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    event_rows = 83
    profile_rows = 54

    event_ids = np.arange(1000, 1000 + event_rows, dtype=np.int64)
    amounts = np.exp(rng.normal(3.25, 0.42, event_rows))
    latencies = rng.gamma(2.1, 7.5, event_rows)
    event_amount_nulls = _nulls(event_rows, 11, 3)
    event_latency_nulls = _nulls(event_rows, 17, 5)
    event_id_nulls = [0] * event_rows

    profile_ids = np.arange(5000, 5000 + profile_rows, dtype=np.int64)
    risk_scores = rng.uniform(5.0, 95.0, profile_rows)
    tenure_days = rng.integers(1, 3650, profile_rows, dtype=np.int64)
    risk_nulls = _nulls(profile_rows, 13, 7)
    tenure_nulls = _nulls(profile_rows, 19, 9)
    profile_id_nulls = [0] * profile_rows

    def int_values(values: np.ndarray, nulls: list[int]) -> list[int]:
        return [0 if nulls[index] else int(value) for index, value in enumerate(values)]

    def float_values(values: np.ndarray, nulls: list[int]) -> list[str]:
        return [
            "0" if nulls[index] else _float_text(float(value))
            for index, value in enumerate(values)
        ]

    return {
        "format": DATASET_FORMAT,
        "generator": {
            "name": "numpy.default_rng",
            "seed": seed,
            "description": "deterministic local demo sample",
        },
        "sources": [
            {
                "id": "events",
                "rows": event_rows,
                "columns": [
                    {
                        "name": "event_id",
                        "type": "int",
                        "values": int_values(event_ids, event_id_nulls),
                        "nulls": event_id_nulls,
                    },
                    {
                        "name": "amount",
                        "type": "float",
                        "values": float_values(amounts, event_amount_nulls),
                        "nulls": event_amount_nulls,
                    },
                    {
                        "name": "latency_ms",
                        "type": "float",
                        "values": float_values(latencies, event_latency_nulls),
                        "nulls": event_latency_nulls,
                    },
                ],
            },
            {
                "id": "profiles",
                "rows": profile_rows,
                "columns": [
                    {
                        "name": "profile_id",
                        "type": "int",
                        "values": int_values(profile_ids, profile_id_nulls),
                        "nulls": profile_id_nulls,
                    },
                    {
                        "name": "risk_score",
                        "type": "float",
                        "values": float_values(risk_scores, risk_nulls),
                        "nulls": risk_nulls,
                    },
                    {
                        "name": "tenure_days",
                        "type": "int",
                        "values": int_values(tenure_days, tenure_nulls),
                        "nulls": tenure_nulls,
                    },
                ],
            },
        ],
    }


def build_pipeline(fill_version: int = 1) -> dict[str, Any]:
    return {
        "format": PIPELINE_FORMAT,
        "fixed_scale": 3,
        "sharding": {"target_rows": 32, "remainder_policy": "keep_partial"},
        "stages": [
            {
                "id": "canonicalize",
                "type": "canonicalize",
                "version": 1,
                "upstream": None,
                "params": {"saturate": 1},
            },
            {
                "id": "fill_missing",
                "type": "fill_missing",
                "version": fill_version,
                "upstream": "canonicalize",
                "params": {"fill_fixed": 0},
            },
            {
                "id": "normalize_amount",
                "type": "normalize",
                "version": 1,
                "upstream": "fill_missing",
                "params": {"column": 1},
            },
            {
                "id": "clip_amount",
                "type": "clip_outliers",
                "version": 1,
                "upstream": "normalize_amount",
                "params": {
                    "column": 1,
                    "z_limit_permille": 2200,
                    "max_outlier_permille": 100,
                },
            },
            {
                "id": "pack",
                "type": "shard_pack",
                "version": 1,
                "upstream": "clip_amount",
                "params": {"remainder_policy": 1},
            },
        ],
    }


def build_evolved_dataset(seed: int = DEMO_SEED) -> dict[str, Any]:
    dataset = build_dataset(seed)
    amount_column = dataset["sources"][0]["columns"][1]
    row = 50
    while amount_column["nulls"][row]:
        row += 1
    current = float(amount_column["values"][row])
    amount_column["values"][row] = _float_text(current + 0.125)
    dataset["generator"]["description"] = "one mid-range amount value changed"
    return dataset


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_demo_inputs(root: str | Path, evolved: bool = False) -> tuple[Path, Path]:
    root = Path(root)
    if evolved:
        dataset = build_evolved_dataset()
        pipeline = build_pipeline(fill_version=2)
        data_path = root / "sample_evolved.json"
        pipeline_path = root / "pipeline_evolved.json"
    else:
        dataset = build_dataset()
        pipeline = build_pipeline()
        data_path = root / "sample.json"
        pipeline_path = root / "pipeline.json"
    write_json(data_path, dataset)
    write_json(pipeline_path, pipeline)
    return data_path, pipeline_path


def corrupt_sample(dataset: dict[str, Any]) -> dict[str, Any]:
    corrupted = copy.deepcopy(dataset)
    corrupted["sources"][0]["columns"][0]["nulls"].append(1)
    return corrupted


def out_of_bounds_pipeline(pipeline: dict[str, Any]) -> dict[str, Any]:
    broken = copy.deepcopy(pipeline)
    broken["stages"][3]["params"]["max_outlier_permille"] = 1001
    return broken


def amplify_outliers(
    dataset: dict[str, Any], column_index: int, permille: int
) -> dict[str, Any]:
    """Inject extreme values into the flattened target column.

    This modifies only the JSON sample payload; no workbench source is changed.
    """

    amplified = copy.deepcopy(dataset)
    columns = [column for source in amplified["sources"] for column in source["columns"]]
    column = columns[column_index]
    eligible = [index for index, null in enumerate(column["nulls"]) if not null]
    count = max(1, len(eligible) * permille // 1000)
    for row in eligible[:count]:
        column["values"][row] = "999999.000"
    amplified["generator"]["description"] = (
        f"outlier amplification: {count} rows in flattened column {column_index}"
    )
    return amplified


def materialise_demo_files(root: str | Path, force: bool = False) -> dict[str, str]:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    data_path = root / "sample.json"
    pipeline_path = root / "pipeline.json"
    if force or not data_path.exists() or not pipeline_path.exists():
        write_demo_inputs(root, evolved=False)
    return {"data": str(data_path), "pipeline": str(pipeline_path)}
