#!/usr/bin/env python3
"""Generate the deterministic local sample used by the acceptance checks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


SEED = 20261123
ROW_COUNT = 24
GROUPS = ("g_numeric", "g_sensor", "g_ratio")
COLUMNS = ("age", "income", "score", "temp", "pressure", "latency", "ratio", "load")
ROW_FIELD = "row_id"
GROUP_FIELD = "group_id"
MISSING = "__MISSING__"


def _round(value: float) -> float:
    return float(f"{value:.6f}")


def _base_values(group_id: str, rng: random.Random) -> dict[str, float]:
    if group_id == "g_numeric":
        return {
            "age": 24.0 + rng.gauss(0.0, 9.0),
            "income": 45.0 + abs(rng.gauss(0.0, 28.0)),
            "score": rng.gauss(0.2, 1.7),
            "temp": 18.0 + rng.gauss(0.0, 3.0),
            "pressure": 101.0 + rng.gauss(0.0, 1.2),
            "latency": 0.4 + abs(rng.gauss(0.0, 0.15)),
            "ratio": 0.55 + rng.gauss(0.0, 0.08),
            "load": 42.0 + rng.gauss(0.0, 7.0),
        }
    if group_id == "g_sensor":
        return {
            "age": 47.0 + rng.gauss(0.0, 12.0),
            "income": 110.0 + abs(rng.gauss(0.0, 46.0)),
            "score": -0.8 + rng.gauss(0.0, 0.9),
            "temp": 31.0 + rng.gauss(0.0, 7.0),
            "pressure": 104.0 + rng.gauss(0.0, 2.4),
            "latency": 1.3 + abs(rng.gauss(0.0, 0.55)),
            "ratio": 0.28 + rng.gauss(0.0, 0.05),
            "load": 68.0 + rng.gauss(0.0, 12.0),
        }
    return {
        "age": 36.0 + rng.gauss(0.0, 6.0),
        "income": 75.0 + abs(rng.gauss(0.0, 20.0)),
        "score": 1.4 + rng.gauss(0.0, 2.1),
        "temp": 24.0 + rng.gauss(0.0, 4.0),
        "pressure": 99.0 + rng.gauss(0.0, 1.7),
        "latency": 0.85 + abs(rng.gauss(0.0, 0.3)),
        "ratio": 1.15 + rng.gauss(0.0, 0.28),
        "load": 27.0 + rng.gauss(0.0, 5.5),
    }


def build_rows() -> list[dict[str, Any]]:
    rng = random.Random(SEED)
    rows: list[dict[str, Any]] = []
    for index in range(ROW_COUNT):
        group_id = GROUPS[index % len(GROUPS)]
        values = _base_values(group_id, rng)
        row: dict[str, Any] = {
            ROW_FIELD: f"r{index:04d}",
            GROUP_FIELD: group_id,
        }
        for column in COLUMNS:
            row[column] = _round(values[column])
        rows.append(row)

    # A handful of missing markers and deliberate tail values ensure that both
    # missing-data and saturation branches are exercised.
    rows[2]["income"] = MISSING
    rows[4]["score"] = MISSING
    rows[9]["temp"] = MISSING
    rows[14]["pressure"] = MISSING
    rows[17]["ratio"] = MISSING
    rows[21]["latency"] = MISSING

    rows[0]["income"] = 4200.0
    rows[0]["age"] = -35.0
    rows[1]["temp"] = -18.0
    rows[1]["pressure"] = 130.0
    rows[2]["score"] = 31.0
    rows[6]["score"] = 0.0
    rows[8]["load"] = 0.0
    rows[5]["latency"] = 14.0
    rows[10]["load"] = 210.0
    rows[13]["ratio"] = 9.5
    rows[16]["income"] = 1800.0
    rows[20]["pressure"] = 72.0
    rows[22]["score"] = -24.0
    return rows


def build_config() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "table_name": "synthetic_feature_table",
        "tensor_dimensions": [ROW_COUNT, len(COLUMNS)],
        "column_axis": 1,
        "row_id_field": ROW_FIELD,
        "group_id_field": GROUP_FIELD,
        "missing_value_markers": [None, MISSING],
        "column_groups": {
            "g_numeric": ["age", "income", "score"],
            "g_sensor": ["temp", "pressure", "latency"],
            "g_ratio": ["ratio", "load"],
        },
        "quantile_permille": {"lower": 50, "upper": 950},
        "scheme": "asymmetric",
        "quantization": {
            "bit_width_candidates": [4, 6, 8],
            "fractional_bits": 3,
            "gear": 2,
        },
        "clipping_strategy": "saturate",
        "tolerance": {
            "absolute": 0.08,
            "relative": 0.06,
            "comparison": "either",
            "outlier_ratio_limit": 0.2,
            "tie_epsilon": 1e-9,
            "zero_relative_branch": "absolute_only",
        },
        "thread_pool_size": 3,
        "element_budget_per_batch": 48,
        "interpolation": "linear",
        "tie_break": "lower",
    }


def build_plan() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "batches": [
            {
                "batch_id": "b01_correct_append_delete",
                "actions": [
                    {"op": "update", "row_id": "r0003", "values": {"score": 7.75}},
                    {
                        "op": "append",
                        "row": {
                            ROW_FIELD: "r0025",
                            GROUP_FIELD: "g_sensor",
                            "age": 52.0,
                            "income": 132.0,
                            "score": -2.25,
                            "temp": 41.5,
                            "pressure": 108.25,
                            "latency": 2.4,
                            "ratio": 0.22,
                            "load": 83.0,
                        },
                    },
                    {"op": "delete", "row_id": "r0011"},
                ],
            },
            {
                "batch_id": "b02_row_regroup",
                "actions": [
                    {"op": "regroup", "row_id": "r0005", "group_id": "g_ratio"},
                ],
            },
            {
                "batch_id": "b03_small_corrections",
                "actions": [
                    {"op": "update", "row_id": "r0016", "values": {"income": 190.5}},
                    {"op": "delete", "row_id": "r0021"},
                    {"op": "append", "row": {"row_id": "r0026", "group_id": "g_numeric", "age": 44.0}},
                ],
            },
            {
                "batch_id": "b04_column_repartition",
                "actions": [
                    {
                        "op": "repartition_columns",
                        "column_groups": {
                            "g_numeric": ["age", "income"],
                            "g_sensor": ["temp", "pressure", "score"],
                            "g_ratio": ["ratio", "load", "latency"],
                        },
                    }
                ],
            },
            {
                "batch_id": "b05_post_repartition_edit",
                "actions": [
                    {"op": "update", "row_id": "r0010", "values": {"load": 58.0}},
                    {"op": "append", "row": {"row_id": "r0027", "group_id": "g_ratio", "ratio": 1.9}},
                ],
            },
        ],
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) for row in rows
    )
    path.write_text(text + "\n", encoding="utf-8", newline="\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="inputs", help="sample directory")
    namespace = parser.parse_args()
    out = Path(namespace.out)
    rows = build_rows()
    config = build_config()
    plan = build_plan()
    write_json(out / "config.json", config)
    write_jsonl(out / "features.jsonl", rows)
    write_json(out / "plan.json", plan)
    print(
        json.dumps(
            {
                "out": str(out),
                "rows": len(rows),
                "groups": len(config["column_groups"]),
                "columns": len(COLUMNS),
                "plan_batches": len(plan["batches"]),
                "seed": SEED,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
