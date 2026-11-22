#!/usr/bin/env python3
"""Generate the complete deterministic local fixture set.

Run with: .venv/bin/python scripts/generate_fixtures.py
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from feature_backfill.config_loader import load_feature_file, load_plan, parse_config
from feature_backfill.data_loader import load_samples
from feature_backfill.fingerprints import canonical_json
from feature_backfill.plan import materialize_plan_state


DATA = ROOT / "data"
DEFAULT = DATA / "default"
INJECTIONS = ROOT / "injections"
SEED = 20261123


FEATURES = {
    "raw_fields": ["amount", "qty", "score"],
    "features": [
        {
            "name": "amount_count_5000",
            "source_field": "amount",
            "window_ms": 5000,
            "aggregation": "count",
            "depends_on": [],
        },
        {
            "name": "amount_sum_5000",
            "source_field": "amount",
            "window_ms": 5000,
            "aggregation": "sum",
            "depends_on": [],
        },
        {
            "name": "amount_mean_5000",
            "source_field": "amount",
            "window_ms": 5000,
            "aggregation": "mean",
            "depends_on": [],
        },
        {
            "name": "amount_latest_5000",
            "source_field": "amount",
            "window_ms": 5000,
            "aggregation": "latest",
            "depends_on": [],
        },
        {
            "name": "qty_sum_5000",
            "source_field": "qty",
            "window_ms": 5000,
            "aggregation": "sum",
            "depends_on": [],
        },
        {
            "name": "score_mean_5000",
            "source_field": "score",
            "window_ms": 5000,
            "aggregation": "mean",
            "depends_on": [],
        },
        {
            "name": "amount_chain_5000",
            "source_feature": "amount_mean_5000",
            "window_ms": 5000,
            "aggregation": "mean",
            "depends_on": ["amount_mean_5000"],
        },
    ],
}


CONFIG = {
    "batch_rows": 1000,
    "entity_shards": 4,
    "batch_point_quota": 100000,
    "lease_ttl_ms": 500,
    "lease_renew_interval_ms": 100,
    "thread_pool_concurrency": 4,
    "snapshot_every_batches": 1,
    "cache_resident_blocks": 16,
    "clock_source": "monotonic_ns",
    "missing_marker": "__MISSING__",
    "lease_wait_timeout_ms": 5000,
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def write_yaml(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(value, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def value_or_missing(rng: np.random.Generator, entity_index: int, row_index: int, field: str):
    if field == "amount":
        if (entity_index + row_index) % 9 == 0:
            return "__MISSING__"
        return round(float(rng.uniform(3.0, 60.0)), 3)
    if field == "qty":
        if (entity_index + 2 * row_index) % 13 == 0:
            return None
        return int(rng.integers(1, 10))
    if field == "score":
        if (entity_index * 2 + row_index) % 11 == 0:
            return "__MISSING__"
        return round(float(rng.uniform(0.0, 1.0)), 5)
    raise KeyError(field)


def make_row(
    rng: np.random.Generator,
    *,
    entity_index: int,
    row_index: int,
    entity_id: str,
    timestamp_ms: int,
    row_tag: str | None = None,
) -> dict[str, object]:
    row_suffix = f"{row_index + 1:03d}" if row_tag is None else row_tag
    row = {
        "row_id": f"R-{entity_id}-{row_suffix}",
        "entity_id": entity_id,
        "timestamp_ms": timestamp_ms,
    }
    for field in ("amount", "qty", "score"):
        value = value_or_missing(rng, entity_index, row_index, field)
        if value is not None:
            row[field] = value
    missing = [
        field
        for field in ("amount", "qty", "score")
        if field not in row
    ]
    if missing:
        row["missing_fields"] = missing
    return row


def build_base_rows() -> list[dict[str, object]]:
    rng = np.random.Generator(np.random.PCG64(SEED))
    rows: list[dict[str, object]] = []
    for entity_index in range(8):
        entity_id = f"E{entity_index + 1:03d}"
        base_timestamp = (entity_index + 1) * 100
        for row_index, offset in enumerate((0, 1200, 2600, 4100)):
            rows.append(
                make_row(
                    rng,
                    entity_index=entity_index,
                    row_index=row_index,
                    entity_id=entity_id,
                    timestamp_ms=base_timestamp + offset,
                )
            )
    return rows


def build_appended_rows() -> dict[str, list[dict[str, object]]]:
    rng = np.random.Generator(np.random.PCG64(SEED + 1))
    return {
        "batch1_e001": [
            make_row(
                rng,
                entity_index=0,
                row_index=4,
                entity_id="E001",
                timestamp_ms=6000,
                row_tag="005",
            ),
            make_row(
                rng,
                entity_index=0,
                row_index=5,
                entity_id="E001",
                timestamp_ms=7000,
                row_tag="006",
            ),
        ],
        "batch1_e002": [
            make_row(
                rng,
                entity_index=1,
                row_index=4,
                entity_id="E002",
                timestamp_ms=5400,
                row_tag="005",
            )
        ],
        "batch4_e009_discarded": [
            make_row(
                rng,
                entity_index=8,
                row_index=0,
                entity_id="E009",
                timestamp_ms=1000,
                row_tag="discarded-001",
            ),
            make_row(
                rng,
                entity_index=8,
                row_index=1,
                entity_id="E009",
                timestamp_ms=2200,
                row_tag="discarded-002",
            ),
        ],
        "batch6_e009": [
            make_row(
                rng,
                entity_index=9,
                row_index=0,
                entity_id="E009",
                timestamp_ms=1000,
                row_tag="001",
            ),
            make_row(
                rng,
                entity_index=9,
                row_index=1,
                entity_id="E009",
                timestamp_ms=2200,
                row_tag="002",
            ),
        ],
    }


def build_plan(rows: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    return {
        "batches": [
            {
                "batch_id": 1,
                "operations": [
                    {
                        "op": "append",
                        "rows": rows["batch1_e001"] + rows["batch1_e002"],
                    }
                ],
            },
            {
                "batch_id": 2,
                "operations": [
                    {
                        "op": "correct",
                        "row_id": "R-E003-002",
                        "changes": {"amount": 42.75},
                    }
                ],
            },
            {
                "batch_id": 3,
                "operations": [
                    {
                        "op": "window_adjust",
                        "feature": "amount_sum_5000",
                        "window_ms": 9000,
                    }
                ],
            },
            {
                "batch_id": 4,
                "operations": [
                    {"op": "append", "rows": rows["batch4_e009_discarded"]}
                ],
            },
            {
                "batch_id": 5,
                "operations": [{"op": "rollback", "to_batch": 2}],
            },
            {
                "batch_id": 6,
                "operations": [{"op": "append", "rows": rows["batch6_e009"]}],
            },
            {
                "batch_id": 7,
                "operations": [
                    {
                        "op": "correct",
                        "row_id": "R-E001-001",
                        "changes": {"score": 0.777},
                    },
                    {
                        "op": "window_adjust",
                        "feature": "score_mean_5000",
                        "window_ms": 12000,
                    },
                ],
            },
        ]
    }


def write_default_fixture(base_rows: list[dict[str, object]], plan: dict[str, object]) -> None:
    DEFAULT.mkdir(parents=True, exist_ok=True)
    with (DEFAULT / "samples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in base_rows:
            handle.write(canonical_json(row) + "\n")
    write_json(DEFAULT / "features.json", FEATURES)
    write_yaml(DEFAULT / "config.yaml", CONFIG)
    write_json(DEFAULT / "plan.json", plan)


def write_scenario_configs() -> None:
    scenarios = {
        "quota": {**CONFIG, "batch_point_quota": 60},
        "cache_tiny": {**CONFIG, "cache_resident_blocks": 1},
        "serial": {**CONFIG, "thread_pool_concurrency": 1},
        "snapshot_interval2": {**CONFIG, "snapshot_every_batches": 2},
        "lease_short": {
            **CONFIG,
            "lease_ttl_ms": 120,
            "lease_renew_interval_ms": 30,
            "lease_wait_timeout_ms": 3000,
        },
    }
    for name, config in scenarios.items():
        write_yaml(DATA / "scenarios" / name / "config.yaml", config)


def write_injections() -> None:
    manifests = {
        "reference": [
            {
                "id": "INJ-001-NONFINITE",
                "type": "nonfinite_value",
                "stage": "window_features",
                "entity_id": "E003",
                "row_id": "R-E003-002",
                "field": "amount",
                "expected_code": "E_ROW_INVALID",
            },
            {
                "id": "INJ-002-TIMESTAMP-REVERSAL",
                "type": "timestamp_reversal",
                "stage": "window_features",
                "entity_id": "E004",
                "row_id": "R-E004-002",
                "expected_code": "E_ROW_INVALID",
            },
            {
                "id": "INJ-003-LEASE-TTL-ZERO",
                "type": "lease_ttl_zero",
                "stage": "lease",
                "entity_id": "E005",
                "expected_code": "E_CONFIG_INVALID",
            },
            {
                "id": "INJ-004-CACHE-BUDGET-ZERO",
                "type": "cache_budget_lt_block",
                "stage": "cache",
                "entity_id": "E006",
                "expected_code": "E_CONFIG_INVALID",
            },
        ],
        "nonfinite": [
            {
                "id": "INJ-001-NONFINITE",
                "type": "nonfinite_value",
                "stage": "window_features",
                "entity_id": "E003",
                "row_id": "R-E003-002",
                "field": "amount",
                "expected_code": "E_ROW_INVALID",
            }
        ],
        "timestamp_reversal": [
            {
                "id": "INJ-002-TIMESTAMP-REVERSAL",
                "type": "timestamp_reversal",
                "stage": "window_features",
                "entity_id": "E004",
                "row_id": "R-E004-002",
                "expected_code": "E_ROW_INVALID",
            }
        ],
        "lease_ttl_zero": [
            {
                "id": "INJ-003-LEASE-TTL-ZERO",
                "type": "lease_ttl_zero",
                "stage": "lease",
                "entity_id": "E005",
                "expected_code": "E_CONFIG_INVALID",
            }
        ],
        "cache_budget_zero": [
            {
                "id": "INJ-004-CACHE-BUDGET-ZERO",
                "type": "cache_budget_lt_block",
                "stage": "cache",
                "entity_id": "E006",
                "expected_code": "E_CONFIG_INVALID",
            }
        ],
    }
    for name, items in manifests.items():
        write_json(INJECTIONS / f"{name}.json", {"injections": items})
    write_json(ROOT / "data" / "scenarios" / "empty_plan.json", {"batches": []})


def write_illegal_fixtures(base_rows: list[dict[str, object]], plan: dict[str, object]) -> None:
    common_features = FEATURES
    common_config = CONFIG

    def copy_common(
        directory: Path,
        *,
        samples: Path | None = None,
        features: Path | None = None,
        config: dict[str, object] | None = None,
        plan_payload: dict[str, object] | None = None,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if samples is None:
            if not (directory / "samples.jsonl").exists():
                shutil.copy2(DEFAULT / "samples.jsonl", directory / "samples.jsonl")
        else:
            shutil.copy2(samples, directory / "samples.jsonl")
        if features is None:
            if not (directory / "features.json").exists():
                write_json(directory / "features.json", common_features)
        else:
            shutil.copy2(features, directory / "features.json")
        write_yaml(directory / "config.yaml", config or common_config)
        write_json(directory / "plan.json", plan_payload or plan)

    bad_rows_dir = DATA / "illegal" / "bad_rows_all"
    bad_rows_dir.mkdir(parents=True, exist_ok=True)
    with (bad_rows_dir / "samples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in base_rows[:5]:
            handle.write(canonical_json(row) + "\n")
        handle.write(canonical_json(base_rows[0]) + "\n")
        handle.write(
            '{"row_id":"R-BAD-NAN","entity_id":"E099","timestamp_ms":1000,'
            '"amount":NaN,"qty":1,"score":0.5}\n'
        )
        handle.write(
            '{"row_id":"R-BAD-ORDER","entity_id":"E002","timestamp_ms":0,'
            '"amount":1.0,"qty":1,"score":0.5}\n'
        )
    write_json(bad_rows_dir / "features.json", FEATURES)
    write_yaml(bad_rows_dir / "config.yaml", CONFIG)
    write_json(bad_rows_dir / "plan.json", plan)

    duplicate_dir = DATA / "illegal" / "duplicate_row"
    duplicate_dir.mkdir(parents=True, exist_ok=True)
    with (duplicate_dir / "samples.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in base_rows[:5]:
            handle.write(canonical_json(row) + "\n")
        handle.write(canonical_json(base_rows[0]) + "\n")
    copy_common(duplicate_dir)

    nonfinite_dir = DATA / "illegal" / "nonfinite_row"
    nonfinite_dir.mkdir(parents=True, exist_ok=True)
    with (nonfinite_dir / "samples.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in base_rows[:5]:
            handle.write(canonical_json(row) + "\n")
        handle.write(
            '{"row_id":"R-BAD-NAN","entity_id":"E099","timestamp_ms":1000,'
            '"amount":NaN,"qty":1,"score":0.5}\n'
        )
    copy_common(nonfinite_dir)

    reversal_dir = DATA / "illegal" / "timestamp_reversal"
    reversal_dir.mkdir(parents=True, exist_ok=True)
    with (reversal_dir / "samples.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for row in base_rows[:5]:
            handle.write(canonical_json(row) + "\n")
        handle.write(
            '{"row_id":"R-BAD-ORDER","entity_id":"E002","timestamp_ms":0,'
            '"amount":1.0,"qty":1,"score":0.5}\n'
        )
    copy_common(reversal_dir)

    bad_dag_dir = DATA / "illegal" / "bad_dag"
    bad_dag_dir.mkdir(parents=True, exist_ok=True)
    bad_dag = json.loads(json.dumps(FEATURES))
    bad_dag["features"].append(
        {
            "name": "cycle_a",
            "source_feature": "cycle_b",
            "window_ms": 5000,
            "aggregation": "mean",
            "depends_on": ["cycle_b"],
        }
    )
    bad_dag["features"].append(
        {
            "name": "cycle_b",
            "source_feature": "cycle_a",
            "window_ms": 5000,
            "aggregation": "mean",
            "depends_on": ["cycle_a"],
        }
    )
    write_json(bad_dag_dir / "features.json", bad_dag)
    shutil.copy2(DEFAULT / "samples.jsonl", bad_dag_dir / "samples.jsonl")
    write_yaml(bad_dag_dir / "config.yaml", CONFIG)
    write_json(bad_dag_dir / "plan.json", plan)

    missing_upstream_dir = DATA / "illegal" / "missing_upstream"
    missing_upstream_dir.mkdir(parents=True, exist_ok=True)
    missing_upstream = json.loads(json.dumps(FEATURES))
    missing_upstream["features"].append(
        {
            "name": "broken_upstream",
            "source_feature": "not_defined",
            "window_ms": 5000,
            "aggregation": "mean",
            "depends_on": ["not_defined"],
        }
    )
    write_json(missing_upstream_dir / "features.json", missing_upstream)
    copy_common(missing_upstream_dir)

    missing_field_dir = DATA / "illegal" / "missing_raw_field"
    missing_field_dir.mkdir(parents=True, exist_ok=True)
    missing_field = json.loads(json.dumps(FEATURES))
    missing_field["features"].append(
        {
            "name": "broken_field",
            "source_field": "not_a_column",
            "window_ms": 5000,
            "aggregation": "sum",
            "depends_on": [],
        }
    )
    write_json(missing_field_dir / "features.json", missing_field)
    copy_common(missing_field_dir)

    for name, changes in {
        "bad_config_quota": {"batch_point_quota": 0},
        "bad_config_lease_ttl": {"lease_ttl_ms": 0},
    }.items():
        directory = DATA / "illegal" / name
        directory.mkdir(parents=True, exist_ok=True)
        copy_common(directory, config={**CONFIG, **changes})

    bad_plan_dir = DATA / "illegal" / "bad_plan"
    bad_plan_dir.mkdir(parents=True, exist_ok=True)
    copy_common(
        bad_plan_dir,
        plan_payload={
            "batches": [
                {
                    "batch_id": 1,
                    "operations": [{"op": "not_a_real_operation"}],
                }
            ]
        },
    )


def write_final_fixture(base_rows: list[dict[str, object]], plan: dict[str, object]) -> None:
    config = parse_config(DEFAULT / "config.yaml")
    features, raw_fields, feature_failures = load_feature_file(DEFAULT / "features.json")
    if feature_failures:
        raise RuntimeError(feature_failures)
    rows, row_failures = load_samples(
        DEFAULT / "samples.jsonl", config=config, raw_fields=raw_fields
    )
    if row_failures:
        raise RuntimeError(row_failures)
    batches, plan_failures = load_plan(DEFAULT / "plan.json")
    if plan_failures:
        raise RuntimeError(plan_failures)
    final_rows, final_features, materialization_failures = materialize_plan_state(
        rows,
        features,
        batches,
        config=config,
        raw_fields=raw_fields,
    )
    if materialization_failures:
        raise RuntimeError(materialization_failures)
    final_dir = DATA / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    with (final_dir / "samples.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in sorted(final_rows, key=lambda item: (item.entity_id, item.timestamp_ms, item.row_id)):
            handle.write(canonical_json(row.to_dict()) + "\n")
    write_json(
        final_dir / "features.json",
        {
            "raw_fields": sorted(raw_fields),
            "features": [final_features[name].to_dict() for name in sorted(final_features)],
        },
    )
    write_yaml(final_dir / "config.yaml", CONFIG)
    write_json(final_dir / "plan.json", {"batches": []})


def main() -> None:
    base_rows = build_base_rows()
    appended_rows = build_appended_rows()
    plan = build_plan(appended_rows)
    DEFAULT.parent.mkdir(parents=True, exist_ok=True)
    INJECTIONS.mkdir(parents=True, exist_ok=True)
    write_default_fixture(base_rows, plan)
    write_scenario_configs()
    write_injections()
    write_illegal_fixtures(base_rows, plan)
    write_final_fixture(base_rows, plan)
    print(
        json.dumps(
            {
                "status": "generated",
                "seed": SEED,
                "base_rows": len(base_rows),
                "entities": 8,
                "feature_count": len(FEATURES["features"]),
                "plan_batches": len(plan["batches"]),
                "default_dir": str(DEFAULT),
                "final_dir": str(DATA / "final"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
