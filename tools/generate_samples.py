"""Deterministic local sample generator for the three exchange formats."""

from __future__ import annotations

import argparse
import csv
import json
import random
import struct
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from feature_pipeline.models import V1, V2, V3, Cell, EntityTable
from feature_pipeline.parsers import v1_fixedwidth, v2_json, v3_binary


FEATURES = [
    ("f_income", "continuous", ("per_mille", "float32", "float64")),
    ("f_age_score", "ratio", ("per_mille", "float32", "float64")),
    ("f_risk_level", "continuous", ("float32", "float64")),
    ("f_engagement", "ratio", ("per_mille", "float32", "float64")),
    ("f_stability", "continuous", ("float32", "float64")),
    ("f_velocity", "continuous", ("per_mille", "float32", "float64")),
]


def build_logical_dataset(seed: int = 20261118):
    rng = random.Random(seed)
    entities = [f"ent_{index:04d}" for index in range(24)]
    truth: dict[str, dict[str, float]] = {}
    for entity_id in entities:
        truth[entity_id] = {
            feature: round(rng.uniform(-5.0, 95.0), 6)
            for feature, _, _ in FEATURES
        }

    missing_plan = {
        V1: {("ent_0003", "f_income"), ("ent_0010", "f_risk_level"),
             ("ent_0021", "f_velocity")},
        V2: {("ent_0003", "f_income"), ("ent_0011", "f_engagement"),
             ("ent_0007", "f_stability")},
        V3: {("ent_0003", "f_income"), ("ent_0011", "f_engagement"),
             ("ent_0015", "f_age_score")},
    }
    return entities, truth, missing_plan


def make_tables(seed: int = 20261118):
    entities, truth, missing_plan = build_logical_dataset(seed)
    feature_names = [item[0] for item in FEATURES]
    tables = {
        V1: EntityTable(source_version=V1, feature_order=list(feature_names)),
        V2: EntityTable(source_version=V2, feature_order=list(feature_names)),
        V3: EntityTable(source_version=V3, feature_order=list(feature_names)),
    }

    allowed_map = {feature: allowed for feature, _, allowed in FEATURES}

    # Deterministic source-specific perturbation creates real merge conflicts.
    bias = {V1: 0.0, V2: 0.004, V3: -0.011}
    for version, table in tables.items():
        for entity_id in entities:
            for feature in feature_names:
                missing = (entity_id, feature) in missing_plan[version]
                if missing:
                    value = float("nan")
                else:
                    value = truth[entity_id][feature] + bias[version]
                if version == V1:
                    allowed = allowed_map[feature]
                    precision = "per_mille" if "per_mille" in allowed else allowed[0]
                elif version == V3:
                    precision = "float32"
                else:
                    precision = "float64"
                table.cells.append(
                    Cell(
                        entity_id=entity_id,
                        feature=feature,
                        source_version=version,
                        value=value,
                        missing=missing,
                        precision_label=precision,
                        partition_id=None,
                    )
                )

    # v3 keeps explicit partition boundaries (3 partitions of 8 entities).
    v3_table = tables[V3]
    for partition_id, start in enumerate(range(0, len(entities), 8)):
        v3_table.partition_boundaries[partition_id] = entities[start : start + 8]
        for cell in v3_table.cells:
            if cell.entity_id in v3_table.partition_boundaries[partition_id]:
                cell.partition_id = partition_id
    return tables, entities, feature_names, truth, missing_plan


def write_metadata(path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature", "semantic_type", "allowed_precisions"])
        for feature, semantic, allowed in FEATURES:
            writer.writerow([feature, semantic, "|".join(allowed)])


def write_config(path: Path) -> None:
    config = {
        "feature_whitelist": [item[0] for item in FEATURES],
        "scaling_method": "zscore",
        "missing_fill": {"strategy": "mean"},
        "conflict_priority": "source_version",
        "source_order": [V1, V2, V3],
        "precision_rank": ["float64", "float32", "per_mille"],
        "numeric_tolerance": 0.02,
        "max_workers": 4,
        "batch_size": 8,
        "percentile_tie_rule": "average_rank",
    }
    path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="samples")
    parser.add_argument("--seed", type=int, default=20261118)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tables, entities, feature_names, truth, missing_plan = make_tables(args.seed)
    v1_fixedwidth.write_file(out / "features_v1.txt", tables[V1])
    v2_json.write_file(out / "features_v2.json", tables[V2])
    v3_binary.write_file(out / "features_v3.bin", tables[V3])
    write_metadata(out / "feature_metadata.csv")
    write_config(out / "merge_config.json")

    plan = {
        "seed": args.seed,
        "entity_count": len(entities),
        "feature_count": len(feature_names),
        "entities": entities,
        "features": feature_names,
        "missing_plan": {
            version: sorted(f"{e}|{f}" for e, f in pairs)
            for version, pairs in missing_plan.items()
        },
        "source_bias": {V1: 0.0, V2: 0.004, V3: -0.011},
        "v3_partition_boundaries": tables[V3].partition_boundaries,
        "logical_truth": truth,
    }
    (out / "generation_plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"generated deterministic samples in {out}")


if __name__ == "__main__":
    main()
