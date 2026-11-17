"""Cross-version interop diagnostic report."""

from __future__ import annotations
import json
import math
from pathlib import Path

from .conversion import to_v3, v1_to_v2, v2_to_v1, v3_to_v2
from .models import V1, V2, V3, Cell, EntityTable, MergeConfig
from .parsers import v1_fixedwidth, v2_json, v3_binary


HISTOGRAM_EDGES = (0.0, 1e-9, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, math.inf)
HISTOGRAM_LABELS = (
    "0",
    "(0,1e-9]",
    "(1e-9,1e-6]",
    "(1e-6,1e-4]",
    "(1e-4,1e-3]",
    "(1e-3,1e-2]",
    "(1e-2,1e-1]",
    "(1e-1,1.0]",
    ">1.0",
)


def _bucket(deviation: float) -> int:
    for index in range(len(HISTOGRAM_EDGES) - 1):
        low, high = HISTOGRAM_EDGES[index], HISTOGRAM_EDGES[index + 1]
        if low < deviation <= high or (index == 0 and deviation == 0.0):
            return index
    return len(HISTOGRAM_LABELS) - 1


def _empty_histogram() -> dict:
    return {label: 0 for label in HISTOGRAM_LABELS}


def _index_cells(table: EntityTable) -> dict[tuple[str, str], Cell]:
    return {(cell.entity_id, cell.feature): cell for cell in table.cells}


def _round_trip(base: EntityTable, metadata: dict, work_dir: Path) -> dict:
    """Serialize one logical feature set in all three formats, then reparse."""
    work_dir.mkdir(parents=True, exist_ok=True)
    round_trips: dict[str, EntityTable] = {}
    downgrade_losses: list[dict] = []

    if base.source_version == V1:
        v1_path = work_dir / "rt_v1.txt"
        v1_fixedwidth.write_file(v1_path, base)
        v1_rt = v1_fixedwidth.parse_file(v1_path)

        v2_table = v1_to_v2(base, metadata)
        v2_path = work_dir / "rt_v2.json"
        v2_json.write_file(v2_path, v2_table)
        v2_rt = v2_json.parse_file(v2_path)

        v3_table, v3_losses = to_v3(base)
        downgrade_losses.extend(v3_losses)
        v3_path = work_dir / "rt_v3.bin"
        v3_binary.write_file(v3_path, v3_table)
        v3_rt = v3_binary.parse_file(v3_path)

    elif base.source_version == V2:
        v2_path = work_dir / "rt_v2.json"
        v2_json.write_file(v2_path, base)
        v2_rt = v2_json.parse_file(v2_path)

        sidecar = work_dir / "rt_v2_to_v1_sidecar.json"
        v1_table = v2_to_v1(base, sidecar)
        v1_path = work_dir / "rt_v1.txt"
        v1_fixedwidth.write_file(v1_path, v1_table)
        v1_rt = v1_fixedwidth.parse_file(v1_path)
        sidecar_payload = json.loads(sidecar.read_text(encoding="utf-8"))
        downgrade_losses.extend(sidecar_payload.get("downgrade_losses", []))

        v3_table, v3_losses = to_v3(base)
        downgrade_losses.extend(v3_losses)
        v3_path = work_dir / "rt_v3.bin"
        v3_binary.write_file(v3_path, v3_table)
        v3_rt = v3_binary.parse_file(v3_path)
    else:
        v3_path = work_dir / "rt_v3.bin"
        v3_binary.write_file(v3_path, base)
        v3_rt = v3_binary.parse_file(v3_path)

        v2_table, v2_losses = v3_to_v2(base)
        downgrade_losses.extend(v2_losses)
        v2_path = work_dir / "rt_v2.json"
        v2_json.write_file(v2_path, v2_table)
        v2_rt = v2_json.parse_file(v2_path)

        sidecar = work_dir / "rt_v3_to_v1_sidecar.json"
        v1_table, v1_losses = _v3_to_v1_simple(base, sidecar)
        downgrade_losses.extend(v1_losses)
        v1_path = work_dir / "rt_v1.txt"
        v1_fixedwidth.write_file(v1_path, v1_table)
        v1_rt = v1_fixedwidth.parse_file(v1_path)

    round_trips = {V1: v1_rt, V2: v2_rt, V3: v3_rt}
    return {"tables": round_trips, "downgrade_losses": downgrade_losses}


def _v3_to_v1_simple(table: EntityTable, sidecar: Path):
    from .conversion import v3_to_v1

    return v3_to_v1(table, sidecar)


def build_interop_report(
    base: EntityTable,
    metadata: dict,
    config: MergeConfig,
    work_dir: Path,
) -> dict:
    trip = _round_trip(base, metadata, work_dir)
    tables: dict[str, EntityTable] = trip["tables"]
    base_index = _index_cells(base)

    entity_sets = {
        version: sorted(table.entity_ids()) for version, table in tables.items()
    }
    feature_sets = {
        version: sorted(table.feature_names())
        for version, table in tables.items()
    }

    per_feature: dict[str, dict] = {}
    missing_diffs: list[dict] = []
    precision_restore: list[dict] = []
    max_deviation = {"deviation": -1.0, "feature": None, "entity_id": None,
                     "source_version": None}

    all_features = sorted(base.feature_names())
    for feature in all_features:
        histogram = _empty_histogram()
        deviations: list[float] = []
        for version, table in tables.items():
            parsed_index = _index_cells(table)
            for key, base_cell in base_index.items():
                entity_id, cell_feature = key
                if cell_feature != feature:
                    continue
                parsed_cell = parsed_index.get(key)
                if parsed_cell is None:
                    missing_diffs.append(
                        {
                            "entity_id": entity_id,
                            "feature": feature,
                            "source_version": version,
                            "difference": "slot_missing_after_roundtrip",
                        }
                    )
                    continue
                if parsed_cell.missing != base_cell.missing:
                    missing_diffs.append(
                        {
                            "entity_id": entity_id,
                            "feature": feature,
                            "source_version": version,
                            "difference": "missing_flag_changed",
                            "base_missing": base_cell.missing,
                            "roundtrip_missing": parsed_cell.missing,
                        }
                    )
                same_label = (
                    base_cell.precision_label == parsed_cell.precision_label
                )
                expected_downgrade = (
                    version in (V1, V3) and not same_label
                )
                precision_restore.append(
                    {
                        "entity_id": entity_id,
                        "feature": feature,
                        "source_version": version,
                        "base_precision": base_cell.precision_label,
                        "roundtrip_precision": parsed_cell.precision_label,
                        "restored": same_label,
                        "expected_storage_downgrade": expected_downgrade,
                        "label_interop_ok": same_label or expected_downgrade,
                    }
                )
                if not base_cell.missing and not parsed_cell.missing:
                    deviation = abs(float(parsed_cell.value) - float(base_cell.value))
                    deviations.append(deviation)
                    histogram[HISTOGRAM_LABELS[_bucket(deviation)]] += 1
                    if deviation > max_deviation["deviation"]:
                        max_deviation = {
                            "deviation": deviation,
                            "feature": feature,
                            "entity_id": entity_id,
                            "source_version": version,
                        }
        per_feature[feature] = {
            "sample_count": len(deviations),
            "max_abs_deviation": max(deviations) if deviations else 0.0,
            "mean_abs_deviation": sum(deviations) / len(deviations)
            if deviations
            else 0.0,
            "deviation_histogram": histogram,
        }

    entity_count_match = all(
        set(ids) == set(entity_sets[base.source_version])
        for ids in entity_sets.values()
    )
    feature_set_match = all(
        set(names) == set(feature_sets[base.source_version])
        for names in feature_sets.values()
    )
    missing_match = not missing_diffs
    precision_match = all(item["label_interop_ok"] for item in precision_restore)
    tolerance = config.numeric_tolerance
    within_tolerance = all(
        entry["max_abs_deviation"] <= tolerance for entry in per_feature.values()
    )
    consistent = all(
        (entity_count_match, feature_set_match, missing_match,
         precision_match, within_tolerance)
    )

    return {
        "report_type": "cross_version_interop",
        "base_version": base.source_version,
        "entity_count": {
            version: len(ids) for version, ids in entity_sets.items()
        },
        "entity_count_match": entity_count_match,
        "feature_name_sets": feature_sets,
        "feature_set_match": feature_set_match,
        "per_feature_deviation": per_feature,
        "max_deviation_feature_entity": max_deviation,
        "missing_position_differences": missing_diffs,
        "missing_positions_match": missing_match,
        "precision_label_restore": precision_restore,
        "precision_labels_restored": precision_match,
        "downgrade_losses": trip["downgrade_losses"],
        "numeric_tolerance": tolerance,
        "within_numeric_tolerance": within_tolerance,
        "cross_version_consistent": consistent,
        "consistency_judgement": (
            "CONSISTENT" if consistent else "INCONSISTENT"
        ),
    }
