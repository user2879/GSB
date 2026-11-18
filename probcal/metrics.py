from __future__ import annotations

from typing import Any

from .util import (
    decimal_to_fixed,
    fixed_to_decimal,
    round_half_up_ratio,
    scheme_rank,
)


def build_metrics_stage(
    data_artifact: dict[str, Any],
    calibration_artifact: dict[str, Any],
    evaluation_config: dict[str, Any],
    fixed: dict[str, int],
    metrics_key: str,
) -> dict[str, Any]:
    labels_by_id = {
        sample_id: int(label)
        for sample_id, label in zip(
            data_artifact["validation"]["sample_ids"],
            data_artifact["validation"]["labels"],
            strict=True,
        )
    }
    calibration_bins = int(evaluation_config["calibration_bins"])
    scale = fixed["scale"]
    metric_scale = fixed["metric_scale"]
    decimal_places = fixed["decimal_places"]
    thresholds = _threshold_grid(evaluation_config["threshold_step"], scale)
    scheme_metrics: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    for result in calibration_artifact["results"]:
        records = sorted(result["probabilities"], key=lambda item: item["sample_id"])
        labels = [labels_by_id[record["sample_id"]] for record in records]
        metric = _calibration_metric(records, labels, calibration_bins, scale, metric_scale)
        metric.update(
            {
                "model_name": result["model_name"],
                "model_id": result["model_id"],
                "model_type": result["model_type"],
                "params_hash": result["params_hash"],
                "fold_id": result["fold_id"],
                "scheme": result["scheme"],
                "ece": fixed_to_decimal(metric["ece_fixed"], metric_scale, decimal_places),
                "mce": fixed_to_decimal(metric["mce_fixed"], metric_scale, decimal_places),
            }
        )
        scheme_metrics.append(metric)
        for threshold_fixed in thresholds:
            row = _threshold_metric(
                records,
                labels,
                threshold_fixed,
                scale,
                metric_scale,
                decimal_places,
            )
            row.update(
                {
                    "model_name": result["model_name"],
                    "model_id": result["model_id"],
                    "model_type": result["model_type"],
                    "params_hash": result["params_hash"],
                    "fold_id": result["fold_id"],
                    "scheme": result["scheme"],
                }
            )
            threshold_rows.append(row)
    scheme_metrics.sort(
        key=lambda item: (
            item["model_name"],
            scheme_rank(item["scheme"]),
            item["fold_id"],
            item["params_hash"],
        )
    )
    threshold_rows.sort(
        key=lambda item: (
            item["model_name"],
            scheme_rank(item["scheme"]),
            item["fold_id"],
            item["params_hash"],
            item["threshold_fixed"],
        )
    )
    return {
        "metrics_key": metrics_key,
        "calibration_key": calibration_artifact["calibration_key"],
        "data_key": data_artifact["data_key"],
        "evaluation": {
            "calibration_bins": calibration_bins,
            "threshold_step": str(evaluation_config["threshold_step"]),
            "fixed_point": fixed,
            "bin_boundary_rule": "lower-inclusive, upper-exclusive; 1.0 belongs to the last bin",
            "threshold_boundary_rule": "prediction is positive when probability_fixed >= threshold_fixed",
        },
        "scheme_metrics": scheme_metrics,
        "threshold_grid": threshold_rows,
        "evaluation_audit": {
            "sample_order": "validation sample_id ascending",
            "probability_order": "validated calibration output, sample_id ascending",
        },
    }


def _calibration_metric(
    records: list[dict[str, int]],
    labels: list[int],
    bin_count: int,
    scale: int,
    metric_scale: int,
) -> dict[str, Any]:
    bins = [
        {
            "bin": index,
            "lower_fixed_inclusive": index * scale // bin_count,
            "upper_fixed_exclusive": (index + 1) * scale // bin_count
            if index < bin_count - 1
            else scale,
            "count": 0,
            "positive_count": 0,
            "probability_sum_fixed": 0,
            "boundary_sample_ids": [],
        }
        for index in range(bin_count)
    ]
    boundary_ids: list[int] = []
    for record, label in zip(records, labels, strict=True):
        probability = int(record["calibrated_probability_fixed"])
        index = min(bin_count - 1, probability * bin_count // scale)
        bucket = bins[index]
        bucket["count"] += 1
        bucket["positive_count"] += int(label)
        bucket["probability_sum_fixed"] += probability
        if (probability * bin_count) % scale == 0:
            bucket["boundary_sample_ids"].append(record["sample_id"])
            boundary_ids.append(record["sample_id"])
    total_difference = 0
    max_gap_fixed = 0
    for bucket in bins:
        count = bucket["count"]
        if count:
            difference = abs(
                bucket["probability_sum_fixed"] - bucket["positive_count"] * scale
            )
            total_difference += difference
            bucket["average_probability_fixed"] = round_half_up_ratio(
                bucket["probability_sum_fixed"], count
            )
            bucket["observed_rate_fixed"] = round_half_up_ratio(
                bucket["positive_count"] * scale, count
            )
            bucket["gap_fixed"] = abs(
                bucket["average_probability_fixed"] - bucket["observed_rate_fixed"]
            )
            max_gap_fixed = max(max_gap_fixed, bucket["gap_fixed"])
        else:
            bucket["average_probability_fixed"] = 0
            bucket["observed_rate_fixed"] = 0
            bucket["gap_fixed"] = 0
    sample_count = len(records)
    ece_fixed = (
        round_half_up_ratio(total_difference * metric_scale, sample_count * scale)
        if sample_count
        else 0
    )
    mce_fixed = (
        round_half_up_ratio(max_gap_fixed * metric_scale, scale)
        if sample_count
        else 0
    )
    return {
        "ece_fixed": ece_fixed,
        "mce_fixed": mce_fixed,
        "calibration_bins": bin_count,
        "sample_count": sample_count,
        "bin_details": bins,
        "boundary_sample_ids": sorted(boundary_ids),
    }


def _threshold_metric(
    records: list[dict[str, int]],
    labels: list[int],
    threshold_fixed: int,
    scale: int,
    metric_scale: int,
    decimal_places: int,
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    boundary_ids: list[int] = []
    for record, label in zip(records, labels, strict=True):
        probability = int(record["calibrated_probability_fixed"])
        if probability == threshold_fixed:
            boundary_ids.append(record["sample_id"])
        predicted = 1 if probability >= threshold_fixed else 0
        if predicted == 1 and label == 1:
            tp += 1
        elif predicted == 1 and label == 0:
            fp += 1
        elif predicted == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    total = tp + fp + tn + fn
    accuracy_fixed = round_half_up_ratio((tp + tn) * metric_scale, total) if total else 0
    recall_fixed = round_half_up_ratio(tp * metric_scale, tp + fn) if tp + fn else 0
    precision_fixed = round_half_up_ratio(tp * metric_scale, tp + fp) if tp + fp else 0
    f1_fixed = (
        round_half_up_ratio(2 * tp * metric_scale, 2 * tp + fp + fn)
        if 2 * tp + fp + fn
        else 0
    )
    return {
        "threshold_fixed": int(threshold_fixed),
        "threshold": fixed_to_decimal(threshold_fixed, scale, decimal_places),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy_fixed": accuracy_fixed,
        "recall_fixed": recall_fixed,
        "precision_fixed": precision_fixed,
        "f1_fixed": f1_fixed,
        "accuracy": fixed_to_decimal(accuracy_fixed, metric_scale, decimal_places),
        "recall": fixed_to_decimal(recall_fixed, metric_scale, decimal_places),
        "precision": fixed_to_decimal(precision_fixed, metric_scale, decimal_places),
        "f1": fixed_to_decimal(f1_fixed, metric_scale, decimal_places),
        "boundary_sample_ids": sorted(boundary_ids),
    }


def _threshold_grid(step: Any, scale: int) -> list[int]:
    step_fixed = decimal_to_fixed(step, scale)
    if step_fixed <= 0:
        raise ValueError("threshold step rounds to zero")
    thresholds = list(range(0, scale + 1, step_fixed))
    if thresholds[-1] != scale:
        thresholds.append(scale)
    return thresholds
