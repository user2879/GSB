from __future__ import annotations

from typing import Any


def _stratum_bucket() -> dict[str, Any]:
    return {
        "sample_ids": set(),
        "key_count": 0,
        "scores": [],
        "truth_scores": [],
        "score_abs_errors": [],
        "probabilities": [],
        "truth_probabilities": [],
        "probability_abs_errors": [],
    }


def _finalize_bucket(bucket: dict[str, Any]) -> dict[str, Any]:
    scores = bucket["scores"]
    truth_scores = bucket["truth_scores"]
    probabilities = bucket["probabilities"]
    truth_probabilities = bucket["truth_probabilities"]
    score_mean = sum(scores) / len(scores) if scores else None
    probability_mean = sum(probabilities) / len(probabilities) if probabilities else None
    score_errors = bucket["score_abs_errors"]
    probability_errors = bucket["probability_abs_errors"]
    max_deviation_from_mean = max((abs(value - score_mean) for value in scores), default=None) if score_mean is not None else None
    return {
        "sample_count": len(bucket["sample_ids"]),
        "key_count": bucket["key_count"],
        "score_mean": score_mean,
        "truth_score_mean": sum(truth_scores) / len(truth_scores) if truth_scores else None,
        "probability_mean": probability_mean,
        "truth_probability_mean": sum(truth_probabilities) / len(truth_probabilities) if truth_probabilities else None,
        "max_deviation": max(score_errors, default=None),
        "score_max_abs_error_vs_truth": max(score_errors, default=None),
        "score_max_deviation_from_mean": max_deviation_from_mean,
        "probability_max_abs_error_vs_truth": max(probability_errors, default=None),
    }


def build_scheme_statistics(
    scheme_name: str,
    views: list[dict[str, Any]],
) -> dict[str, Any]:
    buckets = {"training": _stratum_bucket(), "extrapolation": _stratum_bucket()}
    sample_first_breaches: list[dict[str, Any]] = []
    for view in sorted(views, key=lambda item: item["sample_id"]):
        sample_id = view["sample_id"]
        errors = view["errors"]
        if errors["stages"].get("first_breach"):
            breach = dict(errors["stages"]["first_breach"])
            breach["sample_id"] = sample_id
            sample_first_breaches.append(breach)
        for key_metric in errors["keys"]:
            stratum = key_metric["stratum"]
            bucket = buckets[stratum]
            bucket["sample_ids"].add(sample_id)
            bucket["key_count"] += 1
            bucket["scores"].append(float(key_metric["fixed_score"]))
            bucket["truth_scores"].append(float(key_metric["truth_score"]))
            bucket["score_abs_errors"].append(abs(float(key_metric["fixed_score"]) - float(key_metric["truth_score"])))
            bucket["probabilities"].append(float(key_metric["fixed_probability"]))
            bucket["truth_probabilities"].append(float(key_metric["truth_probability"]))
            bucket["probability_abs_errors"].append(
                abs(float(key_metric["fixed_probability"]) - float(key_metric["truth_probability"]))
            )

    finalized = {name: _finalize_bucket(bucket) for name, bucket in buckets.items()}
    breach_stage_counts = {"position": 0, "rotation": 0, "dot": 0, "normalization": 0}
    for breach in sample_first_breaches:
        breach_stage_counts[breach["stage"]] += 1
    return {
        "scheme_id": scheme_name,
        "sample_count_available": len(views),
        "strata": finalized,
        "first_breaches": sorted(sample_first_breaches, key=lambda item: (item["sample_id"], item["stage"])),
        "first_breach_stage_counts": breach_stage_counts,
    }


def build_cross_scheme_comparison(
    scheme_names: list[str],
    views_by_scheme: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    views_index: dict[str, dict[tuple[str, int], dict[str, Any]]] = {}
    for scheme_name in scheme_names:
        index: dict[tuple[str, int], dict[str, Any]] = {}
        for view in views_by_scheme.get(scheme_name, []):
            for key_metric in view["errors"]["keys"]:
                if key_metric["stratum"] == "extrapolation":
                    index[(view["sample_id"], int(key_metric["key_index"]))] = key_metric
        views_index[scheme_name] = index

    positions = sorted(
        {position for index in views_index.values() for position in index},
        key=lambda item: (item[0], item[1]),
    )
    baseline = scheme_names[0]
    rows: list[dict[str, Any]] = []
    for sample_id, key_index in positions:
        baseline_metric = views_index[baseline].get((sample_id, key_index))
        if baseline_metric is None:
            continue
        scheme_rows: dict[str, Any] = {}
        differences: dict[str, Any] = {}
        ranges: dict[str, float | None] = {}
        for stage in ("position", "rotation", "dot", "normalization"):
            values: list[float] = []
            for scheme_name in scheme_names:
                metric = views_index[scheme_name].get((sample_id, key_index))
                if metric is None:
                    continue
                abs_error = float(metric["stages"][stage]["abs_error"])
                values.append(abs_error)
                scheme_rows.setdefault(scheme_name, {})[stage] = {
                    "abs_error": abs_error,
                    "rel_error": float(metric["stages"][stage]["rel_error"]),
                    "exceeded": bool(metric["stages"][stage]["exceeded"]),
                }
                baseline_error = float(baseline_metric["stages"][stage]["abs_error"])
                differences.setdefault(scheme_name, {})[stage] = abs_error - baseline_error
            ranges[stage] = max(values) - min(values) if values else None
        for scheme_name in scheme_names:
            metric = views_index[scheme_name].get((sample_id, key_index))
            if metric is not None:
                scheme_rows[scheme_name]["fixed_score"] = metric["fixed_score"]
                scheme_rows[scheme_name]["fixed_probability"] = metric["fixed_probability"]
        rows.append(
            {
                "sample_id": sample_id,
                "key_index": key_index,
                "key_position": baseline_metric["key_position"],
                "relative_distance": baseline_metric["relative_distance"],
                "baseline_scheme": baseline,
                "schemes": scheme_rows,
                "differences_vs_baseline": differences,
                "stage_error_ranges": ranges,
            }
        )
    return rows
