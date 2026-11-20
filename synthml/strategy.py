"""Auditable scale-aware strategy selection."""

from __future__ import annotations

from typing import Any

from .config import ExperimentConfig
from .util import rounded


def scale_bucket(sample_count: int) -> str:
    if sample_count < 32:
        return "small"
    if sample_count < 64:
        return "medium"
    return "large"


def _classifier_cost(config: ExperimentConfig, classifier: str, strategy: dict[str, Any]) -> float:
    params = config.classifier_params(classifier, strategy)
    if classifier == "nearest_centroid":
        return 1.0
    if classifier == "knn":
        return 1.0 + float(params["k"]) / 10.0
    if classifier == "logistic_regression":
        return 1.0 + float(params["epochs"]) / 120.0
    raise ValueError(f"unknown classifier: {classifier}")


def select_strategy(
    config: ExperimentConfig,
    tier: dict[str, Any],
    labels: list[str],
    sample_label_sets: list[tuple[str, ...]],
    *,
    force_strategy: str | None = None,
    force_feature_method: str | None = None,
    force_color_bins: int | None = None,
    force_hog_bins: int | None = None,
    force_classifier: str | None = None,
    force_knn_k: int | None = None,
    force_threshold: float | None = None,
    precision: str | None = None,
) -> dict[str, Any]:
    sample_count = len(sample_label_sets)
    label_count = len(labels)
    class_counts = {
        label: sum(1 for sample_labels in sample_label_sets if label in sample_labels)
        for label in labels
    }
    maximum = max(class_counts.values(), default=0)
    minimum = min(class_counts.values(), default=0)
    imbalance_ratio = rounded(float(maximum) / float(max(1, minimum)), 8)
    bucket = scale_bucket(sample_count)
    label_ratio = float(label_count) / float(max(1, len(config.labels)))
    cost_weight = float(config.evaluation["cost_weight"])
    candidates: list[dict[str, Any]] = []
    strategy_by_id = {strategy["id"]: strategy for strategy in config.strategies}
    for strategy in sorted(config.strategies, key=lambda item: item["id"]):
        feature = config.feature_params(strategy, precision=precision)
        classifier = str(strategy["classifier"])
        classifier_cost = _classifier_cost(config, classifier, strategy)
        estimated_cost = rounded(feature["feature_dimension"] / 24.0 * classifier_cost, 8)
        score = float(strategy.get("base_score", 1.0))
        score += float(strategy.get("scale_bonus", {}).get(bucket, 0.0))
        imbalance_bonus = 0.0
        if imbalance_ratio > 1.6:
            imbalance_bonus = float(strategy.get("imbalance_bonus", 0.0))
            score += imbalance_bonus
        label_bonus = float(strategy.get("label_bonus", 0.0)) * label_ratio
        score += label_bonus
        score -= cost_weight * estimated_cost
        candidates.append(
            {
                "id": strategy["id"],
                "score": rounded(score, 10),
                "estimated_cost": estimated_cost,
                "feature_method": strategy["feature_method"],
                "feature_dimension": feature["feature_dimension"],
                "classifier": classifier,
                "breakdown": {
                    "base_score": rounded(float(strategy.get("base_score", 1.0)), 10),
                    "scale_bonus": rounded(float(strategy.get("scale_bonus", {}).get(bucket, 0.0)), 10),
                    "imbalance_bonus": rounded(imbalance_bonus, 10),
                    "label_bonus": rounded(label_bonus, 10),
                    "cost_penalty": rounded(cost_weight * estimated_cost, 10),
                },
            }
        )
    ranked = sorted(candidates, key=lambda item: (-item["score"], item["id"]))
    auto = strategy_by_id[ranked[0]["id"]] if ranked else None
    if auto is None:
        raise ValueError("no strategy candidates available")
    override_notes: list[str] = []
    if force_strategy is not None:
        if force_strategy not in strategy_by_id:
            raise ValueError(f"unknown forced strategy: {force_strategy}")
        auto = strategy_by_id[force_strategy]
        override_notes.append(f"strategy force-switched to {force_strategy}")
    selected = dict(auto)
    effective_feature: dict[str, Any]
    effective_classifier: str = str(selected["classifier"])
    if force_feature_method is not None:
        if force_feature_method not in config.features["supported"]:
            raise ValueError(f"unsupported forced feature method: {force_feature_method}")
        selected["feature_method"] = force_feature_method
        selected["color_bins"] = int(config.features["color_bins"])
        selected["hog_bins"] = int(config.features["hog_bins"])
        override_notes.append(f"feature force-switched to {force_feature_method}")
    if force_color_bins is not None:
        selected["color_bins"] = int(force_color_bins)
        override_notes.append(f"color bins force-set to {force_color_bins}")
    if force_hog_bins is not None:
        selected["hog_bins"] = int(force_hog_bins)
        override_notes.append(f"HOG bins force-set to {force_hog_bins}")
    if force_classifier is not None:
        if force_classifier not in config.classifiers["supported"]:
            raise ValueError(f"unsupported forced classifier: {force_classifier}")
        effective_classifier = force_classifier
        selected["classifier"] = force_classifier
        override_notes.append(f"classifier force-switched to {force_classifier}")
    if force_knn_k is not None:
        if effective_classifier != "knn":
            raise ValueError("knn-k override requires the kNN classifier")
        if force_knn_k not in config.classifiers["knn"]["k_candidates"]:
            raise ValueError("knn-k override must be one of classifiers.knn.k_candidates")
        selected["knn_k"] = int(force_knn_k)
        override_notes.append(f"kNN k force-set to {force_knn_k}")
    selected["classifier"] = effective_classifier
    effective_feature = config.feature_params(selected, precision=precision)
    key = (
        effective_feature["feature_method"],
        int(effective_feature["color_bins"]),
        int(effective_feature["hog_bins"]),
    )
    if key not in config.expected_dimensions:
        raise ValueError(
            "forced feature combination has no declared expected dimension: "
            f"{key[0]} color={key[1]} hog={key[2]}"
        )
    if config.expected_dimensions[key] != effective_feature["feature_dimension"]:
        raise ValueError("forced feature dimension does not match configuration")
    classifier_params = config.classifier_params(effective_classifier, selected)
    effective_estimated_cost = rounded(
        effective_feature["feature_dimension"] / 24.0
        * _classifier_cost(config, effective_classifier, selected),
        8,
    )
    base_threshold = float(config.threshold)
    threshold_delta = float(selected.get("threshold_delta", 0.0))
    if force_threshold is not None:
        threshold = float(force_threshold)
        override_notes.append(f"threshold force-set to {threshold}")
    else:
        threshold = min(1.0, max(0.0, base_threshold + threshold_delta))
    threshold = rounded(threshold, 10)
    selected_id = str(auto["id"])
    effective_id = selected_id
    if override_notes:
        effective_id = selected_id + ":overridden"
    reason_parts = [
        f"bucket={bucket}",
        f"samples={sample_count}",
        f"labels={label_count}",
        f"imbalance={imbalance_ratio}",
        f"highest_score={ranked[0]['score'] if ranked else 0}",
    ]
    if imbalance_ratio > 1.6:
        reason_parts.append("imbalance_bonus_applied")
    if override_notes:
        reason_parts.append("; ".join(override_notes))
    else:
        reason_parts.append(f"auto-selected {selected_id}")
    return {
        "tier_id": tier["id"],
        "input_metrics": {
            "sample_count": sample_count,
            "label_count": label_count,
            "max_class_count": maximum,
            "min_class_count": minimum,
            "imbalance_ratio": imbalance_ratio,
            "scale_bucket": bucket,
            "tier_max_samples": tier["max_samples"],
            "tier_max_labels": tier["max_labels"],
            "tier_estimated_cost": effective_estimated_cost,
        },
        "precision_level": precision or config.precision,
        "candidates": candidates,
        "selected_strategy_id": selected_id,
        "effective_strategy_id": effective_id,
        "selected": {
            "feature_method": effective_feature["feature_method"],
            "color_bins": effective_feature["color_bins"],
            "hog_bins": effective_feature["hog_bins"],
            "feature_dimension": effective_feature["feature_dimension"],
            "classifier": effective_classifier,
            "classifier_hyperparameters": classifier_params,
            "feature_parameters": effective_feature,
        },
        "threshold": {
            "base": base_threshold,
            "delta": threshold_delta,
            "final": threshold,
        },
        "overrides": {
            "strategy": force_strategy,
            "feature_method": force_feature_method,
            "color_bins": force_color_bins,
            "hog_bins": force_hog_bins,
            "classifier": force_classifier,
            "knn_k": force_knn_k,
            "threshold": force_threshold,
            "precision": precision,
        },
        "reason": "; ".join(reason_parts),
    }
