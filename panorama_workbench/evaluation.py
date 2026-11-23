from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np

from .config import WorkbenchConfig
from .failures import Failure
from .merging import MergedInstance
from .validation import Sample, TruthInstance


@dataclass
class EvaluationResult:
    semantic_accuracy: float
    instance_quality: float
    mean_matched_iou: float
    true_positive_count: int
    false_positive_count: int
    false_negative_count: int
    matched_pairs: tuple[dict[str, Any], ...]
    predicted_diagnostics: tuple[dict[str, Any], ...]
    unmatched_predicted_instances: tuple[dict[str, Any], ...]
    missed_truth_instances: tuple[dict[str, Any], ...]
    failures: tuple[Failure, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "false_negative_count": self.false_negative_count,
            "false_positive_count": self.false_positive_count,
            "instance_quality": self.instance_quality,
            "matched_pairs": list(self.matched_pairs),
            "mean_matched_iou": self.mean_matched_iou,
            "missed_truth_instances": list(self.missed_truth_instances),
            "panoptic_quality": self.instance_quality,
            "predicted_diagnostics": list(self.predicted_diagnostics),
            "semantic_accuracy": self.semantic_accuracy,
            "true_positive_count": self.true_positive_count,
            "unmatched_predicted_instances": list(
                self.unmatched_predicted_instances
            ),
        }


def evaluate_predictions(
    sample: Sample,
    instances: tuple[MergedInstance, ...],
    semantic_raster: np.ndarray,
    instance_raster: np.ndarray,
    config: WorkbenchConfig,
) -> EvaluationResult:
    truth_semantic = sample.truth_semantic_raster()
    total_pixels = sample.width * sample.height
    correct = int((semantic_raster == truth_semantic).sum())
    semantic_accuracy = correct / total_pixels if total_pixels else 1.0

    candidates: list[tuple[Fraction, MergedInstance, TruthInstance]] = []
    truth_masks = {
        truth.instance_id: truth.mask(sample.width, sample.height)
        for truth in sample.truth_instances
    }
    for prediction in instances:
        pred_pixels = prediction.pixels
        for truth in sample.truth_instances:
            if truth.category_code != prediction.category_code:
                continue
            truth_mask = truth_masks[truth.instance_id]
            truth_pixels = set(np.flatnonzero(truth_mask).tolist())
            intersection = len(set(pred_pixels).intersection(truth_pixels))
            if intersection == 0:
                continue
            union = len(pred_pixels) + len(truth_pixels) - intersection
            candidates.append((Fraction(intersection, union), prediction, truth))

    candidates.sort(
        key=lambda item: (-item[0], item[1].instance_id, item[2].instance_id)
    )
    threshold = Fraction(str(config.instance_match_iou))
    matched_predictions: set[int] = set()
    matched_truth: set[int] = set()
    matched_pairs: list[dict[str, Any]] = []
    for iou, prediction, truth in candidates:
        if iou < threshold:
            continue
        if (
            prediction.instance_id in matched_predictions
            or truth.instance_id in matched_truth
        ):
            continue
        matched_predictions.add(prediction.instance_id)
        matched_truth.add(truth.instance_id)
        matched_pairs.append(
            {
                "category_code": prediction.category_code,
                "instance_id": prediction.instance_id,
                "iou": float(iou),
                "truth_instance_id": truth.instance_id,
            }
        )
    matched_pairs.sort(key=lambda item: item["instance_id"])

    true_positive = len(matched_pairs)
    false_positive = len(instances) - true_positive
    false_negative = len(sample.truth_instances) - true_positive
    denominator = true_positive + 0.5 * false_positive + 0.5 * false_negative
    if denominator == 0:
        instance_quality = 1.0
    else:
        instance_quality = sum(pair["iou"] for pair in matched_pairs) / denominator
    mean_iou = (
        sum(pair["iou"] for pair in matched_pairs) / true_positive
        if true_positive
        else 0.0
    )

    match_by_prediction = {
        pair["instance_id"]: pair for pair in matched_pairs
    }
    diagnostics: list[dict[str, Any]] = []
    unmatched_predicted: list[dict[str, Any]] = []
    for prediction in instances:
        matched = match_by_prediction.get(prediction.instance_id)
        diagnostic = {
            "area": prediction.area,
            "category_code": prediction.category_code,
            "instance_id": prediction.instance_id,
            "matched_iou": None if matched is None else matched["iou"],
            "matched_truth_instance_id": (
                None if matched is None else matched["truth_instance_id"]
            ),
            "reason_code": (
                None if matched is not None else "E_INSTANCE_FALSE_POSITIVE"
            ),
        }
        diagnostics.append(diagnostic)
        if matched is None:
            unmatched_predicted.append(dict(diagnostic))
    missed_truth: list[dict[str, Any]] = []
    for truth in sorted(sample.truth_instances, key=lambda item: item.instance_id):
        if truth.instance_id in matched_truth:
            continue
        missed_truth.append(
            {
                "area": truth.area,
                "category_code": truth.category_code,
                "reason_code": "E_INSTANCE_FALSE_NEGATIVE",
                "truth_instance_id": truth.instance_id,
            }
        )

    failures: list[Failure] = []
    if (
        semantic_accuracy < config.min_semantic_accuracy
        or instance_quality < config.min_instance_quality
    ):
        failures.append(
            Failure(
                code="E_QUALITY_BELOW",
                item=sample.name,
                stage="evaluation",
                message="semantic accuracy or instance quality is below configured threshold",
                details={
                    "instance_quality": _rounded(instance_quality),
                    "min_instance_quality": config.min_instance_quality,
                    "min_semantic_accuracy": config.min_semantic_accuracy,
                    "semantic_accuracy": _rounded(semantic_accuracy),
                },
            )
        )

    return EvaluationResult(
        semantic_accuracy=_rounded(semantic_accuracy),
        instance_quality=_rounded(instance_quality),
        mean_matched_iou=_rounded(mean_iou),
        true_positive_count=true_positive,
        false_positive_count=false_positive,
        false_negative_count=false_negative,
        matched_pairs=tuple(matched_pairs),
        predicted_diagnostics=tuple(diagnostics),
        unmatched_predicted_instances=tuple(unmatched_predicted),
        missed_truth_instances=tuple(missed_truth),
        failures=tuple(failures),
    )


def _rounded(value: float) -> float:
    return round(float(value), 12)
