from __future__ import annotations

import math
from typing import Any

import numpy as np

from .fingerprint import stable_float


def _bin_for_value(artifact: dict[str, Any], value: Any) -> dict[str, Any]:
    if value is None:
        missing_index = artifact.get("missing_index")
        if missing_index is None:
            raise ValueError(f"missing value cannot be encoded for column {artifact['column']}")
        return artifact["bins"][int(missing_index)]

    if artifact["type"] == "numeric":
        number = int(value)
        for bin_value in artifact["bins"]:
            if bin_value["is_missing"]:
                continue
            if int(bin_value["lower"]) <= number <= int(bin_value["upper"]):
                return bin_value
        raise ValueError(f"value {value!r} is outside persisted bins for column {artifact['column']}")

    mapping = artifact.get("category_to_group", {})
    key = str(value)
    if key not in mapping:
        raise ValueError(f"unknown category {value!r} for column {artifact['column']}")
    return artifact["bins"][int(mapping[key])]


def build_scores(
    columns: dict[str, list[Any]],
    target: list[Any],
    positive_value: Any,
    artifacts: dict[str, dict[str, Any]],
) -> tuple[list[float], list[int]]:
    rows = len(target)
    scores = [0.0] * rows
    for column in sorted(artifacts):
        artifact = artifacts[column]
        values = columns[column]
        for row in range(rows):
            bin_value = _bin_for_value(artifact, values[row])
            scores[row] += float(bin_value["woe"])
    scores = [stable_float(value) for value in scores]
    labels = [1 if value == positive_value else 0 for value in target]
    return scores, labels


def _auc(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: (scores[index], index))
    ranks = [0.0] * len(order)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and scores[order[end]] == scores[order[position]]:
            end += 1
        average_rank = (position + 1 + end) / 2.0
        for rank_index in range(position, end):
            ranks[order[rank_index]] = average_rank
        position = end
    positive_rank_sum = sum(ranks[index] for index, label in enumerate(labels) if label == 1)
    value = (positive_rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)
    return stable_float(value)


def _ks(scores: list[float], labels: list[int]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    order = sorted(range(len(scores)), key=lambda index: (scores[index], index))
    cumulative_bad = 0
    cumulative_good = 0
    best = 0.0
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and scores[order[end]] == scores[order[position]]:
            end += 1
        for index in order[position:end]:
            if labels[index] == 1:
                cumulative_bad += 1
            else:
                cumulative_good += 1
        best = max(best, abs(cumulative_bad / positives - cumulative_good / negatives))
        position = end
    return stable_float(best)


def _metrics(scores: list[float], labels: list[int]) -> dict[str, Any]:
    if not scores:
        return {
            "rows": 0,
            "auc": None,
            "ks": None,
            "log_loss": None,
            "brier": None,
            "accuracy": None,
            "event_rate": None,
        }
    log_loss = 0.0
    brier = 0.0
    correct = 0
    for score, label in zip(scores, labels):
        probability = 1.0 / (1.0 + math.exp(-max(-35.0, min(35.0, score))))
        clipped = max(1e-15, min(1.0 - 1e-15, probability))
        if label == 1:
            log_loss -= math.log(clipped)
        else:
            log_loss -= math.log(1.0 - clipped)
        brier += (probability - label) ** 2
        if (score >= 0.0 and label == 1) or (score < 0.0 and label == 0):
            correct += 1
    count = len(labels)
    return {
        "rows": count,
        "auc": _auc(scores, labels),
        "ks": _ks(scores, labels),
        "log_loss": stable_float(log_loss / count),
        "brier": stable_float(brier / count),
        "accuracy": stable_float(correct / count),
        "event_rate": stable_float(sum(labels) / count),
    }


def evaluate(
    columns: dict[str, list[Any]],
    target: list[Any],
    positive_value: Any,
    artifacts: dict[str, dict[str, Any]],
    folds: int,
    fold_seed: int,
) -> dict[str, Any]:
    rows = len(target)
    if not artifacts:
        return {
            "scored_columns": [],
            "folds": folds,
            "fold_seed": fold_seed,
            "fold_sizes": [],
            "overall": None,
            "per_fold": [],
        }
    scores, labels = build_scores(columns, target, positive_value, artifacts)
    permutation = np.random.default_rng(fold_seed).permutation(rows)
    fold_ids = [0] * rows
    for position, row in enumerate(permutation):
        fold_ids[int(row)] = position % folds

    per_fold: list[dict[str, Any]] = []
    for fold in range(folds):
        indices = [index for index, assigned in enumerate(fold_ids) if assigned == fold]
        fold_scores = [scores[index] for index in indices]
        fold_labels = [labels[index] for index in indices]
        per_fold.append({"fold": fold, "rows": len(indices), **_metrics(fold_scores, fold_labels)})

    return {
        "scored_columns": sorted(artifacts),
        "folds": folds,
        "fold_seed": fold_seed,
        "fold_sizes": [sum(1 for value in fold_ids if value == fold) for fold in range(folds)],
        "overall": _metrics(scores, labels),
        "per_fold": per_fold,
    }
