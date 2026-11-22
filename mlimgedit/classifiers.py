"""Stage 3: deterministic multi-label classifiers and per-label decisions.

Three classifiers are trained independently per label on the fixed-point
dequantized feature matrix:

* ``logreg``    - full-batch gradient descent with fixed learning rate, fixed
                  sample order and an auditable per-epoch loss curve;
* ``centroid``  - class-mean nearest centroid with Euclidean distance;
* ``knn``       - deterministic neighbor ordering plus uniform/distance voting.

Raw scores are mapped to probabilities through one shared stable sigmoid, then
quantized to fixed-point integers. Exact threshold equality is resolved by an
explicit tie rule and flagged as a boundary sample.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .errors import ConfigError
from .fixedpoint import FixedPointQuantizer, quantize_probability, threshold_integer

CLASSIFIER_ORDER = ("logreg", "centroid", "knn")


def stable_sigmoid(values: np.ndarray) -> np.ndarray:
    """Branch-free numerically stable sigmoid, identical on every run."""
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_pos = np.exp(values[~positive])
    result[~positive] = exp_pos / (1.0 + exp_pos)
    return result


def _standardize(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if matrix.size == 0:
        return matrix, np.zeros(0), np.ones(0)
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    safe_stds = np.where(stds > 1e-12, stds, 1.0)
    standardized = (matrix - means) / safe_stds
    return standardized.astype(np.float64), means, safe_stds


def _logistic_regression(matrix: np.ndarray, targets: np.ndarray, params: dict):
    n_samples, n_features = matrix.shape
    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0
    lr = float(params["learning_rate"])
    l2 = float(params["l2"])
    epochs = int(params["epochs"])
    losses: list[float] = []
    for _ in range(epochs):
        logits = matrix.dot(weights) + bias
        probabilities = stable_sigmoid(logits)
        # Full-batch gradient in the fixed sample order encoded by the matrix.
        error = probabilities - targets
        grad_w = matrix.T.dot(error) / max(n_samples, 1) + l2 * weights
        grad_b = float(error.sum() / max(n_samples, 1))
        weights = weights - lr * grad_w
        bias = bias - lr * grad_b
        eps = 1e-12
        loss = -(
            targets * np.log(probabilities + eps)
            + (1.0 - targets) * np.log(1.0 - probabilities + eps)
        ).mean()
        losses.append(round(float(loss), 10))
    raw = matrix.dot(weights) + bias
    return raw, {"losses": losses, "bias": float(bias), "weights": weights.tolist()}


def _centroid(matrix: np.ndarray, targets: np.ndarray, params: dict):
    positive_mask = targets == 1
    negative_mask = ~positive_mask
    if positive_mask.any():
        positive_center = matrix[positive_mask].mean(axis=0)
    else:
        positive_center = np.zeros(matrix.shape[1], dtype=np.float64)
    if negative_mask.any():
        negative_center = matrix[negative_mask].mean(axis=0)
    else:
        negative_center = np.zeros(matrix.shape[1], dtype=np.float64)
    distance_positive = np.sqrt(((matrix - positive_center) ** 2).sum(axis=1))
    distance_negative = np.sqrt(((matrix - negative_center) ** 2).sum(axis=1))
    # Closer to the positive centroid yields a larger raw score.
    raw = distance_negative - distance_positive
    return raw, {
        "distance": params["distance"],
        "positive_center": positive_center.tolist(),
        "negative_center": negative_center.tolist(),
    }


def _knn(matrix: np.ndarray, targets: np.ndarray, params: dict):
    k = min(int(params["k"]), max(1, matrix.shape[0]))
    raw = np.zeros(matrix.shape[0], dtype=np.float64)
    neighbor_table: list[list[int]] = []
    for idx in range(matrix.shape[0]):
        distances = np.sqrt(((matrix - matrix[idx]) ** 2).sum(axis=1))
        # Lexicographic (distance, sample index) gives a total deterministic order.
        order = sorted(range(matrix.shape[0]), key=lambda j: (float(distances[j]), j))
        neighbors = order[:k]
        neighbor_table.append(neighbors)
        neighbor_targets = targets[neighbors]
        if params["vote"] == "uniform":
            score = float(neighbor_targets.sum()) / float(k)
        else:
            neighbor_distances = distances[neighbors]
            weights = 1.0 / (neighbor_distances + 1e-6)
            if weights.sum() <= 0:
                score = float(neighbor_targets.sum()) / float(k)
            else:
                score = float((weights * neighbor_targets).sum() / weights.sum())
        # Map [0, 1] vote into a raw score whose sigmoid preserves ordering.
        raw[idx] = math.log((score + 1e-9) / (1.0 - score + 1e-9))
    return raw, {"k": k, "vote": params["vote"], "neighbors": neighbor_table}


_TRAINERS = {
    "logreg": _logistic_regression,
    "centroid": _centroid,
    "knn": _knn,
}


@dataclass(frozen=True)
class LabelScore:
    classifier: str
    label: str
    sample: str
    raw_score: float
    probability: float
    probability_int: int
    decision: bool
    boundary: bool
    threshold: float
    threshold_int: int
    tie_rule: str


@dataclass(frozen=True)
class LabelModelArtifact:
    classifier: str
    label: str
    referenced_samples: tuple[str, ...]
    scores: tuple[LabelScore, ...]
    trainer_meta: dict
    hyper_fingerprint_input: dict
    scale_tier: str

    def to_dict(self) -> dict:
        return {
            "classifier": self.classifier,
            "label": self.label,
            "referenced_samples": list(self.referenced_samples),
            "scale_tier": self.scale_tier,
            "hyper": self.hyper_fingerprint_input,
            "trainer_meta": self.trainer_meta,
            "scores": [
                {
                    "sample": score.sample,
                    "raw_score": round(score.raw_score, 10),
                    "probability": round(score.probability, 12),
                    "probability_int": score.probability_int,
                    "decision": score.decision,
                    "boundary": score.boundary,
                    "threshold": score.threshold,
                    "threshold_int": score.threshold_int,
                    "tie_rule": score.tie_rule,
                }
                for score in self.scores
            ],
        }


def hyper_fingerprint_input(classifier: str, config) -> dict:
    return {
        "classifier": classifier,
        "params": config.classifiers[classifier],
        "feature_mode": config.feature_mode,
        "bins": config.bins,
        "hog_cell": config.hog_cell,
        "quant_level": config.quant_level,
        "frac_bits": config.frac_bits,
    }


def train_label_model(
    classifier: str,
    label: str,
    sample_names: list[str],
    matrix: np.ndarray,
    label_matrix: np.ndarray,
    config,
    quantizer: FixedPointQuantizer,
    scale_tier: str,
) -> LabelModelArtifact:
    if classifier not in _TRAINERS:
        raise ConfigError(f"unknown classifier {classifier!r}")
    if len(sample_names) != matrix.shape[0]:
        raise ConfigError("sample/matrix size mismatch")
    standardized, means, stds = _standardize(matrix)
    targets = label_matrix.astype(np.float64)
    raw, trainer_meta = _TRAINERS[classifier](
        standardized, targets, config.classifiers[classifier]
    )
    trainer_meta["standardization_means"] = [round(float(v), 12) for v in means]
    trainer_meta["standardization_stds"] = [round(float(v), 12) for v in stds]
    probabilities = stable_sigmoid(raw)
    threshold = config.effective_threshold(label)
    threshold_int = threshold_integer(threshold, quantizer)
    tie_rule = "positive_on_equal"
    scores: list[LabelScore] = []
    for sample, raw_value, probability in zip(sample_names, raw, probabilities):
        probability_int = quantize_probability(float(probability), quantizer)
        boundary = probability_int == threshold_int
        # Exact integer comparison removes floating-point ambiguity at equality.
        decision = probability_int >= threshold_int
        scores.append(
            LabelScore(
                classifier=classifier,
                label=label,
                sample=sample,
                raw_score=float(raw_value),
                probability=float(probability),
                probability_int=probability_int,
                decision=bool(decision),
                boundary=bool(boundary),
                threshold=float(threshold),
                threshold_int=threshold_int,
                tie_rule=tie_rule,
            )
        )
    return LabelModelArtifact(
        classifier=classifier,
        label=label,
        referenced_samples=tuple(sample_names),
        scores=tuple(scores),
        trainer_meta=trainer_meta,
        hyper_fingerprint_input=hyper_fingerprint_input(classifier, config),
        scale_tier=scale_tier,
    )
