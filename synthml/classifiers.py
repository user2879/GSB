"""Deterministic one-vs-rest multi-label classifiers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .util import rounded


@dataclass(frozen=True)
class TrainingInstance:
    name: str
    features: np.ndarray
    labels: tuple[str, ...]


def _sigmoid(values: np.ndarray | float) -> np.ndarray | float:
    array = np.asarray(values, dtype=np.float64)
    positive = array >= 0
    result = np.empty_like(array, dtype=np.float64)
    result[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponent = np.exp(array[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    if np.ndim(values) == 0:
        return float(result)
    return result


def _clip_probability(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def _matrix(instances: list[TrainingInstance]) -> np.ndarray:
    if not instances:
        return np.zeros((0, 0), dtype=np.float64)
    return np.stack([instance.features for instance in instances], axis=0).astype(np.float64)


def _label_targets(instances: list[TrainingInstance], label: str) -> np.ndarray:
    return np.array([1.0 if label in instance.labels else 0.0 for instance in instances], dtype=np.float64)


def _base_payload(classifier: str, params: dict[str, Any], labels: list[str], instances: list[TrainingInstance]) -> dict[str, Any]:
    return {
        "classifier": classifier,
        "params": params,
        "labels": list(labels),
        "training_order": [instance.name for instance in instances],
        "training_sample_count": len(instances),
    }


def train_logistic_regression(
    params: dict[str, Any],
    labels: list[str],
    instances: list[TrainingInstance],
) -> dict[str, Any]:
    matrix = _matrix(instances)
    if matrix.size == 0:
        raise ValueError("logistic regression requires at least one training sample")
    mean = matrix.mean(axis=0)
    standard_deviation = matrix.std(axis=0)
    standard_deviation = np.where(standard_deviation < 1e-12, 1.0, standard_deviation)
    normalized = (matrix - mean) / standard_deviation
    learning_rate = float(params["learning_rate"])
    epochs = int(params["epochs"])
    l2 = float(params["l2"])
    epoch_losses: dict[str, list[float]] = {}
    weights_by_label: dict[str, list[float]] = {}
    biases: dict[str, float] = {}
    references: dict[str, dict[str, list[str]]] = {}
    training_names = [instance.name for instance in instances]
    for label in labels:
        target = _label_targets(instances, label)
        positive_count = int(target.sum())
        prior = (positive_count + 0.5) / (len(target) + 1.0)
        bias = math.log(prior / (1.0 - prior))
        weights = np.zeros(normalized.shape[1], dtype=np.float64)
        losses: list[float] = []
        for _epoch in range(epochs):
            scores = normalized @ weights + bias
            probabilities = _sigmoid(scores)
            errors = probabilities - target
            gradient = (normalized.T @ errors) / len(target) + l2 * weights
            gradient_bias = float(errors.mean())
            weights -= learning_rate * gradient
            bias -= learning_rate * gradient_bias
            probabilities = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
            loss = -float(
                np.mean(target * np.log(probabilities) + (1.0 - target) * np.log(1.0 - probabilities))
            )
            loss += 0.5 * l2 * float(np.dot(weights, weights))
            losses.append(rounded(loss, 10))
        epoch_losses[label] = losses
        weights_by_label[label] = [rounded(value, 12) for value in weights.tolist()]
        biases[label] = rounded(bias, 12)
        references[label] = {"training": sorted(training_names), "positive": sorted(
            instance.name for instance in instances if label in instance.labels
        )}
    payload = _base_payload("logistic_regression", params, labels, instances)
    payload.update(
        {
            "feature_mean": [rounded(value, 12) for value in mean.tolist()],
            "feature_std": [rounded(value, 12) for value in standard_deviation.tolist()],
            "weights": weights_by_label,
            "biases": biases,
            "loss_history": epoch_losses,
            "references": references,
        }
    )
    return payload


def train_nearest_centroid(
    params: dict[str, Any],
    labels: list[str],
    instances: list[TrainingInstance],
) -> dict[str, Any]:
    matrix = _matrix(instances)
    if matrix.size == 0:
        raise ValueError("nearest centroid requires at least one training sample")
    positive_centroids: dict[str, list[float] | None] = {}
    negative_centroids: dict[str, list[float] | None] = {}
    references: dict[str, dict[str, list[str]]] = {}
    for label in labels:
        target = _label_targets(instances, label).astype(bool)
        positive = matrix[target]
        negative = matrix[~target]
        positive_centroids[label] = (
            [rounded(value, 12) for value in positive.mean(axis=0).tolist()] if positive.size else None
        )
        negative_centroids[label] = (
            [rounded(value, 12) for value in negative.mean(axis=0).tolist()] if negative.size else None
        )
        references[label] = {
            "training": sorted(instance.name for instance in instances),
            "positive": sorted(instance.name for instance, flag in zip(instances, target) if flag),
        }
    payload = _base_payload("nearest_centroid", params, labels, instances)
    payload.update(
        {
            "positive_centroids": positive_centroids,
            "negative_centroids": negative_centroids,
            "references": references,
        }
    )
    return payload


def train_knn(
    params: dict[str, Any],
    labels: list[str],
    instances: list[TrainingInstance],
) -> dict[str, Any]:
    if not instances:
        raise ValueError("kNN requires at least one training sample")
    training = [
        {
            "name": instance.name,
            "features": [rounded(value, 12) for value in instance.features.tolist()],
            "labels": list(instance.labels),
        }
        for instance in instances
    ]
    references: dict[str, dict[str, list[str]]] = {}
    for label in labels:
        references[label] = {
            "training": sorted(instance.name for instance in instances),
            "positive": sorted(instance.name for instance in instances if label in instance.labels),
        }
    payload = _base_payload("knn", params, labels, instances)
    payload.update(
        {
            "k": int(params["k"]),
            "effective_k": min(int(params["k"]), len(instances)),
            "training_vectors": training,
            "references": references,
        }
    )
    return payload


def train_model(
    classifier: str,
    params: dict[str, Any],
    labels: list[str],
    instances: list[TrainingInstance],
) -> dict[str, Any]:
    if classifier == "logistic_regression":
        return train_logistic_regression(params, labels, instances)
    if classifier == "nearest_centroid":
        return train_nearest_centroid(params, labels, instances)
    if classifier == "knn":
        return train_knn(params, labels, instances)
    raise ValueError(f"unknown classifier: {classifier}")


class ClassifierModel:
    def __init__(self, payload: dict[str, Any], probability_decimals: int) -> None:
        self.payload = payload
        self.classifier = str(payload["classifier"])
        self.labels = list(payload["labels"])
        self.probability_decimals = int(probability_decimals)
        self.params = dict(payload["params"])
        if self.classifier == "logistic_regression":
            self.weights = {
                label: np.asarray(payload["weights"][label], dtype=np.float64) for label in self.labels
            }
            self.biases = {label: float(payload["biases"][label]) for label in self.labels}
            self.feature_mean = np.asarray(payload["feature_mean"], dtype=np.float64)
            self.feature_std = np.asarray(payload["feature_std"], dtype=np.float64)
        elif self.classifier == "nearest_centroid":
            self.positive_centroids = {
                label: (
                    np.asarray(payload["positive_centroids"][label], dtype=np.float64)
                    if payload["positive_centroids"][label] is not None
                    else None
                )
                for label in self.labels
            }
            self.negative_centroids = {
                label: (
                    np.asarray(payload["negative_centroids"][label], dtype=np.float64)
                    if payload["negative_centroids"][label] is not None
                    else None
                )
                for label in self.labels
            }
        elif self.classifier == "knn":
            self.k = int(payload["effective_k"])
            self.training_names = [str(item["name"]) for item in payload["training_vectors"]]
            self.training_vectors = np.asarray(
                [item["features"] for item in payload["training_vectors"]], dtype=np.float64
            )
            self.training_label_sets = [set(item["labels"]) for item in payload["training_vectors"]]
        else:
            raise ValueError(f"unknown model classifier: {self.classifier}")

    def score(self, sample_name: str, features: np.ndarray) -> dict[str, Any]:
        if self.classifier == "logistic_regression":
            normalized = (features.astype(np.float64) - self.feature_mean) / self.feature_std
            raw_scores: dict[str, float] = {}
            probabilities: dict[str, float] = {}
            for label in self.labels:
                score = float(normalized @ self.weights[label] + self.biases[label])
                probability = float(_sigmoid(score))
                raw_scores[label] = rounded(score, 10)
                probabilities[label] = rounded(_clip_probability(probability), self.probability_decimals)
            references = {
                label: list(self.payload["references"][label]["training"]) for label in self.labels
            }
        elif self.classifier == "nearest_centroid":
            raw_scores = {}
            probabilities = {}
            for label in self.labels:
                positive_centroid = self.positive_centroids[label]
                negative_centroid = self.negative_centroids[label]
                if positive_centroid is None:
                    score = -1.0
                elif negative_centroid is None:
                    score = 1.0
                else:
                    positive_distance = float(np.linalg.norm(features - positive_centroid))
                    negative_distance = float(np.linalg.norm(features - negative_centroid))
                    score = (negative_distance - positive_distance) / (
                        negative_distance + positive_distance + 1e-12
                    )
                probability = float(_sigmoid(4.0 * score))
                raw_scores[label] = rounded(score, 10)
                probabilities[label] = rounded(_clip_probability(probability), self.probability_decimals)
            references = {
                label: list(self.payload["references"][label]["training"]) for label in self.labels
            }
        elif self.classifier == "knn":
            distances = np.linalg.norm(self.training_vectors - features.astype(np.float64), axis=1)
            ordering = sorted(
                range(len(self.training_names)),
                key=lambda index: (float(distances[index]), self.training_names[index]),
            )
            neighbors = ordering[: self.k]
            neighbor_names = [self.training_names[index] for index in neighbors]
            raw_scores = {}
            probabilities = {}
            for label in self.labels:
                votes = sum(label in self.training_label_sets[index] for index in neighbors)
                raw_scores[label] = rounded(votes / float(self.k), 10)
                probability = (votes + 1.0) / (self.k + 2.0)
                probabilities[label] = rounded(
                    _clip_probability(probability), self.probability_decimals
                )
            references = {label: list(neighbor_names) for label in self.labels}
        else:  # pragma: no cover - constructor rejects this path
            raise AssertionError("unreachable")
        return {
            "sample": sample_name,
            "classifier": self.classifier,
            "raw_scores": {label: raw_scores[label] for label in self.labels},
            "probabilities": {label: probabilities[label] for label in self.labels},
            "references": {label: references[label] for label in self.labels},
        }


def decide_labels(
    probabilities: dict[str, float], threshold: float, boundary_policy: str
) -> dict[str, Any]:
    if boundary_policy != "equality_positive_labels_lexicographic":
        raise ValueError(f"unsupported boundary policy: {boundary_policy}")
    sorted_labels = sorted(probabilities)
    boundary_labels: list[str] = []
    included: list[str] = []
    decisions: dict[str, dict[str, Any]] = {}
    for rank, label in enumerate(sorted_labels):
        score = float(probabilities[label])
        if score > threshold:
            comparison = "greater_than_threshold"
            positive = True
        elif score == threshold:
            comparison = "equal_to_threshold_boundary"
            positive = True
            boundary_labels.append(label)
        else:
            comparison = "less_than_threshold"
            positive = False
        if positive:
            included.append(label)
        decisions[label] = {
            "probability": score,
            "lexicographic_rank": rank,
            "comparison": comparison,
            "included": positive,
        }
    return {
        "predicted_labels": sorted(included),
        "boundary_labels": sorted(boundary_labels),
        "boundary": bool(boundary_labels),
        "boundary_policy": boundary_policy,
        "decisions": decisions,
    }
