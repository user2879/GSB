"""Clustering metrics, density likelihood, reproducibility and ranking."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from .config import Candidate
from .errors import E_INPUT_NON_FINITE, BusinessError
from .models import FitResult
from .util import max_abs_diff


METRIC_NAMES = ("silhouette", "calinski_harabasz", "log_likelihood")


def assert_finite_matrix(matrix: np.ndarray, stage: str = "evaluation") -> None:
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            if not math.isfinite(float(matrix[row, column])):
                raise BusinessError(
                    E_INPUT_NON_FINITE,
                    f"non-finite feature at sample {row}, feature {column}",
                    stage=stage,
                    path=f"matrix[{row}][{column}]",
                    extra={"sample": row, "feature_index": column},
                )


@dataclass
class Evaluation:
    candidate: Candidate
    fit: FitResult
    metrics: dict[str, float | None]
    metric_diagnostics: dict[str, Any]
    reproducibility: dict[str, Any]
    diagnostics: dict[str, Any]
    rank_score: float | None = None
    rank: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "fit": self.fit.to_dict(),
            "metrics": self.metrics,
            "metric_diagnostics": self.metric_diagnostics,
            "reproducibility": self.reproducibility,
            "diagnostics": self.diagnostics,
            "rank_score": self.rank_score,
            "rank": self.rank,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Evaluation":
        candidate_payload = payload["candidate"]
        candidate = Candidate(
            method=candidate_payload["method"],
            params=candidate_payload["params"],
            params_key=candidate_payload["params_key"],
            sort_key=candidate_payload["sort_key"],
        )
        return cls(
            candidate=candidate,
            fit=FitResult.from_dict(payload["fit"]),
            metrics=payload["metrics"],
            metric_diagnostics=payload["metric_diagnostics"],
            reproducibility=payload["reproducibility"],
            diagnostics=payload["diagnostics"],
            rank_score=payload.get("rank_score"),
            rank=payload.get("rank"),
        )


def _safe_metric(name: str, function: Any, labels: list[int], matrix: np.ndarray) -> tuple[float | None, dict[str, Any]]:
    unique = sorted(set(labels))
    valid_unique = [label for label in unique if label != -1]
    if len(unique) < 2 or len(valid_unique) == 0:
        return None, {
            "status": "unavailable",
            "reason": "fewer than two distinct assignment labels",
            "labels": unique,
        }
    try:
        value = float(function(matrix, labels))
        if not math.isfinite(value):
            raise ValueError("metric returned a non-finite value")
        return value, {"status": "ok", "labels": unique}
    except Exception as exc:  # sklearn uses several exception classes for degenerate inputs
        return None, {
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "labels": unique,
        }


def _density_log_likelihood(matrix: np.ndarray, labels: list[int]) -> tuple[float | None, dict[str, Any]]:
    unique_labels = sorted(set(labels))
    if not unique_labels:
        return None, {"status": "unavailable", "reason": "no labels"}
    rows, columns = matrix.shape
    components: list[dict[str, Any]] = []
    for label in unique_labels:
        indices = [index for index, value in enumerate(labels) if value == label]
        if not indices:
            continue
        mean = np.asarray(
            [
                math.fsum(float(matrix[index, column]) for index in indices) / len(indices)
                for column in range(columns)
            ],
            dtype=float,
        )
        variance = np.asarray(
            [
                max(
                    math.fsum((float(matrix[index, column]) - mean[column]) ** 2 for index in indices)
                    / len(indices),
                    1e-6,
                )
                for column in range(columns)
            ],
            dtype=float,
        )
        components.append(
            {
                "label": label,
                "weight": len(indices) / rows,
                "mean": mean,
                "variance": variance,
            }
        )
    if not components:
        return None, {"status": "unavailable", "reason": "no components"}
    normalizer = columns * math.log(2.0 * math.pi)
    log_likelihoods: list[float] = []
    for row in range(rows):
        log_terms: list[float] = []
        for component in components:
            variance = component["variance"]
            mean = component["mean"]
            mahalanobis = math.fsum(
                ((float(matrix[row, column]) - float(mean[column])) ** 2) / float(variance[column])
                for column in range(columns)
            )
            log_det = math.fsum(math.log(float(value)) for value in variance)
            log_terms.append(
                math.log(float(component["weight"]))
                - 0.5 * (normalizer + log_det + mahalanobis)
            )
        maximum = max(log_terms)
        log_sum = maximum + math.log(math.fsum(math.exp(term - maximum) for term in log_terms))
        log_likelihoods.append(log_sum)
    mean_likelihood = math.fsum(log_likelihoods) / len(log_likelihoods)
    return mean_likelihood, {
        "status": "ok",
        "component_count": len(components),
        "covariance": "diagonal_population_with_floor_1e-6",
        "mean_log_likelihood": mean_likelihood,
    }


def evaluate_fit(candidate: Candidate, fit: FitResult, matrix: np.ndarray) -> Evaluation:
    assert_finite_matrix(matrix)
    labels = fit.labels
    silhouette, silhouette_diag = _safe_metric(
        "silhouette", silhouette_score, labels, matrix
    )
    calinski, calinski_diag = _safe_metric(
        "calinski_harabasz", calinski_harabasz_score, labels, matrix
    )
    if fit.density_log_likelihood is not None and math.isfinite(float(fit.density_log_likelihood)):
        log_likelihood = float(fit.density_log_likelihood)
        likelihood_diag = {
            "status": "ok",
            "source": "fitted_gaussian_mixture_score_samples",
            "mean_log_likelihood": log_likelihood,
        }
    else:
        log_likelihood, likelihood_diag = _density_log_likelihood(matrix, labels)
        likelihood_diag["source"] = "cluster_diagonal_density"
    metrics = {
        "silhouette": silhouette,
        "calinski_harabasz": calinski,
        "log_likelihood": log_likelihood,
    }
    diagnostics = dict(fit.diagnostics)
    if diagnostics.get("all_noise"):
        diagnostics["all_noise_note"] = "all samples are noise; candidate retained in ranking"
    if diagnostics.get("empty_clusters"):
        diagnostics["empty_cluster_note"] = "empty clusters are diagnostic, not discarded"
    return Evaluation(
        candidate=candidate,
        fit=fit,
        metrics=metrics,
        metric_diagnostics={
            "silhouette": silhouette_diag,
            "calinski_harabasz": calinski_diag,
            "log_likelihood": likelihood_diag,
        },
        reproducibility={},
        diagnostics=diagnostics,
    )


def attach_reproducibility(
    evaluation: Evaluation,
    first: Evaluation,
) -> None:
    metric_diffs: dict[str, float | None] = {}
    metric_availability_changed = False
    for metric in METRIC_NAMES:
        left = first.metrics.get(metric)
        right = evaluation.metrics.get(metric)
        if left is None or right is None:
            metric_diffs[metric] = None
            if (left is None) != (right is None):
                metric_availability_changed = True
        else:
            metric_diffs[metric] = abs(float(left) - float(right))

    labels_equal = first.fit.labels == evaluation.fit.labels
    assignment_diff = sum(
        1 for left, right in zip(first.fit.labels, evaluation.fit.labels) if left != right
    )
    responsibility_diff: float | None = None
    if first.fit.responsibilities is not None and evaluation.fit.responsibilities is not None:
        flattened_first = [value for row in first.fit.responsibilities for value in row]
        flattened_second = [value for row in evaluation.fit.responsibilities for value in row]
        responsibility_diff = max_abs_diff(flattened_first, flattened_second)
    deterministic = labels_equal and not metric_availability_changed and all(
        diff is None or diff <= 1e-12 for diff in metric_diffs.values()
    )
    if responsibility_diff is not None and responsibility_diff > 1e-12:
        deterministic = False
    evaluation.reproducibility = {
        "second_run_is_independent": True,
        "labels_equal": labels_equal,
        "assignment_diff_count": assignment_diff,
        "metric_abs_diffs": metric_diffs,
        "metric_availability_changed": metric_availability_changed,
        "responsibility_max_abs_diff": responsibility_diff,
        "deterministic": deterministic,
    }


def _finite_values(values: list[float | None]) -> list[float]:
    return [float(value) for value in values if value is not None and math.isfinite(float(value))]


def _normalize(value: float | None, finite: list[float]) -> float:
    if value is None or not finite:
        return 0.0
    minimum = min(finite)
    maximum = max(finite)
    if maximum - minimum <= 1e-15:
        return 1.0
    return (float(value) - minimum) / (maximum - minimum)


def build_ranking(evaluations: list[Evaluation], evaluation_config: dict[str, Any]) -> list[Evaluation]:
    weights = evaluation_config["weights"]
    primary = evaluation_config["primary"]
    finite_by_metric = {
        metric: _finite_values([evaluation.metrics.get(metric) for evaluation in evaluations])
        for metric in METRIC_NAMES
    }
    for evaluation in evaluations:
        composite = math.fsum(
            float(weights.get(metric, 0.0))
            * _normalize(evaluation.metrics.get(metric), finite_by_metric[metric])
            for metric in METRIC_NAMES
        )
        weight_sum = math.fsum(float(weights.get(metric, 0.0)) for metric in METRIC_NAMES)
        evaluation.rank_score = composite / weight_sum if weight_sum > 0 else 0.0
        evaluation.diagnostics["ranking_primary"] = primary
        evaluation.diagnostics["composite_score"] = evaluation.rank_score

    def score_for(evaluation: Evaluation) -> float:
        if primary == "composite":
            value = evaluation.rank_score
        else:
            value = evaluation.metrics.get(primary)
        if value is None or not math.isfinite(float(value)):
            return -math.inf
        return float(value)

    ranked = sorted(
        evaluations,
        key=lambda evaluation: (
            -score_for(evaluation),
            evaluation.candidate.method,
            evaluation.candidate.params_key,
            evaluation.candidate.sort_key,
        ),
    )
    for rank, evaluation in enumerate(ranked, start=1):
        evaluation.rank = rank
        evaluation.diagnostics["rank"] = rank
    return ranked


def metrics_table(evaluations: list[Evaluation]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for evaluation in sorted(evaluations, key=lambda item: item.candidate.sort_key):
        rows.append(
            {
                "sort_key": evaluation.candidate.sort_key,
                "method": evaluation.candidate.method,
                "params": evaluation.candidate.params,
                "metrics": evaluation.metrics,
                "rank": evaluation.rank,
            }
        )
    return rows
