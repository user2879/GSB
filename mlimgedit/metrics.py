"""Stage 4: deterministic per-label and aggregate multi-label metrics."""

from __future__ import annotations

from dataclasses import dataclass

from .classifiers import LabelModelArtifact


def _safe_div(numerator: float, denominator: float) -> float:
    return round(float(numerator) / float(denominator), 10) if denominator > 0 else 0.0


@dataclass(frozen=True)
class MetricArtifact:
    classifier: str
    label: str
    threshold: float
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    boundary_count: int
    precision: float
    recall: float
    f1: float
    accuracy: float

    def to_dict(self) -> dict:
        return {
            "classifier": self.classifier,
            "label": self.label,
            "threshold": self.threshold,
            "tp": self.true_positive,
            "fp": self.false_positive,
            "tn": self.true_negative,
            "fn": self.false_negative,
            "boundary_count": self.boundary_count,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "accuracy": self.accuracy,
        }


def evaluate_label(
    model: LabelModelArtifact,
    ground_truth: dict[str, tuple[str, ...]],
) -> MetricArtifact:
    tp = fp = tn = fn = 0
    boundary_count = 0
    threshold = model.scores[0].threshold if model.scores else 0.0
    for score in model.scores:
        truth = score.label in ground_truth.get(score.sample, ())
        if score.boundary:
            boundary_count += 1
        if score.decision and truth:
            tp += 1
        elif score.decision and not truth:
            fp += 1
        elif not score.decision and truth:
            fn += 1
        else:
            tn += 1
    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if precision + recall > 0 else 0.0
    accuracy = _safe_div(tp + tn, tp + tn + fp + fn)
    return MetricArtifact(
        classifier=model.classifier,
        label=model.label,
        threshold=threshold,
        true_positive=tp,
        false_positive=fp,
        true_negative=tn,
        false_negative=fn,
        boundary_count=boundary_count,
        precision=precision,
        recall=recall,
        f1=f1,
        accuracy=accuracy,
    )


def summarize(metrics: list[MetricArtifact]) -> dict:
    ordered = sorted(metrics, key=lambda item: (item.classifier, item.label))
    by_classifier: dict[str, list[MetricArtifact]] = {}
    for metric in ordered:
        by_classifier.setdefault(metric.classifier, []).append(metric)
    summary: dict = {"per_label": {}, "aggregate": {}}
    for classifier, items in by_classifier.items():
        summary["per_label"][classifier] = [item.to_dict() for item in items]
        count = len(items)
        summary["aggregate"][classifier] = {
            "precision": round(sum(item.precision for item in items) / count, 10) if count else 0.0,
            "recall": round(sum(item.recall for item in items) / count, 10) if count else 0.0,
            "f1": round(sum(item.f1 for item in items) / count, 10) if count else 0.0,
            "accuracy": round(sum(item.accuracy for item in items) / count, 10) if count else 0.0,
            "boundary_count": sum(item.boundary_count for item in items),
            "label_count": count,
        }
    return summary
