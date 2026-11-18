"""Evaluation metrics with matching batched and streaming algorithm paths.

The two paths are intentionally separate implementations. The pipeline runs
the streaming path (incremental friendly); tests assert both paths emit the
same values, which is how the scale-adaptive algorithm switch is validated.
"""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9_]+", re.IGNORECASE)


def _words(text: str) -> list:
    return [token.lower() for token in _WORD_RE.findall(text)]


def _numbers(text: str) -> list:
    return [float(token) for token in re.findall(r"[-+]?\d+(?:\.\d+)?", text)]


def per_sample(sample: dict) -> dict:
    model_words = _words(sample["model_answer"])
    reference_words = _words(sample["reference_answer"])
    model_counts, reference_counts = {}, {}
    for word in model_words:
        model_counts[word] = model_counts.get(word, 0) + 1
    for word in reference_words:
        reference_counts[word] = reference_counts.get(word, 0) + 1
    overlap = sum(min(count, reference_counts.get(word, 0)) for word, count in model_counts.items())

    model_numbers = _numbers(sample["model_answer"])
    reference_numbers = _numbers(sample["reference_answer"])
    if model_numbers and reference_numbers:
        pair_count = max(len(model_numbers), len(reference_numbers))
        padded_model = model_numbers + [0.0] * (pair_count - len(model_numbers))
        padded_reference = reference_numbers + [0.0] * (pair_count - len(reference_numbers))
        mae = sum(abs(a - b) for a, b in zip(padded_model, padded_reference)) / pair_count
    else:
        mae = None

    return {
        "sample_id": sample["sample_id"],
        "exact_match": int(" ".join(model_words) == " ".join(reference_words)),
        "tp": overlap,
        "pred_count": len(model_words),
        "ref_count": len(reference_words),
        "numeric_mae": mae,
    }


def _aggregate(rows: list) -> dict:
    count = len(rows)
    exact = sum(row["exact_match"] for row in rows)
    tp = sum(row["tp"] for row in rows)
    pred = sum(row["pred_count"] for row in rows)
    ref = sum(row["ref_count"] for row in rows)
    precision = tp / pred if pred else 0.0
    recall = tp / ref if ref else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    mae_rows = [row["numeric_mae"] for row in rows if row["numeric_mae"] is not None]
    return {
        "sample_count": count,
        "exact_match_rate": exact / count if count else 0.0,
        "token_precision": precision,
        "token_recall": recall,
        "token_f1": f1,
        "numeric_mae": (sum(mae_rows) / len(mae_rows)) if mae_rows else None,
    }


class StreamingAggregator:
    """Incremental aggregator that stores running integer/value accumulators.

    The final divisions are identical to :func:`_aggregate`, so the two
    algorithm paths produce bit-equal results.
    """

    def __init__(self):
        self.count = 0
        self.exact = 0
        self.tp = 0
        self.pred = 0
        self.ref = 0
        self.mae_sum = 0.0
        self.mae_count = 0

    def add(self, row: dict) -> None:
        self.count += 1
        self.exact += row["exact_match"]
        self.tp += row["tp"]
        self.pred += row["pred_count"]
        self.ref += row["ref_count"]
        if row["numeric_mae"] is not None:
            self.mae_sum += row["numeric_mae"]
            self.mae_count += 1

    def result(self) -> dict:
        precision = self.tp / self.pred if self.pred else 0.0
        recall = self.tp / self.ref if self.ref else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        return {
            "sample_count": self.count,
            "exact_match_rate": self.exact / self.count if self.count else 0.0,
            "token_precision": precision,
            "token_recall": recall,
            "token_f1": f1,
            "numeric_mae": (self.mae_sum / self.mae_count) if self.mae_count else None,
        }


def compute_metrics(samples: list, metric_names: tuple, streaming: bool = False) -> dict:
    rows = [per_sample(sample) for sample in samples]
    if streaming:
        aggregator = StreamingAggregator()
        for row in rows:
            aggregator.add(row)
        summary = aggregator.result()
    else:
        summary = _aggregate(rows)
    return {name: summary[name] for name in metric_names if name in summary}
