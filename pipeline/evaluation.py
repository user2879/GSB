from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Optional

from .data import Qrel
from .errors import RelevanceGradeError
from .injections import Injection


def _dcg(grades: list[int]) -> float:
    return sum((2**grade - 1) / math.log2(index + 2) for index, grade in enumerate(grades))


def evaluate_query_result(
    query_result: dict,
    qrels: Iterable[Qrel],
    config: dict,
    injections: Iterable[Injection],
) -> dict:
    injection_list = sorted(injections, key=lambda item: item.injection_id)
    injection = injection_list[0] if injection_list else None
    judgments = {item.doc_id: int(item.grade) for item in qrels if item.query_id == query_result["query_id"]}
    max_grade = int(config["max_relevance_grade"])

    if injection is not None and injection.kind == "relevance_grade_out_of_range":
        target_doc = injection.params.get("doc_id")
        if not target_doc:
            if query_result.get("results"):
                target_doc = query_result["results"][0]["doc_id"]
            elif judgments:
                target_doc = sorted(judgments)[0]
            else:
                target_doc = "__injected_doc__"
        injected_grade = max_grade + 1
        judgments[str(target_doc)] = injected_grade
        if injected_grade > max_grade:
            raise RelevanceGradeError(
                f"injected relevance grade {injected_grade} exceeds configured maximum {max_grade} for {query_result['query_id']}/{target_doc}",
                injection_id=injection.injection_id,
            )
        raise AssertionError("unreachable")

    invalid = [(doc_id, grade) for doc_id, grade in sorted(judgments.items()) if grade < 0 or grade > max_grade]
    if invalid:
        doc_id, grade = invalid[0]
        raise RelevanceGradeError(f"relevance grade {grade} for {query_result['query_id']}/{doc_id} is outside [0, {max_grade}]")

    k = int(config["evaluation"]["k"])
    metrics_requested = list(config["evaluation"]["metrics"])
    retrieved = query_result.get("results", [])[:k]
    relevant_total = sum(1 for grade in judgments.values() if grade > 0)
    retrieved_relevant = sum(1 for item in retrieved if judgments.get(item["doc_id"], 0) > 0)

    metrics: dict[str, Optional[float]] = {}
    if "precision_at_k" in metrics_requested:
        metrics["precision_at_k"] = round(retrieved_relevant / k, 12) if k else 0.0
    if "recall_at_k" in metrics_requested:
        metrics["recall_at_k"] = round(retrieved_relevant / relevant_total, 12) if relevant_total else 0.0
    if "mrr_at_k" in metrics_requested:
        reciprocal_rank = 0.0
        for rank, item in enumerate(retrieved, start=1):
            if judgments.get(item["doc_id"], 0) > 0:
                reciprocal_rank = 1.0 / rank
                break
        metrics["mrr_at_k"] = round(reciprocal_rank, 12)
    if "ndcg_at_k" in metrics_requested:
        actual_grades = [judgments.get(item["doc_id"], 0) for item in retrieved]
        ideal_grades = sorted(judgments.values(), reverse=True)[:k]
        dcg = _dcg(actual_grades)
        idcg = _dcg(ideal_grades)
        metrics["ndcg_at_k"] = round(dcg / idcg, 12) if idcg > 0.0 else 0.0

    judged_retrieved = [item for item in retrieved if item["doc_id"] in judgments]
    if "grade_mae" in metrics_requested:
        if judged_retrieved:
            metrics["grade_mae"] = round(
                sum(abs(int(item["predicted_grade"]) - judgments[item["doc_id"]]) for item in judged_retrieved)
                / len(judged_retrieved),
                12,
            )
        else:
            metrics["grade_mae"] = None
    if "grade_accuracy" in metrics_requested:
        if judged_retrieved:
            metrics["grade_accuracy"] = round(
                sum(1 for item in judged_retrieved if int(item["predicted_grade"]) == judgments[item["doc_id"]])
                / len(judged_retrieved),
                12,
            )
        else:
            metrics["grade_accuracy"] = None

    return {
        "query_id": query_result["query_id"],
        "status": "ok",
        "code": None,
        "k": k,
        "qrels_count": len(judgments),
        "relevant_qrels_count": relevant_total,
        "retrieved_count": len(retrieved),
        "retrieved_relevant_count": retrieved_relevant,
        "metrics": metrics,
        "graded_predictions": [
            {
                "doc_id": item["doc_id"],
                "predicted_grade": int(item["predicted_grade"]),
                "qrel_grade": judgments.get(item["doc_id"]),
            }
            for item in retrieved
        ],
    }


def aggregate_metrics(per_query: Iterable[dict], requested: Iterable[str]) -> dict:
    values: dict[str, list[float]] = defaultdict(list)
    query_count = 0
    for item in per_query:
        if item.get("status") != "ok":
            continue
        query_count += 1
        for key, value in item.get("metrics", {}).items():
            if value is not None:
                values[key].append(float(value))
    aggregate = {
        key: (round(sum(values[key]) / len(values[key]), 12) if values[key] else None)
        for key in sorted(requested)
    }
    return {
        "query_count": query_count,
        "metrics": aggregate,
        "metric_denominators": {key: len(values[key]) for key in sorted(requested)},
    }

