from __future__ import annotations

from fractions import Fraction
from typing import Any


def _permille_fraction(item: dict[str, Any]) -> Fraction:
    return Fraction(int(item["mean_permille_exact_num"]), int(item["mean_permille_exact_den"]))


def attribute_verdict_flip(
    item_a: dict[str, Any],
    item_b: dict[str, Any],
    scores_a: dict[tuple[str, int], dict[str, Any]] | None = None,
    scores_b: dict[tuple[str, int], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Classify why two quantization configurations produce different verdicts.

    The classifier follows the computation pipeline.  A changed per-record
    quantized value that changes the aggregate is a score-quantization flip.
    Unchanged record values but a changed weighted aggregate is accumulation;
    unchanged record and aggregate values with a changed code is thresholding.
    """
    scores_a = scores_a or {}
    scores_b = scores_b or {}
    session_id = item_a["session_id"]
    turns = sorted(set(item_a.get("turns", ())) | set(item_b.get("turns", ())))
    quant_changes: list[dict[str, Any]] = []
    quant_changed = False

    for turn in turns:
        row_a = scores_a.get((session_id, turn))
        row_b = scores_b.get((session_id, turn))
        if row_a is None or row_b is None:
            continue
        value_a = Fraction(int(row_a["quantized_scaled"]), int(row_a["scale"]))
        value_b = Fraction(int(row_b["quantized_scaled"]), int(row_b["scale"]))
        if value_a != value_b:
            quant_changed = True
            quant_changes.append({"turn": turn, "value_a": str(value_a), "value_b": str(value_b)})

    mean_a = _permille_fraction(item_a)
    mean_b = _permille_fraction(item_b)
    aggregate_changed = mean_a != mean_b

    if quant_changed and aggregate_changed:
        source = "score_quantization"
    elif not quant_changed and aggregate_changed:
        source = "weighted_accumulation"
    else:
        source = "threshold_comparison"

    return {
        "scope_id": item_a["scope_id"],
        "kind": item_a.get("kind"),
        "verdict_a": item_a["verdict_code"],
        "verdict_b": item_b["verdict_code"],
        "mean_permille_a": str(mean_a),
        "mean_permille_b": str(mean_b),
        "aggregate_changed": aggregate_changed,
        "quantized_record_changes": quant_changes,
        "source": source,
        "basis": {
            "score_quantization": "normalized per-record fixed-point values changed and moved the aggregate",
            "weighted_accumulation": "record values stayed equal but fixed-point span weighting/rounding changed the aggregate",
            "threshold_comparison": "record and aggregate values stayed equal; boundary half-step or threshold comparison changed",
        }[source],
    }


def compare_run_verdicts(
    result_a: dict[str, Any],
    result_b: dict[str, Any],
    score_report_a: list[dict[str, Any]] | None = None,
    score_report_b: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    report_a = result_a["report"] if "report" in result_a else result_a
    report_b = result_b["report"] if "report" in result_b else result_b
    score_report_a = score_report_a or report_a.get("record_scores", [])
    score_report_b = score_report_b or report_b.get("record_scores", [])
    scores_a = {(row["session_id"], row["turn"]): row for row in score_report_a}
    scores_b = {(row["session_id"], row["turn"]): row for row in score_report_b}
    flips: list[dict[str, Any]] = []
    for stage in ("windows", "partitions"):
        map_a = {item["scope_id"]: item for item in report_a.get(stage, [])}
        map_b = {item["scope_id"]: item for item in report_b.get(stage, [])}
        for scope_id in sorted(set(map_a) & set(map_b)):
            item_a = map_a[scope_id]
            item_b = map_b[scope_id]
            if item_a["verdict_code"] != item_b["verdict_code"]:
                flips.append(attribute_verdict_flip(item_a, item_b, scores_a, scores_b))
    return flips
