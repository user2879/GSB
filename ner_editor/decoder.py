"""Integer log-domain BIESO Viterbi decoding with explicit tie provenance."""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from .constants import E_EMISSION_INVALID, FIXED_ALGORITHM, LABELS
from .models import Failure, InteractionConfig, ModelParams
from .stages import StageFailure
from .util import digest, quantize_decimal


def _quantize_matrix(
    model: ModelParams,
    config: InteractionConfig,
    *,
    fixed_decimals: int,
    level: int,
    sample_id: str,
) -> tuple[list[list[int]], list[int]]:
    transition: list[list[int]] = []
    for row_index, row in enumerate(model.transition_weights):
        quantized_row: list[int] = []
        for column_index, value in enumerate(row):
            if not isinstance(value, Decimal):
                value = Decimal(str(value))
            if not value.is_finite():
                raise StageFailure(
                    Failure(
                        code=E_EMISSION_INVALID,
                        stage="decode",
                        message="non-finite transition parameter reached decoding",
                        sample_id=sample_id,
                        details={
                            "parameter": f"transition_weights[{row_index}][{column_index}]",
                            "value": str(value),
                        },
                    )
                )
            units, _ = quantize_decimal(value, fixed_decimals, level)
            quantized_row.append(units)
        transition.append(quantized_row)
    start: list[int] = []
    for index, value in enumerate(model.start_weights):
        if not isinstance(value, Decimal):
            value = Decimal(str(value))
        if not value.is_finite():
            raise StageFailure(
                Failure(
                    code=E_EMISSION_INVALID,
                    stage="decode",
                    message="non-finite start parameter reached decoding",
                    sample_id=sample_id,
                    details={"parameter": f"start_weights[{index}]", "value": str(value)},
                )
            )
        units, _ = quantize_decimal(value, fixed_decimals, level)
        start.append(units)
    return transition, start


def transition_parameter_fingerprint(
    model: ModelParams,
    *,
    fixed_decimals: int,
    level: int,
    position_count: int | None = None,
) -> str:
    transitions_used = position_count is None or position_count >= 2
    return digest(
        {
            "schema": "transition-params-v1",
            "labels": list(LABELS),
            "transition_weights": (
                model.transition_weights if transitions_used else "unused-for-single-position"
            ),
            "start_weights": model.start_weights,
            "fixed_decimals": fixed_decimals,
            "quantization_level": level,
            "algorithm": FIXED_ALGORITHM,
        }
    )


def decode_key(
    emission_output: str,
    model: ModelParams,
    *,
    fixed_decimals: int,
    level: int,
    position_count: int | None = None,
) -> str:
    return digest(
        {
            "stage": "decode",
            "upstream": emission_output,
            "parameters": transition_parameter_fingerprint(
                model,
                fixed_decimals=fixed_decimals,
                level=level,
                position_count=position_count,
            ),
            "fixed_decimals": fixed_decimals,
            "quantization_level": level,
        }
    )


def decode_emissions(
    emission_payload: dict[str, Any],
    model: ModelParams,
    config: InteractionConfig,
    *,
    fixed_decimals: int | None = None,
    quantization_level: int | None = None,
) -> dict[str, Any]:
    sample_id = str(emission_payload["sample_id"])
    decimals = config.fixed_decimals if fixed_decimals is None else fixed_decimals
    level = config.quantization_level if quantization_level is None else quantization_level
    transition, start = _quantize_matrix(
        model,
        config,
        fixed_decimals=decimals,
        level=level,
        sample_id=sample_id,
    )
    score_rows = emission_payload["scores"]
    position_count = len(score_rows)
    tie_records: list[dict[str, Any]] = []
    trace: list[dict[str, Any]] = []
    previous_scores: list[int] = []

    for position in range(position_count):
        emissions = [int(value) for value in score_rows[position]]
        current_scores: list[int] = []
        candidate_details: list[dict[str, Any]] = []
        for to_code, to_label in enumerate(LABELS):
            if position == 0:
                score = emissions[to_code] + start[to_code]
                predecessors: list[int] = []
                chosen: int | None = None
            else:
                candidate_pairs = [
                    (previous_scores[from_code] + transition[from_code][to_code], from_code)
                    for from_code in range(len(LABELS))
                ]
                best_score = max(pair[0] for pair in candidate_pairs)
                predecessors = [
                    from_code for score, from_code in candidate_pairs if score == best_score
                ]
                chosen = predecessors[0]
                score = emissions[to_code] + best_score
                if len(predecessors) > 1:
                    tie_records.append(
                        {
                            "stage": "transition",
                            "position": position,
                            "to_label": to_label,
                            "to_code": to_code,
                            "tied_from_labels": [LABELS[code] for code in predecessors],
                            "tied_from_codes": list(predecessors),
                            "chosen_from_label": LABELS[chosen],
                            "chosen_from_code": chosen,
                        }
                    )
            current_scores.append(score)
            candidate_details.append(
                {
                    "label": to_label,
                    "code": to_code,
                    "score": score,
                    "chosen_predecessor_label": (
                        LABELS[chosen] if chosen is not None else None
                    ),
                    "chosen_predecessor_code": chosen,
                    "tied_predecessor_labels": (
                        [LABELS[code] for code in predecessors] if predecessors else []
                    ),
                }
            )
        trace.append(
            {
                "position": position,
                "candidates": candidate_details,
                "best_label": LABELS[
                    min(
                        range(len(LABELS)),
                        key=lambda code: (-current_scores[code], code),
                    )
                ],
            }
        )
        previous_scores = current_scores

    final_scores = previous_scores
    best_final_score = max(final_scores)
    final_ties = [
        code for code, score in enumerate(final_scores) if score == best_final_score
    ]
    chosen_final = final_ties[0]
    if len(final_ties) > 1:
        tie_records.append(
            {
                "stage": "final",
                "position": position_count - 1,
                "tied_labels": [LABELS[code] for code in final_ties],
                "tied_codes": final_ties,
                "chosen_label": LABELS[chosen_final],
                "chosen_code": chosen_final,
            }
        )

    path_codes = [chosen_final]
    for position in range(position_count - 1, 0, -1):
        candidates = trace[position]["candidates"]
        predecessor = candidates[path_codes[-1]]["chosen_predecessor_code"]
        assert predecessor is not None
        path_codes.append(int(predecessor))
    path_codes.reverse()
    path_labels = [LABELS[code] for code in path_codes]

    # Per-position candidate margins are useful to the merge boundary check.
    margins: list[dict[str, Any]] = []
    for position, item in enumerate(trace):
        candidate_scores = sorted(
            (candidate["score"] for candidate in item["candidates"]),
            reverse=True,
        )
        top = candidate_scores[0]
        second = candidate_scores[1] if len(candidate_scores) > 1 else top
        margins.append(
            {
                "position": position,
                "top_score": top,
                "runner_up_score": second,
                "margin": top - second,
            }
        )

    return {
        "sample_id": sample_id,
        "chars": list(emission_payload["chars"]),
        "labels": list(LABELS),
        "label_codes": {label: index for index, label in enumerate(LABELS)},
        "emissions": [list(row) for row in score_rows],
        "trace": trace,
        "path_labels": path_labels,
        "path_codes": path_codes,
        "path_total": best_final_score,
        "tie_records": tie_records,
        "position_margins": margins,
        "fixed_decimals": decimals,
        "scale": 10**decimals,
        "quantization_level": level,
        "algorithm": "integer-log-viterbi-v1",
    }


def decode_output_fingerprint(payload: dict[str, Any]) -> str:
    return digest(
        {
            "sample_id": payload["sample_id"],
            "chars": payload["chars"],
            "labels": payload["labels"],
            "emissions": payload["emissions"],
            "trace": payload["trace"],
            "path_labels": payload["path_labels"],
            "path_total": payload["path_total"],
            "tie_records": payload["tie_records"],
            "position_margins": payload["position_margins"],
            "fixed_decimals": payload["fixed_decimals"],
            "quantization_level": payload["quantization_level"],
            "algorithm": payload["algorithm"],
        }
    )
