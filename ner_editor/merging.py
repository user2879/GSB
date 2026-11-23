"""BIESO repair, entity type assignment and boundary diagnostics."""

from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import Any, Callable

from .constants import (
    MERGE_ALGORITHM,
    TYPE_CODES,
)
from .models import CorpusSample, Failure, InteractionConfig, ModelParams, VocabEntry
from .stages import StageFailure
from .util import (
    decimal_from_units,
    digest,
    fraction_json,
    quantize_decimal,
    residual_direction,
    score_json,
)

ProbeRunner = Callable[[int, int], dict[str, Any]]


def _span_record(start: int, end: int, kind: str = "entity") -> dict[str, Any]:
    return {"start": start, "end": end, "kind": kind}


def repair_sequence(labels: list[str] | tuple[str, ...]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Repair an arbitrary label sequence using a fixed, documented policy."""

    spans: list[dict[str, Any]] = []
    repairs: list[dict[str, Any]] = []
    current_start: int | None = None
    index = 0
    length = len(labels)

    def close_open(next_index: int, next_label: str) -> None:
        nonlocal current_start
        if current_start is None:
            return
        spans.append(_span_record(current_start, next_index - 1))
        repairs.append(
            {
                "reason_code": "ENTITY_NOT_CLOSED_BY_E",
                "basis": "open B/I span encountered O, S, B, or end-of-sample",
                "position": next_index - 1,
                "detected_position": next_index,
                "affected_positions": list(range(current_start, next_index)),
                "observed_label": next_label,
                "repair": "close the open span immediately before the current position",
            }
        )
        current_start = None

    while index < length:
        label = labels[index]
        if current_start is not None:
            if label == "I":
                index += 1
                continue
            if label == "E":
                spans.append(_span_record(current_start, index))
                current_start = None
                index += 1
                continue
            close_open(index, label)

        if label == "B":
            next_label = labels[index + 1] if index + 1 < length else None
            if next_label in {"I", "E"}:
                current_start = index
            else:
                spans.append(_span_record(index, index))
                repairs.append(
                    {
                        "reason_code": "B_WITHOUT_CONTINUATION",
                        "basis": "B must be followed by I or E",
                        "position": index,
                        "detected_position": index,
                        "affected_positions": [index],
                        "observed_label": label,
                        "repair": "treat the B as a singleton S span",
                    }
                )
            index += 1
            continue
        if label == "S":
            spans.append(_span_record(index, index))
            index += 1
            continue
        if label == "I":
            next_label = labels[index + 1] if index + 1 < length else None
            repairs.append(
                {
                    "reason_code": "I_WITHOUT_B",
                    "basis": "I appeared without an open B/I span",
                    "position": index,
                    "detected_position": index,
                    "affected_positions": [index],
                    "observed_label": label,
                    "repair": "rewrite I as B",
                }
            )
            if next_label in {"I", "E"}:
                current_start = index
            else:
                spans.append(_span_record(index, index))
            index += 1
            continue
        if label == "E":
            repairs.append(
                {
                    "reason_code": "E_WITHOUT_B",
                    "basis": "E appeared without an open B/I span",
                    "position": index,
                    "detected_position": index,
                    "affected_positions": [index],
                    "observed_label": label,
                    "repair": "treat E as a singleton S span",
                }
            )
            spans.append(_span_record(index, index))
            index += 1
            continue
        # The decoder normally only returns BIESO, but keeping an explicit
        # unknown-label repair makes merge total even for future label sets.
        repairs.append(
            {
                "reason_code": "UNKNOWN_LABEL_AS_O",
                "basis": "label is outside BIESO",
                "position": index,
                "detected_position": index,
                "affected_positions": [index],
                "observed_label": label,
                "repair": "treat as O",
            }
        )
        index += 1

    if current_start is not None:
        close_open(length, "<EOS>")
    spans.sort(key=lambda span: (span["start"], span["end"]))
    repairs.sort(key=lambda item: (item["position"], item["reason_code"]))
    return spans, repairs


def _gold_spans(sample: CorpusSample) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    index = 0
    while index < len(sample.labels):
        if sample.labels[index] in {"B", "S"}:
            start = index
            entity_type = sample.types[index]
            if sample.labels[index] == "S":
                end = index
                index += 1
            else:
                index += 1
                while index < len(sample.labels) and sample.labels[index] == "I":
                    index += 1
                if index < len(sample.labels) and sample.labels[index] == "E":
                    end = index
                    index += 1
                else:
                    end = max(start, index - 1)
            spans.append(
                {
                    "start": start,
                    "end": end,
                    "type": entity_type,
                    "text": "".join(sample.chars[start : end + 1]),
                }
            )
        else:
            index += 1
    return spans


def _type_assignment(
    sample: CorpusSample,
    vocabulary: dict[str, VocabEntry],
    start: int,
    end: int,
    default_type: str,
) -> dict[str, Any]:
    text = "".join(sample.chars[start : end + 1])
    scores: dict[str, int] = {}
    sources: list[dict[str, Any]] = []

    for span in _gold_spans(sample):
        if span["start"] == start and span["end"] == end:
            scores[span["type"]] = scores.get(span["type"], 0) + 3
            sources.append(
                {
                    "kind": "gold_exact_span",
                    "type": span["type"],
                    "weight": 3,
                }
            )
    for index in range(start, min(end + 1, len(sample.types))):
        type_code = sample.types[index]
        if type_code != "_":
            scores[type_code] = scores.get(type_code, 0) + 1
            sources.append(
                {
                    "kind": "gold_char_evidence",
                    "position": index,
                    "type": type_code,
                    "weight": 1,
                }
            )
    for entry in sorted(vocabulary.values(), key=lambda item: (item.text, item.entry_id)):
        if entry.text == text:
            scores[entry.type_code] = scores.get(entry.type_code, 0) + 2
            sources.append(
                {
                    "kind": "vocabulary_exact_match",
                    "entry_id": entry.entry_id,
                    "type": entry.type_code,
                    "weight": 2,
                }
            )

    if not scores:
        scores[default_type] = 1
        sources.append({"kind": "default_type", "type": default_type, "weight": 1})

    ranked = sorted(
        scores.items(),
        key=lambda item: (-item[1], TYPE_CODES.index(item[0]) if item[0] in TYPE_CODES else 99),
    )
    top_score = ranked[0][1]
    top_types = [type_code for type_code, score in ranked if score == top_score]
    chosen_type = sorted(
        top_types,
        key=lambda type_code: TYPE_CODES.index(type_code) if type_code in TYPE_CODES else 99,
    )[0]
    tie = (
        {
            "tied_types": top_types,
            "chosen_type": chosen_type,
            "rule": "type_code_ascending",
        }
        if len(top_types) > 1
        else None
    )
    return {
        "type": chosen_type,
        "type_candidates": [
            {"type": type_code, "score": score} for type_code, score in ranked
        ],
        "type_tie": tie,
        "type_decision_basis": sources,
    }


def _fraction_decimal(value: Fraction) -> str:
    scale = Decimal(10) ** 12
    rounded = Decimal(value.numerator) / Decimal(value.denominator)
    return format(rounded.quantize(Decimal(1) / scale), "f")


def _decision_for_entity(
    sample: CorpusSample,
    vocabulary: dict[str, VocabEntry],
    labels: list[str],
    start: int,
    end: int,
    default_type: str,
) -> dict[str, Any]:
    spans, _ = repair_sequence(labels)
    matched = None
    for span in spans:
        if span["start"] == start and span["end"] == end:
            matched = span
            break
    if matched is None and start < len(labels):
        for span in spans:
            if span["start"] <= start <= span["end"]:
                matched = span
                break
    if matched is None:
        return {
            "start": start,
            "end": end,
            "labels": labels[start : end + 1],
            "type": None,
            "exists": False,
        }
    assignment = _type_assignment(
        sample,
        vocabulary,
        matched["start"],
        matched["end"],
        default_type,
    )
    return {
        "start": matched["start"],
        "end": matched["end"],
        "labels": labels[matched["start"] : matched["end"] + 1],
        "text": "".join(sample.chars[matched["start"] : matched["end"] + 1]),
        "type": assignment["type"],
        "exists": True,
    }


def _boundary_probe(
    *,
    sample: CorpusSample,
    vocabulary: dict[str, VocabEntry],
    active_decoder: dict[str, Any],
    entity: dict[str, Any],
    config: InteractionConfig,
    default_type: str,
    probe_runner: ProbeRunner | None,
) -> dict[str, Any] | None:
    decimals = int(active_decoder["fixed_decimals"])
    level = int(active_decoder["quantization_level"])
    tolerance_units, tolerance_residual = quantize_decimal(
        config.boundary_score_tolerance,
        decimals,
        level,
    )
    confidence_units = int(entity["confidence"]["units"])
    if confidence_units != tolerance_units:
        return None

    start = int(entity["start"])
    end = int(entity["end"])
    active_decision = _decision_for_entity(
        sample,
        vocabulary,
        active_decoder["path_labels"],
        start,
        end,
        default_type,
    )
    high_decimals = min(decimals + 1, 12)
    if high_decimals == decimals:
        high_decimals = decimals
    high_decoder: dict[str, Any] | None = None
    if probe_runner is not None and high_decimals != decimals:
        high_decoder = probe_runner(high_decimals, level)

    reference_score = None
    difference = None
    high_decision = None
    flip_stage = None
    if high_decoder is not None:
        high_margins = high_decoder.get("position_margins", [])
        high_units_values = [
            int(high_margins[position]["margin"])
            for position in range(start, min(end + 1, len(high_margins)))
        ]
        reference_units = min(high_units_values) if high_units_values else 0
        high_scale = int(high_decoder["scale"])
        reference_score = score_json(reference_units, high_scale, high_decimals)
        active_value = Fraction(confidence_units, int(active_decoder["scale"]))
        high_value = Fraction(reference_units, high_scale)
        delta = high_value - active_value
        difference = {
            **fraction_json(delta),
            "decimal": _fraction_decimal(delta),
        }
        high_decision = _decision_for_entity(
            sample,
            vocabulary,
            high_decoder["path_labels"],
            start,
            end,
            default_type,
        )
        if high_decoder["path_labels"][start : end + 1] != active_decoder["path_labels"][start : end + 1]:
            flip_stage = "decode"
        elif (
            high_decision.get("type") is not None
            and high_decision.get("type") != active_decision.get("type")
        ):
            flip_stage = "merge"

    alternate = None
    levels = sorted(set(config.quantization_levels) | {level})
    alternate_levels = [candidate for candidate in levels if candidate != level]
    if alternate_levels and probe_runner is not None:
        alternate_level = min(
            alternate_levels,
            key=lambda candidate: (abs(candidate - level), candidate),
        )
        alternate_decoder = probe_runner(decimals, alternate_level)
        alternate_decision = _decision_for_entity(
            sample,
            vocabulary,
            alternate_decoder["path_labels"],
            start,
            end,
            default_type,
        )
        alternate_flip = None
        if (
            alternate_decoder["path_labels"][start : end + 1]
            != active_decoder["path_labels"][start : end + 1]
        ):
            alternate_flip = "decode"
        elif (
            alternate_decision.get("type") is not None
            and alternate_decision.get("type") != active_decision.get("type")
        ):
            alternate_flip = "merge"
        alternate = {
            "quantization_level": alternate_level,
            "decision": alternate_decision,
            "decision_equal": alternate_decision == active_decision,
            "flip_stage": alternate_flip,
        }
        if flip_stage is None and alternate_flip is not None:
            flip_stage = alternate_flip

    return {
        "kind": "confidence_equals_boundary_tolerance",
        "active_score": score_json(confidence_units, int(active_decoder["scale"]), decimals),
        "tolerance": {
            **score_json(tolerance_units, int(active_decoder["scale"]), decimals),
            "residual": fraction_json(tolerance_residual),
            "rounding": residual_direction(tolerance_residual),
        },
        "reference_fixed_decimals": high_decimals,
        "reference_score": reference_score,
        "difference": difference,
        "active_decision": active_decision,
        "reference_decision": high_decision,
        "alternate_quantization": alternate,
        "flip_stage": flip_stage,
    }


def _relevant_vocabulary_fingerprint(
    sample: CorpusSample,
    vocabulary: dict[str, VocabEntry],
) -> str:
    words = sample.text
    relevant = []
    for entry in sorted(vocabulary.values(), key=lambda item: (item.text, item.entry_id)):
        if entry.text and entry.text in words:
            relevant.append(entry.as_dict())
    return digest(relevant)


def merge_key(
    decode_output: str,
    sample: CorpusSample,
    vocabulary: dict[str, VocabEntry],
    config: InteractionConfig,
    default_type: str,
) -> str:
    return digest(
        {
            "stage": "merge",
            "upstream": decode_output,
            "algorithm": MERGE_ALGORITHM,
            "sample_labels": list(sample.labels),
            "sample_types": list(sample.types),
            "vocabulary": _relevant_vocabulary_fingerprint(sample, vocabulary),
            "boundary_score_tolerance": config.boundary_score_tolerance,
            "fixed_decimals": config.fixed_decimals,
            "quantization_level": config.quantization_level,
            "quantization_levels": list(config.quantization_levels),
            "reference_fixed_decimals": min(config.fixed_decimals + 1, 12),
            "default_type": default_type,
        }
    )


def merge_entities(
    sample: CorpusSample,
    vocabulary: dict[str, VocabEntry],
    decoder_payload: dict[str, Any],
    config: InteractionConfig,
    model: ModelParams,
    *,
    probe_runner: ProbeRunner | None = None,
) -> dict[str, Any]:
    labels = list(decoder_payload["path_labels"])
    spans, repairs = repair_sequence(labels)
    margins = decoder_payload.get("position_margins", [])
    default_type = model.default_entity_type or config.default_entity_type
    entities: list[dict[str, Any]] = []
    boundary_records: list[dict[str, Any]] = []
    for span in spans:
        start = int(span["start"])
        end = int(span["end"])
        assignment = _type_assignment(sample, vocabulary, start, end, default_type)
        span_margins = [
            int(margins[position]["margin"])
            for position in range(start, min(end + 1, len(margins)))
        ]
        confidence_units = min(span_margins) if span_margins else 0
        entity = {
            "start": start,
            "end": end,
            "text": "".join(sample.chars[start : end + 1]),
            "type": assignment["type"],
            "type_candidates": assignment["type_candidates"],
            "type_tie": assignment["type_tie"],
            "type_decision_basis": assignment["type_decision_basis"],
            "labels": labels[start : end + 1],
            "confidence": {
                "kind": "minimum_position_candidate_margin",
                "units": confidence_units,
                "scale": int(decoder_payload["scale"]),
                "fixed_decimals": int(decoder_payload["fixed_decimals"]),
                "decimal": format(
                    decimal_from_units(confidence_units, int(decoder_payload["scale"])),
                    "f",
                ),
            },
            "boundary": None,
        }
        boundary = _boundary_probe(
            sample=sample,
            vocabulary=vocabulary,
            active_decoder=decoder_payload,
            entity=entity,
            config=config,
            default_type=default_type,
            probe_runner=probe_runner,
        )
        if boundary is not None:
            entity["boundary"] = boundary
            boundary_records.append(
                {
                    "sample_id": sample.sample_id,
                    "start": start,
                    "end": end,
                    **boundary,
                }
            )
        entities.append(entity)

    return {
        "sample_id": sample.sample_id,
        "chars": list(sample.chars),
        "gold_labels": list(sample.labels),
        "gold_types": list(sample.types),
        "path_labels": labels,
        "path_total": int(decoder_payload["path_total"]),
        "position_margins": margins,
        "entities": entities,
        "repairs": repairs,
        "boundary_records": boundary_records,
        "algorithm": MERGE_ALGORITHM,
    }


def merge_output_fingerprint(payload: dict[str, Any]) -> str:
    return digest(
        {
            "sample_id": payload["sample_id"],
            "chars": payload["chars"],
            "gold_labels": payload["gold_labels"],
            "gold_types": payload["gold_types"],
            "path_labels": payload["path_labels"],
            "path_total": payload["path_total"],
            "position_margins": payload["position_margins"],
            "entities": payload["entities"],
            "repairs": payload["repairs"],
            "boundary_records": payload["boundary_records"],
            "algorithm": payload["algorithm"],
        }
    )
