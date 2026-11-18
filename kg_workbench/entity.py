"""Quantized BIESO entity recognition and entity type classification."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from .errors import ConfigError, ModelShapeError
from .loader import (
    ENTITY_LABEL_PRIORITY,
    FixedPointConfig,
    ModelPackage,
    Sentence,
)
from .util import (
    format_fixed,
    permille,
    quantize_probability,
    quantize_units,
    round_div_half_up,
    sigmoid_from_units,
    units_decimal,
)


VALID_TRANSITIONS: dict[str, tuple[str, ...]] = {
    "B": ("I", "E"),
    "I": ("I", "E"),
    "E": ("B", "S", "O"),
    "S": ("B", "S", "O"),
    "O": ("B", "S", "O"),
}
VALID_STARTS = ("B", "S", "O")
VALID_ENDS = ("E", "S", "O")
NEG_INF = -(10**30)


@dataclass(frozen=True)
class CharacterSlot:
    char: str
    word_index: int
    char_index: int
    pos: str
    offset_in_word: int
    word_length: int


def character_slots(sentence: Sentence) -> tuple[CharacterSlot, ...]:
    slots: list[CharacterSlot] = []
    for word_index, word in enumerate(sentence.words):
        chars = tuple(word)
        for char_index, char in enumerate(chars):
            slots.append(
                CharacterSlot(
                    char=char,
                    word_index=word_index,
                    char_index=char_index,
                    pos=sentence.pos[word_index],
                    offset_in_word=char_index,
                    word_length=len(chars),
                )
            )
    return tuple(slots)


def _lexicon_hit_length(chars: str, start: int, lexicon: tuple[str, ...]) -> int:
    best = 0
    for term in lexicon:
        if len(term) > best and chars.startswith(term, start):
            best = len(term)
    return best


def character_feature_keys(
    slots: tuple[CharacterSlot, ...], index: int, lexicon: tuple[str, ...]
) -> tuple[str, ...]:
    slot = slots[index]
    current_string = "".join(item.char for item in slots)
    previous = slots[index - 1] if index > 0 else None
    following = slots[index + 1] if index + 1 < len(slots) else None
    prev_char = previous.char if previous is not None else "<BOS>"
    next_char = following.char if following is not None else "<EOS>"
    prev_pos = previous.pos if previous is not None else "<BOS>"
    next_pos = following.pos if following is not None else "<EOS>"
    hit_length = _lexicon_hit_length(current_string, index, lexicon)
    return (
        f"char:{slot.char}",
        f"char_prev:{prev_char}",
        f"char_next:{next_char}",
        f"pos:{slot.pos}",
        f"pos_prev:{prev_pos}",
        f"pos_next:{next_pos}",
        f"lexlen:{hit_length}",
    )


def entity_span_feature_keys(
    sentence: Sentence, start: int, end: int, lexicon: tuple[str, ...]
) -> tuple[str, ...]:
    words = sentence.words[start : end + 1]
    pos = sentence.pos[start : end + 1]
    joined = "".join(words)
    keys: list[str] = [
        f"span_len:{len(words)}",
        f"span_first:{joined[0]}",
        f"span_last:{joined[-1]}",
    ]
    for word in words:
        keys.append(f"span_word:{word}")
    for tag in pos:
        keys.append(f"span_pos:{tag}")
    hit = _lexicon_hit_length(joined, 0, lexicon)
    keys.append(f"span_lexlen:{hit}")
    return tuple(dict.fromkeys(keys))


def _linear_score(
    model: ModelPackage,
    bias: Decimal,
    weights: tuple[Decimal, ...],
    feature_keys: tuple[str, ...],
    fixed: FixedPointConfig,
) -> tuple[int, list[dict[str, Any]]]:
    terms: list[dict[str, Any]] = []
    score = bias
    for key in sorted(feature_keys):
        bucket = model.bucket_for(key)
        weight = weights[bucket]
        score += weight
        terms.append(
            {
                "feature": key,
                "bucket": bucket,
                "weight_units": quantize_units(
                    weight, fixed.decimals, fixed.quantization_decimals
                ),
            }
        )
    return quantize_units(score, fixed.decimals, fixed.quantization_decimals), terms


def emission_scores(
    sentence: Sentence,
    model: ModelPackage,
    fixed: FixedPointConfig,
) -> tuple[tuple[CharacterSlot, ...], dict[str, tuple[int, ...]]]:
    slots = character_slots(sentence)
    buckets_per_position: list[dict[str, int]] = []
    for index in range(len(slots)):
        keys = character_feature_keys(slots, index, model.lexicon)
        buckets_per_position.append({key: model.bucket_for(key) for key in keys})

    result: dict[str, list[int]] = {label: [] for label in ENTITY_LABEL_PRIORITY}
    for label in ENTITY_LABEL_PRIORITY:
        weights = model.entity_weights[label]
        bias = model.entity_bias[label]
        for bucket_map in buckets_per_position:
            raw = bias
            for key in sorted(bucket_map):
                raw += weights[bucket_map[key]]
            result[label].append(
                quantize_units(raw, fixed.decimals, fixed.quantization_decimals)
            )
    return slots, {label: tuple(values) for label, values in result.items()}


def _transition_units(model: ModelPackage, fixed: FixedPointConfig) -> dict[str, dict[str, int]]:
    return {
        source: {
            target: quantize_units(
                model.transitions[source][target],
                fixed.decimals,
                fixed.quantization_decimals,
            )
            for target in ENTITY_LABEL_PRIORITY
        }
        for source in ENTITY_LABEL_PRIORITY
    }


def viterbi_decode(
    emissions: dict[str, tuple[int, ...]], model: ModelPackage, fixed: FixedPointConfig
) -> tuple[tuple[str, ...], int, str]:
    length = len(next(iter(emissions.values())))
    if length == 0:
        raise ConfigError("cannot decode an empty sentence")
    transitions = _transition_units(model, fixed)
    priority = {label: index for index, label in enumerate(ENTITY_LABEL_PRIORITY)}

    previous_scores: dict[str, int | None] = {label: None for label in ENTITY_LABEL_PRIORITY}
    for label in VALID_STARTS:
        previous_scores[label] = emissions[label][0]
    backpointers: list[dict[str, str | None]] = []

    for position in range(1, length):
        current_scores: dict[str, int | None] = {
            label: None for label in ENTITY_LABEL_PRIORITY
        }
        current_back: dict[str, str | None] = {
            label: None for label in ENTITY_LABEL_PRIORITY
        }
        for target in ENTITY_LABEL_PRIORITY:
            best_score: int | None = None
            best_source: str | None = None
            for source in ENTITY_LABEL_PRIORITY:
                if target not in VALID_TRANSITIONS[source]:
                    continue
                source_score = previous_scores[source]
                if source_score is None:
                    continue
                candidate = source_score + transitions[source][target]
                if (
                    best_score is None
                    or candidate > best_score
                    or (candidate == best_score and priority[source] < priority[best_source])
                ):
                    best_score = candidate
                    best_source = source
            if best_score is not None:
                current_scores[target] = best_score + emissions[target][position]
                current_back[target] = best_source
        previous_scores = current_scores
        backpointers.append(current_back)

    final_score: int | None = None
    final_label: str | None = None
    for label in ENTITY_LABEL_PRIORITY:
        if label not in VALID_ENDS:
            continue
        score = previous_scores[label]
        if score is None:
            continue
        if (
            final_score is None
            or score > final_score
            or (score == final_score and priority[label] < priority[final_label])
        ):
            final_score = score
            final_label = label
    if final_score is None or final_label is None:
        raise ModelShapeError("no structurally valid BIESO path exists for the sentence")

    labels: list[str] = [final_label]
    current = final_label
    for position in range(length - 1, 0, -1):
        source = backpointers[position - 1][current]
        if source is None:
            raise ModelShapeError("incomplete Viterbi backpointer")
        labels.append(source)
        current = source
    labels.reverse()
    return tuple(labels), final_score, format_fixed(final_score, fixed.decimals)


def _segments_from_labels(
    labels: tuple[str, ...], slots: tuple[CharacterSlot, ...]
) -> list[tuple[int, int, int, int]]:
    segments: list[tuple[int, int, int, int]] = []
    start: int | None = None
    for index, label in enumerate(labels):
        if label == "B":
            start = index
        elif label == "S":
            segments.append((index, index, slots[index].word_index, slots[index].word_index))
            start = None
        elif label == "E":
            if start is None:
                raise ModelShapeError("BIESO path has E without B/I")
            segments.append(
                (start, index, slots[start].word_index, slots[index].word_index)
            )
            start = None
    if start is not None:
        raise ModelShapeError("BIESO path ends with an open entity")
    return segments


def _entity_confidence(
    labels: tuple[str, ...],
    start: int,
    end: int,
    emissions: dict[str, tuple[int, ...]],
    fixed: FixedPointConfig,
) -> tuple[int, Decimal, int]:
    margins: list[int] = []
    for position in range(start, end + 1):
        chosen_label = labels[position]
        chosen = emissions[chosen_label][position]
        alternatives = [
            emissions[label][position]
            for label in ENTITY_LABEL_PRIORITY
            if label != chosen_label
        ]
        margins.append(chosen - max(alternatives))
    average_margin = round_div_half_up(sum(margins), len(margins))
    raw_probability = sigmoid_from_units(average_margin, fixed.decimals)
    units = quantize_probability(
        raw_probability, fixed.decimals, fixed.quantization_decimals
    )
    return units, raw_probability, average_margin


def recognize_entities(
    sentence: Sentence,
    model: ModelPackage,
    config: object,
    fixed: FixedPointConfig,
    *,
    missing_pos_injection: bool = False,
) -> dict[str, Any]:
    """Run the entity stage and return a JSON-safe, deterministic result record."""

    if missing_pos_injection:
        if not sentence.pos:
            raise ConfigError("injected missing POS tag leaves an empty POS stream")
        effective = Sentence(
            sentence_id=sentence.sentence_id,
            words=sentence.words,
            pos=sentence.pos[:-1],
            entities=sentence.entities,
            relations=sentence.relations,
            source_line=sentence.source_line,
        )
        if len(effective.words) != len(effective.pos):
            raise ConfigError(
                f"injected missing POS tag: word count ({len(effective.words)}) "
                f"does not match POS count ({len(effective.pos)})"
            )
    else:
        effective = sentence

    slots, emissions = emission_scores(effective, model, fixed)
    labels, path_score, path_score_text = viterbi_decode(emissions, model, fixed)
    segments = _segments_from_labels(labels, slots)
    threshold_units = quantize_units(
        config.entity_threshold, fixed.decimals, fixed.quantization_decimals
    )

    candidates: list[dict[str, Any]] = []
    for char_start, char_end, word_start, word_end in segments:
        type_scores: dict[str, int] = {}
        feature_terms: dict[str, list[dict[str, Any]]] = {}
        for entity_type in sorted(model.entity_types):
            score, terms = _linear_score(
                model,
                model.entity_type_bias[entity_type],
                model.entity_type_weights[entity_type],
                entity_span_feature_keys(effective, word_start, word_end, model.lexicon),
                fixed,
            )
            type_scores[entity_type] = score
            feature_terms[entity_type] = terms
        best_type = min(
            type_scores,
            key=lambda value: (-type_scores[value], value),
        )
        confidence_units, confidence, margin = _entity_confidence(
            labels, char_start, char_end, emissions, fixed
        )
        accepted = confidence_units >= threshold_units
        candidates.append(
            {
                "start": word_start,
                "end": word_end,
                "start_char": char_start,
                "end_char": char_end,
                "type": best_type,
                "type_scores": [
                    {
                        "type": entity_type,
                        "score_units": type_scores[entity_type],
                        "score": format_fixed(type_scores[entity_type], fixed.decimals),
                        "features": feature_terms[entity_type],
                    }
                    for entity_type in sorted(type_scores)
                ],
                "confidence_units": confidence_units,
                "confidence": format_fixed(confidence_units, fixed.decimals),
                "confidence_permille": permille(confidence_units, fixed.decimals),
                "raw_confidence": format_fixed(
                    quantize_units(confidence, fixed.decimals, fixed.quantization_decimals),
                    fixed.decimals,
                ),
                "margin_units": margin,
                "accepted": accepted,
            }
        )
    candidates.sort(key=lambda item: (item["start"], item["end"], item["type"]))
    accepted_entities = [item for item in candidates if item["accepted"]]

    position_scores = []
    for position in range(len(slots)):
        slot = slots[position]
        position_scores.append(
            {
                "char_index": position,
                "word_index": slot.word_index,
                "char": slot.char,
                "pos": slot.pos,
                "scores": [
                    {
                        "label": label,
                        "score_units": emissions[label][position],
                        "score": format_fixed(emissions[label][position], fixed.decimals),
                    }
                    for label in ENTITY_LABEL_PRIORITY
                ],
            }
        )

    return {
        "sentence_id": sentence.sentence_id,
        "precision": {
            "decimals": fixed.decimals,
            "quantization_decimals": fixed.quantization_decimals,
            "scale": fixed.scale,
        },
        "labels": list(labels),
        "path_score_units": path_score,
        "path_score": path_score_text,
        "position_scores": position_scores,
        "candidate_entities": candidates,
        "entities": accepted_entities,
        "entity_threshold_units": threshold_units,
        "entity_threshold": format_fixed(threshold_units, fixed.decimals),
    }
