"""Evidence-constrained relation candidate generation and classification."""

from __future__ import annotations

from typing import Any

from .errors import ModelShapeError
from .loader import (
    EvidenceTable,
    FixedPointConfig,
    ModelPackage,
    Sentence,
    WorkbenchConfig,
)
from .util import (
    format_fixed,
    permille,
    quantize_probability,
    quantize_units,
    sigmoid_from_units,
)


def _pair_key(subject: dict[str, Any], object_: dict[str, Any]) -> str:
    return (
        f"{subject['start']}:{subject['end']}->{object_['start']}:{object_['end']}"
    )


def _trigger_for_relation(
    sentence: Sentence,
    relation_type: str,
    subject: dict[str, Any],
    object_: dict[str, Any],
    triggers: dict[str, tuple[str, ...]],
) -> tuple[str, int | None]:
    trigger_set = set(triggers.get(relation_type, ()))
    excluded = set(range(subject["start"], subject["end"] + 1)).union(
        range(object_["start"], object_["end"] + 1)
    )
    for index in range(len(sentence.words)):
        if index in excluded:
            continue
        if sentence.words[index] in trigger_set:
            return sentence.words[index], index
    return "<NONE>", None


def relation_feature_keys(
    sentence: Sentence,
    subject: dict[str, Any],
    object_: dict[str, Any],
    relation_type: str,
    trigger: str,
    between_start: int,
    between_end: int,
) -> tuple[str, ...]:
    keys: list[str] = [
        f"candidate:{relation_type}",
        f"subject_type:{subject['type']}",
        f"object_type:{object_['type']}",
        f"trigger:{trigger}",
    ]
    for word in sentence.words[subject["start"] : subject["end"] + 1]:
        keys.append(f"subject_word:{word}")
    for word in sentence.words[object_["start"] : object_["end"] + 1]:
        keys.append(f"object_word:{word}")
    if between_start <= between_end:
        keys.append(
            "between_seq:" + "|".join(sentence.words[between_start : between_end + 1])
        )
        for word in sentence.words[between_start : between_end + 1]:
            keys.append(f"between_word:{word}")
    else:
        keys.append("between_seq:<EMPTY>")
    return tuple(dict.fromkeys(keys))


def _score_relation_type(
    model: ModelPackage,
    feature_keys: tuple[str, ...],
    relation_type: str,
    fixed: FixedPointConfig,
) -> tuple[int, list[dict[str, Any]]]:
    bias = model.relation_bias[relation_type]
    weights = model.relation_weights[relation_type]
    score = bias
    terms: list[dict[str, Any]] = []
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


def classify_relations(
    sentence: Sentence,
    entity_result: dict[str, Any],
    evidence: EvidenceTable,
    model: ModelPackage,
    config: WorkbenchConfig,
    fixed: FixedPointConfig,
    *,
    feature_hash_bucket_injection: bool = False,
) -> dict[str, Any]:
    if feature_hash_bucket_injection:
        raise ModelShapeError(
            "injected feature hash bucket is outside the model bucket range"
        )
    recognized: dict[tuple[int, int], dict[str, Any]] = {}
    for entity in entity_result["entities"]:
        key = (entity["start"], entity["end"])
        current = recognized.get(key)
        if current is None or (-entity["confidence_units"], entity["type"]) < (
            -current["confidence_units"],
            current["type"],
        ):
            recognized[key] = entity

    output: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    evidence_pairs = evidence.candidates_by_sentence.get(sentence.sentence_id, ())
    for pair_index, candidate in enumerate(evidence_pairs):
        subject = recognized.get(candidate.subject_span)
        object_ = recognized.get(candidate.object_span)
        pair_key = (
            f"{candidate.subject_span[0]}:{candidate.subject_span[1]}->"
            f"{candidate.object_span[0]}:{candidate.object_span[1]}"
        )
        if subject is None or object_ is None:
            skipped.append(
                {
                    "pair_key": pair_key,
                    "subject_span": list(candidate.subject_span),
                    "object_span": list(candidate.object_span),
                    "reason": "entity_not_recognized_or_below_threshold",
                }
            )
            continue
        if candidate.subject_span == candidate.object_span:
            skipped.append(
                {
                    "pair_key": pair_key,
                    "subject_span": list(candidate.subject_span),
                    "object_span": list(candidate.object_span),
                    "reason": "self_relation_not_supported",
                }
            )
            continue

        between_start = subject["end"] + 1
        between_end = object_["start"] - 1
        type_scores: list[dict[str, Any]] = []
        for relation_type in sorted(candidate.relations):
            trigger, trigger_index = _trigger_for_relation(
                sentence,
                relation_type,
                subject,
                object_,
                config.trigger_words,
            )
            features = relation_feature_keys(
                sentence,
                subject,
                object_,
                relation_type,
                trigger,
                between_start,
                between_end,
            )
            score_units, terms = _score_relation_type(
                model, features, relation_type, fixed
            )
            confidence_raw = sigmoid_from_units(score_units, fixed.decimals)
            confidence_units = quantize_probability(
                confidence_raw, fixed.decimals, fixed.quantization_decimals
            )
            type_scores.append(
                {
                    "type": relation_type,
                    "score_units": score_units,
                    "score": format_fixed(score_units, fixed.decimals),
                    "confidence_units": confidence_units,
                    "confidence": format_fixed(confidence_units, fixed.decimals),
                    "confidence_permille": permille(confidence_units, fixed.decimals),
                    "trigger": trigger,
                    "trigger_index": trigger_index,
                    "features": terms,
                }
            )
        if not type_scores:
            skipped.append(
                {
                    "pair_key": pair_key,
                    "subject_span": list(candidate.subject_span),
                    "object_span": list(candidate.object_span),
                    "reason": "no_allowed_relation",
                }
            )
            continue

        selected = min(
            type_scores, key=lambda item: (-item["score_units"], item["type"])
        )
        allowed_types = {item["type"] for item in type_scores}
        suppressed_set = set(evidence.exclusive_relations.get(selected["type"], ()))
        suppressed = sorted(allowed_types.intersection(suppressed_set))
        entity_units = min(
            subject["confidence_units"], object_["confidence_units"]
        )
        output.append(
            {
                "pair_index": pair_index,
                "pair_key": pair_key,
                "subject": {
                    "start": subject["start"],
                    "end": subject["end"],
                    "type": subject["type"],
                    "confidence_units": subject["confidence_units"],
                    "confidence": subject["confidence"],
                },
                "object": {
                    "start": object_["start"],
                    "end": object_["end"],
                    "type": object_["type"],
                    "confidence_units": object_["confidence_units"],
                    "confidence": object_["confidence"],
                },
                "entity_score_units": entity_units,
                "entity_score": format_fixed(entity_units, fixed.decimals),
                "type": selected["type"],
                "score_units": selected["score_units"],
                "score": selected["score"],
                "confidence_units": selected["confidence_units"],
                "confidence": selected["confidence"],
                "confidence_permille": selected["confidence_permille"],
                "trigger": selected["trigger"],
                "trigger_index": selected["trigger_index"],
                "type_scores": type_scores,
                "suppressed": suppressed,
            }
        )

    output.sort(key=lambda item: (item["pair_key"], item["type"]))
    skipped.sort(key=lambda item: (item["pair_key"], item["reason"]))
    return {
        "sentence_id": sentence.sentence_id,
        "precision": {
            "decimals": fixed.decimals,
            "quantization_decimals": fixed.quantization_decimals,
            "scale": fixed.scale,
        },
        "relations": output,
        "skipped_pairs": skipped,
    }
