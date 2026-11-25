"""Deterministic feature hashing and fixed-point emission scoring."""

from __future__ import annotations

from typing import Any

from .config import QuantConfig
from .io_utils import stable_hash_text
from .labels import LABEL_CODES, LABELS


FEATURE_FAMILIES = ("current_char", "prev_char", "next_char", "position_band", "lexicon_length")


def _position_band(index: int, length: int) -> int:
    if length <= 1 or index == 0:
        return 0
    if index == length - 1:
        return 2
    return 1


def sample_lexicon_refs(chars: list[str], entries: list[dict[str, str]]) -> list[str]:
    text = "".join(chars)
    refs = {entry["id"] for entry in entries if entry.get("text") and entry["text"] in text}
    return sorted(refs)


def feature_coordinates(
    sample: dict[str, Any],
    entries: list[dict[str, str]],
    feature_dim: int,
    bucket_count: int,
) -> dict[str, Any]:
    chars = sample["chars"]
    length = len(chars)
    refs = sample_lexicon_refs(chars, entries)
    entries_by_id = {entry["id"]: entry for entry in entries}
    ref_entries = [entries_by_id[entry_id] for entry_id in refs if entry_id in entries_by_id]
    positions: list[dict[str, Any]] = []

    for index, char in enumerate(chars):
        lex_matches = []
        lex_length = 0
        for entry in ref_entries:
            word = entry["text"]
            word_length = len(word)
            if chars[index : index + word_length] == list(word):
                lex_matches.append(entry["id"])
                lex_length = max(lex_length, word_length)

        raw_features = (
            ("current_char", f"cur:{char}"),
            ("prev_char", f"prev:{chars[index - 1] if index > 0 else '^'}"),
            ("next_char", f"next:{chars[index + 1] if index + 1 < length else '$'}"),
            ("position_band", f"pos:{_position_band(index, length)}"),
            ("lexicon_length", f"lexlen:{lex_length}"),
        )
        coordinates = []
        for family_index, (family, feature_key) in enumerate(raw_features):
            feature_index = family_index % feature_dim
            bucket = stable_hash_text(feature_key) % bucket_count
            coordinates.append(
                {
                    "family": family,
                    "family_index": family_index,
                    "feature_index": feature_index,
                    "bucket": bucket,
                    "hash_key": feature_key,
                    "lexicon_length": lex_length if family == "lexicon_length" else None,
                    "lexicon_match_ids": sorted(lex_matches) if family == "lexicon_length" else [],
                }
            )
        coordinates.sort(key=lambda item: (item["feature_index"], item["bucket"], item["family_index"]))
        positions.append({"position": index, "char": char, "coordinates": coordinates})

    return {"sample_id": sample["id"], "lexicon_refs": refs, "positions": positions}


def quantize_value(value: float, config: QuantConfig) -> tuple[int, dict[str, Any]]:
    scale = config.scale
    scaled = float(value) * scale
    if scaled >= 0:
        quantized = int(scaled + 0.5)
    else:
        quantized = -int(-scaled + 0.5)
    pre_clamp = quantized
    clamped = max(config.clamp_min, min(config.clamp_max, quantized))
    restored = clamped / scale
    if restored > value:
        direction = "up"
    elif restored < value:
        direction = "down"
    else:
        direction = "exact"
    trace = {
        "raw": value,
        "scale": scale,
        "step": config.quantization_step,
        "effective_bits": config.effective_bits,
        "pre_clamp_integer": pre_clamp,
        "integer": clamped,
        "direction": direction,
        "clamped": clamped != pre_clamp,
        "rounding": "nearest_ties_away_from_zero",
    }
    return clamped, trace


def rounded_product_term(qh: int, qw: int, config: QuantConfig) -> tuple[int, str]:
    product = qh * qw
    scale = config.scale
    divisor = scale
    half = divisor // 2
    if product >= 0:
        term = (product + half) // divisor
    else:
        term = -((-product + half) // divisor)
    reconstructed = term * divisor
    if reconstructed > product:
        direction = "up"
    elif reconstructed < product:
        direction = "down"
    else:
        direction = "exact"
    term = max(config.clamp_min, min(config.clamp_max, term))
    return term, direction


def direction_code(direction: str) -> str:
    return {"up": "u", "down": "d", "exact": "e"}.get(direction, "?")


def emission_scores(
    features: dict[str, Any],
    package: dict[str, Any],
    config: QuantConfig,
) -> dict[str, Any]:
    feature_dim = package["feature_dim"]
    bucket_count = package["bucket_count"]
    hash_table = package["hash_table"]
    weights = package["emission_weights"]
    rows: list[list[float]] = []
    integer_rows: list[list[int]] = []
    trace_positions: list[dict[str, Any]] = []

    for position in features["positions"]:
        score_row: list[float] = []
        integer_row: list[int] = []
        label_traces: list[dict[str, Any]] = []
        coordinates = position["coordinates"]
        quantized_coordinates = []
        coordinate_traces = []
        for coordinate in coordinates:
            feature_index = coordinate["feature_index"]
            bucket = coordinate["bucket"]
            if not (0 <= feature_index < feature_dim and 0 <= bucket < bucket_count):
                raise IndexError(f"feature coordinate out of range: f={feature_index}, b={bucket}")
            h_raw = hash_table[bucket][feature_index]
            qh, h_trace = quantize_value(h_raw, config)
            quantized_coordinates.append((coordinate, qh))
            coordinate_traces.append(
                {
                    "feature_index": feature_index,
                    "bucket": bucket,
                    "family": coordinate["family"],
                    "h_integer": qh,
                    "h_direction": h_trace["direction"],
                }
            )
        for label in LABELS:
            label_index = LABEL_CODES[label]
            total = 0
            term_integers: list[int] = []
            weight_directions: list[str] = []
            shift_directions: list[str] = []
            for coordinate, qh in quantized_coordinates:
                feature_index = coordinate["feature_index"]
                bucket = coordinate["bucket"]
                w_raw = weights[label_index][feature_index][bucket]
                qw, w_trace = quantize_value(w_raw, config)
                term, shift_direction = rounded_product_term(qh, qw, config)
                total += term
                term_integers.append(term)
                weight_directions.append(direction_code(w_trace["direction"]))
                shift_directions.append(direction_code(shift_direction))
            total = max(config.clamp_min, min(config.clamp_max, total))
            integer_row.append(total)
            score_row.append(total / config.scale)
            label_traces.append(
                {
                    "label": label,
                    "integer_score": total,
                    "term_integers": term_integers,
                    "weight_directions": "".join(weight_directions),
                    "product_shift_directions": "".join(shift_directions),
                    "accumulation_order": "feature_index_asc,bucket_asc,family_index_asc",
                }
            )
        rows.append(score_row)
        integer_rows.append(integer_row)
        trace_positions.append(
            {
                "position": position["position"],
                "char": position["char"],
                "quantization_step": config.quantization_step,
                "effective_bits": config.effective_bits,
                "decimal_bits": config.decimal_bits,
                "quant_level": config.quant_level,
                "coordinates": coordinate_traces,
                "labels": label_traces,
            }
        )

    return {
        "sample_id": features["sample_id"],
        "labels": list(LABELS),
        "scores": rows,
        "integer_scores": integer_rows,
        "quantization_trace": trace_positions,
        "quant_fingerprint": config.fingerprint(),
    }
