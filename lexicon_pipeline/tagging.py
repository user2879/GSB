"""Fixed-point integer Viterbi POS tagging."""

from __future__ import annotations

import math
from typing import Any

from .fixedpoint import quantize_int
from .hashing import fingerprint


ALGORITHM_VERSION = "fixed-point-viterbi-v1"


def normalize_transition_weights(weights_document: dict[str, Any], index: dict[str, Any]) -> dict[str, Any]:
    pos_to_code = index["pos_to_code"]
    raw_table = weights_document.get("transition_weights", {})
    transitions: dict[tuple[int, int], float] = {}
    named: dict[str, float] = {}

    for source, row in raw_table.items():
        if not isinstance(row, dict):
            continue
        for target, value in row.items():
            if source in pos_to_code and target in pos_to_code:
                key = (pos_to_code[source], pos_to_code[target])
                transitions[key] = float(value)
                named[f"{source}>{target}"] = float(value)
        if isinstance(row, (int, float)) and source in pos_to_code:
            # Also accept flat {"NN>VV": 1.2} records.
            pass

    for key, value in raw_table.items():
        if ">" in key and isinstance(value, (int, float)):
            source, target = key.split(">", 1)
            if source in pos_to_code and target in pos_to_code:
                transitions[(pos_to_code[source], pos_to_code[target])] = float(value)
                named[key] = float(value)

    default = float(weights_document.get("default_transition_weight", 0.0))
    fp = fingerprint({"named": dict(sorted(named.items())), "default": default}, 20)
    return {"by_code": transitions, "named": named, "default": default, "fingerprint": fp}


def _candidate_tags(fragment: dict[str, Any], index: dict[str, Any]) -> list[str]:
    tags: set[str] = set()
    for entry_id in fragment.get("candidate_entry_ids", []):
        entry = index["by_id"].get(entry_id)
        if entry and entry.get("enabled", True):
            tags.update(entry.get("pos_tags") or [entry["pos"]])
    if not tags:
        declared = index["entries"] and None
        lowest_code_tag = min(index["code_to_pos"].items())[1]
        tags.add("NN" if "NN" in index["pos_to_code"] else lowest_code_tag)
    return sorted(tags, key=lambda name: index["pos_to_code"][name])


def _emission_raw(fragment: dict[str, Any], tag: str, index: dict[str, Any]) -> float:
    candidates = [index["by_id"][entry_id] for entry_id in fragment.get("candidate_entry_ids", []) if entry_id in index["by_id"]]
    candidates = [entry for entry in candidates if entry.get("enabled", True)]
    if not candidates:
        return 0.0
    candidates.sort(key=lambda entry: (-int(entry["frequency"]), entry["id"]))
    for entry in candidates:
        weights = entry.get("tag_weights") or {}
        if tag in weights:
            return float(weights[tag])
    return 0.0


def _transition_raw(prev_code: int, code: int, transitions: dict[str, Any], index: dict[str, Any]) -> float:
    return transitions["by_code"].get((prev_code, code), transitions["default"])


def _softmax_confidence(scores: list[int], scale: int, temperature: float) -> float:
    if not scores:
        return 0.0
    values = [score / scale / temperature for score in scores]
    maximum = max(values)
    exps = [math.exp(value - maximum) for value in values]
    total = sum(exps)
    return exps[0] / total if total else 0.0


def tag_document(
    segmentation: dict[str, Any],
    index: dict[str, Any],
    transitions: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    fp_config = config["fixed_point"]
    bits = int(config["effective_fraction_bits"])
    scale = int(config["fixed_point_scale"])
    high_bits = int(config["high_precision_bits"])
    high_scale = int(config["high_precision_scale"])
    mode = fp_config.get("rounding_mode", "ROUND_HALF_UP")
    threshold = float(fp_config.get("boundary_threshold", 0.0))
    threshold_int = quantize_int(threshold, scale, mode)
    temperature = float(fp_config.get("score_temperature", 1.0))

    fragments = sorted(segmentation["fragments"], key=lambda item: (item["start"], item["fragment_seq"]))
    dp_steps: list[dict[int, int]] = []
    backpointers: list[dict[int, int | None]] = []
    fragment_outputs: list[dict[str, Any]] = []

    for step, fragment in enumerate(fragments):
        tag_names = _candidate_tags(fragment, index)
        tag_codes = [index["pos_to_code"][name] for name in tag_names]
        emission_raw = {code: _emission_raw(fragment, index["code_to_pos"][code], index) for code in tag_codes}
        emission_int = {code: quantize_int(emission_raw[code], scale, mode) for code in tag_codes}
        emission_high = {code: quantize_int(emission_raw[code], high_scale, mode) for code in tag_codes}

        current_dp: dict[int, int] = {}
        current_bp: dict[int, int | None] = {}
        weight_refs: list[str] = []

        if step == 0:
            for code in sorted(tag_codes):
                current_dp[code] = emission_int[code]
                current_bp[code] = None
        else:
            previous_dp = dp_steps[-1]
            previous_codes = sorted(previous_dp)
            for code in sorted(tag_codes):
                best_score: int | None = None
                best_prev: int | None = None
                for prev_code in previous_codes:
                    weight_name = f"{index['code_to_pos'][prev_code]}>{index['code_to_pos'][code]}"
                    if weight_name not in weight_refs:
                        weight_refs.append(weight_name)
                    trans_raw = _transition_raw(prev_code, code, transitions, index)
                    trans_int = quantize_int(trans_raw, scale, mode)
                    candidate_score = previous_dp[prev_code] + trans_int
                    if best_score is None or candidate_score > best_score:
                        best_score = candidate_score
                        best_prev = prev_code
                    # Equal scores keep the first predecessor because codes are
                    # visited in ascending order; this is the stable tie-break.
                assert best_score is not None
                current_dp[code] = best_score + emission_int[code]
                current_bp[code] = best_prev

        dp_steps.append(current_dp)
        backpointers.append(current_bp)

        candidate_scores = []
        ordered_codes = sorted(current_dp)
        score_values = [current_dp[code] for code in ordered_codes]
        confidence_by_code: dict[int, float] = {}
        if score_values:
            raw_values = [value / scale / temperature for value in score_values]
            maximum = max(raw_values)
            exps = [math.exp(value - maximum) for value in raw_values]
            denominator = sum(exps)
            confidence_by_code = {code: exps[i] / denominator for i, code in enumerate(ordered_codes)}

        for code in ordered_codes:
            raw = emission_raw[code]
            candidate_scores.append(
                {
                    "tag": index["code_to_pos"][code],
                    "code": code,
                    "score_int": current_dp[code],
                    "score_float": current_dp[code] / scale,
                    "emission_int": emission_int[code],
                    "emission_float": emission_int[code] / scale,
                    "emission_raw": raw,
                    "rounding_residual": raw - emission_int[code] / scale,
                    "confidence": confidence_by_code.get(code, 0.0),
                    "predecessor_code": current_bp.get(code),
                }
            )

        fragment_outputs.append(
            {
                "fragment_id": fragment["fragment_id"],
                "fragment_seq": fragment["fragment_seq"],
                "start": fragment["start"],
                "end": fragment["end"],
                "surface": fragment["surface"],
                "tag_codes": sorted(tag_codes),
                "tag_names": tag_names,
                "weight_refs": sorted(weight_refs),
                "candidate_scores": candidate_scores,
                "dp_by_code": dict(sorted(current_dp.items())),
                "backpointers": {str(key): value for key, value in current_bp.items()},
                "emission_high": emission_high,
            }
        )

    selected_codes: list[int] = []
    path_total_int = 0
    if fragment_outputs:
        final_dp = dp_steps[-1]
        final_code = min(sorted(final_dp), key=lambda code: (-final_dp[code], code, fragments[-1]["start"]))
        path_total_int = final_dp[final_code]
        cursor: int | None = final_code
        selected_codes = [0] * len(fragments)
        for step in range(len(fragments) - 1, -1, -1):
            assert cursor is not None
            selected_codes[step] = cursor
            cursor = backpointers[step][cursor]

    selected_tags = [index["code_to_pos"][code] for code in selected_codes]

    # High-width reference for the exact selected path.  Operations follow the
    # same start/code order as the production integer recurrence.
    boundary_fragments: list[dict[str, Any]] = []
    selected_prefix_current = 0
    selected_prefix_high = 0
    for i, output in enumerate(fragment_outputs):
        code = selected_codes[i]
        emission_current = next(item["emission_int"] for item in output["candidate_scores"] if item["code"] == code)
        emission_high = output["emission_high"][code]
        if i == 0:
            selected_prefix_current = emission_current
            selected_prefix_high = emission_high
        else:
            prev_code = selected_codes[i - 1]
            trans_raw = _transition_raw(prev_code, code, transitions, index)
            selected_prefix_current += quantize_int(trans_raw, scale, mode)
            selected_prefix_current += emission_current
            selected_prefix_high += quantize_int(trans_raw, high_scale, mode)
            selected_prefix_high += emission_high

        top_scores = sorted(output["dp_by_code"].values(), reverse=True)
        margin_int = top_scores[0] - top_scores[1] if len(top_scores) > 1 else None
        delta_high_int = selected_prefix_high - 2 * selected_prefix_current
        delta_raw = delta_high_int / high_scale
        high_raw = selected_prefix_high / high_scale
        reasons: list[str] = []
        if selected_prefix_current == threshold_int:
            reasons.append("threshold_equal")
        if abs(high_raw - threshold) <= 0.5 / scale:
            reasons.append("threshold_half_step")
        if margin_int == 0:
            reasons.append("candidate_tie")
        if reasons:
            boundary_fragments.append(
                {
                    "doc_id": segmentation["doc_id"],
                    "doc_seq": segmentation["doc_seq"],
                    "fragment_id": output["fragment_id"],
                    "fragment_seq": output["fragment_seq"],
                    "start": output["start"],
                    "end": output["end"],
                    "selected_tag": index["code_to_pos"][code],
                    "selected_code": code,
                    "reasons": reasons,
                    "prefix_total_int": selected_prefix_current,
                    "threshold_int": threshold_int,
                    "threshold_raw": threshold,
                    "current_scale": scale,
                    "current_fraction_bits": bits,
                    "high_precision_bits": high_bits,
                    "high_precision_scale": high_scale,
                    "high_reference_total_int": selected_prefix_high,
                    "delta_high_int": delta_high_int,
                    "delta_raw": delta_raw,
                    "candidate_margin_int": margin_int,
                }
            )

    path_total_high = 0
    if selected_codes:
        for i, code in enumerate(selected_codes):
            tag = index["code_to_pos"][code]
            path_total_high += quantize_int(_emission_raw(fragments[i], tag, index), high_scale, mode)
            if i > 0:
                prev_code = selected_codes[i - 1]
                path_total_high += quantize_int(_transition_raw(prev_code, code, transitions, index), high_scale, mode)

    path_boundary: dict[str, Any] | None = None
    if selected_codes and fragment_outputs:
        path_delta_high_int = path_total_high - 2 * path_total_int
        path_high_raw = path_total_high / high_scale
        path_reasons: list[str] = []
        if path_total_int == threshold_int:
            path_reasons.append("path_total_threshold_equal")
        if abs(path_high_raw - threshold) <= 0.5 / scale:
            path_reasons.append("path_total_threshold_half_step")
        if path_reasons:
            last_fragment = fragment_outputs[-1]
            path_boundary = {
                "doc_id": segmentation["doc_id"],
                "doc_seq": segmentation["doc_seq"],
                "fragment_id": last_fragment["fragment_id"],
                "fragment_seq": last_fragment["fragment_seq"],
                "start": last_fragment["start"],
                "end": last_fragment["end"],
                "selected_tag": selected_tags[-1],
                "selected_code": selected_codes[-1],
                "boundary_scope": "full_path_total",
                "reasons": path_reasons,
                "prefix_total_int": path_total_int,
                "path_total_int": path_total_int,
                "threshold_int": threshold_int,
                "threshold_raw": threshold,
                "current_scale": scale,
                "current_fraction_bits": bits,
                "high_precision_bits": high_bits,
                "high_precision_scale": high_scale,
                "high_reference_total_int": path_total_high,
                "delta_high_int": path_delta_high_int,
                "delta_raw": path_delta_high_int / high_scale,
            }
            boundary_fragments.append(path_boundary)

    receipts: list[dict[str, Any]] = []
    tagged_fragments: list[dict[str, Any]] = []
    for i, output in enumerate(fragment_outputs):
        code = selected_codes[i]
        selected_candidate = next(item for item in output["candidate_scores"] if item["code"] == code)
        confidence = selected_candidate["confidence"]
        receipts.append(
            {
                "doc_id": segmentation["doc_id"],
                "doc_seq": segmentation["doc_seq"],
                "fragment_seq": output["fragment_seq"],
                "score": confidence,
                "pos": index["code_to_pos"][code],
                "rounding_mode": mode,
                "precision_label": f"fixed-point-Q{bits}:softmax-confidence",
                "precision_labels": [f"Q{bits}", config["fixed_point"]["quantization_tier"], "softmax-confidence"],
            }
        )
        tagged_fragments.append(
            {
                **{key: value for key, value in output.items() if key not in {"dp_by_code", "backpointers", "emission_high"}},
                "selected_tag": index["code_to_pos"][code],
                "selected_code": code,
                "selected_confidence": confidence,
                "selected_prefix_total_int": dp_steps[i][code],
            }
        )

    annotation_key = fingerprint(
        {
            "algorithm": ALGORITHM_VERSION,
            "segmentation_fingerprint": segmentation["segmentation_fingerprint"],
            "selected_codes": selected_codes,
            "path_total_int": path_total_int,
            "bits": bits,
            "tier": config["fixed_point"]["quantization_tier"],
            "transition_fingerprint": transitions["fingerprint"],
        },
        20,
    )
    return {
        "doc_id": segmentation["doc_id"],
        "doc_seq": segmentation["doc_seq"],
        "algorithm_version": ALGORITHM_VERSION,
        "fraction_bits": bits,
        "scale": scale,
        "high_precision_bits": high_bits,
        "high_precision_scale": high_scale,
        "tags": selected_tags,
        "selected_codes": selected_codes,
        "path_total_int": path_total_int,
        "path_total_float": path_total_int / scale,
        "high_reference_path_total_int": path_total_high,
        "path_delta_high_int": path_total_high - 2 * path_total_int,
        "path_delta_raw": (path_total_high - 2 * path_total_int) / high_scale if selected_codes else 0.0,
        "path_boundary": path_boundary,
        "annotation_key": annotation_key,
        "boundary_fragments": boundary_fragments,
        "receipts": receipts,
        "fragments": tagged_fragments,
        "_dp_steps": dp_steps,
        "_backpointers": backpointers,
    }
