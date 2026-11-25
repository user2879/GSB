"""Per-fragment nearest-representative recognition and rejection decisions."""

from __future__ import annotations

from typing import Any

from .fingerprint import sha256_json
from .similarity import fixed_similarity_permille


def quantization_half_step_permille(fractional_bits: int) -> int:
    scale = 1 << fractional_bits
    return max(1, 1000 // scale)


def recognition_cache_key(
    clip_integer_fingerprint: str,
    representatives: dict[str, dict[str, Any]],
    representative_templates: dict[str, dict[str, Any]],
    *,
    reject_threshold_permille: int,
    overlap_candidates: int,
    fractional_bits: int,
    label: dict[str, Any] | None = None,
) -> str:
    return "recognition:" + sha256_json(
        {
            "clip": clip_integer_fingerprint,
            "label": (
                {
                    "clip_id": label.get("clip_id"),
                    "path": label.get("path"),
                    "true_command": label.get("true_command"),
                    "confusion_with": label.get("confusion_with"),
                    "kind": label.get("kind"),
                }
                if label is not None
                else None
            ),
            "representatives": [
                {
                    "command_word": command_word,
                    "template_key": representatives[command_word]["template_key"],
                    "integer_fingerprint": representative_templates[command_word]["integer_fingerprint"],
                    "score": representatives[command_word]["composite_score_permille"],
                }
                for command_word in sorted(representatives)
            ],
            "reject_threshold_permille": reject_threshold_permille,
            "overlap_candidates": overlap_candidates,
            "fractional_bits": fractional_bits,
        }
    )


def recognize_clip(
    clip: dict[str, Any],
    label: dict[str, Any],
    representatives: dict[str, dict[str, Any]],
    representative_templates: dict[str, dict[str, Any]],
    *,
    reject_threshold_permille: int,
    overlap_candidates: int,
    fractional_bits: int,
) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for command_word in sorted(representatives):
        representative = representatives[command_word]
        template = representative_templates[command_word]
        similarity = fixed_similarity_permille(clip["integers"], template["integers"])
        candidates.append(
            {
                "command_word": command_word,
                "template_key": representative["template_key"],
                "similarity_permille": similarity,
                "representative_composite_score_permille": representative["composite_score_permille"],
            }
        )
    candidates.sort(key=lambda item: (-item["similarity_permille"], item["command_word"]))
    for rank, candidate in enumerate(candidates, 1):
        candidate["rank"] = rank
    top = candidates[0] if candidates else None
    runner_up = candidates[1] if len(candidates) > 1 else None
    rejected = top is None or top["similarity_permille"] < reject_threshold_permille
    decision = "<REJECTED>" if rejected else top["command_word"]
    difference = (top["similarity_permille"] - runner_up["similarity_permille"]) if top and runner_up else None
    half_step = quantization_half_step_permille(fractional_bits)
    boundary = bool(top and runner_up and difference <= half_step)
    if difference is None:
        direction = "no_runner_up"
    elif difference > 0:
        direction = "top_ahead"
    else:
        direction = "tie"
    selected_candidates = candidates[: max(1, overlap_candidates)]
    references = [candidate["template_key"] for candidate in selected_candidates]
    result = {
        "clip_id": label["clip_id"],
        "path": label["path"],
        "content_sha256": clip["content_sha256"],
        "true_command": label["true_command"],
        "label_confusion_with": label["confusion_with"],
        "kind": label["kind"],
        "clip_integer_fingerprint": clip["integer_fingerprint"],
        "clip_effective_frames": clip.get("effective_frames", {}),
        "top": top,
        "runner_up": runner_up,
        "decision": decision,
        "rejected": rejected,
        "reject_reason": "top_similarity_below_reject_threshold" if rejected else None,
        "reject_threshold_permille": reject_threshold_permille,
        "candidates": selected_candidates,
        "overlap_candidate_count": len(selected_candidates),
        "boundary": boundary,
        "quantization_half_step_permille": half_step,
        "boundary_delta": {
            "top_minus_runner_up_permille": difference,
            "direction": direction,
            "within_half_step": boundary,
        }
        if runner_up
        else {
            "top_minus_runner_up_permille": None,
            "direction": direction,
            "within_half_step": False,
        },
        "comparison_references": [
            {
                "command_word": candidate["command_word"],
                "template_key": candidate["template_key"],
                "clip_integer_fingerprint": clip["integer_fingerprint"],
                "template_integer_fingerprint": representative_templates[candidate["command_word"]][
                    "integer_fingerprint"
                ],
                "similarity_permille": candidate["similarity_permille"],
            }
            for candidate in selected_candidates
        ],
        "references": references,
    }
    result["cache_key"] = recognition_cache_key(
        clip["integer_fingerprint"],
        representatives,
        representative_templates,
        reject_threshold_permille=reject_threshold_permille,
        overlap_candidates=overlap_candidates,
        fractional_bits=fractional_bits,
        label=label,
    )
    return result


def reference_comparison(
    low_result: dict[str, Any],
    high_result: dict[str, Any],
    *,
    low_profile: str,
    high_profile: str,
) -> dict[str, Any]:
    low_top = low_result.get("top")
    high_top = high_result.get("top")
    low_similarity = low_top["similarity_permille"] if low_top else None
    high_similarity = high_top["similarity_permille"] if high_top else None
    if low_similarity is None or high_similarity is None:
        direction = "unavailable"
    elif high_similarity > low_similarity:
        direction = "reference_similarity_up"
    elif high_similarity < low_similarity:
        direction = "reference_similarity_down"
    else:
        direction = "reference_similarity_equal"
    return {
        "reference_profile": high_profile,
        "source_profile": low_profile,
        "reference_conclusion": high_result["decision"],
        "source_conclusion": low_result["decision"],
        "conclusion_changed": high_result["decision"] != low_result["decision"],
        "reference_top_similarity_permille": high_similarity,
        "source_top_similarity_permille": low_similarity,
        "difference_direction": direction,
        "reference_runner_up": high_result.get("runner_up"),
        "references": high_result.get("references", []),
    }
