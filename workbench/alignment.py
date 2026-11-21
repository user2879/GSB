"""Cross-modal stable alignment scoring and deterministic retrieval."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .config import AlignmentConfig


def _vectors(artifact: dict[str, Any], key: str = "quantized", scale_key: str = "fixed_scale") -> tuple[np.ndarray, np.ndarray]:
    entries = (
        artifact.get("blocks")
        or artifact.get("entity_groups")
        or artifact.get("entries")
        or []
    )
    vectors = np.asarray([entry[key] for entry in entries], dtype=np.float64)
    scales = np.asarray([entry[scale_key] for entry in entries], dtype=np.float64)
    if vectors.size == 0:
        return np.zeros((0, 0), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    vectors = vectors / scales[:, None]
    return vectors, scales


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    safe = np.where(norms > 0.0, norms, 1.0)
    return values / safe


def _common_vectors(visual: np.ndarray, text: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    common = min(visual.shape[1], text.shape[1])
    return visual[:, :common], text[:, :common]


def _fixed(value: float, scale: int) -> int:
    if not math.isfinite(value):
        raise ValueError("non-finite score")
    return int(np.rint(value * scale))


def align_score(
    visual_artifact: dict[str, Any],
    text_artifact: dict[str, Any],
    config: AlignmentConfig,
) -> dict[str, Any]:
    visual, _ = _vectors(visual_artifact)
    text, _ = _vectors(text_artifact)
    if visual.size == 0 or text.size == 0:
        raise ValueError("alignment requires at least one visual block and one text assertion")
    visual, text = _common_vectors(visual, text)
    visual_norm = _normalize_rows(visual)
    text_norm = _normalize_rows(text)
    match_entries = text_artifact.get("entity_groups") or text_artifact["entries"]
    weights = np.asarray(
        [float(entry.get("weight", 1.0)) for entry in match_entries],
        dtype=np.float64,
    )
    cosine_matrix = text_norm @ visual_norm.T if text_norm.size and visual_norm.size else np.zeros((0, 0))
    assignment: dict[int, int] = {}
    candidate_pairs = [
        (float(cosine_matrix[group_index, block_index]), group_index, block_index)
        for group_index in range(cosine_matrix.shape[0])
        for block_index in range(cosine_matrix.shape[1])
    ]
    candidate_pairs.sort(key=lambda item: (-item[0], item[1], item[2]))
    used_groups: set[int] = set()
    used_blocks: set[int] = set()
    for similarity, group_index, block_index in candidate_pairs:
        if group_index in used_groups or block_index in used_blocks:
            continue
        assignment[group_index] = block_index
        used_groups.add(group_index)
        used_blocks.add(block_index)
        if len(used_groups) == cosine_matrix.shape[0]:
            break
    visual_blocks = visual_artifact.get("blocks", [])
    assertion_matches: list[dict[str, Any]] = []
    match_values: list[float] = []
    for index in range(text_norm.shape[0]):
        raw = cosine_matrix[index]
        if raw.size == 0:
            continue
        row_max = float(np.max(raw))
        shifted = raw - row_max
        attention = np.exp(shifted / config.temperature)
        attention_sum = float(np.sum(attention))
        if not math.isfinite(attention_sum) or attention_sum <= 0.0:
            attention = np.full_like(raw, 1.0 / raw.size)
        else:
            attention = attention / attention_sum
        fused_visual = np.sum(attention[:, None] * visual_norm, axis=0)
        fused_norm = float(np.linalg.norm(fused_visual))
        if fused_norm > 0.0:
            fused_visual = fused_visual / fused_norm
        fused_cosine = float(np.dot(fused_visual, text_norm[index]))
        fused_cosine = max(-1.0, min(1.0, fused_cosine))
        quality = (fused_cosine + 1.0) / 2.0
        assigned_block = assignment.get(index)
        assigned_similarity = (
            float(cosine_matrix[index, assigned_block])
            if assigned_block is not None
            else -1.0
        )
        assigned_quality = (assigned_similarity + 1.0) / 2.0
        match_values.append(assigned_quality)
        assertion_matches.append(
            {
                "match_id": match_entries[index].get("entity_id")
                or match_entries[index].get("assertion_id"),
                "entity_id": match_entries[index].get("entity_id"),
                "assertion_ids": match_entries[index].get("assertion_ids")
                or [match_entries[index].get("assertion_id")],
                "padded": bool(match_entries[index].get("padded", False)),
                "weight": float(weights[index]),
                "max_cosine": row_max,
                "stable_shift": shifted.tolist(),
                "attention": attention.tolist(),
                "fused_cosine": fused_cosine,
                "fused_quality": quality,
                "assigned_block_index": assigned_block,
                "assigned_block_id": (
                    visual_blocks[assigned_block].get("block_id")
                    if assigned_block is not None and assigned_block < len(visual_blocks)
                    else None
                ),
                "assigned_similarity": assigned_similarity,
                "quality": assigned_quality,
            }
        )
    match_values_array = np.asarray(match_values, dtype=np.float64)
    weighted = float(np.average(match_values_array, weights=weights)) if weights.size else 0.0

    pooled_visual = np.mean(visual, axis=0) if visual.size else np.zeros(0, dtype=np.float64)
    pooled_text = np.average(text, axis=0, weights=weights) if weights.size else np.zeros(0, dtype=np.float64)
    pv_norm = float(np.linalg.norm(pooled_visual))
    pt_norm = float(np.linalg.norm(pooled_text))
    if pv_norm > 0.0 and pt_norm > 0.0:
        pooled_cosine = float(np.dot(pooled_visual, pooled_text) / (pv_norm * pt_norm))
        pooled_cosine = max(-1.0, min(1.0, pooled_cosine))
        pooled_quality = (pooled_cosine + 1.0) / 2.0
    else:
        pooled_cosine = -1.0
        pooled_quality = 0.0

    scale = config.fixed_scale
    visual_component_fixed = _fixed(
        config.visual_weight_normalized * pooled_quality,
        scale,
    )
    text_component_fixed = _fixed(
        config.text_weight_normalized * weighted,
        scale,
    )
    score_fixed = visual_component_fixed + text_component_fixed
    if score_fixed > scale:
        text_component_fixed -= score_fixed - scale
        score_fixed = scale
    if score_fixed < 0:
        visual_component_fixed -= score_fixed
        score_fixed = 0
    score = score_fixed / scale
    return {
        "stage": "alignment",
        "visual_sample": visual_artifact.get("sample"),
        "text_sample": text_artifact.get("sample"),
        "score_fixed": score_fixed,
        "score": score,
        "fixed_scale": scale,
        "fixed_decimal_bits": config.effective_decimal_bits,
        "quantization_level": config.quantization_level,
        "contributions": {
            "visual_component_fixed": visual_component_fixed,
            "text_component_fixed": text_component_fixed,
            "visual_contribution": visual_component_fixed / scale,
            "text_contribution": text_component_fixed / scale,
            "visual_weight": config.visual_weight_normalized,
            "text_weight": config.text_weight_normalized,
        },
        "pooled_cosine": pooled_cosine,
        "pooled_quality": pooled_quality,
        "stable_assertion_quality": weighted,
        "stable_entity_quality": weighted,
        "unique_assignment_count": len(assignment),
        "assertion_matches": assertion_matches,
    }


def query_against_visual(
    query_text: dict[str, Any],
    candidate_visual: dict[str, Any],
    config: AlignmentConfig,
) -> dict[str, Any]:
    return align_score(candidate_visual, query_text, config)


def retrieve(
    query_texts: dict[str, dict[str, Any]],
    visual_artifacts: dict[str, dict[str, Any]],
    config: AlignmentConfig,
) -> dict[str, Any]:
    scale = config.fixed_scale
    threshold_fixed = _fixed(config.similarity_threshold, scale)
    all_candidates = sorted(visual_artifacts)
    queries: list[dict[str, Any]] = []
    for query_name in sorted(query_texts):
        text_artifact = query_texts[query_name]
        scored: list[dict[str, Any]] = []
        for candidate_name in all_candidates:
            result = query_against_visual(text_artifact, visual_artifacts[candidate_name], config)
            boundary = result["score_fixed"] == threshold_fixed
            scored.append(
                {
                    "sample": candidate_name,
                    "score_fixed": result["score_fixed"],
                    "score": result["score"],
                    "contributions": result["contributions"],
                    "pooled_cosine": result["pooled_cosine"],
                    "stable_assertion_quality": result["stable_assertion_quality"],
                    "passes_threshold": result["score_fixed"] >= threshold_fixed,
                    "boundary": boundary,
                    "tie_rule": "score_fixed_desc_then_sample_asc",
                }
            )
        scored.sort(key=lambda item: (-item["score_fixed"], item["sample"]))
        for rank, item in enumerate(scored, start=1):
            item["rank_all"] = rank
        returned = [item for item in scored if item["passes_threshold"]][: config.top_k]
        for rank, item in enumerate(returned, start=1):
            item["rank"] = rank
        queries.append(
            {
                "query": query_name,
                "threshold_fixed": threshold_fixed,
                "threshold": threshold_fixed / scale,
                "top_k": config.top_k,
                "candidate_count": len(scored),
                "returned_count": len(returned),
                "results": returned,
                "all_candidates": scored,
            }
        )
    return {
        "stage": "retrieval_index",
        "fixed_scale": scale,
        "fixed_decimal_bits": config.effective_decimal_bits,
        "quantization_level": config.quantization_level,
        "threshold": threshold_fixed / scale,
        "threshold_fixed": threshold_fixed,
        "top_k": config.top_k,
        "tie_breaker": "score_fixed_desc_then_sample_asc",
        "boundary_rule": "score_fixed >= threshold_fixed; equality is a boundary sample",
        "queries": queries,
    }


def retrieval_quality(retrieval: dict[str, Any]) -> dict[str, Any]:
    query_count = len(retrieval["queries"])
    self_hits = 0
    hit_at_k = 0
    reciprocal_rank_sum = 0.0
    details: list[dict[str, Any]] = []
    for query in retrieval["queries"]:
        query_name = query["query"]
        found_rank: int | None = None
        for result in query["results"]:
            if result["sample"] == query_name:
                found_rank = result["rank"]
                break
        if found_rank == 1:
            self_hits += 1
        if found_rank is not None:
            hit_at_k += 1
            reciprocal_rank_sum += 1.0 / found_rank
        details.append(
            {
                "query": query_name,
                "returned_count": query["returned_count"],
                "self_rank": found_rank,
                "self_hit": found_rank is not None,
                "top1_self_hit": found_rank == 1,
            }
        )
    return {
        "queries": query_count,
        "self_hit_at_k_count": hit_at_k,
        "self_hit_at_k_rate": hit_at_k / query_count if query_count else 0.0,
        "top1_self_hit_count": self_hits,
        "top1_self_hit_rate": self_hits / query_count if query_count else 0.0,
        "mrr": reciprocal_rank_sum / query_count if query_count else 0.0,
        "per_query": details,
    }
