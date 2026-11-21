"""Deterministic score fusion, truncation and observability."""

from __future__ import annotations

import math
from typing import Any

from .config import FusionConfig
from .diagnostics import failure
from .hashing import fusion_fingerprint
from .io_utils import sha256_json


def build_contribution_index(normalized_state: dict[str, Any], config: FusionConfig) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for channel in config.channel_names:
        channel_state = normalized_state["channels"][channel]
        for candidate_id in sorted(channel_state["candidates"]):
            state = channel_state["candidates"][candidate_id]
            index.setdefault(candidate_id, []).append(
                {
                    "candidate_fingerprint": state["fingerprint"],
                    "candidate_id": candidate_id,
                    "channel": channel,
                    "dequantized": float(state["dequantized"]),
                    "fixed": int(state["fixed"]),
                }
            )
    return {candidate_id: sorted(items, key=lambda item: item["channel"]) for candidate_id, items in sorted(index.items())}


def expected_fingerprint(
    candidate_id: str,
    contributions: list[dict[str, Any]],
    config: FusionConfig,
) -> str:
    ordered = sorted(contributions, key=lambda item: item["channel"])
    return fusion_fingerprint(
        candidate_id=candidate_id,
        candidate_fingerprints=[item["candidate_fingerprint"] for item in ordered],
        channel_weights=[(item["channel"], config.channels[item["channel"]].weight) for item in ordered],
        trunc_k=config.trunc_k,
    )


def channel_contribution(
    channel: str,
    candidate_id: str,
    normalized_channel_state: dict[str, Any],
    config: FusionConfig,
) -> dict[str, Any]:
    state = normalized_channel_state["candidates"][candidate_id]
    weight = config.channels[channel].weight
    dequantized = float(state["dequantized"])
    return {
        "candidate_fingerprint": state["fingerprint"],
        "candidate_id": candidate_id,
        "channel": channel,
        "dequantized": dequantized,
        "fixed": int(state["fixed"]),
        "weight": weight,
        "weighted": dequantized * weight,
    }


def assemble_fusion_record(
    candidate_id: str,
    contributions: list[dict[str, Any]],
    config: FusionConfig,
) -> dict[str, Any]:
    ordered = sorted(contributions, key=lambda item: item["channel"])
    public_contributions: list[dict[str, Any]] = []
    fused_score = 0.0
    for order, item in enumerate(ordered):
        weighted = float(item["weighted"])
        fused_score += weighted
        public_contributions.append(
            {
                "candidate_fingerprint": item["candidate_fingerprint"],
                "channel": item["channel"],
                "dequantized": float(item["dequantized"]),
                "fixed": int(item["fixed"]),
                "order": order,
                "weight": float(item["weight"]),
                "weighted": weighted,
            }
        )
    fingerprint = fusion_fingerprint(
        candidate_id=candidate_id,
        candidate_fingerprints=[item["candidate_fingerprint"] for item in ordered],
        channel_weights=[(item["channel"], config.channels[item["channel"]].weight) for item in ordered],
        trunc_k=config.trunc_k,
    )
    return {
        "candidate_id": candidate_id,
        "contributions": public_contributions,
        "fused_score": fused_score,
        "fusion_fingerprint": fingerprint,
    }


def rank_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda item: (-float(item["fused_score"]), str(item["candidate_id"])))


def truncate_records(
    records: list[dict[str, Any]],
    config: FusionConfig,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if config.trunc_k == 0:
        return [], [
            failure(
                "truncate",
                "E_TRUNC_ZERO",
                "trunc_k is zero; truncation intentionally returns an empty set",
            )
        ]
    return rank_records(records)[: config.trunc_k], []


def build_observability(
    records: list[dict[str, Any]],
    kept: list[dict[str, Any]],
    reference_channels: dict[str, dict[str, float]],
    config: FusionConfig,
) -> dict[str, Any]:
    kept_ids = {item["candidate_id"] for item in kept}
    boundary_score = float(kept[-1]["fused_score"]) if kept else None
    max_weight = max((channel.weight for channel in config.channels.values()), default=0.0)
    boundary_eps = 4.0 * max_weight * config.quant_step
    neighborhood: list[dict[str, Any]] = []
    max_abs_error = 0.0
    if boundary_score is not None:
        for record in records:
            fused_score = float(record["fused_score"])
            if abs(fused_score - boundary_score) > boundary_eps:
                continue
            reference_score = 0.0
            for contribution in sorted(record["contributions"], key=lambda item: item["order"]):
                channel = contribution["channel"]
                reference_score += reference_channels[channel][record["candidate_id"]] * config.channels[channel].weight
            abs_error = abs(fused_score - reference_score)
            max_abs_error = max(max_abs_error, abs_error)
            neighborhood.append(
                {
                    "abs_error": abs_error,
                    "candidate_id": record["candidate_id"],
                    "fused_score": fused_score,
                    "kept": record["candidate_id"] in kept_ids,
                    "reference_score": reference_score,
                }
            )
    neighborhood.sort(key=lambda item: (-item["fused_score"], item["candidate_id"]))
    return {
        "boundary_eps": boundary_eps,
        "boundary_neighborhood": neighborhood,
        "boundary_score": boundary_score,
        "max_abs_error_at_boundary": max_abs_error if neighborhood else 0.0,
        "normalization": config.normalization,
        "quant_bits": config.quant_bits,
        "quant_step": config.quant_step,
        "rounding": config.rounding,
    }


def quantization_evidence(
    normalized_state: dict[str, Any],
    records: list[dict[str, Any]],
    kept: list[dict[str, Any]],
    config: FusionConfig,
) -> dict[str, Any]:
    fixed_values = {
        channel: [
            [candidate_id, normalized_state["channels"][channel]["candidates"][candidate_id]["fixed"]]
            for candidate_id in sorted(normalized_state["channels"][channel]["candidates"])
        ]
        for channel in config.channel_names
    }
    fused_values = [[record["candidate_id"], record["fused_score"]] for record in records]
    return {
        "fixed_checksum": sha256_json(fixed_values),
        "fused_checksum": sha256_json(fused_values),
        "kept_ids": [record["candidate_id"] for record in kept],
        "normalization": config.normalization,
        "quant_bits": config.quant_bits,
        "quant_step": config.quant_step,
        "rounding": config.rounding,
    }
