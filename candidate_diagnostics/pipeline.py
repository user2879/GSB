from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal
from typing import Any

from . import VERSION_FINGERPRINT
from .config import Config
from .fingerprinting import derive_cache_key, fingerprint, stable_seed
from .models import Candidate, Request


ROUNDING = {
    "half_up": ROUND_HALF_UP,
    "half_even": ROUND_HALF_EVEN,
    "floor": ROUND_FLOOR,
    "ceil": ROUND_CEILING,
}


def round_decimal_integer(value: Decimal, rounding: str) -> int:
    return int(value.to_integral_value(rounding=ROUNDING[rounding]))


def quantize_score(
    normalized_score: float,
    quant_config: dict[str, Any],
    *,
    reference: bool = False,
) -> tuple[int, float, int]:
    """Return fixed integer, dequantized score and decimal width."""
    if reference:
        decimals = int(
            quant_config.get(
                "reference_fixed_decimals", int(quant_config["fixed_decimals"]) + 1
            )
        )
        gear = 0
        bit_width = int(quant_config.get("reference_bit_width", quant_config["bit_width"] + 1))
    else:
        decimals = int(quant_config["fixed_decimals"])
        gear = int(quant_config.get("gear_exponent", 0))
        bit_width = int(quant_config.get("bit_width", 32))

    rounding = str(quant_config.get("rounding", "half_up"))
    scale = Decimal(10) ** decimals
    scaled = Decimal(repr(float(normalized_score))) * scale
    fixed = round_decimal_integer(scaled, rounding)
    grid = 10**gear
    if grid > 1:
        grid_decimal = Decimal(grid)
        fixed = round_decimal_integer(Decimal(fixed) / grid_decimal, rounding) * grid

    max_fixed = 2 ** (bit_width - 1) - 1
    if fixed > max_fixed or fixed < -(2 ** (bit_width - 1)):
        fixed = max(-(2 ** (bit_width - 1)), min(max_fixed, fixed))
    return fixed, float(Decimal(fixed) / scale), decimals


def normalize_candidate(
    candidate: Candidate, config: Config, *, reference: bool = False
) -> dict[str, Any]:
    norm = config.raw["normalization"]
    raw_scale = float(norm["raw_scale"])
    weight = float(config.raw["channels"][candidate.channel])
    raw_ratio = candidate.raw_score / raw_scale
    method = norm["method"]
    if method == "weighted_linear":
        normalized = (
            float(norm["output_scale"])
            * (float(norm["slope"]) * weight * raw_ratio + float(norm["bias"]))
        )
    elif method == "unweighted_linear":
        normalized = float(norm["output_scale"]) * (
            float(norm["slope"]) * raw_ratio + float(norm["bias"])
        )
    else:
        raise ValueError(f"unsupported normalization method: {method}")

    if norm.get("clamp", True):
        normalized = max(0.0, min(float(norm["output_scale"]), normalized))

    fixed, dequant, decimals = quantize_score(normalized, config.raw["quantization"], reference=reference)
    return {
        "candidate_id": candidate.candidate_id,
        "channel": candidate.channel,
        "aux_feature": candidate.aux_feature,
        "content_key": candidate.content_key,
        "raw_score": candidate.raw_score,
        "channel_weight": weight,
        "normalized_float": normalized,
        "fixed_score": fixed,
        "dequant_score": dequant,
        "fixed_decimals": decimals,
    }


def request_content_fingerprint(request: Request) -> str:
    return fingerprint(request.to_dict(), 20)


def segment_fingerprint(candidates: list[Candidate], channel: str, config: Config) -> str:
    effective_quantization = {
        key: value
        for key, value in config.raw["quantization"].items()
        if not key.startswith("reference_")
    }
    payload = {
        "channel": channel,
        "candidates": [candidate.to_dict() for candidate in sorted(candidates, key=lambda c: c.candidate_id)],
        "weight": config.raw["channels"][channel],
        "normalization": config.raw["normalization"],
        "quantization": effective_quantization,
    }
    return fingerprint(payload, 20)


@dataclass
class RequestResult:
    request_id: str
    content_fingerprint: str
    channels: list[str]
    candidate_ids: list[int]
    scores: list[dict[str, Any]]
    ranked_ids: list[int]
    top_k_ids: list[int]
    truncated_ids: list[int]
    cutoff_score: int | None
    boundary: dict[str, Any]
    layers: dict[str, dict[str, Any]]
    sampled_ids: list[int]
    unsampled_topk_ids: list[int]
    shuffle_order: list[int]
    accumulation_trace: list[dict[str, Any]]
    segment_fingerprints: dict[str, str]
    segment_cache_keys: dict[str, str]
    cache_keys: dict[str, str]
    config_hashes: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "content_fingerprint": self.content_fingerprint,
            "channels": self.channels,
            "candidate_ids": self.candidate_ids,
            "scores": self.scores,
            "ranked_ids": self.ranked_ids,
            "top_k_ids": self.top_k_ids,
            "truncated_ids": self.truncated_ids,
            "cutoff_score": self.cutoff_score,
            "boundary": self.boundary,
            "layers": self.layers,
            "sampled_ids": self.sampled_ids,
            "unsampled_topk_ids": self.unsampled_topk_ids,
            "shuffle_order": self.shuffle_order,
            "accumulation_trace": self.accumulation_trace,
            "segment_fingerprints": self.segment_fingerprints,
            "segment_cache_keys": self.segment_cache_keys,
            "cache_keys": self.cache_keys,
            "config_hashes": self.config_hashes,
        }


def _normalization_cache_key(request: Request, config: Config, content_fp: str) -> str:
    referenced_weights = {
        channel: config.raw["channels"][channel] for channel in sorted(request.channels)
    }
    return derive_cache_key(
        "normalization",
        upstream_content=content_fp,
        channel_weights=fingerprint(referenced_weights, 16),
        sampling=config.sampling_fingerprint,
        version=config.derive_version_fingerprint(VERSION_FINGERPRINT),
        quantization=config.quantization_fingerprint,
    )


def analyze_request(
    request: Request,
    config: Config,
    segment_cache: dict[str, list[dict[str, Any]]] | None = None,
) -> RequestResult:
    content_fp = request_content_fingerprint(request)
    channels = sorted(request.channels)
    segment_fps: dict[str, str] = {}
    segment_keys: dict[str, str] = {}
    trace: list[dict[str, Any]] = []
    accumulated = 0.0
    order_no = 0
    scores: list[dict[str, Any]] = []

    for channel in channels:
        channel_candidates = sorted(
            [c for c in request.candidates if c.channel == channel],
            key=lambda candidate: candidate.candidate_id,
        )
        segment_fps[channel] = segment_fingerprint(channel_candidates, channel, config)
        segment_key = derive_cache_key(
            "normalization_segment",
            segment_fingerprint=segment_fps[channel],
            version=config.derive_version_fingerprint(VERSION_FINGERPRINT),
        )
        segment_keys[channel] = segment_key
        if segment_cache is not None and segment_key in segment_cache:
            segment_scores = segment_cache[segment_key]
        else:
            segment_scores = [normalize_candidate(c, config) for c in channel_candidates]
            if segment_cache is not None:
                segment_cache[segment_key] = segment_scores
        scores.extend(segment_scores)

        # Accumulation audit always walks channel asc/candidate id asc.
        for candidate in channel_candidates:
            order_no += 1
            contribution = (
                float(config.raw["channels"][channel])
                * candidate.raw_score
                / float(config.raw["normalization"]["raw_scale"])
            )
            accumulated += contribution
            trace.append(
                {
                    "order": order_no,
                    "channel": channel,
                    "candidate_id": candidate.candidate_id,
                    "weighted_contribution": contribution,
                    "accumulated": accumulated,
                }
            )

    score_by_id = {item["candidate_id"]: item for item in scores}
    ranked = sorted(
        scores,
        key=lambda item: (-int(item["fixed_score"]), int(item["candidate_id"])),
    )
    ranked_ids = [int(item["candidate_id"]) for item in ranked]
    top_k = min(config.top_k, len(ranked_ids))
    top_k_ids = ranked_ids[:top_k]
    truncated_ids = ranked_ids[top_k:]

    cutoff_score = int(score_by_id[top_k_ids[-1]]["fixed_score"]) if top_k_ids else None
    boundary: dict[str, Any] = {
        "cutoff_score": cutoff_score,
        "boundary_selected_id": None,
        "nearest_rejected_id": None,
        "tied_ids": [],
        "boundary_selected_ids": [],
        "boundary_rejected_ids": [],
        "tie_at_cutoff": False,
    }
    if cutoff_score is not None:
        tied_ids = sorted(
            cid for cid, item in score_by_id.items() if int(item["fixed_score"]) == cutoff_score
        )
        selected_set = set(top_k_ids)
        boundary["tied_ids"] = tied_ids
        boundary["boundary_selected_ids"] = [cid for cid in tied_ids if cid in selected_set]
        boundary["boundary_rejected_ids"] = [cid for cid in tied_ids if cid not in selected_set]
        boundary["tie_at_cutoff"] = bool(boundary["boundary_rejected_ids"])
        if top_k_ids:
            last_id = top_k_ids[-1]
            if int(score_by_id[last_id]["fixed_score"]) == cutoff_score:
                boundary["boundary_selected_id"] = last_id
        if truncated_ids:
            next_id = truncated_ids[0]
            if int(score_by_id[next_id]["fixed_score"]) == cutoff_score:
                boundary["nearest_rejected_id"] = next_id

    layers = _sample_layers(request, config, top_k_ids, score_by_id)
    sampled_ids = sorted(
        cid for layer in layers.values() for cid in layer["selected_ids"]
    )
    unsampled_topk_ids = sorted(
        cid for layer in layers.values() for cid in layer["unselected_ids"]
    )
    shuffle_order = _shuffle_sampled(config, request.request_id, sampled_ids)

    norm_key = _normalization_cache_key(request, config, content_fp)
    trunc_key = derive_cache_key(
        "truncation",
        upstream=norm_key,
        top_k=config.top_k,
        version=config.derive_version_fingerprint(VERSION_FINGERPRINT),
    )
    sample_key = derive_cache_key(
        "sampling",
        upstream=trunc_key,
        sampling=config.sampling_fingerprint,
        content=content_fp,
        version=config.derive_version_fingerprint(VERSION_FINGERPRINT),
    )
    shuffle_key = derive_cache_key(
        "shuffle",
        upstream=sample_key,
        content=content_fp,
        version=config.derive_version_fingerprint(VERSION_FINGERPRINT),
    )

    return RequestResult(
        request_id=request.request_id,
        content_fingerprint=content_fp,
        channels=channels,
        candidate_ids=sorted(candidate.candidate_id for candidate in request.candidates),
        scores=scores,
        ranked_ids=ranked_ids,
        top_k_ids=top_k_ids,
        truncated_ids=truncated_ids,
        cutoff_score=cutoff_score,
        boundary=boundary,
        layers=layers,
        sampled_ids=sampled_ids,
        unsampled_topk_ids=unsampled_topk_ids,
        shuffle_order=shuffle_order,
        accumulation_trace=trace,
        segment_fingerprints=segment_fps,
        segment_cache_keys=segment_keys,
        cache_keys={
            "normalization": norm_key,
            "truncation": trunc_key,
            "sampling": sample_key,
            "shuffle": shuffle_key,
        },
        config_hashes={
            "business_config": config.business_fingerprint,
            "channel_weights": config.channel_weights_fingerprint,
            "sampling": config.sampling_fingerprint,
            "quantization": config.quantization_fingerprint,
            "version": config.derive_version_fingerprint(VERSION_FINGERPRINT),
        },
    )


def _sample_layers(
    request: Request,
    config: Config,
    top_k_ids: list[int],
    score_by_id: dict[int, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    top_set = set(top_k_ids)
    ratio = config.ratio
    total_population = len(top_k_ids)
    total_quota = int(total_population * ratio)
    top_candidates = [c for c in request.candidates if c.candidate_id in top_set]
    layers: dict[str, dict[str, Any]] = {}
    channels = sorted({c.channel for c in top_candidates})
    populations = {
        channel: sorted(
            [c.candidate_id for c in top_candidates if c.channel == channel]
        )
        for channel in channels
    }
    base_quotas = {
        channel: int(len(ids) * ratio)
        for channel, ids in populations.items()
    }
    remainder = total_quota - sum(base_quotas.values())
    quotas = dict(base_quotas)
    top_ups: dict[str, dict[str, Any]] = {}
    for index, channel in enumerate(channels):
        if index < remainder:
            quotas[channel] += 1
            top_ups[channel] = {
                "top_up": 1,
                "basis": "remainder_by_channel_name_asc",
                "remainder_index": index,
                "remainder_count": remainder,
            }
        else:
            top_ups[channel] = {"top_up": 0, "basis": None, "remainder_count": remainder}

    for channel in channels:
        ids = populations[channel]
        feature_groups: dict[int, list[int]] = {}
        candidate_by_id = {c.candidate_id: c for c in top_candidates if c.channel == channel}
        for cid in ids:
            feature_groups.setdefault(candidate_by_id[cid].aux_feature, []).append(cid)
        for group in feature_groups.values():
            group.sort()

        draw_keys: dict[int, int] = {}
        for aux_value in sorted(feature_groups):
            for cid in feature_groups[aux_value]:
                draw_keys[cid] = stable_seed(
                    config.sampling_fingerprint,
                    request.request_id,
                    "sample-draw",
                    channel,
                    aux_value,
                    cid,
                )
        draw_ranking = sorted(ids, key=lambda cid: (draw_keys[cid], cid))
        quota = min(int(quotas[channel]), len(ids))
        selected = sorted(draw_ranking[:quota])
        unselected = sorted(cid for cid in ids if cid not in set(selected))
        layers[channel] = {
            "channel": channel,
            "population_ids": ids,
            "population": len(ids),
            "exact_quota": len(ids) * ratio,
            "base_quota": base_quotas[channel],
            "quota": quota,
            "ratio_permille": config.ratio_permille,
            "remainder": remainder,
            **top_ups[channel],
            "feature_groups": {str(k): sorted(v) for k, v in sorted(feature_groups.items())},
            "draw_keys": {str(k): draw_keys[k] for k in sorted(draw_keys)},
            "draw_ranking": draw_ranking,
            "selected_ids": selected,
            "unselected_ids": unselected,
            "scheme": config.raw["sampling"]["scheme"],
            "random_seed": stable_seed(
                config.sampling_fingerprint, request.request_id, "layer", channel
            ),
            "random_source": "sha256_content_fingerprint_prf",
        }
    return layers


def _shuffle_sampled(config: Config, request_id: str, sampled_ids: list[int]) -> list[int]:
    keys = {
        cid: stable_seed(
            config.sampling_fingerprint, request_id, "shuffle-position", cid
        )
        for cid in sorted(sampled_ids)
    }
    return sorted(sampled_ids, key=lambda cid: (keys[cid], cid))


def analyze_many(
    requests: dict[str, Request] | list[Request], config: Config
) -> dict[str, RequestResult]:
    if isinstance(requests, list):
        request_map = {request.request_id: request for request in requests}
    else:
        request_map = requests
    return {rid: analyze_request(request_map[rid], config) for rid in sorted(request_map)}
