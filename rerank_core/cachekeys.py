"""请求内容指纹、参数指纹与各阶段缓存键的确定性派生。"""

from __future__ import annotations

from fractions import Fraction

from .config import RerankConfig
from .fixedpoint import compound_key, stable_hash
from .models import Candidate


def candidate_payload(candidate: Candidate, config: RerankConfig) -> list:
    return [
        candidate.item_id,
        [candidate.recall_scores[ch] for ch in config.recall_channels],
        list(candidate.dense),
        list(candidate.sparse),
    ]


def request_content_fingerprint(
    request_id: str, candidates: list[Candidate], config: RerankConfig
) -> str:
    # 按物品 ID 规范化排序，使指纹与文件行顺序无关，仅由内容决定
    payload = [
        candidate_payload(candidate, config)
        for candidate in sorted(candidates, key=lambda c: c.item_id)
    ]
    return stable_hash(["request", request_id, payload])


def item_feature_fingerprint(candidate: Candidate, config: RerankConfig) -> str:
    return stable_hash(["item", candidate_payload(candidate, config)])


def normalization_param_fingerprint(config: RerankConfig) -> str:
    return stable_hash(
        [
            "normalize_param",
            config.normalization,
            config.frac_bits,
            config.quant_levels,
        ]
    )


def fusion_param_fingerprint(config: RerankConfig) -> str:
    return stable_hash(
        [
            "fusion_param",
            [str(weight) for weight in config.weights],
            str(config.fusion_temperature),
            [str(config.score_clip[0]), str(config.score_clip[1])],
            config.frac_bits,
            config.quant_levels,
        ]
    )


def rerank_param_fingerprint(config: RerankConfig) -> str:
    return stable_hash(
        [
            "rerank_param",
            config.rerank_window,
            str(config.diversity_penalty),
            str(config.similarity_threshold),
            config.frac_bits,
        ]
    )


def normalize_cache_key(request_fp: str, norm_param_fp: str) -> str:
    return compound_key("stage:normalize", request_fp, norm_param_fp)


def fusion_cache_key(
    request_fp: str, item_fp: str, item_id: str, fusion_param_fp: str
) -> str:
    return compound_key("stage:fusion", request_fp, item_fp, item_id, fusion_param_fp)


def rerank_cache_key(request_fp: str, fusion_bundle_fp: str, rerank_param_fp: str) -> str:
    return compound_key("stage:rerank", request_fp, fusion_bundle_fp, rerank_param_fp)


def discount_cache_key(
    rerank_key: str, step: int, target: str, reference: str
) -> str:
    return compound_key("discount", rerank_key, step, target, reference)
