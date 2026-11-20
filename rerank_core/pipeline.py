"""单请求阶段计算：归一化 -> 融合 -> 重排，供串行与并发调度共用。"""

from __future__ import annotations

from dataclasses import dataclass

from . import cachekeys
from .config import RerankConfig
from .fixedpoint import stable_task_key
from .fusion import fuse_request, sort_fusion
from .normalize import compute_stats, quantize_request, raw_matrix
from .rerank import rerank_request
from .models import Candidate, RerankResult
from .serialize import fusion_from_dict, rerank_from_dict


@dataclass
class StageArtifacts:
    request_id: str
    request_fp: str
    normalize_key: str
    fusion_keys: dict[str, str]
    rerank_key: str
    discount_keys: list[str]
    stats: list
    quantized: list[list[int]]
    fusion_results: list
    fusion_order: list
    rerank: RerankResult


def compute_request_stages(
    request_id: str,
    candidates: list[Candidate],
    config: RerankConfig,
    *,
    need_normalize: bool,
    cached_normalize=None,
    need_fusion: bool,
    cached_fusion: dict | None,
    need_rerank: bool,
) -> StageArtifacts:
    request_fp = cachekeys.request_content_fingerprint(request_id, candidates, config)
    norm_param_fp = cachekeys.normalization_param_fingerprint(config)
    fusion_param_fp = cachekeys.fusion_param_fingerprint(config)
    rerank_param_fp = cachekeys.rerank_param_fingerprint(config)
    normalize_key = cachekeys.normalize_cache_key(request_fp, norm_param_fp)

    if need_normalize or cached_normalize is None:
        matrix = raw_matrix(candidates)
        stats = compute_stats(
            request_id, matrix, config.feature_dim, config.normalization
        )
        quantized = quantize_request(request_id, candidates, stats, config)
        normalize_payload = {
            "key": normalize_key,
            "stats": [stat.to_dict() for stat in stats],
            "quantized": quantized,
            "item_order": [c.item_id for c in candidates],
        }
    else:
        normalize_payload = cached_normalize
        stats = normalize_payload["stats"]
        quantized = normalize_payload["quantized"]

    # 候选任务稳定排序键：无论线程如何调度，计算与产出顺序都以该键为准
    ordered_indexes = sorted(
        range(len(candidates)),
        key=lambda pos: (stable_task_key(request_id, candidates[pos].item_id),
                         candidates[pos].item_id),
    )

    fusion_results_by_item: dict[str, object] = {}
    fusion_keys: dict[str, str] = {}
    if need_fusion:
        # 以稳定排序键顺序执行候选打分，保证并发与串行完全一致
        ordered_candidates = [candidates[pos] for pos in ordered_indexes]
        ordered_quantized = [quantized[pos] for pos in ordered_indexes]
        results = fuse_request(ordered_candidates, ordered_quantized, config)
        for candidate, result in zip(ordered_candidates, results):
            item_fp = cachekeys.item_feature_fingerprint(candidate, config)
            key = cachekeys.fusion_cache_key(
                request_fp, item_fp, candidate.item_id, fusion_param_fp
            )
            fusion_keys[candidate.item_id] = key
            fusion_results_by_item[candidate.item_id] = result
    else:
        assert cached_fusion is not None
        for item_id, payload in cached_fusion.items():
            fusion_results_by_item[item_id] = fusion_from_dict(payload)
            fusion_keys[item_id] = payload["key"]

    # 恢复为按候选稳定键顺序的结果列表，再做业务排序
    ordered_results = [fusion_results_by_item[candidates[pos].item_id]
                       for pos in ordered_indexes]
    fusion_order = sort_fusion(ordered_results, candidates, config)
    fusion_order_items = [result.item_id for result in fusion_order]

    rerank_key = cachekeys.rerank_cache_key(
        request_fp,
        cachekeys.stable_hash(fusion_order_items),
        rerank_param_fp,
    )
    if need_rerank:
        rerank_result = rerank_request(
            request_id, fusion_order, candidates, quantized, config
        )
    else:
        rerank_result = rerank_from_dict(cached_fusion)  # 调用方保证传入重排产物

    discount_keys = [
        cachekeys.discount_cache_key(
            rerank_key, record.step, record.target_item, record.reference_item
        )
        for record in rerank_result.discounts
    ]

    return StageArtifacts(
        request_id=request_id,
        request_fp=request_fp,
        normalize_key=normalize_key,
        fusion_keys=fusion_keys,
        rerank_key=rerank_key,
        discount_keys=discount_keys,
        stats=stats,
        quantized=quantized,
        fusion_results=fusion_order,
        fusion_order=fusion_order_items,
        rerank=rerank_result,
    )
