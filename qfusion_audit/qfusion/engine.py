"""定点融合引擎（要求 2/3）。

纯函数式处理单个请求：在每个 (frac_bits, gear) 组合下，于定点整数域完成
通道融合、特征叠加、裁剪与排序，逐候选输出逐通道/逐特征的有序分解。
所有运算仅依赖 Python int，结果与线程调度无关、跨次运行逐字段一致。
"""
from __future__ import annotations

from typing import Any, Dict, List

from .config import FusionConfig
from .fixedpoint import (STAGE_CLIP, STAGE_FEATURE, STAGE_FUSION,
                         STAGE_NORM, clip_int, quantize,
                         shift_right_half_up)
from .normalization import normalize_channel


def _q(value: float, frac_bits: int, gear: int) -> Dict[str, object]:
    return quantize(value, frac_bits, gear).to_dict()


def compute_request(request_id: str, request_data: Dict[str, List],
                    item_order: List[str], feature_lookup: Dict[str, List[float]],
                    cfg: FusionConfig) -> Dict[str, Any]:
    channel_names = cfg.channel_names_sorted
    # ---- 浮点归一化（与位宽/档位无关，每通道一次）----
    norm_float: Dict[str, Dict[str, float]] = {}
    events: List[Dict[str, object]] = []
    for chan in channel_names:
        spec = cfg.channels[chan]
        if chan in request_data and request_data[chan]:
            vals, ev = normalize_channel(request_data[chan], spec)
        else:
            vals, ev = {}, [{"channel": chan, "rule": "missing_channel",
                             "action": "set_zero"}]
        norm_float[chan] = vals
        events.extend(ev)

    settings: List[Dict[str, Any]] = []
    for frac_bits in cfg.frac_bits_choices:
        for gear in cfg.gears:
            setting = _compute_setting(
                request_id, frac_bits, gear, channel_names, item_order,
                feature_lookup, norm_float, events, cfg)
            settings.append(setting)

    return {
        "request_id": request_id,
        "events": events,
        "settings": settings,
        "item_order": item_order,
    }


def _compute_setting(request_id, frac_bits, gear, channel_names, item_order,
                     feature_lookup, norm_float, base_events, cfg):
    entries: List[Dict[str, Any]] = []
    # 温度在当前档位下的定点表示。
    temp_q = quantize(cfg.fusion_temperature, frac_bits, gear).q
    clip_lo = int(cfg.score_clip[0] * (1 << frac_bits))
    clip_hi = int(cfg.score_clip[1] * (1 << frac_bits))
    weight_q = {
        chan: quantize(cfg.channels[chan].weight, frac_bits, gear).q
        for chan in channel_names
    }
    feat_weight_q = [
        quantize(w, frac_bits, gear).q for w in cfg.feature_weights
    ]

    for item_id in item_order:
        channel_detail: Dict[str, Dict[str, Any]] = {}
        fusion_sum = 0
        for chan in channel_names:
            norm_val = norm_float[chan].get(item_id, 0.0)
            if item_id not in norm_float[chan]:
                pass
            nq = quantize(norm_val, frac_bits, gear)
            raw_product = nq.q * weight_q[chan]
            contribution = shift_right_half_up(raw_product, frac_bits)
            fusion_sum += contribution
            channel_detail[chan] = {
                "norm_float": norm_val,
                "norm_quant": nq.to_dict(),
                "weight_q": weight_q[chan],
                "raw_product": raw_product,
                "contribution_q": contribution,
            }

        # 固定运算顺序：通道字典序求和后，再除以温度。
        fusion_q = shift_right_half_up(fusion_sum * temp_q, frac_bits)

        features = feature_lookup[item_id]
        feature_detail: List[Dict[str, Any]] = []
        feature_sum = 0
        for idx, feat_val in enumerate(features):
            fq = quantize(feat_val, frac_bits, gear)
            raw_product = fq.q * feat_weight_q[idx]
            contribution = shift_right_half_up(raw_product, frac_bits)
            feature_sum += contribution
            feature_detail.append({
                "feature_index": idx,
                "feature_float": feat_val,
                "feature_quant": fq.to_dict(),
                "weight_q": feat_weight_q[idx],
                "raw_product": raw_product,
                "contribution_q": contribution,
            })

        overlay_q = fusion_q + feature_sum
        clipped_q = clip_int(overlay_q, clip_lo, clip_hi)
        clipped = not (clip_lo <= overlay_q <= clip_hi)

        entries.append({
            "request_id": request_id,
            "item_id": item_id,
            "frac_bits": frac_bits,
            "gear": gear,
            "channels": channel_detail,
            "features": feature_detail,
            "stages": {
                STAGE_NORM: {
                    chan: channel_detail[chan]["norm_quant"]["q"]
                    for chan in channel_names
                },
                STAGE_FUSION: fusion_q,
                STAGE_FEATURE: overlay_q,
                STAGE_CLIP: clipped_q,
            },
            "clipped": clipped,
            "sort_key": [clipped_q, fusion_q, feature_sum, item_id],
        })

    # 排序：裁剪分 -> 融合分 -> 特征贡献 -> 物品 ID 字典序。
    ranked = sorted(entries, key=lambda e: (
        -e["stages"][STAGE_CLIP],
        -e["stages"][STAGE_FUSION],
        -(e["stages"][STAGE_FEATURE] - e["stages"][STAGE_FUSION]),
        e["item_id"],
    ))
    rank_of: Dict[str, int] = {}
    for pos, entry in enumerate(ranked, 1):
        entry["rank"] = pos
        rank_of[entry["item_id"]] = pos

    return {
        "frac_bits": frac_bits,
        "gear": gear,
        "temp_q": temp_q,
        "clip_bounds_q": [clip_lo, clip_hi],
        "weight_q": weight_q,
        "entries": entries,
        "rank_of": rank_of,
        "ranked_item_ids": [e["item_id"] for e in ranked],
    }
