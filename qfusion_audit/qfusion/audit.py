"""跨位宽/跨档位量化审计账本（要求 4/5）。

不同小数位宽的定点整数不在同一尺度，直接相减会把尺度差异误判为量化误差。
本模块把所有中间值统一到公共定点单位 ``U = 2^-F``（F 为候选集合中的最大
小数位宽）：q_F = q_f << (F-f)，再做逐环节比较，保证差值反映真实量化差异。

产物：
- ledger：逐候选、逐 (位宽, 档位)、逐环节定点中间值、舍入方向与名次；
- setting_pair_audit：两组档位间名次变化、首个分歧环节及该环节定点差值；
- flips：翻转候选 + 对端候选 + 各通道贡献差 + 并列决断标记；
- boundary_pairs：每个 (位宽, 档位) 下融合差落入量化半步的候选对，
  附最高位宽参考差值与判定结论。
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .fixedpoint import (STAGE_CLIP, STAGE_FEATURE, STAGE_FUSION,
                         STAGE_NORM, STAGE_ORDER, quantization_step)


def _setting_index(request_result: Dict[str, Any]
                   ) -> Dict[Tuple[int, int], Dict[str, Any]]:
    return {(s["frac_bits"], s["gear"]): s
            for s in request_result["settings"]}


def _channel_contribs(entry: Dict[str, Any]) -> Dict[str, int]:
    return {chan: detail["contribution_q"]
            for chan, detail in entry["channels"].items()}


def build_request_ledger(request_result: Dict[str, Any],
                         max_frac: int) -> Dict[str, Any]:
    item_order = request_result["item_order"]
    idx = _setting_index(request_result)
    settings = sorted(idx)

    per_item: Dict[str, Dict[str, Any]] = {}
    for (frac_bits, gear), setting in idx.items():
        for entry in setting["entries"]:
            slot = per_item.setdefault(entry["item_id"], {
                "request_id": entry["request_id"],
                "item_id": entry["item_id"],
                "settings": [],
            })
            slot["settings"].append({
                "frac_bits": frac_bits,
                "gear": gear,
                "rank": entry["rank"],
                "stages": {
                    stage: _stage_at_unit(entry, stage, max_frac)
                    for stage in STAGE_ORDER
                },
                "channel_contributions_q": {
                    chan: detail["contribution_q"] << (max_frac - frac_bits)
                    for chan, detail in entry["channels"].items()
                },
                "feature_contributions_q": [
                    f["contribution_q"] << (max_frac - frac_bits)
                    for f in entry["features"]
                ],
                "rounding_directions": {
                    chan: entry["channels"][chan]["norm_quant"]["direction"]
                    for chan in entry["channels"]
                },
            })

    for item_id in per_item:
        per_item[item_id]["settings"].sort(
            key=lambda s: (s["frac_bits"], s["gear"]))

    pair_audits: List[Dict[str, Any]] = []
    flips: List[Dict[str, Any]] = []
    for a in range(len(settings)):
        for b in range(a + 1, len(settings)):
            key_a, key_b = settings[a], settings[b]
            audit, pair_flips = _compare_settings(
                request_result, idx[key_a], idx[key_b], max_frac)
            pair_audits.append(audit)
            flips.extend(pair_flips)

    boundary_pairs = _all_boundary_pairs(idx, settings, item_order, max_frac)

    return {
        "request_id": request_result["request_id"],
        "item_order": item_order,
        "common_unit_frac_bits": max_frac,
        "per_item": per_item,
        "setting_pair_audit": pair_audits,
        "flips": flips,
        "boundary_pairs": boundary_pairs,
    }


def _stage_at_unit(entry: Dict[str, Any], stage: str,
                   max_frac: int) -> Any:
    shift = max_frac - entry["frac_bits"]
    value = entry["stages"][stage]
    if stage == STAGE_NORM:
        return {chan: int(q) << shift for chan, q in value.items()}
    return int(value) << shift



def _common_unit_rank(setting_a, setting_b, max_frac, item_order):
    """在公共定点单位下，对两个档位的条目按同一排序键重算名次。

    名次比较必须在同一尺度上进行，否则不同小数位宽的原始整数排序会把
    纯尺度差异误判为翻转。返回 (rank_a, rank_b, entry_a, entry_b)。
    """
    shift_a = max_frac - setting_a["frac_bits"]
    shift_b = max_frac - setting_b["frac_bits"]
    ea = {e["item_id"]: e for e in setting_a["entries"]}
    eb = {e["item_id"]: e for e in setting_b["entries"]}

    def key_of(entry, shift):
        stages = entry["stages"]
        feat_sum = stages[STAGE_FEATURE] - stages[STAGE_FUSION]
        return (-int(stages[STAGE_CLIP]) << shift,
                -int(stages[STAGE_FUSION]) << shift,
                -feat_sum << shift,
                entry["item_id"])

    order_a = sorted(item_order, key=lambda i: key_of(ea[i], shift_a))
    order_b = sorted(item_order, key=lambda i: key_of(eb[i], shift_b))
    rank_a = {item: pos for pos, item in enumerate(order_a, 1)}
    rank_b = {item: pos for pos, item in enumerate(order_b, 1)}
    return rank_a, rank_b, ea, eb


def _stage_diff_at_unit(stage: str, entry_a: Dict[str, Any],
                        entry_b: Dict[str, Any], max_frac: int) -> Any:
    va = _stage_at_unit(entry_a, stage, max_frac)
    vb = _stage_at_unit(entry_b, stage, max_frac)
    if isinstance(va, dict):
        return {chan: vb[chan] - va[chan] for chan in va}
    return vb - va


def _first_diverged_stage(entry_a: Dict[str, Any], entry_b: Dict[str, Any],
                          max_frac: int) -> Tuple[str | None, Any]:
    for stage in STAGE_ORDER:
        diff = _stage_diff_at_unit(stage, entry_a, entry_b, max_frac)
        changed = any(v != 0 for v in diff.values()) \
            if isinstance(diff, dict) else diff != 0
        if changed:
            return stage, diff
    return None, 0


def _compare_settings(request_result, setting_a, setting_b, max_frac):
    key_a = (setting_a["frac_bits"], setting_a["gear"])
    key_b = (setting_b["frac_bits"], setting_b["gear"])
    item_order = request_result["item_order"]
    rank_a, rank_b, entries_a, entries_b = _common_unit_rank(
        setting_a, setting_b, max_frac, item_order)

    item_changes: List[Dict[str, Any]] = []
    flips: List[Dict[str, Any]] = []
    for item_id in item_order:
        ea, eb = entries_a[item_id], entries_b[item_id]
        first_stage, stage_diff = _first_diverged_stage(ea, eb, max_frac)
        ra, rb = rank_a[item_id], rank_b[item_id]
        item_changes.append({
            "item_id": item_id,
            "rank_a": ra,
            "rank_b": rb,
            "rank_delta": rb - ra,
            "first_diverged_stage": first_stage,
            "stage_diff_common_unit": stage_diff,
        })
        if ra != rb:
            flips.append(_build_flip(item_id, ea, eb, rank_a, rank_b,
                                     setting_a, setting_b,
                                     key_a, key_b, item_order, max_frac))

    audit = {
        "setting_a": {"frac_bits": key_a[0], "gear": key_a[1]},
        "setting_b": {"frac_bits": key_b[0], "gear": key_b[1]},
        "common_unit_frac_bits": max_frac,
        "item_changes": item_changes,
        "flipped_item_ids": [c["item_id"] for c in item_changes
                             if c["rank_a"] != c["rank_b"]],
    }
    return audit, flips


def _build_flip(item_id, ea, eb, rank_a, rank_b, setting_a, setting_b,
                key_a, key_b, item_order, max_frac) -> Dict[str, Any]:
    counterpart = next(
        (other for other in item_order
         if rank_b[other] == rank_a[item_id] and other != item_id),
        None)
    shift_a = max_frac - key_a[0]
    shift_b = max_frac - key_b[0]
    contribs_a = {c: d["contribution_q"] << shift_a
                  for c, d in ea["channels"].items()}
    contribs_b = {c: d["contribution_q"] << shift_b
                  for c, d in eb["channels"].items()}
    channel_delta = {chan: contribs_b[chan] - contribs_a[chan]
                     for chan in contribs_a}
    tie_break_triggered = (
        _stage_at_unit(eb, STAGE_CLIP, max_frac)
        == _stage_at_unit(ea, STAGE_CLIP, max_frac)
        and _stage_at_unit(eb, STAGE_FUSION, max_frac)
        == _stage_at_unit(ea, STAGE_FUSION, max_frac)
    )
    return {
        "item_id": item_id,
        "rank_before": rank_a[item_id],
        "rank_after": rank_b[item_id],
        "setting_before": {"frac_bits": key_a[0], "gear": key_a[1]},
        "setting_after": {"frac_bits": key_b[0], "gear": key_b[1]},
        "counterpart_item_id": counterpart,
        "fusion_q_common_unit_before":
            ea["stages"][STAGE_FUSION] << shift_a,
        "fusion_q_common_unit_after":
            eb["stages"][STAGE_FUSION] << shift_b,
        "channel_contribution_delta_common_unit": channel_delta,
        "tie_break_triggered": tie_break_triggered,
    }


def _all_boundary_pairs(idx, settings, item_order, max_frac
                        ) -> List[Dict[str, Any]]:
    """每个 (位宽, 档位) 独立判定边界对，并附最高位宽参考差值。"""
    finest = idx[(max_frac, min(s[1] for s in settings))]
    finest_entries = {e["item_id"]: e for e in finest["entries"]}
    finest_scale = 1 << max_frac
    out: List[Dict[str, Any]] = []

    for frac_bits, gear in settings:
        setting = idx[(frac_bits, gear)]
        entries = {e["item_id"]: e for e in setting["entries"]}
        rank_finest, rank_here, _, _ = _common_unit_rank(
            finest, setting, max_frac, item_order)
        half_step = quantization_step(frac_bits, gear) / 2.0
        scale = 1 << frac_bits
        for i in range(len(item_order)):
            for j in range(i + 1, len(item_order)):
                x, y = item_order[i], item_order[j]
                ex, ey = entries[x], entries[y]
                diff_q = abs(ex["stages"][STAGE_FUSION]
                             - ey["stages"][STAGE_FUSION])
                diff_real = diff_q / scale
                if diff_real <= half_step:
                    ref_q = abs(
                        finest_entries[x]["stages"][STAGE_FUSION]
                        - finest_entries[y]["stages"][STAGE_FUSION])
                    ref_real = ref_q / finest_scale
                    rank_changed = (
                        rank_finest[x] != rank_here[x]
                        or rank_finest[y] != rank_here[y])
                    out.append({
                        "item_a": x,
                        "item_b": y,
                        "frac_bits": frac_bits,
                        "gear": gear,
                        "fusion_diff_q": diff_q,
                        "fusion_diff_real": diff_real,
                        "half_step": half_step,
                        "within_half_step": True,
                        "highest_frac_ref_diff_q": ref_q,
                        "highest_frac_ref_diff_real": ref_real,
                        "rank_changed_vs_finest": rank_changed,
                        "verdict": "flip" if rank_changed
                        else "stable_within_half_step",
                    })
    return out


def highest_frac_reference(request_results: Dict[str, Any],
                           cfg) -> Dict[str, Any]:
    """最高位宽、最细档位（gear 最小值）下的参考名次。"""
    max_f = max(cfg.frac_bits_choices)
    finest_gear = cfg.gears[0]
    reference: Dict[str, Any] = {}
    for req_id, result in request_results.items():
        idx = _setting_index(result)
        setting = idx[(max_f, finest_gear)]
        reference[req_id] = {
            "frac_bits": max_f,
            "gear": finest_gear,
            "rank_of": setting["rank_of"],
        }
    return reference
