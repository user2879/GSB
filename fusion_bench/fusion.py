"""跨模态定点融合打分：减最大值稳定加权融合 + 冲突折减 + 区间裁剪。"""

import numpy as np

from .fixedpoint import div_round, exp_fixed, mean_fixed, mul_fixed


def fuse(visual_feat, text_feat, cfg):
    """返回融合结果字典（全部定点整数，确定性舍入）。"""
    fb = cfg.frac_bits
    v_summary = mean_fixed(visual_feat)
    t_summary = mean_fixed(text_feat)
    # logit = w * summary / temperature（三者均为定点，SCALE 相约）
    logit_v = div_round(cfg.w_visual * v_summary, cfg.temperature)
    logit_t = div_round(cfg.w_text * t_summary, cfg.temperature)
    m = max(logit_v, logit_t)
    e_v = exp_fixed(logit_v - m, fb)
    e_t = exp_fixed(logit_t - m, fb)
    total = e_v + e_t
    alpha_v = div_round(e_v * cfg.scale, total)
    alpha_t = div_round(e_t * cfg.scale, total)
    contrib_v = mul_fixed(alpha_v, v_summary, fb)
    contrib_t = mul_fixed(alpha_t, t_summary, fb)
    score = int(np.clip(contrib_v + contrib_t,
                        cfg.score_clip_lo, cfg.score_clip_hi))
    borderline = score == cfg.decision_threshold
    # 固定规则：得分 >= 阈值判定 accept；恰等于阈值额外标注边界样本
    decision = "accept" if score >= cfg.decision_threshold else "reject"
    return {
        "score": score,
        "decision": decision,
        "borderline": borderline,
        "alpha": {"visual": alpha_v, "text": alpha_t},
        "contributions": {
            "visual": {"summary": v_summary, "weighted": contrib_v},
            "text": {"summary": t_summary, "weighted": contrib_t},
        },
    }
