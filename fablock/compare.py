"""定点主路径与 float64 参考路径的逐块逐环节误差比对。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

STAGES = ("score", "softmax_state", "output")


@dataclass
class BlockDeviation:
    query_block: int
    key_block: int
    stage: str
    max_abs_diff: float

    def to_dict(self):
        return {"query_block": self.query_block, "key_block": self.key_block,
                "stage": self.stage, "max_abs_diff": self.max_abs_diff}


@dataclass
class CompareResult:
    # (query_block, key_block) -> 各环节最大绝对差
    stage_diffs: dict = field(default_factory=dict)
    # 超差块：[(query_block, key_block, first_stage)]
    out_of_tolerance: list = field(default_factory=list)
    # (query_block, key_block) -> 输出环节残差
    residuals: dict = field(default_factory=dict)
    # 最终输出逐元素差
    final_out_diffs: dict = field(default_factory=dict)

    def stats(self):
        per_stage = {s: 0.0 for s in STAGES}
        for diffs in self.stage_diffs.values():
            for s in STAGES:
                per_stage[s] = max(per_stage[s], diffs.get(s, 0.0))
        return {
            "max_abs_diff_per_stage": per_stage,
            "out_of_tolerance_count": len(self.out_of_tolerance),
            "out_of_tolerance": [
                {"query_block": qi, "key_block": ki, "first_stage": st}
                for qi, ki, st in self.out_of_tolerance],
        }


def compare_query_block(int_trace, score_blocks, ref_trace, cfg,
                        compare: CompareResult) -> None:
    """比对单个查询块：块内打分、运行最大值与分母、输出累加三个环节。"""
    frac = cfg.fixed_point_frac_bits
    scale2 = float(1 << (2 * frac))
    scale1 = float(1 << frac)
    qi = int_trace.query_block
    for key_block, (sblk, state, ref_state) in enumerate(
            zip(score_blocks, int_trace.states, ref_trace.states)):
        s_int = np.array(sblk.scores, dtype=np.float64) / scale2
        d_score = float(np.max(np.abs(s_int - ref_state.scores))) \
            if s_int.size else 0.0
        m_int = np.float64(state.m_fp) / scale2
        l_int = np.array(state.l_fp, dtype=np.float64) / scale2
        d_state = max(float(np.max(np.abs(m_int - ref_state.m))),
                      float(np.max(np.abs(l_int - ref_state.denom))))
        o_int = np.array(state.out_acc, dtype=np.float64) / scale2
        d_out = float(np.max(np.abs(o_int - ref_state.out_acc))) \
            if o_int.size else 0.0
        diffs = {"score": d_score, "softmax_state": d_state, "output": d_out}
        compare.stage_diffs[(qi, sblk.key_block)] = diffs
        compare.residuals[(qi, sblk.key_block)] = d_out
        for stage in STAGES:
            if diffs[stage] > cfg.tolerance_for(stage):
                compare.out_of_tolerance.append((qi, sblk.key_block, stage))
                break
    out_int = np.array(int_trace.out_fp, dtype=np.float64) / scale1
    compare.final_out_diffs[qi] = out_int - ref_trace.out
