"""整数域分块乘加 + 在线 softmax 内核。

数据流定点标尺（f = fixed_point_frac_bits）：
    Q,K,V 量化值            2^f
    打分 S = Q·K / sqrt(d)  2^(2f)（累加缓冲按声明位宽饱和）
    softmax P / 分母 l      2^(2f)
    输出累加 O = P·V        2^(2f)（P 先降到 2^f 再乘）
    最终输出 out = O / l    2^f
所有中间量均为 Python 整数，同一输入与配置下多次运行逐元素一致。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import OperatorConfig
from .errors import Code
from .fixedpoint import fp_exp, inv_sqrt_fp, saturate

STAGE_MATMUL = "matmul"
STAGE_SOFTMAX = "softmax"


@dataclass
class OverflowEvent:
    stage: str
    query_block: int
    key_block: int
    row: int
    col: int
    dim: int

    def to_dict(self):
        return {
            "code": Code.E_ACCUMULATOR_OVERFLOW,
            "stage": self.stage,
            "query_block": self.query_block,
            "key_block": self.key_block,
            "row": self.row,
            "col": self.col,
            "dim": self.dim,
        }


@dataclass
class ScoreBlock:
    """一个 (查询块, 键块) 打分乘积，整数域，标尺 2^(2f)。"""
    query_block: int
    key_block: int
    scores: list  # list[list[int]]
    overflows: list = field(default_factory=list)


def compute_score_block(q_blk: np.ndarray, k_blk: np.ndarray,
                        cfg: OperatorConfig, query_block: int,
                        key_block: int) -> ScoreBlock:
    """分块乘加：按键维度升序在声明位宽的整数缓冲中累加，溢出前饱和并记录。"""
    frac = cfg.fixed_point_frac_bits
    acc_bits = cfg.accumulator_int_bits
    inv_sqrt = inv_sqrt_fp(cfg.head_dim, frac)
    offset_fp2 = int(round(cfg.stability_offset * (1 << (2 * frac))))
    qb, dim = q_blk.shape
    kb = k_blk.shape[0]
    scores = []
    overflows = []
    for a in range(qb):
        row = []
        for b in range(kb):
            acc = 0
            for d in range(dim):  # 键维度升序累加
                acc += int(q_blk[a, d]) * int(k_blk[b, d])
                acc, ovf = saturate(acc, acc_bits)
                if ovf:
                    overflows.append(OverflowEvent(
                        STAGE_MATMUL, query_block, key_block, a, b, d))
            s = (acc * inv_sqrt) >> frac  # 乘 1/sqrt(d)，标尺保持 2^(2f)
            s -= offset_fp2               # 数值稳定偏移
            row.append(s)
        scores.append(row)
    return ScoreBlock(query_block, key_block, scores, overflows)


@dataclass
class SoftmaxState:
    """一个键块归约完成后的在线 softmax 运行状态。"""
    key_block: int
    m_fp: int          # 运行最大值，标尺 2^(2f)
    l_fp: list         # 运行分母（逐行），标尺 2^(2f)
    out_acc: list      # 输出累加 O，list[list[int]]，标尺 2^(2f)


@dataclass
class QueryBlockTrace:
    query_block: int
    states: list       # list[SoftmaxState]，按键块顺序
    out_fp: list       # 最终输出，list[list[int]]，标尺 2^f
    overflows: list = field(default_factory=list)


def online_softmax(score_blocks: list, v_blocks: list,
                   cfg: OperatorConfig, query_block: int) -> QueryBlockTrace:
    """在线 softmax 归约：以数值稳定偏移逐块更新运行最大值与分母。"""
    frac = cfg.fixed_point_frac_bits
    frac2 = 2 * frac
    acc_bits = cfg.accumulator_int_bits
    dim = cfg.head_dim
    qb = len(score_blocks[0].scores)
    m = None
    l = [0] * qb
    out_acc = [[0] * dim for _ in range(qb)]
    states = []
    overflows = []
    for key_block, (sblk, v_blk) in enumerate(zip(score_blocks, v_blocks)):
        kb = len(sblk.scores[0])
        m_blk = max(max(row) for row in sblk.scores)
        m_new = m_blk if m is None else max(m, m_blk)
        alpha = 0 if m is None else fp_exp(m - m_new, frac2)
        # 旧分母与旧输出按 alpha  rescale
        for a in range(qb):
            l[a] = (l[a] * alpha) >> frac2
            for d in range(dim):
                out_acc[a][d] = (out_acc[a][d] * alpha) >> frac2
        # 新块贡献
        for a in range(qb):
            p_f = []
            for b in range(kb):
                p = fp_exp(sblk.scores[a][b] - m_new, frac2)  # 标尺 2^(2f)
                l[a] += p
                p_f.append(p >> frac)                          # 降到 2^f 再乘 V
            for d in range(dim):
                acc = out_acc[a][d]
                for b in range(kb):
                    acc += p_f[b] * int(v_blk[b, d])
                    acc, ovf = saturate(acc, acc_bits)
                    if ovf:
                        overflows.append(OverflowEvent(
                            STAGE_SOFTMAX, query_block, key_block, a, b, d))
                out_acc[a][d] = acc
        states.append(SoftmaxState(key_block, m_new, list(l),
                                   [row[:] for row in out_acc]))
        m = m_new
    out_fp = [[(out_acc[a][d] << frac) // l[a] for d in range(dim)]
              for a in range(qb)]
    return QueryBlockTrace(query_block, states, out_fp, overflows)
