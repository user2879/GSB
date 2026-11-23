"""float64 参考路径：与整数主路径相同的分块结构，逐块记录三个环节的中间量。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RefKeyBlockState:
    key_block: int
    scores: np.ndarray      # 块内打分（含稳定偏移），float64
    m: np.ndarray           # 更新后的运行最大值（每行）
    denom: np.ndarray       # 更新后的分母（每行）
    out_acc: np.ndarray     # 更新后的输出累加


@dataclass
class RefQueryBlockTrace:
    query_block: int
    states: list = field(default_factory=list)  # list[RefKeyBlockState]
    out: np.ndarray = None                       # 最终输出，float64


def reference_query_block(q_blk: np.ndarray, k_blocks: list, v_blocks: list,
                          stability_offset: float,
                          query_block: int) -> RefQueryBlockTrace:
    head_dim = q_blk.shape[1]
    m = None
    denom = np.zeros(q_blk.shape[0], dtype=np.float64)
    out_acc = np.zeros((q_blk.shape[0], head_dim), dtype=np.float64)
    trace = RefQueryBlockTrace(query_block)
    for key_block, (k_blk, v_blk) in enumerate(zip(k_blocks, v_blocks)):
        scores = q_blk @ k_blk.T / np.sqrt(head_dim) - stability_offset
        # 与整数内核一致：运行最大值为查询块级标量
        m_blk = float(scores.max())
        m_new = m_blk if m is None else max(m, m_blk)
        alpha = 0.0 if m is None else float(np.exp(m - m_new))
        p = np.exp(scores - m_new)
        denom = denom * alpha + p.sum(axis=1)
        out_acc = out_acc * alpha + p @ v_blk
        trace.states.append(RefKeyBlockState(
            key_block, scores, m_new, denom.copy(), out_acc.copy()))
        m = m_new
    trace.out = out_acc / denom[:, None]
    return trace


def split_blocks(arr: np.ndarray, block_len: int) -> list:
    return [arr[i:i + block_len] for i in range(0, arr.shape[0], block_len)]


def run_reference(q: np.ndarray, k: np.ndarray, v: np.ndarray,
                  query_block_len: int, key_block_len: int,
                  stability_offset: float) -> list:
    """返回每个查询块的 RefQueryBlockTrace。"""
    k_blocks = split_blocks(k, key_block_len)
    v_blocks = split_blocks(v, key_block_len)
    traces = []
    for qi, q_blk in enumerate(split_blocks(q, query_block_len)):
        traces.append(reference_query_block(
            q_blk.astype(np.float64), [b.astype(np.float64) for b in k_blocks],
            [b.astype(np.float64) for b in v_blocks], stability_offset, qi))
    return traces
