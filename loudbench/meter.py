"""BS.1770-4 门控响度测量单元与整段参考实现。

关键不变量：400ms 积分块锚定在“整段素材起点”的全局网格上，而不是每个任务
各自的起点。因此无论窗口怎么切分，每个积分块的样本集合都唯一，门控统计在
归并后与一次性整段计算完全一致（同一组块、同一 float64 求和顺序）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .kweight import (
    CHANNEL_WEIGHTS,
    fftconvolve_chunk,
    kweight_impulse,
    to_db_tp,
    true_peak_fir,
    upsample_max,
)
from .wavio import SAMPLE_RATE

BLOCK_SECONDS = 0.4
HOP_SECONDS = 0.1
BLOCK_SAMPLES = int(round(BLOCK_SECONDS * SAMPLE_RATE))   # 19200
HOP_SAMPLES = int(round(HOP_SECONDS * SAMPLE_RATE))       # 4800
ABS_GATE_LKFS = -70.0
REL_GATE_OFFSET_LU = -10.0
LOUDNESS_EPS = 1e-12


def block_anchors(total_samples: int) -> List[int]:
    """全局 100ms 网格上的积分块起点，仅保留完整 400ms 块。"""
    anchors: List[int] = []
    if total_samples < BLOCK_SAMPLES:
        return anchors
    last = total_samples - BLOCK_SAMPLES
    for a in range(0, last + 1, HOP_SAMPLES):
        anchors.append(a)
    return anchors


def block_meansq(weighted: np.ndarray, start: int) -> float:
    seg = weighted[:, start:start + BLOCK_SAMPLES]
    w = CHANNEL_WEIGHTS.reshape(-1, 1)
    return float(np.mean(np.sum((seg * seg) * w, axis=0)))


def block_lufs(mean_sq: float) -> float:
    return -0.691 + 10.0 * math.log10(mean_sq + LOUDNESS_EPS)


def gated_lufs(block_ms: Sequence[float]) -> Tuple[Optional[float], Optional[float], int, int]:
    """两级门控，返回 (绝对门控响度, 相对门控响度, 绝对块数, 相对块数)。"""
    if not block_ms:
        return None, None, 0, 0
    abs_kept = [m for m in block_ms if block_lufs(m) > ABS_GATE_LKFS]
    abs_count = len(abs_kept)
    if not abs_kept:
        return None, None, 0, 0
    abs_lufs = -0.691 + 10.0 * math.log10(sum(abs_kept) / abs_count + LOUDNESS_EPS)
    rel_threshold = abs_lufs + REL_GATE_OFFSET_LU
    rel_kept = [m for m in abs_kept if block_lufs(m) > rel_threshold]
    rel_count = len(rel_kept)
    rel_lufs = -0.691 + 10.0 * math.log10(sum(rel_kept) / rel_count + LOUDNESS_EPS)
    return abs_lufs, rel_lufs, abs_count, rel_count


def percentile_lra(block_lufs_values: Sequence[float]) -> Optional[float]:
    """简化 EBU R128 LRA：门控后块响度的 P95-P10（LU≈LKFS 跨度）。"""
    vals = [v for v in block_lufs_values if v > ABS_GATE_LKFS]
    if len(vals) < 2:
        return None
    arr = np.sort(np.asarray(vals, dtype=np.float64))
    p10 = float(np.percentile(arr, 10))
    p95 = float(np.percentile(arr, 95))
    return p95 - p10


@dataclass
class TaskResult:
    task_key: str
    index: int
    start: int
    end: int
    short_lufs: Optional[float]
    gated_abs_lufs: Optional[float]
    gated_rel_lufs: Optional[float]
    abs_block_count: int
    rel_block_count: int
    true_peak_db: float
    true_peak_channel: int
    true_peak_sample: int
    lra: Optional[float]
    block_count: int
    block_ms_sum: float
    block_ms: List[float] = field(default_factory=list)
    cache_hits: int = 0
    cache_misses: int = 0
    recomputes: int = 0
    uncached: int = 0
    ledger_events: List[Dict] = field(default_factory=list)
    error: Optional[Dict] = None


def process_task(
    task,
    raw_context: np.ndarray,
    kw_impulse: np.ndarray,
    tp_fir: np.ndarray,
) -> TaskResult:
    """对单个测量任务做 K 加权、积分块统计与真峰值。

    raw_context 覆盖 [start - warmup, end)，前 warmup 样本仅用于重建滤波状态。
    """
    assert raw_context.shape[1] == task.context_end - task.context_start

    weighted_ctx = np.empty_like(raw_context)
    for c in range(raw_context.shape[0]):
        weighted_ctx[c] = fftconvolve_chunk(raw_context[c], kw_impulse)
    weighted = weighted_ctx  # 保留前置 warm-up，供锚点早于任务起点的跨边界块使用

    meansq: List[float] = []
    lufs_values: List[float] = []
    for anchor in task.block_anchors:
        local = anchor - task.context_start
        m = block_meansq(weighted, local)
        meansq.append(m)
        lufs_values.append(block_lufs(m))

    if meansq:
        short_lufs = block_lufs(float(np.mean(np.asarray(meansq))))
    else:
        short_lufs = None
    abs_lufs, rel_lufs, abs_count, rel_count = gated_lufs(meansq)
    lra = percentile_lra(lufs_values)

    peak_db = -math.inf
    peak_ch = -1
    peak_idx = -1
    peak_lo = task.start - task.context_start
    peak_hi = task.end - task.context_start
    for c in range(raw_context.shape[0]):
        p, idx = upsample_max(raw_context[c], tp_fir, out_start=peak_lo, out_end=peak_hi)
        db = to_db_tp(p)
        if db > peak_db:
            peak_db, peak_ch, peak_idx = db, c, idx

    return TaskResult(
        task_key=task.task_key,
        index=task.index,
        start=task.start,
        end=task.end,
        short_lufs=short_lufs,
        gated_abs_lufs=abs_lufs,
        gated_rel_lufs=rel_lufs,
        abs_block_count=abs_count,
        rel_block_count=rel_count,
        true_peak_db=peak_db,
        true_peak_channel=peak_ch,
        true_peak_sample=peak_idx,
        lra=lra,
        block_count=len(meansq),
        block_ms_sum=float(np.sum(np.asarray(meansq))) if meansq else 0.0,
        block_ms=meansq,
    )


def measure_whole_program(samples: np.ndarray) -> Dict:
    """整段一次算完的参考实现（用于验收与切分一致性比对）。"""
    kw = kweight_impulse()
    fir = true_peak_fir()
    weighted = np.empty_like(samples)
    for c in range(samples.shape[0]):
        weighted[c] = fftconvolve_chunk(samples[c], kw)

    anchors = block_anchors(samples.shape[1])
    ms = [block_meansq(weighted, a) for a in anchors]
    lufs_values = [block_lufs(m) for m in ms]
    abs_lufs, rel_lufs, abs_count, rel_count = gated_lufs(ms)
    lra = percentile_lra(lufs_values)

    peak_db = -math.inf
    peak_ch = -1
    peak_idx = -1
    for c in range(samples.shape[0]):
        p, idx = upsample_max(samples[c], fir)
        db = to_db_tp(p)
        if db > peak_db:
            peak_db, peak_ch, peak_idx = db, c, idx

    return {
        "samples": int(samples.shape[1]),
        "block_count": len(ms),
        "abs_gated_lufs": abs_lufs,
        "rel_gated_lufs": rel_lufs,
        "abs_block_count": abs_count,
        "rel_block_count": rel_count,
        "lra_lu": lra,
        "true_peak_db": peak_db,
        "true_peak_channel": peak_ch,
        "true_peak_sample": peak_idx,
        "block_ms_sum": float(np.sum(np.asarray(ms))) if ms else 0.0,
        "block_ms_mean": float(np.mean(np.asarray(ms))) if ms else None,
    }
