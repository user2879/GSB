"""单进程参考计算：真值梯度、一致性校验、误差界与误差传播归因。

真值来源（ground truth）始终是在单进程内按固定累加顺序
rank 0 -> rank 1 -> ... -> rank n-1 重算的参考梯度，不经过网络。
所有误差结论都与工作进程数量和块大小无关，同一输入重复执行输出相同。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import Config
from .data import all_local_gradients
from .digest import array_digest, bytewise_equal, first_diff_index
from .quantize import (Quantizer, combined_elements_bound,
                       vector_relative_bound)


def canonical_fp_global(cfg: Config, step: int) -> np.ndarray:
    """固定累加顺序（rank 升序）的全精度参考全局梯度。"""
    grads = all_local_gradients(cfg, step)
    total = np.zeros(cfg.dimension, dtype=np.float64)
    for rank in range(cfg.workers):
        total = total + grads[rank]
    return total


def ring_order_fp_global(cfg: Config, step: int) -> np.ndarray:
    """模拟环形路径的浮点累加顺序（每个分片按轮转顺序求和）。

    与参考梯度的差异即“累加顺序误差”。
    """
    grads = all_local_gradients(cfg, step)
    size = cfg.chunk_size
    out = np.zeros(cfg.dimension, dtype=np.float64)
    for chunk in range(cfg.workers):
        start = chunk * size
        # 环上该分片的首持者为 chunk-1，累加顺序随之轮转。
        first_holder = (chunk - 1) % cfg.workers
        acc = np.zeros(size, dtype=np.float64)
        for k in range(cfg.workers):
            rank = (first_holder + k) % cfg.workers
            acc = acc + grads[rank, start:start + size]
        out[start:start + size] = acc
    return out


def compressed_global(cfg: Config, step: int) -> np.ndarray:
    """压缩路径的单进程重算（整数域精确求和后解压），与环结果一致。"""
    grads = all_local_gradients(cfg, step)
    quantizer = Quantizer(cfg.scale_exponent, cfg.truncation)
    total = np.zeros(cfg.dimension, dtype=np.int64)
    for rank in range(cfg.workers):
        q, _ = quantizer.quantize(grads[rank])
        total = total + q.astype(np.int64)
    return quantizer.dequantize(total)


@dataclass
class StepAnalysis:
    step: int
    compressed: bool
    global_gradient: np.ndarray
    reference_gradient: np.ndarray
    consistent: bool
    first_diff: int | None
    worker_digests: list[str]
    reference_digest: str
    error_bounds: dict[str, Any]
    attribution: dict[str, Any]
    parameter_deviation: np.ndarray
    quantizer_info: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "compressed": self.compressed,
            "consistent": self.consistent,
            "first_diff_index": self.first_diff,
            "worker_global_digests": list(self.worker_digests),
            "reference_digest": self.reference_digest,
            "error_bounds": self.error_bounds,
            "attribution": self.attribution,
            "parameter_deviation_norm": float(
                np.linalg.norm(self.parameter_deviation)),
            "parameter_deviation_digest": array_digest(self.parameter_deviation),
            "quantizer": self.quantizer_info,
        }


def _error_bounds(cfg: Config, grads: np.ndarray,
                  total_trunc_error: np.ndarray) -> dict[str, Any]:
    quantizer = Quantizer(cfg.scale_exponent, cfg.truncation)
    scale = quantizer.scale
    flat = grads.reshape(-1)
    grad_norm = float(np.linalg.norm(flat))
    vec = vector_relative_bound(scale, flat.size, grad_norm, cfg.truncation)
    # n 个工作进程在同一元素上各自量化后合成的上界（合成 n 份单值误差）。
    combined = combined_elements_bound(scale, cfg.workers, cfg.truncation)
    observed = {
        "min": float(flat.min()), "max": float(flat.max()),
        "max_abs": float(np.abs(flat).max()),
    }
    measured_vec = float(np.linalg.norm(total_trunc_error))
    return {
        "depends_on": {
            "scale": scale, "scale_exponent": cfg.scale_exponent,
            "truncation": cfg.truncation,
            "observed_gradient_range": observed,
            "workers": cfg.workers, "dimension": cfg.dimension,
        },
        "single_value_bound": quantizer.single_value_bound(),
        "vector_relative_bound": vec,
        "combined_workers_bound": combined,
        "measured_truncation_error_2norm": measured_vec,
        "vector_bound_respected": bool(
            measured_vec <=
            vec["vector_abs_bound_2norm"] * cfg.workers * (1 + 1e-9) + 1e-12),
    }


def analyze_step(cfg: Config, step: int,
                 worker_globals: list[np.ndarray],
                 parameters: np.ndarray) -> StepAnalysis:
    """汇总各工作进程结果并给出一致性、误差界与归因结论。"""
    reference = canonical_fp_global(cfg, step)
    worker_digests = [array_digest(g) for g in worker_globals]
    reference_digest = array_digest(reference)

    consistent = all(bytewise_equal(g, worker_globals[0])
                     for g in worker_globals)
    first_diff = None
    if not consistent:
        for rank, g in enumerate(worker_globals[1:], start=1):
            idx = first_diff_index(worker_globals[0], g)
            if idx is not None:
                first_diff = idx
                break

    if cfg.compressed:
        quantizer = Quantizer(cfg.scale_exponent, cfg.truncation)
        grads = all_local_gradients(cfg, step)
        # 压缩截断误差：压缩路径结果与“同累加顺序浮点结果”的差异。
        ring_fp = ring_order_fp_global(cfg, step)
        comp = compressed_global(cfg, step)
        truncation_error = comp - ring_fp
        # 累加顺序误差：环形浮点顺序与规范固定顺序的差异。
        order_error = ring_fp - reference
        quantizer_info = {
            "scale": quantizer.scale,
            "scale_exponent": quantizer.scale_exponent,
            "truncation": quantizer.truncation,
            "single_value_bound": quantizer.single_value_bound(),
        }
        error_bounds = _error_bounds(cfg, grads, truncation_error)
    else:
        truncation_error = np.zeros(cfg.dimension, dtype=np.float64)
        order_error = np.zeros(cfg.dimension, dtype=np.float64)
        quantizer_info = {"compressed": False}
        error_bounds = {"note": "未压缩训练步，无量化误差"}

    total_error = truncation_error + order_error
    attribution = _attribute(cfg, truncation_error, order_error)
    parameter_deviation = -cfg.learning_rate * total_error

    return StepAnalysis(
        step=step, compressed=cfg.compressed,
        global_gradient=worker_globals[0], reference_gradient=reference,
        consistent=consistent, first_diff=first_diff,
        worker_digests=worker_digests, reference_digest=reference_digest,
        error_bounds=error_bounds, attribution=attribution,
        parameter_deviation=parameter_deviation,
        quantizer_info=quantizer_info)


def _attribute(cfg: Config, truncation_error: np.ndarray,
               order_error: np.ndarray) -> dict[str, Any]:
    """逐元素区分两类误差来源，按贡献绝对值排序输出主导来源与占比。"""
    trunc_abs = np.abs(truncation_error)
    order_abs = np.abs(order_error)
    total_abs = trunc_abs + order_abs
    total_energy = float(total_abs.sum())
    trunc_share = float(trunc_abs.sum() / total_energy) if total_energy > 0 else 0.0
    order_share = float(order_abs.sum() / total_energy) if total_energy > 0 else 0.0

    dominant = np.where(trunc_abs >= order_abs, "compression_truncation",
                        "accumulation_order")
    order = np.argsort(-total_abs, kind="stable")
    top_k = min(8, cfg.dimension)
    top = [{
        "element": int(idx),
        "dominant_source": str(dominant[idx]),
        "truncation_error": float(truncation_error[idx]),
        "order_error": float(order_error[idx]),
        "abs_contribution": float(total_abs[idx]),
        "share_of_total": (float(total_abs[idx] / total_energy)
                           if total_energy > 0 else 0.0),
    } for idx in order[:top_k]]
    counts = {
        "compression_truncation": int(np.count_nonzero(
            dominant == "compression_truncation")),
        "accumulation_order": int(np.count_nonzero(
            dominant == "accumulation_order")),
    }
    return {
        "global_share": {
            "compression_truncation": trunc_share,
            "accumulation_order": order_share,
        },
        "dominant_element_counts": counts,
        "top_elements": top,
        "truncation_error_2norm": float(np.linalg.norm(truncation_error)),
        "order_error_2norm": float(np.linalg.norm(order_error)),
    }
