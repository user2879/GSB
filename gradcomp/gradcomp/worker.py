"""工作进程：本地梯度生成、量化压缩、环形通信与一步参数更新。

每个 rank 是独立进程（TCP 传输，multiprocessing spawn）或独立线程
（loopback 传输，主要用于测试与无头演示）。训练步内的数值逻辑对两种
传输完全一致，因此结果与传输方式、块大小无关。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import protocol as proto
from .config import Config
from .data import local_gradient
from .digest import array_digest
from .quantize import Quantizer
from .ring import RingCounters, RingEngine
from .transport import LoopbackTransport, TcpEndpoint


@dataclass
class WorkerStepResult:
    rank: int
    step: int
    global_gradient: np.ndarray
    updated_parameters: np.ndarray
    quantized: list[dict[str, Any]]      # 每个分片的量化摘要
    counters: RingCounters
    local_gradient_digest: str
    shard_digests: list[str]

    def to_payload(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "step": self.step,
            "global_gradient": self.global_gradient,
            "updated_parameters": self.updated_parameters,
            "quantized": self.quantized,
            "counters": self.counters.to_dict(),
            "local_gradient_digest": self.local_gradient_digest,
            "shard_digests": list(self.shard_digests),
            "global_gradient_digest": array_digest(self.global_gradient),
            "parameter_digest": array_digest(self.updated_parameters),
        }


def _quantize_chunks(cfg: Config, gradient: np.ndarray
                     ) -> tuple[list[np.ndarray], list[dict[str, Any]], np.dtype]:
    quantizer = Quantizer(cfg.scale_exponent, cfg.truncation)
    size = cfg.chunk_size
    chunks_q: list[np.ndarray] = []
    summaries: list[dict[str, Any]] = []
    common_dtype: np.dtype | None = None
    for c in range(cfg.workers):
        piece = gradient[c * size:(c + 1) * size]
        q, dtype = quantizer.quantize(piece)
        if common_dtype is None:
            common_dtype = dtype
        elif dtype != common_dtype:
            # 环上累加要求 dtype 一致；统一提升到最宽的整数类型。
            common_dtype = max((common_dtype, dtype),
                               key=lambda d: np.iinfo(d).bits)
        rt = quantizer.roundtrip(piece)
        summaries.append({
            "chunk": c,
            "quantized_dtype": rt["quantized_dtype"],
            "observed_range": rt["observed_range"],
            "single_value_bound": rt["single_value_bound"],
            "measured_max_abs_error": rt["measured_max_abs_error"],
            "bound_respected": rt["bound_respected"],
            "digest_before": array_digest(piece),
        })
        chunks_q.append(q)
    chunks_q = [q.astype(common_dtype) for q in chunks_q]
    for summary in summaries:
        summary["ring_dtype"] = str(common_dtype)
    return chunks_q, summaries, common_dtype


def run_rank(cfg: Config, rank: int, step: int, parameters: np.ndarray,
             transport) -> WorkerStepResult:
    """执行单个 rank 的一个训练步（不含传输的建立与关闭）。"""
    n = cfg.workers
    gradient = local_gradient(cfg, rank, step)
    counters = RingCounters()
    engine = RingEngine(rank, n, cfg.block_size, transport, counters)
    quantizer = Quantizer(cfg.scale_exponent, cfg.truncation)

    if cfg.compressed:
        chunks, quant_summaries, dtype = _quantize_chunks(cfg, gradient)
        owned = engine.reduce_scatter(chunks, step)
        gathered_int = engine.all_gather(owned, step)
        global_gradient = quantizer.dequantize(gathered_int)
    else:
        size = cfg.chunk_size
        chunks = [gradient[c * size:(c + 1) * size].copy()
                  for c in range(n)]
        owned = engine.reduce_scatter(chunks, step)
        global_gradient = engine.all_gather(owned, step)
        quant_summaries = []

    engine.barrier(step)
    updated = parameters - cfg.learning_rate * global_gradient

    return WorkerStepResult(
        rank=rank, step=step, global_gradient=global_gradient,
        updated_parameters=updated, quantized=quant_summaries,
        counters=counters, local_gradient_digest=array_digest(gradient),
        shard_digests=[array_digest(c) for c in (
            chunks if cfg.compressed else chunks)])


def tcp_worker_entry(cfg_dict: dict[str, Any], rank: int, step: int,
                     parameters: np.ndarray, result_queue) -> None:
    """multiprocessing spawn 入口：建立 TCP 端点，跑一步，回传结果。"""
    cfg = Config(**cfg_dict)
    endpoint = TcpEndpoint(rank, cfg)
    try:
        endpoint.setup()
        result = run_rank(cfg, rank, step, parameters, endpoint)
        result_queue.put(result.to_payload())
        # 等待协调进程通过独立控制连接发来 FINALIZE 后优雅退出。
        try:
            control, _ = endpoint.accept_control()
            ack = proto.recv_message(control)
            control.close()
        except OSError:
            ack = None
        if ack is None or ack.msg_type != proto.MSG_FINALIZE:
            raise RuntimeError("未收到 FINALIZE 确认")
    except Exception as exc:  # noqa: BLE001 - 回传给父进程统一分类
        result_queue.put({"__error__": repr(exc),
                          "rank": rank})
    finally:
        endpoint.close()


def loopback_worker(cfg: Config, rank: int, step: int,
                    parameters: np.ndarray,
                    broker: LoopbackTransport) -> WorkerStepResult:
    """进程内（线程）执行一个 rank 的一步训练。"""
    return run_rank(cfg, rank, step, parameters, broker)
