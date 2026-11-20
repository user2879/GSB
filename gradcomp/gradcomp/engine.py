"""训练会话编排：驱动 4 个工作进程执行多步训练并集中分析。

TCP 模式使用 multiprocessing spawn 启动独立工作进程，各自监听唯一回环
端口；loopback 模式在当前进程内以线程跑相同的 rank 逻辑（用于测试与
无头环境）。两种模式产生逐位一致的结果。
"""
from __future__ import annotations

import multiprocessing as mp
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .config import Config
from .data import (all_local_gradients, data_summary, gradient_shards,
                      initial_parameters)
from .digest import array_digest, bytewise_equal, first_diff_index
from .errors import E_DIVERGE, E_INTERNAL, GradCompError
from .reference import analyze_step
from .snapshot import write_snapshot
from .transport import LoopbackTransport
from .worker import loopback_worker, tcp_worker_entry


@dataclass
class StepRecord:
    step: int
    analysis: dict[str, Any]
    parameters: np.ndarray
    parameter_version: int
    worker_payloads: list[dict[str, Any]]


@dataclass
class SessionReport:
    config: dict[str, Any]
    data_summary: dict[str, Any]
    steps: list[dict[str, Any]]
    final_parameters_digest: str
    final_parameter_version: int
    snapshot_files: list[str] = field(default_factory=list)


def _run_tcp_step(cfg: Config, step: int,
                  parameters: np.ndarray) -> list[dict[str, Any]]:
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=tcp_worker_entry,
                    args=(cfg.to_dict(), rank, step, parameters, queue),
                    name=f"gc-worker-{rank}")
        for rank in range(cfg.workers)
    ]
    for proc in procs:
        proc.start()
    payloads: list[dict[str, Any] | None] = [None] * cfg.workers
    for _ in range(cfg.workers):
        item = queue.get()
        if "__error__" in item:
            for proc in procs:
                proc.terminate()
            raise GradCompError(E_INTERNAL,
                                f"工作进程失败: {item['__error__']}",
                                details=item)
        payloads[item["rank"]] = item
    _send_finalize(cfg)
    for proc in procs:
        proc.join(timeout=30)
        if proc.exitcode not in (0, None):
            raise GradCompError(E_INTERNAL,
                                f"工作进程异常退出 code={proc.exitcode}")
    return payloads  # type: ignore[return-value]


def _send_finalize(cfg: Config) -> None:
    import socket as _socket
    from . import protocol as _proto
    for rank in range(cfg.workers):
        port = cfg.base_port + rank
        last = None
        for _ in range(50):
            try:
                with _socket.create_connection(("127.0.0.1", port),
                                                timeout=5.0) as ctrl:
                    msg = _proto.Message(
                        msg_type=_proto.MSG_FINALIZE,
                        phase=_proto.PHASE_FINALIZE, step=0,
                        src=-1 & 0xFFFF, dst=rank, chunk=0, round_index=0,
                        block=0, payload=np.zeros(0, dtype=np.int64))
                    _proto.send_message(ctrl, msg)
                break
            except OSError as exc:  # worker 可能尚未进入 accept_control
                last = exc
                import time as _time
                _time.sleep(0.05)
        else:
            raise GradCompError(E_INTERNAL,
                                f"FINALIZE 无法送达 rank {rank}: {last}")


def _run_loopback_step(cfg: Config, step: int,
                       parameters: np.ndarray) -> list[dict[str, Any]]:
    broker = LoopbackTransport(cfg.workers)
    payloads: list[dict[str, Any]] = [None] * cfg.workers  # type: ignore[list-item]

    def _target(rank: int) -> None:
        result = loopback_worker(cfg, rank, step, parameters, broker)
        payloads[rank] = result.to_payload()

    threads = [threading.Thread(target=_target, args=(rank,),
                                name=f"gc-loop-{rank}")
               for rank in range(cfg.workers)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    broker.close()
    if any(item is None for item in payloads):
        raise GradCompError(E_INTERNAL, "loopback 工作线程未返回结果")
    return payloads


def run_step(cfg: Config, step: int,
             parameters: np.ndarray) -> tuple[list[dict[str, Any]], np.ndarray]:
    if cfg.transport == "tcp":
        payloads = _run_tcp_step(cfg, step, parameters)
    else:
        payloads = _run_loopback_step(cfg, step, parameters)

    worker_globals = [item["global_gradient"] for item in payloads]
    # 跨进程一致性：所有 rank 的全局梯度必须逐字节一致。
    for rank, grad in enumerate(worker_globals[1:], start=1):
        if not bytewise_equal(worker_globals[0], grad):
            idx = first_diff_index(worker_globals[0], grad)
            raise GradCompError(
                E_DIVERGE,
                "各工作进程全局梯度不一致",
                field=f"worker[{rank}]",
                details={"first_diff_index": idx,
                         "digests": [p["global_gradient_digest"]
                                     for p in payloads]})
    updated = payloads[0]["updated_parameters"]
    return payloads, updated


def train(cfg: Config, workdir: str | Path | None = None,
          start_step: int = 0,
          initial_params: np.ndarray | None = None,
          snapshot_every_step: bool = True,
          on_step=None) -> SessionReport:
    """执行 cfg.steps 个训练步，可选在每步后原子写快照。"""
    parameters = (initial_params.copy() if initial_params is not None
                  else initial_parameters(cfg))
    snapshot_files: list[str] = []
    records: list[dict[str, Any]] = []

    for offset in range(cfg.steps):
        step = start_step + offset
        if workdir is not None and snapshot_every_step:
            # 先写“待决”快照（与本步完成后的版本号同名），
            # 强制终止时残留为未完成训练步标记，恢复时要求重跑。
            pending = write_snapshot(
                workdir, cfg, parameters, completed_steps=step + 1,
                shard_digests=[], global_gradient_digest="",
                complete=False)
        payloads, updated = run_step(cfg, step, parameters)
        analysis = analyze_step(cfg, step,
                                [p["global_gradient"] for p in payloads],
                                parameters)
        if not analysis.consistent:
            raise GradCompError(E_DIVERGE, "全局梯度一致性校验失败",
                                details=analysis.to_dict())

        grads = all_local_gradients(cfg, step)
        shards = gradient_shards(grads, cfg)
        shard_digests = [[array_digest(shards[r][c])
                          for c in range(cfg.workers)]
                         for r in range(cfg.workers)]
        parameters = updated
        version = step + 1
        if workdir is not None and snapshot_every_step:
            path = write_snapshot(
                workdir, cfg, parameters, completed_steps=version,
                shard_digests=shard_digests,
                global_gradient_digest=array_digest(analysis.global_gradient),
                complete=True)
            # 完整快照落盘后删除同版本的待决标记（二者文件名不同）。
            if pending.exists():
                pending.unlink()
            snapshot_files.append(str(path))

        record = {
            **analysis.to_dict(),
            "parameter_version": version,
            "parameters_digest": array_digest(parameters),
            "workers": [_worker_view(p) for p in payloads],
        }
        records.append(record)
        if on_step is not None:
            on_step(record)

    return SessionReport(
        config=cfg.summary(), data_summary=data_summary(cfg),
        steps=records, final_parameters_digest=array_digest(parameters),
        final_parameter_version=start_step + cfg.steps,
        snapshot_files=snapshot_files)


def _worker_view(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": payload["rank"],
        "counters": payload["counters"],
        "local_gradient_digest": payload["local_gradient_digest"],
        "global_gradient_digest": payload["global_gradient_digest"],
        "parameter_digest": payload["parameter_digest"],
        "quantized": payload["quantized"],
    }
