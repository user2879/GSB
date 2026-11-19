"""确定性分块并发渲染、内存上限与中断处理（业务要求 6、8）。

调度模型
--------
* 单派发线程严格按分块索引序把任务放入有界队列；
* 有界线程池中的工作线程取块渲染，队列容量同时受并发上限与内存预算约束；
* 因此“在途采样缓冲总量超上限时按分块索引序等待”由派发线程天然实现，
  任何情况下都不会通过降低步长或精度来迁就内存；
* 任一分块异常即取消所有待处理分块并向上抛出首个错误；
* 中断信号只在分块边界（派发前 / 取块后开工前）生效。
"""
from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from .errors import E_INTERRUPTED, E_MEMORY, E_TILE, RenderError
from .renderer import TileResult, render_tile, tile_ray_memory
from .scene import Scene

INTERRUPT_FILENAME = "INTERRUPT"
_SENTINEL = object()


@dataclass(frozen=True)
class TileSpec:
    index: int
    x0: int
    y0: int
    width: int
    height: int


def build_tile_specs(scene: Scene) -> list[TileSpec]:
    """按行序生成完整覆盖超采样网格的分块清单。"""
    cfg = scene.config
    grid_w = cfg.width * cfg.supersample
    grid_h = cfg.height * cfg.supersample
    specs: list[TileSpec] = []
    idx = 0
    for y0 in range(0, grid_h, cfg.tile_height):
        for x0 in range(0, grid_w, cfg.tile_width):
            specs.append(TileSpec(idx, x0, y0, cfg.tile_width, cfg.tile_height))
            idx += 1
    return specs


class TileScheduler:
    def __init__(self, scene: Scene, interrupted: threading.Event | None = None):
        self.scene = scene
        self.interrupted = interrupted or threading.Event()
        self.specs = build_tile_specs(scene)
        cfg = scene.config
        biggest = max(tile_ray_memory(s.width, s.height, cfg.max_steps) for s in self.specs)
        if biggest > cfg.memory_limit_bytes:
            raise RenderError(
                E_MEMORY, "单个分块的在途采样缓冲需求即超过内存上限",
                {"tile_need_bytes": biggest, "limit_bytes": cfg.memory_limit_bytes},
            )
        # 队列容量：在不超内存预算的前提下允许的最大在途块数，且不超过并发上限。
        capacity = 0
        accounted = 0
        for spec in sorted(self.specs, key=lambda s: s.index):
            need = tile_ray_memory(spec.width, spec.height, cfg.max_steps)
            if accounted + need > cfg.memory_limit_bytes:
                break
            accounted += need
            capacity += 1
        self._queue_capacity = max(1, min(capacity, cfg.concurrency))

    def run(self, skip: set[int] | None = None) -> list[TileResult]:
        skip = skip or set()
        pending = [s for s in self.specs if s.index not in skip]
        if not pending:
            return []

        cfg = self.scene.config
        work_q: queue.Queue = queue.Queue(maxsize=self._queue_capacity)
        results: list[TileResult] = []
        first_error: list[BaseException] = []
        error_lock = threading.Lock()
        cancel_event = threading.Event()

        def worker() -> None:
            while True:
                item = work_q.get()
                try:
                    if item is _SENTINEL:
                        return
                    spec: TileSpec = item
                    # 分块边界检查：中断/取消均不在块中强制打断。
                    if self.interrupted.is_set() or cancel_event.is_set():
                        continue
                    try:
                        tile = render_tile(self.scene, spec.index, spec.x0,
                                          spec.y0, spec.width, spec.height)
                    except RenderError as exc:
                        with error_lock:
                            if not first_error:
                                first_error.append(exc)
                        cancel_event.set()
                        continue
                    except BaseException as exc:  # noqa: BLE001 - 需要回传首个错误
                        with error_lock:
                            if not first_error:
                                first_error.append(
                                    RenderError(E_TILE, "分块渲染未预期异常",
                                                {"type": type(exc).__name__,
                                                 "message": str(exc)}))
                        cancel_event.set()
                        continue
                    if not cancel_event.is_set():
                        results.append(tile)
                finally:
                    work_q.task_done()

        def dispatcher() -> None:
            try:
                for spec in pending:  # 已按索引序
                    if self.interrupted.is_set() or cancel_event.is_set():
                        break
                    work_q.put(spec)  # 队列满即阻塞 = 按索引序等待内存
            finally:
                for _ in range(cfg.concurrency):
                    work_q.put(_SENTINEL)

        with ThreadPoolExecutor(max_workers=cfg.concurrency) as pool:
            workers = [pool.submit(worker) for _ in range(cfg.concurrency)]
            dispatch = threading.Thread(target=dispatcher, name="tile-dispatcher")
            dispatch.start()
            dispatch.join()
            for w in workers:
                w.result()

        if first_error:
            raise first_error[0]
        # 中断发生在分块边界且没有任何块被处理时，显式上抛中断错误，
        # 避免静默返回空结果被误判为成功完成。
        if self.interrupted.is_set() and not results:
            raise RenderError(E_INTERRUPTED, "调度在分块边界被中断，未处理任何分块",
                              {"pending": len(pending)})
        return results
