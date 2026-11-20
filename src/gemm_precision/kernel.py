from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from .model import BlockShape, LoopOrder
from .reduction import MERGE_SCHEDULES, merge_partial_arrays
from .util import array_sha256
from .validation import prepare_bmm_arrays, prepare_gemm_arrays


@dataclass(frozen=True)
class BlockTask:
    key: tuple[int, int, int]
    batch_index: int | None
    block_row: int
    block_col: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "stable_task_key": list(self.key),
            "batch_index": self.batch_index,
            "block_row": self.block_row,
            "block_col": self.block_col,
            "row_range": [self.row_start, self.row_end],
            "col_range": [self.col_start, self.col_end],
        }


@dataclass(frozen=True)
class KernelResult:
    c: np.ndarray
    mode: Literal["gemm", "bmm"]
    block_shape: BlockShape
    loop_order: LoopOrder
    tasks: tuple[BlockTask, ...]
    element_metadata: tuple[dict[str, Any], ...]
    max_workers: int
    input_sha256: str

    @property
    def output_sha256(self) -> str:
        return array_sha256(self.c)

    def summary(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "shape": [int(item) for item in self.c.shape],
            "block_shape": self.block_shape.as_dict(),
            "loop_order": self.loop_order,
            "task_count": len(self.tasks),
            "stable_task_order": [list(task.key) for task in self.tasks],
            "input_sha256": self.input_sha256,
            "output_sha256": self.output_sha256,
            "element_metadata_count": len(self.element_metadata),
        }


def _chunk_ranges(k: int, bk: int) -> list[tuple[int, int]]:
    return [(start, min(start + bk, k)) for start in range(0, k, bk)]


def _accumulate_partial(
    a_tile: np.ndarray,
    b_tile: np.ndarray,
    loop_order: LoopOrder,
) -> np.ndarray:
    f32 = np.float32
    rows, k_size = a_tile.shape
    _, cols = b_tile.shape
    partial = np.zeros((rows, cols), dtype=np.float32)

    if loop_order == "ijk":
        for i in range(rows):
            for j in range(cols):
                acc = f32(0.0)
                for k in range(k_size):
                    acc = f32(acc + a_tile[i, k] * b_tile[k, j])
                partial[i, j] = acc
    elif loop_order == "ikj":
        for i in range(rows):
            for k in range(k_size):
                aval = a_tile[i, k]
                for j in range(cols):
                    partial[i, j] = f32(partial[i, j] + aval * b_tile[k, j])
    elif loop_order == "jik":
        for j in range(cols):
            for i in range(rows):
                acc = f32(0.0)
                for k in range(k_size):
                    acc = f32(acc + a_tile[i, k] * b_tile[k, j])
                partial[i, j] = acc
    else:
        raise ValueError(f"Unsupported loop order: {loop_order}")
    return partial


def _compute_tile(
    A2: np.ndarray,
    B2: np.ndarray,
    C2: np.ndarray,
    task: BlockTask,
    block_shape: BlockShape,
    loop_order: LoopOrder,
    alpha: float,
    beta: float,
) -> np.ndarray:
    i0, i1 = task.row_start, task.row_end
    j0, j1 = task.col_start, task.col_end
    k_size = A2.shape[1]

    partials: list[np.ndarray] = []
    for k0, k1 in _chunk_ranges(k_size, block_shape.bk):
        partials.append(
            _accumulate_partial(A2[i0:i1, k0:k1], B2[k0:k1, j0:j1], loop_order)
        )
    row_sum = merge_partial_arrays(partials, loop_order)

    f32 = np.float32
    alpha32 = f32(alpha)
    beta32 = f32(beta)
    result = np.empty_like(row_sum)
    beta_nonzero = float(beta) != 0.0
    if beta_nonzero:
        c_tile = C2[i0:i1, j0:j1]
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                scaled = f32(alpha32 * row_sum[i, j])
                result[i, j] = f32(scaled + f32(beta32 * c_tile[i, j]))
    else:
        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = f32(alpha32 * row_sum[i, j])
    return result


def _accumulation_order_id(
    loop_order: LoopOrder,
    block_shape: BlockShape,
    k_size: int,
) -> str:
    chunk_count = len(_chunk_ranges(k_size, block_shape.bk))
    return (
        f"order={loop_order}|K={k_size}|BK={block_shape.bk}|chunks={chunk_count}|"
        "local_chain=k-ascending|local=float32|"
        f"merge={MERGE_SCHEDULES[loop_order]}|"
        "scale=alpha-after-sum|beta=post-add"
    )


def _make_tasks(
    batch_size: int,
    M: int,
    N: int,
    block_shape: BlockShape,
    mode: Literal["gemm", "bmm"],
) -> list[BlockTask]:
    tasks: list[BlockTask] = []
    for batch_index in range(batch_size):
        public_batch_index = batch_index if mode == "bmm" else None
        for block_row, row_start in enumerate(range(0, M, block_shape.bm)):
            row_end = min(row_start + block_shape.bm, M)
            for block_col, col_start in enumerate(range(0, N, block_shape.bn)):
                col_end = min(col_start + block_shape.bn, N)
                tasks.append(
                    BlockTask(
                        key=(batch_index, block_row, block_col),
                        batch_index=public_batch_index,
                        block_row=block_row,
                        block_col=block_col,
                        row_start=row_start,
                        row_end=row_end,
                        col_start=col_start,
                        col_end=col_end,
                    )
                )
    tasks.sort(key=lambda task: task.key)
    return tasks


def _execute_tasks(
    A: np.ndarray,
    B: np.ndarray,
    C: np.ndarray,
    tasks: list[BlockTask],
    block_shape: BlockShape,
    loop_order: LoopOrder,
    alpha: float,
    beta: float,
    max_workers: int,
) -> np.ndarray:
    output = np.empty_like(C, dtype=np.float32)
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="gemm-block") as executor:
        futures: list[tuple[BlockTask, Future[np.ndarray]]] = []
        for task in tasks:
            batch_index = task.key[0]
            future = executor.submit(
                _compute_tile,
                A[batch_index],
                B[batch_index],
                C[batch_index],
                task,
                block_shape,
                loop_order,
                alpha,
                beta,
            )
            futures.append((task, future))

        tile_results: dict[tuple[int, int, int], np.ndarray] = {}
        first_error: BaseException | None = None
        for task, future in futures:
            try:
                tile_results[task.key] = future.result()
            except BaseException as exc:  # preserve the first deterministic task failure
                if first_error is None:
                    first_error = exc
        if first_error is not None:
            raise first_error

    for task in tasks:
        batch_index, _, _ = task.key
        output[
            batch_index,
            task.row_start : task.row_end,
            task.col_start : task.col_end,
        ] = tile_results[task.key]
    return output


def _build_element_metadata(
    mode: Literal["gemm", "bmm"],
    shape: tuple[int, ...],
    k_size: int,
    block_shape: BlockShape,
    loop_order: LoopOrder,
    beta_nonzero: bool,
) -> tuple[dict[str, Any], ...]:
    accumulation_id = _accumulation_order_id(loop_order, block_shape, k_size)
    chunk_count = len(_chunk_ranges(k_size, block_shape.bk))
    rounding_steps = k_size + max(chunk_count - 1, 0) + 1 + (1 if beta_nonzero else 0)
    items: list[dict[str, Any]] = []
    if mode == "gemm":
        M, N = int(shape[0]), int(shape[1])
        for row in range(M):
            for col in range(N):
                items.append(
                    {
                        "batch_index": None,
                        "row": row,
                        "col": col,
                        "stable_task_key": [0, row // block_shape.bm, col // block_shape.bn],
                        "loop_order": loop_order,
                        "accumulation_order_id": accumulation_id,
                        "chain_length": k_size,
                        "k_chunk_count": chunk_count,
                        "rounding_step_count": rounding_steps,
                    }
                )
    else:
        batch_size, M, N = int(shape[0]), int(shape[1]), int(shape[2])
        for batch_index in range(batch_size):
            for row in range(M):
                for col in range(N):
                    items.append(
                        {
                            "batch_index": batch_index,
                            "row": row,
                            "col": col,
                            "stable_task_key": [
                                batch_index,
                                row // block_shape.bm,
                                col // block_shape.bn,
                            ],
                            "loop_order": loop_order,
                            "accumulation_order_id": accumulation_id,
                            "chain_length": k_size,
                            "k_chunk_count": chunk_count,
                            "rounding_step_count": rounding_steps,
                        }
                    )
    return tuple(items)


def gemm(
    A: Any,
    B: Any,
    C: Any,
    alpha: float = 1.0,
    beta: float = 0.0,
    *,
    block_shape: BlockShape | tuple[int, int, int] = BlockShape(8, 8, 8),
    loop_order: LoopOrder = "ijk",
    max_workers: int = 1,
) -> KernelResult:
    block = BlockShape.from_any(block_shape)
    block.validate(stage="kernel_block_validation")
    if loop_order not in ("ijk", "ikj", "jik"):
        raise ValueError(f"Unsupported loop order: {loop_order}")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    A32, B32, C32, alpha_value, beta_value = prepare_gemm_arrays(A, B, C, alpha, beta)
    M, K = A32.shape
    N = B32.shape[1]
    tasks = _make_tasks(1, M, N, block, mode="gemm")
    output = _execute_tasks(
        A32[None, ...],
        B32[None, ...],
        C32[None, ...],
        tasks,
        block,
        loop_order,
        alpha_value,
        beta_value,
        max_workers,
    )[0]
    return KernelResult(
        c=output,
        mode="gemm",
        block_shape=block,
        loop_order=loop_order,
        tasks=tuple(tasks),
        element_metadata=_build_element_metadata(
            "gemm", output.shape, K, block, loop_order, beta_value != 0.0
        ),
        max_workers=max_workers,
        input_sha256=array_sha256(A32, B32, C32),
    )


def bmm(
    A: Any,
    B: Any,
    C: Any,
    alpha: float = 1.0,
    beta: float = 0.0,
    *,
    block_shape: BlockShape | tuple[int, int, int] = BlockShape(8, 8, 8),
    loop_order: LoopOrder = "ijk",
    max_workers: int = 1,
) -> KernelResult:
    block = BlockShape.from_any(block_shape)
    block.validate(stage="kernel_block_validation")
    if loop_order not in ("ijk", "ikj", "jik"):
        raise ValueError(f"Unsupported loop order: {loop_order}")
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")
    A32, B32, C32, alpha_value, beta_value = prepare_bmm_arrays(A, B, C, alpha, beta)
    batch_size, M, K = A32.shape
    N = B32.shape[2]
    tasks = _make_tasks(batch_size, M, N, block, mode="bmm")
    output = _execute_tasks(
        A32,
        B32,
        C32,
        tasks,
        block,
        loop_order,
        alpha_value,
        beta_value,
        max_workers,
    )
    return KernelResult(
        c=output,
        mode="bmm",
        block_shape=block,
        loop_order=loop_order,
        tasks=tuple(tasks),
        element_metadata=_build_element_metadata(
            "bmm", output.shape, K, block, loop_order, beta_value != 0.0
        ),
        max_workers=max_workers,
        input_sha256=array_sha256(A32, B32, C32),
    )
