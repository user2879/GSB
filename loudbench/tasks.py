"""按时间窗的确定性任务切分。

任务窗口在样本域连续且互不重叠（边界归属唯一）；任务额外声明 warm-up 重叠
上下文（滤波/过采样启动样本），这些重叠样本只用于重建滤波器状态，不计入
任何积分块的归属，因此不会被重复统计。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, List

from .errors import BlockBoundsError, WindowInvalidError
from .kweight import KW_WARMUP, TP_WARMUP
from .meter import BLOCK_SAMPLES, HOP_SAMPLES, block_anchors
from .wavio import SAMPLE_RATE

WARMUP = max(KW_WARMUP, TP_WARMUP)
DEFAULT_WINDOW_SECONDS = 3.0


@dataclass
class MeasurementTask:
    task_key: str
    index: int
    start: int
    end: int
    context_start: int
    context_end: int
    overlap_lead_samples: int
    block_anchors: List[int] = field(default_factory=list)
    buffer_declaration: Dict[str, int] = field(default_factory=dict)

    @property
    def span_samples(self) -> int:
        return self.end - self.start

    @property
    def context_samples(self) -> int:
        return self.context_end - self.context_start


def _declare_buffers(task: MeasurementTask, channels: int) -> Dict[str, int]:
    span = task.context_samples
    bytes_per = 8  # float64
    raw = channels * span * bytes_per
    weighted = channels * span * bytes_per
    integration = len(task.block_anchors) * bytes_per
    tp_over = channels * span * 4 * bytes_per
    resample = raw  # 48k 直通，重采样缓冲与原始上下文同尺寸（恒等占位）
    total = raw + weighted + integration + tp_over + resample
    return {
        "raw_context_bytes": raw,
        "resample_bytes": resample,
        "kweighted_bytes": weighted,
        "integration_bytes": integration,
        "true_peak_bytes": tp_over,
        "total_bytes": total,
    }


def validate_window_params(window_seconds: float, overlap_seconds: float) -> None:
    if not isinstance(window_seconds, (int, float)) or window_seconds <= 0:
        raise WindowInvalidError(
            "窗口时长必须为正数",
            evidence={"window_seconds": window_seconds},
        )
    if not isinstance(overlap_seconds, (int, float)) or overlap_seconds < 0:
        raise WindowInvalidError(
            "重叠时长不能为负",
            evidence={"overlap_seconds": overlap_seconds},
        )
    if overlap_seconds >= window_seconds:
        raise WindowInvalidError(
            "重叠时长必须小于窗口时长，否则窗口区间被重叠完全覆盖",
            evidence={
                "window_seconds": float(window_seconds),
                "overlap_seconds": float(overlap_seconds),
            },
        )
    window_samples = int(round(window_seconds * SAMPLE_RATE))
    overlap_samples = int(round(overlap_seconds * SAMPLE_RATE))
    if window_samples < BLOCK_SAMPLES:
        raise WindowInvalidError(
            f"窗口至少需要一个完整 {BLOCK_SAMPLES} 样本（400ms）积分块",
            evidence={"window_samples": window_samples, "block_samples": BLOCK_SAMPLES},
        )
    if overlap_samples > WARMUP:
        raise BlockBoundsError(
            "声明的窗口重叠超出可用 warm-up 区间",
            location={"warmup_samples": WARMUP},
            evidence={"overlap_samples": overlap_samples},
        )


def split_tasks(
    total_samples: int,
    channels: int,
    window_seconds: float = DEFAULT_WINDOW_SECONDS,
    overlap_seconds: float = 0.0,
) -> List[MeasurementTask]:
    validate_window_params(window_seconds, overlap_seconds)
    if total_samples < BLOCK_SAMPLES:
        raise BlockBoundsError(
            "素材短于一个完整积分块，无法切分",
            evidence={"total_samples": int(total_samples), "block_samples": BLOCK_SAMPLES},
        )

    window = int(round(window_seconds * SAMPLE_RATE))
    anchors_global = block_anchors(total_samples)

    tasks: List[MeasurementTask] = []
    start = 0
    idx = 0
    while start < total_samples:
        end = min(start + window, total_samples)
        if end - start < BLOCK_SAMPLES and tasks:
            # 末段不足一个积分块时并入上一任务，保证无样本遗漏
            tasks[-1].end = total_samples
            tasks[-1].context_end = total_samples
            prev = tasks[-1]
            prev.block_anchors = [
                a for a in anchors_global
                if prev.start < a + BLOCK_SAMPLES <= prev.end
                or (prev.index == 0 and a + BLOCK_SAMPLES <= prev.end)
            ]
            prev.buffer_declaration = _declare_buffers(prev, channels)
            break

        # 块末端归属：块 (a, a+B] 归到包含其末端的任务，跨边界块经 warm-up 上下文读取
        owned = [
            a for a in anchors_global if start < a + BLOCK_SAMPLES <= end or (idx == 0 and a + BLOCK_SAMPLES <= end)
        ]
        if owned:
            need_start = owned[0]
            need_end = owned[-1] + BLOCK_SAMPLES
        else:
            need_start, need_end = start, end
        context_start = max(0, min(start - WARMUP, need_start))
        context_end = min(total_samples, max(need_end, end) + TP_WARMUP)
        task = MeasurementTask(
            task_key=f"T{idx:04d}",
            index=idx,
            start=start,
            end=end,
            context_start=context_start,
            context_end=context_end,
            overlap_lead_samples=start - context_start,
        )
        task.block_anchors = owned
        task.buffer_declaration = _declare_buffers(task, channels)
        tasks.append(task)
        start = end
        idx += 1

    _validate_partition(tasks, total_samples)
    return tasks


def _validate_partition(tasks: List[MeasurementTask], total_samples: int) -> None:
    if not tasks:
        raise BlockBoundsError("切分结果为空", evidence={"total_samples": total_samples})
    if tasks[0].start != 0 or tasks[-1].end != total_samples:
        raise BlockBoundsError(
            "任务分区未覆盖完整素材",
            evidence={"first": tasks[0].start, "last_end": tasks[-1].end, "total": total_samples},
        )
    for a, b in zip(tasks, tasks[1:]):
        if a.end != b.start:
            raise BlockBoundsError(
                "相邻任务边界不连续，存在重复或遗漏样本",
                location={"left_key": a.task_key, "right_key": b.task_key},
                evidence={"left_end": a.end, "right_start": b.start},
            )
    # 积分块归属唯一
    seen: Dict[int, str] = {}
    for t in tasks:
        for anchor in t.block_anchors:
            if anchor in seen:
                raise BlockBoundsError(
                    "积分块被多个任务重复归属",
                    location={"anchor": anchor},
                    evidence={"owners": [seen[anchor], t.task_key]},
                )
            seen[anchor] = t.task_key
    expected = block_anchors(total_samples)
    if sorted(seen.keys()) != expected:
        raise BlockBoundsError(
            "积分块归属集合与全局网格不一致，存在遗漏",
            evidence={"owned": len(seen), "expected": len(expected)},
        )


def manifest_to_dict(tasks: List[MeasurementTask], params: Dict) -> Dict:
    return {
        "schema": "loudbench.task-manifest/v1",
        "params": params,
        "hop_samples": HOP_SAMPLES,
        "block_samples": BLOCK_SAMPLES,
        "warmup_samples": WARMUP,
        "task_count": len(tasks),
        "tasks": [asdict(t) for t in tasks],
    }


def serialize_manifest(tasks: List[MeasurementTask], params: Dict) -> str:
    return json.dumps(manifest_to_dict(tasks, params), sort_keys=True, ensure_ascii=False, indent=2)
