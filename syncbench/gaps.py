"""音视频空洞检测与可复现填补。

策略：
- linear_interp     端点线性插值；
- drift_extrapolate 依据漂移估计版本，按速率比做外推填充；
- mark_unfillable   保持零值并标注不可填补。

填补结果记录：漂移估计版本戳、插值区间、策略、内容哈希；
同素材同参数两次填补逐字节一致（无随机源参与）。
空洞长度超声明阈值 => E_GAP_TOO_LONG，并给出首个超限区间。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import GAP_DRIFT, GAP_LINEAR, GAP_MARK, WorkspaceConfig
from .detection import Gap
from .errors import E_GAP_TOO_LONG, make_record
from .hashing import hash_array


@dataclass(frozen=True)
class FillRecord:
    domain: str
    start: int
    end: int
    length: int
    policy: str
    drift_version: str
    filled: bool
    interval_ref: tuple[float, float] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "start": self.start,
            "end": self.end,
            "length": self.length,
            "policy": self.policy,
            "drift_version": self.drift_version,
            "filled": self.filled,
            "interval_ref": list(self.interval_ref) if self.interval_ref else None,
        }


@dataclass(frozen=True)
class GapFillResult:
    channel: int
    policy: str
    drift_version: str
    records: tuple[FillRecord, ...]
    audio_hash: str
    video_hash: str
    errors: tuple = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "policy": self.policy,
            "drift_version": self.drift_version,
            "records": [r.to_dict() for r in self.records],
            "audio_hash": self.audio_hash,
            "video_hash": self.video_hash,
            "errors": [e.to_dict() for e in self.errors],
        }


def _linear_fill(values: np.ndarray, gap: Gap) -> None:
    left = values[gap.start - 1] if gap.start > 0 else 0.0
    right = values[gap.end] if gap.end < len(values) else 0.0
    span = gap.end - gap.start
    values[gap.start:gap.end] = np.linspace(left, right, span, endpoint=False,
                                            dtype=values.dtype)


def _drift_fill(values: np.ndarray, gap: Gap, rate_ratio: float) -> None:
    left = values[gap.start - 1] if gap.start > 0 else 0.0
    right = values[gap.end] if gap.end < len(values) else 0.0
    span = gap.end - gap.start
    # 按漂移速率比重新分配插值位置（确定性，无随机）
    weights = (np.arange(1, span + 1, dtype=np.float64) / span) ** (1.0 / rate_ratio)
    values[gap.start:gap.end] = left + (right - left) * weights


def fill_channel(channel: int,
                 audio: np.ndarray,
                 video: np.ndarray,
                 audio_gaps: list[Gap],
                 video_gaps: list[Gap],
                 policy: str,
                 drift_version: str,
                 rate_ratio: float,
                 cfg: WorkspaceConfig) -> GapFillResult:
    a = audio.astype(np.float64).copy()
    v = video.copy()
    records: list[FillRecord] = []
    errors = []
    first_over: Gap | None = None

    for gap in audio_gaps:
        if gap.length > cfg.audio_gap_max and first_over is None:
            first_over = gap
        interval = (float(gap.start / cfg.audio_rate),
                    float(gap.end / cfg.audio_rate))
        if policy == GAP_LINEAR:
            _linear_fill(a, gap)
            filled = True
        elif policy == GAP_DRIFT:
            _drift_fill(a, gap, rate_ratio)
            filled = True
        elif policy == GAP_MARK:
            filled = False
        else:  # pragma: no cover - 配置校验已拦截
            raise ValueError(f"unknown gap policy {policy}")
        records.append(FillRecord("audio", gap.start, gap.end, gap.length,
                                  policy, drift_version, filled, interval))

    for gap in video_gaps:
        if gap.length > cfg.video_gap_max and first_over is None:
            first_over = gap
        interval = (float(gap.start / cfg.video_fps),
                    float(gap.end / cfg.video_fps))
        if policy == GAP_LINEAR:
            _fill_video_linear(v, gap)
            filled = True
        elif policy == GAP_DRIFT:
            _fill_video_drift(v, gap, rate_ratio)
            filled = True
        elif policy == GAP_MARK:
            filled = False
        else:  # pragma: no cover
            raise ValueError(f"unknown gap policy {policy}")
        records.append(FillRecord("video", gap.start, gap.end, gap.length,
                                  policy, drift_version, filled, interval))

    if first_over is not None:
        errors.append(make_record(
            E_GAP_TOO_LONG,
            f"通道 {channel} 首个超限空洞 {first_over.domain} "
            f"[{first_over.start},{first_over.end}) 长度 {first_over.length}",
            channel=channel,
            domain=first_over.domain,
            first_start=first_over.start,
            first_end=first_over.end,
            length=first_over.length,
            audio_max=cfg.audio_gap_max,
            video_max=cfg.video_gap_max))

    return GapFillResult(
        channel=channel,
        policy=policy,
        drift_version=drift_version,
        records=tuple(records),
        audio_hash=hash_array(a),
        video_hash=hash_array(v),
        errors=tuple(errors),
    )


def _fill_video_linear(video: np.ndarray, gap: Gap) -> None:
    left = video[gap.start - 1].astype(np.float64) if gap.start > 0 else np.zeros_like(video[0])
    right = video[gap.end].astype(np.float64) if gap.end < len(video) else np.zeros_like(video[0])
    span = gap.end - gap.start
    for i in range(span):
        t = (i + 1) / span
        video[gap.start + i] = np.clip((1 - t) * left + t * right, 0, 255).astype(np.uint8)


def _fill_video_drift(video: np.ndarray, gap: Gap, rate_ratio: float) -> None:
    left = video[gap.start - 1].astype(np.float64) if gap.start > 0 else np.zeros_like(video[0])
    right = video[gap.end].astype(np.float64) if gap.end < len(video) else np.zeros_like(video[0])
    span = gap.end - gap.start
    for i in range(span):
        t = ((i + 1) / span) ** (1.0 / rate_ratio)
        video[gap.start + i] = np.clip((1 - t) * left + t * right, 0, 255).astype(np.uint8)
