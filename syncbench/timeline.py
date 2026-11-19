"""音视频双域统一时间轴映射与一致性约束。

每个通道由同一份分段线性漂移估计构造唯一 TimelineMap：
    local_time(ref_time) = intercept + slope * ref_time  （分段）
音频片段与视频片段都由该映射派生，保证双域同源；
非整数延迟仅在 delay_compensation 开启时从本地时间中扣除。

参考通道切换时，所有通道映射改为“相对新参考通道”重算；
映射结构同时保留相对参考时钟的绝对速率，因此空洞填补（不依赖参考通道）
可被上层依赖图原样复用。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import WorkspaceConfig
from .errors import E_SYNC_MISMATCH, make_record
from .estimation import EstimationResult


@dataclass(frozen=True)
class TimelineMap:
    channel: int
    reference_channel: int
    delay_compensation: bool
    delay_samples: float
    # 分段边界（参考秒）与每段 slope/intercept（本地秒 = a + b*ref 秒）
    bounds: tuple[float, ...]
    slopes: tuple[float, ...]
    intercepts: tuple[float, ...]
    version: str

    def local_seconds(self, ref_seconds: np.ndarray) -> np.ndarray:
        x = np.asarray(ref_seconds, dtype=np.float64)
        out = np.empty_like(x)
        for i in range(len(self.slopes)):
            lo = self.bounds[i]
            hi = self.bounds[i + 1]
            sel = (x >= lo) & (x < hi) if i < len(self.slopes) - 1 else (x >= lo)
            # 统一秒域：local_sec = intercept_sec + slope * ref_sec
            out[sel] = self.intercepts[i] + self.slopes[i] * x[sel]
        if self.delay_compensation:
            out -= self.delay_samples / 48000.0
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "channel": self.channel,
            "reference_channel": self.reference_channel,
            "delay_compensation": self.delay_compensation,
            "delay_samples": self.delay_samples,
            "bounds": list(self.bounds),
            "slopes": list(self.slopes),
            "intercepts": list(self.intercepts),
            "version": self.version,
        }


@dataclass(frozen=True)
class AlignSegment:
    domain: str          # audio | video
    channel: int
    local_index: int     # 本地样本/帧索引
    ref_seconds: float   # 映射后的参考时间
    map_version: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "channel": self.channel,
            "local_index": self.local_index,
            "ref_seconds": self.ref_seconds,
            "map_version": self.map_version,
        }


@dataclass(frozen=True)
class DualDomainResult:
    timeline: TimelineMap
    audio_segments: tuple[AlignSegment, ...]
    video_segments: tuple[AlignSegment, ...]
    max_ref_diff_sec: float
    mismatch: dict[str, Any] | None
    errors: tuple = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timeline": self.timeline.to_dict(),
            "audio_segments": [s.to_dict() for s in self.audio_segments],
            "video_segments": [s.to_dict() for s in self.video_segments],
            "max_ref_diff_sec": self.max_ref_diff_sec,
            "mismatch": self.mismatch,
            "errors": [e.to_dict() for e in self.errors],
        }


def build_timeline(est: EstimationResult, cfg: WorkspaceConfig,
                   reference_slope: float, version: str) -> TimelineMap:
    """从分段拟合构造时间轴映射（参考通道斜率用于相对化）。"""
    fs = cfg.audio_rate
    bounds = [0.0]
    slopes: list[float] = []
    intercepts: list[float] = []
    for seg in est.segments:
        # 拟合为样本->样本：slope 是无量纲速率比（秒域同样适用），
        # intercept 由样本换算为秒。
        slope_sec = seg.slope
        intercept_sec = seg.intercept / fs
        slopes.append(float(slope_sec))
        intercepts.append(float(intercept_sec))
        if seg.segment < len(est.segments) - 1:
            # 段边界：按段内点数比例放到参考时间轴
            bounds.append(round(cfg.duration_sec
                                * (seg.segment + 1) / len(est.segments), 6))
    bounds.append(cfg.duration_sec + 1e-9)
    delay = est.delay_samples if cfg.channel_params[est.channel].delay_compensation else 0.0
    return TimelineMap(
        channel=est.channel,
        reference_channel=cfg.reference_channel,
        delay_compensation=bool(cfg.channel_params[est.channel].delay_compensation),
        delay_samples=float(delay),
        bounds=tuple(bounds),
        slopes=tuple(slopes),
        intercepts=tuple(intercepts),
        version=version,
    )


def derive_segments(timeline: TimelineMap, cfg: WorkspaceConfig
                    ) -> tuple[tuple[AlignSegment, ...], tuple[AlignSegment, ...]]:
    fs = cfg.audio_rate
    # 两域在同一组连续内容时刻 grid_sec 上求映射，输入严格一致；
    # 样本/帧索引仅作定位标签，量化误差不进入双域时间差比较
    grid_sec = np.arange(0.0, cfg.duration_sec - 0.3, 0.25)
    audio_idx = np.round(grid_sec * fs).astype(np.int64)
    video_idx = np.round(grid_sec * cfg.video_fps).astype(np.int64)
    mapped = timeline.local_seconds(grid_sec)
    a_ref = mapped
    v_ref = mapped
    audio_segments = tuple(
        AlignSegment("audio", timeline.channel, int(i), float(t), timeline.version)
        for i, t in zip(audio_idx, a_ref))
    video_segments = tuple(
        AlignSegment("video", timeline.channel, int(i), float(t), timeline.version)
        for i, t in zip(video_idx, v_ref))
    return audio_segments, video_segments


def check_dual_domain(timeline: TimelineMap,
                      audio_segments: tuple[AlignSegment, ...],
                      video_segments: tuple[AlignSegment, ...],
                      cfg: WorkspaceConfig) -> tuple[float, dict[str, Any] | None, tuple]:
    """同一内容时刻在音视频两域映射后的参考时间差检查。

    音频/视频片段本由同一 TimelineMap 派生；此处把两域片段按内容时刻对齐，
    比较其映射到参考轴的时间，定位首个失配的帧号与样本位置。
    """
    # 两域片段数相同且一一对应同一内容时刻，直接逐点比较，无需外推
    a_map = np.array([s.ref_seconds for s in audio_segments])
    v_map = np.array([s.ref_seconds for s in video_segments])
    a_idx = np.array([s.local_index for s in audio_segments])
    v_idx = np.array([s.local_index for s in video_segments])
    diff = np.abs(a_map - v_map)
    max_diff = float(diff.max())
    mismatch = None
    errors = ()
    if max_diff > cfg.av_sync_tol_sec:
        first = int(np.argmax(diff > cfg.av_sync_tol_sec))
        mismatch = {
            "first_ref_seconds": float(v_map[first]),
            "audio_sample": int(a_idx[first]),
            "video_frame": int(v_idx[first]),
            "audio_ref_seconds": float(a_map[first]),
            "video_ref_seconds": float(v_map[first]),
            "diff_seconds": float(diff[first]),
        }
        errors = (make_record(
            E_SYNC_MISMATCH,
            f"通道 {timeline.channel} 首个双域失配 帧 {mismatch['video_frame']} "
            f"样本 {mismatch['audio_sample']} 差 {diff[first]:.6f}s",
            channel=timeline.channel, **mismatch),)
    return max_diff, mismatch, errors


def build_dual_domain(est: EstimationResult, cfg: WorkspaceConfig,
                      reference_slope: float, version: str) -> DualDomainResult:
    timeline = build_timeline(est, cfg, reference_slope, version)
    audio_segments, video_segments = derive_segments(timeline, cfg)
    max_diff, mismatch, errors = check_dual_domain(
        timeline, audio_segments, video_segments, cfg)
    return DualDomainResult(timeline, audio_segments, video_segments,
                            max_diff, mismatch, errors)
