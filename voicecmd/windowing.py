"""重叠分析窗切分：全局采样坐标、显式区间与跨片段引用集合。"""

from dataclasses import dataclass, field

from .config import WindowConfig
from .vad import SpeechInterval


@dataclass(frozen=True)
class AnalysisWindow:
    window_index: int
    start_sample: int          # 全局采样坐标，左闭
    end_sample: int            # 全局采样坐标，右开
    fragment_seqs: tuple[int, ...]
    speech_region: tuple[int, int] = field(compare=False)


def fragment_offsets(fragments) -> list[tuple[int, int, int]]:
    """返回 (seq, 全局起点, 全局终点)，按 seq 排序。"""
    offsets = []
    cursor = 0
    for fragment in sorted(fragments, key=lambda f: f.seq):
        offsets.append((fragment.seq, cursor, cursor + fragment.sample_count))
        cursor += fragment.sample_count
    return offsets


def _global_intervals(
    intervals: list[SpeechInterval],
    offsets: list[tuple[int, int, int]],
    bridge_gap: int = 0,
) -> list[tuple[int, int]]:
    local_map = {seq: (start, end) for seq, start, end in offsets}
    globals_ = []
    for interval in sorted(intervals, key=lambda item: local_map[item.fragment_seq][0]):
        base, _ = local_map[interval.fragment_seq]
        globals_.append(
            (base + interval.start_sample, base + interval.end_sample)
        )
    globals_.sort()
    merged: list[list[int]] = []
    for start, end in globals_:
        if merged and start - merged[-1][1] <= bridge_gap:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def _referenced_fragments(start: int, end: int,
                          offsets: list[tuple[int, int, int]]) -> tuple[int, ...]:
    refs = tuple(
        seq
        for seq, base, fragment_end in offsets
        if fragment_end > start and base < end
    )
    return refs


def build_windows(
    intervals: list[SpeechInterval],
    fragments,
    cfg: WindowConfig,
) -> list[AnalysisWindow]:
    offsets = fragment_offsets(fragments)
    bridge_gap = cfg.window_length  # 小于一个分析窗的片段边界间隙视为同一语音区
    regions = _global_intervals(intervals, offsets, bridge_gap=bridge_gap)
    windows: list[AnalysisWindow] = []
    window_index = 0
    for region_start, region_end in regions:
        start = region_start
        while start + cfg.window_length <= region_end:
            end = start + cfg.window_length
            refs = _referenced_fragments(start, end, offsets)
            windows.append(
                AnalysisWindow(
                    window_index=window_index,
                    start_sample=start,
                    end_sample=end,
                    fragment_seqs=refs,
                    speech_region=(region_start, region_end),
                )
            )
            window_index += 1
            start += cfg.window_shift
        if not cfg.drop_partial and start < region_end:
            end = region_end
            refs = _referenced_fragments(start, end, offsets)
            windows.append(
                AnalysisWindow(
                    window_index=window_index,
                    start_sample=start,
                    end_sample=end,
                    fragment_seqs=refs,
                    speech_region=(region_start, region_end),
                )
            )
            window_index += 1
    return windows
