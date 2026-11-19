"""记谱拍位到整数 tick 的确定性展开（业务要求 2）。

取整规则唯一：``tick = beat * ticks_per_beat`` 四舍五入到最近偶数
（Python ``round`` 的银行家舍入对整数倍结果无影响，这里拍位均为整数，
实际等价于精确相乘；保留统一入口以便表达层复用同一规则）。
"""
from __future__ import annotations

from dataclasses import dataclass

from .errors import E_OVERLAP_SAME_VOICE, WorkbenchError
from .models import Note, Score


@dataclass
class PlacedNote:
    note_id: str
    voice: str
    pitch: int
    start_tick: int
    end_tick: int          # 不含该 tick（半开区间 [start, end)）
    velocity: int
    articulation: str
    stretch_rule: str

    @property
    def nominal_ticks(self) -> int:
        return self.end_tick - self.start_tick


def beat_to_tick(beat: float, ticks_per_beat: int) -> int:
    """唯一取整规则：四舍五入到最近整数（.5 取偶数），结果非负。"""
    tick = round(float(beat) * ticks_per_beat)
    return int(tick)


def expand_score(score: Score) -> tuple[list[PlacedNote], int, list[dict]]:
    """返回 (放置后的音符, 时间轴总 tick 数, 跨声部重叠区间列表)。

    同声部音符若在同一 tick 重叠直接返回 E_OVERLAP_SAME_VOICE；
    跨声部重叠只记录，写入诊断报告。
    """
    tpb = score.ticks_per_beat
    placed: list[PlacedNote] = []
    for note in score.notes:
        start_tick = beat_to_tick(note.start_beat, tpb)
        end_tick = start_tick + note.duration_beats * tpb
        placed.append(
            PlacedNote(
                note_id=note.note_id,
                voice=note.voice,
                pitch=note.pitch,
                start_tick=start_tick,
                end_tick=end_tick,
                velocity=note.velocity,
                articulation=note.articulation,
                stretch_rule=note.stretch_rule,
            )
        )

    # 同声部重叠检测（半开区间相交即非法）
    by_voice: dict[str, list[PlacedNote]] = {}
    for item in placed:
        by_voice.setdefault(item.voice, []).append(item)
    for voice, items in by_voice.items():
        items.sort(key=lambda n: (n.start_tick, n.end_tick, n.note_id))
        for prev, cur in zip(items, items[1:]):
            if cur.start_tick < prev.end_tick:
                raise WorkbenchError(
                    E_OVERLAP_SAME_VOICE,
                    "同声部音符在同一 tick 重叠",
                    {
                        "voice": voice,
                        "note_a": prev.note_id,
                        "note_b": cur.note_id,
                        "tick": cur.start_tick,
                    },
                )

    total_ticks = max(item.end_tick for item in placed)
    cross_voice_overlaps = _cross_voice_overlaps(by_voice)
    placed.sort(key=lambda n: (n.start_tick, n.voice, n.note_id))
    return placed, total_ticks, cross_voice_overlaps


def _cross_voice_overlaps(by_voice: dict[str, list[PlacedNote]]) -> list[dict]:
    voices = sorted(by_voice)
    results: list[dict] = []
    for i in range(len(voices)):
        for j in range(i + 1, len(voices)):
            va, vb = voices[i], voices[j]
            intervals: list[tuple[int, int]] = []
            for na in by_voice[va]:
                for nb in by_voice[vb]:
                    lo = max(na.start_tick, nb.start_tick)
                    hi = min(na.end_tick, nb.end_tick)
                    if lo < hi:
                        intervals.append((lo, hi))
            for lo, hi in _merge_intervals(intervals):
                results.append(
                    {
                        "voice_a": va,
                        "voice_b": vb,
                        "start_tick": lo,
                        "end_tick": hi,
                        "tick_count": hi - lo,
                    }
                )
    return results


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for lo, hi in intervals[1:]:
        last_lo, last_hi = merged[-1]
        if lo <= last_hi:
            merged[-1] = (last_lo, max(last_hi, hi))
        else:
            merged.append((lo, hi))
    return merged
