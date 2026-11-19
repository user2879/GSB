"""固定 tick 块的表达状态计算与依赖记录（业务要求 3、4）。

每个块是纯函数：相同 (乐谱展开结果, 表达参数, tick 区间) 必然得到
逐 tick 相同的结果，因此并发度变化不影响最终拼装结果。
"""
from __future__ import annotations

import math

from .curves import pedal_track, tempo_at_ticks, velocity_map
from .expand import PlacedNote
from .models import ExpressionParams

# 连断规则缺省发音比例：legato 满发音，normal 90%，staccato 50%
DEFAULT_GATE = {"legato": 1.0, "normal": 0.9, "staccato": 0.5}


def gate_for(params: ExpressionParams, articulation: str) -> float:
    if articulation in params.articulation_gate:
        return float(params.articulation_gate[articulation])
    return DEFAULT_GATE.get(articulation, 0.9)


def stretch_for(params: ExpressionParams, rule: str) -> float:
    if rule in params.stretch_rules:
        return float(params.stretch_rules[rule])
    return float(params.default_stretch)


def compute_note_segments(
    placed: list[PlacedNote],
    params: ExpressionParams,
    ticks_per_beat: int,
    tempo: list[float],
) -> list[dict]:
    """计算每个音符的实际发音段/静音段边界与实际力度。"""
    base_bpm = float(params.tempo_curve[0].bpm)
    segments: list[dict] = []
    for note in placed:
        stretch = stretch_for(params, note.stretch_rule)
        start_tempo = float(tempo[note.start_tick])
        tempo_factor = base_bpm / start_tempo
        actual_ticks = max(1, int(round(note.nominal_ticks * stretch * tempo_factor)))
        actual_end = note.start_tick + actual_ticks
        gate = gate_for(params, note.articulation)
        gate_ticks = max(1, int(round(actual_ticks * gate)))
        gate_ticks = min(gate_ticks, actual_ticks)
        active_end = note.start_tick + gate_ticks
        actual_velocity = velocity_map(params, note.velocity)
        segments.append(
            {
                "note_id": note.note_id,
                "voice": note.voice,
                "pitch": note.pitch,
                "start_tick": note.start_tick,
                "nominal_end_tick": note.end_tick,
                "actual_end_tick": actual_end,
                "active_end_tick": active_end,
                "actual_duration_ticks": actual_ticks,
                "gate_ticks": gate_ticks,
                "tempo_at_start": start_tempo,
                "stretch_factor": stretch,
                "gate_ratio": gate,
                "actual_velocity": actual_velocity,
                "notated_velocity": note.velocity,
            }
        )
    return segments


def tempo_dependency_keys(params: ExpressionParams, tick_lo: int, tick_hi: int, ticks_per_beat: int) -> list[str]:
    """覆盖 [tick_lo, tick_hi) 的速度控制点依赖（含夹住区间的两个端点）。"""
    keys: list[str] = []
    ctrl = sorted(params.tempo_curve, key=lambda p: p.beat)
    for idx, point in enumerate(ctrl):
        pt = point.beat * ticks_per_beat
        # 控制点在区间内，或作为区间左侧/右侧的插值端点
        if tick_lo <= pt < tick_hi:
            keys.append(f"tempo:{idx}")
        elif pt < tick_lo and idx + 1 < len(ctrl) and ctrl[idx + 1].beat * ticks_per_beat > tick_lo:
            keys.append(f"tempo:{idx}")
        elif pt >= tick_hi and idx > 0 and ctrl[idx - 1].beat * ticks_per_beat < tick_hi:
            keys.append(f"tempo:{idx}")
    return sorted(set(keys))


def velocity_dependency_keys(segments_in_chunk: list[dict]) -> list[str]:
    keys = {f"velocity:{seg['notated_velocity']}" for seg in segments_in_chunk}
    return sorted(keys)


def compute_chunk(
    chunk_index: int,
    chunk_size: int,
    total_ticks: int,
    placed: list[PlacedNote],
    params: ExpressionParams,
    ticks_per_beat: int,
    tempo_full: list[float] | None = None,
    pedal_full: list[int] | None = None,
) -> dict:
    """计算单个 tick 块的完整表达状态与依赖摘要。"""
    tick_lo = chunk_index * chunk_size
    tick_hi = min(total_ticks, tick_lo + chunk_size)
    declared = tick_hi - tick_lo

    if tempo_full is None:
        tempo_full = tempo_at_ticks(params, ticks_per_beat, total_ticks).tolist()
    if pedal_full is None:
        pedal_full = pedal_track(params, ticks_per_beat, total_ticks).tolist()

    all_segments = compute_note_segments(placed, params, ticks_per_beat, tempo_full)
    # 落在本块内（有任何 tick 交集）的音符段
    segments_in_chunk = [
        seg for seg in all_segments if seg["start_tick"] < tick_hi and seg["actual_end_tick"] > tick_lo
    ]

    tempo_chunk = [round(float(v), 6) for v in tempo_full[tick_lo:tick_hi]]
    pedal_chunk = [int(v) for v in pedal_full[tick_lo:tick_hi]]

    # 逐 tick 音符状态：每个 tick 记录处于发音/静音段的音符及力度
    per_tick_notes: list[list[dict]] = [[] for _ in range(declared)]
    stretch_keys: set[str] = set()
    articulation_keys: set[str] = set()
    for seg, placed_note in zip(all_segments, placed):
        if not (seg["start_tick"] < tick_hi and seg["actual_end_tick"] > tick_lo):
            continue
        stretch_keys.add(f"stretch:{placed_note.stretch_rule}")
        articulation_keys.add(f"articulation:{placed_note.articulation}")
        lo = max(seg["start_tick"], tick_lo)
        hi = min(seg["actual_end_tick"], tick_hi)
        for tick in range(lo, hi):
            state = "active" if tick < seg["active_end_tick"] else "silent"
            per_tick_notes[tick - tick_lo].append(
                {
                    "note_id": seg["note_id"],
                    "state": state,
                    "velocity": seg["actual_velocity"],
                    "tempo": round(float(tempo_full[tick]), 6),
                    "pedal": int(pedal_full[tick]),
                }
            )

    pedal_keys = sorted(
        {
            f"pedal:{seg.pedal_id}"
            for seg in params.pedal_segments
            if seg.start_beat * ticks_per_beat < tick_hi and seg.end_beat * ticks_per_beat > tick_lo
        }
    )

    dependency_keys = sorted(
        set(tempo_dependency_keys(params, tick_lo, tick_hi, ticks_per_beat))
        | set(velocity_dependency_keys(segments_in_chunk))
        | stretch_keys
        | articulation_keys
        | set(pedal_keys)
    )

    return {
        "chunk_index": chunk_index,
        "tick_lo": tick_lo,
        "tick_hi": tick_hi,
        "declared_ticks": declared,
        "tempo": tempo_chunk,
        "pedal": pedal_chunk,
        "note_segments": [
            {
                "note_id": seg["note_id"],
                "voice": seg["voice"],
                "pitch": seg["pitch"],
                "start_tick": seg["start_tick"],
                "active_end_tick": seg["active_end_tick"],
                "actual_end_tick": seg["actual_end_tick"],
                "actual_duration_ticks": seg["actual_duration_ticks"],
                "actual_velocity": seg["actual_velocity"],
            }
            for seg in segments_in_chunk
        ],
        "per_tick_notes": per_tick_notes,
        "dependency_keys": dependency_keys,
    }
