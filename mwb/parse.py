"""JSON 样本解析与严格校验（业务要求 1）。

非法输入抛出带 E_xxx 业务码的 :class:`WorkbenchError`，
由 CLI 边界统一转成 JSON 错误，且不触发任何持久化写出。
"""
from __future__ import annotations

import math

from .atomicio import read_json
from .errors import (
    E_BEAT,
    E_EDIT_OP,
    E_EDIT_TARGET,
    E_PARSE,
    E_PITCH,
    E_SCHEMA,
    E_TEMPO,
    E_VELOCITY_CURVE,
    E_VOICE,
    WorkbenchError,
)
from .models import EDIT_OPS, Edit, ExpressionParams, Note, PedalSegment, Score, TempoPoint, VelocityPoint


def _is_int(value) -> bool:
    # 布尔值是 int 的子类，需排除
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _require(obj: dict, key: str, ctx: str):
    if key not in obj:
        raise WorkbenchError(E_SCHEMA, f"{ctx} 缺少必填字段: {key}")
    return obj[key]


def parse_score(path: str) -> Score:
    data = read_json(path)
    if not isinstance(data, dict):
        raise WorkbenchError(E_SCHEMA, "乐谱根节点必须是对象")
    title = data.get("title", "untitled")
    tpb = _require(data, "ticks_per_beat", "乐谱")
    if not _is_int(tpb) or tpb <= 0:
        raise WorkbenchError(E_SCHEMA, "ticks_per_beat 必须为正整数", {"value": tpb})
    raw_notes = _require(data, "notes", "乐谱")
    if not isinstance(raw_notes, list) or not raw_notes:
        raise WorkbenchError(E_SCHEMA, "notes 必须为非空数组")

    notes: list[Note] = []
    seen_ids: set[str] = set()
    voices: set[str] = set()
    for index, raw in enumerate(raw_notes):
        ctx = f"notes[{index}]"
        if not isinstance(raw, dict):
            raise WorkbenchError(E_SCHEMA, f"{ctx} 必须是对象")
        note_id = str(_require(raw, "note_id", ctx))
        if note_id in seen_ids:
            raise WorkbenchError(E_VOICE, "音符标识必须唯一", {"note_id": note_id})
        seen_ids.add(note_id)

        pitch = _require(raw, "pitch", ctx)
        if not _is_int(pitch) or not 0 <= pitch <= 127:
            raise WorkbenchError(E_PITCH, "音高必须在 0 至 127 之间", {"note_id": note_id, "pitch": pitch})

        start_beat = _require(raw, "start_beat", ctx)
        if not _is_int(start_beat) or start_beat < 0:
            raise WorkbenchError(E_BEAT, "记谱起始拍必须为非负整数", {"note_id": note_id})

        duration = _require(raw, "duration_beats", ctx)
        if not _is_int(duration) or duration <= 0:
            raise WorkbenchError(E_BEAT, "记谱时值必须为正整数", {"note_id": note_id})

        voice = str(_require(raw, "voice", ctx))
        voices.add(voice)
        velocity = _require(raw, "velocity", ctx)
        if not _is_int(velocity) or not 0 <= velocity <= 127:
            raise WorkbenchError(E_SCHEMA, "力度标记必须在 0 至 127 之间", {"note_id": note_id})

        notes.append(
            Note(
                note_id=note_id,
                pitch=pitch,
                start_beat=start_beat,
                duration_beats=duration,
                voice=voice,
                velocity=velocity,
                articulation=str(raw.get("articulation", "normal")),
                stretch_rule=str(raw.get("stretch_rule", "default")),
            )
        )
    return Score(title=title, ticks_per_beat=tpb, notes=notes)
