"""编辑脚本逐步重演、目标语义校验与依赖失效分析（业务要求 4）。"""
from __future__ import annotations

import math

from .errors import E_EDIT_TARGET, E_TEMPO, E_VELOCITY_CURVE, WorkbenchError
from .expand import PlacedNote
from .models import Edit, ExpressionParams, PedalSegment, TempoPoint, VelocityPoint


def validate_targets(edits: list[Edit], params: ExpressionParams, placed: list[PlacedNote]) -> None:
    """干跑校验所有编辑引用的目标音符/参数存在，非法即 E_EDIT_TARGET。"""
    note_ids = {n.note_id for n in placed}
    sim = params.clone()
    for edit in edits:
        op, args = edit.op, edit.args
        if op == "tempo_delete":
            if not any(p.beat == float(args["beat"]) for p in sim.tempo_curve):
                raise WorkbenchError(E_EDIT_TARGET, "删除的速度控制点不存在", {"step": edit.step, **args})
        elif op == "velocity_delete":
            if not any(p.x == int(args["x"]) for p in sim.velocity_curve):
                raise WorkbenchError(E_EDIT_TARGET, "删除的力度控制点不存在", {"step": edit.step, **args})
        elif op == "pedal_remove":
            if not any(p.pedal_id == str(args["pedal_id"]) for p in sim.pedal_segments):
                raise WorkbenchError(E_EDIT_TARGET, "移除的踏板段落不存在", {"step": edit.step, **args})
        elif op == "note_articulation_set":
            if str(args["note_id"]) not in note_ids:
                raise WorkbenchError(E_EDIT_TARGET, "编辑引用的音符不存在", {"step": edit.step, **args})
        # upsert / set 类是创建或覆盖，不存在时创建，无需目标存在性
        apply_edit(sim, edit, placed)


def apply_edit(params: ExpressionParams, edit: Edit, placed: list[PlacedNote]) -> None:
    """把单个编辑就地应用到参数副本，并维护曲线约束。"""
    op, args = edit.op, edit.args
    if op == "tempo_upsert":
        beat, bpm = float(args["beat"]), float(args["bpm"])
        if not (math.isfinite(beat) and math.isfinite(bpm) and bpm > 0 and beat >= 0):
            raise WorkbenchError(E_TEMPO, "速度控制点取值非法", {"step": edit.step})
        params.tempo_curve = [p for p in params.tempo_curve if p.beat != beat]
        params.tempo_curve.append(TempoPoint(beat, bpm))
        params.tempo_curve.sort(key=lambda p: p.beat)
    elif op == "tempo_delete":
        target = float(args["beat"])
        params.tempo_curve = [p for p in params.tempo_curve if p.beat != target]
        if not params.tempo_curve:
            raise WorkbenchError(E_TEMPO, "删除后速度曲线不能为空", {"step": edit.step})
    elif op == "velocity_upsert":
        x, y = int(args["x"]), int(args["y"])
        if not (0 <= x <= 127 and 0 <= y <= 127):
            raise WorkbenchError(E_VELOCITY_CURVE, "力度控制点取值非法", {"step": edit.step})
        params.velocity_curve = [p for p in params.velocity_curve if p.x != x]
        params.velocity_curve.append(VelocityPoint(x, y))
        params.velocity_curve.sort(key=lambda p: p.x)
        ys = [p.y for p in params.velocity_curve]
        if not (all(b >= a for a, b in zip(ys, ys[1:])) or all(b <= a for a, b in zip(ys, ys[1:]))):
            raise WorkbenchError(E_VELOCITY_CURVE, "编辑后力度曲线不再单调", {"step": edit.step})
    elif op == "velocity_delete":
        target = int(args["x"])
        params.velocity_curve = [p for p in params.velocity_curve if p.x != target]
        if not params.velocity_curve:
            raise WorkbenchError(E_VELOCITY_CURVE, "删除后力度曲线不能为空", {"step": edit.step})
    elif op == "stretch_set":
        factor = float(args["factor"])
        if not (math.isfinite(factor) and factor > 0):
            raise WorkbenchError(E_TEMPO, "伸缩系数必须为有限正数", {"step": edit.step})
        params.stretch_rules[str(args["name"])] = factor
    elif op == "articulation_set":
        gate = float(args["gate"])
        if not 0.0 < gate <= 1.0:
            raise WorkbenchError(E_TEMPO, "连断发音比例必须在 (0,1]", {"step": edit.step})
        params.articulation_gate[str(args["name"])] = gate
    elif op == "note_articulation_set":
        note_id = str(args["note_id"])
        for note in placed:
            if note.note_id == note_id:
                note.articulation = str(args["articulation"])
    elif op == "pedal_add":
        pedal_id, start, end = str(args["pedal_id"]), float(args["start_beat"]), float(args["end_beat"])
        if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end):
            raise WorkbenchError(E_TEMPO, "踏板段落区间非法", {"step": edit.step})
        params.pedal_segments = [p for p in params.pedal_segments if p.pedal_id != pedal_id]
        params.pedal_segments.append(PedalSegment(pedal_id, start, end))
        params.pedal_segments.sort(key=lambda p: p.start_beat)
    elif op == "pedal_remove":
        pedal_id = str(args["pedal_id"])
        params.pedal_segments = [p for p in params.pedal_segments if p.pedal_id != pedal_id]
    else:  # pragma: no cover - 白名单已在解析阶段保证
        raise WorkbenchError(E_EDIT_TARGET, f"未知算子 {op}")


def edit_dependency_keys(edit: Edit, placed: list[PlacedNote]) -> list[str]:
    """返回单步编辑所影响的依赖键集合（用于块失效判定）。"""
    op, args = edit.op, edit.args
    if op in ("tempo_upsert", "tempo_delete"):
        # 控制点索引可能重排，保守标记全部速度键失效
        return sorted({f"tempo:{i}" for i in range(64)})
    if op in ("velocity_upsert", "velocity_delete"):
        return [f"velocity:{int(args['x'])}"]
    if op == "stretch_set":
        return [f"stretch:{args['name']}"]
    if op == "articulation_set":
        return [f"articulation:{args['name']}"]
    if op == "note_articulation_set":
        note_id = str(args["note_id"])
        for note in placed:
            if note.note_id == note_id:
                return [f"articulation:{note.articulation}"]
        return []
    if op in ("pedal_add", "pedal_remove"):
        return [f"pedal:{args['pedal_id']}"]
    return []

def edit_tick_range(edit: Edit, ticks_per_beat: int) -> tuple[int, int] | None:
    """新增/移除类编辑的显式影响 tick 区间；无区间信息返回 None。"""
    op, args = edit.op, edit.args
    if op == "pedal_add":
        start = int(round(float(args["start_beat"]) * ticks_per_beat))
        end = int(round(float(args["end_beat"]) * ticks_per_beat))
        return start, end
    if op == "pedal_remove":
        return None  # 移除走依赖键匹配（旧块登记了该 pedal 键）
    if op == "note_articulation_set":
        return None
    return None
