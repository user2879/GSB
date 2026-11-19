"""演奏表达参数与编辑脚本解析校验。"""
from __future__ import annotations

import math

from .atomicio import read_json
from .errors import (
    E_EDIT_OP,
    E_EDIT_TARGET,
    E_SCHEMA,
    E_TEMPO,
    E_VELOCITY_CURVE,
    WorkbenchError,
)
from .models import EDIT_OPS, Edit, ExpressionParams, PedalSegment, TempoPoint, VelocityPoint


def _finite_positive(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def parse_expression(path: str) -> ExpressionParams:
    data = read_json(path)
    if not isinstance(data, dict):
        raise WorkbenchError(E_SCHEMA, "表达参数根节点必须是对象")

    tempo_raw = data.get("tempo_curve", [])
    if not isinstance(tempo_raw, list) or not tempo_raw:
        raise WorkbenchError(E_TEMPO, "速度曲线必须为非空控制点数组")
    tempo: list[TempoPoint] = []
    last_beat = None
    for point in tempo_raw:
        if not isinstance(point, dict) or "beat" not in point or "bpm" not in point:
            raise WorkbenchError(E_TEMPO, "速度控制点需包含 beat 与 bpm")
        beat, bpm = point["beat"], point["bpm"]
        if not isinstance(beat, (int, float)) or isinstance(beat, bool) or not math.isfinite(beat) or beat < 0:
            raise WorkbenchError(E_TEMPO, "速度控制点拍位必须为非负有限数", {"point": point})
        if not _finite_positive(bpm):
            raise WorkbenchError(E_TEMPO, "速度必须为有限正数", {"point": point})
        if last_beat is not None and not beat > last_beat:
            raise WorkbenchError(E_TEMPO, "速度控制点拍位必须严格递增", {"beat": beat, "prev": last_beat})
        last_beat = beat
        tempo.append(TempoPoint(float(beat), float(bpm)))

    vel_raw = data.get("velocity_curve", [])
    if not isinstance(vel_raw, list) or not vel_raw:
        raise WorkbenchError(E_VELOCITY_CURVE, "力度映射曲线必须为非空控制点数组")
    velocity: list[VelocityPoint] = []
    last_x = None
    last_y = None
    monotonic_up = True
    monotonic_down = True
    for point in vel_raw:
        if not isinstance(point, dict) or "x" not in point or "y" not in point:
            raise WorkbenchError(E_VELOCITY_CURVE, "力度控制点需包含 x 与 y")
        x, y = point["x"], point["y"]
        if not isinstance(x, int) or isinstance(x, bool) or not 0 <= x <= 127:
            raise WorkbenchError(E_VELOCITY_CURVE, "力度控制点 x 必须为 0..127 整数", {"point": point})
        if not isinstance(y, int) or isinstance(y, bool) or not 0 <= y <= 127:
            raise WorkbenchError(E_VELOCITY_CURVE, "力度控制点 y 必须为 0..127 整数", {"point": point})
        if last_x is not None and x <= last_x:
            raise WorkbenchError(E_VELOCITY_CURVE, "力度控制点 x 必须严格递增", {"x": x, "prev": last_x})
        if last_y is not None:
            monotonic_up &= y >= last_y
            monotonic_down &= y <= last_y
        last_x, last_y = x, y
        velocity.append(VelocityPoint(x, y))
    if not (monotonic_up or monotonic_down):
        raise WorkbenchError(E_VELOCITY_CURVE, "力度映射曲线控制点必须单调（单调增或单调减）")

    stretch_rules = data.get("stretch_rules", {})
    if not isinstance(stretch_rules, dict):
        raise WorkbenchError(E_SCHEMA, "stretch_rules 必须为对象")
    for name, factor in stretch_rules.items():
        if not _finite_positive(factor):
            raise WorkbenchError(E_SCHEMA, f"伸缩规则 {name} 的系数必须为有限正数")

    articulation_gate = data.get("articulation_gate", {})
    if not isinstance(articulation_gate, dict):
        raise WorkbenchError(E_SCHEMA, "articulation_gate 必须为对象")
    for name, gate in articulation_gate.items():
        if not isinstance(gate, (int, float)) or isinstance(gate, bool) or not 0.0 < gate <= 1.0:
            raise WorkbenchError(E_SCHEMA, f"连断规则 {name} 的发音比例必须在 (0,1] 内")

    pedal_raw = data.get("pedal_segments", [])
    if not isinstance(pedal_raw, list):
        raise WorkbenchError(E_SCHEMA, "pedal_segments 必须为数组")
    pedals: list[PedalSegment] = []
    pedal_ids: set[str] = set()
    for segment in pedal_raw:
        if not isinstance(segment, dict):
            raise WorkbenchError(E_SCHEMA, "踏板段落必须是对象")
        pedal_id = str(segment.get("pedal_id", ""))
        if not pedal_id or pedal_id in pedal_ids:
            raise WorkbenchError(E_SCHEMA, "踏板段落标识缺失或重复", {"segment": segment})
        pedal_ids.add(pedal_id)
        start, end = segment.get("start_beat"), segment.get("end_beat")
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            raise WorkbenchError(E_SCHEMA, "踏板段落起止拍必须是数值", {"segment": segment})
        if not (math.isfinite(start) and math.isfinite(end) and start >= 0 and end > start):
            raise WorkbenchError(E_SCHEMA, "踏板段落需满足 0<=start<end", {"segment": segment})
        pedals.append(PedalSegment(pedal_id, float(start), float(end)))

    default_stretch = data.get("default_stretch", 1.0)
    if not _finite_positive(default_stretch):
        raise WorkbenchError(E_SCHEMA, "default_stretch 必须为有限正数")

    return ExpressionParams(
        tempo_curve=tempo,
        velocity_curve=velocity,
        stretch_rules={k: float(v) for k, v in stretch_rules.items()},
        articulation_gate={k: float(v) for k, v in articulation_gate.items()},
        pedal_segments=pedals,
        default_stretch=float(default_stretch),
    )
