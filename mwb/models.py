"""乐谱、演奏表达参数与编辑脚本的数据模型。

模型全部可 JSON 化，便于持久化与规范化摘要。
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

# 允许的编辑算子集合（业务要求 1）
EDIT_OPS = frozenset(
    {
        "tempo_upsert",
        "tempo_delete",
        "velocity_upsert",
        "velocity_delete",
        "stretch_set",
        "articulation_set",
        "note_articulation_set",
        "pedal_add",
        "pedal_remove",
    }
)


@dataclass
class Note:
    note_id: str
    pitch: int
    start_beat: int
    duration_beats: int
    voice: str
    velocity: int
    articulation: str = "normal"
    stretch_rule: str = "default"


@dataclass
class TempoPoint:
    beat: float
    bpm: float


@dataclass
class VelocityPoint:
    x: int          # 输入记谱力度 0..127
    y: int          # 输出实际力度 0..127


@dataclass
class PedalSegment:
    pedal_id: str
    start_beat: float
    end_beat: float


@dataclass
class ExpressionParams:
    tempo_curve: list[TempoPoint] = field(default_factory=list)
    velocity_curve: list[VelocityPoint] = field(default_factory=list)
    stretch_rules: dict[str, float] = field(default_factory=dict)
    articulation_gate: dict[str, float] = field(default_factory=dict)
    pedal_segments: list[PedalSegment] = field(default_factory=list)
    default_stretch: float = 1.0

    def clone(self) -> "ExpressionParams":
        return ExpressionParams(
            tempo_curve=[TempoPoint(p.beat, p.bpm) for p in self.tempo_curve],
            velocity_curve=[VelocityPoint(p.x, p.y) for p in self.velocity_curve],
            stretch_rules=dict(self.stretch_rules),
            articulation_gate=dict(self.articulation_gate),
            pedal_segments=[PedalSegment(p.pedal_id, p.start_beat, p.end_beat) for p in self.pedal_segments],
            default_stretch=self.default_stretch,
        )


@dataclass
class Edit:
    step: int
    op: str
    args: dict


@dataclass
class Score:
    title: str
    ticks_per_beat: int
    notes: list[Note]


def params_to_dict(params: ExpressionParams) -> dict:
    return asdict(params)
