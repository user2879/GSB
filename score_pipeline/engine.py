"""演奏时间轴合成引擎：验证、渲染、失效闭包、版本化增量演进。"""

from __future__ import annotations

from dataclasses import dataclass

from . import SHARD_BEATS
from .errors import (
    E_BEAT_RANGE,
    E_PEDAL_OVERLAP,
    E_TIMELINE_HASH,
    PipelineError,
)
from .model import Note, Score, content_hash, round6
from .tempo import TempoMap

ARTICULATION_DURATION_FACTOR = {"legato": 1.0, "normal": 0.95, "staccato": 0.5}
ARTICULATION_VELOCITY_FACTOR = {"legato": 1.0, "normal": 0.95, "staccato": 0.9}


def validate_expressions(score: Score, tempo=None, dynamics=None, pedals=None) -> None:
    """参数层业务校验：覆盖 E_BEAT_RANGE / E_VELOCITY_MONO / E_PEDAL_OVERLAP。"""
    tempo = score.tempo if tempo is None else tempo
    dynamics = score.dynamics if dynamics is None else dynamics
    pedals = score.pedals if pedals is None else pedals

    TempoMap(tempo, score.total_beats)  # 非单调/越界在此抛 E_VELOCITY_MONO/E_BEAT_RANGE

    for part_id, notes in score.parts.items():
        for note in notes:
            if note.duration_beats <= 0:
                raise PipelineError(
                    E_BEAT_RANGE,
                    "音符时值必须为正",
                    {"note_id": note.note_id, "duration_beats": note.duration_beats},
                )
            if note.start_beat < -1e-12 or note.end_beat > score.total_beats + 1e-9:
                raise PipelineError(
                    E_BEAT_RANGE,
                    "音符拍区间超出 [0,total_beats]",
                    {"note_id": note.note_id, "start": note.start_beat, "end": note.end_beat},
                )

    for interval in dynamics:
        if interval["start_beat"] < -1e-12 or interval["end_beat"] > score.total_beats + 1e-9:
            raise PipelineError(
                E_BEAT_RANGE, "力度区间越界", {"interval": interval}
            )
        if interval["end_beat"] <= interval["start_beat"]:
            raise PipelineError(
                E_BEAT_RANGE, "力度区间长度必须为正", {"interval": interval}
            )

    by_part: dict[str, list[dict]] = {}
    for pedal in pedals:
        by_part.setdefault(pedal["part_id"], []).append(pedal)
        if pedal["start_beat"] < -1e-12 or pedal["end_beat"] > score.total_beats + 1e-9:
            raise PipelineError(E_BEAT_RANGE, "踏板区间越界", {"interval": pedal})
        if pedal["end_beat"] <= pedal["start_beat"]:
            raise PipelineError(E_BEAT_RANGE, "踏板区间长度必须为正", {"interval": pedal})
    for part_id, intervals in by_part.items():
        ordered = sorted(intervals, key=lambda item: item["start_beat"])
        for prev, cur in zip(ordered, ordered[1:]):
            if cur["start_beat"] < prev["end_beat"] - 1e-12:
                raise PipelineError(
                    E_PEDAL_OVERLAP,
                    "同声部踏板区间交叉",
                    {"part_id": part_id, "prev": prev, "current": cur},
                )


def dynamics_gain(dynamics: list[dict], beat: float) -> float:
    """力度包络增益：包含 beat 的区间内线性插值；无匹配则 1.0。"""
    for interval in dynamics:
        lo, hi = float(interval["start_beat"]), float(interval["end_beat"])
        if lo - 1e-12 <= beat < hi - 1e-12 or abs(beat - hi) <= 1e-12:
            if hi <= lo:
                return float(interval["gain"])
            ratio = (beat - lo) / (hi - lo)
            gain = interval["gain"] + (interval["end_gain"] - interval["gain"]) * ratio
            return float(gain)
    return 1.0


def render_note(note: Note, tempo_map: TempoMap, dynamics: list[dict], beat_unit_ms: float) -> dict:
    start_ms = tempo_map.beat_to_ms_r(note.start_beat)
    end_ms = tempo_map.beat_to_ms_r(note.end_beat)
    factor = ARTICULATION_DURATION_FACTOR[note.articulation]
    duration_ms = round6((end_ms - start_ms) * factor)
    gain = dynamics_gain(dynamics, note.start_beat)
    vel_factor = ARTICULATION_VELOCITY_FACTOR[note.articulation]
    envelope_velocity = round6(note.velocity * gain * vel_factor)
    return {
        "note_id": note.note_id,
        "pitch": note.pitch,
        "start_beat": round6(note.start_beat),
        "end_beat": round6(note.end_beat),
        "start_ms": start_ms,
        "end_ms": end_ms,
        "duration_ms": duration_ms,
        "velocity_base": note.velocity,
        "dynamics_gain": round6(gain),
        "velocity_envelope": envelope_velocity,
        "articulation": note.articulation,
    }


def render_pedals(pedals: list[dict], tempo_map: TempoMap) -> list[dict]:
    rendered = []
    for pedal in pedals:
        rendered.append(
            {
                "part_id": pedal["part_id"],
                "start_beat": round6(pedal["start_beat"]),
                "end_beat": round6(pedal["end_beat"]),
                "start_ms": tempo_map.beat_to_ms_r(pedal["start_beat"]),
                "end_ms": tempo_map.beat_to_ms_r(pedal["end_beat"]),
            }
        )
    return sorted(rendered, key=lambda item: (item["part_id"], item["start_beat"]))


def shard_index(beat: float) -> int:
    return int(beat // SHARD_BEATS)


def compute_closure(
    kind: str,
    changed_range: tuple[float, float],
    tempo_changed: bool,
    tempo_map: TempoMap,
    old_tempo_map: TempoMap | None,
    notes: list[Note],
) -> dict:
    """拍区间失效闭包。

    返回 recompute_note_ids / shift_note_ids / shifted_count / range / offset_ms。
    速度变更：首个分歧拍起全部音符重算（offset 仅用于外部分片平移核对）；
    力度/踏板变更：只重演与变更拍区间相交的音符，其余逐音符不动。
    """
    lo, hi = changed_range
    recompute: list[str] = []
    shift: list[str] = []
    offset_ms = 0.0

    if tempo_changed and old_tempo_map is not None:
        divergence = None
        for note in notes:
            if abs(
                tempo_map.beat_to_ms(note.start_beat)
                - old_tempo_map.beat_to_ms(note.start_beat)
            ) > 5e-7:
                divergence = note.start_beat
                break
        divergence = lo if divergence is None else min(lo, divergence)
        offset_ms = round6(
            tempo_map.beat_to_ms(notes[-1].start_beat)
            - old_tempo_map.beat_to_ms(notes[-1].start_beat)
        )
        for note in notes:
            if note.start_beat + 1e-12 >= divergence:
                recompute.append(note.note_id)
            else:
                shift.append(note.note_id)
        return {
            "kind": kind,
            "range_beats": [round6(lo), round6(hi)],
            "divergence_beat": round6(divergence),
            "recompute_note_ids": recompute,
            "shift_note_ids": shift,
            "shifted_note_count": len(shift),
            "tail_offset_ms": offset_ms,
        }

    for note in notes:
        intersects = note.start_beat <= hi + 1e-12 and note.end_beat >= lo - 1e-12
        if intersects:
            recompute.append(note.note_id)
        else:
            shift.append(note.note_id)
    return {
        "kind": kind,
        "range_beats": [round6(lo), round6(hi)],
        "divergence_beat": None,
        "recompute_note_ids": recompute,
        "shift_note_ids": shift,
        "shifted_note_count": 0,
        "tail_offset_ms": 0.0,
    }


@dataclass
class TimelineState:
    version: int
    parent_version: int
    tempo: list
    dynamics: list
    pedals: list
    rendered_notes: dict  # part_id -> list[rendered note dict]
    rendered_pedals: list
    shards: dict  # f"{part}:{idx}" -> shard content dict (含 notes 与 hash)
    timeline_hash: str


class Engine:
    """版本化演奏时间轴引擎。incremental_* 路径只重渲染闭包内音符。"""

    def __init__(self, score: Score):
        self.score = score

    def full_rebuild(self, tempo=None, dynamics=None, pedals=None, version: int = 0) -> TimelineState:
        """整段重建（仅用于基线与核对；增量路径严禁调用本方法）。"""
        tempo = self.score.tempo if tempo is None else tempo
        dynamics = self.score.dynamics if dynamics is None else dynamics
        pedals = self.score.pedals if pedals is None else pedals
        validate_expressions(self.score, tempo, dynamics, pedals)
        tempo_map = TempoMap(tempo, self.score.total_beats)
        rendered_notes = {}
        shards: dict[str, dict] = {}
        for part_id in sorted(self.score.parts):
            rows = [
                render_note(note, tempo_map, dynamics, self.score.beat_unit_ms)
                for note in self.score.parts[part_id]
            ]
            rendered_notes[part_id] = rows
            part_shards = self._build_shards(part_id, rows)
            shards.update(part_shards)
        rendered_pedals = render_pedals(pedals, tempo_map)
        return self._assemble_state(
            version=version,
            parent_version=version - 1,
            tempo=tempo,
            dynamics=dynamics,
            pedals=pedals,
            rendered_notes=rendered_notes,
            rendered_pedals=rendered_pedals,
            shards=shards,
        )

    def _build_shards(self, part_id: str, rows: list[dict]) -> dict[str, dict]:
        grouped: dict[int, list[dict]] = {}
        for row in rows:
            grouped.setdefault(shard_index(row["start_beat"]), []).append(row)
        result = {}
        for idx in sorted(grouped):
            payload = {
                "part_id": part_id,
                "shard_index": idx,
                "beat_range": [idx * SHARD_BEATS, (idx + 1) * SHARD_BEATS],
                "notes": grouped[idx],
            }
            payload["content_hash"] = content_hash(self._hashable_shard(payload))
            result[f"{part_id}:{idx}"] = payload
        return result

    @staticmethod
    def _hashable_shard(payload: dict) -> dict:
        return {key: value for key, value in payload.items() if key != "content_hash"}

    def _assemble_state(
        self,
        version: int,
        parent_version: int,
        tempo,
        dynamics,
        pedals,
        rendered_notes,
        rendered_pedals,
        shards,
    ) -> TimelineState:
        ordered_hashes = [shards[key]["content_hash"] for key in sorted(shards)]
        timeline_hash = content_hash(
            {
                "schema": "timeline-hash/1",
                "shard_hashes": ordered_hashes,
                "pedals": rendered_pedals,
            }
        )
        return TimelineState(
            version=version,
            parent_version=parent_version,
            tempo=tempo,
            dynamics=dynamics,
            pedals=pedals,
            rendered_notes=rendered_notes,
            rendered_pedals=rendered_pedals,
            shards=shards,
            timeline_hash=timeline_hash,
        )

    def incremental_apply(
        self,
        parent: TimelineState,
        op: "Operation",
    ) -> tuple[TimelineState, dict]:
        """只重演失效闭包内音符；闭包外仅做平移或完全不动。

        本方法绝不调用 full_rebuild，且重算集合必须等于闭包清单。
        """
        if op.parent_version != parent.version:
            raise PipelineError(
                E_VERSION_REGRESS,
                "操作父版本与当前状态不一致",
                {"op_parent": op.parent_version, "current": parent.version},
            )
        validate_expressions(self.score, op.tempo, op.dynamics, op.pedals)

        old_tempo_map = TempoMap(parent.tempo, self.score.total_beats)
        new_tempo_map = TempoMap(op.tempo, self.score.total_beats)
        tempo_changed = parent.tempo != op.tempo
        dynamics_changed = parent.dynamics != op.dynamics
        pedals_changed = parent.pedals != op.pedals

        rendered_notes = {
            part_id: [dict(row) for row in rows]
            for part_id, rows in parent.rendered_notes.items()
        }
        all_new_notes: dict[str, list[dict]] = {}
        closure_parts: dict[str, dict] = {}

        for part_id in sorted(self.score.parts):
            notes = self.score.parts[part_id]
            closure = compute_closure(
                kind=op.kind,
                changed_range=op.changed_range,
                tempo_changed=tempo_changed,
                tempo_map=new_tempo_map,
                old_tempo_map=old_tempo_map,
                notes=notes,
            )
            closure_parts[part_id] = closure
            recompute_ids = set(closure["recompute_note_ids"])
            shift_ids = set(closure["shift_note_ids"])
            actual_ids = {note.note_id for note in notes}
            if recompute_ids | shift_ids != actual_ids or recompute_ids & shift_ids:
                raise PipelineError(
                    E_TIMELINE_HASH,
                    "失效闭包分区与实际音符集合不一致",
                    {"part_id": part_id},
                )

            rows = rendered_notes[part_id]
            note_by_id = {note.note_id: note for note in notes}
            new_rows: list[dict] = []
            recomputed: list[str] = []
            for row in rows:
                note_id = row["note_id"]
                if note_id in recompute_ids:
                    new_rows.append(
                        render_note(
                            note_by_id[note_id], new_tempo_map, op.dynamics,
                            self.score.beat_unit_ms,
                        )
                    )
                    recomputed.append(note_id)
                elif tempo_changed:
                    moved = dict(row)
                    note = note_by_id[note_id]
                    moved["start_ms"] = new_tempo_map.beat_to_ms_r(note.start_beat)
                    moved["end_ms"] = new_tempo_map.beat_to_ms_r(note.end_beat)
                    moved["duration_ms"] = round6(
                        (moved["end_ms"] - moved["start_ms"])
                        * ARTICULATION_DURATION_FACTOR[note.articulation]
                    )
                    new_rows.append(moved)
                else:
                    new_rows.append(dict(row))  # 力度/踏板变更：闭包外逐音符不变
            if sorted(recomputed) != sorted(recompute_ids):
                raise PipelineError(
                    E_TIMELINE_HASH,
                    "实际重算音符集合与闭包清单不一致",
                    {"part_id": part_id},
                )
            all_new_notes[part_id] = new_rows

        new_pedals = (
            render_pedals(op.pedals, new_tempo_map)
            if pedals_changed or tempo_changed
            else [dict(row) for row in parent.rendered_pedals]
        )

        touched_shards = self._rebuild_touched_shards(
            parent=parent,
            rendered_notes=all_new_notes,
            closure_parts=closure_parts,
            tempo_changed=tempo_changed,
        )
        shards = dict(parent.shards)
        shards.update(touched_shards)

        state = self._assemble_state(
            version=op.version,
            parent_version=op.parent_version,
            tempo=op.tempo,
            dynamics=op.dynamics,
            pedals=op.pedals,
            rendered_notes=all_new_notes,
            rendered_pedals=new_pedals,
            shards=shards,
        )
        closure_record = {
            "op_version": op.version,
            "kind": op.kind,
            "changed_range_beats": [round6(op.changed_range[0]), round6(op.changed_range[1])],
            "tempo_changed": tempo_changed,
            "dynamics_changed": dynamics_changed,
            "pedals_changed": pedals_changed,
            "parts": closure_parts,
            "recomputed_note_total": sum(
                len(item["recompute_note_ids"]) for item in closure_parts.values()
            ),
            "touched_shard_ids": sorted(touched_shards),
        }
        return state, closure_record

    def _rebuild_touched_shards(
        self,
        parent: TimelineState,
        rendered_notes: dict,
        closure_parts: dict,
        tempo_changed: bool,
    ) -> dict:
        touched: set[str] = set()
        for part_id, closure in closure_parts.items():
            for note_id in closure["recompute_note_ids"]:
                note = next(item for item in self.score.parts[part_id] if item.note_id == note_id)
                touched.add(f"{part_id}:{shard_index(note.start_beat)}")
            if tempo_changed:
                # 平移的闭包外音符也落在分片内（内容随之变化），同样重写其分片
                first_shift = closure["shift_note_ids"]
                for note_id in first_shift:
                    note = next(item for item in self.score.parts[part_id] if item.note_id == note_id)
                    touched.add(f"{part_id}:{shard_index(note.start_beat)}")
        rebuilt = {}
        for shard_id in sorted(touched):
            part_id, idx_text = shard_id.split(":")
            idx = int(idx_text)
            rows = [
                row
                for row in rendered_notes[part_id]
                if shard_index(row["start_beat"]) == idx
            ]
            payload = {
                "part_id": part_id,
                "shard_index": idx,
                "beat_range": [idx * SHARD_BEATS, (idx + 1) * SHARD_BEATS],
                "notes": rows,
            }
            payload["content_hash"] = content_hash(self._hashable_shard(payload))
            rebuilt[shard_id] = payload
        return rebuilt

    def replay_fragment(self, op_history: list["Operation"], version: int, shard_id: str) -> dict:
        """从操作历史重算指定历史版本的分片内容（不依赖当时内存状态）。"""
        state = self.full_rebuild(version=0)
        target = None
        for op in op_history:
            if op.version <= version:
                state, _ = self.incremental_apply(state, op)
            if op.version == version:
                target = state
        if target is None:
            raise PipelineError(E_BEAT_RANGE, "历史版本不存在", {"version": version})
        return target.shards[shard_id]


@dataclass(frozen=True)
class Operation:
    version: int
    parent_version: int
    kind: str  # tempo / dynamics / pedals
    changed_range: tuple[float, float]
    tempo: list
    dynamics: list
    pedals: list

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "parent_version": self.parent_version,
            "kind": self.kind,
            "changed_range": [round6(self.changed_range[0]), round6(self.changed_range[1])],
            "tempo": self.tempo,
            "dynamics": self.dynamics,
            "pedals": self.pedals,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Operation":
        return cls(
            version=int(data["version"]),
            parent_version=int(data["parent_version"]),
            kind=data["kind"],
            changed_range=(
                float(data["changed_range"][0]),
                float(data["changed_range"][1]),
            ),
            tempo=data["tempo"],
            dynamics=data["dynamics"],
            pedals=data["pedals"],
        )


def make_operation(
    version: int,
    parent: TimelineState,
    kind: str,
    tempo=None,
    dynamics=None,
    pedals=None,
    total_beats: float | None = None,
) -> Operation:
    tempo = parent.tempo if tempo is None else tempo
    dynamics = parent.dynamics if dynamics is None else dynamics
    pedals = parent.pedals if pedals is None else pedals
    if kind == "tempo":
        lo = next(
            (
                float(a["beat"])
                for a, b in zip(parent.tempo, tempo)
                if a != b
            ),
            min(float(p["beat"]) for p in tempo if p not in parent.tempo)
            if len(tempo) != len(parent.tempo)
            else 0.0,
        )
        changed_range = (lo, float(total_beats) if total_beats is not None else 1e12)
    else:
        old = parent.dynamics if kind == "dynamics" else parent.pedals
        new = dynamics if kind == "dynamics" else pedals
        old_set, new_set = {tuple(sorted(x.items())) for x in old}, {
            tuple(sorted(x.items())) for x in new
        }
        diffs = [dict(items) for items in (old_set ^ new_set)]
        lo = min(float(d["start_beat"]) for d in diffs)
        hi = max(float(d["end_beat"]) for d in diffs)
        changed_range = (lo, hi)
    return Operation(
        version=version,
        parent_version=parent.version,
        kind=kind,
        changed_range=changed_range,
        tempo=tempo,
        dynamics=dynamics,
        pedals=pedals,
    )
