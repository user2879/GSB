"""确定性乐谱与演奏表达事件生成器。

随机源固定为 numpy.random.Generator(PCG64(seed))，所有抽样顺序固定，
不使用集合迭代、不使用时间戳；同一配置两次生成逐字节一致。
"""

from __future__ import annotations

import numpy as np

from . import BEAT_UNIT_MS
from .model import Note, Score, content_hash, round6, write_json

ARTICULATIONS = ("legato", "normal", "staccato")


def generate_score(
    seed: int = 20261119,
    part_count: int = 4,
    bars_per_part: int = 16,
    beats_per_bar: int = 4,
    beat_unit_ms: float = BEAT_UNIT_MS,
) -> Score:
    rng = np.random.Generator(np.random.PCG64(seed))
    total_beats = float(bars_per_part * beats_per_bar)

    parts: dict[str, list[Note]] = {}
    event_index = 0
    for part_index in range(part_count):
        part_id = f"P{part_index:02d}"
        notes: list[Note] = []
        cursor = 0.0
        while cursor < total_beats - 1e-12:
            duration_choices = (0.5, 1.0, 1.0, 1.5, 2.0, 3.0, 4.0)
            duration = float(rng.choice(duration_choices))
            if cursor + duration > total_beats:
                duration = round6(total_beats - cursor)
            pitch = int(rng.integers(48, 85))
            velocity = int(rng.integers(48, 105))
            articulation = ARTICULATIONS[int(rng.integers(0, len(ARTICULATIONS)))]
            notes.append(
                Note(
                    note_id=f"{part_id}-N{len(notes):04d}",
                    pitch=pitch,
                    start_beat=round6(cursor),
                    duration_beats=round6(duration),
                    velocity=velocity,
                    articulation=articulation,
                )
            )
            event_index += 1
            cursor = round6(cursor + duration)
        parts[part_id] = notes

    # 速度曲线关键点：拍号严格递增，覆盖 0..total_beats
    tempo = [{"beat": 0.0, "bpm": 96.0}]
    for bar in (4, 8, 12):
        tempo.append(
            {
                "beat": round6(bar * beats_per_bar),
                "bpm": round6(float(rng.integers(72, 132))),
            }
        )
    tempo.append({"beat": round6(total_beats), "bpm": 100.0})

    # 力度区间（渐强/渐弱）：互不重叠、按拍排序
    dynamics = []
    boundaries = [0.0] + [round6(b * beats_per_bar) for b in (4, 8, 12)] + [total_beats]
    for lo, hi in zip(boundaries, boundaries[1:]):
        gain = round6(float(rng.uniform(0.8, 1.0)))
        end_gain = round6(float(rng.uniform(0.85, 1.15)))
        dynamics.append(
            {"start_beat": lo, "end_beat": hi, "gain": gain, "end_gain": end_gain}
        )

    # 踏板区间：每声部 2 个不交叉区间
    pedals = []
    for part_index in range(part_count):
        part_id = f"P{part_index:02d}"
        first_start = round6(float(rng.integers(0, 2)) * beats_per_bar)
        first_end = round6(first_start + 2 * beats_per_bar)
        second_start = round6(first_end + 2 * beats_per_bar)
        second_end = round6(min(second_start + 2 * beats_per_bar, total_beats))
        pedals.append(
            {"part_id": part_id, "start_beat": first_start, "end_beat": first_end}
        )
        if second_end > second_start:
            pedals.append(
                {"part_id": part_id, "start_beat": second_start, "end_beat": second_end}
            )

    return Score(
        seed=seed,
        parts=parts,
        tempo=tempo,
        dynamics=dynamics,
        pedals=pedals,
        total_beats=total_beats,
        beat_unit_ms=beat_unit_ms,
    )


def write_score_bundle(score: Score, out_dir: str) -> dict:
    """写乐谱、逐声部逻辑时间轴真值清单与生成报告，返回报告字典。"""
    out_dir = str(out_dir)
    score_path = f"{out_dir}/score.json"
    write_json(score_path, score.to_dict())

    note_total = 0
    part_truth = {}
    for part_id in sorted(score.parts):
        notes = score.parts[part_id]
        note_total += len(notes)
        part_truth[part_id] = {
            "part_id": part_id,
            "beat_unit_ms": score.beat_unit_ms,
            "total_beats": round6(score.total_beats),
            "notes": [note.to_dict() for note in notes],
            "pedals": [p for p in score.pedals if p["part_id"] == part_id],
        }
        write_json(f"{out_dir}/truth_{part_id}.json", part_truth[part_id])

    report = {
        "schema": score.to_dict()["schema"],
        "seed": score.seed,
        "part_count": len(score.parts),
        "note_event_total": note_total,
        "tempo_point_total": len(score.tempo),
        "dynamics_event_total": len(score.dynamics),
        "pedal_event_total": len(score.pedals),
        "total_beats": round6(score.total_beats),
        "beat_unit_ms": score.beat_unit_ms,
        "beat_unit_meaning": "一个四分音符逻辑拍的参考毫秒口径，实际毫秒由速度曲线插值决定",
        "score_file": "score.json",
        "score_sha256": content_hash(score.to_dict()),
        "truth_files": [f"truth_{pid}.json" for pid in sorted(score.parts)],
    }
    write_json(f"{out_dir}/generation_report.json", report)
    return report
