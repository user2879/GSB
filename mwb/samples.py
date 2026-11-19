"""内置确定性样本生成器。

固定常量、无随机源，重复生成字节级一致，满足“本地样本由程序按
确定性规则生成”的要求。
"""
from __future__ import annotations

import os

from .atomicio import atomic_write_json

TICKS_PER_BEAT = 480
CHUNK_SIZE = 960


def generate_score() -> dict:
    # 两个声部，各 8 个音符，共 16 拍；两声部在中段刻意跨声部重叠
    notes = []
    melody = [60, 62, 64, 65, 67, 69, 71, 72]
    bass = [36, 40, 43, 47, 36, 40, 43, 48]
    for i, pitch in enumerate(melody):
        notes.append(
            {
                "note_id": f"M{i+1:02d}",
                "pitch": pitch,
                "start_beat": i * 2,
                "duration_beats": 2,
                "voice": "melody",
                "velocity": 64 + (i % 3) * 8,
                "articulation": "legato" if i % 2 == 0 else "normal",
                "stretch_rule": "default",
            }
        )
    for i, pitch in enumerate(bass):
        notes.append(
            {
                "note_id": f"B{i+1:02d}",
                "pitch": pitch,
                "start_beat": i * 2 + 1,
                "duration_beats": 2,
                "voice": "bass",
                "velocity": 48 + (i % 2) * 10,
                "articulation": "staccato" if i in (2, 6) else "normal",
                "stretch_rule": "broad" if i >= 4 else "default",
            }
        )
    return {"title": "deterministic-demo", "ticks_per_beat": TICKS_PER_BEAT, "notes": notes}


def generate_expression() -> dict:
    return {
        "tempo_curve": [
            {"beat": 0, "bpm": 96.0},
            {"beat": 4, "bpm": 120.0},
            {"beat": 10, "bpm": 108.0},
            {"beat": 16, "bpm": 132.0},
        ],
        "velocity_curve": [
            {"x": 0, "y": 10},
            {"x": 64, "y": 70},
            {"x": 127, "y": 120},
        ],
        "stretch_rules": {"broad": 1.12, "tight": 0.92},
        "articulation_gate": {"legato": 0.98, "normal": 0.9, "staccato": 0.55},
        "pedal_segments": [
            {"pedal_id": "P1", "start_beat": 0.0, "end_beat": 4.0},
            {"pedal_id": "P2", "start_beat": 8.0, "end_beat": 12.0},
        ],
        "default_stretch": 1.0,
    }


def generate_edit_script(concurrency: int = 2, memory_budget_chunks: int = 2) -> dict:
    return {
        "concurrency": concurrency,
        "memory_budget_chunks": memory_budget_chunks,
        "edits": [
            {"op": "tempo_upsert", "args": {"beat": 6, "bpm": 132.0}},
            {"op": "velocity_upsert", "args": {"x": 96, "y": 110}},
            {"op": "stretch_set", "args": {"name": "broad", "factor": 1.18}},
            {"op": "articulation_set", "args": {"name": "staccato", "gate": 0.45}},
            {"op": "note_articulation_set", "args": {"note_id": "M03", "articulation": "staccato"}},
            {"op": "pedal_add", "args": {"pedal_id": "P3", "start_beat": 13.0, "end_beat": 15.0}},
            {"op": "pedal_remove", "args": {"pedal_id": "P1"}},
            {"op": "tempo_delete", "args": {"beat": 10}},
        ],
    }


def generate_workspace(samples_dir: str) -> dict:
    os.makedirs(samples_dir, exist_ok=True)
    paths = {
        "score": os.path.join(samples_dir, "score.json"),
        "expression": os.path.join(samples_dir, "expression.json"),
        "edits": os.path.join(samples_dir, "edits.json"),
    }
    atomic_write_json(paths["score"], generate_score())
    atomic_write_json(paths["expression"], generate_expression())
    atomic_write_json(paths["edits"], generate_edit_script())
    return paths
