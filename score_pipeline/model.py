"""数据模型、确定性 JSON 序列化与内容哈希。

所有落盘 JSON 都走 canonical_bytes：sort_keys + 紧凑分隔符，
因此哈希与文件字节不依赖字典遍历顺序，跨运行/跨平台逐字节一致。
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from . import SCHEMA


def canonical_bytes(obj) -> bytes:
    """确定性 JSON 编码：键排序、紧凑分隔、UTF-8、禁止 NaN。"""
    return json.dumps(
        obj,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def content_hash(obj) -> str:
    return hashlib.sha256(canonical_bytes(obj)).hexdigest()


def write_json(path: str | Path, obj) -> None:
    """原子写确定性 JSON：同目录临时文件 + fsync + os.replace。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = canonical_bytes(obj)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def read_json(path: str | Path):
    with open(path, "rb") as handle:
        return json.loads(handle.read().decode("utf-8"))


def round6(value: float) -> float:
    return round(float(value), 6)


@dataclass(frozen=True)
class Note:
    note_id: str
    pitch: int
    start_beat: float
    duration_beats: float
    velocity: int
    articulation: str

    @property
    def end_beat(self) -> float:
        return self.start_beat + self.duration_beats

    def to_dict(self) -> dict:
        data = asdict(self)
        data["end_beat"] = round6(self.end_beat)
        return data


@dataclass(frozen=True)
class Score:
    seed: int
    parts: dict  # part_id -> list[Note]
    tempo: list  # list[{"beat","bpm"}]
    dynamics: list  # list[{"start_beat","end_beat","gain","end_gain"}]
    pedals: list  # list[{"part_id","start_beat","end_beat"}]
    total_beats: float
    beat_unit_ms: float

    def all_notes(self) -> list:
        notes = []
        for part_id in sorted(self.parts):
            notes.extend((part_id, note) for note in self.parts[part_id])
        return notes

    def to_dict(self) -> dict:
        return {
            "schema": SCHEMA,
            "seed": self.seed,
            "total_beats": round6(self.total_beats),
            "beat_unit_ms": self.beat_unit_ms,
            "parts": [
                {"part_id": part_id, "notes": [n.to_dict() for n in self.parts[part_id]]}
                for part_id in sorted(self.parts)
            ],
            "tempo": self.tempo,
            "dynamics": self.dynamics,
            "pedals": self.pedals,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Score":
        parts = {}
        for part in data["parts"]:
            parts[part["part_id"]] = [
                Note(**{key: value for key, value in n.items() if key != "end_beat"})
                for n in part["notes"]
            ]
        return cls(
            seed=data["seed"],
            parts=parts,
            tempo=data["tempo"],
            dynamics=data["dynamics"],
            pedals=data["pedals"],
            total_beats=float(data["total_beats"]),
            beat_unit_ms=float(data["beat_unit_ms"]),
        )
