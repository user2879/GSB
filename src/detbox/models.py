"""Unified intermediate representation (UIR) and shared data classes."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

FORMAT_V1 = "v1"
FORMAT_V2 = "v2"

# Fixed-width v1 column layout (total line width 46 chars + newline).
V1_LAYOUT = {
    "sequence": (0, 10),
    "frame": (10, 15),
    "class_code": (15, 18),
    "x1": (18, 23),
    "y1": (23, 28),
    "x2": (28, 33),
    "y2": (33, 38),
    "score": (38, 45),
    "missing": (45, 46),
}
V1_LINE_WIDTH = 46


@dataclass(frozen=True)
class ClassTable:
    """Bidirectional mapping between integer class codes and names."""

    code_to_name: dict[int, str]

    @property
    def name_to_code(self) -> dict[str, int]:
        return {name: code for code, name in self.code_to_name.items()}

    def name_for(self, code: int) -> str:
        if code not in self.code_to_name:
            raise KeyError(code)
        return self.code_to_name[code]

    def code_for(self, name: str) -> int:
        table = self.name_to_code
        if name not in table:
            raise KeyError(name)
        return table[name]

    def to_dict(self) -> dict[str, Any]:
        return {"code_to_name": {str(k): v for k, v in sorted(self.code_to_name.items())}}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ClassTable":
        return cls({int(k): v for k, v in data["code_to_name"].items()})


@dataclass
class Candidate:
    """One candidate box in the UIR.

    Float geometry/score are the canonical values.  For v1-sourced rows the
    exact integers are retained so round trips are lossless; v2 rows carry a
    precision label and an optional high-precision reference score.
    """

    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_code: int
    source: str = FORMAT_V1
    precision_label: str = "int-exact"
    ref_score: float | None = None
    int_box: tuple[int, int, int, int] | None = None
    int_score: int | None = None
    missing: int = 0

    def box_key(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    def int_box_tuple(self) -> tuple[int, int, int, int]:
        if self.int_box is not None:
            return self.int_box
        return tuple(half_up(v) for v in (self.x1, self.y1, self.x2, self.y2))

    def int_score_value(self) -> int:
        if self.int_score is not None:
            return self.int_score
        return half_up(self.score * 1000.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Candidate":
        int_box = data.get("int_box")
        return cls(
            x1=float(data["x1"]),
            y1=float(data["y1"]),
            x2=float(data["x2"]),
            y2=float(data["y2"]),
            score=float(data["score"]),
            class_code=int(data["class_code"]),
            source=data.get("source", FORMAT_V1),
            precision_label=data.get("precision_label", "int-exact"),
            ref_score=None if data.get("ref_score") is None else float(data["ref_score"]),
            int_box=None if int_box is None else tuple(int(v) for v in int_box),
            int_score=None if data.get("int_score") is None else int(data["int_score"]),
            missing=int(data.get("missing", 0)),
        )


@dataclass
class FrameCandidates:
    sequence: str
    frame: int
    width: int
    height: int
    candidates: list[Candidate] = field(default_factory=list)
    source_format: str = FORMAT_V1

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "frame": self.frame,
            "width": self.width,
            "height": self.height,
            "source_format": self.source_format,
            "candidates": [c.to_dict() for c in self.candidates],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FrameCandidates":
        return cls(
            sequence=data["sequence"],
            frame=int(data["frame"]),
            width=int(data["width"]),
            height=int(data["height"]),
            source_format=data.get("source_format", FORMAT_V1),
            candidates=[Candidate.from_dict(c) for c in data.get("candidates", [])],
        )


@dataclass
class Track:
    track_id: str
    class_code: int
    sequence: str
    frames: list[int] = field(default_factory=list)
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    scores: list[int] = field(default_factory=list)
    center_x: float = 0.0
    center_y: float = 0.0
    vx: float = 0.0
    vy: float = 0.0
    last_seen: int = 0
    closed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "class_code": self.class_code,
            "sequence": self.sequence,
            "frames": list(self.frames),
            "boxes": [list(b) for b in self.boxes],
            "scores": list(self.scores),
            "center_x": self.center_x,
            "center_y": self.center_y,
            "vx": self.vx,
            "vy": self.vy,
            "last_seen": self.last_seen,
            "closed": self.closed,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Track":
        return cls(
            track_id=data["track_id"],
            class_code=int(data["class_code"]),
            sequence=data["sequence"],
            frames=[int(v) for v in data["frames"]],
            boxes=[tuple(int(x) for x in b) for b in data["boxes"]],
            scores=[int(v) for v in data["scores"]],
            center_x=float(data.get("center_x", 0.0)),
            center_y=float(data.get("center_y", 0.0)),
            vx=float(data.get("vx", 0.0)),
            vy=float(data.get("vy", 0.0)),
            last_seen=int(data.get("last_seen", 0)),
            closed=bool(data.get("closed", False)),
        )


@dataclass
class FailureRecord:
    sequence: str
    frame: int
    stage: str
    exception_type: str
    code: str
    message: str
    injection_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def half_up(value: float) -> int:
    """Deterministic half-away-from-zero rounding used by v2 -> v1."""
    if value >= 0:
        return int(value + 0.5)
    return -int(-value + 0.5)
