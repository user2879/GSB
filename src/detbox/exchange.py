"""v1 fixed-width text and v2 JSON candidate exchange formats.

v1 line layout (46 chars + newline):
  sequence[0:10] frame[10:15] class[15:18] x1[18:23] y1[23:28]
  x2[28:33] y2[33:38] score_per_mille[38:45] missing[45:46]
"""
from __future__ import annotations

import json
from pathlib import Path

from .errors import CandidateFormatError, UnknownClassError
from .models import (
    Candidate,
    ClassTable,
    FORMAT_V1,
    FORMAT_V2,
    FrameCandidates,
    V1_LAYOUT,
    V1_LINE_WIDTH,
    half_up,
)
from .util import atomic_write_json, atomic_write_text, read_json


def _slice(line: str, field: str) -> str:
    start, end = V1_LAYOUT[field]
    return line[start:end]


def parse_v1_line(line: str, classes: ClassTable, width: int, height: int) -> Candidate:
    if len(line) != V1_LINE_WIDTH:
        raise CandidateFormatError(
            f"v1 line width {len(line)} != {V1_LINE_WIDTH}: {line!r}"
        )
    try:
        frame = int(_slice(line, "frame"))
        class_code = int(_slice(line, "class_code"))
        x1 = int(_slice(line, "x1"))
        y1 = int(_slice(line, "y1"))
        x2 = int(_slice(line, "x2"))
        y2 = int(_slice(line, "y2"))
        score_mille = int(_slice(line, "score"))
        missing = int(_slice(line, "missing"))
    except ValueError as exc:
        raise CandidateFormatError(f"non-integer v1 field: {exc}") from exc
    sequence = _slice(line, "sequence").strip()
    if not sequence:
        raise CandidateFormatError("empty sequence field")
    if missing not in (0, 1):
        raise CandidateFormatError(f"missing flag must be 0/1, got {missing}")
    if class_code not in classes.code_to_name:
        raise UnknownClassError(f"class code {class_code} not in class table")
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise CandidateFormatError(
            f"v1 box out of bounds: ({x1},{y1},{x2},{y2}) vs {width}x{height}"
        )
    if not (0 <= score_mille <= 1000):
        raise CandidateFormatError(f"score per mille out of range: {score_mille}")
    return Candidate(
        x1=float(x1),
        y1=float(y1),
        x2=float(x2),
        y2=float(y2),
        score=score_mille / 1000.0,
        class_code=class_code,
        source=FORMAT_V1,
        precision_label="int-exact",
        ref_score=None,
        int_box=(x1, y1, x2, y2),
        int_score=score_mille,
        missing=missing,
    )


def parse_v1_text(
    text: str, classes: ClassTable, width: int, height: int
) -> tuple[list[FrameCandidates], list[dict]]:
    """Parse all v1 lines; bad lines go to the failure list."""
    frames: dict[tuple[str, int], FrameCandidates] = {}
    failures: list[dict] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            candidate = parse_v1_line(raw, classes, width, height)
        except (CandidateFormatError, UnknownClassError) as exc:
            failures.append(
                {
                    "line": lineno,
                    "stage": "parse_v1",
                    "exception_type": type(exc).__name__,
                    "code": exc.code,
                    "message": str(exc),
                    "raw": raw,
                }
            )
            continue
        sequence = raw[V1_LAYOUT["sequence"][0] : V1_LAYOUT["sequence"][1]].strip()
        frame = int(raw[V1_LAYOUT["frame"][0] : V1_LAYOUT["frame"][1]])
        key = (sequence, frame)
        if key not in frames:
            frames[key] = FrameCandidates(
                sequence=sequence,
                frame=frame,
                width=width,
                height=height,
                source_format=FORMAT_V1,
            )
        frames[key].candidates.append(candidate)
    return list(frames.values()), failures


def candidate_to_v1_line(candidate: Candidate, sequence: str, frame: int) -> str:
    x1, y1, x2, y2 = candidate.int_box_tuple()
    score_mille = candidate.int_score_value()
    parts = {
        "sequence": sequence,
        "frame": frame,
        "class_code": candidate.class_code,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "score": score_mille,
        "missing": candidate.missing,
    }
    line = (
        parts["sequence"].ljust(10)[:10]
        + str(parts["frame"]).rjust(5)
        + str(parts["class_code"]).rjust(3)
        + str(parts["x1"]).rjust(5)
        + str(parts["y1"]).rjust(5)
        + str(parts["x2"]).rjust(5)
        + str(parts["y2"]).rjust(5)
        + str(parts["score"]).rjust(7)
        + str(parts["missing"]).rjust(1)
    )
    if len(line) != V1_LINE_WIDTH:
        raise CandidateFormatError(f"serialized v1 line width {len(line)} != {V1_LINE_WIDTH}")
    return line


def serialize_v1(frame_lists: list[FrameCandidates]) -> str:
    lines: list[str] = []
    for frame in sorted(frame_lists, key=lambda f: (f.sequence, f.frame)):
        for index, candidate in enumerate(frame.candidates):
            lines.append(candidate_to_v1_line(candidate, frame.sequence, frame.frame))
    return "\n".join(lines) + ("\n" if lines else "")


def parse_v2_document(
    document: dict, classes: ClassTable, width: int, height: int
) -> tuple[list[FrameCandidates], list[dict]]:
    failures: list[dict] = []
    if not isinstance(document, dict) or "sequences" not in document:
        raise CandidateFormatError("v2 document must be an object with 'sequences'")
    frames: list[FrameCandidates] = []
    for seq_entry in document["sequences"]:
        if not isinstance(seq_entry, dict) or "sequence" not in seq_entry or "frames" not in seq_entry:
            raise CandidateFormatError("v2 sequence entry needs 'sequence' and 'frames'")
        sequence = str(seq_entry["sequence"])
        for frame_entry in seq_entry["frames"]:
            if not isinstance(frame_entry, dict) or "frame" not in frame_entry:
                raise CandidateFormatError("v2 frame entry needs a 'frame' field")
            frame_no = int(frame_entry["frame"])
            fc = FrameCandidates(
                sequence=sequence,
                frame=frame_no,
                width=width,
                height=height,
                source_format=FORMAT_V2,
            )
            for cand_entry in frame_entry.get("candidates", []):
                try:
                    fc.candidates.append(_parse_v2_candidate(cand_entry, classes, width, height))
                except (CandidateFormatError, UnknownClassError) as exc:
                    failures.append(
                        {
                            "line": frame_no,
                            "stage": "parse_v2",
                            "exception_type": type(exc).__name__,
                            "code": exc.code,
                            "message": str(exc),
                            "raw": cand_entry,
                        }
                    )
            frames.append(fc)
    return frames, failures


def _parse_v2_candidate(
    entry: dict, classes: ClassTable, width: int, height: int
) -> Candidate:
    required = ["box", "score", "class_name", "precision_label"]
    missing = [key for key in required if key not in entry]
    if missing:
        raise CandidateFormatError(f"v2 candidate missing fields: {missing}")
    box = entry["box"]
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise CandidateFormatError("v2 box must be a 4-number array")
    x1, y1, x2, y2 = (float(v) for v in box)
    score = float(entry["score"])
    if not (0.0 <= score <= 1.0):
        raise CandidateFormatError(f"v2 score out of [0,1]: {score}")
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise CandidateFormatError(
            f"v2 box out of bounds: ({x1},{y1},{x2},{y2}) vs {width}x{height}"
        )
    class_name = str(entry["class_name"])
    if class_name not in classes.name_to_code:
        raise UnknownClassError(f"class name {class_name!r} not in class table")
    ref_score = entry.get("ref_score")
    if ref_score is not None:
        ref_score = float(ref_score)
        if not (0.0 <= ref_score <= 1.0):
            raise CandidateFormatError(f"v2 ref_score out of [0,1]: {ref_score}")
    return Candidate(
        x1=x1,
        y1=y1,
        x2=x2,
        y2=y2,
        score=score,
        class_code=classes.code_for(class_name),
        source=FORMAT_V2,
        precision_label=str(entry["precision_label"]),
        ref_score=ref_score,
        int_box=None,
        int_score=None,
        missing=int(bool(entry.get("missing", False))),
    )


def serialize_v2_document(frame_lists: list[FrameCandidates], classes: ClassTable) -> dict:
    grouped: dict[str, list[FrameCandidates]] = {}
    for frame in frame_lists:
        grouped.setdefault(frame.sequence, []).append(frame)
    sequences = []
    for sequence in sorted(grouped):
        seq_frames = []
        for frame in sorted(grouped[sequence], key=lambda f: f.frame):
            candidates = []
            for candidate in frame.candidates:
                candidates.append(
                    {
                        "box": [candidate.x1, candidate.y1, candidate.x2, candidate.y2],
                        "score": candidate.score,
                        "class_name": classes.name_for(candidate.class_code),
                        "precision_label": candidate.precision_label,
                        "ref_score": candidate.ref_score,
                        "missing": bool(candidate.missing),
                    }
                )
            seq_frames.append({"frame": frame.frame, "candidates": candidates})
        sequences.append({"sequence": sequence, "frames": seq_frames})
    return {"format": FORMAT_V2, "sequences": sequences}


# ---------------------------------------------------------------------------
# Bidirectional conversion
# ---------------------------------------------------------------------------


def v1_to_v2(frames: list[FrameCandidates]) -> list[FrameCandidates]:
    """Lossless upgrade: integers -> exact floats, codes -> class names."""
    converted: list[FrameCandidates] = []
    for frame in frames:
        new_candidates = []
        for candidate in frame.candidates:
            new_candidates.append(
                Candidate(
                    x1=float(candidate.int_box[0]),
                    y1=float(candidate.int_box[1]),
                    x2=float(candidate.int_box[2]),
                    y2=float(candidate.int_box[3]),
                    score=candidate.int_score / 1000.0,
                    class_code=candidate.class_code,
                    source=FORMAT_V2,
                    precision_label="upgraded-from-v1",
                    ref_score=None,
                    int_box=candidate.int_box,
                    int_score=candidate.int_score,
                    missing=candidate.missing,
                )
            )
        converted.append(
            FrameCandidates(
                sequence=frame.sequence,
                frame=frame.frame,
                width=frame.width,
                height=frame.height,
                candidates=new_candidates,
                source_format=FORMAT_V2,
            )
        )
    return converted


def v2_to_v1(
    frames: list[FrameCandidates], classes: ClassTable
) -> tuple[list[FrameCandidates], dict]:
    """Downgrade with a sidecar capturing all dropped precision."""
    converted: list[FrameCandidates] = []
    sidecar: dict = {
        "sidecar_version": 1,
        "source_format": FORMAT_V2,
        "target_format": FORMAT_V1,
        "class_mapping": {
            classes.name_for(code): code for code in sorted(classes.code_to_name)
        },
        "records": [],
    }
    for frame in frames:
        new_candidates = []
        for index, candidate in enumerate(frame.candidates):
            rounded = tuple(half_up(v) for v in (candidate.x1, candidate.y1, candidate.x2, candidate.y2))
            score_mille = half_up(candidate.score * 1000.0)
            new_candidates.append(
                Candidate(
                    x1=float(rounded[0]),
                    y1=float(rounded[1]),
                    x2=float(rounded[2]),
                    y2=float(rounded[3]),
                    score=score_mille / 1000.0,
                    class_code=candidate.class_code,
                    source=FORMAT_V1,
                    precision_label="int-exact",
                    ref_score=None,
                    int_box=rounded,
                    int_score=score_mille,
                    missing=candidate.missing,
                )
            )
            sidecar["records"].append(
                {
                    "sequence": frame.sequence,
                    "frame": frame.frame,
                    "candidate_index": index,
                    "coordinate_residual": [
                        rounded[0] - candidate.x1,
                        rounded[1] - candidate.y1,
                        rounded[2] - candidate.x2,
                        rounded[3] - candidate.y2,
                    ],
                    "score_residual": score_mille / 1000.0 - candidate.score,
                    "dropped_precision_label": candidate.precision_label,
                    "dropped_ref_score": candidate.ref_score,
                    "class_name": classes.name_for(candidate.class_code),
                    "class_code": candidate.class_code,
                }
            )
        converted.append(
            FrameCandidates(
                sequence=frame.sequence,
                frame=frame.frame,
                width=frame.width,
                height=frame.height,
                candidates=new_candidates,
                source_format=FORMAT_V1,
            )
        )
    return converted, sidecar


def load_exchange_file(
    path: str | Path,
    fmt: str,
    classes: ClassTable,
    width: int,
    height: int,
) -> tuple[list[FrameCandidates], list[dict]]:
    path = Path(path)
    if fmt == FORMAT_V1:
        text = path.read_text(encoding="utf-8")
        return parse_v1_text(text, classes, width, height)
    if fmt == FORMAT_V2:
        try:
            document = read_json(path)
        except json.JSONDecodeError as exc:
            raise CandidateFormatError(f"invalid v2 JSON: {exc}") from exc
        return parse_v2_document(document, classes, width, height)
    raise CandidateFormatError(f"unknown exchange format: {fmt}")


def write_exchange_file(
    path: str | Path,
    fmt: str,
    frames: list[FrameCandidates],
    classes: ClassTable,
) -> None:
    if fmt == FORMAT_V1:
        atomic_write_text(path, serialize_v1(frames))
    elif fmt == FORMAT_V2:
        atomic_write_json(path, serialize_v2_document(frames, classes))
    else:
        raise CandidateFormatError(f"unknown exchange format: {fmt}")
