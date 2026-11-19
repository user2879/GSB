"""ASS subtitle export and an independent file-only consistency verifier.

The exported ``.ass`` carries two complementary representations:

* a standard ``[Script Info]`` / ``[V4+ Styles]`` / ``[Events]`` section that
  any ASS renderer can display, including per-cue ``MarginV`` (the exact lane
  Y coordinate), ``Layer`` (the lane index) and a named style per lane/type;
* a deterministic ``[SWB_META]`` section (canonical JSON per cue) holding the
  exact frame interval, lane geometry and wrapped lines so a parser can restore
  the layout without centisecond rounding.

The verifier reads *only* the exported file and a snapshot file; it never
imports the in-memory objects it is checking.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import FPS
from .snapshot import LayoutSnapshot

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes
"""

STYLE_LINE = ("Style: {name},{font},{size},{primary},{secondary},{outline},"
              "{back},{bold},{italic},{underline},{strikeout},{scale_x},"
              "{scale_y},{spacing},{angle},{border_style},{outline_w},"
              "{shadow},{alignment},{margin_l},{margin_r},{margin_v},{encoding}")

DIALOGUE_PRIMARY = "&H00FFFFFF"   # white
DIALOGUE_OUTLINE = "&H00000000"   # black
AD_PRIMARY = "&H0000FFFF"         # yellow
AD_OUTLINE = "&H00000000"


def frame_to_timecode(frame: int, fps: int = FPS) -> str:
    centis = round(frame * 100 / fps)
    hours, rem = divmod(centis, 3600 * 100)
    minutes, rem = divmod(rem, 60 * 100)
    seconds, centiseconds = divmod(rem, 100)
    return f"{hours}:{minutes:02d}:{seconds:02d}.{centiseconds:02d}"


def timecode_to_frame(timecode: str, fps: int = FPS) -> int:
    hours, minutes, rest = timecode.split(":")
    seconds, centiseconds = rest.split(".")
    centis = (int(hours) * 3600 + int(minutes) * 60 + int(seconds)) * 100 \
        + int(centiseconds)
    return round(centis * fps / 100)


def _style_name(lane_index: int, is_ad: bool) -> str:
    return ("AD" if is_ad else "DLG") + f"_L{lane_index}"


def _style_rows(snapshot: LayoutSnapshot) -> list[str]:
    config = snapshot.layout.config
    seen: dict[str, dict[str, Any]] = {}
    for cue in snapshot.layout.cues:
        name = _style_name(cue.lane_index, cue.item.is_ad)
        if name in seen:
            continue
        seen[name] = {
            "name": name,
            "font": "sans-serif",
            "size": cue.wrap.font_size,
            "primary": AD_PRIMARY if cue.item.is_ad else DIALOGUE_PRIMARY,
            "secondary": "&H000000FF",
            "outline": AD_OUTLINE if cue.item.is_ad else DIALOGUE_OUTLINE,
            "back": "&H64000000",
            "bold": -1 if cue.item.is_ad else 0,
            "italic": 0, "underline": 0, "strikeout": 0,
            "scale_x": 100, "scale_y": 100, "spacing": 0, "angle": 0,
            "border_style": 1, "outline_w": 2, "shadow": 1,
            "alignment": 8 if cue.item.is_ad else 2,
            "margin_l": cue.side_margin, "margin_r": cue.side_margin,
            "margin_v": cue.y, "encoding": 1,
        }
    return [STYLE_LINE.format(**row) for _, row in sorted(seen.items())]


def export_ass(snapshot: LayoutSnapshot, path: str | Path) -> None:
    config = snapshot.layout.config
    lines: list[str] = [
        ASS_HEADER.format(width=config.width, height=config.height).rstrip(),
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
        "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
        "Alignment, MarginL, MarginR, MarginV, Encoding",
        *_style_rows(snapshot),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text",
    ]
    for cue in sorted(snapshot.layout.cues, key=lambda c: c.item.item_id):
        text = "\\N".join(cue.wrap.lines)
        lines.append(
            "Dialogue: {layer},{start},{end},{style},{name},{ml},{mr},{mv},,"
            "{text}".format(
                layer=cue.lane_index,
                start=frame_to_timecode(cue.item.start_frame),
                end=frame_to_timecode(cue.item.end_frame),
                style=_style_name(cue.lane_index, cue.item.is_ad),
                name=cue.item.item_id,
                ml=cue.side_margin, mr=cue.side_margin, mv=cue.y,
                text=text,
            ))
    lines.append("")
    lines.append("[SWB_META]")
    lines.append("# deterministic canonical JSON, one cue per line")
    for cue in sorted(snapshot.layout.cues, key=lambda c: c.item.item_id):
        record = {
            "item": cue.item.to_dict(),
            "lane_index": cue.lane_index,
            "x": cue.x,
            "y": cue.y,
            "width": cue.width,
            "side_margin": cue.side_margin,
            "wrap": cue.wrap.to_dict(),
        }
        lines.append(json.dumps(record, sort_keys=True, ensure_ascii=False,
                                separators=(",", ":")))
    Path(path).write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


# ----------------------------------------------------------------------
@dataclass
class VerificationReport:
    ok: bool
    first_mismatch: str | None
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {"ok": self.ok, "first_mismatch": self.first_mismatch,
                "details": self.details}


def parse_ass(path: str | Path) -> dict[str, Any]:
    """Parse an exported ASS file back into structured cue records."""
    text = Path(path).read_text(encoding="utf-8")
    sections: dict[str, list[str]] = {}
    current = None
    for raw in text.splitlines():
        if raw.startswith("[") and raw.endswith("]"):
            current = raw[1:-1]
            sections[current] = []
        elif current is not None:
            sections[current].append(raw)

    events: dict[str, dict[str, Any]] = {}
    for row in sections.get("Events", []):
        if not row.startswith("Dialogue:"):
            continue
        payload = row[len("Dialogue:"):].lstrip()
        parts = payload.split(",", 9)
        (layer, start, end, style, name, margin_l, margin_r, margin_v,
         _effect, text) = parts
        events[name] = {
            "lane_index": int(layer),
            "start_frame": timecode_to_frame(start),
            "end_frame": timecode_to_frame(end),
            "style": style,
            "y": int(margin_v),
            "side_margin": int(margin_l),
            "lines": text.split("\\N"),
        }

    meta: dict[str, dict[str, Any]] = {}
    for row in sections.get("SWB_META", []):
        if not row or row.startswith("#"):
            continue
        record = json.loads(row)
        meta[record["item"]["item_id"]] = record
    return {"events": events, "meta": meta}


def verify_export(ass_path: str | Path,
                  snapshot_path: str | Path) -> VerificationReport:
    """Independent check: file + snapshot only, field by field."""
    parsed = parse_ass(ass_path)
    snapshot = LayoutSnapshot.from_dict(
        json.loads(Path(snapshot_path).read_text(encoding="utf-8")))
    expected = {cue.item.item_id: cue for cue in snapshot.layout.cues}

    details: dict[str, Any] = {
        "snapshot_cues": len(expected),
        "ass_meta_cues": len(parsed["meta"]),
        "ass_event_cues": len(parsed["events"]),
    }
    if len(parsed["meta"]) != len(expected) or len(parsed["events"]) != len(expected):
        return VerificationReport(
            False, "cue count mismatch (count equality alone is never enough)",
            details)

    for item_id in sorted(expected):
        cue = expected[item_id]
        meta = parsed["meta"].get(item_id)
        event = parsed["events"].get(item_id)
        if meta is None or event is None:
            return VerificationReport(False, f"{item_id}: missing in ASS",
                                      details)
        checks = {
            "lane_index": (meta["lane_index"], cue.lane_index),
            "y": (meta["y"], cue.y),
            "x": (meta["x"], cue.x),
            "width": (meta["width"], cue.width),
            "side_margin": (meta["side_margin"], cue.side_margin),
            "start_frame": (meta["item"]["start_frame"],
                            cue.item.start_frame),
            "end_frame": (meta["item"]["end_frame"], cue.item.end_frame),
            "text": (meta["item"]["text"], cue.item.text),
            "is_ad": (meta["item"]["is_ad"], cue.item.is_ad),
            "priority": (meta["item"]["priority"], cue.item.priority),
            "font_size": (meta["wrap"]["font_size"], cue.wrap.font_size),
            "lines": (tuple(meta["wrap"]["lines"]), cue.wrap.lines),
            "event_start_frame": (event["start_frame"],
                                  cue.item.start_frame),
            "event_end_frame": (event["end_frame"], cue.item.end_frame),
            "event_lane": (event["lane_index"], cue.lane_index),
            "event_y": (event["y"], cue.y),
            "event_lines": (tuple(event["lines"]), cue.wrap.lines),
        }
        for field_name, (actual, wanted) in checks.items():
            if actual != wanted:
                details["item_id"] = item_id
                details["field"] = field_name
                details["actual"] = actual
                details["expected"] = wanted
                return VerificationReport(
                    False,
                    f"{item_id}: field '{field_name}' actual={actual!r} "
                    f"expected={wanted!r}",
                    details)
    details["matched_cues"] = len(expected)
    return VerificationReport(True, None, details)
