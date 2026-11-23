"""v1 fixed-width text exchange format.

Line layout (1-based character positions)::

    01..06  segment index        width 6
    07..14  start frame          width 8
    15..22  end frame            width 8
    23..28  intensity permille   width 6
    29..34  target timbre code   width 6
    35..36  missing flag         width 2  (spaces = present)

Any short line, non-numeric numeric field or non-blank missing flag produces
``E_FIELD_TRUNCATED`` (or ``E_UNKNOWN_TIMBRE`` for an unknown code) carrying the
physical line number and the offending field name.
"""
from __future__ import annotations

from pathlib import Path

from .config import PipelineConfig
from .errors import E_FIELD_TRUNCATED, E_UNKNOWN_TIMBRE, FailureLog, PipelineError
from .model import Segment

FIELD_WIDTHS = {
    "segment": 6,
    "start_frame": 8,
    "end_frame": 8,
    "intensity_permille": 6,
    "timbre_code": 6,
    "missing_flag": 2,
}
FIELD_ORDER = list(FIELD_WIDTHS)
TOTAL_WIDTH = sum(FIELD_WIDTHS.values())


def _field_at_position(position: int) -> str:
    cursor = 0
    for name in FIELD_ORDER:
        if position < cursor + FIELD_WIDTHS[name]:
            return name
        cursor += FIELD_WIDTHS[name]
    return FIELD_ORDER[-1]


def _parse_int(token: str) -> int:
    token = token.strip()
    if not token or not token.lstrip("-").isdigit():
        raise ValueError("not an integer field")
    return int(token)


def parse_v1(path: str | Path, config: PipelineConfig,
             failures: FailureLog) -> list[Segment]:
    segments: list[Segment] = []
    text = Path(path).read_text(encoding="utf-8")
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\r")
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if len(line) < TOTAL_WIDTH:
            failures.add(
                E_FIELD_TRUNCATED, "load", line=line_no,
                field=_field_at_position(len(line)),
                message=f"line width {len(line)} < {TOTAL_WIDTH}",
            )
            continue
        offsets, cursor = {}, 0
        for name in FIELD_ORDER:
            offsets[name] = (cursor, cursor + FIELD_WIDTHS[name])
            cursor += FIELD_WIDTHS[name]
        tokens = {name: line[a:b] for name, (a, b) in offsets.items()}
        values: dict = {}
        bad_field = ""
        for name in ("segment", "start_frame", "end_frame",
                     "intensity_permille", "timbre_code"):
            try:
                values[name] = _parse_int(tokens[name])
            except ValueError:
                bad_field = name
                break
        if bad_field:
            failures.add(
                E_FIELD_TRUNCATED, "load", line=line_no, field=bad_field,
                message=f"non-numeric or truncated field {bad_field!r}",
            )
            continue
        if tokens["missing_flag"].strip() != "":
            failures.add(
                E_FIELD_TRUNCATED, "load", line=line_no,
                field="missing_flag",
                message="missing flag is non-blank",
            )
            continue
        permille = values["intensity_permille"]
        if not 0 <= permille <= 1000:
            failures.add(
                E_FIELD_TRUNCATED, "load", line=line_no,
                field="intensity_permille",
                message="permille outside [0,1000]",
            )
            continue
        timbre = config.code_to_name(values["timbre_code"])
        if timbre is None:
            failures.add(
                E_UNKNOWN_TIMBRE, "load", segment=values["segment"],
                line=line_no, field="timbre_code",
                message=f"unknown timbre code {values['timbre_code']}",
            )
            continue
        segments.append(
            Segment(
                index=values["segment"],
                start_frame=float(values["start_frame"]),
                end_frame=float(values["end_frame"]),
                intensity=permille / 1000.0,
                timbre=timbre,
                source_version="v1",
                intensity_permille=permille,
                precision_tag="exact-permille",
                source_line=line_no,
                timbre_code=values["timbre_code"],
            )
        )
    segments.sort(key=lambda s: s.index)
    return segments


def _fixed(token: str, width: int, field: str) -> str:
    if len(token) > width:
        raise PipelineError(
            E_FIELD_TRUNCATED,
            f"value {token!r} exceeds width {width} of field {field!r}",
        )
    return token.rjust(width)


def render_v1(segments: list[Segment], config: PipelineConfig) -> str:
    lines = [
        "# fixed-width v1: seg(6) start(8) end(8) permille(6) code(6) miss(2)"
    ]
    for seg in sorted(segments, key=lambda s: s.index):
        code = config.name_to_code(seg.timbre)
        if code is None:
            raise PipelineError(
                E_UNKNOWN_TIMBRE, f"timbre {seg.timbre!r} not in code table"
            )
        permille = (
            seg.intensity_permille
            if seg.intensity_permille is not None
            else int(round(seg.intensity * 1000.0))
        )
        line = (
            _fixed(str(seg.index), FIELD_WIDTHS["segment"], "segment")
            + _fixed(str(int(round(seg.start_frame))),
                     FIELD_WIDTHS["start_frame"], "start_frame")
            + _fixed(str(int(round(seg.end_frame))),
                     FIELD_WIDTHS["end_frame"], "end_frame")
            + _fixed(str(permille),
                     FIELD_WIDTHS["intensity_permille"], "intensity_permille")
            + _fixed(str(code), FIELD_WIDTHS["timbre_code"], "timbre_code")
            + " " * FIELD_WIDTHS["missing_flag"]
        )
        lines.append(line)
    return "\n".join(lines) + "\n"


def write_v1(path: str | Path, segments: list[Segment],
             config: PipelineConfig) -> None:
    Path(path).write_text(render_v1(segments, config), encoding="utf-8")
