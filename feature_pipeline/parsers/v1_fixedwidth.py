"""v1 fixed-width text exchange format.

Layout (no separator, every line has exactly ``LINE_WIDTH`` characters before
the platform newline):

    entity_id   width 12, left justified, padded with spaces
    feature     width 24, left justified, padded with spaces
    permille    width 10, right justified, signed integer in per-mille units
    missing     width 1, literal ``0`` (present) or ``1`` (missing)

Values are stored as per-mille integers, i.e. canonical value = integer/1000.
"""

from __future__ import annotations

from pathlib import Path

from ..errors import E_LINE_OVERFLOW, E_NONFINITE_INPUT, LineOverflowError, NonFiniteInputError
from ..models import V1, Cell, EntityTable


ID_WIDTH = 12
FEATURE_WIDTH = 24
PERMILLE_WIDTH = 10
MISSING_WIDTH = 1
LINE_WIDTH = ID_WIDTH + FEATURE_WIDTH + PERMILLE_WIDTH + MISSING_WIDTH
MAGIC = "FEATV1"
HEADER_PREFIX = MAGIC + " "


def format_line(entity_id: str, feature: str, permille: int | None, missing: bool) -> str:
    if missing or permille is None:
        number_field = " " * (PERMILLE_WIDTH - 1) + "0"
        missing_flag = "1"
    else:
        token = str(int(permille))
        if len(token) > PERMILLE_WIDTH:
            raise LineOverflowError(
                f"per-mille token {token!r} exceeds field width {PERMILLE_WIDTH}",
                entity_id=entity_id,
                stage="v1_serialize",
            )
        number_field = token.rjust(PERMILLE_WIDTH)
        missing_flag = "0"
    line = (
        entity_id.ljust(ID_WIDTH)
        + feature.ljust(FEATURE_WIDTH)
        + number_field
        + missing_flag
    )
    if len(line) != LINE_WIDTH:
        raise LineOverflowError(
            f"assembled line width {len(line)} != {LINE_WIDTH}",
            entity_id=entity_id,
            stage="v1_serialize",
        )
    return line


def write_file(path: str | Path, table: EntityTable) -> None:
    lines = [HEADER_PREFIX + f"width={LINE_WIDTH}"]
    for cell in table.cells:
        if cell.missing:
            permille = None
        else:
            permille = int(round(cell.value * 1000))
        lines.append(format_line(cell.entity_id, cell.feature, permille, cell.missing))
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_file(path: str | Path) -> EntityTable:
    text = Path(path).read_text(encoding="utf-8")
    return parse_text(text, source_path=str(path))


def parse_text(text: str, source_path: str = "<memory>") -> EntityTable:
    table = EntityTable(source_version=V1)
    feature_seen: dict[str, None] = {}
    lines = text.split("\n")
    expected_width = LINE_WIDTH

    for line_index, raw_line in enumerate(lines):
        line = raw_line.rstrip("\r")
        if line == "":
            continue
        if line.startswith(HEADER_PREFIX):
            declared = _parse_header(line)
            if declared is not None:
                expected_width = declared
            continue
        if len(line) > expected_width:
            raise LineOverflowError(
                f"line {line_index + 1} length {len(line)} exceeds fixed width "
                f"{expected_width} in {source_path}",
                stage="v1_parse",
            )
        if len(line) < expected_width:
            # Short lines are invalid fixed-width records, surfaced with the same
            # business code as overflow because the width contract is broken.
            raise LineOverflowError(
                f"line {line_index + 1} length {len(line)} is shorter than fixed "
                f"width {expected_width} in {source_path}",
                stage="v1_parse",
            )

        entity_id = line[0:ID_WIDTH].strip()
        feature = line[ID_WIDTH : ID_WIDTH + FEATURE_WIDTH].strip()
        number_token = line[
            ID_WIDTH + FEATURE_WIDTH : ID_WIDTH + FEATURE_WIDTH + PERMILLE_WIDTH
        ]
        missing_token = line[-MISSING_WIDTH]

        if missing_token not in ("0", "1"):
            raise LineOverflowError(
                f"line {line_index + 1} missing flag must be 0/1, got {missing_token!r}",
                entity_id=entity_id,
                stage="v1_parse",
            )
        missing = missing_token == "1"

        if missing:
            value = float("nan")
        else:
            try:
                permille = int(number_token)
            except ValueError as exc:
                raise LineOverflowError(
                    f"line {line_index + 1} per-mille field {number_token!r} is not "
                    "an integer",
                    entity_id=entity_id,
                    stage="v1_parse",
                ) from exc
            value = permille / 1000.0
            if not _is_finite(value):
                raise NonFiniteInputError(
                    f"non-finite v1 value for {entity_id}/{feature}",
                    entity_id=entity_id,
                    stage="v1_parse",
                )

        feature_seen.setdefault(feature, None)
        table.cells.append(
            Cell(
                entity_id=entity_id,
                feature=feature,
                source_version=V1,
                value=value,
                missing=missing,
                precision_label="per_mille",
                raw_text=line,
            )
        )

    table.feature_order = list(feature_seen)
    return table


def _parse_header(line: str) -> int | None:
    body = line[len(HEADER_PREFIX) :]
    for part in body.split():
        if part.startswith("width="):
            try:
                return int(part.split("=", 1)[1])
            except ValueError:
                return None
    return None


def _is_finite(value: float) -> bool:
    return value == value and value not in (float("inf"), float("-inf"))


BUSINESS_CODES = (E_LINE_OVERFLOW, E_NONFINITE_INPUT)
