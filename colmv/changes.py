"""NDJSON change-stream parsing and row-shape validation."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, NoReturn

from .errors import ColmvError
from .model import BaseTable, Change


def _change_error(message: str, *, seq: Any = None, line: int | None = None, field: str | None = None, **details: Any) -> NoReturn:
    raise ColmvError(
        "E_CHANGE",
        message,
        field=field,
        details={"seq": seq, "line": line, **details},
    )


def _validate_row(row: Any, table: BaseTable, *, seq: int, line: int) -> dict[str, Any]:
    if not isinstance(row, dict):
        _change_error("insert/update row must be an object", seq=seq, line=line, field="row")
    expected_names = [column.name for column in table.columns]
    expected_set = set(expected_names)
    actual_set = set(row)
    missing = [name for name in expected_names if name not in actual_set]
    extra = sorted(actual_set - expected_set)
    if missing:
        _change_error(
            "change row is missing base-table columns",
            seq=seq,
            line=line,
            field="row",
            missing_columns=missing,
        )
    if extra:
        _change_error(
            "change row contains unknown base-table columns",
            seq=seq,
            line=line,
            field="row",
            extra_columns=extra,
        )
    for column in table.columns:
        value = row[column.name]
        if value is None:
            continue
        if column.type == "int":
            if not isinstance(value, int) or isinstance(value, bool):
                _change_error(
                    "integer column contains a non-integer value",
                    seq=seq,
                    line=line,
                    field=f"row.{column.name}",
                )
            if not -(2**63) <= value <= 2**63 - 1:
                _change_error(
                    "integer column value exceeds signed int64",
                    seq=seq,
                    line=line,
                    field=f"row.{column.name}",
                )
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                _change_error(
                    "float column contains a non-numeric value",
                    seq=seq,
                    line=line,
                    field=f"row.{column.name}",
                )
            if not math.isfinite(float(value)):
                _change_error(
                    "float column contains a non-finite value",
                    seq=seq,
                    line=line,
                    field=f"row.{column.name}",
                )
            row[column.name] = float(value)
    return row


def load_changes(path: str | Path, table: BaseTable) -> list[Change]:
    changes_path = Path(path)
    try:
        lines = changes_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        _change_error(f"cannot read NDJSON change stream {changes_path}: {exc}", field="file")

    changes: list[Change] = []
    previous_seq: int | None = None
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            _change_error(
                "invalid NDJSON line",
                line=line_number,
                field="ndjson",
                reason=exc.msg,
                column=exc.colno,
            )
        if not isinstance(raw, dict):
            _change_error("change line must be a JSON object", line=line_number, field="document")
        seq = raw.get("seq")
        if not isinstance(seq, int) or isinstance(seq, bool):
            _change_error("change seq must be an integer", line=line_number, field="seq")
        if previous_seq is not None and seq <= previous_seq:
            _change_error(
                "change seq must be strictly increasing in file order",
                seq=seq,
                line=line_number,
                field="seq",
                previous_seq=previous_seq,
            )
        previous_seq = seq
        op = raw.get("op")
        if op not in ("insert", "update", "delete"):
            _change_error(
                "change op must be insert, update, or delete",
                seq=seq,
                line=line_number,
                field="op",
            )
        row_id = raw.get("row_id")
        if not isinstance(row_id, int) or isinstance(row_id, bool):
            _change_error("change row_id must be an integer", seq=seq, line=line_number, field="row_id")

        if op == "delete":
            row = raw.get("row")
            if row is not None:
                if not isinstance(row, dict):
                    _change_error("delete row, when present, must be an object", seq=seq, line=line_number, field="row")
                row = _validate_row(row, table, seq=seq, line=line_number)
        else:
            if "row" not in raw:
                _change_error(f"{op} change requires a row", seq=seq, line=line_number, field="row")
            row = _validate_row(raw["row"], table, seq=seq, line=line_number)
            key_value = row.get(table.row_key)
            if key_value != row_id:
                _change_error(
                    "row_id must equal the row_key column value in the post-change row",
                    seq=seq,
                    line=line_number,
                    field="row_id",
                    row_key=table.row_key,
                    row_key_value=key_value,
                )
        changes.append(Change(seq=seq, op=op, row_id=row_id, row=row))
    if not changes:
        _change_error("change stream is empty", field="file")
    return changes
