"""v1 定宽文本行与 v2 JSON 权重文件的序列化、解析与字段校验。"""

from __future__ import annotations

import json
import math

from .errors import ErrorCode, FailureList, PipelineError
from .fixedpoint import ALLOWED_BIT_WIDTHS
from .models import (
    V1_BIAS_MAX,
    V1_BIAS_MIN,
    V1_FIELD_WIDTHS,
    V1_LINE_WIDTH,
    V1_SCALE_MAX,
    V1_SCALE_MIN,
    V1WeightRow,
    V2WeightRow,
)

V1_HEADER = "# v1 fixed-width weights: layer(24) row(6) scale_permille(12) bias_code(12) bit_width(10)"


# ---------------------------------------------------------------------- #
# v1
# ---------------------------------------------------------------------- #
def format_v1_line(row: V1WeightRow) -> str:
    parts = [
        f"{row.layer:<{V1_FIELD_WIDTHS['layer']}}",
        f"{row.row_index:>{V1_FIELD_WIDTHS['row']}}",
        f"{row.scale_permille:>{V1_FIELD_WIDTHS['scale_permille']}}",
        f"{row.bias_code:>{V1_FIELD_WIDTHS['bias_code']}}",
        f"{row.bit_width:>{V1_FIELD_WIDTHS['bit_width']}}",
    ]
    line = "".join(parts)
    if len(line) != V1_LINE_WIDTH:
        raise PipelineError(
            ErrorCode.E_V1_LINE_MALFORMED,
            f"v1 行宽 {len(line)} 不等于定宽 {V1_LINE_WIDTH}",
            {"layer": row.layer, "row": row.row_index},
        )
    return line


def parse_v1_line(line: str, source: str, failures: FailureList) -> V1WeightRow | None:
    if len(line) != V1_LINE_WIDTH:
        failures.add(
            ErrorCode.E_V1_LINE_MALFORMED,
            source,
            f"行宽 {len(line)} 不等于定宽 {V1_LINE_WIDTH}",
            {"line": line},
        )
        return None
    widths = list(V1_FIELD_WIDTHS.values())
    offsets = [0]
    for width in widths[:-1]:
        offsets.append(offsets[-1] + width)
    fields = [line[offsets[i]: offsets[i] + widths[i]] for i in range(len(widths))]
    names = list(V1_FIELD_WIDTHS.keys())
    try:
        parsed = {
            "layer": fields[0].strip(),
            "row_index": int(fields[1]),
            "scale_permille": int(fields[2]),
            "bias_code": int(fields[3]),
            "bit_width": int(fields[4]),
        }
    except ValueError:
        failures.add(
            ErrorCode.E_V1_LINE_MALFORMED,
            source,
            f"数值字段无法解析: {line!r}",
            dict(zip(names, fields)),
        )
        return None
    return V1WeightRow(**parsed)


def validate_v1_row(row: V1WeightRow, source: str, failures: FailureList) -> bool:
    ok = True
    if not V1_SCALE_MIN <= row.scale_permille <= V1_SCALE_MAX:
        failures.add(
            ErrorCode.E_FIELD_RANGE, source,
            f"千分比缩放因子 {row.scale_permille} 越界 "
            f"[{V1_SCALE_MIN},{V1_SCALE_MAX}]",
            {"layer": row.layer, "row": row.row_index},
        )
        ok = False
    if not V1_BIAS_MIN <= row.bias_code <= V1_BIAS_MAX:
        failures.add(
            ErrorCode.E_FIELD_RANGE, source,
            f"整数偏置码 {row.bias_code} 越界 [{V1_BIAS_MIN},{V1_BIAS_MAX}]",
            {"layer": row.layer, "row": row.row_index},
        )
        ok = False
    if row.bit_width not in ALLOWED_BIT_WIDTHS:
        failures.add(
            ErrorCode.E_BITWIDTH_ENUM, source,
            f"位宽 {row.bit_width} 不在 {ALLOWED_BIT_WIDTHS}",
            {"layer": row.layer, "row": row.row_index},
        )
        ok = False
    return ok


def write_v1(path: str, rows: list[V1WeightRow]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(V1_HEADER + "\n")
        for row in rows:
            handle.write(format_v1_line(row) + "\n")


def read_v1(path: str, failures: FailureList) -> list[V1WeightRow]:
    rows: list[V1WeightRow] = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            source = f"{path}#L{lineno}"
            row = parse_v1_line(line, source, failures)
            if row is not None and validate_v1_row(row, source, failures):
                rows.append(row)
    return rows


# ---------------------------------------------------------------------- #
# v2
# ---------------------------------------------------------------------- #
def write_v2(path: str, rows: list[V2WeightRow]) -> None:
    payload = {
        "format": "v2",
        "count": len(rows),
        "rows": [
            {
                "layer": row.layer,
                "row_index": row.row_index,
                "scale": row.scale,
                "scheme": row.scheme,
                "rounding_mode": row.rounding_mode,
            }
            for row in rows
        ],
    }
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def read_v2(path: str, failures: FailureList) -> list[V2WeightRow]:
    with open(path, "r", encoding="utf-8") as handle:
        try:
            payload = json.load(handle)
        except json.JSONDecodeError as exc:
            failures.add(ErrorCode.E_V2_SCHEMA, path, f"JSON 解析失败: {exc}")
            return []
    if not isinstance(payload, dict) or not isinstance(payload.get("rows"), list):
        failures.add(ErrorCode.E_V2_SCHEMA, path, "缺少 rows 数组")
        return []
    rows: list[V2WeightRow] = []
    for index, item in enumerate(payload["rows"]):
        source = f"{path}#rows[{index}]"
        required = ("layer", "row_index", "scale", "scheme", "rounding_mode")
        if not isinstance(item, dict) or any(key not in item for key in required):
            failures.add(ErrorCode.E_V2_SCHEMA, source, f"缺少必填字段 {required}", item or {})
            continue
        if not math.isfinite(item["scale"]):
            failures.add(
                ErrorCode.E_NON_FINITE, source,
                f"浮点缩放因子非有限值 {item['scale']}",
                {"layer": item["layer"], "row": item["row_index"]},
            )
            continue
        rows.append(
            V2WeightRow(
                layer=item["layer"],
                row_index=int(item["row_index"]),
                scale=float(item["scale"]),
                scheme=item["scheme"],
                rounding_mode=item["rounding_mode"],
            )
        )
    return rows
