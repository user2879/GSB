"""v1 定宽文本行 与 v2 JSON 标定结果的解析、统一表示与双向转换。"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict, field
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .enums import (
    DEFAULT_ROUNDING,
    ROUNDING_MODES,
    SCHEME_CODE_TO_NAME,
    SCHEME_NAME_TO_CODE,
    V1_FIELD_WIDTHS,
    V1_LINE_WIDTH,
    V1_PRECISION_LABEL,
    qint_range,
)
from .errors import (
    E_CONFIG_INVALID,
    E_ENUM_INVALID,
    E_FIELD_TRUNCATED,
    E_SCALE_INVALID,
    E_ZERO_OUT_OF_RANGE,
    FailureCollector,
)
from .fingerprint import fp_float, version_id


@dataclass
class CalibEntry:
    """统一中间表示（IR）。"""

    group_id: str
    column: str
    scale: float                 # 浮点缩放因子（v1 由 permille/1000 精确换算）
    zero_point: int
    qbits: int
    scheme: str                  # v2 命名方案
    rounding: str                # 命名舍入模式
    precision_label: str
    per_channel: bool = False
    source_version: int = 2
    # v1 来源时保留原始 permille，便于无损往返
    scale_permille: Optional[int] = None
    version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------
# v1 定宽文本
# --------------------------------------------------------------------------
def _slice_fields(line: str) -> Optional[Dict[str, str]]:
    if len(line) < V1_LINE_WIDTH:
        return None
    cursor = 0
    out: Dict[str, str] = {}
    for name, width in V1_FIELD_WIDTHS.items():
        out[name] = line[cursor:cursor + width]
        cursor += width
    return out


def encode_v1_line(entry: CalibEntry) -> str:
    """把 IR 编码为定宽文本行。permille 优先取原始值，否则四舍五入。"""
    if entry.scale_permille is not None:
        permille = entry.scale_permille
    else:
        permille = _float_to_permille(entry.scale)
    code = SCHEME_NAME_TO_CODE[entry.scheme]
    parts = [
        entry.group_id.ljust(V1_FIELD_WIDTHS["group"]),
        entry.column.ljust(V1_FIELD_WIDTHS["column"]),
        str(permille).rjust(V1_FIELD_WIDTHS["scale_permille"]),
        str(entry.zero_point).rjust(V1_FIELD_WIDTHS["zero_point"]),
        str(entry.qbits).rjust(V1_FIELD_WIDTHS["qbits"]),
    ]
    line = "".join(parts)
    assert len(line) == V1_LINE_WIDTH
    # 方案码不占独立字段宽度：附加在 qbits 之后的短标记（不参与定宽切分）
    return line + code


def parse_v1(text: str, default_rounding: str = DEFAULT_ROUNDING) -> Tuple[List[CalibEntry], FailureCollector]:
    collector = FailureCollector(source="v1")
    entries: List[CalibEntry] = []
    for lineno, raw in enumerate(text.splitlines(), start=1):
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        loc = f"line {lineno}"
        slices = _slice_fields(line)
        if slices is None:
            collector.add(
                E_FIELD_TRUNCATED,
                f"定宽行宽度不足/错位: 长度 {len(line)} < {V1_LINE_WIDTH}",
                loc,
                length=len(line),
                expected=V1_LINE_WIDTH,
            )
            continue
        group_id = slices["group"].strip()
        column = slices["column"].strip()
        if not group_id or not column:
            collector.add(E_FIELD_TRUNCATED, "组标识或列名字段为空（疑似错位）", loc)
            continue
        try:
            permille = int(slices["scale_permille"].strip())
            zero_point = int(slices["zero_point"].strip())
            qbits = int(slices["qbits"].strip())
        except ValueError as exc:
            collector.add(E_CONFIG_INVALID, f"整数字段解析失败: {exc}", loc)
            continue
        scheme_code = line[V1_LINE_WIDTH:V1_LINE_WIDTH + 1].strip() or "A"
        if scheme_code not in SCHEME_CODE_TO_NAME:
            collector.add(E_ENUM_INVALID, f"未知方案码: {scheme_code!r}", loc)
            continue
        scheme = SCHEME_CODE_TO_NAME[scheme_code]

        scale = permille / 1000.0
        if not math.isfinite(scale) or scale <= 0.0:
            collector.add(E_SCALE_INVALID, f"缩放因子非有限正数: permille={permille}", loc)
            continue
        lo, hi = qint_range(qbits, scheme)
        if not (lo <= zero_point <= hi):
            collector.add(
                E_ZERO_OUT_OF_RANGE,
                f"零点 {zero_point} 超出 qbits={qbits}/{scheme} 区间 [{lo},{hi}]",
                loc,
            )
            continue
        if default_rounding not in ROUNDING_MODES:
            collector.add(E_ENUM_INVALID, f"未知舍入模式: {default_rounding!r}", loc)
            continue
        entry = CalibEntry(
            group_id=group_id,
            column=column,
            scale=scale,
            zero_point=zero_point,
            qbits=qbits,
            scheme=scheme,
            rounding=default_rounding,
            precision_label=V1_PRECISION_LABEL,
            per_channel=True,
            source_version=1,
            scale_permille=permille,
        )
        entry.version = version_id(1, _entry_fp_payload(entry))
        entries.append(entry)
    return entries, collector


def write_v1(path: str | Path, entries: List[CalibEntry]) -> None:
    Path(path).write_text(
        "\n".join(encode_v1_line(e) for e in entries) + ("\n" if entries else ""),
        encoding="utf-8",
    )


def read_v1(path: str | Path, default_rounding: str = DEFAULT_ROUNDING) -> Tuple[List[CalibEntry], FailureCollector]:
    return parse_v1(Path(path).read_text(encoding="utf-8"), default_rounding)


# --------------------------------------------------------------------------
# v2 JSON
# --------------------------------------------------------------------------
def _entry_fp_payload(entry: CalibEntry) -> Dict[str, Any]:
    return {
        "group_id": entry.group_id,
        "column": entry.column,
        "scale": repr(entry.scale),
        "zero_point": entry.zero_point,
        "qbits": entry.qbits,
        "scheme": entry.scheme,
        "rounding": entry.rounding,
        "precision_label": entry.precision_label,
        "per_channel": entry.per_channel,
    }


def encode_v2_doc(entries: List[CalibEntry]) -> Dict[str, Any]:
    doc_entries = []
    for entry in entries:
        payload = _entry_fp_payload(entry)
        doc_entries.append(
            {
                "group_id": entry.group_id,
                "column": entry.column,
                "scale": entry.scale,
                "zero_point": entry.zero_point,
                "qbits": entry.qbits,
                "scheme": entry.scheme,
                "rounding": entry.rounding,
                "precision_label": entry.precision_label,
                "per_channel": entry.per_channel,
                "version": entry.version or version_id(entry.source_version, payload),
            }
        )
    return {"format_version": 2, "entries": doc_entries}


def write_v2(path: str | Path, entries: List[CalibEntry]) -> None:
    Path(path).write_text(
        json.dumps(encode_v2_doc(entries), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def parse_v2(text: str) -> Tuple[List[CalibEntry], FailureCollector]:
    collector = FailureCollector(source="v2")
    entries: List[CalibEntry] = []
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        collector.add(E_CONFIG_INVALID, f"v2 JSON 解析失败: {exc}")
        return entries, collector
    raw_entries = doc.get("entries") if isinstance(doc, dict) else None
    if not isinstance(raw_entries, list):
        collector.add(E_CONFIG_INVALID, "v2 文档缺少 entries 数组")
        return entries, collector

    for index, obj in enumerate(raw_entries):
        loc = f"entries[{index}]"
        if not isinstance(obj, dict):
            collector.add(E_CONFIG_INVALID, "条目必须是对象", loc)
            continue
        group_id = obj.get("group_id")
        column = obj.get("column")
        if not isinstance(group_id, str) or not isinstance(column, str):
            collector.add(E_CONFIG_INVALID, "group_id/column 必须是字符串", loc)
            continue
        scale = obj.get("scale")
        if not isinstance(scale, (int, float)) or isinstance(scale, bool) or not math.isfinite(float(scale)) or float(scale) <= 0:
            collector.add(E_SCALE_INVALID, f"缩放因子非有限正数: {scale!r}", loc)
            continue
        scale = float(scale)
        qbits = obj.get("qbits")
        zero_point = obj.get("zero_point")
        scheme = obj.get("scheme")
        rounding = obj.get("rounding")
        if scheme not in SCHEME_NAME_TO_CODE:
            collector.add(E_ENUM_INVALID, f"未知量化方案: {scheme!r}", loc)
            continue
        if rounding not in ROUNDING_MODES:
            collector.add(E_ENUM_INVALID, f"未知舍入模式: {rounding!r}", loc)
            continue
        if not isinstance(qbits, int) or isinstance(qbits, bool):
            collector.add(E_CONFIG_INVALID, "qbits 必须是整数", loc)
            continue
        if not isinstance(zero_point, int) or isinstance(zero_point, bool):
            collector.add(E_CONFIG_INVALID, "zero_point 必须是整数", loc)
            continue
        lo, hi = qint_range(qbits, scheme)
        if not (lo <= zero_point <= hi):
            collector.add(
                E_ZERO_OUT_OF_RANGE,
                f"零点 {zero_point} 超出 qbits={qbits}/{scheme} 区间 [{lo},{hi}]",
                loc,
            )
            continue
        label = obj.get("precision_label") or f"float-{fp_float(scale)}"
        if not isinstance(label, str):
            collector.add(E_CONFIG_INVALID, "precision_label 必须是字符串", loc)
            continue
        entry = CalibEntry(
            group_id=group_id,
            column=column,
            scale=scale,
            zero_point=zero_point,
            qbits=qbits,
            scheme=scheme,
            rounding=rounding,
            precision_label=label,
            per_channel=bool(obj.get("per_channel", False)),
            source_version=2,
            scale_permille=None,
            version=obj.get("version", ""),
        )
        if not entry.version:
            entry.version = version_id(2, _entry_fp_payload(entry))
        entries.append(entry)
    return entries, collector


def read_v2(path: str | Path) -> Tuple[List[CalibEntry], FailureCollector]:
    return parse_v2(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# 双向转换与旁路
# --------------------------------------------------------------------------
def _float_to_permille(scale: float) -> int:
    """四舍五入（ROUND_HALF_UP）还原千分比。"""
    return int(
        (Decimal(repr(scale)) * Decimal(1000)).quantize(
            Decimal("1"), rounding=ROUND_HALF_UP
        )
    )


def upgrade_v1_to_v2(entries: List[CalibEntry]) -> List[CalibEntry]:
    """v1 -> v2：permille/1000 精确换算，方案码表映射，生成精度标签。"""
    out: List[CalibEntry] = []
    for entry in entries:
        if entry.source_version != 1:
            raise ValueError("upgrade_v1_to_v2 仅接受 source_version==1 的条目")
        upgraded = CalibEntry(
            group_id=entry.group_id,
            column=entry.column,
            scale=entry.scale_permille / 1000.0,
            zero_point=entry.zero_point,
            qbits=entry.qbits,
            scheme=entry.scheme,
            rounding=entry.rounding,
            precision_label=f"float-from-{V1_PRECISION_LABEL}",
            per_channel=entry.per_channel,
            source_version=2,
            scale_permille=entry.scale_permille,
        )
        upgraded.version = version_id(2, _entry_fp_payload(upgraded))
        out.append(upgraded)
    return out


def downgrade_v2_to_v1(entries: List[CalibEntry]) -> Tuple[List[CalibEntry], Dict[str, Any]]:
    """v2 -> v1：四舍五入还原 permille，并产出可再次解析的旁路元数据。"""
    downgraded: List[CalibEntry] = []
    side_entries: List[Dict[str, Any]] = []
    for entry in entries:
        permille = _float_to_permille(entry.scale)
        restored_scale = permille / 1000.0
        residual = entry.scale - restored_scale
        dg = CalibEntry(
            group_id=entry.group_id,
            column=entry.column,
            scale=restored_scale,
            zero_point=entry.zero_point,
            qbits=entry.qbits,
            scheme=entry.scheme,
            rounding=entry.rounding,
            precision_label=V1_PRECISION_LABEL,
            per_channel=entry.per_channel,
            source_version=1,
            scale_permille=permille,
        )
        dg.version = version_id(1, _entry_fp_payload(dg))
        downgraded.append(dg)
        side_entries.append(
            {
                "group_id": entry.group_id,
                "column": entry.column,
                "original_scale": entry.scale,
                "permille": permille,
                "restored_scale": restored_scale,
                "truncation_residual": residual,
                "precision_label": entry.precision_label,
                "rounding": entry.rounding,
                "per_channel": entry.per_channel,
                "entry_version": entry.version,
            }
        )
    sidecar: Dict[str, Any] = {
        "format_version": 1,
        "kind": "v2-to-v1-sidecar",
        "rounding_for_permille": "half_up",
        "entries": side_entries,
    }
    sidecar["version"] = version_id(2, {"entries": side_entries})
    return downgraded, sidecar


def write_sidecar(path: str | Path, sidecar: Dict[str, Any]) -> None:
    Path(path).write_text(
        json.dumps(sidecar, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def read_sidecar(path: str | Path) -> Dict[str, Any]:
    """旁路元数据须可被再次解析用于比较。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))
