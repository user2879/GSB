"""配置、特征表与列组映射的加载校验层。

非法条目分别打上业务码并保留失败明细，绝不在单条非法时中止整体加载。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .errors import (
    E_CONFIG_INVALID,
    E_DUPLICATE_ROW,
    E_EMPTY_GROUP,
    E_ENUM_INVALID,
    E_GROUP_UNKNOWN,
    E_MISSING_COLUMN,
    FailureCollector,
)
from .enums import (
    ALLOWED_QBITS,
    DEFAULT_ROUNDING,
    DEFAULT_SATURATION,
    ROUNDING_MODES,
    SATURATION_STRATEGIES,
    SCHEME_NAME_TO_CODE,
)
from .fingerprint import content_fingerprint


@dataclass
class FeatureTable:
    row_ids: List[str]
    columns: List[str]
    values: Dict[str, Dict[str, Optional[float]]]

    def column_vector(self, column: str) -> List[Optional[float]]:
        return [self.values[rid].get(column) for rid in self.row_ids]


@dataclass
class GroupConfig:
    group_id: str
    columns: List[str]
    scheme: str
    q_lo_permille: int
    q_hi_permille: int
    qbits: int
    fbits: int
    rounding: str
    saturation: str
    abs_tol: float
    rel_tol: float
    params_fingerprint: str = ""

    def param_parts(self) -> Dict[str, Any]:
        return {
            "group_id": self.group_id,
            "columns": list(self.columns),
            "scheme": self.scheme,
            "q_lo_permille": self.q_lo_permille,
            "q_hi_permille": self.q_hi_permille,
            "qbits": self.qbits,
            "fbits": self.fbits,
            "rounding": self.rounding,
            "saturation": self.saturation,
            "abs_tol": self.abs_tol,
            "rel_tol": self.rel_tol,
        }


@dataclass
class CalibrationConfig:
    partition: List[str]
    groups: List[GroupConfig]
    group_by_id: Dict[str, GroupConfig] = field(default_factory=dict)


# --------------------------------------------------------------------------
def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def load_feature_table(path: str | Path) -> Tuple[Optional[FeatureTable], FailureCollector]:
    """逐行解析特征表 JSONL。

    每行结构：{"row_id": "r0", "values": {"f0": 1.2, "f1": null}}
    缺失标记为 JSON null。坏行被记录但不中止后续行解析。
    """
    collector = FailureCollector(source=str(path))
    path = Path(path)
    row_ids: List[str] = []
    columns: List[str] = []
    values: Dict[str, Dict[str, Optional[float]]] = {}
    seen_rows: set[str] = set()

    with path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            loc = f"line {lineno}"
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                collector.add(E_CONFIG_INVALID, f"JSON 解析失败: {exc}", loc)
                continue
            if not isinstance(obj, dict) or "row_id" not in obj:
                collector.add(E_CONFIG_INVALID, "缺少 row_id 字段", loc)
                continue
            row_id = obj["row_id"]
            if not isinstance(row_id, str) or not row_id:
                collector.add(E_CONFIG_INVALID, "row_id 必须是非空字符串", loc)
                continue
            if row_id in seen_rows:
                collector.add(E_DUPLICATE_ROW, f"行标识重复: {row_id}", loc, row_id=row_id)
                continue
            raw_values = obj.get("values", {})
            if not isinstance(raw_values, dict):
                collector.add(E_CONFIG_INVALID, "values 必须是对象", loc, row_id=row_id)
                continue

            clean: Dict[str, Optional[float]] = {}
            row_ok = True
            for col, val in raw_values.items():
                if val is None:
                    clean[col] = None
                    if col not in columns:
                        columns.append(col)
                    continue
                if not _is_finite_number(val):
                    collector.add(
                        E_CONFIG_INVALID,
                        f"取值不是有限浮点或缺失标记: {col}={val!r}",
                        loc,
                        row_id=row_id,
                        column=col,
                    )
                    row_ok = False
                    continue
                clean[col] = float(val)
                if col not in columns:
                    columns.append(col)
            if not row_ok:
                # 行内存在非法取值：保留合法字段，但该行整体仍记入失败明细
                pass
            seen_rows.add(row_id)
            row_ids.append(row_id)
            values[row_id] = clean

    if not row_ids:
        return None, collector
    return FeatureTable(row_ids=row_ids, columns=columns, values=values), collector


def load_calibration_config(
    path: str | Path, table: FeatureTable
) -> Tuple[Optional[CalibrationConfig], FailureCollector]:
    """解析逐组标定配置并对照特征表做完整校验。"""
    collector = FailureCollector(source=str(path))
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    partition = payload.get("partition")
    raw_groups = payload.get("groups")
    if not isinstance(partition, list) or not all(isinstance(g, str) for g in partition):
        collector.add(E_CONFIG_INVALID, "partition 必须是字符串数组")
        partition = partition if isinstance(partition, list) else []
    if not isinstance(raw_groups, list):
        collector.add(E_CONFIG_INVALID, "groups 必须是数组")
        return None, collector

    declared = set(partition)
    groups: List[GroupConfig] = []
    for index, entry in enumerate(raw_groups):
        loc = f"groups[{index}]"
        if not isinstance(entry, dict):
            collector.add(E_CONFIG_INVALID, "组配置必须是对象", loc)
            continue
        group_id = entry.get("group_id")
        if not isinstance(group_id, str) or not group_id:
            collector.add(E_CONFIG_INVALID, "group_id 必须是非空字符串", loc)
            continue
        if group_id not in declared:
            collector.add(
                E_GROUP_UNKNOWN,
                f"组标识未在列组划分中声明: {group_id}",
                loc,
                group_id=group_id,
                declared=sorted(declared),
            )
            continue

        cols = entry.get("columns", [])
        if not isinstance(cols, list) or not all(isinstance(c, str) for c in cols):
            collector.add(E_CONFIG_INVALID, "columns 必须是字符串数组", loc, group_id=group_id)
            continue
        if len(cols) == 0:
            collector.add(E_EMPTY_GROUP, "组内列集合为空", loc, group_id=group_id)
            continue
        missing = [c for c in cols if c not in table.columns]
        if missing:
            collector.add(
                E_MISSING_COLUMN,
                f"列在表内不存在: {missing}",
                loc,
                group_id=group_id,
                missing=missing,
            )
            continue
        if len(set(cols)) != len(cols):
            collector.add(E_CONFIG_INVALID, "组内列重复", loc, group_id=group_id)
            continue

        scheme = entry.get("scheme", "asymmetric")
        if scheme not in SCHEME_NAME_TO_CODE:
            collector.add(
                E_ENUM_INVALID,
                f"未知量化方案: {scheme!r}",
                loc,
                group_id=group_id,
                allowed=sorted(SCHEME_NAME_TO_CODE),
            )
            continue

        q_lo_permille = entry.get("q_lo_permille")
        q_hi_permille = entry.get("q_hi_permille")
        if (
            not (isinstance(q_lo_permille, int) and isinstance(q_hi_permille, int))
            or isinstance(q_lo_permille, bool) or isinstance(q_hi_permille, bool)
        ):
            collector.add(E_CONFIG_INVALID, "分位千分比必须是整数", loc, group_id=group_id)
            continue
        if not (0 <= q_lo_permille < q_hi_permille <= 1000):
            collector.add(
                E_CONFIG_INVALID,
                f"需要 0<=lo<hi<=1000，实际 lo={q_lo_permille} hi={q_hi_permille}",
                loc,
                group_id=group_id,
            )
            continue

        qbits = entry.get("qbits")
        if qbits not in ALLOWED_QBITS:
            collector.add(
                E_ENUM_INVALID,
                f"量化位宽不在允许枚举内: {qbits!r}",
                loc,
                group_id=group_id,
                allowed=list(ALLOWED_QBITS),
            )
            continue

        fbits = entry.get("fbits")
        if not isinstance(fbits, int) or isinstance(fbits, bool) or fbits <= 0:
            collector.add(E_CONFIG_INVALID, f"定点小数位宽必须是正整数: {fbits!r}", loc, group_id=group_id)
            continue

        rounding = entry.get("rounding", DEFAULT_ROUNDING)
        if rounding not in ROUNDING_MODES:
            collector.add(
                E_ENUM_INVALID,
                f"未知舍入模式: {rounding!r}",
                loc,
                group_id=group_id,
                allowed=list(ROUNDING_MODES),
            )
            continue

        saturation = entry.get("saturation", DEFAULT_SATURATION)
        if saturation not in SATURATION_STRATEGIES:
            collector.add(
                E_ENUM_INVALID,
                f"未知饱和策略: {saturation!r}",
                loc,
                group_id=group_id,
                allowed=list(SATURATION_STRATEGIES),
            )
            continue

        abs_tol = entry.get("abs_tol", 0.0)
        rel_tol = entry.get("rel_tol", 0.0)
        if not _is_finite_number(abs_tol) or not _is_finite_number(rel_tol) or abs_tol < 0 or rel_tol < 0:
            collector.add(E_CONFIG_INVALID, "容差必须是非负有限数", loc, group_id=group_id)
            continue

        cfg = GroupConfig(
            group_id=group_id,
            columns=list(cols),
            scheme=scheme,
            q_lo_permille=q_lo_permille,
            q_hi_permille=q_hi_permille,
            qbits=qbits,
            fbits=fbits,
            rounding=rounding,
            saturation=saturation,
            abs_tol=float(abs_tol),
            rel_tol=float(rel_tol),
        )
        cfg.params_fingerprint = content_fingerprint(cfg.param_parts())
        groups.append(cfg)

    # 稳定顺序：按 partition 声明顺序输出
    order = {gid: i for i, gid in enumerate(partition)}
    groups.sort(key=lambda g: (order.get(g.group_id, len(order)), g.group_id))
    config = CalibrationConfig(
        partition=list(partition),
        groups=groups,
        group_by_id={g.group_id: g for g in groups},
    )
    return config, collector
