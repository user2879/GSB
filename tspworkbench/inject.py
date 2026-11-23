"""独立故障注入配置与真实输入故障施加。

注入项按编号在报告中留痕并映射到稳定业务码；所有注入都作用于内存中的
序列/窗口/量化配置副本，不污染源样本文件，实现故障隔离。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .constants import QUANT_GEARS
from .errors import (
    WorkbenchError, E_TIMESTAMP_IRREGULAR, E_NON_FINITE,
    E_WINDOW_TOO_LONG, E_QUANT_GEAR_INVALID,
)

INJECTION_CODE_MAP = {
    "irregular_timestamp": E_TIMESTAMP_IRREGULAR,
    "non_finite_value": E_NON_FINITE,
    "window_too_long": E_WINDOW_TOO_LONG,
    "quant_gear_out_of_range": E_QUANT_GEAR_INVALID,
}


@dataclass
class Injection:
    injection_id: str
    kind: str
    series_id: str
    window_index: int | None = None

    @property
    def code(self) -> str:
        return INJECTION_CODE_MAP[self.kind]


def load_injections(path: str | Path | None) -> list[Injection]:
    if not path:
        return []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    out: list[Injection] = []
    for item in data.get("injections", []):
        kind = item["kind"]
        if kind not in INJECTION_CODE_MAP:
            raise ValueError(f"未知注入类型 {kind}")
        out.append(Injection(
            injection_id=item["id"],
            kind=kind,
            series_id=item["series_id"],
            window_index=item.get("window_index"),
        ))
    return out


def apply_row_injections(rows: list[dict], injections: list[Injection]) -> list[Injection]:
    """施加作用于原始行的故障：时间戳非等间隔、观测非有限。返回命中的注入项。"""
    hit: list[Injection] = []
    for inj in injections:
        if inj.series_id != rows[0]["series_id"]:
            continue
        if inj.kind == "irregular_timestamp" and len(rows) >= 2:
            target = min(1 if inj.window_index is None else inj.window_index, len(rows) - 1)
            rows[target]["timestamp_ms"] = int(rows[target]["timestamp_ms"]) + 1
            hit.append(inj)
        elif inj.kind == "non_finite_value":
            target = 0 if inj.window_index is None else min(inj.window_index, len(rows) - 1)
            rows[target]["value"] = float("nan")
            rows[target]["missing"] = False
            hit.append(inj)
    return hit


def window_injection_error(series_id: str, window_index: int,
                           injections: list[Injection], available_len: int,
                           required_len: int) -> WorkbenchError | None:
    """施加作用于窗口级的故障：窗口超长、量化档位越界。"""
    for inj in injections:
        if inj.series_id != series_id:
            continue
        if inj.window_index is not None and inj.window_index != window_index:
            continue
        if inj.kind == "window_too_long":
            return WorkbenchError(
                E_WINDOW_TOO_LONG, f"[注入 {inj.injection_id}] 窗口长度超过可用序列长度",
                injection_id=inj.injection_id,
                context={"series_id": series_id, "window_index": window_index,
                         "available": available_len, "required": required_len})
        if inj.kind == "quant_gear_out_of_range":
            bad_gear = max(QUANT_GEARS) + 1
            return WorkbenchError(
                E_QUANT_GEAR_INVALID, f"[注入 {inj.injection_id}] 量化档位 {bad_gear} 越界",
                injection_id=inj.injection_id,
                context={"series_id": series_id, "window_index": window_index,
                         "gear": bad_gear, "allowed": list(QUANT_GEARS)})
    return None
