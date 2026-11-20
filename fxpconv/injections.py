"""异常注入清单加载与隔离引擎。

四类真实故障与业务码映射：
  nonfinite_scale         -> E_INJECT_NONFINITE_SCALE
  qbits_out_of_range      -> E_INJECT_QBITS_OUT_OF_RANGE
  data_out_of_shape       -> E_INJECT_DATA_OUT_OF_SHAPE
  illegal_strategy_combo  -> E_INJECT_ILLEGAL_STRATEGY

注入在独立配置中声明，命中位置被隔离（只影响该位置该环节），
其余位置照常计算，注入编号写入最终报告。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from .errors import (
    E_DESC_INVALID,
    E_INJECT_DATA_OUT_OF_SHAPE,
    E_INJECT_ILLEGAL_STRATEGY,
    E_INJECT_NONFINITE_SCALE,
    E_INJECT_QBITS_OUT_OF_RANGE,
    FailureLedger,
)
from .models import OpDesc
from .tensorpack import Tensor

KINDS = (
    "nonfinite_scale",
    "qbits_out_of_range",
    "data_out_of_shape",
    "illegal_strategy_combo",
)

KIND_CODE = {
    "nonfinite_scale": E_INJECT_NONFINITE_SCALE,
    "qbits_out_of_range": E_INJECT_QBITS_OUT_OF_RANGE,
    "data_out_of_shape": E_INJECT_DATA_OUT_OF_SHAPE,
    "illegal_strategy_combo": E_INJECT_ILLEGAL_STRATEGY,
}


@dataclass(frozen=True)
class Injection:
    injection_id: str
    kind: str
    target_position: int | None
    target_tensor: str | None
    stage: str
    params: dict

    def to_dict(self) -> dict:
        return {
            "id": self.injection_id,
            "kind": self.kind,
            "target_position": self.target_position,
            "target_tensor": self.target_tensor,
            "stage": self.stage,
            "params": self.params,
            "business_code": KIND_CODE[self.kind],
        }


def load_injections(path: str | Path, desc: OpDesc,
                    ledger: FailureLedger) -> list[Injection]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        ledger.add(E_DESC_INVALID, f"无法解析注入清单: {exc}", "inject", (0, "inject_file"))
        return []
    items = raw.get("injections") if isinstance(raw, dict) else None
    if not isinstance(items, list):
        ledger.add(E_DESC_INVALID, "注入清单必须含 injections 列表", "inject", (0, "inject_list"))
        return []

    result: list[Injection] = []
    seen: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            ledger.add(E_DESC_INVALID, f"注入项 {index} 必须是对象", "inject", (index, "type"))
            continue
        injection_id = str(item.get("id", f"INJ-{index:02d}"))
        if injection_id in seen:
            ledger.add(
                E_DESC_INVALID,
                f"注入编号重复: {injection_id}",
                "inject",
                (index, "dup", injection_id),
            )
            continue
        kind = item.get("kind")
        if kind not in KINDS:
            ledger.add(
                E_DESC_INVALID,
                f"注入 {injection_id} 类型 {kind!r} 非法，允许 {KINDS}",
                "inject",
                (index, injection_id, "kind"),
            )
            continue
        target_position = item.get("target_position")
        if target_position is not None and (
            not isinstance(target_position, int)
            or not 0 <= target_position < desc.total_positions
        ):
            ledger.add(
                E_DESC_INVALID,
                f"注入 {injection_id} 目标位置越界: {target_position!r}",
                "inject",
                (index, injection_id, "position"),
            )
            continue
        seen.add(injection_id)
        result.append(
            Injection(
                injection_id=injection_id,
                kind=kind,
                target_position=target_position,
                target_tensor=item.get("target_tensor"),
                stage=str(item.get("stage", "dequant")),
                params=item.get("params", {}) if isinstance(item.get("params"), dict) else {},
            )
        )
    return result


class InjectionEngine:
    """按位置/环节查询注入参数，并把命中映射成业务码失败记录。"""

    def __init__(self, injections: list[Injection], desc: OpDesc,
                 ledger: FailureLedger):
        self.desc = desc
        self.ledger = ledger
        self.by_position: dict[int, list[Injection]] = {}
        self.global_items: list[Injection] = []
        self.hits: list[dict] = []
        for injection in injections:
            if injection.target_position is None:
                self.global_items.append(injection)
            else:
                self.by_position.setdefault(injection.target_position, []).append(injection)

    def for_position(self, position: int) -> list[Injection]:
        return list(self.by_position.get(position, ()))

    def global_injections(self) -> list[Injection]:
        return list(self.global_items)

    def record_hit(self, injection: Injection, position: int | None, batch: int | None,
                   message: str) -> None:
        self.ledger.add(
            KIND_CODE[injection.kind],
            message,
            f"inject:{injection.stage}",
            (batch if batch is not None else -1, position if position is not None else -1,
             injection.injection_id),
            injection_id=injection.injection_id,
            position=position,
            batch=batch,
        )
        self.hits.append(
            {
                "id": injection.injection_id,
                "kind": injection.kind,
                "business_code": KIND_CODE[injection.kind],
                "position": position,
                "batch": batch,
                "stage": injection.stage,
                "message": message,
            }
        )

    def audit_global(self) -> dict[str, object]:
        """在运行前执行全局注入（数据越界、非法策略组合）。"""
        artifacts: dict[str, object] = {"bad_tensors": [], "illegal_combo": None}
        for injection in self.global_items:
            if injection.kind == "data_out_of_shape":
                artifacts["bad_tensors"].append(injection.to_dict())
                self.record_hit(
                    injection,
                    None,
                    None,
                    f"张量 {injection.target_tensor} 存在超出形状声明的整数数据"
                    f"（越界索引 {injection.params.get('bad_index')}）",
                )
            elif injection.kind == "illegal_strategy_combo":
                artifacts["illegal_combo"] = injection.to_dict()
                self.record_hit(
                    injection,
                    None,
                    None,
                    f"饱和策略 {injection.params.get('saturation')} 与位宽组合非法 "
                    f"(acc_bits={injection.params.get('acc_bits')}, "
                    f"qbits={injection.params.get('qbits')})",
                )
        return artifacts

    def nonfinite_scale(self, position: int, batch: int) -> float | None:
        for injection in self.for_position(position):
            if injection.kind != "nonfinite_scale":
                continue
            value = float(injection.params.get("value", "nan"))
            if not math.isfinite(value):
                self.record_hit(
                    injection,
                    position,
                    batch,
                    f"位置 {position} 反量化缩放因子非有限: {value}",
                )
                return value
        return None

    def qbits_override(self, position: int, batch: int) -> tuple[int, Injection] | None:
        for injection in self.for_position(position):
            if injection.kind != "qbits_out_of_range":
                continue
            qbits = int(injection.params.get("qbits", 3))
            if qbits not in (4, 8, 16):
                self.record_hit(
                    injection,
                    position,
                    batch,
                    f"位置 {position} 量化位宽 {qbits} 超出允许枚举 (4,8,16)",
                )
                return qbits, injection
        return None

    def build_bad_tensor(self, base: Tensor, injection: Injection) -> Tensor:
        bad_index = int(injection.params.get("bad_index", len(base.data)))
        bad_value = int(injection.params.get("bad_value", 999999))
        data = list(base.data)
        if 0 <= bad_index < len(data):
            data[bad_index] = bad_value
        else:
            data.append(bad_value)
        return Tensor(
            name=base.name,
            shape=base.shape,
            dtype=base.dtype,
            signed=base.signed,
            data=tuple(data),
            row_ids=base.row_ids,
        )
