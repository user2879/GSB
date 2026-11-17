"""版本双向转换器与信息保留策略。

信息保留策略：
- v1 -> v2：千分比整数按精确换算（p / 1000）转浮点，精度标签记为 permille。
- v2 -> v1：按四舍五入（ROUND_HALF_UP）还原千分比；截断残差、精度标签与
  步骤对象参数写入旁路元数据文件，并逐条记入降级损失清单。
- v3 互转：步骤枚举与字符串步骤名通过可配置名称表映射；名称表缺项时输出
  E_STEP_MAP_MISSING，步骤在 IR 中保留不丢失；float32 舍入损失逐条记录。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np

from .errors import E_STEP_MAP_MISSING, Failure
from .ir import PRECISION_FLOAT32, PRECISION_PERMILLE, IR


def round_half_up_permille(value: float) -> int:
    return int(Decimal(repr(value * 1000)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))


@dataclass
class ConvertContext:
    name_map: dict[str, int]
    sidecar_dir: Path
    sidecar: dict = field(default_factory=dict)
    ledger: list[dict] = field(default_factory=list)
    failures: list[Failure] = field(default_factory=list)

    def load_sidecar(self) -> None:
        path = self.sidecar_dir / "sidecar.json"
        if path.exists():
            self.sidecar = json.loads(path.read_text(encoding="utf-8"))

    def flush_sidecar(self) -> None:
        self.sidecar_dir.mkdir(parents=True, exist_ok=True)
        path = self.sidecar_dir / "sidecar.json"
        tmp = self.sidecar_dir / "sidecar.json.tmp"
        tmp.write_text(json.dumps(self.sidecar, ensure_ascii=False, indent=1,
                                  sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(path)


def convert_ir(ir: IR, target_version: str, ctx: ConvertContext) -> IR:
    """把 IR 调整到目标版本语义，返回新 IR；损失逐条写入 ctx.ledger。"""
    source = ir.source_version
    if source == target_version:
        return ir
    out = ir.clone()
    out.source_version = target_version
    seq = len(ctx.ledger)

    if target_version == "v1":
        residuals: dict[str, dict[str, float]] = {}
        for entity in sorted(out.features.values):
            for feature in sorted(out.features.values[entity]):
                value = out.features.values[entity][feature]
                permille = round_half_up_permille(value)
                residual = value - permille / 1000.0
                out.features.values[entity][feature] = permille / 1000.0
                old_label = out.features.precision.get(entity, {}).get(feature)
                out.features.precision.setdefault(entity, {})[feature] = PRECISION_PERMILLE
                if residual != 0.0:
                    residuals.setdefault(entity, {})[feature] = residual
                    ctx.ledger.append({
                        "seq": seq, "kind": "permille_truncation",
                        "conversion": f"{source}->v1", "entity": entity,
                        "feature": feature, "residual": residual,
                    })
                    seq += 1
                if old_label and old_label != PRECISION_PERMILLE:
                    ctx.ledger.append({
                        "seq": seq, "kind": "precision_label_downgrade",
                        "conversion": f"{source}->v1", "entity": entity,
                        "feature": feature, "from": old_label, "to": PRECISION_PERMILLE,
                    })
                    seq += 1
        ctx.sidecar.setdefault("v1_residuals", {}).update(residuals)
        ctx.sidecar["precision_labels"] = {
            e: dict(p) for e, p in sorted(ir.features.precision.items())}
        params = {str(i): s.params for i, s in enumerate(ir.steps) if s.params}
        if params:
            ctx.sidecar["step_params"] = params
            ctx.ledger.append({
                "seq": seq, "kind": "step_params_sidecar",
                "conversion": f"{source}->v1",
                "detail": f"{len(params)} 个步骤的参数写入旁路元数据",
            })
            seq += 1

    if target_version == "v2":
        for entity in sorted(out.features.values):
            for feature in sorted(out.features.values[entity]):
                label = out.features.precision.get(entity, {}).get(feature)
                if label is None:
                    out.features.precision.setdefault(entity, {})[feature] = (
                        PRECISION_PERMILLE if source == "v1" else PRECISION_FLOAT32
                        if source == "v3" else "double")

    if target_version == "v3":
        for entity in sorted(out.features.values):
            for feature in sorted(out.features.values[entity]):
                value = out.features.values[entity][feature]
                rounded = float(np.float32(value))
                if rounded != value:
                    ctx.ledger.append({
                        "seq": seq, "kind": "float32_rounding",
                        "conversion": f"{source}->v3", "entity": entity,
                        "feature": feature, "before": value, "after": rounded,
                    })
                    seq += 1
                out.features.values[entity][feature] = rounded
                out.features.precision.setdefault(entity, {})[feature] = PRECISION_FLOAT32
        for step in out.steps:
            if step.name not in ctx.name_map and not step.name.startswith("__unknown_enum_"):
                ctx.failures.append(Failure(
                    entity_id=None, stage="convert_v3", code=E_STEP_MAP_MISSING,
                    exc_type="StepMapMissing",
                    detail=f"步骤 {step.name} 不在名称映射表，IR 中保留不丢步骤"))
                ctx.ledger.append({
                    "seq": seq, "kind": "step_enum_unmapped",
                    "conversion": f"{source}->v3", "step": step.name,
                })
                seq += 1

    if source == "v3" and target_version in ("v1", "v2"):
        for step in out.steps:
            if step.name.startswith("__unknown_enum_"):
                ctx.ledger.append({
                    "seq": seq, "kind": "unknown_enum_step_carried",
                    "conversion": f"v3->{target_version}", "step": step.name,
                })
                seq += 1

    if target_version in ("v1", "v3"):
        lost = [(e, f) for e in sorted(out.features.explicit_null)
                for f in sorted(out.features.explicit_null[e])]
        for entity, feature in lost:
            ctx.ledger.append({
                "seq": seq, "kind": "explicit_null_lost",
                "conversion": f"{source}->{target_version}",
                "entity": entity, "feature": feature,
            })
            seq += 1
        if lost:
            out.features.explicit_null = {e: set() for e in out.features.explicit_null}
    return out
