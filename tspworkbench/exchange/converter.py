"""v1 <-> v2 双向转换与降级信息旁路。

- 升版 v1->v2：千分比整数权重精确换算为浮点，生成精度标签；模型码经码表映射为
  命名模型；若存在旁路且指纹匹配，则还原原始浮点权重、精度标签与命名缺失策略。
- 降版 v2->v1：浮点权重四舍五入回千分比，截断残差、精度标签与被丢弃的命名
  缺失处理方式写入旁路元数据，旁路携带配置指纹与版本指纹。
"""
from __future__ import annotations

from pathlib import Path

from ..util import fingerprint
from .ir import ExchangeIR, WindowWeight
from .sidecar import Sidecar, SidecarEntry, read_sidecar, assert_sidecar_matches
from .v1_codec import parse_v1, write_v1, fingerprint_v1
from .v2_codec import parse_v2, write_v2


def fingerprint_v2(ir: ExchangeIR) -> str:
    """v2 版本指纹：基于 v2 可表达的全部内容（含命名策略与精度标签）。"""
    return fingerprint([
        "v2", ir.config_fingerprint, ir.missing_strategy,
        [(w.series_id, w.window_index, w.model, w.weight, w.lag_order, w.precision_label)
         for w in ir.weights],
        [(p.series_id, p.window_index, p.model, p.step_index, p.pred_q, p.actual_q)
         for p in ir.predictions],
        [(m.series_id, m.window_index, m.mae_q, m.rmse_q, m.mape_bp, m.near_zero)
         for m in ir.metrics],
    ])


def v1_to_v2(v1_text: str, sidecar: Sidecar | None = None) -> ExchangeIR:
    ir = parse_v1(v1_text)
    restore: dict = {}
    if sidecar is not None:
        assert_sidecar_matches(sidecar, ir.config_fingerprint, ir.version_fingerprint)
        restore = sidecar.entry_key()
        ir.missing_strategy = sidecar.missing_strategy

    upgraded: list[WindowWeight] = []
    for w in ir.weights:
        key = (w.series_id, w.window_index, w.model)
        if key in restore:
            entry = restore[key]
            upgraded.append(WindowWeight(
                w.series_id, w.window_index, w.model,
                weight=entry.original_weight,
                lag_order=w.lag_order,
                precision_label=entry.precision_label,
            ))
        else:
            permille = int(round(w.weight * 1000))
            upgraded.append(WindowWeight(
                w.series_id, w.window_index, w.model,
                weight=permille / 1000.0,
                lag_order=w.lag_order,
                precision_label="permille-exact",
            ))
    ir.weights = upgraded
    ir.exchange_version = "v2"
    ir.version_fingerprint = fingerprint_v2(ir)
    ir.validate()
    return ir


def v2_to_v1(ir_v2: ExchangeIR) -> tuple[ExchangeIR, Sidecar]:
    ir_v2.validate()
    downgraded: list[WindowWeight] = []
    entries: list[SidecarEntry] = []
    for w in ir_v2.weights:
        permille = int(round(w.weight * 1000))
        rounded_weight = permille / 1000.0
        residual = w.weight - rounded_weight
        entries.append(SidecarEntry(
            series_id=w.series_id,
            window_index=w.window_index,
            model=w.model,
            original_weight=w.weight,
            rounded_permille=permille,
            residual=residual,
            precision_label=w.precision_label,
        ))
        downgraded.append(WindowWeight(
            w.series_id, w.window_index, w.model,
            weight=rounded_weight,
            lag_order=w.lag_order,
            precision_label="permille",
        ))

    ir_v1 = ExchangeIR(
        exchange_version="v1",
        config_fingerprint=ir_v2.config_fingerprint,
        version_fingerprint="",
        missing_strategy="forward_fill",  # v1 无法表达命名缺失策略，真值进旁路
        weights=downgraded,
        predictions=list(ir_v2.predictions),
        metrics=list(ir_v2.metrics),
    )
    ir_v1.version_fingerprint = fingerprint_v1(ir_v1)
    sidecar = Sidecar(
        config_fingerprint=ir_v1.config_fingerprint,
        version_fingerprint=ir_v1.version_fingerprint,
        missing_strategy=ir_v2.missing_strategy,
        entries=entries,
    )
    ir_v1.validate()
    return ir_v1, sidecar


def convert_file(src_path: str, dst_path: str) -> dict:
    src, dst = Path(src_path), Path(dst_path)
    text = src.read_text(encoding="utf-8")
    summary: dict = {"src": str(src), "dst": str(dst)}

    if src.name.endswith(".v1.txt"):
        sidecar = read_sidecar(src)
        ir = v1_to_v2(text, sidecar)
        write_v2(ir, dst)
        summary["direction"] = "v1->v2"
        summary["restored_from_sidecar"] = sidecar is not None
    elif src.name.endswith(".v2.json"):
        ir_v2 = parse_v2(text)
        ir_v1, sidecar = v2_to_v1(ir_v2)
        write_v1(ir_v1, dst)
        sidecar_path = sidecar.write(dst)
        summary["direction"] = "v2->v1"
        summary["sidecar"] = str(sidecar_path)
        summary["max_abs_residual"] = max(
            (abs(e.residual) for e in sidecar.entries), default=0.0)
    else:
        raise ValueError(f"无法从文件名识别交换版本: {src.name}")
    return summary
