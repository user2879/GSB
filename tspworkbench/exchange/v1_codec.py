"""v1 定宽扁平文本行的解析与写入。

行类型：HD 头、MW 模型权重（千分比整数）、PR 逐窗逐模型预测、MT 窗口指标。
解析严格按 :data:`V1_LAYOUT` 切片，任何长度不符或字段非整数都判定列错位。
"""
from __future__ import annotations

from pathlib import Path

from ..constants import (
    V1_LAYOUT, v1_slices, v1_line_length, CODE_TO_MODEL, MODEL_TO_CODE,
)
from ..errors import WorkbenchError, E_FIELD_LAYOUT, E_MODEL_ENUM
from ..util import fingerprint
from .ir import ExchangeIR, WindowWeight, WindowPrediction, WindowMetric


def _format_int(value: int, width: int) -> str:
    text = str(int(value))
    if len(text) > width:
        raise WorkbenchError(E_FIELD_LAYOUT,
                             f"整数 {value} 宽度 {len(text)} 超过字段宽度 {width}")
    return text.rjust(width, "0")


def _format_text(value: str, width: int) -> str:
    text = str(value)
    if len(text) > width:
        raise WorkbenchError(E_FIELD_LAYOUT,
                             f"文本 {text!r} 宽度 {len(text)} 超过字段宽度 {width}")
    return text.ljust(width, " ")


def encode_line(tag: str, fields: dict) -> str:
    parts = [tag]
    for name, width in V1_LAYOUT[tag].items():
        if name == "tag":
            continue
        value = fields[name]
        if isinstance(value, str) and not value.lstrip("-").isdigit():
            parts.append(_format_text(value, width))
        else:
            parts.append(_format_int(int(value), width))
    return "".join(parts)


def _parse_int(token: str, tag: str, name: str, line_no: int) -> int:
    token = token.strip()
    if not token or not (token.lstrip("-").isdigit()):
        raise WorkbenchError(E_FIELD_LAYOUT,
                             f"第 {line_no} 行 {tag}.{name} 字段非整数: {token!r}",
                             context={"line_no": line_no, "field": name})
    return int(token)



def fingerprint_v1(ir: "ExchangeIR") -> str:
    """v1 版本指纹：仅由 v1 可表达内容派生，解析与转换两侧必须一致。"""
    return fingerprint([
        "v1", ir.config_fingerprint,
        [(w.series_id, w.window_index, w.model, round(w.weight, 6), w.lag_order)
         for w in ir.weights],
        [(pr.series_id, pr.window_index, pr.model, pr.step_index, pr.pred_q, pr.actual_q)
         for pr in ir.predictions],
        [(m.series_id, m.window_index, m.mae_q, m.rmse_q, m.mape_bp, m.near_zero)
         for m in ir.metrics],
    ])


def parse_v1(text: str, failures: list | None = None) -> ExchangeIR:
    failures = failures if failures is not None else []
    header: dict = {}
    weights: list[WindowWeight] = []
    predictions: list[WindowPrediction] = []
    metrics: list[WindowMetric] = []

    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.rstrip("\n")
        if not line.strip():
            continue
        tag = line[:2]
        if tag not in V1_LAYOUT:
            failures.append(WorkbenchError(E_FIELD_LAYOUT,
                                           f"第 {line_no} 行未知行类型 {tag!r}",
                                           context={"line_no": line_no}).to_record())
            continue
        expected = v1_line_length(tag)
        if len(line) != expected:
            failures.append(WorkbenchError(
                E_FIELD_LAYOUT,
                f"第 {line_no} 行长度 {len(line)} 与定宽 {expected} 不符（列错位）",
                context={"line_no": line_no, "expected": expected, "actual": len(line)},
            ).to_record())
            continue

        fields: dict[str, str] = {}
        for name, lo, hi in v1_slices(tag):
            fields[name] = line[lo:hi]
        try:
            if tag == "HD":
                header = {
                    "series_id": fields["series_id"].strip(),
                    "window_index": _parse_int(fields["window_index"], tag, "window_index", line_no),
                    "ex_version": fields["ex_version"].strip(),
                }
            elif tag == "MW":
                code = _parse_int(fields["model_code"], tag, "model_code", line_no)
                if code not in CODE_TO_MODEL:
                    raise WorkbenchError(E_MODEL_ENUM, f"v1 模型码 {code} 无映射",
                                         context={"code": code})
                weights.append(WindowWeight(
                    series_id=fields["series_id"].strip(),
                    window_index=_parse_int(fields["window_index"], tag, "window_index", line_no),
                    model=CODE_TO_MODEL[code],
                    weight=_parse_int(fields["weight_permille"], tag, "weight_permille", line_no) / 1000.0,
                    lag_order=_parse_int(fields["lag_order"], tag, "lag_order", line_no),
                    precision_label="permille",
                ))
            elif tag == "PR":
                code = _parse_int(fields["model_code"], tag, "model_code", line_no)
                if code not in CODE_TO_MODEL:
                    raise WorkbenchError(E_MODEL_ENUM, f"v1 模型码 {code} 无映射",
                                         context={"code": code})
                predictions.append(WindowPrediction(
                    series_id=fields["series_id"].strip(),
                    window_index=_parse_int(fields["window_index"], tag, "window_index", line_no),
                    model=CODE_TO_MODEL[code],
                    step_index=_parse_int(fields["step_index"], tag, "step_index", line_no),
                    pred_q=_parse_int(fields["pred_q"], tag, "pred_q", line_no),
                    actual_q=_parse_int(fields["actual_q"], tag, "actual_q", line_no),
                ))
            elif tag == "MT":
                metrics.append(WindowMetric(
                    series_id=fields["series_id"].strip(),
                    window_index=_parse_int(fields["window_index"], tag, "window_index", line_no),
                    mae_q=_parse_int(fields["mae_q"], tag, "mae_q", line_no),
                    rmse_q=_parse_int(fields["rmse_q"], tag, "rmse_q", line_no),
                    mape_bp=_parse_int(fields["mape_bp"], tag, "mape_bp", line_no),
                    near_zero=_parse_int(fields["near_zero"], tag, "near_zero", line_no) == 1,
                ))
        except WorkbenchError as exc:
            failures.append(exc.to_record())

    ir = ExchangeIR(
        exchange_version="v1",
        config_fingerprint=header.get("ex_version", ""),
        version_fingerprint="",
        missing_strategy="forward_fill",   # v1 不携带命名缺失策略，默认值（可被旁路覆盖）
        weights=weights,
        predictions=predictions,
        metrics=metrics,
    )
    ir.version_fingerprint = fingerprint_v1(ir)
    return ir


def write_v1(ir: ExchangeIR, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    head_series = ir.weights[0].series_id if ir.weights else (
        ir.metrics[0].series_id if ir.metrics else "")
    head_window = ir.weights[0].window_index if ir.weights else (
        ir.metrics[0].window_index if ir.metrics else 0)
    lines.append(encode_line("HD", {
        "series_id": head_series, "window_index": head_window,
        "ex_version": ir.config_fingerprint,
    }))
    for w in ir.weights:
        lines.append(encode_line("MW", {
            "series_id": w.series_id, "window_index": w.window_index,
            "model_code": MODEL_TO_CODE[w.model],
            "weight_permille": int(round(w.weight * 1000)),
            "lag_order": w.lag_order,
        }))
    for p in ir.predictions:
        lines.append(encode_line("PR", {
            "series_id": p.series_id, "window_index": p.window_index,
            "model_code": MODEL_TO_CODE[p.model],
            "step_index": p.step_index, "pred_q": p.pred_q, "actual_q": p.actual_q,
        }))
    for m in ir.metrics:
        lines.append(encode_line("MT", {
            "series_id": m.series_id, "window_index": m.window_index,
            "mae_q": m.mae_q, "rmse_q": m.rmse_q, "mape_bp": m.mape_bp,
            "near_zero": 1 if m.near_zero else 0,
        }))
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path
