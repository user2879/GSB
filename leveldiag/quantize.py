"""连续量 -> 网格图块的量化规则与误差度量。

规则（写入映射表头部，复算器按同一规则独立复算）：
  q(x) = clamp( round_dir(x / step) * step, 0, 1 )
  round_dir:
    nearest -> floor(x/step + 0.5)
    down    -> floor(x/step)
    up      -> ceil(x/step)
  量化差值 delta = q(x) - x（双精度），理论误差上界：
    nearest: step/2；down/up: step
  另叠加一次双精度表示误差（1 ULP of q(x)），合成声明上界。

映射表按区域分片；terrain 为类别量，不参与浮点量化，仅登记来源。
E_QUANT_BOUNDS：量化结果越出 [0,1] 或实际 |delta| 超过声明上界。
"""

from __future__ import annotations

import math

from .errors import ErrorCode, LevelDiagError
from .spec import Spec
from .util import content_hash, ulp


def _round_index(x: float, step: float, rounding: str) -> int:
    ratio = x / step
    if rounding == "nearest":
        return int(math.floor(ratio + 0.5))
    if rounding == "down":
        return int(math.floor(ratio))
    if rounding == "up":
        return int(math.ceil(ratio))
    raise ValueError(f"未知舍入方向 {rounding}")


def quantize_value(x: float, step: float, rounding: str) -> tuple[float, int]:
    """返回 (量化值, 量化档级 index)。"""
    idx = _round_index(float(x), float(step), rounding)
    q = idx * float(step)
    if q < 0.0 or q > 1.0 + 1e-15:
        raise LevelDiagError(
            ErrorCode.E_QUANT_BOUNDS,
            f"量化值 {q!r} 越出有效域 [0,1]",
            {"continuous": x, "step": step, "quantized": q},
        )
    return float(min(max(q, 0.0), 1.0)), idx


def theoretical_bound(step: float, rounding: str) -> float:
    half = float(step) / 2.0 if rounding == "nearest" else float(step)
    # 表示误差按 1 个 ULP(1.0) 保守计入（足够覆盖 q(x) 自身 ULP）
    return half + ulp(1.0)


def quantize(spec: Spec, continuous: dict) -> dict:
    """把连续域结果量化为按区域分片的映射表，并汇总误差上界。"""
    raw = spec.raw
    elev_step = float(raw["quantization"]["elevation_step"])
    dens_step = float(raw["quantization"]["density_step"])
    rounding = raw["quantization"]["rounding"]
    elev_bound = theoretical_bound(elev_step, rounding)
    dens_bound = theoretical_bound(dens_step, rounding)

    shards: dict[str, dict] = {}
    max_elev_delta = 0.0
    max_dens_delta = 0.0
    for rid in sorted(continuous["region_shards"]):
        entries = []
        for item in continuous["region_shards"][rid]["cells"]:
            x, y = item["cell"]
            e_q, e_idx = quantize_value(item["elevation"], elev_step, rounding)
            d_q, d_idx = quantize_value(item["density"], dens_step, rounding)
            e_delta = e_q - float(item["elevation"])
            d_delta = d_q - float(item["density"])
            max_elev_delta = max(max_elev_delta, abs(e_delta))
            max_dens_delta = max(max_dens_delta, abs(d_delta))
            if abs(e_delta) > elev_bound + ulp(elev_bound):
                raise LevelDiagError(
                    ErrorCode.E_QUANT_BOUNDS,
                    f"高程量化误差 {e_delta!r} 超过声明上界 {elev_bound!r}",
                    {"cell": [x, y], "field": "elevation"},
                )
            if abs(d_delta) > dens_bound + ulp(dens_bound):
                raise LevelDiagError(
                    ErrorCode.E_QUANT_BOUNDS,
                    f"密度量化误差 {d_delta!r} 超过声明上界 {dens_bound!r}",
                    {"cell": [x, y], "field": "density"},
                )
            entries.append({
                "cell": [x, y],
                "terrain": item["terrain"],
                "elevation": {
                    "continuous": float(item["elevation"]),
                    "quantized": float(e_q),
                    "level_index": e_idx,
                    "delta": float(e_delta),
                },
                "density": {
                    "continuous": float(item["density"]),
                    "quantized": float(d_q),
                    "level_index": d_idx,
                    "delta": float(d_delta),
                },
            })
        shards[rid] = {"region_id": rid, "mappings": entries}

    table = {
        "spec_hash": spec.spec_hash,
        "continuous_hash": continuous["continuous_hash"],
        "rules": {
            "rounding": rounding,
            "elevation_step": elev_step,
            "density_step": dens_step,
            "formula": "q=clamp(round_dir(x/step)*step,0,1)",
            "elevation_error_bound": elev_bound,
            "density_error_bound": dens_bound,
        },
        "region_shards": shards,
        "observed_max_error": {
            "elevation": float(max_elev_delta),
            "density": float(max_dens_delta),
        },
    }
    table["mapping_hash"] = content_hash(
        {"rules": table["rules"], "region_shards": shards})
    return table
