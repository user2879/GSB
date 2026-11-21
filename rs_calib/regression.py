"""跨传感器回归对齐阶段：确定性最小二乘（岭回归）求解与逐瓦片映射。"""
from __future__ import annotations

import math

import numpy as np

from .errors import E_REGRESS_FAILED, STAGE_REGRESS, StageError
from .fingerprint import (
    content_fingerprint,
    derive_cache_key,
    regression_config_fingerprint,
    sensor_coeff_fingerprint,
)
from .fixedpoint import to_fixed_point
from .models import AlignmentRecord, CalibRecord, RegressionModel


def model_id(source: str, band: int) -> str:
    return f"model:{source}->ref:b{band}"


def _feature_matrix(x: np.ndarray, orders: list[int]) -> np.ndarray:
    xf = x.astype(np.float64)
    return np.column_stack([np.power(xf, k) for k in orders])


def solve_model(source: str, band: int, config: dict,
                calib_index: dict[tuple[str, int], CalibRecord],
                table: dict[str, dict], scene_pairs: list[tuple[str, str]]
                ) -> RegressionModel:
    ref_sensor = config["reference_sensor"]
    orders = [int(k) for k in config["feature_orders"]]
    reg = float(config["regularization"])
    rcond = float(config["near_singular_rcond"])
    total_bits = int(config["fixed_point_bits"])
    frac_bits = int(config["fixed_point_frac_bits"])

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for src_tile, ref_tile in scene_pairs:
        src_rec = calib_index.get((src_tile, band))
        ref_rec = calib_index.get((ref_tile, band))
        if src_rec is None or ref_rec is None:
            continue
        xs.append(np.asarray(src_rec["fixed_values"], dtype=np.float64))
        ys.append(np.asarray(ref_rec["fixed_values"], dtype=np.float64))
    if not xs:
        raise StageError(E_REGRESS_FAILED,
                         f"传感器 {source} 波段 {band} 缺少与参考传感器的同名场景样本",
                         stage=STAGE_REGRESS, sensor_id=source, band=band)
    fp_scale = float(1 << int(config["fixed_point_frac_bits"]))
    x_all = np.concatenate(xs) / fp_scale
    y_all = np.concatenate(ys) / fp_scale
    X = _feature_matrix(x_all, orders)

    # 固定初始权重：标量广播或按阶数展开
    iw = config["initial_weights"]
    if isinstance(iw, (int, float)):
        w0 = np.full(X.shape[1], float(iw), dtype=np.float64)
    else:
        w0 = np.asarray([float(v) for v in iw], dtype=np.float64)
        if w0.size != X.shape[1]:
            raise StageError(E_REGRESS_FAILED,
                             f"初始权重长度 {w0.size} 与特征阶数 {X.shape[1]} 不一致",
                             stage=STAGE_REGRESS, sensor_id=source, band=band)

    gram = X.T @ X
    rhs = X.T @ y_all
    # 岭正则：对非常数项施加（常数项对应 order==0，不惩罚）
    penalty = np.eye(gram.shape[0], dtype=np.float64) * reg
    for col, order in enumerate(orders):
        if order == 0:
            penalty[col, col] = 0.0
    near_singular = False
    condition_number = None
    try:
        cond = float(np.linalg.cond(gram))
        condition_number = cond if math.isfinite(cond) else None
        near_singular = (not math.isfinite(cond)) or cond > (1.0 / max(rcond, 1e-300))
    except Exception:
        near_singular = True
    try:
        weights = np.linalg.solve(gram + penalty, rhs + penalty @ w0)
    except np.linalg.LinAlgError as exc:
        # 近奇异兜底：提升正则强度后再解一次
        near_singular = True
        fallback = penalty + np.eye(gram.shape[0]) * max(reg * 1e6, 1e-8)
        weights = np.linalg.solve(gram + fallback, rhs + fallback @ w0)

    params_fp = content_fingerprint({
        "source": source, "band": band, "ref": ref_sensor,
        "orders": orders, "regularization": reg, "rcond": rcond,
        "initial_weights": [float(v) for v in w0],
        "config_fp": regression_config_fingerprint(config),
        "source_sensor_fp": sensor_coeff_fingerprint(table[source]),
        "ref_sensor_fp": sensor_coeff_fingerprint(table[ref_sensor]),
        "sample_scenes": sorted([f"{a}->{b}" for a, b in scene_pairs]),
        "sample_n": int(x_all.size),
    })
    key = derive_cache_key("model", params_fp)
    return RegressionModel(
        model_id=model_id(source, band),
        source_sensor=source,
        reference_sensor=ref_sensor,
        band=band,
        feature_orders=orders,
        weights=[float(v) for v in weights],
        regularization=reg,
        near_singular=near_singular,
        condition_number=condition_number,
        sample_count=int(x_all.size),
        key=key,
        referenced_sensors=sorted({source, ref_sensor}),
    )


def align_tile(record: CalibRecord, model: RegressionModel, config: dict,
               ref_records: dict[tuple[str, int], CalibRecord]) -> AlignmentRecord:
    total_bits = int(config["fixed_point_bits"])
    frac_bits = int(config["fixed_point_frac_bits"])
    scale = float(1 << frac_bits)
    x_phys = np.asarray(record.fixed_values, dtype=np.float64) / scale
    X = _feature_matrix(x_phys, model.feature_orders)
    mapped_phys = X @ np.asarray(model.weights, dtype=np.float64)
    aligned = to_fixed_point(mapped_phys, total_bits, frac_bits,
                             tile_id=record.tile_id, band=record.band,
                             stage=STAGE_REGRESS)
    ref = ref_records.get((record.scene_id, record.band))
    if ref is not None:
        residual = aligned.astype(np.float64) - np.asarray(ref["fixed_values"], dtype=np.float64)
        residual_mean = float(np.mean(residual))
        residual_max = float(np.max(np.abs(residual)))
        rmse = float(np.sqrt(np.mean(residual ** 2)))
    else:
        residual_mean = residual_max = rmse = 0.0
    upstream_fp = content_fingerprint({
        "calib_key": record.key,
        "model_key": model.key,
        "fixed_point_bits": total_bits,
        "fixed_point_frac_bits": frac_bits,
    })
    return AlignmentRecord(
        tile_id=record.tile_id,
        scene_id=record.scene_id,
        source_sensor=record.sensor_id,
        reference_sensor=model.reference_sensor,
        band=record.band,
        aligned_values=[int(v) for v in aligned],
        residual_mean=residual_mean,
        residual_max=residual_max,
        rmse=rmse,
        model_id=model.model_id,
        model_key=model.key,
        key=derive_cache_key("align", upstream_fp),
    )
