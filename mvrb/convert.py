"""v1 <-> v2 双向转换与降级信息旁路（sidecar）。

- v1 -> v2：整数坐标/千分比按精确换算转浮点，生成精度标签；度量码按码表映射命名度量。
- v2 -> v1：浮点四舍五入还原整数/千分比，截断残差、精度标签、命名度量与舍入模式
  写入旁路元数据文件；旁路可在再次升版时逐字段还原原始 v2。
"""
from __future__ import annotations

import copy
import os

from .errors import E_UNKNOWN_METRIC, MVRBError
from .exchange_v1 import read_v1, write_v1
from .exchange_v2 import read_v2, write_v2
from .ir import ReconstructionIR
from .metrics import code_to_name, name_to_code
from .utils import read_json, round_half_up, write_json

SIDECAR_SUFFIX = ".sidecar.json"


def upgrade_v1_to_v2(ir_v1: ReconstructionIR, sidecar: dict | None = None) -> ReconstructionIR:
    """v1 IR 升级为 v2 IR；提供 sidecar 时按旁路逐字段还原原始 v2。"""
    ir = copy.deepcopy(ir_v1)
    ir.source_version = "v2"
    sf = sidecar or {}
    feat_side = {tuple(map(int, k.split(":"))): v for k, v in sf.get("features", {}).items()}
    match_side = {tuple(map(int, k.split(":"))): v for k, v in sf.get("matches", {}).items()}
    voxel_side = {tuple(map(int, k.split(":"))): v for k, v in sf.get("voxel_points", {}).items()}

    for f in ir.features:
        key = (f.view, f.feature_id)
        if key in feat_side:
            rec = feat_side[key]
            f.x = float(rec["x"])
            f.y = float(rec["y"])
            f.precision = rec["precision"]
        else:
            f.x = float(round_half_up(f.x))
            f.y = float(round_half_up(f.y))
            f.precision = "float:v2(exact-int)"

    for m in ir.matches:
        key = (m.view_a, m.feature_a, m.view_b, m.feature_b)
        if key in match_side:
            rec = match_side[key]
            m.score_permille = float(rec["score_permille"])
            m.metric = rec["metric"]
            m.metric_code = name_to_code(rec["metric"])
            m.precision = rec["precision"]
        else:
            m.score_permille = float(round_half_up(m.score_permille))
            try:
                m.metric = code_to_name(m.metric_code)
            except MVRBError:
                ir.failures.fail(
                    E_UNKNOWN_METRIC,
                    f"cannot map metric code {m.metric_code}",
                    view=m.view_a,
                    stage="upgrade",
                )
                m.metric = "unknown"
            m.precision = "float:v2(exact-int)"

    for p in ir.voxel_points:
        key = (p.vx, p.vy, p.vz, p.landmark_id)
        if key in voxel_side:
            rec = voxel_side[key]
            p.x, p.y, p.z = float(rec["x"]), float(rec["y"]), float(rec["z"])
            p.precision = rec["precision"]
        else:
            p.precision = "float:v2(exact-fixed)"

    ir.rounding_mode = sf.get("rounding_mode", "ROUND_HALF_UP")
    return ir


def downgrade_v2_to_v1(ir_v2: ReconstructionIR) -> tuple[ReconstructionIR, dict]:
    """v2 IR 降级为 v1 IR，同时生成旁路元数据。"""
    ir = copy.deepcopy(ir_v2)
    ir.source_version = "v1"
    sidecar = {
        "format": "MVRB-SIDECAR-1",
        "rounding_mode": ir_v2.rounding_mode,
        "features": {},
        "matches": {},
        "voxel_points": {},
    }
    for f in ir.features:
        ix, iy = round_half_up(f.x), round_half_up(f.y)
        sidecar["features"][f"{f.view}:{f.feature_id}"] = {
            "x": f.x,
            "y": f.y,
            "residual_x": f.x - ix,
            "residual_y": f.y - iy,
            "precision": f.precision,
        }
        f.x, f.y = float(ix), float(iy)
        f.precision = "fixed:v1"

    for m in ir.matches:
        iscore = round_half_up(m.score_permille)
        try:
            code = name_to_code(m.metric)
        except MVRBError:
            raise
        sidecar["matches"][f"{m.view_a}:{m.feature_a}:{m.view_b}:{m.feature_b}"] = {
            "score_permille": m.score_permille,
            "residual_score": m.score_permille - iscore,
            "metric": m.metric,
            "metric_code": code,
            "rounding_mode": ir_v2.rounding_mode,
            "precision": m.precision,
        }
        m.score_permille = float(iscore)
        m.metric_code = code
        m.precision = "fixed:v1"

    for p in ir.voxel_points:
        sidecar["voxel_points"][f"{p.vx}:{p.vy}:{p.vz}:{p.landmark_id}"] = {
            "x": p.x,
            "y": p.y,
            "z": p.z,
            "precision": p.precision,
        }
        p.precision = "fixed:v1"

    return ir, sidecar


# ---- 文件级便捷 API ----
def convert_file(v1_path: str, v2_path: str, sidecar_path: str | None = None) -> None:
    ir_v1 = read_v1(v1_path)
    sidecar = read_json(sidecar_path) if sidecar_path and os.path.exists(sidecar_path) else None
    ir_v2 = upgrade_v1_to_v2(ir_v1, sidecar)
    write_v2(v2_path, ir_v2)


def downgrade_file(v2_path: str, v1_path: str, sidecar_path: str) -> None:
    ir_v2 = read_v2(v2_path)
    ir_v1, sidecar = downgrade_v2_to_v1(ir_v2)
    write_v1(v1_path, ir_v1)
    write_json(sidecar_path, sidecar)
