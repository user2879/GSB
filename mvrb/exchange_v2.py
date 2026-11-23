"""v2 交换文件：JSON，浮点坐标、命名匹配度量、命名舍入模式与精度标签。"""
from __future__ import annotations

from .ir import Feature, Match, ReconstructionIR, VoxelPoint
from .utils import read_json, write_json

V2_FORMAT = "MVRB2"


def parse_v2(data: dict) -> ReconstructionIR:
    header = data.get("header", {})
    ir = ReconstructionIR(
        image_size=tuple(header["image_size"]),
        voxel_size=float(header["voxel_size"]),
        fixed_point_bits=int(header.get("fixed_point_bits", 8)),
        rounding_mode=header.get("rounding_mode", "ROUND_HALF_UP"),
        source_version="v2",
    )
    for d in data.get("features", []):
        ir.features.append(
            Feature(
                view=int(d["view"]),
                feature_id=int(d["feature_id"]),
                x=float(d["x"]),
                y=float(d["y"]),
                landmark_id=int(d["landmark_id"]),
                precision=str(d.get("precision", "float:v2")),
            )
        )
    for d in data.get("matches", []):
        ir.matches.append(
            Match(
                view_a=int(d["view_a"]),
                feature_a=int(d["feature_a"]),
                view_b=int(d["view_b"]),
                feature_b=int(d["feature_b"]),
                score_permille=float(d["score_permille"]),
                metric=str(d.get("metric", "ncc")),
                metric_code=int(d.get("metric_code", 0)),
                precision=str(d.get("precision", "float:v2")),
            )
        )
    for d in data.get("voxel_points", []):
        ir.voxel_points.append(
            VoxelPoint(
                vx=int(d["vx"]),
                vy=int(d["vy"]),
                vz=int(d["vz"]),
                x=float(d["x"]),
                y=float(d["y"]),
                z=float(d["z"]),
                landmark_id=int(d["landmark_id"]),
                precision=str(d.get("precision", "float:v2")),
            )
        )
    return ir


def serialize_v2(ir: ReconstructionIR) -> dict:
    return {
        "format": V2_FORMAT,
        "header": {
            "image_size": list(ir.image_size),
            "voxel_size": ir.voxel_size,
            "fixed_point_bits": ir.fixed_point_bits,
            "rounding_mode": ir.rounding_mode,
        },
        "features": [f.to_dict() for f in ir.features],
        "matches": [m.to_dict() for m in ir.matches],
        "voxel_points": [p.to_dict() for p in ir.voxel_points],
    }


def write_v2(path: str, ir: ReconstructionIR) -> None:
    write_json(path, serialize_v2(ir))


def read_v2(path: str) -> ReconstructionIR:
    return parse_v2(read_json(path))
