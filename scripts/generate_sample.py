"""固定种子生成确定性本地样本：多视图 PNG、相机参数、重建配置、v1/v2 交换文件。

用法:
    python scripts/generate_sample.py --out samples/demo [--views 6] [--seed 20261124]
不访问网络/外部数据集。
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mvrb.camera import save_cameras
from mvrb.config import ReconConfig
from mvrb.convert import downgrade_v2_to_v1
from mvrb.exchange_v1 import write_v1
from mvrb.exchange_v2 import write_v2
from mvrb.ir import Feature, Match, ReconstructionIR, VoxelPoint
from mvrb.metrics import name_to_code
from mvrb.synthetic import (
    IMAGE_H,
    IMAGE_W,
    make_cameras,
    make_scene,
    render_view,
    save_png,
)
from mvrb.utils import round_half_up, snap_to_grid, write_json


def build_exchange_ir(scene, cameras, config: ReconConfig) -> ReconstructionIR:
    features, matches, voxel_points = [], [], []
    voxel_size = config.voxel_size
    # 每个图元在每个可见视图中的整数像素特征
    projected = {}
    for view, cam in sorted(cameras.items()):
        for lid, point in scene:
            (u, v), depth = cam.project(point)
            if depth <= 0:
                continue
            if not (2 <= u < IMAGE_W - 2 and 2 <= v < IMAGE_H - 2):
                continue
            projected.setdefault(lid, {})[view] = (round_half_up(u), round_half_up(v))
    # 稳定 feature_id：每视图按 (y, x)
    fid_per_view = {view: {} for view in cameras}
    for view in sorted(cameras):
        rows = sorted(
            ((lid, projected[lid][view][0], projected[lid][view][1])
             for lid in projected if view in projected[lid]),
            key=lambda r: (r[2], r[1]),
        )
        for fid, (lid, u, v) in enumerate(rows):
            fid_per_view[view][lid] = fid
            features.append(Feature(view=view, feature_id=fid, x=float(u), y=float(v),
                                    landmark_id=lid, precision="float:v2(exact-int)"))
    # 相邻视图匹配（共享图元），确定性 NCC 千分比
    views = sorted(cameras)
    for i in range(len(views) - 1):
        va, vb = views[i], views[i + 1]
        common = sorted(
            lid for lid in projected
            if va in projected[lid] and vb in projected[lid]
        )
        for lid in common:
            fa = fid_per_view[va][lid]
            fb = fid_per_view[vb][lid]
            score = float(800 + (lid * 13) % 200)  # 固定千分比得分
            matches.append(Match(
                view_a=va, feature_a=fa, view_b=vb, feature_b=fb,
                score_permille=score, metric="ncc",
                metric_code=name_to_code("ncc"), precision="float:v2(exact-int)",
            ))
    # 体素点：场景点按 voxel_size 吸附
    for lid, point in scene:
        gx = snap_to_grid(point[0], voxel_size, config.fixed_point_bits)
        gy = snap_to_grid(point[1], voxel_size, config.fixed_point_bits)
        gz = snap_to_grid(point[2], voxel_size, config.fixed_point_bits)
        voxel_points.append(VoxelPoint(
            vx=round_half_up(point[0] / voxel_size),
            vy=round_half_up(point[1] / voxel_size),
            vz=round_half_up(point[2] / voxel_size),
            x=float(gx), y=float(gy), z=float(gz),
            landmark_id=lid, precision="float:v2(exact-fixed)",
        ))
    ir = ReconstructionIR(
        image_size=(IMAGE_W, IMAGE_H),
        voxel_size=voxel_size,
        fixed_point_bits=config.fixed_point_bits,
        features=features, matches=matches, voxel_points=voxel_points,
        rounding_mode="ROUND_HALF_UP", source_version="v2",
    )
    ir.sort_canonical()
    return ir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="samples/demo")
    ap.add_argument("--views", type=int, default=6)
    ap.add_argument("--seed", type=int, default=20261124)
    args = ap.parse_args()

    out = os.path.abspath(args.out)
    img_dir = os.path.join(out, "images")
    os.makedirs(img_dir, exist_ok=True)

    scene = make_scene(seed=args.seed)
    cameras = make_cameras(n_views=args.views)
    config = ReconConfig.from_dict({})

    for view, cam in cameras.items():
        image = render_view(cam, scene, seed=args.seed)
        save_png(os.path.join(img_dir, f"view_{view:02d}.png"), image)
    save_cameras(os.path.join(out, "cameras.json"), cameras)
    write_json(os.path.join(out, "recon_config.json"), config.to_dict())

    ir_v2 = build_exchange_ir(scene, cameras, config)
    ir_v2.validate()
    write_v2(os.path.join(out, "result_v2.json"), ir_v2)
    ir_v1, sidecar = downgrade_v2_to_v1(ir_v2)
    write_v1(os.path.join(out, "result_v1.txt"), ir_v1)
    write_json(os.path.join(out, "result_v1.sidecar.json"), sidecar)

    print(f"generated sample at {out}")
    print(f"  views={args.views} features={len(ir_v2.features)} "
          f"matches={len(ir_v2.matches)} voxels={len(ir_v2.voxel_points)} "
          f"failures={len(ir_v2.failures)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
