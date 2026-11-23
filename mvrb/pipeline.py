"""重建管线编排：加载校验 -> 特征 -> 匹配 -> 三角化 -> 深度融合 -> 体素合并。

每个产物都由上游内容指纹 + 参数指纹派生缓存键；命中即复用、绝不重算。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .cache import ArtifactCache
from .camera import camera_fingerprint, load_cameras
from .config import ReconConfig
from .depth import build_all_depth_maps
from .errors import (
    E_MATCH_INLIERS,
    E_MISSING_CAMERA,
    FailureList,
)
from .features import ViewFeatures, extract_features, feature_cache_key, image_bytes_fingerprint
from .ir import Feature, Match, ReconstructionIR, VoxelPoint
from .match import match_cache_key, match_views, pair_id
from .metrics import name_to_code
from .scheduler import BudgetScheduler, ViewTask
from .triangulate import LandmarkPoint3D, triangulate_cache_key, triangulate_landmarks
from .utils import combine_fingerprint, fingerprint
from .voxel import VoxelRecord, merge_point_cloud, voxel_cache_key

STAGES = ("load", "features", "match", "triangulate", "depth", "voxel")


@dataclass
class PipelineResult:
    config: ReconConfig
    view_ids: list
    view_features: dict
    pair_matches: list
    points3d: list
    depth_maps: dict
    voxels: list
    ir: ReconstructionIR
    failures: FailureList
    cache_ledger: dict
    invalidated: list
    schedule_report: dict = field(default_factory=dict)
    fingerprints: dict = field(default_factory=dict)


class ReconstructionPipeline:
    def __init__(self, sample_dir: str, work_dir: str, force_serial: bool = False):
        self.sample_dir = sample_dir
        self.work_dir = work_dir
        self.cache = ArtifactCache(os.path.join(work_dir, "cache"))
        self.image_dir = os.path.join(sample_dir, "images")
        self._load_config = None
        self._load_image_size = (0, 0)
        self.force_serial = force_serial

    # ------------------------------------------------------------------
    def run(self) -> PipelineResult:
        self.cache.reset_ledger()
        failures = FailureList()
        invalidated: list = []

        config, cameras, view_ids, image_fps = self._stage_load(failures)
        cfg_fp = config.fingerprint()
        feat_param_fp = fingerprint({'feature_threshold': config.feature_threshold})
        match_param_fp = fingerprint({'match_min_inliers': config.match_min_inliers})
        tri_param_fp = fingerprint({'min_disparity': config.min_disparity})
        depth_param_fp = fingerprint({
            'depth_resolution': list(config.depth_resolution),
            'depth_quant_levels': config.depth_quant_levels,
            'point_budget': config.point_budget,
            'view_point_quota': config.view_point_quota,
            'max_workers': config.max_workers,
        })
        voxel_param_fp = fingerprint({
            'voxel_size': config.voxel_size,
            'fixed_point_bits': config.fixed_point_bits,
        })
        cam_fp = camera_fingerprint(cameras) if cameras else fingerprint({"no_camera": True})
        fps = {"config": cfg_fp, "camera": cam_fp, "images": image_fps}

        view_features = self._stage_features(
            view_ids, cameras, image_fps, config, feat_param_fp, cam_fp, failures, invalidated
        )
        pair_matches, match_fps = self._stage_match(
            view_ids, view_features, match_param_fp, failures, invalidated
        )
        points3d, points_fp = self._stage_triangulate(
            cameras, view_features, pair_matches, match_fps, cam_fp, tri_param_fp,
            failures, invalidated,
        )
        depth_maps, schedule_report = self._stage_depth(
            view_ids, points3d, config, points_fp, depth_param_fp, failures, invalidated,
        )
        voxels, voxel_fp = self._stage_voxel(
            points3d, config, points_fp, voxel_param_fp, failures, invalidated,
        )
        ir = self._build_ir(config, view_features, pair_matches, voxels, failures)
        ir.failures = failures
        ir.validate()

        return PipelineResult(
            config=config,
            view_ids=view_ids,
            view_features=view_features,
            pair_matches=pair_matches,
            points3d=points3d,
            depth_maps=depth_maps,
            voxels=voxels,
            ir=ir,
            failures=failures,
            cache_ledger=self.cache.ledger(),
            invalidated=invalidated,
            schedule_report=schedule_report,
            fingerprints={**fps, "points": points_fp, "voxel": voxel_fp,
                          "match": combine_fingerprint(*[m for m in match_fps.values()])
                          if match_fps else fingerprint({"empty": "match"})},
        )

    # ------------------------------------------------------------------
    def _stage_load(self, failures: FailureList):
        config = ReconConfig.load(os.path.join(self.sample_dir, "recon_config.json"))
        cam_path = os.path.join(self.sample_dir, "cameras.json")
        cameras = {}
        if os.path.exists(cam_path):
            cameras = load_cameras(cam_path)
        image_files = sorted(
            f for f in os.listdir(self.image_dir) if f.endswith(".png")
        )
        view_ids = [int(f.split("_")[1].split(".")[0]) for f in image_files]
        image_fps = {}
        for view, fname in zip(view_ids, image_files):
            image_fps[view] = image_bytes_fingerprint(os.path.join(self.image_dir, fname))
            if view not in cameras:
                failures.fail(
                    E_MISSING_CAMERA,
                    f"view {view} has no camera parameters",
                    view=view,
                    stage="load",
                    sort_key=f"load|{view:04d}|{E_MISSING_CAMERA}",
                )
        self._load_config = config
        if image_files:
            from .synthetic import load_png_gray
            for fname in image_files:
                try:
                    probe = load_png_gray(os.path.join(self.image_dir, fname))
                    self._load_image_size = (probe.shape[1], probe.shape[0])
                    break
                except Exception:
                    continue
        return config, cameras, sorted(view_ids), image_fps

    def _stage_features(self, view_ids, cameras, image_fps, config, cfg_fp, cam_fp,
                        failures, invalidated):
        out = {}
        for view in view_ids:
            img_path = os.path.join(self.image_dir, f"view_{view:02d}.png")
            key = feature_cache_key(view, image_fps[view], cam_fp, cfg_fp)
            cached = self.cache.get(key)
            if cached is not None:
                vf = ViewFeatures.from_dict(cached)
            else:
                invalidated.append({"stage": "features", "view": view, "cache_key": key})
                vf = extract_features(view, img_path, config.feature_threshold, failures)
                self.cache.put(key, vf.to_dict())
            out[view] = vf
        return out

    # ------------------------------------------------------------------
    def _stage_match(self, view_ids, view_features, cfg_fp, failures, invalidated):
        min_inliers = self._load_config.match_min_inliers
        feature_fps = {v: fingerprint(view_features[v].to_dict()) for v in view_ids}
        pair_matches = []
        match_fps = {}
        for i, va in enumerate(view_ids):
            for vb in view_ids[i + 1:]:
                pid = pair_id(va, vb)
                key = match_cache_key(pid, {va: feature_fps[va], vb: feature_fps[vb]}, cfg_fp)
                cached = self.cache.get(key)
                if cached is not None:
                    from .match import ViewPairMatch
                    pm = ViewPairMatch.from_dict(cached)
                else:
                    invalidated.append({"stage": "match", "pair": pid, "cache_key": key})
                    pm = match_views(view_features[va], view_features[vb], min_inliers)
                    self.cache.put(key, pm.to_dict())
                if pm.inliers < min_inliers:
                    failures.fail(
                        E_MATCH_INLIERS,
                        f"pair {pid} inliers {pm.inliers} < min {min_inliers}",
                        view=va,
                        stage="match",
                        sort_key=f"match|{pid}|{E_MATCH_INLIERS}",
                    )
                    continue
                pair_matches.append(pm)
                match_fps[pid] = fingerprint(pm.to_dict())
        pair_matches.sort(key=lambda m: m.sort_key)
        return pair_matches, match_fps

    def _stage_triangulate(self, cameras, view_features, pair_matches, match_fps,
                           cam_fp, cfg_fp, failures, invalidated):
        match_fp = combine_fingerprint(*sorted(match_fps.values())) if match_fps \
            else fingerprint({"empty": "match"})
        key = triangulate_cache_key(match_fp, cam_fp, cfg_fp)
        cached = self.cache.get(key)
        if cached is not None:
            points = [LandmarkPoint3D.from_dict(d) for d in cached]
        else:
            invalidated.append({"stage": "triangulate", "cache_key": key})
            points = triangulate_landmarks(
                cameras, view_features, pair_matches,
                self._load_config.min_disparity, failures,
            )
            self.cache.put(key, [p.to_dict() for p in points])
        fp = fingerprint([p.to_dict() for p in points])
        return points, fp

    def _stage_depth(self, view_ids, points3d, config, points_fp, depth_param_fp,
                     failures, invalidated):
        # 每个视图的可计入点数 = 该视图观测到的三维点数
        per_view_counts = {v: 0 for v in view_ids}
        for p in points3d:
            for (view, _fid, _u, _v, _d) in p.observations:
                if view in per_view_counts:
                    per_view_counts[view] += 1

        scheduler = BudgetScheduler(
            config.max_workers, config.point_budget, config.view_point_quota,
            force_serial=self.force_serial,
        )

        def build_one(view: int):
            key = combine_fingerprint("depth", str(view), points_fp, depth_param_fp)
            cached = self.cache.get(key)
            if cached is not None:
                from .depth import DepthMap
                return DepthMap.from_dict(cached)
            invalidated.append({"stage": "depth", "view": view, "cache_key": key})
            dm = build_all_depth_maps(
                points3d, [view], self._image_size,
                config.depth_resolution, config.depth_quant_levels,
            )[view]
            self.cache.put(key, dm.to_dict())
            return dm

        tasks = [
            ViewTask(view=view, sort_key=f"depth|{view:04d}",
                     work=lambda v=view: build_one(v),
                     point_count=per_view_counts[view])
            for view in view_ids
        ]
        report = scheduler.run(tasks, failures)
        depth_maps = {view: report.results[view] for view in report.order}
        return depth_maps, {
            "order": report.order,
            "exhausted_views": report.exhausted_views,
            "todo_queue": report.todo_queue,
            "global_exhausted": report.global_exhausted,
            "peak_concurrency": report.peak_concurrency,
            "budget_remaining": report.budget_remaining,
        }

    def _stage_voxel(self, points3d, config, points_fp, voxel_param_fp,
                     failures, invalidated):
        if config.voxel_size <= 0:
            key = voxel_cache_key(points_fp, voxel_param_fp)
            invalidated.append({"stage": "voxel", "cache_key": key})
            merge_point_cloud(points3d, config.voxel_size,
                              config.fixed_point_bits, failures)
            return [], fingerprint({"empty": "voxel"})
        key = voxel_cache_key(points_fp, voxel_param_fp)
        cached = self.cache.get(key)
        if cached is not None:
            records = [VoxelRecord.from_dict(d) for d in cached]
        else:
            invalidated.append({"stage": "voxel", "cache_key": key})
            records = merge_point_cloud(
                points3d, config.voxel_size, config.fixed_point_bits, failures,
            )
            self.cache.put(key, [r.to_dict() for r in records])
        return records, fingerprint([r.to_dict() for r in records])

    # ------------------------------------------------------------------
    def _build_ir(self, config, view_features, pair_matches, voxels, failures):
        features, matches, vpoints = [], [], []
        for view in sorted(view_features):
            for f in view_features[view].features:
                features.append(Feature(
                    view=view, feature_id=f.feature_id,
                    x=round(f.x, 6), y=round(f.y, 6),
                    landmark_id=f.landmark_id, precision="float:v2",
                ))
        for pm in pair_matches:
            for fa_id, fb_id, score, _lid in pm.pairs:
                matches.append(Match(
                    view_a=pm.view_a, feature_a=fa_id,
                    view_b=pm.view_b, feature_b=fb_id,
                    score_permille=score, metric=pm.metric,
                    metric_code=name_to_code(pm.metric), precision="float:v2",
                ))
        for r in voxels:
            lid = r.landmark_ids[0] if r.landmark_ids else -1
            vpoints.append(VoxelPoint(
                vx=r.vx, vy=r.vy, vz=r.vz, x=r.x, y=r.y, z=r.z,
                landmark_id=lid, precision="float:v2",
            ))
        ir = ReconstructionIR(
            image_size=self._image_size,
            voxel_size=config.voxel_size,
            fixed_point_bits=config.fixed_point_bits,
            features=features, matches=matches, voxel_points=vpoints,
            rounding_mode="ROUND_HALF_UP", source_version="v2",
        )
        ir.sort_canonical()
        return ir

    @property
    def _image_size(self) -> tuple:
        return self._load_image_size
