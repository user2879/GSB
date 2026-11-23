"""工作台门面：把管线、快照、异常注入、跨版本诊断编排为一次完整运行。"""
from __future__ import annotations

import os

from .camera import camera_fingerprint, load_cameras
from .config import ReconConfig
from .diagnostics import cross_version_report
from .errors import FailureList
from .pipeline import ReconstructionPipeline
from .snapshot import SnapshotStore, STAGE_ORDER
from .utils import ensure_dir, fingerprint


class Workbench:
    def __init__(self, sample_dir: str, work_dir: str, force_serial: bool = False):
        self.sample_dir = sample_dir
        self.work_dir = ensure_dir(work_dir)
        self.force_serial = force_serial
        self.snapshot = SnapshotStore(os.path.join(work_dir, "snapshot"))

    def _fingerprints(self) -> tuple:
        config = ReconConfig.load(os.path.join(self.sample_dir, "recon_config.json"))
        cam_path = os.path.join(self.sample_dir, "cameras.json")
        cam_fp = (camera_fingerprint(load_cameras(cam_path))
                  if os.path.exists(cam_path) else fingerprint({"no_camera": True}))
        # 交换版本指纹：v1/v2 文件内容共同决定
        exchange_payload = {}
        for name in ("result_v1.txt", "result_v2.json"):
            path = os.path.join(self.sample_dir, name)
            if os.path.exists(path):
                with open(path, "rb") as fh:
                    exchange_payload[name] = fh.read().hex()
        exchange_fp = fingerprint(exchange_payload)
        return config, config.fingerprint(), cam_fp, exchange_fp

    def run(self) -> dict:
        failures = FailureList()
        config, cfg_fp, cam_fp, exchange_fp = self._fingerprints()
        start_stage, old_manifest = self.snapshot.check_resume(
            cfg_fp, cam_fp, exchange_fp, failures,
        )

        pipeline = ReconstructionPipeline(
            self.sample_dir, os.path.join(self.work_dir, "pipeline"),
            force_serial=self.force_serial,
        )
        result = pipeline.run()
        for f in result.failures.sorted():
            failures.add(f)

        # 阶段化产物整体替换提交
        stage_payloads = {
            "load": {
                "view_ids": result.view_ids,
                "config": config.to_dict(),
                "failures": [f.to_dict() for f in result.failures.sorted()
                             if f.stage == "load"],
            },
            "features_match": {
                "features": {v: vf.to_dict() for v, vf in result.view_features.items()},
                "matches": [m.to_dict() for m in result.pair_matches],
            },
            "triangulate": {"points": [p.to_dict() for p in result.points3d]},
            "depth_voxel": {
                "depth": {v: d.to_dict() for v, d in result.depth_maps.items()},
                "voxels": [v.to_dict() for v in result.voxels],
            },
        }
        stage_versions = {}
        commit_actions = {}
        for stage in STAGE_ORDER:
            payload = stage_payloads[stage]
            stage_version = fingerprint(payload)
            wrapped = {"__stage_version__": stage_version, "data": payload}
            commit_actions[stage] = self.snapshot.commit_stage(stage, wrapped, stage_version)
            stage_versions[stage] = stage_version

        manifest = self.snapshot.build_manifest(
            cfg_fp, cam_fp, exchange_fp, result.view_ids, stage_versions,
        )
        self.snapshot.save_manifest(manifest)

        diagnostics = cross_version_report(result.ir)

        report = {
            "sample_dir": self.sample_dir,
            "work_dir": self.work_dir,
            "force_serial": self.force_serial,
            "resume": {
                "start_stage": start_stage,
                "had_manifest": old_manifest is not None,
                "stale": start_stage is not None and old_manifest is not None,
            },
            "stage_commit_actions": commit_actions,
            "manifest": manifest,
            "cache_ledger": result.cache_ledger,
            "invalidated": result.invalidated,
            "schedule": result.schedule_report,
            "failures": failures.dicts(),
            "diagnostics": diagnostics,
            "counts": {
                "views": len(result.view_ids),
                "features": len(result.ir.features),
                "matches": len(result.ir.matches),
                "points3d": len(result.points3d),
                "depth_maps": len(result.depth_maps),
                "voxels": len(result.voxels),
            },
        }
        return report
