"""重建流水线编排：并发执行、精确失效复用、预算调度、快照提交与诊断报告。"""

from __future__ import annotations

import os
import signal
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from .budget import BudgetLedger
from .camera import load_camera, load_gray_png
from .cloud import cloud_artifact_key, merge_point_cloud
from .config import ReconstructionConfig
from .depth import build_depth_map, depth_artifact_key
from .editor import load_config
from .errors import StageError
from .features import extract_features, feature_artifact_key
from .injection import InjectionPlan
from .matching import match_and_triangulate, match_artifact_key
from .snapshot import SNAPSHOT_VERSION, STAGE_NAMES, SnapshotStore
from .util import (
    content_fingerprint,
    pair_sort_key,
    read_json,
    timed_stage,
    view_pair_id,
)


@dataclass
class RunOutcome:
    report: dict[str, Any]
    exit_code: int = 0
    stale_error: dict | None = None
    interrupted: bool = False


@dataclass
class _Runtime:
    cfg: ReconstructionConfig
    store: SnapshotStore
    injections: InjectionPlan
    reuse: list[dict] = field(default_factory=list)
    recompute: list[dict] = field(default_factory=list)
    failures: list[dict] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    stop_requested: bool = False
    graceful: bool = False
    crash_after_stage: str | None = None
    crash_after_view: int | None = None


class ReconstructionPipeline:
    def __init__(
        self,
        input_dir: str,
        snapshot_dir: str,
        injection_config: str | None = None,
        *,
        crash_env: dict[str, str] | None = None,
    ) -> None:
        self.input_dir = input_dir
        self.store = SnapshotStore(snapshot_dir)
        self.injections = InjectionPlan.load(injection_config)
        env = crash_env or os.environ
        self.crash_after_stage = env.get("MVR_CRASH_AFTER_STAGE")
        self.crash_after_view = (
            int(env["MVR_CRASH_AFTER_VIEW"]) if env.get("MVR_CRASH_AFTER_VIEW") else None
        )
        self.graceful_after_stage = env.get("MVR_GRACEFUL_AFTER_STAGE")
        self.graceful_after_view = (
            int(env["MVR_GRACEFUL_AFTER_VIEW"]) if env.get("MVR_GRACEFUL_AFTER_VIEW") else None
        )
        self.stop_flag = None
        self._last_parallax_injection = None

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------

    def run(self) -> RunOutcome:
        cfg = load_config(self.input_dir)
        runtime = _Runtime(
            cfg=cfg,
            store=self.store,
            injections=self.injections,
            crash_after_stage=self.crash_after_stage,
            crash_after_view=self.crash_after_view,
        )
        runtime.stop_flag = self.stop_flag
        config_fp = cfg.fingerprint()
        camera_fp = self._camera_bundle_fingerprint()
        recovery = {"resumable": False, "reason": "fresh", "recovery_view_id": None}
        stale_error = None
        try:
            recovery = self.store.check_resumable(config_fp, camera_fp)
        except StageError as exc:
            stale_error = exc.to_failure()
            stale_error["recovery_view_id"] = getattr(exc, "recovery_view_id", None)
            runtime.failures.append(stale_error)
            # 陈旧快照不可续跑：清理后全量重建，报告保留 E_SNAPSHOT_STALE 判定
            for stage in STAGE_NAMES:
                stage_dir = self.store.handle.stage_dir(stage)
                if os.path.isdir(stage_dir):
                    import shutil
                    shutil.rmtree(stage_dir)
            if os.path.exists(self.store.handle.manifest_path):
                os.remove(self.store.handle.manifest_path)

        cached = self.store.load_artifact_map() if (recovery["resumable"] and stale_error is None) else {}
        views, invalid_views = self._load_views(runtime)
        valid_ids = [view.view_id for view in views]
        view_map = {view.view_id: view for view in views}

        feature_param_fp = cfg.section_fingerprint(
            "feature_threshold", "per_view_quota", "fixed_point_frac_bits"
        )
        match_param_fp = cfg.section_fingerprint(
            "match_min_inliers", "triang_min_parallax_deg", "fixed_point_frac_bits"
        )
        depth_param_fp = cfg.section_fingerprint(
            "depth_resolution", "depth_quant_levels", "depth_smooth_passes",
            "fixed_point_frac_bits",
        )
        cloud_param_fp = cfg.section_fingerprint("voxel_size", "fixed_point_frac_bits")

        features = self._run_feature_stage(runtime, views, cached, feature_param_fp)
        if self._should_stop(runtime, "feature"):
            return self._finalize_interrupted(runtime, cfg, valid_ids, recovery, stale_error)

        all_candidate_ids = sorted(
            set(view_map) | {item["view_id"] for item in invalid_views
                             if item.get("view_id") is not None}
        )
        matches, requested_by_owner, ledger = self._run_matching_stage(
            runtime, view_map, features, cached, match_param_fp, all_candidate_ids
        )
        if self._should_stop(runtime, "matching"):
            return self._finalize_interrupted(runtime, cfg, valid_ids, recovery, stale_error)

        depths = self._run_depth_stage(
            runtime, view_map, matches, ledger, cached, depth_param_fp
        )
        if self._should_stop(runtime, "depth"):
            return self._finalize_interrupted(runtime, cfg, valid_ids, recovery, stale_error)

        cloud = self._run_cloud_stage(runtime, valid_ids, matches, depths, cached, cloud_param_fp)
        manifest = self._write_manifest(
            cfg, camera_fp, valid_ids, ledger, features, matches, depths, cloud,
            recovery_view_id=None,
        )
        report = self._build_report(
            runtime, cfg, valid_ids, invalid_views, features, matches, depths, cloud,
            ledger, recovery, stale_error, manifest,
        )
        return RunOutcome(report=report, stale_error=stale_error)

    # ------------------------------------------------------------------
    # 输入加载（逐视图隔离故障，不合规视图进失败清单，其余继续）
    # ------------------------------------------------------------------

    def _load_views(self, runtime: _Runtime):
        views = []
        invalid_views: list[dict] = []
        # 先扫描目录拿到候选视图序号（扫描级错误直接抛出）
        views_dir = os.path.join(self.input_dir, "views")
        candidate_ids = sorted(
            {
                int(name[5:9])
                for name in os.listdir(views_dir)
                if name.startswith("view_") and (name.endswith(".png") or name.endswith(".cam.json"))
            }
        )
        seen_ids: set[int] = set()
        for view_id in candidate_ids:
            if view_id in seen_ids:
                invalid_views.append({
                    "code": "E_VIEW_INVALID", "message": "视图序号不唯一",
                    "stage": "load", "view_id": view_id,
                })
                continue
            seen_ids.add(view_id)
            image_path = os.path.join(views_dir, f"view_{view_id:04d}.png")
            camera_path = os.path.join(views_dir, f"view_{view_id:04d}.cam.json")
            try:
                self.injections.require_camera(view_id, camera_path)
                camera = load_camera(camera_path, view_id)
                corrupted_path, injection_id = self._maybe_corrupt_png(view_id, image_path)
                try:
                    gray, raw_bytes = load_gray_png(corrupted_path, view_id)
                except StageError as decode_error:
                    if injection_id is not None:
                        decode_error.injection_id = injection_id
                    raise
                if gray.shape != (camera.height, camera.width):
                    raise StageError(
                        "E_VIEW_INVALID",
                        f"视图 {view_id} 分辨率 {gray.shape} 与相机参数不一致",
                        stage="load", view_id=view_id,
                    )
                from .camera import ViewInput
                views.append(
                    ViewInput(
                        view_id=view_id, image_path=image_path, camera_path=camera_path,
                        camera=camera, image_fingerprint=content_fingerprint(raw_bytes),
                        camera_fingerprint=camera.fingerprint(), gray=gray,
                    )
                )
            except StageError as exc:
                failure = exc.to_failure()
                invalid_views.append(failure)
                runtime.failures.append(failure)
        if not views:
            raise StageError("E_VIEW_INVALID", "不存在任何合法视图", stage="load")
        return views, invalid_views

    def _maybe_corrupt_png(self, view_id: int, image_path: str):
        items = [i for i in self.injections.items if i["type"] == "corrupt_png"
                 and i.get("view_id") == view_id]
        if not items:
            return image_path, None
        with open(image_path, "rb") as handle:
            raw = handle.read()
        corrupted, injection_id = self.injections.corrupt_image_bytes(view_id, raw)
        tmp_path = os.path.join(
            self.store.handle.tmp_root, f"corrupt-view-{view_id:04d}.png"
        )
        with open(tmp_path, "wb") as handle:
            handle.write(corrupted)
        return tmp_path, injection_id

    def _camera_bundle_fingerprint(self) -> str:
        views_dir = os.path.join(self.input_dir, "views")
        payload = []
        for name in sorted(os.listdir(views_dir)):
            if name.endswith(".cam.json"):
                payload.append({"name": name, "content": read_json(os.path.join(views_dir, name))})
        return content_fingerprint(payload)

    # ------------------------------------------------------------------
    # 特征阶段（线程池并发，稳定排序键，精确复用）
    # ------------------------------------------------------------------

    def _run_feature_stage(self, runtime, views, cached, param_fp):
        features: dict[int, dict] = {}
        tasks = []
        for view in sorted(views, key=lambda item: item.view_id):
            key = feature_artifact_key(
                view.view_id, view.image_fingerprint, view.camera_fingerprint, param_fp
            )
            cached_artifact = cached.get(f"feat-{view.view_id:04d}")
            if cached_artifact and cached_artifact.get("cache_key") == key:
                runtime.reuse.append({
                    "stage": "feature", "artifact_id": cached_artifact["artifact_id"],
                    "view_id": view.view_id, "cache_key": key,
                })
                features[view.view_id] = cached_artifact
            else:
                if cached_artifact:
                    runtime.recompute.append({
                        "stage": "feature", "artifact_id": cached_artifact["artifact_id"],
                        "view_id": view.view_id, "reason": "cache-key-mismatch",
                    })
                tasks.append((view, key))

        def worker(task):
            view, key = task
            self.injections.raise_feature_fault(view.view_id)
            artifact = extract_features(view.view_id, view.gray, runtime.cfg)
            artifact["cache_key"] = key
            artifact["upstream"] = [
                f"image:{view.view_id}:{view.image_fingerprint}",
                f"camera:{view.view_id}:{view.camera_fingerprint}",
                f"params:feature:{param_fp}",
            ]
            artifact["sort_key"] = ["feature", view.view_id]
            return artifact

        produced = []
        if tasks:
            with timed_stage(runtime.timings, "feature"):
                with ThreadPoolExecutor(max_workers=runtime.cfg.thread_workers) as pool:
                    futures = [pool.submit(worker, task) for task in tasks]
                    for future in futures:
                        try:
                            produced.append(future.result())
                        except StageError as exc:
                            runtime.failures.append(exc.to_failure())
            # 按视图序号稳定排序后落盘，线程调度顺序不影响最终顺序
            produced.sort(key=lambda artifact: artifact["view_id"])
            for artifact in produced:
                self.store.stage_artifact(artifact)
                features[artifact["view_id"]] = artifact
                runtime.recompute.append({
                    "stage": "feature", "artifact_id": artifact["artifact_id"],
                    "view_id": artifact["view_id"], "cache_key": artifact["cache_key"],
                })
            self.store.commit_stage("feature")
        return features

    # ------------------------------------------------------------------
    # 匹配三角化阶段（线程池并发 + 主线程确定性预算结算）
    # ------------------------------------------------------------------

    def _run_matching_stage(self, runtime, view_map, features, cached, param_fp,
                            all_candidate_ids=None):
        cfg = runtime.cfg
        pair_specs = self._resolve_pairs(cfg, sorted(view_map))
        ledger = BudgetLedger(cfg.global_point_budget, cfg.per_view_quota)
        for view_id in (all_candidate_ids if all_candidate_ids is not None
                        else sorted(view_map)):
            ledger.reserve(view_id)

        matches: dict[str, dict] = {}
        tasks = []
        for view_a, view_b in pair_specs:
            pair_id = view_pair_id(view_a, view_b)
            if view_a not in features or view_b not in features:
                continue
            key = match_artifact_key(
                pair_id,
                features[view_a]["cache_key"], features[view_b]["cache_key"],
                view_map[view_a].camera_fingerprint,
                view_map[view_b].camera_fingerprint, param_fp,
            )
            cached_artifact = cached.get(f"match-{pair_id}")
            if cached_artifact and cached_artifact.get("cache_key") == key:
                runtime.reuse.append({
                    "stage": "matching", "artifact_id": cached_artifact["artifact_id"],
                    "pair_id": pair_id, "cache_key": key,
                })
                matches[pair_id] = cached_artifact
            else:
                tasks.append((view_a, view_b, pair_id, key))

        def worker(task):
            view_a, view_b, pair_id, key = task
            injection_id = self.injections.parallax_fault(pair_id)
            if injection_id is not None:
                self._last_parallax_injection = injection_id
            artifact = match_and_triangulate(
                view_a, view_b, features[view_a], features[view_b],
                view_map[view_a].camera, view_map[view_b].camera, cfg,
                force_parallax_fault=injection_id is not None,
            )
            artifact["cache_key"] = key
            artifact["upstream"] = [
                features[view_a]["artifact_id"], features[view_b]["artifact_id"],
                f"camera:{view_a}:{view_map[view_a].camera_fingerprint}",
                f"camera:{view_b}:{view_map[view_b].camera_fingerprint}",
                f"params:matching:{param_fp}",
            ]
            artifact["sort_key"] = ["matching", *pair_sort_key(pair_id)]
            return artifact

        produced: list[dict] = []
        if tasks:
            with timed_stage(runtime.timings, "matching"):
                with ThreadPoolExecutor(max_workers=cfg.thread_workers) as pool:
                    futures = [pool.submit(worker, task) for task in tasks]
                    for future in futures:
                        try:
                            artifact = future.result()
                        except StageError as exc:
                            if exc.injection_id is None:
                                exc.injection_id = self._last_parallax_injection
                            runtime.failures.append(exc.to_failure())
                            continue
                        produced.append(artifact)

        # 结果按视图对标识稳定排序后落盘，线程调度顺序不影响最终顺序
        produced.sort(key=lambda item: pair_sort_key(item["pair_id"]))
        for artifact in produced:
            self.store.stage_artifact(artifact)
            matches[artifact["pair_id"]] = artifact
            runtime.recompute.append({
                "stage": "matching", "artifact_id": artifact["artifact_id"],
                "pair_id": artifact["pair_id"], "cache_key": artifact["cache_key"],
            })
        if produced:
            self.store.commit_stage("matching")

        # 主线程按视图序号稳定顺序结算预算，并按预算截断点集
        requested_by_owner = self._settle_budget(runtime, ledger, matches)
        return matches, requested_by_owner, ledger

    def _settle_budget(self, runtime, ledger: BudgetLedger, matches: dict[str, dict]):
        requested_by_owner: dict[int, int] = {view_id: 0 for view_id in ledger.queue_order}
        for pair_id in sorted(matches, key=pair_sort_key):
            artifact = matches[pair_id]
            content = artifact["content"]
            if not content.get("accepted"):
                continue
            for point in content["points"]:
                requested_by_owner[point["owner_view_id"]] = (
                    requested_by_owner.get(point["owner_view_id"], 0) + 1
                )

        accepted_keys: set[str] = set()
        for view_id in ledger.queue_order:
            requested = requested_by_owner.get(view_id, 0)
            accepted, exhausted, _ = ledger.settle(view_id, requested)
            if exhausted:
                runtime.failures.append(ledger.budget_failure(view_id))
            # 该视图拥有的点按 point_key 稳定排序后截断到接受数量
            owned = sorted(
                (point for pair_id in sorted(matches, key=pair_sort_key)
                 for point in matches[pair_id]["content"]["points"]
                 if point["owner_view_id"] == view_id),
                key=lambda point: point["point_key"],
            )
            for point in owned[:accepted]:
                accepted_keys.add(point["point_key"])

        # 预算截断写回匹配产物（确定性内容，纳入指纹下游）
        for pair_id, artifact in matches.items():
            content = artifact["content"]
            if not content.get("accepted"):
                continue
            kept = [point for point in content["points"] if point["point_key"] in accepted_keys]
            kept_keys = {point["point_key"] for point in kept}
            content["points"] = kept
            content["matches"] = [m for m in content["matches"] if m["point_key"] in kept_keys]
            # 预算截断是运行时调度结果，放在指纹之外的元数据，避免污染下游缓存键
            artifact["budget_accepted_count"] = len(kept)
            self.store.stage_artifact(artifact)
        if any(a["content"].get("accepted") for a in matches.values()):
            self.store.commit_stage("matching")
        return requested_by_owner

    def _resolve_pairs(self, cfg: ReconstructionConfig, valid_ids: list[int]):
        if cfg.view_pairs:
            valid = set(valid_ids)
            pairs = sorted({tuple(sorted(pair)) for pair in cfg.view_pairs
                            if pair[0] in valid and pair[1] in valid})
        else:
            pairs = list(zip(valid_ids, valid_ids[1:]))
        return pairs

    # ------------------------------------------------------------------
    # 深度图阶段（线程池并发，仅消费预算接受的匹配产物）
    # ------------------------------------------------------------------

    def _run_depth_stage(self, runtime, view_map, matches, ledger, cached, param_fp):
        cfg = runtime.cfg
        match_list = [matches[pair_id] for pair_id in sorted(matches, key=pair_sort_key)]
        depths: dict[int, dict] = {}
        tasks = []
        for view_id in sorted(view_map):
            upstream_keys = sorted({
                artifact["cache_key"]
                for artifact in match_list
                if artifact["content"].get("accepted")
                and view_id in (artifact["content"]["view_a"], artifact["content"]["view_b"])
                and any(view_id in point["source_view_ids"]
                        for point in artifact["content"]["points"])
            })
            key = depth_artifact_key(view_id, upstream_keys, param_fp)
            cached_artifact = cached.get(f"depth-{view_id:04d}")
            if cached_artifact and cached_artifact.get("cache_key") == key:
                runtime.reuse.append({
                    "stage": "depth", "artifact_id": cached_artifact["artifact_id"],
                    "view_id": view_id, "cache_key": key,
                })
                depths[view_id] = cached_artifact
            else:
                tasks.append((view_id, key, upstream_keys))

        def worker(task):
            view_id, key, upstream_keys = task
            artifact = build_depth_map(
                view_id, view_map[view_id].camera, match_list, cfg
            )
            artifact["cache_key"] = key
            artifact["upstream"] = [
                *(f"match:{up_key}" for up_key in upstream_keys),
                f"camera:{view_id}:{view_map[view_id].camera_fingerprint}",
                f"params:depth:{param_fp}",
            ]
            artifact["sort_key"] = ["depth", view_id]
            return artifact

        produced = []
        if tasks:
            with timed_stage(runtime.timings, "depth"):
                with ThreadPoolExecutor(max_workers=cfg.thread_workers) as pool:
                    produced = list(pool.map(worker, tasks))
            produced.sort(key=lambda item: item["view_id"])
            for artifact in produced:
                self.store.stage_artifact(artifact)
                depths[artifact["view_id"]] = artifact
                runtime.recompute.append({
                    "stage": "depth", "artifact_id": artifact["artifact_id"],
                    "view_id": artifact["view_id"], "cache_key": artifact["cache_key"],
                })
            self.store.commit_stage("depth")
        return depths

    # ------------------------------------------------------------------
    # 体素点云合并（串行，顺序与按视图序号串行执行一致）
    # ------------------------------------------------------------------

    def _run_cloud_stage(self, runtime, valid_ids, matches, depths, cached, param_fp):
        depth_keys = sorted(depths[view_id]["cache_key"] for view_id in sorted(depths))
        key = cloud_artifact_key(depth_keys, param_fp)
        cached_artifact = cached.get("cloud-global")
        if cached_artifact and cached_artifact.get("cache_key") == key:
            runtime.reuse.append({
                "stage": "cloud", "artifact_id": "cloud-global", "cache_key": key,
            })
            return cached_artifact

        with timed_stage(runtime.timings, "cloud"):
            artifact = merge_point_cloud(
                valid_ids,
                [matches[pair_id] for pair_id in sorted(matches, key=pair_sort_key)],
                runtime.cfg,
            )
        artifact["cache_key"] = key
        artifact["upstream"] = [
            *(f"depth:{depth_key}" for depth_key in depth_keys), f"params:cloud:{param_fp}",
        ]
        artifact["sort_key"] = ["cloud"]
        self.store.stage_artifact(artifact)
        self.store.commit_stage("cloud")
        runtime.recompute.append({
            "stage": "cloud", "artifact_id": "cloud-global", "cache_key": key,
        })
        return artifact

    # ------------------------------------------------------------------
    # 中断 / 强制终止控制
    # ------------------------------------------------------------------

    def _should_stop(self, runtime: _Runtime, stage: str) -> bool:
        if getattr(runtime, "stop_flag", None) is not None and runtime.stop_flag.get("stop"):
            runtime.stop_requested = True
        if runtime.stop_requested:
            runtime.graceful = True
            return True
        if self.graceful_after_stage == stage and (
            self.graceful_after_view is None
            or self._last_committed_view(runtime, stage) >= self.graceful_after_view
        ):
            runtime.stop_requested = True
            runtime.graceful = True
            return True
        if self.crash_after_stage == stage and (
            self.crash_after_view is None
            or self._last_committed_view(runtime, stage) >= self.crash_after_view
        ):
            # 模拟强制终止：不写清单、不生成退出报告，直接退出进程
            os._exit(137)
        return False

    def _last_committed_view(self, runtime: _Runtime, stage: str) -> int:
        if stage == "feature":
            ids = [item.get("view_id") for item in runtime.recompute
                   if item["stage"] == "feature" and item.get("view_id") is not None]
        elif stage == "depth":
            ids = [item.get("view_id") for item in runtime.recompute
                   if item["stage"] == "depth" and item.get("view_id") is not None]
        else:
            ids = []
        return max(ids) if ids else -1

    def _finalize_interrupted(self, runtime, cfg, valid_ids, recovery, stale_error):
        manifest = self._write_manifest(
            cfg, self._camera_bundle_fingerprint(), valid_ids,
            BudgetLedger(cfg.global_point_budget, cfg.per_view_quota),
            {}, {}, {}, None, recovery_view_id=self._recovery_point(runtime),
        )
        report = self._build_report(
            runtime, cfg, valid_ids, [], {}, {}, {}, None,
            BudgetLedger(cfg.global_point_budget, cfg.per_view_quota),
            recovery, stale_error, manifest,
        )
        report["interrupted"] = True
        report["recovery_point_view_id"] = self._recovery_point(runtime)
        return RunOutcome(report=report, exit_code=0, interrupted=True)

    def _recovery_point(self, runtime: _Runtime) -> int:
        completed = sorted({
            item["view_id"] for item in runtime.recompute + runtime.reuse
            if item.get("view_id") is not None
        })
        return completed[-1] if completed else 0

    # ------------------------------------------------------------------
    # 清单与诊断报告
    # ------------------------------------------------------------------

    def _write_manifest(self, cfg, camera_fp, valid_ids, ledger,
                        features, matches, depths, cloud, *, recovery_view_id):
        stage_versions = {
            "feature": sorted(a["artifact_id"] for a in features.values()),
            "matching": sorted(a["artifact_id"] for a in matches.values()),
            "depth": sorted(a["artifact_id"] for a in depths.values()),
            "cloud": [cloud["artifact_id"]] if cloud else [],
        }
        manifest = {
            "snapshot_version": SNAPSHOT_VERSION,
            "config_fingerprint": cfg.fingerprint(),
            "camera_bundle_fingerprint": camera_fp,
            "completed_views": list(valid_ids),
            "consumed_point_budget": ledger.consumed,
            "stage_versions": stage_versions,
            "recovery_view_id": recovery_view_id,
        }
        self.store.write_manifest(manifest)
        return manifest

    def _build_report(self, runtime, cfg, valid_ids, invalid_views,
                      features, matches, depths, cloud, ledger,
                      recovery, stale_error, manifest):
        view_summaries = []
        # 仅报告成功完成特征阶段的视图；故障隔离掉的视图进入 failures 而非 views
        successful_ids = sorted(features)
        for view_id in successful_ids:
            artifact = features.get(view_id)
            depth_artifact = depths.get(view_id)
            view_summaries.append({
                "view_id": view_id,
                "feature_count": len(artifact["content"]["features"]) if artifact else 0,
                "feature_fingerprint": artifact["content_fingerprint"] if artifact else None,
                "depth_cells": len(depth_artifact["content"]["entries"]) if depth_artifact else 0,
                "depth_fingerprint": depth_artifact["content_fingerprint"] if depth_artifact else None,
                "sort_key": ["feature", view_id],
            })
        pair_summaries = []
        for pair_id in sorted(matches, key=pair_sort_key):
            content = matches[pair_id]["content"]
            pair_summaries.append({
                "pair_id": pair_id,
                "accepted": content.get("accepted", False),
                "inlier_count": content.get("inlier_count", 0),
                "point_count": len(content.get("points", [])),
                "match_fingerprint": matches[pair_id]["content_fingerprint"],
                "sort_key": ["matching", *pair_sort_key(pair_id)],
            })
        report = {
            "snapshot_version": SNAPSHOT_VERSION,
            "config_fingerprint": cfg.fingerprint(),
            "views": view_summaries,
            "pairs": pair_summaries,
            "cloud": {
                "point_count": cloud["content"]["point_count"] if cloud else 0,
                "point_order": [item["point_key"] for item in
                                cloud["content"]["representatives"]] if cloud else [],
                "cloud_fingerprint": cloud["content_fingerprint"] if cloud else None,
            },
            "invalidation": {
                "reused": sorted(runtime.reuse, key=lambda item: json_sort_key(item)),
                "recomputed": sorted(runtime.recompute, key=lambda item: json_sort_key(item)),
            },
            "budget_ledger": ledger.to_dict(),
            "recovery": {
                "resumable": recovery.get("resumable", False),
                "reason": recovery.get("reason"),
                "recovery_view_id": recovery.get("recovery_view_id"),
            },
            "failures": sorted(runtime.failures, key=lambda item: json_sort_key(item)),
            "injections": self.injections.report(),
            "stage_timings_ms": {name: round(runtime.timings.get(name, 0.0), 6)
                                 for name in STAGE_NAMES},
            "manifest": manifest,
        }
        return report


def json_sort_key(item: dict):
    return tuple(
        str(item.get(name, "")) for name in ("stage", "view_id", "pair_id", "artifact_id")
    )


def install_signal_handlers(flag_holder: dict[str, bool]) -> None:
    """收到可处理退出信号时置位：流水线在当前视图阶段产物落盘后以 0 退出。"""

    def handler(signum, frame):  # pragma: no cover - 由信号测试覆盖
        flag_holder["stop"] = True

    for signame in ("SIGTERM", "SIGINT"):
        if hasattr(signal, signame):
            signal.signal(getattr(signal, signame), handler)
