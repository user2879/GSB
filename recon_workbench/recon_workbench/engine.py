"""重建引擎：缓存调度、线程池并发、预算约束与快照提交。"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .artifact_store import (
    STAGES,
    ArtifactStore,
    CacheDecision,
)
from .config import RebuildConfig
from .errors import BizCode, FailureRecord, WorkbenchError
from .features import Feature
from .geometry import DepthMap, TriPoint
from .loader import LoadResult, ViewData, input_fingerprint
from .pipeline import (
    PairResult,
    PipelineResult,
    assemble_cloud,
    extract_view_features,
    run_pair,
    stable_task_key,
)


@dataclass
class RunContext:
    config: RebuildConfig
    precision: str
    store: ArtifactStore
    config_hash: str
    input_hash: str
    version: int
    failures: List[FailureRecord] = field(default_factory=list)
    decision: CacheDecision = field(default_factory=CacheDecision)
    budget_deductions: List[Dict[str, Any]] = field(default_factory=list)
    pending: List[Dict[str, Any]] = field(default_factory=list)
    injection_hits: List[Any] = field(default_factory=list)
    shutdown_requested: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class FeatureTask:
    stage = "feature"

    def __init__(self, view_data: ViewData) -> None:
        self.view_data = view_data
        self.key = stable_task_key(view_data.view, self.stage)

    def __call__(self, ctx: RunContext) -> List[Feature]:
        return extract_view_features(
            self.view_data, ctx.config, ctx.precision
        )


class PairTask:
    stage = "match"

    def __init__(self, view_a: ViewData, view_b: ViewData) -> None:
        self.view_a = view_a
        self.view_b = view_b
        self.key = stable_task_key(view_b.view, self.stage)

    def __call__(self, ctx: RunContext,
                 features: Dict[int, List[Feature]]) -> PairResult:
        return run_pair(
            self.view_a, self.view_b,
            features[self.view_a.view], features[self.view_b.view],
            ctx.config, ctx.precision,
        )


def _serialize_features(features: List[Feature]) -> List[Dict[str, Any]]:
    return [f.to_dict() for f in features]


def _serialize_pair(pair: PairResult) -> Dict[str, Any]:
    return {
        "view_a": pair.view_a, "view_b": pair.view_b,
        "matches": [m.to_dict() for m in pair.matches],
        "tri_points": [p.to_dict() for p in pair.tri_points],
        "depth_a": pair.depth_a.to_dict(),
        "depth_b": pair.depth_b.to_dict(),
        "boundary_match": pair.boundary_match,
        "below_inliers": pair.below_inliers,
        "match_key": pair.match_key,
        "depth_key_a": pair.depth_key_a,
        "depth_key_b": pair.depth_key_b,
    }


def _budget_remaining(ctx: RunContext) -> int:
    budget = ctx.config.rebuild_point_budget
    if budget < 0:
        return 1 << 30
    used = sum(item["granted"] for item in ctx.budget_deductions)
    return max(0, budget - used)


def _reserve_budget(ctx: RunContext, pair_label: str,
                    view: int, requested: int) -> int:
    """按稳定顺序在主线程串行扣减预算，返回实际可用额度。"""
    remaining = _budget_remaining(ctx)
    granted = min(requested, remaining)
    ctx.budget_deductions.append(
        {"task": pair_label, "view": view, "requested": requested,
         "granted": granted, "remaining_after": remaining - granted}
    )
    if granted < requested:
        ctx.pending.append(
            {"task": pair_label, "view": view,
             "code": BizCode.E_REBUILD_BUDGET_PENDING,
             "requested": requested, "granted": granted}
        )
    return granted


class WorkbenchEngine:
    """缓存驱动、快照提交的重建引擎，串行/并发共用同一编排逻辑。"""

    def __init__(self, load_result: LoadResult, config: RebuildConfig,
                 store: ArtifactStore, injection_hits: Optional[List[Any]] = None,
                 concurrency: Optional[int] = None) -> None:
        self.load_result = load_result
        self.config = config
        self.store = store
        self.views = sorted(load_result.views, key=lambda v: v.view)
        self.view_map = {v.view: v for v in self.views}
        self.injection_hits = injection_hits or []
        self.concurrency = concurrency if concurrency is not None else config.thread_count
        self.concurrency = min(self.concurrency, config.thread_count)
        self.config_hash = config.fingerprint()
        self.input_hash = input_fingerprint(self.views)
        self.last_context: Optional[RunContext] = None
        self._precision = "float64" 

    # ---- 缓存键派生 ----
    def _feature_fingerprints(self) -> Dict[int, str]:
        # 角点检测只依赖图像字节，特征键不纳入相机指纹，
        # 从而相机参数编辑不会错误地使特征缓存失效。
        return {v.view: v.image_fingerprint for v in self.views}

    def _feature_cache_keys(self) -> Dict[int, str]:
        fps = self._feature_fingerprints()
        precision = self._precision
        return {
            view: ArtifactStore.feature_key(
                fps[view], self.config_hash, precision
            )
            for view in fps
        }

    def _pair_cache_keys(self, feature_keys: Dict[int, str]
                         ) -> Dict[int, Dict[str, str]]:
        keys: Dict[int, Dict[str, str]] = {}
        ordered_views = [v.view for v in self.views]
        for index in range(len(ordered_views) - 1):
            va, vb = ordered_views[index], ordered_views[index + 1]
            mkey = ArtifactStore.match_key(
                feature_keys[va], feature_keys[vb], self.config_hash,
                self._precision,
            )
            dka = ArtifactStore.depth_key(
                mkey, self.view_map[va].camera_fingerprint,
                self.config_hash, self._precision,
            )
            dkb = ArtifactStore.depth_key(
                mkey, self.view_map[vb].camera_fingerprint,
                self.config_hash, self._precision,
            )
            keys[vb] = {"match": mkey, "depth_a": dka, "depth_b": dkb}
        return keys

    def run(self, precision: str, version: int,
            shutdown_event: Optional[threading.Event] = None,
            resume_manifests: Optional[Dict[str, Any]] = None
            ) -> PipelineResult:
        ctx = RunContext(
            config=self.config, precision=precision, store=self.store,
            config_hash=self.config_hash, input_hash=self.input_hash,
            version=version, injection_hits=self.injection_hits,
        )
        ctx.failures.extend(self.load_result.failures)
        self.last_context = ctx
        self._precision = precision
        feature_keys = self._feature_cache_keys()
        pair_keys = self._pair_cache_keys(feature_keys)

        features = self._stage_features(ctx, feature_keys, shutdown_event)
        if ctx.shutdown_requested:
            return self._partial_result(ctx, features, {}, [])
        pairs = self._stage_pairs(ctx, features, pair_keys, shutdown_event)
        if ctx.shutdown_requested:
            return self._partial_result(ctx, features, pairs, [])
        cloud = self._stage_cloud(ctx, pairs, pair_keys)
        result = PipelineResult(precision=precision)
        result.features = features
        result.pairs = pairs
        result.cloud = cloud
        result.failures = ctx.failures
        result.budget_deductions = ctx.budget_deductions
        result.pending = ctx.pending
        return result

    # ---- 阶段 1：特征（缓存复用优先，线程池仅用于重算项）----
    def _stage_features(self, ctx: RunContext,
                        feature_keys: Dict[int, str],
                        shutdown_event: Optional[threading.Event]
                        ) -> Dict[int, List[Feature]]:
        features: Dict[int, List[Feature]] = {}
        tasks: List[FeatureTask] = []
        for view_data in self.views:
            key = feature_keys[view_data.view]
            if self.store.has(key):
                payload = self.store.load(key)
                features[view_data.view] = [
                    self._restore_feature(item)
                    for item in payload["features"]
                ]
                ctx.decision.record(key, True)
            else:
                tasks.append(FeatureTask(view_data))
                ctx.decision.record(key, False)

        tasks.sort(key=lambda task: task.key)
        if tasks:
            executor = ThreadPoolExecutor(max_workers=self.concurrency)
            future_map = {
                executor.submit(task, ctx): task for task in tasks
            }
            for future, task in sorted(
                future_map.items(), key=lambda kv: kv[1].key
            ):
                if shutdown_event is not None and shutdown_event.is_set():
                    ctx.shutdown_requested = True
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                try:
                    feat = future.result()
                    features[task.view_data.view] = feat
                    self.store.save(
                        feature_keys[task.view_data.view],
                        {"view": task.view_data.view,
                         "features": _serialize_features(feat)},
                    )
                except WorkbenchError as exc:
                    exc.view = task.view_data.view
                    ctx.failures.append(exc.to_record())
            if not ctx.shutdown_requested:
                executor.shutdown(wait=True)

        # 阶段快照（整体替换）
        completed_views = sorted(features)
        manifest = self._manifest(ctx, "feature", completed_views)
        self.store.commit_stage(
            ctx.version, "feature",
            {f"view_{v:02d}.json": {
                "cache_key": feature_keys[v],
                "features": _serialize_features(features[v])}
             for v in completed_views},
            manifest,
        )
        return features

    # ---- 阶段 2/3：匹配、三角化与深度图 ----
    def _stage_pairs(self, ctx: RunContext,
                     features: Dict[int, List[Feature]],
                     pair_keys: Dict[int, Dict[str, str]],
                     shutdown_event: Optional[threading.Event]
                     ) -> List[PairResult]:
        ordered_views = [v.view for v in self.views]
        pairs: List[PairResult] = []
        tasks: List[PairTask] = []
        for index in range(len(ordered_views) - 1):
            va, vb = ordered_views[index], ordered_views[index + 1]
            if va not in features or vb not in features:
                ctx.failures.append(FailureRecord(
                    view=vb, stage="match",
                    code=BizCode.E_VIEW_INVALID,
                    reason="上游特征缺失，跳过重算该视图对",
                ))
                continue
            keys = pair_keys[vb]
            # match 与两个 depth 键独立判定；三者齐备才整体复用，
            # 任一缺失（如相机参数变化使深度键失效）则整对重算，
            # 保证 pair 内匹配/三角化/深度产物内部一致。
            all_cached = (
                self.store.has(keys["match"])
                and self.store.has(keys["depth_a"])
                and self.store.has(keys["depth_b"])
            )
            if all_cached:
                payload = self.store.load(keys["match"])
                pairs.append(self._deserialize_pair(payload))
                ctx.decision.record(keys["match"], True)
                ctx.decision.record(keys["depth_a"], True)
                ctx.decision.record(keys["depth_b"], True)
            else:
                tasks.append(PairTask(self.view_map[va], self.view_map[vb]))
                ctx.decision.record(keys["match"], False)
                ctx.decision.record(keys["depth_a"], False)
                ctx.decision.record(keys["depth_b"], False)

        tasks.sort(key=lambda task: task.key)
        if tasks:
            executor = ThreadPoolExecutor(max_workers=self.concurrency)
            future_map = {executor.submit(task, ctx, features): task
                          for task in tasks}
            for future, task in sorted(
                future_map.items(), key=lambda kv: kv[1].key
            ):
                if shutdown_event is not None and shutdown_event.is_set():
                    ctx.shutdown_requested = True
                    executor.shutdown(wait=False, cancel_futures=True)
                    break
                label = f"pair_{task.view_a.view:02d}_{task.view_b.view:02d}"
                # 预算在主线程按稳定顺序预扣，杜绝线程竞争导致的不确定
                requested = len(features[task.view_b.view])
                granted = _reserve_budget(
                    ctx, label, task.view_b.view, requested
                )
                if granted <= 0:
                    continue
                try:
                    pair = future.result()
                except WorkbenchError as exc:
                    exc.view = task.view_b.view
                    ctx.failures.append(exc.to_record())
                    continue
                keys = pair_keys[task.view_b.view]
                pair.match_key = keys["match"]
                pair.depth_key_a = keys["depth_a"]
                pair.depth_key_b = keys["depth_b"]
                self.store.save(keys["match"], _serialize_pair(pair))
                self.store.save(
                    keys["depth_a"], {"depth": pair.depth_a.to_dict()}
                )
                self.store.save(
                    keys["depth_b"], {"depth": pair.depth_b.to_dict()}
                )
                pairs.append(pair)
            if not ctx.shutdown_requested:
                executor.shutdown(wait=True)

        pairs.sort(key=lambda pair: (pair.view_a, pair.view_b))
        manifest = self._manifest(
            ctx, "match", sorted({p.view_b for p in pairs})
        )
        self.store.commit_stage(
            ctx.version, "match",
            {f"pair_{p.view_a:02d}_{p.view_b:02d}.json":
             _serialize_pair(p) for p in pairs},
            manifest,
        )
        depth_manifest = self._manifest(
            ctx, "depth",
            sorted({p.view_a for p in pairs} | {p.view_b for p in pairs}),
        )
        self.store.commit_stage(
            ctx.version, "depth",
            {f"pair_{p.view_a:02d}_{p.view_b:02d}.json":
             {"depth_a": p.depth_a.to_dict(),
              "depth_b": p.depth_b.to_dict()} for p in pairs},
            depth_manifest,
        )
        return pairs

    # ---- 阶段 4：体素级点云合并 ----
    def _stage_cloud(self, ctx: RunContext, pairs: List[PairResult],
                     pair_keys: Dict[int, Dict[str, str]]) -> List[Any]:
        from .geometry import CloudPoint

        depth_keys = sorted(
            key for keys in pair_keys.values()
            for key in (keys["depth_a"], keys["depth_b"])
        )
        cloud_key = ArtifactStore.cloud_key(
            depth_keys, self.config_hash, self._precision
        )
        if self.store.has(cloud_key):
            payload = self.store.load(cloud_key)
            ctx.decision.record(cloud_key, True)
            return [self._restore_cloud_point(item)
                    for item in payload["cloud"]]
        ctx.decision.record(cloud_key, False)
        try:
            cloud = assemble_cloud(pairs, self.config, ctx.precision)
        except WorkbenchError as exc:
            # 点云阶段异常（如零体素）隔离记录，返回空云，不污染上游产物
            exc.stage = "point_cloud"
            ctx.failures.append(exc.to_record())
            return []
        self.store.save(
            cloud_key,
            {"cloud_key": cloud_key,
             "cloud": [point.to_dict() for point in cloud]},
        )
        manifest = self._manifest(
            ctx, "point_cloud",
            sorted({v for p in pairs for v in (p.view_a, p.view_b)}),
        )
        self.store.commit_stage(
            ctx.version, "point_cloud",
            {"cloud.json": {"cloud_key": cloud_key,
                            "cloud": [point.to_dict() for point in cloud]}},
            manifest,
        )
        return cloud

    # ---- 辅助 ----
    def _manifest(self, ctx: RunContext, stage: str,
                  completed_views: List[int]) -> Dict[str, Any]:
        interval = max(1, ctx.config.snapshot_interval_views)
        # 检查点：按 snapshot_interval_views 切分的已完成视图批次，
        # 恢复时据此判断最近一次完整提交覆盖到哪个视图。
        checkpoints = [
            completed_views[i:i + interval]
            for i in range(0, len(completed_views), interval)
        ]
        return {
            "version": ctx.version,
            "stage": stage,
            "config_hash": ctx.config_hash,
            "input_hash": ctx.input_hash,
            "precision": ctx.precision,
            "completed_views": completed_views,
            "snapshot_interval_views": interval,
            "checkpoints": checkpoints,
        }

    def _restore_feature(self, item: Dict[str, Any]) -> Feature:
        if self._precision == "float32":
            item = dict(item)
            for field_name in ("x", "y", "response"):
                item[field_name] = float(
                    np.float32(item[field_name])
                )
        return Feature(**item)

    def _restore_cloud_point(self, item: Dict[str, Any]) -> Any:
        from .geometry import CloudPoint

        record = dict(item)
        if self._precision == "float32":
            for field_name in ("x", "y", "z", "confidence"):
                record[field_name] = float(np.float32(record[field_name]))
        return CloudPoint(**record)

    def _deserialize_pair(self, payload: Dict[str, Any]) -> PairResult:
        from .features import MatchRecord
        from .geometry import DepthMap, TriPoint

        is32 = self._precision == "float32"
        matches = []
        for item in payload["matches"]:
            record = dict(item)
            if is32:
                record["distance"] = float(np.float32(record["distance"]))
            matches.append(MatchRecord(**record))
        tri_points = []
        for item in payload["tri_points"]:
            record = dict(item)
            if is32:
                for field_name in ("x", "y", "z", "disparity",
                                   "confidence"):
                    record[field_name] = float(
                        np.float32(record[field_name])
                    )
            tri_points.append(TriPoint(**record))
        depth_dtype = np.float32 if self._precision == "float32" else np.float64
        depth_a = DepthMap(
            view=payload["depth_a"]["view"],
            width=payload["depth_a"]["width"],
            height=payload["depth_a"]["height"],
            levels=np.array(payload["depth_a"]["levels"], dtype=np.int64),
            depth_values=np.array(
                [np.nan if v is None else v
                 for v in payload["depth_a"]["depth_values"]],
                dtype=depth_dtype,
            ),
            summary=payload["depth_a"]["summary"],
        )
        depth_b = DepthMap(
            view=payload["depth_b"]["view"],
            width=payload["depth_b"]["width"],
            height=payload["depth_b"]["height"],
            levels=np.array(payload["depth_b"]["levels"], dtype=np.int64),
            depth_values=np.array(
                [np.nan if v is None else v
                 for v in payload["depth_b"]["depth_values"]],
                dtype=depth_dtype,
            ),
            summary=payload["depth_b"]["summary"],
        )
        return PairResult(
            view_a=payload["view_a"], view_b=payload["view_b"],
            matches=matches, tri_points=tri_points,
            depth_a=depth_a, depth_b=depth_b,
            match_key=payload.get("match_key", ""),
            depth_key_a=payload.get("depth_key_a", ""),
            depth_key_b=payload.get("depth_key_b", ""),
            boundary_match=payload.get("boundary_match", False),
            below_inliers=payload.get("below_inliers", False),
        )

    def _partial_result(self, ctx: RunContext,
                        features: Dict[int, List[Feature]],
                        pairs: List[PairResult],
                        cloud: List[Any]) -> PipelineResult:
        result = PipelineResult(precision=ctx.precision)
        result.features = features
        result.pairs = pairs
        result.cloud = cloud
        result.failures = ctx.failures
        result.budget_deductions = ctx.budget_deductions
        result.pending = ctx.pending
        return result
