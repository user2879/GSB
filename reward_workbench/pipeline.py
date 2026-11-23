"""增量流水线引擎：批次特征/打分/边界/汇总，带精确失效、预算与快照。

同一份引擎同时服务：
- 增量运行：从初始状态逐命令演进，复用未失效缓存；
- 全量重建：从空白缓存对最终状态一次性计算。
两者产出的规范化结果逐字段一致（见 tests/test_equivalence.py）。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .cache import ArtifactCache
from .commands import Command, apply_command
from .config import AppConfig
from .errors import FailureEntry, FailureLog
from .executor import run_tasks, stable_sort_key
from .pairs import PreferencePair
from .scheduler import BudgetState, Scheduler
from .snapshot import Manifest, SnapshotStore, fingerprint_config, fingerprint_pairs
from .stages import (
    BoundaryResult, FeatureResult, ScoreResult,
    resolve_tie_ranks, run_boundary, run_feature, run_score, summarize_batch,
)


@dataclass
class RunReport:
    failures: FailureLog
    invalidation_events: list[dict[str, Any]] = field(default_factory=list)
    reuse_events: list[dict[str, Any]] = field(default_factory=list)
    batch_plans: list[dict[str, Any]] = field(default_factory=list)
    boundary_list: list[dict[str, Any]] = field(default_factory=list)
    score_breakdown: list[dict[str, Any]] = field(default_factory=list)
    summaries: list[dict[str, Any]] = field(default_factory=list)
    budget_ledger: dict[str, Any] = field(default_factory=dict)

    def canonical_results(self, cache: ArtifactCache) -> dict[str, Any]:
        """逐字段比对用的规范化最终结果。"""
        features, scores, boundaries = [], [], []
        for pid in sorted(cache.feature.pair_index):
            feat: FeatureResult = cache.feature.items[cache.feature.pair_index[pid]]
            features.append({
                "pair_id": pid,
                "bucket_digest": feat.bucket_digest(),
                "nonzero_prompt": feat.prompt_vec.counts,
                "nonzero_chosen": feat.chosen_vec.counts,
                "nonzero_rejected": feat.rejected_vec.counts,
                "referenced_buckets": feat.referenced_buckets,
            })
        for pid in sorted(cache.score.pair_index):
            sc: ScoreResult = cache.score.items[cache.score.pair_index[pid]]
            scores.append(sc.to_dict())
        for pid in sorted(cache.boundary.pair_index):
            bd: BoundaryResult = cache.boundary.items[cache.boundary.pair_index[pid]]
            boundaries.append(bd.to_dict())
        summaries = []
        for key in sorted(cache.summary_items):
            summaries.append(cache.summary_items[key].to_dict())
        return {
            "features": features,
            "scores": scores,
            "boundaries": boundaries,
            "summaries": summaries,
            "cache_manifest": cache.canonical_manifest(),
            "failures": self.failures.to_list(),
        }

    def data_artifacts(self, cache: ArtifactCache) -> dict[str, Any]:
        """纯数据产物口径：只含由 (最终偏好对集合, 最终配置) 决定的内容。

        失败清单/失效事件/预算账本属于有状态运维历史，增量运行与一次性
        全量重建天然不同，故不纳入数据等价判定，单独在报告中呈现。
        """
        full = self.canonical_results(cache)
        full.pop("failures", None)
        return full


@dataclass
class _FullPlan:
    batch_id: str
    process_pairs: list[PreferencePair]
    newly_charged: list[PreferencePair]
    deferred: list[PreferencePair]
    budget_blocked: list[PreferencePair]
    charged_count: int
    quota_used: int
    global_consumed: int
    global_remaining: int


class Pipeline:
    def __init__(self, cfg: AppConfig, pairs: dict[str, PreferencePair],
                 snapshot_store: SnapshotStore | None = None,
                 parallel: bool = True,
                 budget: BudgetState | None = None,
                 global_version: int = 0,
                 full_compute: bool = False) -> None:
        self.cfg = cfg
        self.pairs = pairs
        self.cache = ArtifactCache()
        self.parallel = parallel
        self.budget = budget or BudgetState(global_budget=cfg.global_pair_budget)
        self.snapshot_store = snapshot_store
        self.global_version = global_version
        # 全量验证模式：表达“最终数据的完整计算结果”，绕过预算/配额截断
        self.full_compute = full_compute
        self.report = RunReport(failures=FailureLog())
        self._processed: set[str] = set()  # 已完成阶段计算的对（本引擎生命周期）

    # ------------------------------------------------------------------
    def _data_fingerprint(self) -> str:
        payload = {pid: p.to_dict() for pid, p in sorted(self.pairs.items())}
        return fingerprint_pairs(payload)

    def _config_hash(self) -> str:
        return fingerprint_config(self.cfg.to_dict())

    # ------------------------------------------------------------------
    def invalidate(self, changed_pairs: set[str], reason: str) -> None:
        if not changed_pairs:
            return
        removed = self.cache.cascade_invalidate(changed_pairs)
        self.report.invalidation_events.append({
            "reason": reason,
            "changed_pairs": sorted(changed_pairs),
            "removed_cache_keys": removed,
        })
        # 已删除/被替换但内容指纹变化的对需要重算
        self._processed.difference_update(
            pid for pid in changed_pairs if pid not in self.cache.feature.pair_index
        )

    # ------------------------------------------------------------------
    def _parallel_stage(self, stage: str, pair_list: list[PreferencePair],
                        fn) -> dict[str, Any]:
        tasks = [(stable_sort_key(p.pair_id, stage), (lambda pp=p: fn(pp)))
                 for p in pair_list]
        results = run_tasks(tasks, self.cfg.max_workers, self.parallel)
        out: dict[str, Any] = {}
        for key, value in results:
            pid = key.split("::", 1)[1]
            if isinstance(value, Exception):
                self.report.failures.add(FailureEntry(
                    sort_key=pid, code="E_PAIR_INVALID", stage=stage,
                    message=f"{type(value).__name__}: {value}"))
                continue
            out[pid] = value
        return out

    # ------------------------------------------------------------------
    def process_pairs(self, pair_list: list[PreferencePair]) -> None:
        """对给定对集合做 feature/score/boundary，精确复用未失效缓存。"""
        pair_list = sorted(pair_list, key=lambda p: p.pair_id)
        if not pair_list:
            return

        # ---- 阶段 1：feature（复用判定） ----
        need_feature, reused = [], []
        for pair in pair_list:
            cached = self.cache.feature_for(pair.pair_id)
            if cached is not None and cached.cache_key == self._want_feature_key(pair):
                reused.append(pair.pair_id)
            else:
                need_feature.append(pair)
        if reused:
            self.report.reuse_events.append({
                "stage": "feature", "pair_ids": sorted(reused),
            })
        features = self._parallel_stage("feature", need_feature,
                                        lambda p: run_feature(p, self.cfg))
        for pid, feat in features.items():
            self.cache.put_feature(feat)

        # ---- 阶段 2：score ----
        all_feats = {p.pair_id: self.cache.feature_for(p.pair_id) for p in pair_list}
        need_score, reused = [], []
        for pair in pair_list:
            feat = all_feats[pair.pair_id]
            want_key = self._want_score_key(feat)
            cached = self.cache.score_for(pair.pair_id)
            if cached is not None and cached.cache_key == want_key:
                reused.append(pair.pair_id)
            else:
                need_score.append(pair)
        if reused:
            self.report.reuse_events.append({"stage": "score", "pair_ids": sorted(reused)})
        scores = self._parallel_stage(
            "score", need_score,
            lambda p: run_score(p, all_feats[p.pair_id], self.cfg),
        )
        for pid, sc in scores.items():
            self.cache.put_score(sc)

        # ---- 阶段 3：boundary（需要全局 tie_rank，先算再判） ----
        score_map = {p.pair_id: self.cache.score_for(p.pair_id) for p in pair_list}
        tie_ranks = resolve_tie_ranks(pair_list, score_map, self.cfg)
        need_boundary, reused = [], []
        for pair in pair_list:
            sc = score_map[pair.pair_id]
            want_key = self._want_boundary_key(sc)
            cached = self.cache.boundary_for(pair.pair_id)
            if cached is not None and cached.cache_key == want_key:
                reused.append(pair.pair_id)
            else:
                need_boundary.append(pair)
        if reused:
            self.report.reuse_events.append({
                "stage": "boundary", "pair_ids": sorted(reused),
            })
        for pair in need_boundary:
            sc = score_map[pair.pair_id]
            bd = run_boundary(pair, sc, self.cfg, tie_ranks.get(pair.pair_id, 0))
            self.cache.put_boundary(bd)

        for pair in pair_list:
            self._processed.add(pair.pair_id)

    def _want_feature_key(self, pair: PreferencePair) -> str:
        from .stages import feature_cache_key
        return feature_cache_key(pair, self.cfg)

    def _want_score_key(self, feat: FeatureResult) -> str:
        from .stages import score_cache_key
        return score_cache_key(feat, self.cfg)

    def _want_boundary_key(self, sc: ScoreResult) -> str:
        from .stages import boundary_cache_key
        return boundary_cache_key(sc, self.cfg)

    # ------------------------------------------------------------------
    def run_batch(self, batch_id: str, members: list[PreferencePair]) -> None:
        if self.full_compute:
            plan = _FullPlan(
                batch_id=batch_id, process_pairs=members,
                newly_charged=[], deferred=[], budget_blocked=[],
                charged_count=0, quota_used=len(members),
                global_consumed=self.budget.consumed,
                global_remaining=self.budget.remaining,
            )
        else:
            scheduler = Scheduler(self.budget, self.cfg.per_batch_quota)
            plan = scheduler.plan_batch(batch_id, members, self.report.failures)
        self.process_pairs(plan.process_pairs)

        # ---- 阶段 4：仅对本批已处理成员汇总（先清旧批次汇总，杜绝孤儿） ----
        self.cache.clear_batch_summaries(batch_id)
        processed_here = [p for p in members if p.pair_id in self._processed]
        boundaries = {
            p.pair_id: self.cache.boundary_for(p.pair_id) for p in processed_here
        }
        summaries = summarize_batch(batch_id, processed_here, boundaries, self.cfg)
        for g in summaries:
            self.cache.put_summary(g)
        self.cache.commit_batch_summaries(batch_id, [g.cache_key for g in summaries])

        self.report.batch_plans.append({
            "batch_id": batch_id,
            "processed": [p.pair_id for p in plan.process_pairs],
            "newly_charged": [p.pair_id for p in plan.newly_charged],
            "deferred": [p.pair_id for p in plan.deferred],
            "budget_blocked": [p.pair_id for p in plan.budget_blocked],
            "charged_count": plan.charged_count,
            "quota_used": plan.quota_used,
            "global_consumed": plan.global_consumed,
            "global_remaining": plan.global_remaining,
        })
        self._maybe_snapshot(batch_id)

    def run_all_batches(self) -> None:
        batches = self._group_batches()
        for batch_id in sorted(batches):
            members = sorted(batches[batch_id], key=lambda p: p.pair_id)
            self.run_batch(batch_id, members)
        self._build_audit_lists()

    def _group_batches(self) -> dict[str, list[PreferencePair]]:
        out: dict[str, list[PreferencePair]] = {}
        for pair in self.pairs.values():
            out.setdefault(pair.batch_id, []).append(pair)
        return out

    # ------------------------------------------------------------------
    def apply_and_run(self, command: Command) -> None:
        pairs, cfg, changed = apply_command(
            command, self.pairs, self.cfg, self.cfg.normalize, self.report.failures,
        )
        self.pairs, self.cfg = pairs, cfg
        self.invalidate(changed, reason=f"cmd-{command.seq}:{command.op}")
        # 预算容量若被命令调整，跟随配置
        self.budget.global_budget = cfg.global_pair_budget
        self.global_version += 1
        self.run_all_batches()
        self._build_audit_lists()

    # ------------------------------------------------------------------
    def _build_audit_lists(self) -> None:
        # 逐样本得分分解（按 pair_id 字典序）
        self.report.score_breakdown = []
        for pid in sorted(self.cache.score.pair_index):
            sc: ScoreResult = self.cache.score.items[self.cache.score.pair_index[pid]]
            self.report.score_breakdown.append(sc.to_dict())
        # 边界样本清单：label 秩 -> at_threshold -> tie_rank -> pair_id
        rows = []
        for pid in sorted(self.cache.boundary.pair_index):
            bd: BoundaryResult = self.cache.boundary.items[
                self.cache.boundary.pair_index[pid]]
            rows.append(bd.to_dict())
        from .stages import LABEL_RANK
        rows.sort(key=lambda r: (LABEL_RANK[r["label"]], not r["at_threshold"],
                                 r["tie_rank"], r["pair_id"]))
        self.report.boundary_list = rows
        # 汇总（批次 -> 标注者 -> 分区，已在 summarize 内排序）
        self.report.summaries = [
            self.cache.summary_items[k].to_dict()
            for k in sorted(self.cache.summary_items)
        ]
        self.report.budget_ledger = {
            "global_budget": self.budget.global_budget,
            "consumed": self.budget.consumed,
            "remaining": self.budget.remaining,
            "charged_pairs": sorted(self.budget.charged_pairs),
            "todo": list(self.budget.todo),
            "batch_cursor": dict(sorted(self.budget.batch_cursor.items())),
        }

    # ------------------------------------------------------------------
    def _maybe_snapshot(self, stage_tag: str) -> None:
        if self.snapshot_store is None:
            return
        stage_versions = {s: self.global_version for s in
                          ("load", "feature", "score", "boundary", "summary")}
        manifest = Manifest(
            global_version=self.global_version,
            config_hash=self._config_hash(),
            data_fingerprint=self._data_fingerprint(),
            processed_cursor=len(self._processed),
            consumed_budget=self.budget.consumed,
            stage_versions=stage_versions,
            parallel=self.parallel,
        )
        payload = self.report.canonical_results(self.cache)
        self.snapshot_store.commit_stage(
            self.global_version, "summary", payload, manifest,
        )
