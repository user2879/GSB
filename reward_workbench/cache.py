"""基于内容指纹的对级精确失效缓存。

每个阶段产物以确定性 cache_key 存储，并记录其引用的偏好对集合与特征桶集合。
失效时只摘除引用了受影响对的产物，未受影响对的产物原样保留并复用。
"""
from __future__ import annotations

from dataclasses import dataclass

from .stages import (
    BoundaryResult, FeatureResult, GroupSummary, ScoreResult,
)


@dataclass
class StageCache:
    name: str
    items: dict[str, object]
    # pair_id -> 当前指向的 cache_key（对级阶段）
    pair_index: dict[str, str]
    # cache_key -> 引用的偏好对集合 / 桶集合
    ref_pairs: dict[str, set[str]]
    ref_buckets: dict[str, set[int]]

    @classmethod
    def create(cls, name: str) -> "StageCache":
        return cls(name, {}, {}, {}, {})

    def get(self, key: str) -> object | None:
        return self.items.get(key)

    def put_pair_artifact(self, key: str, artifact: object, pair_id: str,
                          buckets: set[int] | None = None) -> None:
        self.items[key] = artifact
        self.pair_index[pair_id] = key
        self.ref_pairs.setdefault(key, set()).add(pair_id)
        if buckets:
            self.ref_buckets.setdefault(key, set()).update(buckets)

    def invalidate_pairs(self, pair_ids: set[str]) -> set[str]:
        """摘除引用了给定对的全部产物，返回被摘除的 cache_key 集合。"""
        removed: set[str] = set()
        for key, refs in list(self.ref_pairs.items()):
            if refs & pair_ids:
                self.items.pop(key, None)
                self.ref_pairs.pop(key, None)
                self.ref_buckets.pop(key, None)
                removed.add(key)
        for pid in pair_ids:
            self.pair_index.pop(pid, None)
        return removed

    def keys_for_pairs(self, pair_ids: set[str]) -> set[str]:
        return {
            key for pid in pair_ids
            if (key := self.pair_index.get(pid)) is not None
        }


class ArtifactCache:
    def __init__(self) -> None:
        self.feature = StageCache.create("feature")
        self.score = StageCache.create("score")
        self.boundary = StageCache.create("boundary")
        # 汇总以 group cache_key 为键，记录成员对集合
        self.summary_items: dict[str, GroupSummary] = {}
        self.summary_refs: dict[str, set[str]] = {}
        # 批次 -> 该批次当前 group key 列表
        self.batch_summary_keys: dict[str, list[str]] = {}

    # ---- 对级三阶段 ----
    def put_feature(self, result: FeatureResult) -> None:
        self.feature.put_pair_artifact(
            result.cache_key, result, result.pair_id,
            set(result.referenced_buckets),
        )

    def put_score(self, result: ScoreResult) -> None:
        self.score.put_pair_artifact(
            result.cache_key, result, result.pair_id,
        )
        self.score.ref_pairs[result.cache_key].add(result.pair_id)

    def put_boundary(self, result: BoundaryResult) -> None:
        self.boundary.put_pair_artifact(
            result.cache_key, result, result.pair_id,
        )

    # ---- 汇总阶段 ----
    def put_summary(self, summary: GroupSummary) -> None:
        self.summary_items[summary.cache_key] = summary
        self.summary_refs[summary.cache_key] = set(summary.pair_ids)

    def commit_batch_summaries(self, batch_id: str, keys: list[str]) -> None:
        self.batch_summary_keys[batch_id] = keys

    def clear_batch_summaries(self, batch_id: str) -> None:
        """重算批次前，整体摘除该批次此前的全部汇总（权威索引清理）。"""
        for key in self.batch_summary_keys.pop(batch_id, []):
            self.summary_items.pop(key, None)
            self.summary_refs.pop(key, None)

    def invalidate_summaries(self, pair_ids: set[str]) -> set[str]:
        removed: set[str] = set()
        for key, refs in list(self.summary_refs.items()):
            if refs & pair_ids:
                self.summary_items.pop(key, None)
                self.summary_refs.pop(key, None)
                removed.add(key)
        for batch_id, keys in list(self.batch_summary_keys.items()):
            if set(keys) & removed:
                self.batch_summary_keys[batch_id] = [
                    k for k in keys if k not in removed
                ]
        return removed

    def cascade_invalidate(self, changed_pairs: set[str]) -> dict[str, list[str]]:
        """自底向上失效：feature -> score -> boundary -> summary。

        由于对级阶段每对只有一条当前产物，直接按受影响对摘除即可精确失效。
        """
        report = {
            "feature": sorted(self.feature.invalidate_pairs(changed_pairs)),
            "score": sorted(self.score.invalidate_pairs(changed_pairs)),
            "boundary": sorted(self.boundary.invalidate_pairs(changed_pairs)),
            "summary": sorted(self.invalidate_summaries(changed_pairs)),
        }
        return report

    # ---- 复用判定 ----
    def feature_for(self, pair_id: str) -> FeatureResult | None:
        key = self.feature.pair_index.get(pair_id)
        return self.feature.items.get(key) if key else None

    def score_for(self, pair_id: str) -> ScoreResult | None:
        key = self.score.pair_index.get(pair_id)
        return self.score.items.get(key) if key else None

    def boundary_for(self, pair_id: str) -> BoundaryResult | None:
        key = self.boundary.pair_index.get(pair_id)
        return self.boundary.items.get(key) if key else None

    def canonical_manifest(self) -> dict:
        """当前有效产物的规范化清单（增量与全量重建必须逐字段一致）。"""
        def pair_stage(cache: StageCache) -> list[dict]:
            rows = []
            for pid in sorted(cache.pair_index):
                key = cache.pair_index[pid]
                rows.append({
                    "pair_id": pid,
                    "cache_key": key,
                    "ref_pairs": sorted(cache.ref_pairs.get(key, set())),
                    "ref_buckets": sorted(cache.ref_buckets.get(key, set())),
                })
            return rows

        summaries = []
        for key in sorted(self.summary_items):
            g = self.summary_items[key]
            summaries.append({
                "batch_id": g.batch_id,
                "annotator_id": g.annotator_id,
                "time_partition": g.time_partition,
                "cache_key": key,
                "ref_pairs": list(g.pair_ids),
            })
        return {
            "feature": pair_stage(self.feature),
            "score": pair_stage(self.score),
            "boundary": pair_stage(self.boundary),
            "summary": sorted(
                summaries,
                key=lambda r: (r["batch_id"], r["annotator_id"], r["time_partition"]),
            ),
        }
