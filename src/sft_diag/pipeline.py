"""构造管道：采样 -> 打分 -> 去重 三阶段，支持精确失效与增量复用。"""
from __future__ import annotations

from typing import Any, Callable

from .dedup import Deduplicator
from .errors import Failure, OK
from .generator import CandidateGenerator, build_reference_index
from .models import WorkspaceState
from .scorer import QualityScorer
from .snapshot import Manifest, SnapshotStore
from .tokenizer import Tokenizer
from .utils import (
    derive_cache_key,
    fingerprint,
    float64,
    round_sig,
)
import math


class Pipeline:
    def __init__(
        self,
        state: WorkspaceState,
        store: SnapshotStore,
        base_fingerprint: str,
        failures: list[Failure],
        fault_hook: Callable[[str, int], None] | None = None,
    ) -> None:
        self.state = state
        self.store = store
        self.base_fingerprint = base_fingerprint
        self.failures = failures
        self.fault_hook = fault_hook or (lambda stage, batch: None)
        self.manifest: Manifest | None = None
        self.last_report: dict[str, Any] | None = None
        self._checkpoint_commits = 0

    # ------------------------------------------------------------------
    # 阶段一：采样（确定性全量重算序列，按缓存键区分新增/变更/复用）
    # ------------------------------------------------------------------
    def run_sampling(
        self, batch_no: int, previous: dict[str, Any] | None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        generator = CandidateGenerator(self.state)
        candidates = generator.generate_all()
        reference_index = build_reference_index(candidates)
        fps = self.state.fingerprints()
        task_keys: dict[str, str] = {}
        by_task: dict[str, list[dict[str, Any]]] = {}
        for candidate in candidates:
            by_task.setdefault(candidate["task_type"], []).append(candidate)
        for task_type in sorted(self.state.config.task_types):
            task_candidates = sorted(
                by_task.get(task_type, []), key=lambda item: item["candidate_id"]
            )
            task_keys[task_type] = derive_cache_key(
                "sampling",
                fps["seeds_fp"],
                fps["slots_fp"],
                generator.sampling_config_fp,
                fingerprint(
                    [
                        {
                            "candidate_id": c["candidate_id"],
                            "template_id": c["template_id"],
                            "slot_sources": c["slot_sources"],
                            "sample_order": c["sample_order"],
                        }
                        for c in task_candidates
                    ]
                ),
            )
        payload = {
            "stage": "sampling",
            "batch_no": batch_no,
            "task_cache_keys": task_keys,
            "candidates": candidates,
            "reference_index": reference_index,
        }
        stats = self._diff_sampling(previous, payload)
        return payload, stats

    @staticmethod
    def _candidate_sampling_key(candidate: dict[str, Any]) -> str:
        return derive_cache_key(
            "candidate",
            fingerprint(
                {
                    "candidate_id": candidate["candidate_id"],
                    "template_id": candidate["template_id"],
                    "template_occurrence": candidate["template_occurrence"],
                    "slot_sources": candidate["slot_sources"],
                    "slot_values": candidate["slot_values"],
                    "instruction": candidate["instruction"],
                    "reference_answer": candidate["reference_answer"],
                }
            ),
        )

    def _diff_sampling(
        self, previous: dict[str, Any] | None, current: dict[str, Any]
    ) -> dict[str, Any]:
        old_keys: dict[str, str] = {}
        old_index: dict[str, dict[str, list[str]]] = {"template": {}, "slot_source": {}}
        if previous:
            for candidate in previous["candidates"]:
                old_keys[candidate["candidate_id"]] = self._candidate_sampling_key(candidate)
            old_index = previous.get("reference_index", old_index)
        new_ids: list[str] = []
        changed_ids: list[str] = []
        reused_ids: list[str] = []
        for candidate in current["candidates"]:
            identifier = candidate["candidate_id"]
            key = self._candidate_sampling_key(candidate)
            if identifier not in old_keys:
                new_ids.append(identifier)
            elif old_keys[identifier] != key:
                changed_ids.append(identifier)
            else:
                reused_ids.append(identifier)
        removed_ids = sorted(set(old_keys) - {c["candidate_id"] for c in current["candidates"]})
        reasons = self._invalidation_reasons(old_index, current, new_ids, changed_ids, removed_ids)
        return {
            "new_candidate_ids": sorted(new_ids),
            "changed_candidate_ids": sorted(changed_ids),
            "reused_candidate_ids": sorted(reused_ids),
            "removed_candidate_ids": removed_ids,
            "reasons": reasons,
        }

    def _invalidation_reasons(
        self,
        old_index: dict[str, dict[str, list[str]]],
        current: dict[str, Any],
        new_ids: list[str],
        changed_ids: list[str],
        removed_ids: list[str],
    ) -> dict[str, list[str]]:
        current_refs = build_reference_index(current["candidates"])
        old_templates = set(old_index.get("template", {}))
        new_templates = set(current_refs["template"])
        old_sources = set(old_index.get("slot_source", {}))
        new_sources = set(current_refs["slot_source"])
        replaced_or_deleted_templates = old_templates - new_templates
        added_or_deleted_sources = old_sources.symmetric_difference(new_sources)
        reasons: dict[str, list[str]] = {
            "template_replaced_or_deleted": sorted(
                cid
                for template in replaced_or_deleted_templates
                for cid in old_index["template"].get(template, [])
            ),
            "slot_source_added_or_deleted": sorted(
                cid
                for source in added_or_deleted_sources
                for cid in old_index.get("slot_source", {}).get(source, [])
            ),
            "target_or_weight_changed": sorted(changed_ids),
            "newly_generated": sorted(new_ids),
            "removed": removed_ids,
        }
        return reasons

    # ------------------------------------------------------------------
    # 阶段二：打分（仅缓存键未命中的候选才重新打分）
    # ------------------------------------------------------------------
    def run_scoring(
        self,
        sampling_payload: dict[str, Any],
        previous_scores: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        tokenizer = Tokenizer(self.state.vocab)
        scorer = QualityScorer(self.state, tokenizer)
        cached: dict[str, dict[str, Any]] = {}
        if previous_scores:
            for record in previous_scores["scores"]:
                cached[record["cache_key"]] = record
        candidates = sampling_payload["candidates"]
        scores, status = scorer.score_all(
            candidates, self.state.config.max_workers, cached
        )
        payload = {
            "stage": "scoring",
            "batch_no": sampling_payload["batch_no"],
            "scores": scores,
        }
        return payload, status

    # ------------------------------------------------------------------
    # 阶段三：去重（仅任务级缓存键未命中的任务重算）
    # ------------------------------------------------------------------
    def run_dedup(
        self,
        sampling_payload: dict[str, Any],
        scoring_payload: dict[str, Any],
        previous_dedup: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], dict[str, str]]:
        tokenizer = Tokenizer(self.state.vocab)
        deduplicator = Deduplicator(self.state, tokenizer)
        score_by_id = {record["candidate_id"]: record for record in scoring_payload["scores"]}
        scored_candidates = []
        for candidate in sampling_payload["candidates"]:
            merged = dict(candidate)
            merged.update(score_by_id[candidate["candidate_id"]])
            scored_candidates.append(merged)
        cached = {}
        if previous_dedup:
            for result in previous_dedup["results"]:
                cached[result["cache_key"]] = result
        results, status = deduplicator.deduplicate_all(
            scored_candidates, self.state.config.max_workers, cached
        )
        payload = {
            "stage": "dedup",
            "batch_no": sampling_payload["batch_no"],
            "results": results,
        }
        return payload, status

    # ------------------------------------------------------------------
    # 完整批次执行
    # ------------------------------------------------------------------
    def execute_batch(
        self,
        batch_no: int,
        previous: dict[str, dict[str, Any]],
        checkpoint: bool,
    ) -> dict[str, Any]:
        version = batch_no
        prior_stage_versions = (
            dict(self.manifest.stage_versions) if self.manifest is not None else {}
        )
        self.manifest = Manifest(
            global_version=version,
            base_fingerprint=self.base_fingerprint,
            combined_fingerprint=self.state.combined_fingerprint(),
            config_fp=self.state.fingerprints()["config_fp"],
            params_fp=self.state.fingerprints()["params_fp"],
            completed_batch_no=batch_no,
            stage_versions=prior_stage_versions,
            stage_cache_keys={},
            snapshot_interval_batches=self.state.config.snapshot_interval_batches,
        )
        self.store.write_manifest(self.manifest)
        self.fault_hook("manifest", batch_no)

        sampling_payload, sampling_stats = self.run_sampling(
            batch_no, previous.get("sampling")
        )
        sampling_key = fingerprint(
            {task: key for task, key in sampling_payload["task_cache_keys"].items()}
        )
        self.fault_hook("sampling_pre", batch_no)
        self.manifest = self.store.commit_stage(
            "sampling", sampling_payload, sampling_key, self.manifest
        )
        self.fault_hook("sampling_post", batch_no)

        scoring_payload, score_status = self.run_scoring(
            sampling_payload, previous.get("scoring")
        )
        scoring_key = fingerprint(
            [record["cache_key"] for record in scoring_payload["scores"]]
        )
        self.fault_hook("scoring_pre", batch_no)
        self.manifest = self.store.commit_stage(
            "scoring", scoring_payload, scoring_key, self.manifest
        )
        self.fault_hook("scoring_post", batch_no)

        dedup_payload, dedup_status = self.run_dedup(
            sampling_payload, scoring_payload, previous.get("dedup")
        )
        dedup_key = fingerprint(
            [result["cache_key"] for result in dedup_payload["results"]]
        )
        self.fault_hook("dedup_pre", batch_no)
        self.manifest = self.store.commit_stage(
            "dedup", dedup_payload, dedup_key, self.manifest
        )
        self.fault_hook("dedup_post", batch_no)

        if checkpoint:
            self._checkpoint_commits += 1
        report = self.build_report(
            batch_no,
            sampling_payload,
            scoring_payload,
            dedup_payload,
            sampling_stats,
            score_status,
            dedup_status,
            checkpoint,
        )
        self.last_report = report
        return report

    # ------------------------------------------------------------------
    # 报告
    # ------------------------------------------------------------------
    def build_report(
        self,
        batch_no: int,
        sampling_payload: dict[str, Any],
        scoring_payload: dict[str, Any],
        dedup_payload: dict[str, Any],
        sampling_stats: dict[str, Any],
        score_status: dict[str, str],
        dedup_status: dict[str, str],
        checkpoint: bool,
    ) -> dict[str, Any]:
        scores_by_id = {record["candidate_id"]: record for record in scoring_payload["scores"]}
        candidates = sampling_payload["candidates"]
        retained_by_task: dict[str, list[str]] = {
            result["task_type"]: result["retained_ids"]
            for result in dedup_payload["results"]
        }
        all_retained = {cid for ids in retained_by_task.values() for cid in ids}
        task_reports: dict[str, Any] = {}
        for task in self.state.config.task_types:
            task_candidates = [c for c in candidates if c["task_type"] == task]
            task_ids = [c["candidate_id"] for c in task_candidates]
            retained = retained_by_task.get(task, [])
            task_reports[task] = {
                "candidate_count": len(task_ids),
                "retained_count": len(retained),
                "removed_count": len(task_ids) - len(retained),
                "dimension_averages_all": self._average_dimensions(task_ids, scores_by_id),
                "dimension_averages_retained": self._average_dimensions(retained, scores_by_id),
            }
        rescored = sorted(
            cid for cid, status in score_status.items() if status == "recomputed"
        )
        score_reused = sorted(
            cid for cid, status in score_status.items() if status == "reused"
        )
        dedup_recomputed = sorted(
            cid for cid, status in dedup_status.items() if status == "recomputed"
        )
        dedup_reused = sorted(
            cid for cid, status in dedup_status.items() if status == "reused"
        )
        report = {
            "business_code": OK,
            "completed_batch_no": batch_no,
            "global_version": self.manifest.global_version if self.manifest else 0,
            "fingerprints": self.state.fingerprints(),
            "tasks": task_reports,
            "totals": {
                "candidate_count": len(candidates),
                "retained_count": len(all_retained),
                "removed_count": len(candidates) - len(all_retained),
            },
            "invalidation": {
                "new_candidates": sampling_stats["new_candidate_ids"],
                "sampling_changed_candidates": sampling_stats["changed_candidate_ids"],
                "sampling_reused_candidates": sampling_stats["reused_candidate_ids"],
                "removed_candidates": sampling_stats["removed_candidate_ids"],
                "reasons": sampling_stats["reasons"],
                "rescored_candidates": rescored,
                "score_reused_candidates": score_reused,
                "dedup_recomputed_candidates": dedup_recomputed,
                "dedup_reused_candidates": dedup_reused,
            },
            "snapshot": {
                "version": self.manifest.global_version if self.manifest else 0,
                "stage_versions": dict(self.manifest.stage_versions) if self.manifest else {},
                "checkpoint": checkpoint,
                "checkpoint_commits_so_far": self._checkpoint_commits,
                "recovered_from_batch": getattr(self, "_recovered_from", None),
            },
            "failures": [failure.to_dict() for failure in self.failures],
            "diagnostics": {"field_differences": []},
        }
        return report

    @staticmethod
    def _average_dimensions(
        candidate_ids: list[str], scores_by_id: dict[str, dict[str, Any]]
    ) -> dict[str, float]:
        if not candidate_ids:
            return {"clarity": 0.0, "correctness": 0.0, "format": 0.0, "length": 0.0,
                    "total_score": 0.0}
        dims = ("clarity", "correctness", "format", "length")
        averages: dict[str, float] = {}
        for dim in dims:
            averages[dim] = round_sig(
                float64(
                    math.fsum(
                        float64(scores_by_id[cid]["dimensions"][dim])
                        for cid in candidate_ids
                    )
                    / len(candidate_ids)
                )
            )
        averages["total_score"] = round_sig(
            float64(
                math.fsum(
                    float64(scores_by_id[cid]["total_score"]) for cid in candidate_ids
                )
                / len(candidate_ids)
            )
        )
        return averages
