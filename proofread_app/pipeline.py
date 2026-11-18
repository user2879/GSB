"""主流水线：五阶段编排、内容指纹缓存、并发执行与快照提交。"""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .cache import (
    ContentCache,
    STAGE_SEGMENT,
    STAGE_DETECT,
    STAGE_SCORE,
    STAGE_AGGREGATE,
    STAGE_DECISION,
)
from .concurrency import Task, run_tasks
from .config import AppConfig
from .detection import detect_candidates, referenced_rules, segment_sample
from .editing import build_edits
from .fingerprint import combine_fingerprints, content_fingerprint, derive_key, text_fingerprint
from .models import (
    Candidate,
    PrecisionAudit,
    RuleResource,
    Sample,
    SampleResult,
    ScoreBreakdown,
    SuppressedItem,
)
from .precision import audit_sample
from .scoring import score_all
from .signals import SignalController
from .snapshot import Manifest, SnapshotStore


def _config_param_fp(config: AppConfig) -> str:
    return content_fingerprint(config.to_fingerprint_obj())


def _scoring_fp(params_fp: str, precision: str) -> str:
    return content_fingerprint({"params": params_fp, "precision": precision})


class Pipeline:
    """串联检测、双档位打分、审计、编辑与缓存/快照的编排引擎。"""

    def __init__(self, config: AppConfig, resource: RuleResource,
                 scoring_fingerprint: str, cache: ContentCache,
                 snapshots: SnapshotStore,
                 signal_controller: Optional[SignalController] = None) -> None:
        self.config = config
        self.resource = resource
        self.scoring_fingerprint = scoring_fingerprint
        self.cache = cache
        self.snapshots = snapshots
        self.signals = signal_controller or SignalController()
        self.config_fp = _config_param_fp(config)

    # ---- 缓存键派生 ----
    def _segment_key(self, sample: Sample) -> str:
        return derive_key(
            [text_fingerprint(sample.text)],
            {"ngram_order": self.config.ngram_order})

    def _detect_key(self, segment_fp: str, sample: Sample) -> str:
        return derive_key(
            [segment_fp, self.resource.source_fingerprint],
            {"threshold": self.config.similarity_threshold,
             "ngram_order": self.config.ngram_order})

    def _score_key(self, detect_fp: str, precision: str) -> str:
        return derive_key(
            [detect_fp, _scoring_fp(self.scoring_fingerprint, precision)],
            {"quant_levels": self.config.quant_levels,
             "fixed_point_bits": self.config.fixed_point_bits})

    def _aggregate_key(self, score_fp: str, precision: str) -> str:
        return derive_key([score_fp], {"weights": self.config.aggregation_weights,
                                       "precision": precision})

    def _decision_key(self, aggregate_fp: str, precision: str) -> str:
        return derive_key([aggregate_fp],
                          {"threshold": self.config.decision_threshold,
                           "precision": precision})

    # ---- 单样本处理 ----
    def process_sample(self, sample: Sample) -> SampleResult:
        cache = self.cache
        result = SampleResult(sample=sample)

        # 阶段 1：上下文切分
        segment_key = self._segment_key(sample)
        cached = cache.get(sample.sample_id, STAGE_SEGMENT, segment_key)
        if cached is not None:
            segment_payload = cached.payload
        else:
            segmented = segment_sample(sample, self.config)
            segment_payload = segmented.to_fingerprint_obj()
            cache.put(sample.sample_id, STAGE_SEGMENT, segment_key, segment_payload)
        segment_fp = content_fingerprint(segment_payload)

        # 阶段 2：候选检测
        detect_key = self._detect_key(segment_fp, sample)
        cached = cache.get(sample.sample_id, STAGE_DETECT, detect_key)
        if cached is not None:
            detect_payload = cached.payload
            candidates = [Candidate(**item) for item in detect_payload["candidates"]]
            suppressed = [SuppressedItem(**item) for item in detect_payload["suppressed"]]
        else:
            segmented = segment_sample(sample, self.config)
            candidates, suppressed = detect_candidates(
                sample, segmented, self.resource, self.config)
            detect_payload = {
                "candidates": [c.to_fingerprint_obj() for c in candidates],
                "suppressed": [s.to_fingerprint_obj() for s in suppressed],
            }
            cache.put(sample.sample_id, STAGE_DETECT, detect_key, detect_payload,
                      upstream_keys=[segment_key])
        result.candidates = candidates
        result.suppressed = suppressed
        for rule_id in referenced_rules(candidates, suppressed):
            cache.record_rule_usage(rule_id, sample.sample_id)
        result.used_rules = referenced_rules(candidates, suppressed)
        detect_fp = content_fingerprint(detect_payload)

        # 阶段 3/4/5：双档位分项打分、聚合、决策（决策已内嵌于 ScoreBreakdown）
        precision_pair = ("float32", "float64")
        score_payloads: Dict[str, Any] = {}
        for precision in precision_pair:
            score_key = self._score_key(detect_fp, precision)
            cached = cache.get(sample.sample_id, STAGE_SCORE,
                               f"{precision}:{score_key}")
            if cached is not None:
                score_payload = cached.payload
            else:
                breakdowns = score_all(candidates, self.resource,
                                       self._params_view(), self.config, precision)
                score_payload = {"items": [b.to_fingerprint_obj() for b in breakdowns]}
                cache.put(sample.sample_id, STAGE_SCORE,
                          f"{precision}:{score_key}", score_payload,
                          upstream_keys=[detect_key])

            aggregate_key = self._aggregate_key(content_fingerprint(score_payload),
                                                precision)
            agg_cached = cache.get(sample.sample_id, STAGE_AGGREGATE,
                                   f"{precision}:{aggregate_key}")
            if agg_cached is not None:
                aggregate_payload = agg_cached.payload
            else:
                aggregate_payload = {
                    "items": [
                        {"rule_id": item["rule_id"], "start": item["start"],
                         "weighted_detect": item["weighted_detect"],
                         "weighted_confidence": item["weighted_confidence"],
                         "total_confidence": item["total_confidence"]}
                        for item in score_payload["items"]
                    ]
                }
                cache.put(sample.sample_id, STAGE_AGGREGATE,
                          f"{precision}:{aggregate_key}", aggregate_payload,
                          upstream_keys=[score_key])

            decision_key = self._decision_key(
                content_fingerprint(aggregate_payload), precision)
            dec_cached = cache.get(sample.sample_id, STAGE_DECISION,
                                   f"{precision}:{decision_key}")
            if dec_cached is not None:
                decision_payload = dec_cached.payload
            else:
                decision_payload = {
                    "items": [
                        {"rule_id": item["rule_id"], "start": item["start"],
                         "decision": item["decision"], "boundary": item["boundary"]}
                        for item in score_payload["items"]
                    ]
                }
                cache.put(sample.sample_id, STAGE_DECISION,
                          f"{precision}:{decision_key}", decision_payload,
                          upstream_keys=[aggregate_key])
            score_payloads[precision] = score_payload

        result.scores_fp32 = [ScoreBreakdown(**item)
                              for item in score_payloads["float32"]["items"]]
        result.scores_fp64 = [ScoreBreakdown(**item)
                              for item in score_payloads["float64"]["items"]]

        # 逐级误差审计
        result.audits = audit_sample(sample.sample_id, candidates,
                                     result.scores_fp32, result.scores_fp64,
                                     self.config)

        # 修正文本以配置档位决策为准
        primary = self.config.precision
        primary_scores = result.scores_fp32 if primary == "float32" else result.scores_fp64
        result.edits, result.corrected_text = build_edits(
            sample, candidates, primary_scores, self.resource, primary)
        result.boundary = any(
            item.boundary for item in result.scores_fp32
        ) or any(item.boundary for item in result.scores_fp64)
        return result

    def _params_view(self):
        return self._scoring_params

    def bind_scoring_params(self, params) -> None:
        self._scoring_params = params

    # ---- 批量执行 ----
    def run_batch(self, samples: Sequence[Sample],
                  existing_manifest: Optional[Manifest] = None
                  ) -> Tuple[List[SampleResult], Manifest, bool]:
        ordered_samples = sorted(samples, key=lambda item: item.sample_id)
        completed = set(existing_manifest.completed_cursor) if existing_manifest else set()
        pending = [s for s in ordered_samples if s.sample_id not in completed]

        # 并发执行候选检测与分项打分：整条样本流水线进入线程池，
        # 缓存命中在 worker 内部生效，未失效产物直接复用、不会重算。
        tasks: List[Task[Sample]] = [
            Task(sample.sample_id, 0, STAGE_DETECT, sample) for sample in pending
        ]

        def _worker(task: Task[Sample]) -> SampleResult:
            return self.process_sample(task.payload)

        processed = run_tasks(tasks, _worker, self.config.max_workers)

        # 回收后按样本 ID 升序串行提交快照，顺序与单线程完全一致
        results: List[SampleResult] = []
        exited_gracefully = False
        cursor = sorted(completed)
        stage_versions = dict(existing_manifest.stage_versions) if existing_manifest else {}
        global_version = existing_manifest.global_version if existing_manifest else 0
        result_map: Dict[str, SampleResult] = {}

        for index, result in enumerate(processed, start=1):
            result_map[result.sample.sample_id] = result_map.get(
                result.sample.sample_id, result)
            results.append(result)
            cursor.append(result.sample.sample_id)
            if index % self.config.snapshot_interval == 0 or index == len(processed):
                global_version += 1
                stage_versions = self._commit_snapshot(
                    ordered_samples, result_map, sorted(cursor),
                    global_version, stage_versions)
            if self.signals.requested:
                exited_gracefully = True
                global_version += 1
                stage_versions = self._commit_snapshot(
                    ordered_samples, result_map, sorted(cursor),
                    global_version, stage_versions)
                break

        manifest = Manifest(
            global_version=global_version,
            config_hash=self.config_fp,
            rule_fingerprint=self.resource.source_fingerprint,
            scoring_fingerprint=self.scoring_fingerprint,
            completed_cursor=sorted(cursor),
            stage_versions=stage_versions,
        )
        self.snapshots.write_manifest(manifest)
        self.cache.persist()
        return results, manifest, exited_gracefully

    def _commit_snapshot(self, all_samples: Sequence[Sample],
                         result_map: Dict[str, "SampleResult"], cursor: List[str],
                         global_version: int,
                         stage_versions: Dict[str, int]) -> Dict[str, int]:
        versions = dict(stage_versions)
        for stage, builder in (
            (STAGE_SEGMENT, lambda r: r.sample.to_fingerprint_obj()),
            (STAGE_DETECT, lambda r: {
                "candidates": [c.to_fingerprint_obj() for c in r.candidates],
                "suppressed": [s.to_fingerprint_obj() for s in r.suppressed]}),
            (STAGE_SCORE, lambda r: {
                "fp32": [b.to_fingerprint_obj() for b in r.scores_fp32],
                "fp64": [b.to_fingerprint_obj() for b in r.scores_fp64]}),
            (STAGE_AGGREGATE, lambda r: {
                "fp32": [b.to_fingerprint_obj() for b in r.scores_fp32],
                "fp64": [b.to_fingerprint_obj() for b in r.scores_fp64]}),
            (STAGE_DECISION, lambda r: {
                "fp32": [{"rule_id": b.rule_id, "start": b.start,
                          "decision": b.decision, "boundary": b.boundary}
                         for b in r.scores_fp32],
                "fp64": [{"rule_id": b.rule_id, "start": b.start,
                          "decision": b.decision, "boundary": b.boundary}
                         for b in r.scores_fp64]}),
        ):
            payload = {sample_id: builder(result_map[sample_id])
                       for sample_id in cursor if sample_id in result_map}
            self.snapshots.commit_stage(stage, global_version, payload)
            versions[stage] = global_version
        return versions
