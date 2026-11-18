"""Pipeline orchestration: stages, caching, snapshots, isolation and reports.

Semantics:
* the content-addressed cache is the incremental-reuse mechanism;
* staged snapshots are the crash-recovery mechanism (cursor based);
* snapshot lineage covers config, rule table, scale tier *and* input data
  fingerprints, so any edit invalidates a stale snapshot.
"""

from __future__ import annotations

import math
import os

from . import SNAPSHOT_VERSION
from .cache import FingerprintCache
from .config import Config, Injection, load_injection
from .errors import E_CONFIG_MISMATCH, E_NON_FINITE, E_STAGE_FAILED, PipelineError
from .features import (
    FeatureExtractor,
    high_cardinality_ratio,
    inject_non_finite,
    reward_distribution,
    score_pair,
)
from .hallucination import attach_priority_rank, detect, fixed_tie_order
from .loader import sorted_failures
from .metrics import StreamingAggregator, per_sample
from .runner import run_tordered
from .selector import select_strategy
from .snapshot import STAGES, SnapshotStore
from .util import atomic_write_json, fingerprint, read_json

STAGE_FEATURES = "features"
STAGE_SCORING = "scoring"
STAGE_HALLUCINATION = "hallucination"
STAGE_METRICS = "metrics"


class Pipeline:
    def __init__(
        self,
        config: Config,
        rules: tuple,
        preferences: list,
        evaluations: list,
        workdir: str,
        injection_path: str | None = None,
        strict_resume: bool = False,
        crash_after: int | None = None,
    ):
        self.config = config
        self.rules = rules
        self.preferences = preferences
        self.evaluations = evaluations
        self.workdir = workdir
        self.injections = load_injection(injection_path)
        self.strict_resume = strict_resume
        self.crash_after = crash_after
        self.failures = []
        self.injection_hits = []
        self._steps = 0
        self.resumed_stages = []

        # Concurrency is an execution-runtime parameter and never changes the
        # computed artifacts, so it is excluded from the lineage fingerprint.
        self.config_fp = fingerprint(
            {key: value for key, value in config.raw.items() if key != "concurrency"}
        )
        self.rules_fp = fingerprint([rule.__dict__ for rule in rules])
        self.injection_fp = fingerprint(
            read_json(injection_path) if injection_path and os.path.exists(injection_path) else {}
        )
        self.data_fp = fingerprint(
            {"preferences": preferences, "evaluations": evaluations, "injection": self.injection_fp}
        )
        self.cache = FingerprintCache(os.path.join(workdir, "cache"))
        self.store = SnapshotStore(
            os.path.join(workdir, "snapshots"), self.config_fp, self.rules_fp, ""
        )

    def _maybe_crash(self) -> None:
        if self.crash_after is None:
            return
        self._steps += 1
        if self._steps > self.crash_after:
            raise RuntimeError("injected crash for snapshot recovery test")

    def _injection(self, kind: str, target_id: str | None = None):
        for injection in self.injections:
            if injection.kind != kind:
                continue
            if target_id is None or injection.target_id in ("", target_id):
                return injection
        return None

    def _record_injection(self, injection: Injection) -> None:
        self.injection_hits.append(
            {"item_no": injection.item_no, "kind": injection.kind, "target_id": injection.target_id}
        )

    def _record_exception(self, exc: Exception, sample_id: str, stage: str) -> None:
        if isinstance(exc, PipelineError):
            record = exc.to_record()
            record.setdefault("sample_id", sample_id)
            record["stage"] = stage
        else:
            record = {
                "code": E_STAGE_FAILED,
                "message": str(exc),
                "exception_type": type(exc).__name__,
                "sample_id": sample_id,
                "stage": stage,
            }
        self.failures.append(record)

    def _sample_preferences(self) -> list:
        """Deterministic sampling by content hash, never a runtime RNG."""
        ratio = self.config.preference_sample_ratio
        if ratio >= 1.0:
            return list(self.preferences)
        selected = []
        for pair in self.preferences:
            marker = int(fingerprint(pair["pair_id"])[:8], 16) / 0xFFFFFFFF
            if marker < ratio:
                selected.append(pair)
        return selected

    def run(self) -> dict:
        sampled = self._sample_preferences()
        total = len(sampled)
        tier, _tier_index = self.config.tier_for_size(total)

        baseline = FeatureExtractor(
            self.config.precision_level, self.config.hash_buckets, self.config.tokenizer_seed
        )
        probe_vectors = [baseline.vector(pair["prompt"]) for pair in sampled]
        unique_buckets = int(sum(int((vector > 0).sum()) for vector in probe_vectors))
        high_card = high_cardinality_ratio(probe_vectors)
        decision = select_strategy(self.config, total, unique_buckets, high_card)

        self.store.tier_id = tier.name
        lineage_ok = self._validate_snapshot_lineage()
        if not lineage_ok:
            self.store.reset()

        extractor = FeatureExtractor(
            decision["precision_level"],
            decision["effective_hash_buckets"],
            self.config.tokenizer_seed,
        )
        feature_artifacts = self._stage_features(sampled, extractor, tier.name)
        scoring_artifacts = self._stage_scoring(sampled, extractor, tier.name)
        hallucination_artifacts = self._stage_hallucination(decision, tier.name)
        metrics_artifacts = self._stage_metrics(tier.name)

        report = self._build_report(
            tier, decision, scoring_artifacts, hallucination_artifacts, metrics_artifacts
        )
        atomic_write_json(os.path.join(self.workdir, "report.json"), report)
        return report

    def _validate_snapshot_lineage(self) -> bool:
        any_snapshot = False
        for stage in STAGES:
            path = self.store._path(stage)
            if not os.path.exists(path):
                continue
            any_snapshot = True
            snapshot = read_json(path)
            mismatch = (
                snapshot.get("snapshot_version") != SNAPSHOT_VERSION
                or snapshot.get("config_fp") != self.config_fp
                or snapshot.get("rules_fp") != self.rules_fp
                or snapshot.get("tier_id") != self.store.tier_id
                or snapshot.get("data_fp") != self.data_fp
            )
            if mismatch:
                if self.strict_resume:
                    raise PipelineError(
                        E_CONFIG_MISMATCH,
                        f"snapshot {stage} lineage mismatch; resume refused",
                        stage=stage,
                    )
                return False
        return any_snapshot

    def _resume_cursor(self, stage: str):
        snapshot = self.store.load(stage, strict=False)
        if snapshot is None or "artifacts" not in snapshot:
            return 0, None
        if int(snapshot["cursor"]) > 0:
            self.resumed_stages.append(stage)
        return int(snapshot["cursor"]), snapshot["artifacts"]

    def _stage_features(self, sampled: list, extractor: FeatureExtractor, tier_id: str) -> dict:
        cursor, artifacts = self._resume_cursor(STAGE_FEATURES)
        if artifacts is None:
            artifacts = {"features": [], "params": extractor.params}
        tasks = []
        for index in range(cursor, len(sampled)):
            pair = sampled[index]

            def make(pair=pair):
                return {
                    "pair_id": pair["pair_id"],
                    "prompt_buckets": _nonzero(extractor.vector(pair["prompt"])),
                    "chosen_buckets": _nonzero(extractor.vector(pair["chosen"])),
                    "rejected_buckets": _nonzero(extractor.vector(pair["rejected"])),
                }

            tasks.append((index, make))
        for key, result, exc in run_tordered(tasks, self.config.concurrency):
            self._maybe_crash()
            if exc is not None:
                self._record_exception(exc, sampled[key]["pair_id"], STAGE_FEATURES)
                continue
            artifacts["features"].append(result)
            if len(artifacts["features"]) % self.config.snapshot_interval == 0:
                self.store.write(
                    STAGE_FEATURES, len(artifacts["features"]), artifacts, self.data_fp
                )
        self.store.write(STAGE_FEATURES, len(sampled), artifacts, self.data_fp)
        return artifacts

    def _stage_scoring(self, sampled: list, extractor: FeatureExtractor, tier_id: str) -> dict:
        cursor, artifacts = self._resume_cursor(STAGE_SCORING)
        if artifacts is None:
            artifacts = {"scored": []}
        params_fp = fingerprint(
            {"extractor": extractor.params, "boundary": self.config.reward_boundary_threshold}
        )
        tasks = []
        for index in range(cursor, len(sampled)):
            pair = sampled[index]

            def make(pair=pair):
                content_fp = fingerprint(
                    {
                        "pair_id": pair["pair_id"],
                        "prompt": pair["prompt"],
                        "chosen": pair["chosen"],
                        "rejected": pair["rejected"],
                    }
                )

                def compute():
                    return score_pair(pair, extractor, self.config.reward_boundary_threshold)

                artifact, _reused = self.cache.get_or_compute(
                    STAGE_SCORING, content_fp, params_fp, tier_id, compute
                )
                return artifact

            tasks.append((index, make))
        for key, result, exc in run_tordered(tasks, self.config.concurrency):
            pair = sampled[key]
            self._maybe_crash()
            if exc is not None:
                self._record_exception(exc, pair["pair_id"], STAGE_SCORING)
                continue
            injection = self._injection("non_finite_feature", pair["pair_id"])
            if injection is not None:
                self._record_injection(injection)
                result = inject_non_finite(result, injection.item_no)
                self.failures.append(
                    PipelineError(
                        E_NON_FINITE,
                        f"injection {injection.item_no} produced non-finite reward",
                        sample_id=pair["pair_id"],
                        stage=STAGE_SCORING,
                    ).to_record()
                )
            artifacts["scored"].append(result)
            if len(artifacts["scored"]) % self.config.snapshot_interval == 0:
                self.store.write(
                    STAGE_SCORING, len(artifacts["scored"]), artifacts, self.data_fp
                )
        self.store.write(STAGE_SCORING, len(sampled), artifacts, self.data_fp)
        return artifacts

    def _stage_hallucination(self, decision: dict, tier_id: str) -> dict:
        cursor, artifacts = self._resume_cursor(STAGE_HALLUCINATION)
        adjusted_threshold = decision["adjusted_hallucination_threshold"]
        params_fp = fingerprint(
            {
                "rules": self.rules_fp,
                "threshold": adjusted_threshold,
                "priority": attach_priority_rank(self.rules),
            }
        )
        tasks = []
        for index in range(cursor, len(self.evaluations)):
            sample = self.evaluations[index]

            def make(sample=sample):
                content_fp = fingerprint(
                    {
                        "sample_id": sample["sample_id"],
                        "context": sample["context"],
                        "model_answer": sample["model_answer"],
                        "reference_answer": sample["reference_answer"],
                    }
                )

                def compute():
                    return detect(sample, self.rules, adjusted_threshold)

                artifact, _reused = self.cache.get_or_compute(
                    STAGE_HALLUCINATION, content_fp, params_fp, tier_id, compute
                )
                return artifact

            tasks.append((index, make))
        collected = [] if artifacts is None else list(artifacts["raw_results"])
        for key, result, exc in run_tordered(tasks, self.config.concurrency):
            self._maybe_crash()
            if exc is not None:
                self._record_exception(exc, self.evaluations[key]["sample_id"], STAGE_HALLUCINATION)
                continue
            collected.append(result)
            if len(collected) % self.config.snapshot_interval == 0:
                self.store.write(
                    STAGE_HALLUCINATION,
                    len(collected),
                    {"raw_results": collected, "threshold_applied": adjusted_threshold},
                    self.data_fp,
                )
        ordered = fixed_tie_order(collected)
        artifacts = {"results": ordered, "threshold_applied": adjusted_threshold, "raw_results": collected}
        self.store.write(
            STAGE_HALLUCINATION, len(self.evaluations), artifacts, self.data_fp
        )
        return {"results": ordered, "threshold_applied": adjusted_threshold}

    def _stage_metrics(self, tier_id: str) -> dict:
        cursor, artifacts = self._resume_cursor(STAGE_METRICS)
        if artifacts is not None and cursor >= len(self.evaluations):
            return artifacts
        params_fp = fingerprint({"metrics": list(self.config.metrics), "tier": tier_id})
        content_fp = fingerprint(self.evaluations)

        def compute():
            aggregator = StreamingAggregator()
            rows = []
            for sample in self.evaluations:
                row = per_sample(sample)
                aggregator.add(row)
                rows.append(row)
            summary = aggregator.result()
            return {
                "summary": {name: summary[name] for name in self.config.metrics if name in summary},
                "per_sample": rows,
            }

        artifacts, _reused = self.cache.get_or_compute(
            STAGE_METRICS, content_fp, params_fp, tier_id, compute
        )
        self.store.write(STAGE_METRICS, len(self.evaluations), artifacts, self.data_fp)
        return artifacts

    def _build_report(self, tier, decision, scoring, hallucination, metrics) -> dict:
        detection = hallucination["results"]
        suspected = sum(1 for item in detection if item["suspected_hallucination"])
        reward_pairs = []
        for item in scoring["scored"]:
            diff = item.get("reward_diff")
            finite = isinstance(diff, (int, float)) and math.isfinite(diff)
            reward_pairs.append(
                {
                    "pair_id": item["pair_id"],
                    "reward_diff": diff if finite else None,
                    "reward_diff_ref": item.get("reward_diff_ref"),
                    "boundary": item.get("boundary", False),
                }
            )
        return {
            "snapshot_version": SNAPSHOT_VERSION,
            "config_fp": self.config_fp,
            "rules_fp": self.rules_fp,
            "data_fp": self.data_fp,
            "tier": {"name": tier.name, "max_samples": tier.max_samples},
            "strategy": decision,
            "reward_distribution": reward_distribution(scoring["scored"]),
            "reward_pairs": reward_pairs,
            "hallucination": {
                "detection_rate": (suspected / len(detection)) if detection else 0.0,
                "suspected_count": suspected,
                "sample_count": len(detection),
                "results": detection,
            },
            "metrics": metrics["summary"],
            "cache": self.cache.stats(),
            "resume": {
                "snapshot_resume": bool(self.resumed_stages),
                "resumed_stages": sorted(set(self.resumed_stages)),
            },
            "snapshot_versions": self.store.versions(),
            "injections": self.injection_hits,
            "failures": sorted_failures(self.failures),
        }


def _nonzero(vector) -> list:
    return [int(index) for index in vector.nonzero()[0].tolist()]


VOLATILE_REPORT_KEYS = ("cache", "resume")


def normalized_report(report: dict) -> dict:
    """Strip run-state audit counters so core outputs can be compared exactly."""
    return {key: value for key, value in report.items() if key not in VOLATILE_REPORT_KEYS}
