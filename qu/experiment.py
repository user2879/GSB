"""Multi-scale experiment orchestration, verification and reporting."""

from __future__ import annotations

import json
import os
import signal
import tempfile
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .config import PipelineConfig, fingerprint
from .entity import EntityMatcher
from .intent import IntentScorer
from .loader import LoadedBundle
from .pipeline import QueryPipeline, QueryResult
from .rewrite import RewriteEngine
from .runner import TierRunner
from .snapshot import (SNAPSHOT_VERSION, SnapshotStore, cache_key,
                       scope_fingerprint)
from .strategy import StrategySelector


@dataclass
class TierSpec:
    tier_id: str
    samples_path: str
    vocab_path: str
    entity_path: str
    rules_path: str


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return round(float(np.percentile(np.asarray(values, dtype=float), q)), 6)


_WORK_ONLY_KEYS = {
    "entity": {"mode", "match_checks", "candidate_count"},
    "intent": {"mode", "hit_checks"},
    "rewrite": {"pruning_diagnosis", "full_rule_checks", "pruned_rule_checks"},
}


def _semantic_view(record: dict) -> dict:
    """Strip work-statistics fields so comparison covers decisions only."""
    view = {key: value for key, value in record.items()
            if key in ("normalized", "entity", "intent", "rewrite",
                       "failures", "ok")}
    for section, drop_keys in _WORK_ONLY_KEYS.items():
        if isinstance(view.get(section), dict):
            view[section] = {
                key: value for key, value in view[section].items()
                if key not in drop_keys}
    return view


def _equivalent(left: dict, right: dict) -> bool:
    return _semantic_view(left) == _semantic_view(right)


class ExperimentRunner:
    def __init__(self, config: PipelineConfig, workdir: str):
        self.config = config
        self.workdir = workdir
        self.store = SnapshotStore(workdir, config.fingerprint())
        self._stop = threading.Event()
        self._previous_signal = None

    def request_stop(self) -> None:
        self._stop.set()

    def install_signal_handlers(self) -> None:
        """SIGINT/SIGTERM finish the current tier snapshot, then exit 0."""
        if threading.current_thread() is not threading.main_thread():
            return
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                self._previous_signal = signal.signal(sig, self._handle_signal)
            except (ValueError, OSError):
                pass

    def _handle_signal(self, signum, frame) -> None:  # pragma: no cover
        self._stop.set()

    def run_tier(self, spec: TierSpec, bundle: LoadedBundle) -> dict:
        selector = StrategySelector(self.config)
        avg_len = sum(len(s.query) for s in bundle.samples) / max(len(bundle.samples), 1)
        decision = selector.choose(
            query_count=len(bundle.samples),
            vocab_size=len(bundle.vocab),
            dict_size=len(bundle.entities),
            rule_count=len(bundle.rules),
            avg_query_len=avg_len,
        )
        matcher = EntityMatcher(bundle.entities, self.config.dict_priority)
        scorer = IntentScorer(bundle.vocab, bundle.rules, self.config)
        rewriter = RewriteEngine(bundle.rules, self.config.max_rewrites)
        pipeline = QueryPipeline(self.config, matcher, scorer, rewriter, decision)
        runner = TierRunner(pipeline, self.config.max_workers)

        scope = scope_fingerprint(bundle.table_fingerprints["vocab"],
                                  decision.fingerprint, spec.tier_id)
        key_by_query = {
            sample.query_id: cache_key(sample.query_id, scope)
            for sample in bundle.samples
        }
        current_keys = [key_by_query[s.query_id] for s in bundle.samples]
        content_by_key = {
            key_by_query[sample.query_id]: fingerprint(
                {"id": sample.query_id, "query": sample.query,
                 "context": sample.context, "labels": sample.labels})
            for sample in bundle.samples
        }

        previous = self.store.load(spec.tier_id)
        reusable = self.store.reuse_results(
            previous, current_keys, spec.tier_id, scope, content_by_key)

        stale_samples = [s for s in bundle.samples
                         if key_by_query[s.query_id] not in reusable]
        fresh_outcome = runner.run(stale_samples) if stale_samples else {
            "results": [], "parallel_results": [],
            "comparison": {"equal": True, "serial_count": 0,
                           "parallel_count": 0, "ordering_equal": True,
                           "field_mismatch_count": 0,
                           "list_order_mismatches": []},
            "failures": [],
        }

        all_results: List[QueryResult] = []
        for sample in bundle.samples:
            key = key_by_query[sample.query_id]
            hit = key in reusable
            self.store.count_lookup(hit)
            if hit:
                record = reusable[key]
                all_results.append(_record_to_result(record))
            else:
                all_results.append(next(
                    r for r in fresh_outcome["results"]
                    if r.query_id == sample.query_id))
        all_results.sort(key=lambda r: r.sort_key)

        verification = self._verify_equivalence(
            bundle, decision, all_results, spec.tier_id,
            bundle.table_fingerprints["vocab"])
        timings = [r.total_latency_ms for r in all_results if r.ok]
        failures = list(bundle.failures) + [
            failure for r in all_results for failure in r.failures]
        failures.sort(key=lambda item: (
            str(item.get("query_id", "")), item.get("stage", ""),
            item.get("code", "")))

        snapshot_payload = {
            "records": self.store.build_records(all_results, key_by_query),
            "scope_fingerprint": scope,
            "tier_summary": {
                "tier_id": spec.tier_id,
                "metrics": decision.metrics,
                "strategy": decision.to_dict(),
            },
        }
        self.store.save(spec.tier_id, snapshot_payload)

        return {
            "tier_id": spec.tier_id,
            "scale_metrics": {
                "query_count": len(bundle.samples),
                "vocab_entries": len(bundle.vocab),
                "entity_entries": len(bundle.entities),
                "rule_entries": len(bundle.rules),
                "avg_query_len": round(avg_len, 6),
            },
            "strategy": decision.to_dict(),
            "timing_stats": {
                "count": len(timings),
                "avg_latency_ms": round(float(np.mean(timings)), 6) if timings else 0.0,
                "p50_latency_ms": _percentile(timings, 50),
                "p90_latency_ms": _percentile(timings, 90),
                "p99_latency_ms": _percentile(timings, 99),
                "max_latency_ms": round(max(timings), 6) if timings else 0.0,
            },
            "work_stats": self._work_stats(all_results),
            "equivalence": verification,
            "concurrency_comparison": fresh_outcome["comparison"],
            "cache": self.store.counters.to_dict(),
            "failures": failures,
            "snapshot_version": SNAPSHOT_VERSION,
            "stopped_after_snapshot": self._stop.is_set(),
            "results": all_results,
        }

    def _verify_equivalence(self, bundle: LoadedBundle, decision,
                            results: Sequence[QueryResult], tier_id: str,
                            vocab_fp: str) -> dict:
        """Re-run a deterministic subset with the unpruned reference paths."""
        verify_n = min(self.config.exhaustive_verify_sample, len(results))
        by_id = {r.query_id: r for r in results}
        subset_ids = sorted(by_id)[:verify_n]
        ref_matcher = EntityMatcher(bundle.entities, self.config.dict_priority)
        ref_scorer = IntentScorer(bundle.vocab, bundle.rules, self.config)
        ref_rewriter = RewriteEngine(bundle.rules, self.config.max_rewrites)
        ref_pipeline = QueryPipeline(
            self.config, ref_matcher, ref_scorer, ref_rewriter,
            _ReferenceDecision())
        mismatches: List[dict] = []
        check_delta = {"entity_pruned": 0, "entity_exhaustive": 0,
                       "intent_pruned": 0, "intent_exhaustive": 0,
                       "rules_pruned": 0, "rules_full": 0}
        for sample in bundle.samples:
            if sample.query_id not in subset_ids:
                continue
            reference = ref_pipeline.process(
                sample.query_id, sample.query, sample.context, sample.labels)
            current = by_id[sample.query_id]
            if not _equivalent(reference.to_dict(), current.to_dict()):
                mismatches.append({"query_id": sample.query_id,
                                   "reason": "exhaustive_mismatch"})
            check_delta["entity_pruned"] += current.entity["match_checks"]
            check_delta["entity_exhaustive"] += reference.entity["match_checks"]
            check_delta["intent_pruned"] += current.intent["hit_checks"]
            check_delta["intent_exhaustive"] += reference.intent["hit_checks"]
            check_delta["rules_pruned"] += current.rewrite["pruned_rule_checks"]
            check_delta["rules_full"] += current.rewrite["full_rule_checks"]
        return {
            "verified_query_ids": subset_ids,
            "verified_count": len(subset_ids),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches,
            "equal_to_exhaustive": not mismatches,
            "check_counts": check_delta,
            "pruning_reduces_work": (
                check_delta["entity_pruned"] <= check_delta["entity_exhaustive"]
                and check_delta["intent_pruned"] <= check_delta["intent_exhaustive"]
            ),
        }

    @staticmethod
    def _work_stats(results: Sequence[QueryResult]) -> dict:
        entity_checks = [r.entity["match_checks"] for r in results if r.entity]
        intent_checks = [r.intent["hit_checks"] for r in results if r.intent]
        rewrite_lost = [
            r.query_id for r in results
            if r.rewrite and r.rewrite["pruning_diagnosis"]
            ["pruning_lost_suggestion"]
        ]
        return {
            "total_entity_match_checks": sum(entity_checks),
            "total_intent_hit_checks": sum(intent_checks),
            "queries_losing_rewrite_to_pruning": rewrite_lost,
        }

    def run_experiment(self, specs: Sequence[TierSpec],
                       bundle_loader) -> dict:
        tier_reports: List[dict] = []
        stopped = False
        for spec in specs:
            bundle = bundle_loader(spec)
            tier_report = self.run_tier(spec, bundle)
            tier_reports.append(tier_report)
            if self._stop.is_set():
                stopped = True
                break
        report = build_report(self.config, tier_reports,
                              self.store.counters.to_dict(),
                              interrupted=stopped)
        report_path = os.path.join(self.workdir, "experiment_report.json")
        _atomic_write_json(report_path, report)
        return report


class _ReferenceDecision:
    entity_mode = "exhaustive"
    intent_mode = "full_scan"


def _record_to_result(record: dict) -> QueryResult:
    result = QueryResult(
        query_id=record["query_id"],
        original_query=record["original_query"],
        sort_key=record["sort_key"],
        query_fingerprint=record["query_fingerprint"],
        ok=record["ok"],
        normalized=record.get("normalized"),
        entity=record.get("entity"),
        intent=record.get("intent"),
        rewrite=record.get("rewrite"),
        stage_timings_ms=record.get("stage_timings_ms", {}),
        total_latency_ms=record.get("total_latency_ms", 0.0),
        failures=record.get("failures", []),
    )
    return result


def _atomic_write_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix=".report.", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2,
                      sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def build_report(config: PipelineConfig, tier_reports: Sequence[dict],
                 cache_counters: dict, interrupted: bool) -> dict:
    code_counts: Dict[str, int] = {}
    for tier in tier_reports:
        for failure in tier["failures"]:
            code_counts[failure["code"]] = code_counts.get(failure["code"], 0) + 1
    return {
        "report_version": 1,
        "snapshot_version": SNAPSHOT_VERSION,
        "generated_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                          time.gmtime()),
        "config_fingerprint": config.fingerprint(),
        "interrupted_graceful_stop": interrupted,
        "tiers": [
            {key: value for key, value in tier.items() if key != "results"}
            for tier in tier_reports
        ],
        "failure_code_counts": dict(sorted(code_counts.items())),
        "cache_totals": cache_counters,
        "all_tiers_match_exhaustive": all(
            tier["equivalence"]["equal_to_exhaustive"] for tier in tier_reports),
        "all_concurrency_comparisons_equal": all(
            tier["concurrency_comparison"]["equal"] for tier in tier_reports),
    }
