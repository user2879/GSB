"""Top-level orchestration engine for slow-query diagnostics."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from .aggregation import Evidence, aggregate_batch
from .arbitration import QueryVerdict, merge_evidence
from .codec import load_frame_file
from .config import load_declaration
from .diagnosis import DiagnosisReport, build_report
from .invalidation import (
    SchemeDiff,
    ViewReplacement,
    diff_scheme,
    mark_stale_conclusions,
    replace_view,
)
from .ledger import DiagnosticLedger
from .models import (
    CandidateRule,
    DeclarationBundle,
    SourceBatch,
    SourceDeclaration,
)
from .quantization import ErrorBound, Quantizer, build_error_bound
from .scheduler import LoopbackServiceProcess, ReplayJob, ReplayScheduler


@dataclass
class EngineResult:
    scheme_version: int
    bundle: DeclarationBundle
    batches: list[SourceBatch]
    evidence_by_source: dict[str, dict[int, Evidence]]
    verdicts: dict[int, QueryVerdict]
    error_bounds: dict[tuple[str, int], ErrorBound]
    report: DiagnosisReport
    view_replacement: ViewReplacement | None
    diff: SchemeDiff | None
    stale_entries: list
    replay_results: list


class DiagnosticEngine:
    def __init__(self, ledger_path: str, scale_exponent: int = 1) -> None:
        self.ledger = DiagnosticLedger(ledger_path).open()
        self.quantizer = Quantizer(scale_exponent)
        self.scheme_version = 0
        self._previous_sources: tuple[SourceDeclaration, ...] = ()
        self._previous_candidates: tuple[CandidateRule, ...] = ()
        self._previous_view_keys: set[tuple[int, str]] = set()

    def close(self) -> None:
        self.ledger.close()

    # ------------------------------------------------------------------
    def run(self, declaration_path: str,
            replay: bool = False,
            replay_config: dict | None = None) -> EngineResult:
        bundle = load_declaration(declaration_path)
        diff = diff_scheme(
            self._previous_sources, bundle.sources,
            self._previous_candidates, bundle.candidates,
        )
        self.scheme_version += 1
        self.ledger.begin_scheme(self.scheme_version)

        batches: list[SourceBatch] = []
        evidence_by_source: dict[str, dict[int, Evidence]] = {}
        source_to_queries: dict[str, set[int]] = {}
        for source in bundle.sources:
            records = load_frame_file(source)
            batch = SourceBatch(source=source, records=tuple(records))
            evidence = aggregate_batch(batch)
            batches.append(batch)
            evidence_by_source[source.source_id] = evidence
            source_to_queries[source.source_id] = set(evidence)
            self.ledger.write_source_batch(batch, evidence)

        verdicts = merge_evidence(evidence_by_source)
        all_evidence = [
            evidence
            for per_source in evidence_by_source.values()
            for evidence in per_source.values()
        ]
        bounds: list[ErrorBound] = []
        for query_id in sorted(verdicts):
            participating = [
                evidence for evidence in all_evidence
                if evidence.query_id == query_id
            ]
            for metric in ("latency_us", "rows_scanned"):
                bounds.append(
                    build_error_bound(
                        self.quantizer, metric, query_id,
                        participating, batches,
                    )
                )
        error_bounds = {(b.metric, b.query_id): b for b in bounds}
        self.ledger.write_conclusions(verdicts, bounds)

        report = build_report(
            list(bundle.candidates), verdicts, self.quantizer, error_bounds,
            {source.source_id: source.source_type for source in bundle.sources},
        )

        candidate_to_sources = {
            candidate.candidate_id: candidate.target_sources
            for candidate in bundle.candidates
        }
        stale_entries = (
            mark_stale_conclusions(
                diff, source_to_queries, candidate_to_sources,
                self.scheme_version - 1,
            )
            if self._previous_sources else []
        )
        current_keys = sorted(
            (query_id, metric)
            for query_id in verdicts
            for metric in ("latency_us", "rows_scanned")
        )
        view_replacement = replace_view(
            self._previous_view_keys, current_keys, self.scheme_version
        )
        self._previous_view_keys = set(current_keys)
        self._previous_sources = bundle.sources
        self._previous_candidates = bundle.candidates

        replay_results = []
        if replay:
            replay_results = self._dispatch_replays(
                bundle, replay_config or {}, report
            )

        return EngineResult(
            scheme_version=self.scheme_version,
            bundle=bundle,
            batches=batches,
            evidence_by_source=evidence_by_source,
            verdicts=verdicts,
            error_bounds=error_bounds,
            report=report,
            view_replacement=view_replacement,
            diff=diff if self.scheme_version > 1 else None,
            stale_entries=stale_entries,
            replay_results=replay_results,
        )

    # ------------------------------------------------------------------
    def build_replay_catalog(self, bundle: DeclarationBundle,
                             catalog_path: str,
                             estimate_profiles: dict | None = None) -> str:
        profiles = estimate_profiles or {}
        entries = []
        for candidate in bundle.candidates:
            for source_id in candidate.target_sources:
                profile = profiles.get(
                    f"{candidate.candidate_id}@{source_id}",
                    profiles.get(candidate.candidate_id, {}),
                )
                entries.append({
                    "candidate_id": candidate.candidate_id,
                    "source_id": source_id,
                    "est_latency_us": int(profile.get(
                        "est_latency_us",
                        max(1000, 10000 - candidate.expected_gain_bps),
                    )),
                    "est_rows_scanned": int(profile.get(
                        "est_rows_scanned", 500,
                    )),
                    "stall_ms": int(profile.get("stall_ms", 0)),
                })
        with open(catalog_path, "w", encoding="utf-8") as handle:
            json.dump(
                {"catalog_format": 1, "replay_catalog": entries},
                handle, sort_keys=True, indent=2,
            )
        return os.path.abspath(catalog_path)

    def _dispatch_replays(self, bundle: DeclarationBundle,
                          replay_config: dict,
                          report: DiagnosisReport) -> list:
        catalog_path = replay_config["catalog_path"]
        process = LoopbackServiceProcess(catalog_path)
        port = process.start()
        try:
            scheduler = ReplayScheduler(
                port=port,
                global_concurrency=int(replay_config.get("global_concurrency", 2)),
                per_source_quota=dict(replay_config.get(
                    "per_source_quota",
                    {source.source_id: 1 for source in bundle.sources},
                )),
                socket_timeout_ms=int(replay_config.get("socket_timeout_ms", 2000)),
            )
            jobs = []
            for candidate in bundle.candidates:
                for source_id in candidate.target_sources:
                    jobs.append(ReplayJob(
                        candidate_id=candidate.candidate_id,
                        source_id=source_id,
                        rewritten_sql=candidate.rewritten_sql,
                    ))
            for job in jobs:
                self.ledger.write_replay_intent(job)
            results = scheduler.run_all(jobs)
            for result in results:
                self.ledger.write_replay_receipt(result)
                if result.ok and result.ack is not None:
                    self._attach_replay_to_report(result, report)
            self._last_scheduler_stats = scheduler.stats
            return results
        finally:
            process.stop()

    def _attach_replay_to_report(self, result, report: DiagnosisReport) -> None:
        for evaluation in report.evaluations:
            if evaluation.candidate_id != result.job.candidate_id:
                continue
            ack = result.ack
            evaluation.replay_latency_ticks = self.quantizer.quantize(
                int(ack["est_latency_us"])
            )
            evaluation.replay_rows_ticks = self.quantizer.quantize(
                int(ack["est_rows_scanned"])
            )

    def recover_pending_replays(self, replay_config: dict) -> list:
        """Rerun candidates whose receipt never committed (hard-kill case)."""
        scheme_version = self.ledger.latest_scheme_with_unknown_jobs()
        if scheme_version is None:
            return []
        self.ledger.begin_scheme(scheme_version)
        unknown = self.ledger.unknown_replay_jobs(scheme_version)
        if not unknown:
            return []
        process = LoopbackServiceProcess(replay_config["catalog_path"])
        port = process.start()
        try:
            scheduler = ReplayScheduler(
                port=port,
                global_concurrency=int(replay_config.get("global_concurrency", 2)),
                per_source_quota=dict(replay_config["per_source_quota"]),
                socket_timeout_ms=int(replay_config.get("socket_timeout_ms", 2000)),
            )
            jobs = [
                ReplayJob(
                    candidate_id=row["candidate_id"],
                    source_id=row["source_id"],
                    rewritten_sql=row["rewritten_sql"],
                )
                for row in unknown
            ]
            results = scheduler.run_all(jobs)
            for result in results:
                self.ledger.write_replay_receipt(result)
            return results
        finally:
            process.stop()

    def verify_committed_replays(self, fresh_receipts: list[dict]) -> None:
        scheme_version = self.ledger.latest_scheme_with_unknown_jobs()
        if scheme_version is None:
            scheme_version = self.scheme_version
        self.ledger.verify_replay_against_ledger(
            scheme_version, fresh_receipts
        )
