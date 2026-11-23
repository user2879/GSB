"""End-to-end orchestration: load -> schedule -> dedup -> correlate -> commit.

Provides fresh runs, crash-injection runs and recovery runs with deterministic
replay verification. All batch writes are single SQLite transactions.
"""
from __future__ import annotations

import logging
import dataclasses
from dataclasses import dataclass, field

from . import fixedpoint
from .correlation import correlate_all, correlate_one
from .dedup import Deduplicator
from .errors import SpecError
from .evolution import assert_version_monotonic, diff_views
from .ledger import Ledger
from .loader import load_graph, load_rules, load_stream
from .logging_setup import configure_logging
from .models import AggregateAlert
from .replay import verify_replay
from .rules import RuleSpec
from .scheduler import MonotonicClock, Scheduler, Task
from .topology import DependencyGraph
from .util import canonical_digest, read_ndjson, write_json

log = logging.getLogger("alert_pipeline.pipeline")


@dataclass
class PipelineConfig:
    graph_path: str
    rules_path: str
    stream_path: str
    ledger_path: str
    log_dir: str = "logs"
    batch_size: int = 20
    global_limit: int = 4
    per_service_limit: int = 2
    task_timeout_ms: int = 2000
    max_attempts: int = 3
    scale_k: int = 4
    timeout_task_ids: set[str] = field(default_factory=set)


class Pipeline:
    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.clock = MonotonicClock()
        self.graph: DependencyGraph = load_graph(cfg.graph_path)
        self.rules: RuleSpec = load_rules(cfg.rules_path, self.graph)
        self.alerts = load_stream(cfg.stream_path, self.graph, self.rules)
        self.digests = {
            "stream": canonical_digest(
                [text for _, text in read_ndjson(cfg.stream_path)]),
            "graph": self.graph.digest,
            "rules": canonical_digest(dataclasses.asdict(self.rules)),
        }
        self.ledger = Ledger(cfg.ledger_path)
        self.version = self.rules.declared_version
        self.logger = configure_logging(cfg.log_dir)

    # -------------------------------------------------------------- entrypoints
    def run(self, crash_at_batch: int | None = None) -> dict:
        """Fresh run. ``crash_at_batch`` simulates a forced kill: the batch is
        marked UNKNOWN and never committed, then the process 'dies'."""
        self.ledger.verify_snapshots()
        recovery = self.ledger.recovery_state()
        if recovery is not None:
            if recovery.declared_version != self.version:
                raise SpecError(
                    "ledger declared_version differs; use evolve()", 0,
                    "declared_version")
            return self.resume()
        result = self._process(start_line=0, next_batch=1,
                               crash_at_batch=crash_at_batch)
        return result

    def resume(self) -> dict:
        recovery = self.ledger.recovery_state()
        assert recovery is not None
        self.ledger.verify_snapshots()
        for digest_name in ("stream_digest", "graph_digest", "rules_digest"):
            key = digest_name.split("_")[0]
            if getattr(recovery, digest_name) != self.digests[key]:
                raise SpecError(
                    f"input {key} changed without a declared version bump", 0,
                    digest_name)
        unknown = sorted(recovery.unknown_batches)
        if unknown:
            log.warning("UNKNOWN batches require re-run: %s", unknown,
                        extra={"extra_data": {"unknown_batches": unknown}})
        # Deterministic replay verification of every committed conclusion.
        # The progress cursor may describe an UNKNOWN (never committed) batch,
        # so the trusted cursor is the last COMMITTED batch's cursor.
        trusted_cursor = self.ledger.last_committed_cursor(self.version)
        replay_aggs, replay_corrs, replay_state = self._reprocess_through(
            trusted_cursor)
        stored_state = self.ledger.latest_dedup_state(self.version)
        if replay_state != stored_state:
            from .errors import ReplayError
            raise ReplayError(
                "replay divergence for open dedup state", 0, "dedup_state",
                "cursor_line=" + str(trusted_cursor))
        _, ledger_aggs, ledger_corrs = self.ledger.load_committed(self.version)
        ledger_aggs = sorted(ledger_aggs,
                             key=lambda r: r["aggregate_id"])
        ledger_corrs = sorted(ledger_corrs,
                              key=lambda r: r["aggregate_id"])
        verify_replay(
            sorted((a.to_record() for a in replay_aggs),
                   key=lambda r: r["aggregate_id"]),
            ledger_aggs,
            sorted((c.to_record() for c in replay_corrs),
                   key=lambda r: r["aggregate_id"]),
            ledger_corrs)
        result = self._process(start_line=trusted_cursor,
                               next_batch=recovery.last_committed_batch + 1,
                               crash_at_batch=None)
        result["replayed_batches"] = recovery.last_committed_batch
        result["reran_unknown_batches"] = unknown
        return result

    def evolve(self, new_graph_path: str, new_rules_path: str,
               new_stream_path: str) -> dict:
        """Whole-view replacement after a monotonic version bump."""
        old_version = self.version
        old_aggs, _ = self.ledger.current_view(old_version)
        new_graph = load_graph(new_graph_path)
        new_rules = load_rules(new_rules_path, new_graph)
        assert_version_monotonic(old_version, new_rules.declared_version)
        if new_rules.declared_version == old_version:
            raise SpecError(
                "spec changed but declared_version was not bumped", 0,
                "declared_version")
        stale_count = self.ledger.mark_all_stale()
        # Swap live inputs and reset processing against the new declared version.
        self.graph = new_graph
        self.rules = new_rules
        self.alerts = load_stream(new_stream_path, self.graph, self.rules)
        self.digests = {
            "stream": canonical_digest(
                [text for _, text in read_ndjson(new_stream_path)]),
            "graph": self.graph.digest,
            "rules": canonical_digest(dataclasses.asdict(self.rules)),
        }
        self.version = new_rules.declared_version
        result = self._process(start_line=0, next_batch=1,
                               crash_at_batch=None)
        new_aggs, _ = self.ledger.current_view(self.version)
        result["view_diff"] = diff_views(
            old_aggs, old_version, new_aggs, self.version).to_record()
        result["stale_batches_marked"] = stale_count
        return result

    # ------------------------------------------------------------------ core
    def _process(self, start_line: int, next_batch: int,
                 crash_at_batch: int | None) -> dict:
        pending = [a for a in self.alerts if a.line > start_line]
        dedup = Deduplicator(self.rules)
        if start_line:
            dedup.restore_state(
                self.ledger.latest_dedup_state(self.version))
        scheduler = Scheduler(
            global_limit=self.cfg.global_limit,
            per_service_limit=self.cfg.per_service_limit,
            task_timeout_ms=self.cfg.task_timeout_ms,
            max_attempts=self.cfg.max_attempts,
            clock=self.clock,
            timeout_task_ids=set(self.cfg.timeout_task_ids))
        batch_reports: list[dict] = []
        batch_id = next_batch - 1
        index = 0
        while index < len(pending):
            batch_id += 1
            chunk = pending[index:index + self.cfg.batch_size]
            index += len(chunk)
            if crash_at_batch == batch_id:
                cursor = start_line + index - len(chunk)
                self.ledger.mark_unknown(batch_id, self.version, cursor)
                log.error("simulated forced kill at batch %s", batch_id)
                return {
                    "status": "CRASHED",
                    "crashed_batch": batch_id,
                    "next_cursor_line": cursor,
                    "batch_reports": batch_reports,
                }
            tasks = [
                Task(task_id=f"b{batch_id}-l{alert.line}",
                     service_id=alert.service_id,
                     func=(lambda al=alert: al),
                     payload={"line": alert.line})
                for alert in chunk]
            ordered, metrics = scheduler.run_batch(batch_id, tasks)
            sealed_aggs: list[AggregateAlert] = []
            for alert in ordered:
                sealed_aggs.extend(dedup.feed(alert))
            if index < len(pending):
                open_sealed = []
            else:
                # Only still-open windows are sealed at stream end; windows
                # sealed during feeding were already emitted exactly once.
                open_sealed = [
                    obj.seal() for obj in
                    [dedup._open[k] for k in sorted(dedup._open)]]
                dedup._open.clear()
            batch_aggs = sealed_aggs + open_sealed
            batch_corrs = [correlate_one(a, self.graph, self.rules)
                           for a in batch_aggs]
            cursor = chunk[-1].line
            summary = {
                "batch_id": batch_id,
                "input_lines": [a.line for a in chunk],
                "aggregate_ids": [a.aggregate_id for a in batch_aggs],
                "counts": {"processed": len(chunk),
                           "emitted_aggregates": len(batch_aggs),
                           "suppressed": sum(a.suppressed_count
                                             for a in batch_aggs)},
            }
            self.ledger.commit_batch(
                batch_id=batch_id,
                declared_version=self.version,
                cursor_line=cursor,
                summary=summary,
                aggregate_records=[a.to_record() for a in batch_aggs],
                correlation_records=[c.to_record() for c in batch_corrs],
                digests=self.digests,
                clock_ms=self.clock.now_ms(),
                dedup_state=dedup.open_state())
            report = metrics.to_record()
            report["summary"] = summary
            batch_reports.append(report)
            log.info("committed batch %s cursor=%s", batch_id, cursor,
                     extra={"extra_data": {"batch_id": batch_id,
                                           "cursor": cursor}})

        all_aggs, all_corrs = self._materialize_view()
        stats = self._statistics(all_aggs, all_corrs)
        return {
            "status": "OK",
            "declared_version": self.version,
            "total_alerts": len(self.alerts),
            "total_aggregates": len(all_aggs),
            "total_suppressed": sum(a["suppressed_count"] for a in all_aggs),
            "batch_reports": batch_reports,
            "statistics": stats,
        }

    def _materialize_view(self):
        agg_records, corr_records = self.ledger.current_view(self.version)
        return agg_records, corr_records

    def _statistics(self, agg_records: list[dict], corr_records: list[dict]):
        # Rebuild lightweight objects compatible with fixedpoint helpers.
        aggs = [_AggregateShim(r) for r in agg_records]
        corrs = [_CorrelationShim(r) for r in corr_records]
        suppression = fixedpoint.suppression_ratios(aggs, self.cfg.scale_k)
        hit_rate = fixedpoint.root_cause_hit_rate(
            aggs, corrs, self.cfg.scale_k)
        avg = fixedpoint.average_aggregate_count(
            aggs, len(self.graph.services), self.cfg.scale_k)
        ranking = fixedpoint.rank_fingerprints(suppression)
        return {
            "scale_factor": 10 ** self.cfg.scale_k,
            "truncation": "floor",
            "suppression_ratio_by_fingerprint": [s.to_record() for s in
                                                 ranking],
            "global_suppression_ratio": next(
                s.to_record() for s in suppression if s.key == "__global__"),
            "root_cause_hit_rate": hit_rate.to_record(),
            "avg_aggregates_per_service": avg.to_record(),
        }

    def _reprocess_through(self, cursor_line: int):
        """Deterministic re-pipeline of alerts up to and including cursor.

        Replay starts from scratch but seals the same windows the online run
        had open at ``cursor_line``: both committed aggregates and the persisted
        open-window state must be reproduced bit-for-bit.
        """
        replay_dedup = Deduplicator(self.rules)
        replay_aggs: list[AggregateAlert] = []
        for alert in self.alerts:
            if alert.line > cursor_line:
                break
            replay_aggs.extend(replay_dedup.feed(alert))
        # Only sealed windows are committed conclusions; still-open windows
        # live in the persisted dedup state and are verified separately.
        replay_aggs.sort(key=lambda a: (a.first_ts_ms, a.service_id,
                                        a.fingerprint, a.aggregate_id))
        replay_corrs = correlate_all(replay_aggs, self.graph, self.rules)
        return replay_aggs, replay_corrs, replay_dedup.open_state()

    def write_run_report(self, result: dict, path: str) -> None:
        write_json(path, result)

    def close(self) -> None:
        self.ledger.close()


class _AggregateShim:
    def __init__(self, rec: dict):
        self.__dict__.update(rec)
        self.member_lines = tuple(rec["member_lines"])

    @property
    def suppressed_count(self):
        return self.__dict__["suppressed_count"]

    @property
    def fingerprint(self):
        return self.__dict__["fingerprint"]


class _CandidateShim:
    def __init__(self, rec: dict):
        self.direction = rec["direction"]
        self.layer = rec["layer"]


class _CorrelationShim:
    def __init__(self, rec: dict):
        self.candidates = [_CandidateShim(c) for c in rec["candidates"]]
