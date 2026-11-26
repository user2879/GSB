"""End-to-end runner tying validation, scheduling, ledger and reporting."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

from .attribution import attribute_step
from .cache import ConclusionCache
from .commands import Command, CommandApplier
from .config import DecodeConfig
from .errors import BizCode, BizError
from .exchange import render, parse, documents_equal
from .injection import apply_injections, load_injections
from .io_utils import load_config, load_requests
from .ledger import Ledger, Snapshot
from .models import validate_requests
from .pipeline import DecodePipeline
from .report import RunReport
from .scheduler import Scheduler


class Runner:
    def __init__(self, config: DecodeConfig, scheduler_cfg: dict,
                 exchange_version: str = "v2",
                 ledger_dir: str = "ledger"):
        self.config = config
        self.scheduler_cfg = scheduler_cfg
        self.exchange_version = exchange_version
        self.ledger = Ledger(ledger_dir)
        self.cache = ConclusionCache()
        self.snapshot_version = 0

    def _scheduler(self, requests, max_workers=None):
        order = {r.request_id: i + 1 for i, r in
                 enumerate(sorted(requests, key=lambda r: r.request_id))}
        pipeline = DecodePipeline(self.config, self.cache, order)
        cfg = self.scheduler_cfg
        return Scheduler(
            pipeline=pipeline,
            requests=requests,
            cache=self.cache,
            batch_budget=cfg["batch_budget"],
            per_request_quota=cfg["per_request_quota"],
            step_recompute_cap=cfg["step_recompute_cap"],
            max_workers=max_workers or cfg.get("max_workers", 1))

    def validate(self, requests) -> list[BizError]:
        return validate_requests(requests)

    def run_full(self, requests, injections=None, max_workers=None):
        report = RunReport(started_at=time.perf_counter())
        report.request_count = len(requests)
        injection_hits = []
        if injections:
            requests, injection_hits, mutated_config = apply_injections(
                requests, load_injections(injections), self.config)
            if mutated_config is not None:
                self.config = mutated_config
        report.add_failures(injection_hits)
        if injections:
            report.add_failures(self.config.validate())

        valid, invalid = self._partition(requests)
        report.add_failures(invalid)
        report.failed_requests = sorted(
            {e.location.split("/")[0].split("=")[-1] for e in invalid})

        scheduler = self._scheduler(valid, max_workers)
        batches = scheduler.run_to_completion()
        outcomes = []
        for batch in batches:
            outcomes.extend(batch.outcomes)
            report.add_failures(batch.failures)
        # Only the final batch's retained todos are genuinely unfinished.
        report.todos = [t.to_dict() for t in batches[-1].todos]
        outcomes.sort(key=lambda o: o.task_key)

        boundaries, conclusions = self._collect(outcomes)
        report.boundary_token_count = len(boundaries)
        report.cache_hit_count = len(self.cache.hits())
        report.recompute_count = len(self.cache.recomputes())
        report.invalidated_count = len(self.cache.invalidated)

        pipeline = scheduler.pipeline
        doc = pipeline.build_exchange_document(outcomes)
        v1_text = render(doc, "v1")
        v2_text = render(doc, "v2")
        report.receipt_counts = {"v1": doc_roundtrip_count(v1_text, "v1"),
                                 "v2": doc_roundtrip_count(v2_text, "v2")}

        for request in valid:
            for seq_id, step in request.all_steps():
                row = attribute_step(request,
                                     request.sequence_map()[seq_id], step,
                                     self.config)
                report.attribution_rows.append(row.to_dict())

        self.snapshot_version += 1
        snapshot = Snapshot(
            version=self.snapshot_version,
            config_fingerprint=self.config.fingerprint(),
            config=asdict(self.config),
            exchange_version=self.exchange_version,
            conclusions=[c for c in conclusions],
            boundaries=boundaries,
            cache_events=[asdict(e) for e in self.cache.events],
            failures=[f.to_dict() for f in self._all_failures(
                invalid, injection_hits, batches)],
            todos=report.todos,
        )
        report.finished_at = time.perf_counter()
        payload_hash = self.ledger.commit_snapshot(snapshot)
        return {
            "report": report.finalize(),
            "conclusions": conclusions,
            "boundaries": boundaries,
            "snapshot_version": self.snapshot_version,
            "payload_hash": payload_hash,
            "exchange": {"v1": v1_text, "v2": v2_text},
            "todos": report.todos,
        }

    def run_incremental(self, requests, commands: list[dict]):
        before_fp = {r.request_id: r.content_fingerprint() for r in requests}
        applier = CommandApplier(requests, self.config, self.exchange_version)
        applier.apply_many([Command.from_dict(c) for c in commands])
        self.config = applier.config
        self.exchange_version = applier.exchange_version

        # Selective invalidation of only the affected steps.
        request_map = {r.request_id: r for r in requests}
        for request_id, seq_id, step_index in applier.affected:
            self.cache.invalidate_step(
                request_id, before_fp[request_id],
                self.config.fingerprint(), seq_id, step_index)
        if applier.config_changed:
            # Selectively rewrite entries; conclusions depending on changed
            # quantization/sampling config no longer match and are recomputed.
            self.cache.replace_config_fingerprint(
                self.config.fingerprint(), self.config.fingerprint())
        return self.run_full(requests)

    def _partition(self, requests):
        valid, invalid = [], []
        for request in requests:
            errs = validate_requests([request])
            if errs:
                invalid.extend(errs)
            else:
                valid.append(request)
        return valid, invalid

    @staticmethod
    def _collect(outcomes):
        boundaries = []
        conclusions = []
        for outcome in outcomes:
            conclusions.append(outcome.conclusion.to_dict())
            for boundary in outcome.boundary_dicts:
                boundaries.append({
                    "request_id": outcome.request_id,
                    "seq_id": outcome.seq_id,
                    "step_index": outcome.step_index,
                    **boundary,
                })
        return boundaries, conclusions

    @staticmethod
    def _all_failures(invalid, injection_hits, batches):
        all_failures = list(invalid) + list(injection_hits)
        for batch in batches:
            all_failures.extend(batch.failures)
        return sorted(all_failures, key=lambda e: (e.code, e.location))


def doc_roundtrip_count(text: str, version: str) -> int:
    doc = parse(text, version)
    again = render(doc, version)
    assert documents_equal(parse(again, version), doc), "round trip mismatch"
    return len(doc.records)
