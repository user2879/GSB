"""End-to-end evaluation pipeline with incremental reuse and snapshots."""
from __future__ import annotations

import json
import os
import signal
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from . import attribution as attribution_mod
from . import estimation as estimation_mod
from . import hashing
from . import stats as stats_mod
from .cache import (
    STAGE_ATTRIBUTION,
    STAGE_BUCKETING,
    STAGE_ESTIMATION,
    STAGE_STATS,
    ArtifactCache,
)
from .config import ExperimentConfig
from .errors import E_CONFIG_MISMATCH, FailureRecord
from .fingerprint import CALIBER_VERSION, fingerprint_file, fingerprint_json, fingerprint_text
from .injection import InjectionHit, apply_loader_injections
from .loader import load_attributes, load_exposures
from .snapshot import SnapshotStore


@dataclass
class RunResult:
    report: Dict[str, Any]
    failures: List[FailureRecord]
    exit_code: int = 0
    signal_handled: Optional[str] = None

    def to_dict(self):
        data = dict(self.report)
        data["failures"] = [f.to_dict() for f in self.failures]
        data["exit_code"] = self.exit_code
        data["signal_handled"] = self.signal_handled
        return data


class GracefulStop:
    def __init__(self):
        self.requested = False
        self.signal_name: Optional[str] = None

    def handler(self, signum, _frame):
        self.requested = True
        self.signal_name = signal.Signals(signum).name


def _read_jsonl_records(path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def _stage_event_ids(exposures, groups: Optional[List[str]] = None) -> List[str]:
    ids = set()
    for exposure in exposures:
        if groups is None or exposure.group in groups:
            ids.add(exposure.request_id)
    return sorted(ids)


class Workbench:
    def __init__(
        self,
        config: ExperimentConfig,
        snapshot_dir: str,
        cache_dir: str,
        tmp_dir: str,
        concurrent: bool = True,
    ):
        self.config = config
        self.fp = config.fixedpoint
        self.snapshot = SnapshotStore(snapshot_dir)
        self.cache = ArtifactCache(cache_dir)
        self.tmp_dir = tmp_dir
        self.concurrent = concurrent
        os.makedirs(tmp_dir, exist_ok=True)
        self.config_fp = config.fingerprint()

    # ---------- stage 1: bucketing ----------
    def run_bucketing(
        self, attributes: Dict[str, Dict[str, str]]
    ) -> Tuple[Dict[str, Any], List[FailureRecord]]:
        attr_fp = fingerprint_json(attributes)
        key = self.cache.make_key(
            STAGE_BUCKETING, attr_fp, self.config_fp, scope="all-users"
        )
        cached = self.cache.get(key)
        if cached is not None:
            return cached.payload, []
        assignments, hash_failures = hashing.assign_users(
            sorted(attributes.keys()), self.config
        )
        payload = {
            "assignments": [assignments[uid].to_dict() for uid in sorted(assignments)],
            "num_buckets": self.config.num_buckets,
            "hash_seed": self.config.hash_seed,
            "group_order": self.config.group_order,
        }
        self.cache.put(
            STAGE_BUCKETING,
            "all-users",
            attr_fp,
            self.config_fp,
            payload,
            event_ids=set(attributes.keys()),
        )
        return payload, hash_failures

    # ---------- stage 2: group stats ----------
    def run_stats(
        self, exposures, assignments_payload, log_fp: str
    ) -> Tuple[Dict[str, Any], List[FailureRecord]]:
        assignment_map = {
            row["user_id"]: hashing.BucketAssignment(
                row["user_id"], row["bucket"], row["group"]
            )
            for row in assignments_payload["assignments"]
        }
        grouped: Dict[str, List] = {g: [] for g in self.config.group_order}
        for exposure in exposures:
            if exposure.group in grouped:
                grouped[exposure.group].append(exposure)

        stats_payload: Dict[str, Any] = {}
        failures: List[FailureRecord] = []
        expected_keys = []
        for group in self.config.group_order:
            group_exposures = sorted(
                grouped[group], key=lambda e: (e.request_id, e.item_id)
            )
            group_fp = fingerprint_json(
                [e.to_dict() for e in group_exposures]
            )
            key = self.cache.make_key(
                STAGE_STATS, group_fp, self.config_fp, scope=group
            )
            expected_keys.append(key)
            cached = self.cache.get(key)
            if cached is not None:
                stats_payload[group] = cached.payload
                continue
            group_stats_map, group_failures = stats_mod.compute_group_stats(
                group_exposures,
                self.config,
                assignment_map,
                self.fp,
                concurrent=self.concurrent,
            )
            failures.extend(group_failures)
            payload = group_stats_map[group].to_dict()
            self.cache.put(
                STAGE_STATS,
                group,
                group_fp,
                self.config_fp,
                payload,
                event_ids=set(_stage_event_ids(group_exposures)),
            )
            stats_payload[group] = payload
        self.cache.prune_stage_scope(
            STAGE_STATS, self.config.group_order, expected_keys
        )
        failures.sort(key=lambda f: f.sort_key())
        return stats_payload, failures

    # ---------- stage 3: estimation ----------
    def run_estimation(self, stats_payload, exposures) -> Dict[str, Any]:
        group_stats = {
            group: _rehydrate_group_stat(row) for group, row in stats_payload.items()
        }
        output: Dict[str, Any] = {}
        control = self.config.control_group
        expected_keys = []
        event_ids_by_group: Dict[str, Set[str]] = {g: set() for g in self.config.group_order}
        for exposure in exposures:
            event_ids_by_group.setdefault(exposure.group, set()).add(exposure.request_id)
        for treatment in self.config.treatment_groups:
            upstream = fingerprint_json(
                [stats_payload[control], stats_payload[treatment]]
            )
            key = self.cache.make_key(
                STAGE_ESTIMATION, upstream, self.config_fp, scope=treatment
            )
            expected_keys.append(key)
            cached = self.cache.get(key)
            if cached is not None:
                output[treatment] = cached.payload
                continue
            estimate = estimation_mod.estimate_pair(
                group_stats[control],
                group_stats[treatment],
                self.config.primary_metric,
                self.config,
                self.fp,
            )
            event_ids = set(
                stats_payload[treatment].get("drop_reasons", {}).keys()
            )
            self.cache.put(
                STAGE_ESTIMATION,
                treatment,
                upstream,
                self.config_fp,
                estimate.to_dict(),
                event_ids=event_ids_by_group[control] | event_ids_by_group[treatment],
            )
            output[treatment] = estimate.to_dict()
        self.cache.prune_stage_scope(
            STAGE_ESTIMATION, self.config.treatment_groups, expected_keys
        )
        return output

    # ---------- stage 4: attribution ----------
    def run_attribution(
        self, exposures, attributes
    ) -> Tuple[List[Dict[str, Any]], List[FailureRecord]]:
        attr_key_fp = fingerprint_json(
            sorted({k for attrs in attributes.values() for k in attrs})
        )
        exposure_fp = fingerprint_json([e.to_dict() for e in exposures])
        upstream = fingerprint_text(f"{attr_key_fp}|{exposure_fp}")
        key = self.cache.make_key(
            STAGE_ATTRIBUTION, upstream, self.config_fp, scope="all-keys"
        )
        cached = self.cache.get(key)
        if cached is not None:
            payload = cached.payload
            return payload["layers"], _rehydrate_failures(payload.get("drifts", []))
        layers, drifts = attribution_mod.attribute(
            exposures,
            self.config,
            attributes,
            self.fp,
            concurrent=self.concurrent,
        )
        payload = {"layers": layers, "drifts": [d.to_dict() for d in drifts]}
        self.cache.put(
            STAGE_ATTRIBUTION,
            "all-keys",
            upstream,
            self.config_fp,
            payload,
            event_ids=set(_stage_event_ids(exposures)),
        )
        self.cache.prune_stage_scope(
            STAGE_ATTRIBUTION, ["all-keys"], [key]
        )
        return layers, drifts

    # ---------- full run ----------
    def run(
        self,
        log_path: str,
        attributes_path: str,
        injections: Optional[List[Dict[str, Any]]] = None,
        graceful: Optional[GracefulStop] = None,
    ) -> RunResult:
        injections = injections or []
        all_failures: List[FailureRecord] = []
        injection_hits: List[InjectionHit] = []

        mismatch = self.snapshot.resume_check(self.config_fp, CALIBER_VERSION)
        stale_injection = self._stale_injection_requested(injections)
        if mismatch is not None:
            # A legitimate config change invalidates derived artifacts and starts
            # a fresh snapshot lineage; the error code is reserved for the explicit
            # stale-snapshot injection path below.
            self.cache.invalidate_on_config_change(self.config_fp)
            self.snapshot.reset()
            self._config_lineage_reset = True
            if stale_injection:
                injection_hits.append(
                    InjectionHit(
                        injection_id=stale_injection,
                        stage="snapshot",
                        fault="stale_snapshot_reuse",
                        target="global",
                        code=E_CONFIG_MISMATCH,
                    )
                )
                all_failures.append(
                    FailureRecord(
                        stage="snapshot",
                        code=E_CONFIG_MISMATCH,
                        reason="stale_snapshot_reuse",
                        injection_id=stale_injection,
                    )
                )
        else:
            self._config_lineage_reset = False

        records = _read_jsonl_records(log_path)
        loader_hits = apply_loader_injections(records, injections)
        injection_hits.extend(loader_hits)
        injected_path = os.path.join(self.tmp_dir, "injected_exposures.jsonl")
        with open(injected_path, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        load_result = load_exposures(injected_path, set(self.config.group_order))
        all_failures.extend(load_result.failures)
        attributes = load_attributes(attributes_path)

        assignments_payload, hash_failures = self.run_bucketing(attributes)
        all_failures.extend(hash_failures)

        exposures = load_result.sorted_exposures
        batches = _split_batches(exposures, self.config.batch_size)
        completed = 0
        for batch_index, batch in enumerate(batches, start=1):
            completed = batch_index
            if graceful is not None and graceful.requested:
                break
        stats_payload, stats_failures = self.run_stats(
            exposures, assignments_payload, load_result.file_fingerprint
        )
        all_failures.extend(stats_failures)
        estimation_payload = self.run_estimation(stats_payload, exposures)
        layers, drifts = self.run_attribution(exposures, attributes)
        all_failures.extend(drifts)

        self.cache.save()
        global_version = (
            self.snapshot.load_manifest().get("global_version", 0) + 1
            if self.snapshot.has_manifest()
            else 1
        )
        stage_payloads = {
            STAGE_BUCKETING: assignments_payload,
            STAGE_STATS: stats_payload,
            STAGE_ESTIMATION: estimation_payload,
            STAGE_ATTRIBUTION: {"layers": layers},
        }
        manifest = self.snapshot.commit(
            global_version=global_version,
            config_fp=self.config_fp,
            caliber=CALIBER_VERSION,
            completed_batches=completed,
            stage_payloads=stage_payloads,
        )

        report = _base_report(self.config, log_path)
        report.update(
            {
                "input_fingerprint": load_result.file_fingerprint,
                "config_fingerprint": self.config_fp,
                "caliber_version": CALIBER_VERSION,
                "manifest": manifest,
                "bucketing": assignments_payload,
                "group_stats": stats_payload,
                "estimation": estimation_payload,
                "attribution": layers,
                "cache": self.cache.change_report(),
                "injection_hits": [h.to_dict() for h in sorted(injection_hits, key=lambda h: (h.injection_id, h.target))],
                "failure_summary": _summarize_failures(all_failures),
                "mode": "concurrent" if self.concurrent and self.config.concurrency > 1 else "serial",
                "concurrency": self.config.concurrency,
                "completed_batches": completed,
                "total_batches": len(batches),
            }
        )
        all_failures.sort(key=lambda f: f.sort_key())
        signal_name = graceful.signal_name if graceful is not None and graceful.requested else None
        return RunResult(report=report, failures=all_failures, exit_code=0, signal_handled=signal_name)

    def _stale_injection_requested(self, injections) -> Optional[str]:
        from .injection import FAULT_STALE_SNAPSHOT

        for spec in injections:
            if spec.get("fault") != FAULT_STALE_SNAPSHOT:
                continue
            if not self.snapshot.has_manifest():
                continue
            manifest = self.snapshot.load_manifest()
            if manifest.get("config_fingerprint") != self.config_fp:
                return str(spec["id"])
        return None


def _split_batches(exposures, batch_size: int) -> List[List]:
    return [exposures[i : i + batch_size] for i in range(0, len(exposures), batch_size)]


def _rehydrate_group_stat(row: Dict[str, Any]) -> stats_mod.GroupStat:
    return stats_mod.GroupStat(
        group=row["group"],
        exposures=row["exposures"],
        requests=row["requests"],
        metrics=row["metrics"],
        drop_reasons=row["drop_reasons"],
    )


def _rehydrate_failures(rows: List[Dict[str, Any]]) -> List[FailureRecord]:
    return [FailureRecord(**row) for row in rows]


def _summarize_failures(failures: List[FailureRecord]) -> Dict[str, int]:
    summary: Dict[str, int] = {}
    for failure in failures:
        summary[failure.code] = summary.get(failure.code, 0) + 1
    return dict(sorted(summary.items()))


def _base_report(config: ExperimentConfig, log_path: str) -> Dict[str, Any]:
    return {
        "experiment_id": config.experiment_id,
        "log_file": os.path.basename(log_path),
        "primary_metric": config.primary_metric,
        "guardrail_metrics": config.guardrail_metrics,
        "metric_direction": config.metric_direction,
        "min_sample_size": config.min_sample_size,
        "min_lift": config.min_lift,
        "decimal_width": config.decimal_width,
        "quantization_step": config.quantization_step,
    }
