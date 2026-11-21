from __future__ import annotations

import copy
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable

from .attribution import (
    attribution_cache_key,
    attribution_to_report,
    compute_attribution_group,
)
from .bucketing import bucketing_cache_key, compute_bucketing
from .concurrency import ParallelRunner
from .drift import compute_drift_group, drift_cache_key, drift_to_report
from .errors import Failure, sorted_failures
from .fingerprint import digest
from .metrics import compute_metric, metric_cache_key
from .models import Config, Exposure, WorkspaceState
from .snapshot import STAGES, SnapshotStore


class ContentCache:
    def __init__(self, full_rebuild: bool, snapshots: SnapshotStore | None = None):
        self.full_rebuild = full_rebuild
        self.snapshots = snapshots
        self.entries: dict[str, dict[str, Any]] = {stage: {} for stage in STAGES}
        self.entry_origin: dict[str, dict[str, str]] = {stage: {} for stage in STAGES}
        self.reused: list[dict[str, str]] = []
        self.recomputed: list[dict[str, str]] = []
        self.compute_calls = 0

    def load_from_disk(self) -> None:
        if self.snapshots is None:
            return
        for stage, entries in self.snapshots.load_entries().items():
            self.entries[stage] = entries
            self.entry_origin[stage] = {key: "disk" for key in entries}

    def get_or_compute(
        self,
        stage: str,
        key: str,
        compute: Callable[[], dict[str, Any]],
    ) -> dict[str, Any]:
        if not self.full_rebuild and key in self.entries[stage]:
            origin = self.entry_origin[stage].get(key, "memory")
            self.reused.append({"stage": stage, "key": key, "origin": origin})
            return copy.deepcopy(self.entries[stage][key])
        self.compute_calls += 1
        artifact = compute()
        self.entries[stage][key] = copy.deepcopy(artifact)
        self.entry_origin[stage][key] = "memory"
        self.recomputed.append({"stage": stage, "key": key})
        return artifact


class DiagnosticEngine:
    def __init__(
        self,
        *,
        full_rebuild: bool = False,
        snapshot_dir: str | Path | None = None,
        check_snapshot: bool = True,
    ):
        self.full_rebuild = full_rebuild
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.snapshots = SnapshotStore(self.snapshot_dir) if self.snapshot_dir else None
        self.check_snapshot = check_snapshot
        self.startup_checked = False
        self.startup_failures: list[Failure] = []
        self.snapshot_status = "disabled" if self.snapshots is None else "unchecked"
        self.snapshot_manifest: dict[str, Any] | None = None
        self.cache = ContentCache(full_rebuild, self.snapshots)

    def _check_startup(self, state: WorkspaceState) -> None:
        if self.startup_checked:
            return
        self.startup_checked = True
        if self.snapshots is None or not self.check_snapshot:
            return
        inspection = self.snapshots.inspect(state.config, state.specs)
        self.snapshot_status = inspection["status"]
        self.snapshot_manifest = inspection["manifest"]
        self.startup_failures.extend(inspection["failures"])
        if inspection["status"] == "resumed" and not self.full_rebuild:
            self.cache.load_from_disk()
        elif inspection["status"] == "new":
            self.snapshot_status = "new"

    def _maybe_commit(self, state: WorkspaceState, batch_no: int | None, force_commit: bool) -> None:
        if self.snapshots is None:
            return
        interval = state.config.snapshot_commit_interval_batches
        should_commit = force_commit or batch_no is None or (batch_no > 0 and batch_no % interval == 0)
        if not should_commit:
            if self.snapshot_status not in {"stale", "resumed", "new", "rebuilt"}:
                self.snapshot_status = "pending"
            return
        was_stale = self.snapshot_status == "stale"
        self.snapshot_manifest = self.snapshots.commit(
            self.cache.entries,
            state.config,
            state.specs,
            committed_batch=batch_no,
        )
        self.snapshot_status = "rebuilt" if was_stale else "committed"

    def run(
        self,
        state: WorkspaceState,
        *,
        batch_no: int | None = None,
        force_commit: bool = False,
    ) -> dict[str, Any]:
        self._check_startup(state)
        self.cache.reused = []
        self.cache.recomputed = []
        self.cache.compute_calls = 0
        started = time.perf_counter()
        timings: dict[str, float] = {}
        failures: list[Failure] = list(state.failures) + list(self.startup_failures)

        with ParallelRunner(
            state.config.concurrency,
            chunk_size=max(1, state.config.batch_size),
        ) as runner:
            bucket_started = time.perf_counter()
            bucket_key = bucketing_cache_key(
                state.sorted_exposures(), state.properties, state.config
            )
            bucketing = self.cache.get_or_compute(
                "bucketing",
                bucket_key,
                lambda: compute_bucketing(
                    state.sorted_exposures(),
                    state.properties,
                    state.config,
                    runner,
                    cache_key=bucket_key,
                ),
            )
            timings["bucketing_ms"] = (time.perf_counter() - bucket_started) * 1000.0

            assignment_map = bucketing["assignments"]
            group_events: dict[str, list[Exposure]] = {group: [] for group in state.config.group_order}
            mismatch_rows: list[dict[str, str]] = []
            for event in state.sorted_exposures():
                derived_group = assignment_map[event.user_id]["group"]
                group_events[derived_group].append(event)
                if event.experiment_group != derived_group:
                    mismatch_rows.append(
                        {
                            "request_id": event.request_id,
                            "user_id": event.user_id,
                            "logged_group": event.experiment_group,
                            "derived_group": derived_group,
                        }
                    )
            for group in group_events:
                group_events[group].sort(key=lambda item: (item.request_id, item.item_id))
            bucketing_group_report = {
                group: {
                    "user_count": bucketing["groups"][group]["user_count"],
                    "event_count": len(group_events[group]),
                    "user_fingerprint": bucketing["groups"][group]["user_fingerprint"],
                }
                for group in state.config.group_order
            }

            versions = state.spec_versions()
            metric_started = time.perf_counter()
            metrics: dict[str, dict[str, dict[str, Any]]] = {}
            for version in versions:
                spec = state.specs[version]
                metrics[version] = {}
                for group in state.config.group_order:
                    events = group_events[group]
                    events_fingerprint = digest(
                        {"group": group, "events": [event.content_dict() for event in events]}
                    )
                    key = metric_cache_key(
                        version, group, events_fingerprint, spec, state.config
                    )
                    artifact = self.cache.get_or_compute(
                        "metrics",
                        key,
                        lambda v=version, g=group, s=spec, ev=events, fp=events_fingerprint, k=key: compute_metric(
                            ev,
                            g,
                            s,
                            state.config,
                            runner,
                            cache_key=k,
                            upstream={
                                "bucket_cache_key": bucket_key,
                                "event_content_fingerprint": fp,
                                "spec_fingerprint": s.fingerprint(),
                            },
                        ),
                    )
                    metrics[version][group] = artifact
            timings["metrics_ms"] = (time.perf_counter() - metric_started) * 1000.0

            active = state.config.active_spec_version
            active_index = versions.index(active) if active in versions else -1
            from_version = versions[active_index - 1] if active_index > 0 else None
            drift_groups: dict[str, dict[str, Any]] = {}
            attribution_groups: dict[str, dict[str, Any]] = {}

            if from_version is not None:
                from_spec = state.specs[from_version]
                to_spec = state.specs[active]
                drift_started = time.perf_counter()
                for group in state.config.group_order:
                    from_metric = metrics[from_version][group]
                    to_metric = metrics[active][group]
                    key = drift_cache_key(
                        from_version,
                        active,
                        group,
                        from_metric["cache_key"],
                        to_metric["cache_key"],
                        from_spec,
                        to_spec,
                        state.config,
                    )
                    artifact = self.cache.get_or_compute(
                        "drift",
                        key,
                        lambda g=group, fm=from_metric, tm=to_metric, k=key: compute_drift_group(
                            group_events[g],
                            g,
                            from_spec,
                            to_spec,
                            fm,
                            tm,
                            state.config,
                            runner,
                            cache_key=k,
                        ),
                    )
                    drift_groups[group] = artifact
                timings["drift_ms"] = (time.perf_counter() - drift_started) * 1000.0

                attribution_started = time.perf_counter()
                for group in state.config.group_order:
                    drift_artifact = drift_groups[group]
                    key = attribution_cache_key(
                        from_version, active, group, drift_artifact["cache_key"], state.config
                    )
                    artifact = self.cache.get_or_compute(
                        "attribution",
                        key,
                        lambda g=group, da=drift_artifact, k=key: compute_attribution_group(
                            group_events[g],
                            g,
                            from_spec,
                            to_spec,
                            metrics[from_version][g],
                            metrics[active][g],
                            da,
                            state.config,
                            runner,
                            cache_key=k,
                        ),
                    )
                    attribution_groups[group] = artifact
                    for source in artifact["order_sensitive_sources"]:
                        failures.append(
                            Failure(
                                "E_ATTRIB_AMBIGUOUS",
                                "source contribution depends on attribution order; configured order selected",
                                field_path=f"attribution.{group}.{source}",
                                severity="WARNING",
                                context={
                                    "group": group,
                                    "source": source,
                                    "chosen_order": list(state.config.attribution_order),
                                    "range_fixed": artifact["order_sensitivity"][source]["range_fixed"],
                                },
                            )
                        )
                timings["attribution_ms"] = (time.perf_counter() - attribution_started) * 1000.0
            else:
                timings["drift_ms"] = 0.0
                timings["attribution_ms"] = 0.0

            timings["total_ms"] = (time.perf_counter() - started) * 1000.0
            concurrency_stats = runner.stats()

        failures = sorted_failures(failures)
        self._maybe_commit(state, batch_no, force_commit)
        snapshot_report = (
            self.snapshots.report(self.snapshot_status, self.snapshot_manifest)
            if self.snapshots is not None
            else {"enabled": False, "status": "disabled"}
        )
        reuse = self._reuse_report()
        report = {
            "schema_version": 1,
            "report_type": "diagnostic",
            "mode": "full_rebuild" if self.full_rebuild else "incremental",
            "state_fingerprint": state.state_fingerprint(),
            "config": state.config.to_dict(),
            "config_fingerprint": state.config.fingerprint(),
            "spec_versions": versions,
            "specs": [state.specs[version].to_dict() for version in versions],
            "active_spec_version": active,
            "active_pair": {
                "from_version": from_version,
                "to_version": active if from_version else None,
            },
            "bucketing": {
                "cache_key": bucketing["cache_key"],
                "content_fingerprint": bucketing["content_fingerprint"],
                "mapping_rule": bucketing["mapping_rule"],
                "groups": bucketing_group_report,
                "fallbacks": bucketing["fallbacks"],
                "fallback_count": bucketing["fallback_count"],
                "log_group_mismatches": mismatch_rows,
                "log_group_mismatch_count": len(mismatch_rows),
                "assignments": [
                    bucketing["assignments"][user_id]
                    for user_id in sorted(bucketing["assignments"])
                ],
            },
            "metrics": {
                "versions": [
                    {
                        "spec_version": version,
                        "groups": [
                            self._metric_report(metrics[version][group])
                            for group in state.config.group_order
                        ],
                    }
                    for version in versions
                ],
                "active": {
                    "spec_version": active,
                    "groups": [
                        self._metric_report(metrics[active][group])
                        for group in state.config.group_order
                    ],
                },
            },
            "drift": {
                "pair_available": from_version is not None,
                "from_version": from_version,
                "to_version": active if from_version else None,
                "groups": [
                    drift_to_report(drift_groups[group])
                    for group in state.config.group_order
                    if group in drift_groups
                ],
            },
            "attribution": {
                "pair_available": from_version is not None,
                "from_version": from_version,
                "to_version": active if from_version else None,
                "order": list(state.config.attribution_order),
                "groups": [
                    attribution_to_report(attribution_groups[group])
                    for group in state.config.group_order
                    if group in attribution_groups
                ],
            },
            "reuse": reuse,
            "concurrency": concurrency_stats,
            "snapshot": snapshot_report,
            "failures": [failure.to_dict() for failure in failures],
            "timing_ms": {key: round(value, 6) for key, value in timings.items()},
        }
        return report

    @staticmethod
    def _metric_report(artifact: dict[str, Any]) -> dict[str, Any]:
        return {
            "group": artifact["group"],
            "spec_version": artifact["spec_version"],
            "metric_name": artifact["metric_name"],
            "value_fixed": artifact["value_fixed"],
            "value": artifact["value"],
            "numerator_weight": artifact["numerator_weight"],
            "denominator_weight": artifact["denominator_weight"],
            "participant_count": artifact["participant_count"],
            "eligible_before_dedup_count": artifact["eligible_before_dedup_count"],
            "dedup_removed_count": artifact["dedup_removed_count"],
            "request_set_fingerprint": artifact["request_set_fingerprint"],
            "event_refs": artifact["event_refs"],
            "cache_key": artifact["cache_key"],
        }

    def _reuse_report(self) -> dict[str, Any]:
        by_stage: dict[str, dict[str, int]] = {
            stage: {"reused": 0, "recomputed": 0} for stage in STAGES
        }
        for row in self.cache.reused:
            by_stage[row["stage"]]["reused"] += 1
        for row in self.cache.recomputed:
            by_stage[row["stage"]]["recomputed"] += 1
        reused = sorted(self.cache.reused, key=lambda row: (row["stage"], row["key"]))
        recomputed = sorted(self.cache.recomputed, key=lambda row: (row["stage"], row["key"]))
        return {
            "mode": "full_rebuild" if self.full_rebuild else "incremental",
            "counts": {
                "reused": len(reused),
                "recomputed": len(recomputed),
                "actual_compute_calls": self.cache.compute_calls,
                "accounted_attempts": len(reused) + len(recomputed),
            },
            "by_stage": by_stage,
            "reused": reused,
            "recomputed": recomputed,
            "consistent": self.cache.compute_calls == len(recomputed),
        }


def comparable_report(report: dict[str, Any]) -> dict[str, Any]:
    """Fields that must be byte-for-byte equivalent between full and incremental runs."""
    return {
        "active_spec_version": report["active_spec_version"],
        "active_pair": report["active_pair"],
        "bucketing": {
            "content_fingerprint": report["bucketing"]["content_fingerprint"],
            "groups": report["bucketing"]["groups"],
            "fallback_count": report["bucketing"]["fallback_count"],
            "fallbacks": report["bucketing"]["fallbacks"],
            "log_group_mismatch_count": report["bucketing"]["log_group_mismatch_count"],
            "log_group_mismatches": report["bucketing"]["log_group_mismatches"],
            "assignments": report["bucketing"]["assignments"],
            "mapping_rule": report["bucketing"]["mapping_rule"],
        },
        "metrics": report["metrics"],
        "drift": report["drift"],
        "attribution": report["attribution"],
        "failures": report["failures"],
    }
