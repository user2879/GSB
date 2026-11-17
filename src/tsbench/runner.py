"""Batch orchestration, concurrency control, manifests and JSON reports."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from .config import ExperimentConfig
from .errors import SeriesFailure, sort_failures
from .pipeline import SeriesProcessor, series_fingerprint
from .state import RunState
from .synthetic import SeriesData, generate_series, validate_series
from .updates import apply_batch, load_update_plan
from .util import atomic_write_json, content_fingerprint, native


class GracefulExit(Exception):
    """Raised at a safe snapshot boundary after SIGINT/SIGTERM."""

    def __init__(self, signal_name: str = "SIGTERM"):
        super().__init__(f"graceful exit requested via {signal_name}")
        self.signal_name = signal_name


def stable_task_key(series_id: str, stage: str) -> str:
    """Deterministic ordering key derived from series id and stage name."""
    return f"{series_id}::{stage}"


class WorkbenchRunner:
    def __init__(self, config: ExperimentConfig, output_dir: str):
        self.config = config
        self.params = dict(config.prediction)
        self.state = RunState(output_dir)
        self.series_map: dict[str, SeriesData] = {}
        self.failure_records: list[dict] = []
        self.global_version = 0
        self.last_batch = 0
        self.stage_versions: dict[str, int] = {
            "preprocess": 0,
            "features": 0,
            "models": 0,
            "backtest": 0,
        }
        self.signal_gate = None

    # ------------------------------------------------------------------
    def initialize_series(self) -> None:
        start_date = date.fromisoformat(
            self.config.experiment.get("start_date", "2024-01-01")
        )
        for spec in self.config.series_specs:
            try:
                series = generate_series(spec, start_date)
                validate_series(series)
                self.series_map[series.series_id] = series
            except SeriesFailure as failure:
                self.failure_records.append(failure.to_record())

    def load_recovery(self) -> dict:
        manifest = self.state.check_recoverable(self.config.config_hash, mode="run")
        if manifest is None:
            return {"recovered": False, "reason": "fresh workspace"}
        self.global_version = int(manifest["global_version"])
        self.last_batch = int(manifest["last_batch"])
        self.stage_versions = dict(manifest["stage_versions"])
        return self.state.recovery_point()

    # ------------------------------------------------------------------
    def _process_series_task(self, series_id: str) -> dict:
        # The task carries a stable key even though workers run concurrently.
        _ = stable_task_key(series_id, "full-pipeline")
        if self.signal_gate is not None:
            self.signal_gate.checkpoint()
        processor = SeriesProcessor(
            self.state,
            self.params,
            self.config.tier_params,
            self.config.block_size,
        )
        result = processor.process(self.series_map[series_id])
        if self.signal_gate is not None:
            self.signal_gate.checkpoint()
        return result

    def run_once(
        self,
        batch_id: int,
        changed_fragments: dict[str, list[str]],
        changed_indices: dict[str, list[int]],
        operations: dict[str, list[str]],
        signal_gate=None,
    ) -> dict:
        ordered_ids = sorted(self.series_map)
        concurrency = max(1, int(self.params["concurrency"]))
        stage_timings: dict[str, float] = {
            "preprocess": 0.0,
            "features": 0.0,
            "models": 0.0,
            "backtest": 0.0,
            "total": 0.0,
        }
        started = time.perf_counter()
        results: list[dict] = []

        self.signal_gate = signal_gate
        if signal_gate is not None:
            signal_gate.checkpoint()

        if concurrency == 1:
            for series_id in ordered_ids:
                results.append(self._process_series_task(series_id))
        else:
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = {
                    pool.submit(self._process_series_task, sid): sid
                    for sid in ordered_ids
                }
                collected: list[tuple[str, dict]] = []
                for future in futures:
                    sid = futures[future]
                    try:
                        collected.append((sid, future.result()))
                    except SeriesFailure as failure:
                        self.failure_records.append(failure.to_record())
                    except GracefulExit:
                        raise
                # Re-establish canonical series-id order regardless of scheduling.
                collected.sort(key=lambda item: item[0])
                results = [item[1] for item in collected]

        stage_timings["total"] = time.perf_counter() - started
        for result in results:
            for stage, seconds in result.get("stage_seconds", {}).items():
                stage_timings[stage] = stage_timings.get(stage, 0.0) + seconds

        all_reused: list[str] = []
        all_recomputed: list[str] = []
        for result in results:
            sid = result["series_id"]
            all_reused.extend(f"{sid}:{item}" for item in result["audit"]["reused"])
            all_recomputed.extend(
                f"{sid}:{item}" for item in result["audit"]["recomputed"]
            )

        data_fingerprint = content_fingerprint(
            {
                sid: series_fingerprint(self.series_map[sid])
                for sid in ordered_ids
            }
        )

        self.global_version += 1
        self.last_batch = batch_id
        for stage in self.stage_versions:
            self.stage_versions[stage] += 1

        report = {
            "report_version": 1,
            "global_version": self.global_version,
            "batch_id": batch_id,
            "config_hash": self.config.config_hash,
            "prediction_fingerprint": self.config.prediction_fingerprint(),
            "data_fingerprint": data_fingerprint,
            "concurrency": concurrency,
            "stage_versions": dict(self.stage_versions),
            "stage_timings": native(stage_timings),
            "reuse_list": sorted(all_reused),
            "recompute_list": sorted(all_recomputed),
            "changed_fragments": {
                sid: sorted(changed_fragments.get(sid, [])) for sid in ordered_ids
            },
            "changed_indices": {
                sid: sorted(changed_indices.get(sid, [])) for sid in ordered_ids
            },
            "operations": {
                sid: sorted(operations.get(sid, [])) for sid in ordered_ids
            },
            "series_results": results,
            "failures": sort_failures(self.failure_records),
        }

        report_path = os.path.join(
            self.state.report_dir, f"report-batch-{batch_id:05d}.json"
        )
        atomic_write_json(report_path, report)

        manifest = {
            "global_version": self.global_version,
            "last_batch": self.last_batch,
            "config_hash": self.config.config_hash,
            "data_fingerprint": data_fingerprint,
            "completed_series": ordered_ids,
            "stage_versions": dict(self.stage_versions),
            "latest_report": report_path,
        }
        self.state.write_manifest(manifest)
        self.state.append_log(
            {
                "batch_id": batch_id,
                "global_version": self.global_version,
                "data_fingerprint": data_fingerprint,
                "reuse_count": len(report["reuse_list"]),
                "recompute_count": len(report["recompute_list"]),
                "changed_fragments": report["changed_fragments"],
            }
        )
        return report

    # ------------------------------------------------------------------
    def apply_batches(
        self,
        batches: list[dict],
        from_batch: int = 0,
        signal_gate=None,
    ) -> list[dict]:
        reports: list[dict] = []
        known_ids = set(self.series_map)
        for batch in batches:
            batch_id = int(batch["batch_id"])
            if batch_id <= from_batch:
                continue
            try:
                changes = apply_batch(
                    self.series_map,
                    known_ids,
                    batch,
                    self.config.block_size,
                )
            except SeriesFailure as failure:
                self.failure_records.append(failure.to_record())
                changes = {}

            changed_fragments = {
                sid: sorted(record["fragments"]) for sid, record in changes.items()
            }
            changed_indices = {
                sid: sorted(record["indices"]) for sid, record in changes.items()
            }
            operations = {
                sid: sorted(set(record["operations"]))
                for sid, record in changes.items()
            }
            report = self.run_once(
                batch_id,
                changed_fragments,
                changed_indices,
                operations,
                signal_gate=signal_gate,
            )
            reports.append(report)
        return reports

    def materialize_final_state(self, batches: list[dict]) -> None:
        """Apply every update batch without producing intermediate reports."""
        known_ids = set(self.series_map)
        for batch in batches:
            try:
                apply_batch(
                    self.series_map,
                    known_ids,
                    batch,
                    self.config.block_size,
                )
            except SeriesFailure as failure:
                self.failure_records.append(failure.to_record())
