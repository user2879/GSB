"""Incremental orchestration, cache audit, concurrency, and recovery."""

from __future__ import annotations

import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .binning import binning_cache_key, compute_binning
from .config import ExperimentConfig, apply_config_edits, validate_experiment
from .data import PreparedData, prepare_data
from .ensemble import (
    compute_contributions,
    compute_ensemble,
    contributions_cache_key,
    ensemble_cache_key,
    tree_product_fingerprint,
)
from .errors import BusinessFailure, BusinessValidationError, SnapshotVersionChanged, sorted_failures
from .fingerprint import fingerprint
from .report import write_json_report
from .snapshot import Snapshot, SnapshotManager
from .tree import train_tree


@dataclass
class RunResult:
    status: str
    exit_code: int
    failures: list[dict[str, Any]]
    artifacts: dict[str, Any]
    cache_audit: dict[str, Any]
    report: dict[str, Any]
    report_path: str | None
    effective_config: ExperimentConfig | None
    data_summary: dict[str, Any] | None


class Engine:
    def __init__(
        self,
        state_dir: str | Path,
        tmp_dir: str | Path | None = None,
        report_dir: str | Path | None = None,
    ):
        self.state_dir = Path(state_dir)
        self.tmp_dir = Path(tmp_dir) if tmp_dir is not None else self.state_dir / "tmp"
        self.report_dir = Path(report_dir) if report_dir is not None else self.state_dir / "reports"
        self.snapshot_manager = SnapshotManager(self.state_dir)

    def run(
        self,
        config_data: dict[str, Any],
        spec_data: dict[str, Any],
        edit_batches: Iterable[dict[str, Any]] = (),
        *,
        label: str = "run",
        report_name: str | None = None,
        faults: dict[int, str] | None = None,
        stop_event: threading.Event | None = None,
        tree_delay_ms: float = 0.0,
        concurrency_override: int | None = None,
    ) -> RunResult:
        started = time.perf_counter()
        timings: dict[str, float] = {}
        faults = dict(faults or {})
        stop_event = stop_event or threading.Event()
        batches = [batch for batch in edit_batches]
        snapshot_load_started = time.perf_counter()
        loaded = self.snapshot_manager.load_current()
        loaded_recovery = loaded.recovery_info() if loaded is not None else None
        timings["snapshot_load_ms"] = (time.perf_counter() - snapshot_load_started) * 1000.0

        config: ExperimentConfig | None = None
        data: PreparedData | None = None
        artifacts: dict[str, Any] = {}
        audit: dict[str, Any] = self._empty_audit()
        failures: list[BusinessFailure | dict[str, Any]] = []
        last_snapshot: Snapshot | None = loaded

        try:
            if not isinstance(config_data, dict):
                raise BusinessValidationError(
                    [
                        BusinessFailure(
                            code="CFG_TYPE_MISMATCH",
                            field="config",
                            message="configuration root must be an object",
                            stage="configuration",
                        )
                    ]
                )
            if not isinstance(spec_data, dict):
                raise BusinessValidationError(
                    [
                        BusinessFailure(
                            code="CFG_TYPE_MISMATCH",
                            field="data_spec",
                            message="data-spec root must be an object",
                            stage="configuration",
                        )
                    ]
                )
            config, validation_failures = validate_experiment(config_data, spec_data, batches)
            if validation_failures:
                raise BusinessValidationError(validation_failures)
            assert config is not None
            config = apply_config_edits(config, batches)
            if concurrency_override is not None:
                if not isinstance(concurrency_override, int) or isinstance(concurrency_override, bool) or concurrency_override <= 0:
                    raise BusinessValidationError(
                        [
                            BusinessFailure(
                                code="CFG_CONCURRENCY_INVALID",
                                field="runtime.thread_pool_size",
                                message="concurrency override must be a positive integer",
                                stage="configuration",
                            )
                        ]
                    )
                config = config.with_concurrency(concurrency_override)
            self._ensure_tmp_dir()
            config_hash = config.config_hash()

            data_started = time.perf_counter()
            data = prepare_data(config, spec_data, batches)
            timings["data_ms"] = (time.perf_counter() - data_started) * 1000.0

            binning_key = binning_cache_key(config, data)
            baseline_binning = loaded.stages.get("binning") if loaded is not None else None
            binning_started = time.perf_counter()
            if baseline_binning is not None and baseline_binning.get("key") == binning_key:
                binning = baseline_binning
                audit["binning"] = {
                    "status": "reused",
                    "reason": "matching-content-key",
                    "key": binning_key,
                    "fingerprint": binning.get("fingerprint"),
                }
            else:
                binning = compute_binning(config, data, binning_key)
                audit["binning"] = {
                    "status": "recomputed",
                    "reason": self._binning_miss_reason(config, data, baseline_binning, loaded),
                    "key": binning_key,
                    "fingerprint": binning.get("fingerprint"),
                }
            timings["binning_ms"] = (time.perf_counter() - binning_started) * 1000.0
            artifacts["binning"] = binning

            trees_by_index: dict[int, dict[str, Any]] = {}
            tree_keys: dict[int, str] = {}
            missing: list[int] = []
            for tree_index in range(config.tree_count):
                from .tree import tree_cache_key

                key = tree_cache_key(config, data, binning, tree_index, faults.get(tree_index))
                tree_keys[tree_index] = key
                baseline_tree = loaded.trees.get(tree_index) if loaded is not None else None
                if baseline_tree is not None and baseline_tree.get("key") == key:
                    trees_by_index[tree_index] = baseline_tree
                    audit["trees"].append(
                        {
                            "tree_index": tree_index,
                            "status": "reused",
                            "reason": "matching-content-key",
                            "key": key,
                            "artifact_fingerprint": baseline_tree.get("artifact_fingerprint"),
                        }
                    )
                else:
                    missing.append(tree_index)
                    audit["trees"].append(
                        {
                            "tree_index": tree_index,
                            "status": "pending",
                            "reason": self._tree_miss_reason(
                                tree_index,
                                data,
                                binning,
                                baseline_binning,
                                baseline_tree,
                                loaded,
                            ),
                            "key": key,
                            "artifact_fingerprint": None,
                        }
                    )

            current_version = loaded.version if loaded is not None else 0
            timings["snapshot_commit_ms"] = timings.get("snapshot_commit_ms", 0.0)
            new_completed = 0
            peak_inflight = 0

            def commit_snapshot(
                *,
                include_ensemble: bool = False,
                include_contributions: bool = False,
            ) -> None:
                nonlocal current_version, last_snapshot, timings
                commit_started = time.perf_counter()
                last_snapshot = self.snapshot_manager.commit(
                    expected_version=current_version,
                    config_hash=config_hash,
                    data_fingerprint=data.data_fingerprint,
                    binning=binning,
                    trees=trees_by_index,
                    ensemble=artifacts.get("ensemble") if include_ensemble else None,
                    contributions=artifacts.get("contributions") if include_contributions else None,
                )
                current_version = last_snapshot.version
                timings["snapshot_commit_ms"] += (time.perf_counter() - commit_started) * 1000.0

            tree_started = time.perf_counter()
            if missing:
                workers = max(1, min(config.thread_pool_size, len(missing)))
                pending = sorted(missing)
                futures: dict[Future[dict[str, Any]], int] = {}

                def task(index: int) -> dict[str, Any]:
                    result = train_tree(config, data, binning, index, faults.get(index))
                    if tree_delay_ms > 0:
                        time.sleep(float(tree_delay_ms) / 1000.0)
                    return result

                with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tree") as executor:
                    while pending or futures:
                        while pending and len(futures) < workers and not stop_event.is_set():
                            tree_index = pending.pop(0)
                            futures[executor.submit(task, tree_index)] = tree_index
                        if not futures:
                            break
                        peak_inflight = max(peak_inflight, len(futures))
                        done, _ = wait(futures.keys(), return_when=FIRST_COMPLETED)
                        for future in sorted(done, key=lambda item: futures[item]):
                            tree_index = futures.pop(future)
                            try:
                                tree_artifact = future.result()
                            except Exception as exception:  # pragma: no cover - defensive thread boundary.
                                tree_artifact = self._unexpected_tree_failure(tree_index, tree_keys[tree_index], exception)
                            trees_by_index[tree_index] = tree_artifact
                            new_completed += 1
                            audit["trees"][tree_index] = {
                                "tree_index": tree_index,
                                "status": "recomputed",
                                "reason": audit["trees"][tree_index]["reason"],
                                "key": tree_keys[tree_index],
                                "artifact_fingerprint": tree_artifact.get("artifact_fingerprint"),
                            }
                            if new_completed % config.snapshot_interval == 0:
                                commit_snapshot()
                        if stop_event.is_set() and not pending and not futures:
                            break
            timings["tree_training_ms"] = (time.perf_counter() - tree_started) * 1000.0
            artifacts["trees"] = [trees_by_index[index] for index in sorted(trees_by_index)]

            if stop_event.is_set() and len(trees_by_index) < config.tree_count:
                commit_snapshot()
                interrupted_failures = sorted_failures(
                    tree["failure"]
                    for tree in trees_by_index.values()
                    if tree.get("status") != "ok" and tree.get("failure") is not None
                )
                audit["concurrency"] = {
                    "configured_workers": config.thread_pool_size,
                    "actual_workers": max(1, min(config.thread_pool_size, max(1, len(missing)))),
                    "peak_inflight": peak_inflight,
                    "new_trees_completed": new_completed,
                }
                result = self._finish(
                    status="interrupted",
                    exit_code=0,
                    failures=interrupted_failures,
                    artifacts=artifacts,
                    audit=self._finalize_audit(audit),
                    config=config,
                    data=data,
                    timings=timings,
                    loaded_recovery=loaded_recovery,
                    committed_snapshot=last_snapshot,
                    label=label,
                    report_name=report_name,
                    started=started,
                )
                return result

            ordered_trees = [trees_by_index[index] for index in range(config.tree_count)]
            failures.extend(
                tree["failure"]
                for tree in ordered_trees
                if tree.get("status") != "ok" and tree.get("failure") is not None
            )
            product_fingerprint = tree_product_fingerprint(ordered_trees)
            ensemble_key = ensemble_cache_key(config, data, ordered_trees)
            baseline_ensemble = loaded.stages.get("ensemble") if loaded is not None else None
            ensemble_started = time.perf_counter()
            if baseline_ensemble is not None and baseline_ensemble.get("key") == ensemble_key:
                ensemble = baseline_ensemble
                audit["ensemble"] = {
                    "status": "reused",
                    "reason": "matching-content-key",
                    "key": ensemble_key,
                    "fingerprint": ensemble.get("fingerprint"),
                }
            else:
                ensemble = compute_ensemble(config, data, binning, ordered_trees, ensemble_key)
                audit["ensemble"] = {
                    "status": "recomputed",
                    "reason": self._ensemble_miss_reason(loaded, baseline_ensemble, product_fingerprint, config),
                    "key": ensemble_key,
                    "fingerprint": ensemble.get("fingerprint"),
                }
            timings["ensemble_ms"] = (time.perf_counter() - ensemble_started) * 1000.0
            artifacts["ensemble"] = ensemble

            contribution_key = contributions_cache_key(ensemble, ordered_trees)
            baseline_contributions = loaded.stages.get("contributions") if loaded is not None else None
            contributions_started = time.perf_counter()
            if baseline_contributions is not None and baseline_contributions.get("key") == contribution_key:
                contributions = baseline_contributions
                audit["contributions"] = {
                    "status": "reused",
                    "reason": "matching-content-key",
                    "key": contribution_key,
                    "fingerprint": contributions.get("fingerprint"),
                }
            else:
                contributions = compute_contributions(config, data, binning, ordered_trees, ensemble, contribution_key)
                audit["contributions"] = {
                    "status": "recomputed",
                    "reason": (
                        "ensemble_changed"
                        if baseline_contributions is None or baseline_contributions.get("ensemble_fingerprint") != ensemble.get("fingerprint")
                        else "key_mismatch"
                    ),
                    "key": contribution_key,
                    "fingerprint": contributions.get("fingerprint"),
                }
            timings["contributions_ms"] = (time.perf_counter() - contributions_started) * 1000.0
            artifacts["contributions"] = contributions

            commit_snapshot(include_ensemble=True, include_contributions=True)
            audit["concurrency"] = {
                "configured_workers": config.thread_pool_size,
                "actual_workers": max(1, min(config.thread_pool_size, max(1, len(missing)))),
                "peak_inflight": peak_inflight,
                "new_trees_completed": new_completed,
            }
            return self._finish(
                status="completed",
                exit_code=0,
                failures=sorted_failures(failures),
                artifacts=artifacts,
                audit=self._finalize_audit(audit),
                config=config,
                data=data,
                timings=timings,
                loaded_recovery=loaded_recovery,
                committed_snapshot=last_snapshot,
                label=label,
                report_name=report_name,
                started=started,
            )
        except BusinessValidationError as exception:
            failures.extend(exception.failures)
        except SnapshotVersionChanged as exception:
            failures.append(
                BusinessFailure(
                    code="SNAPSHOT_VERSION_CHANGED",
                    field="snapshot.CURRENT",
                    message=str(exception),
                    stage="snapshot",
                )
            )
        except OSError as exception:
            failures.append(
                BusinessFailure(
                    code="RUNTIME_IO_FAILURE",
                    field="filesystem",
                    message=str(exception),
                    stage="runtime",
                    exception_type=type(exception).__name__,
                )
            )
        except Exception as exception:
            failures.append(
                BusinessFailure(
                    code="RUNTIME_UNEXPECTED_FAILURE",
                    field="engine",
                    message=str(exception),
                    stage="runtime",
                    exception_type=type(exception).__name__,
                )
            )

        return self._finish(
            status="failed",
            exit_code=2 if not any(item.get("stage") == "snapshot" for item in sorted_failures(failures)) else 3,
            failures=sorted_failures(failures),
            artifacts=artifacts,
            audit=self._finalize_audit(audit),
            config=config,
            data=data,
            timings=timings,
            loaded_recovery=loaded_recovery,
            committed_snapshot=last_snapshot,
            label=label,
            report_name=report_name,
            started=started,
        )

    def _ensure_tmp_dir(self) -> None:
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.tmp_dir, prefix=".probe-", delete=True):
            pass

    def _empty_audit(self) -> dict[str, Any]:
        return {
            "binning": None,
            "trees": [],
            "ensemble": None,
            "contributions": None,
            "reused": [],
            "recomputed": [],
            "deferred": [],
        }

    def _finalize_audit(self, audit: dict[str, Any]) -> dict[str, Any]:
        reused = []
        recomputed = []
        deferred = []
        if audit.get("binning"):
            target = reused if audit["binning"]["status"] == "reused" else recomputed
            target.append("binning")
        for tree in sorted(audit.get("trees", []), key=lambda item: int(item["tree_index"])):
            if tree["status"] == "reused":
                reused.append(f"tree:{tree['tree_index']}")
            elif tree["status"] == "recomputed":
                recomputed.append(f"tree:{tree['tree_index']}")
            else:
                deferred.append(f"tree:{tree['tree_index']}")
        for stage in ("ensemble", "contributions"):
            if audit.get(stage):
                target = reused if audit[stage]["status"] == "reused" else recomputed
                target.append(stage)
        audit["reused"] = reused
        audit["recomputed"] = recomputed
        audit["deferred"] = deferred
        return audit

    def _binning_miss_reason(
        self,
        config: ExperimentConfig,
        data: PreparedData,
        baseline_binning: dict[str, Any] | None,
        loaded: Snapshot | None,
    ) -> str:
        if loaded is None or baseline_binning is None:
            return "empty-state"
        if loaded.manifest.get("data_fingerprint") != data.data_fingerprint:
            return "data_changed"
        if baseline_binning.get("parameters", {}).get("bin_count") != config.bin_count:
            return "bin_count_changed"
        return "content_key_mismatch"

    def _tree_miss_reason(
        self,
        tree_index: int,
        data: PreparedData,
        binning: dict[str, Any],
        baseline_binning: dict[str, Any] | None,
        baseline_tree: dict[str, Any] | None,
        loaded: Snapshot | None,
    ) -> str:
        if loaded is None:
            return "empty-state"
        if baseline_tree is None:
            if tree_index > max(loaded.trees, default=-1):
                return "tree_count_extended"
            return "tree_not_in_snapshot"
        if loaded.manifest.get("data_fingerprint") != data.data_fingerprint:
            return "data_changed"
        if baseline_binning is None or baseline_binning.get("fingerprint") != binning.get("fingerprint"):
            return "binning_changed"
        return "tree_parameters_or_fault_changed"

    def _ensemble_miss_reason(
        self,
        loaded: Snapshot | None,
        baseline_ensemble: dict[str, Any] | None,
        product_fingerprint: str,
        config: ExperimentConfig,
    ) -> str:
        if loaded is None or baseline_ensemble is None:
            return "empty-state"
        if baseline_ensemble.get("tree_product_fingerprint") != product_fingerprint:
            return "tree_product_changed"
        if baseline_ensemble.get("parameters", {}).get("learning_rate") != config.learning_rate:
            return "learning_rate_changed"
        return "content_key_mismatch"

    def _unexpected_tree_failure(self, tree_index: int, key: str, exception: BaseException) -> dict[str, Any]:
        failure = BusinessFailure(
            code="TREE_UNEXPECTED_FAILURE",
            field=f"tree.{tree_index}",
            message=str(exception),
            stage="tree",
            tree_index=tree_index,
            exception_type=type(exception).__name__,
        ).as_dict()
        artifact = {
            "schema": "tree/v1",
            "tree_index": tree_index,
            "key": key,
            "status": "failed",
            "failure": failure,
            "structure": None,
            "structure_summary": {
                "sample_count": 0,
                "node_count": 0,
                "leaf_count": 0,
                "split_count": 0,
                "max_depth": 0,
                "split_features": [],
            },
            "sampled_row_ids": [],
            "sampled_features": [],
            "structure_fingerprint": None,
            "artifact_fingerprint": None,
        }
        artifact["artifact_fingerprint"] = fingerprint({"schema": "tree-artifact/v1", "body": artifact})
        return artifact

    def _finish(
        self,
        *,
        status: str,
        exit_code: int,
        failures: list[dict[str, Any]],
        artifacts: dict[str, Any],
        audit: dict[str, Any],
        config: ExperimentConfig | None,
        data: PreparedData | None,
        timings: dict[str, float],
        loaded_recovery: dict[str, Any] | None,
        committed_snapshot: Snapshot | None,
        label: str,
        report_name: str | None,
        started: float,
    ) -> RunResult:
        timings = {key: round(float(value), 3) for key, value in sorted(timings.items())}
        timings["total_ms"] = round((time.perf_counter() - started) * 1000.0, 3)
        committed_recovery = committed_snapshot.recovery_info() if committed_snapshot is not None else None
        ensemble = artifacts.get("ensemble")
        contributions = artifacts.get("contributions")
        report = {
            "schema": "run-report/v1",
            "label": label,
            "status": status,
            "exit_code": exit_code,
            "config_hash": config.config_hash() if config is not None else None,
            "effective_config": config.as_dict() if config is not None else None,
            "data": data.summary() if data is not None else None,
            "timings_ms": timings,
            "cache_audit": audit,
            "reused": audit.get("reused", []),
            "recomputed": audit.get("recomputed", []),
            "failures": failures,
            "recovery": {
                "loaded": loaded_recovery,
                "committed": committed_recovery,
                "resumed": loaded_recovery is not None,
            },
            "recovery_point": committed_recovery or loaded_recovery,
            "metrics": ensemble.get("metrics") if ensemble is not None else None,
            "integration": {
                "metrics": ensemble.get("metrics") if ensemble is not None else None,
                "loss_trajectory": ensemble.get("loss_trajectory") if ensemble is not None else None,
                "applied_trees": ensemble.get("applied_trees") if ensemble is not None else None,
                "failed_trees": ensemble.get("failed_trees") if ensemble is not None else None,
                "ensemble_fingerprint": ensemble.get("fingerprint") if ensemble is not None else None,
                "contributions_fingerprint": contributions.get("fingerprint") if contributions is not None else None,
            },
        }
        report_path: str | None = None
        if report_name:
            target = Path(report_name)
            if not target.is_absolute():
                target = self.report_dir / target
            report_path = str(write_json_report(target, report))
        return RunResult(
            status=status,
            exit_code=exit_code,
            failures=failures,
            artifacts=artifacts,
            cache_audit=audit,
            report=report,
            report_path=report_path,
            effective_config=config,
            data_summary=data.summary() if data is not None else None,
        )
