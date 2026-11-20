from __future__ import annotations

import concurrent.futures
import copy
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .binning import bin_column
from .cache import ArtifactCache, CacheStats
from .config import (
    ValidatedConfig,
    validate_generated_table,
    validate_inputs,
)
from .errors import (
    E_COLUMN_BINNING_FAILED,
    E_COLUMN_WOE_FAILED,
    E_CONFIG_INVALID,
    E_EVALUATION_FAILED,
    E_MONOTONIC_FAILED,
    E_MONOTONIC_UNRESOLVED,
    E_SNAPSHOT_STALE,
    Failure,
    SnapshotError,
    sort_failures,
)
from .evaluation import evaluate
from .fingerprint import (
    binning_params_fingerprint,
    column_content_fingerprint,
    config_fingerprint,
    data_fingerprint,
    direction_priority_fingerprint,
    sha256_value,
    stage_cache_key,
)
from .monotonic import solve_monotonic
from .snapshot import SnapshotStore, STAGE_ORDER, atomic_write_json
from .synthetic import Dataset, generate_dataset
from .woe import compute_woe


ALGORITHM_VERSION = "1.0.0"
REPORT_SCHEMA_VERSION = 1


def _column_failure(column: str, stage: str, code: str, exc: BaseException) -> Failure:
    return Failure(
        column=column,
        stage=stage,
        code=code,
        exception_type=type(exc).__name__,
        message=str(exc) or type(exc).__name__,
    )


class ScorecardEngine:
    def __init__(
        self,
        experiment: dict[str, Any],
        schema: dict[str, Any],
        state_dir: str | Path,
        *,
        allow_revision: bool = False,
    ) -> None:
        self.experiment = experiment
        self.schema = schema
        self.state_dir = Path(state_dir)
        self.allow_revision = allow_revision
        self.store = SnapshotStore(self.state_dir, ALGORITHM_VERSION)
        self.cache = ArtifactCache(self.state_dir)
        self.stop_event = threading.Event()
        self.last_report: dict[str, Any] | None = None
        self._stats = CacheStats()
        self._failures: list[Failure] = []
        self._recovery: dict[str, Any] = {}
        self._invalidated_artifacts = 0
        self._invalidated_columns: list[str] = []
        self._previous_eval_key: str | None = None
        self._working_manifest: dict[str, Any] | None = None
        self._key_bundle: dict[str, dict[str, str]] = {}

    def request_stop(self) -> None:
        self.stop_event.set()

    def _maybe_failpoint(self, name: str) -> None:
        configured = os.environ.get("SCORECARD_FAILPOINT", "")
        if configured and configured == name:
            os._exit(137)

    def _task_delay(self) -> None:
        raw = os.environ.get("SCORECARD_TEST_TASK_DELAY_MS")
        if not raw:
            return
        try:
            milliseconds = float(raw)
        except ValueError:
            return
        if milliseconds > 0:
            time.sleep(milliseconds / 1000.0)

    def run(self, *, report_path: str | Path | None = None, allow_revision: bool | None = None) -> dict[str, Any]:
        self._stats = CacheStats()
        self._failures = []
        self._invalidated_artifacts = 0
        self._invalidated_columns = []
        self._previous_eval_key = None
        self._working_manifest = None
        self._key_bundle = {}
        revision_allowed = self.allow_revision if allow_revision is None else allow_revision

        validated, config_failures = validate_inputs(self.experiment, self.schema)
        if config_failures or validated is None:
            report = self._config_error_report(config_failures)
            self._write_report(report, report_path)
            return report

        try:
            dataset = generate_dataset(self.schema)
        except Exception as exc:  # malformed synthetic descriptions are business config errors
            failure = Failure(
                column="",
                stage="config",
                code=E_CONFIG_INVALID,
                exception_type=type(exc).__name__,
                message=f"synthetic data generation failed: {exc}",
            )
            report = self._config_error_report([failure])
            self._write_report(report, report_path)
            return report

        table_failures = validate_generated_table(validated, dataset.columns, dataset.target)
        if table_failures:
            report = self._config_error_report(table_failures)
            self._write_report(report, report_path)
            return report

        current_config_fp = config_fingerprint(self.experiment)
        feature_columns = {name: dataset.columns[name] for name in validated.features}
        current_data_fp = data_fingerprint(
            feature_columns,
            dataset.target,
            validated.target_name,
            self.schema,
        )

        try:
            previous_manifest = self.store.load_manifest()
        except SnapshotError as exc:
            report = self._stale_report(None, current_config_fp, current_data_fp, str(exc))
            self._write_report(report, report_path)
            return report

        progress = self.store.read_progress()
        if progress and (
            progress.get("config_fingerprint") != current_config_fp
            or progress.get("data_fingerprint") != current_data_fp
        ):
            progress = None

        if previous_manifest is not None and validated.expected_global_version is not None:
            if int(previous_manifest.get("global_version", -1)) != validated.expected_global_version:
                message = (
                    "expected snapshot version mismatch: "
                    f"expected {validated.expected_global_version}, "
                    f"found {previous_manifest.get('global_version')}"
                )
                report = self._stale_report(previous_manifest, current_config_fp, current_data_fp, message)
                self._write_report(report, report_path)
                return report
        if previous_manifest is None and validated.expected_global_version not in (None, 0):
            message = f"expected snapshot version {validated.expected_global_version}, but no committed snapshot exists"
            report = self._stale_report(None, current_config_fp, current_data_fp, message)
            self._write_report(report, report_path)
            return report

        identity_matches = bool(
            previous_manifest
            and previous_manifest.get("config_fingerprint") == current_config_fp
            and previous_manifest.get("data_fingerprint") == current_data_fp
        )
        if previous_manifest is not None and not identity_matches and not revision_allowed:
            message = (
                "snapshot input identity mismatch: "
                f"config {previous_manifest.get('config_fingerprint')} -> {current_config_fp}, "
                f"data {previous_manifest.get('data_fingerprint')} -> {current_data_fp}"
            )
            report = self._stale_report(previous_manifest, current_config_fp, current_data_fp, message)
            self._write_report(report, report_path)
            return report

        self._recovery = {
            "recovered": bool(identity_matches and (previous_manifest or {}).get("last_completed_stage")),
            "refused": False,
            "resume_version": int((previous_manifest or {}).get("global_version", 0)),
            "resumed_from_stage": (
                progress.get("stage")
                if progress
                else ((previous_manifest or {}).get("last_completed_stage") or "none")
            ),
            "progress_completed_columns": list((progress or {}).get("completed_columns", [])),
            "revision_started": False,
            "revision_of": None,
        }

        try:
            previous_keys = self._collect_previous_keys(previous_manifest, progress)
            key_bundle = self._compute_keys(validated, dataset)
            self._key_bundle = key_bundle
            self._compute_invalidations(previous_keys, key_bundle)
            self._previous_eval_key = previous_keys.get("evaluation", {}).get("__metrics__")
        except SnapshotError as exc:
            report = self._stale_report(previous_manifest, current_config_fp, current_data_fp, str(exc))
            self._write_report(report, report_path)
            return report

        if previous_manifest is None:
            manifest = self.store.blank_manifest(current_config_fp, current_data_fp)
        elif not identity_matches:
            manifest = self.store.begin_revision(previous_manifest, current_config_fp, current_data_fp)
            self._recovery.update(
                {
                    "recovered": False,
                    "revision_started": True,
                    "revision_of": int(previous_manifest.get("global_version", 0)),
                    "resumed_from_stage": "none",
                }
            )
        else:
            manifest = copy.deepcopy(previous_manifest)
        self._attach_manifest(manifest)

        stage_payloads: dict[str, dict[str, Any]] = {}
        reused_load = self._reuse_load_stage(manifest, dataset, current_data_fp)
        if reused_load is not None:
            load_payload = reused_load
            dataset = self._dataset_from_payload(load_payload)
            stage_payloads["load"] = load_payload
        else:
            load_payload = {
                "stage": "load",
                "rows": dataset.rows,
                "columns": {name: dataset.columns[name] for name in validated.features},
                "target": dataset.target,
                "target_name": dataset.target_name,
                "positive_value": dataset.positive_value,
                "negative_value": dataset.negative_value,
                "data_fingerprint": current_data_fp,
                "missing_counts": {
                    name: sum(1 for value in dataset.columns[name] if value is None)
                    for name in validated.features
                },
            }
            manifest = self.store.commit_stage(
                manifest,
                "load",
                load_payload,
                ["__data__"],
                {},
                [],
            )
            stage_payloads["load"] = load_payload
            self._maybe_failpoint("after_load_stage")

        if self.stop_event.is_set():
            report = self._build_report(
                "interrupted",
                validated,
                manifest,
                stage_payloads,
                current_config_fp,
                current_data_fp,
                interruption_stage="load",
            )
            self._write_report(report, report_path)
            return report

        binning_keys = key_bundle["binning"]
        woe_keys = key_bundle["woe"]
        monotonic_keys = key_bundle["monotonic"]

        binning_results, _, binning_interrupted = self._run_or_reuse_binning(
            manifest, validated, dataset, binning_keys
        )
        if binning_interrupted:
            stage_payloads["binning"] = {"stage": "binning", "columns": binning_results}
            report = self._build_report(
                "interrupted",
                validated,
                manifest,
                stage_payloads,
                current_config_fp,
                current_data_fp,
                interruption_stage="binning",
            )
            self._write_report(report, report_path)
            return report
        manifest = self._current_manifest_after_stage(manifest)
        if "binning" in manifest.get("stage_versions", {}):
            stage_payloads["binning"] = self._load_committed_stage_payload(manifest, "binning", binning_results)

        woe_expected = {
            column: woe_keys[column]
            for column in sorted(binning_results)
            if column in woe_keys
        }
        woe_results, _, woe_interrupted = self._run_or_reuse_woe(
            manifest, validated, dataset, binning_results, woe_expected
        )
        if woe_interrupted:
            stage_payloads["woe"] = {"stage": "woe", "columns": woe_results}
            report = self._build_report(
                "interrupted",
                validated,
                manifest,
                stage_payloads,
                current_config_fp,
                current_data_fp,
                interruption_stage="woe",
            )
            self._write_report(report, report_path)
            return report
        manifest = self._current_manifest_after_stage(manifest)
        if "woe" in manifest.get("stage_versions", {}):
            stage_payloads["woe"] = self._load_committed_stage_payload(manifest, "woe", woe_results)

        monotonic_expected = {
            column: monotonic_keys[column]
            for column in sorted(woe_results)
            if column in monotonic_keys
        }
        monotonic_results, monotonic_keys_used, monotonic_interrupted = self._run_or_reuse_monotonic(
            manifest, validated, dataset, binning_results, woe_results, monotonic_expected
        )
        if monotonic_interrupted:
            stage_payloads["monotonic"] = {"stage": "monotonic", "columns": monotonic_results}
            report = self._build_report(
                "interrupted",
                validated,
                manifest,
                stage_payloads,
                current_config_fp,
                current_data_fp,
                interruption_stage="monotonic",
            )
            self._write_report(report, report_path)
            return report
        manifest = self._current_manifest_after_stage(manifest)
        if "monotonic" in manifest.get("stage_versions", {}):
            stage_payloads["monotonic"] = self._load_committed_stage_payload(
                manifest, "monotonic", monotonic_results
            )

        eligible = {
            column: artifact
            for column, artifact in monotonic_results.items()
            if artifact.get("status") != "unresolved"
        }
        final_keys = {
            column: monotonic_keys_used[column]
            for column in sorted(eligible)
            if column in monotonic_keys_used
        }
        evaluation_key = self._evaluation_key(validated, current_data_fp, final_keys)
        if self._previous_eval_key is not None and self._previous_eval_key != evaluation_key:
            self._invalidated_artifacts += 1
        evaluation_metrics, evaluation_interrupted = self._run_or_reuse_evaluation(
            manifest, validated, dataset, eligible, evaluation_key
        )
        if evaluation_interrupted:
            stage_payloads["evaluation"] = {
                "stage": "evaluation",
                "metrics": evaluation_metrics,
            }
            report = self._build_report(
                "interrupted",
                validated,
                manifest,
                stage_payloads,
                current_config_fp,
                current_data_fp,
                interruption_stage="evaluation",
            )
            self._write_report(report, report_path)
            return report
        manifest = self._current_manifest_after_stage(manifest)
        if "evaluation" in manifest.get("stage_versions", {}):
            committed = self.store.load_stage(manifest, "evaluation")
            if committed is not None:
                evaluation_metrics = committed["payload"].get("metrics")
        stage_payloads["evaluation"] = {"stage": "evaluation", "metrics": evaluation_metrics}

        all_stages_complete = all(stage in manifest.get("stage_versions", {}) for stage in STAGE_ORDER)
        if self.stop_event.is_set() and not all_stages_complete:
            status = "interrupted"
        elif self._failures:
            status = "partial"
        else:
            status = "ok"
        report = self._build_report(
            status,
            validated,
            manifest,
            stage_payloads,
            current_config_fp,
            current_data_fp,
        )
        self._write_report(report, report_path)
        return report

    def _current_manifest_after_stage(self, fallback: dict[str, Any]) -> dict[str, Any]:
        # run/reuse helpers store the latest manifest on this attribute.
        return getattr(self, "_working_manifest", fallback)

    def _load_committed_stage_payload(
        self,
        manifest: dict[str, Any],
        stage: str,
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        document = self.store.load_stage(manifest, stage)
        if document is None:
            return {"stage": stage, "columns": fallback}
        return document["payload"]

    def _compute_keys(self, config: ValidatedConfig, dataset: Dataset) -> dict[str, dict[str, str]]:
        content: dict[str, str] = {}
        bin_params: dict[str, str] = {}
        direction_keys: dict[str, str] = {}
        binning: dict[str, str] = {}
        woe: dict[str, str] = {}
        monotonic: dict[str, str] = {}
        for name in config.features:
            params = config.effective_columns[name]
            content_key = column_content_fingerprint(
                name,
                params.column_type,
                dataset.columns[name],
                dataset.target,
                config.target_name,
                config.positive_value,
            )
            bin_params_key = binning_params_fingerprint(params.max_bins, params.min_bin_ratio, config.prebins)
            bin_key = stage_cache_key("binning", content_key, bin_params_key)
            woe_key = stage_cache_key("woe", bin_key, sha256_value({"smoothing": config.smoothing}))
            if params.column_type == "numeric" and params.direction != "none":
                direction_key = direction_priority_fingerprint(
                    params.direction, params.conflict_priority, params.max_rounds
                )
            else:
                direction_key = direction_priority_fingerprint("none", [], 0)
            mon_key = stage_cache_key("monotonic", woe_key, direction_key)
            content[name] = content_key
            bin_params[name] = bin_params_key
            direction_keys[name] = direction_key
            binning[name] = bin_key
            woe[name] = woe_key
            monotonic[name] = mon_key
        return {
            "content": content,
            "binning_params": bin_params,
            "direction_priority": direction_keys,
            "binning": binning,
            "woe": woe,
            "monotonic": monotonic,
        }

    def _collect_previous_keys(
        self,
        manifest: dict[str, Any] | None,
        progress: dict[str, Any] | None,
    ) -> dict[str, dict[str, str]]:
        result: dict[str, dict[str, str]] = {}
        if manifest is None:
            return result
        for stage in STAGE_ORDER:
            document = self.store.load_stage(manifest, stage)
            if document is not None:
                result[stage] = dict(document.get("keys", {}))
        if progress:
            stage = str(progress.get("stage", ""))
            if stage and stage not in result:
                result[stage] = dict(progress.get("keys", {}))
        return result

    def _compute_invalidations(
        self,
        previous_keys: dict[str, dict[str, str]],
        key_bundle: dict[str, dict[str, str]],
    ) -> None:
        stage_keys = {
            "binning": key_bundle["binning"],
            "woe": key_bundle["woe"],
            "monotonic": key_bundle["monotonic"],
        }
        count = 0
        columns: set[str] = set()
        for stage, expected in stage_keys.items():
            old = previous_keys.get(stage, {})
            for column, old_key in old.items():
                if not column.startswith("__") and expected.get(column) != old_key:
                    count += 1
                    columns.add(column)
        self._invalidated_artifacts = count
        self._invalidated_columns = sorted(columns)

    def _reuse_load_stage(
        self,
        manifest: dict[str, Any],
        dataset: Dataset,
        data_fp: str,
    ) -> dict[str, Any] | None:
        if "load" not in manifest.get("stage_versions", {}):
            return None
        document = self.store.load_stage(manifest, "load")
        if document is None or document.get("failures"):
            return None
        payload = document.get("payload", {})
        if payload.get("data_fingerprint") != data_fp:
            raise SnapshotError("load snapshot data fingerprint differs from generated input")
        if payload.get("rows") != dataset.rows:
            raise SnapshotError("load snapshot row count differs from generated input")
        return payload

    def _dataset_from_payload(self, payload: dict[str, Any]) -> Dataset:
        columns = {name: list(values) for name, values in payload["columns"].items()}
        target = list(payload["target"])
        target_name = str(payload["target_name"])
        columns[target_name] = list(target)
        return Dataset(
            columns=columns,
            target=target,
            target_name=target_name,
            positive_value=payload["positive_value"],
            negative_value=payload["negative_value"],
        )

    def _reuse_stage(
        self,
        manifest: dict[str, Any],
        stage: str,
        expected_keys: dict[str, str],
    ) -> dict[str, Any] | None:
        if stage not in manifest.get("stage_versions", {}):
            return None
        document = self.store.load_stage(manifest, stage)
        if document is None or document.get("failures"):
            return None
        if document.get("keys", {}) != {name: expected_keys[name] for name in sorted(expected_keys)}:
            return None
        self._stats.record_stage_reuse(stage, list(document.get("keys", {})))
        return document

    def _attach_manifest(self, manifest: dict[str, Any]) -> None:
        self._working_manifest = manifest

    def _run_pool(
        self,
        stage: str,
        columns: list[str],
        keys: dict[str, str],
        compute: Callable[[str], dict[str, Any]],
        code: str,
        manifest: dict[str, Any],
    ) -> tuple[dict[str, dict[str, Any]], bool]:
        results: dict[str, dict[str, Any]] = {}
        stage_failures: list[Failure] = []
        ordered_columns = sorted(columns)
        submitted_order: dict[concurrent.futures.Future[Any], str] = {}
        missing = [column for column in ordered_columns if column not in keys]
        if missing:
            raise ValueError(f"missing cache keys for stage {stage}: {missing}")

        if self.stop_event.is_set():
            return results, True

        max_workers = getattr(self, "_max_workers", 1)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(
                    self._cached_column_task,
                    stage,
                    column,
                    f"{stage}:{column}",
                    keys[column],
                    lambda c=column: compute(c),
                    code,
                ): column
                for column in ordered_columns
            }
            for future, column in futures.items():
                submitted_order[future] = column
            pending = set(futures)
            while pending:
                done, pending = concurrent.futures.wait(
                    pending,
                    return_when=concurrent.futures.FIRST_COMPLETED,
                )
                for future in done:
                    column = submitted_order[future]
                    if future.cancelled():
                        continue
                    try:
                        payload, failure = future.result()
                    except Exception as exc:
                        payload, failure = None, _column_failure(column, stage, code, exc)
                    if failure is not None:
                        stage_failures.append(failure)
                    elif payload is not None:
                        results[column] = payload
                    self.store.write_progress(
                        stage=stage,
                        config_fingerprint=manifest["config_fingerprint"],
                        data_fingerprint=manifest["data_fingerprint"],
                        base_global_version=int(manifest.get("global_version", 0)),
                        completed_columns=sorted(results),
                        keys={name: keys[name] for name in results},
                        failures=[item.to_dict() for item in sort_failures(self._failures + stage_failures)],
                    )
                if self.stop_event.is_set() and pending:
                    for future in list(pending):
                        future.cancel()

        self._failures.extend(stage_failures)
        interrupted = self.stop_event.is_set() and len(results) + len(stage_failures) < len(ordered_columns)
        return results, interrupted

    def _cached_column_task(
        self,
        stage: str,
        column: str,
        sort_key: str,
        key: str,
        compute: Callable[[], dict[str, Any]],
        code: str,
    ) -> tuple[dict[str, Any] | None, Failure | None]:
        del sort_key  # The stable stage:column key is carried by each dispatched task.
        self._task_delay()
        cached = self.cache.load(stage, key)
        if cached is not None:
            self._stats.record(stage, column, True)
            return cached, None
        self._stats.record(stage, column, False)
        try:
            payload = compute()
            self.cache.store(stage, key, payload)
        except Exception as exc:
            return None, _column_failure(column, stage, code, exc)
        self._maybe_failpoint(f"after_column_cache:{stage}:{column}")
        self._maybe_failpoint("after_column_cache")
        return payload, None

    def _run_or_reuse_binning(
        self,
        manifest: dict[str, Any],
        config: ValidatedConfig,
        dataset: Dataset,
        keys: dict[str, str],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str], bool]:
        self._attach_manifest(manifest)
        reused = self._reuse_stage(manifest, "binning", keys)
        if reused is not None:
            return reused["payload"].get("columns", {}), dict(reused.get("keys", {})), False
        self._max_workers = config.max_workers

        def compute(column: str) -> dict[str, Any]:
            params = config.effective_columns[column]
            return bin_column(
                column,
                params.column_type,
                dataset.columns[column],
                dataset.target,
                config.positive_value,
                params,
                config.prebins,
            )

        results, interrupted = self._run_pool(
            "binning",
            config.features,
            keys,
            compute,
            E_COLUMN_BINNING_FAILED,
            manifest,
        )
        if interrupted:
            return results, {column: keys[column] for column in results}, True
        payload = {"stage": "binning", "columns": {name: results[name] for name in sorted(results)}}
        completed_keys = {name: keys[name] for name in sorted(results)}
        manifest = self.store.commit_stage(
            manifest,
            "binning",
            payload,
            sorted(results),
            completed_keys,
            [item.to_dict() for item in sort_failures(self._failures)],
        )
        self._attach_manifest(manifest)
        self._maybe_failpoint("after_binning_stage")
        return results, completed_keys, False

    def _run_or_reuse_woe(
        self,
        manifest: dict[str, Any],
        config: ValidatedConfig,
        dataset: Dataset,
        binning_results: dict[str, dict[str, Any]],
        keys: dict[str, str],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str], bool]:
        self._attach_manifest(manifest)
        reused = self._reuse_stage(manifest, "woe", keys)
        if reused is not None:
            return reused["payload"].get("columns", {}), dict(reused.get("keys", {})), False
        self._max_workers = config.max_workers

        def compute(column: str) -> dict[str, Any]:
            binning = binning_results[column]
            result = compute_woe(column, binning["type"], binning["bins"], config.smoothing)
            result["category_to_group"] = dict(binning.get("category_to_group", {}))
            result["missing_index"] = binning.get("missing_index")
            return result

        results, interrupted = self._run_pool(
            "woe",
            sorted(binning_results),
            keys,
            compute,
            E_COLUMN_WOE_FAILED,
            manifest,
        )
        if interrupted:
            return results, {column: keys[column] for column in results}, True
        payload = {"stage": "woe", "columns": {name: results[name] for name in sorted(results)}}
        completed_keys = {name: keys[name] for name in sorted(results)}
        manifest = self.store.commit_stage(
            manifest,
            "woe",
            payload,
            sorted(results),
            completed_keys,
            [item.to_dict() for item in sort_failures(self._failures)],
        )
        self._attach_manifest(manifest)
        self._maybe_failpoint("after_woe_stage")
        return results, completed_keys, False

    def _run_or_reuse_monotonic(
        self,
        manifest: dict[str, Any],
        config: ValidatedConfig,
        dataset: Dataset,
        binning_results: dict[str, dict[str, Any]],
        woe_results: dict[str, dict[str, Any]],
        keys: dict[str, str],
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str], bool]:
        del dataset
        self._attach_manifest(manifest)
        reused = self._reuse_stage(manifest, "monotonic", keys)
        if reused is not None:
            return reused["payload"].get("columns", {}), dict(reused.get("keys", {})), False
        self._max_workers = config.max_workers

        def compute(column: str) -> dict[str, Any]:
            params = config.effective_columns[column]
            binning = binning_results[column]
            base = woe_results[column]
            return solve_monotonic(
                column,
                params.column_type,
                base["bins"],
                params.direction,
                params.conflict_priority,
                params.max_rounds,
                config.smoothing,
                binning.get("category_to_group", {}),
            )

        results, interrupted = self._run_pool(
            "monotonic",
            sorted(woe_results),
            keys,
            compute,
            E_MONOTONIC_FAILED,
            manifest,
        )
        for column in sorted(results):
            if results[column].get("status") == "unresolved":
                self._failures.append(
                    Failure(
                        column=column,
                        stage="monotonic",
                        code=E_MONOTONIC_UNRESOLVED,
                        exception_type=None,
                        message=(
                            "monotonic constraints remain unresolved after "
                            f"{results[column].get('rounds', 0)} merge rounds"
                        ),
                    )
                )
        self._failures = sort_failures(self._failures)
        if interrupted:
            return results, {column: keys[column] for column in results}, True
        payload = {"stage": "monotonic", "columns": {name: results[name] for name in sorted(results)}}
        completed_keys = {name: keys[name] for name in sorted(results)}
        manifest = self.store.commit_stage(
            manifest,
            "monotonic",
            payload,
            sorted(results),
            completed_keys,
            [item.to_dict() for item in self._failures],
        )
        self._attach_manifest(manifest)
        self._maybe_failpoint("after_monotonic_stage")
        return results, completed_keys, False

    def _evaluation_key(
        self,
        config: ValidatedConfig,
        data_fp: str,
        final_keys: dict[str, str],
    ) -> str:
        metadata = sha256_value(
            {
                "folds": config.folds,
                "fold_seed": config.fold_seed,
                "positive_value": config.positive_value,
                "target_name": config.target_name,
                "columns": {name: final_keys[name] for name in sorted(final_keys)},
            }
        )
        return stage_cache_key("evaluation", data_fp, metadata)

    def _run_or_reuse_evaluation(
        self,
        manifest: dict[str, Any],
        config: ValidatedConfig,
        dataset: Dataset,
        eligible: dict[str, dict[str, Any]],
        key: str,
    ) -> tuple[dict[str, Any] | None, bool]:
        self._attach_manifest(manifest)
        expected_keys = {"__metrics__": key}
        reused = self._reuse_stage(manifest, "evaluation", expected_keys)
        if reused is not None:
            return reused["payload"].get("metrics"), False
        if self.stop_event.is_set():
            return None, True
        self._max_workers = 1

        def compute() -> dict[str, Any]:
            return evaluate(
                dataset.columns,
                dataset.target,
                config.positive_value,
                eligible,
                config.folds,
                config.fold_seed,
            )

        cached = self.cache.load("evaluation", key)
        if cached is not None:
            self._stats.record("evaluation", "__metrics__", True)
            metrics = cached
        else:
            self._stats.record("evaluation", "__metrics__", False)
            try:
                metrics = compute()
                self.cache.store("evaluation", key, metrics)
            except Exception as exc:
                self._failures.append(_column_failure("__metrics__", "evaluation", E_EVALUATION_FAILED, exc))
                return None, False
            self._maybe_failpoint("after_column_cache:evaluation:__metrics__")
            self._maybe_failpoint("after_column_cache")

        payload = {"stage": "evaluation", "metrics": metrics}
        manifest = self.store.commit_stage(
            manifest,
            "evaluation",
            payload,
            ["__metrics__"],
            expected_keys,
            [item.to_dict() for item in sort_failures(self._failures)],
        )
        self._attach_manifest(manifest)
        self._maybe_failpoint("after_evaluation_stage")
        return metrics, False

    def _config_error_report(self, failures: list[Failure]) -> dict[str, Any]:
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "status": "error",
            "error": "config_invalid",
            "business_codes": [E_CONFIG_INVALID],
            "snapshot_version": 0,
            "recovery_point_version": 0,
            "input": {
                "config_fingerprint": sha256_value({"kind": "raw_experiment", "value": self.experiment}),
                "data_fingerprint": sha256_value({"kind": "raw_schema", "value": self.schema}),
            },
            "snapshot": {
                "version": 0,
                "recovery_point_version": 0,
                "last_completed_stage": None,
            },
            "recovery": {
                "recovered": False,
                "refused": True,
                "resume_version": 0,
                "resumed_from_stage": "none",
            },
            "columns": [],
            "evaluation": None,
            "cache": {
                "hits": 0,
                "misses": 0,
                "invalidated_artifacts": 0,
                "invalidated_columns": [],
                "by_column": {},
            },
            "failures": [item.to_dict() for item in sort_failures(failures)],
        }

    def _stale_report(
        self,
        manifest: dict[str, Any] | None,
        config_fp: str,
        data_fp: str,
        message: str,
    ) -> dict[str, Any]:
        version = int((manifest or {}).get("global_version", 0))
        failure = Failure(
            column="",
            stage="snapshot",
            code=E_SNAPSHOT_STALE,
            exception_type=None,
            message=message,
        )
        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "status": "error",
            "error": "snapshot_stale",
            "business_codes": [E_SNAPSHOT_STALE],
            "snapshot_version": version,
            "recovery_point_version": version,
            "input": {
                "config_fingerprint": config_fp,
                "data_fingerprint": data_fp,
            },
            "snapshot": {
                "version": version,
                "recovery_point_version": version,
                "last_completed_stage": (manifest or {}).get("last_completed_stage"),
            },
            "recovery": {
                "recovered": False,
                "refused": True,
                "resume_version": version,
                "resumed_from_stage": (manifest or {}).get("last_completed_stage") or "none",
            },
            "columns": [],
            "evaluation": None,
            "cache": {
                "hits": 0,
                "misses": 0,
                "invalidated_artifacts": 0,
                "invalidated_columns": [],
                "by_column": {},
            },
            "failures": [failure.to_dict()],
        }

    def _stage_columns_payload(
        self,
        stage: str,
        stage_payloads: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        return dict(stage_payloads.get(stage, {}).get("columns", {}))

    def _build_report(
        self,
        status: str,
        config: ValidatedConfig,
        manifest: dict[str, Any],
        stage_payloads: dict[str, dict[str, Any]],
        config_fp: str,
        data_fp: str,
        *,
        interruption_stage: str | None = None,
    ) -> dict[str, Any]:
        binning = self._stage_columns_payload("binning", stage_payloads)
        woe = self._stage_columns_payload("woe", stage_payloads)
        monotonic = self._stage_columns_payload("monotonic", stage_payloads)
        cache_report = self._stats.to_dict()
        cache_report["hit_count"] = cache_report["hits"]
        cache_report["miss_count"] = cache_report["misses"]
        cache_report["invalidated_artifacts"] = self._invalidated_artifacts
        cache_report["invalidated_columns"] = self._invalidated_columns
        cache_report["invalidated_artifact_count"] = self._invalidated_artifacts
        cache_report["invalidated_column_count"] = len(self._invalidated_columns)

        columns_report: list[dict[str, Any]] = []
        for name in sorted(binning):
            bin_artifact = binning[name]
            woe_artifact = woe.get(name)
            final_artifact = monotonic.get(name, woe_artifact)
            if final_artifact is None:
                bins = bin_artifact.get("bins", [])
                iv = None
                corrections = []
                final_status = "binning_only"
                direction = config.effective_columns[name].direction
            else:
                bins = final_artifact.get("bins", [])
                iv = final_artifact.get("iv")
                corrections = final_artifact.get("corrections", [])
                final_status = final_artifact.get("status", "ok")
                direction = final_artifact.get("direction", config.effective_columns[name].direction)
                if (
                    woe_artifact is not None
                    and name not in monotonic
                    and ("monotonic" in stage_payloads or "monotonic" in manifest.get("stage_versions", {}))
                ):
                    final_status = "monotonic_failed"
            columns_report.append(
                {
                    "name": name,
                    "type": bin_artifact.get("type"),
                    "direction": direction,
                    "status": final_status,
                    "bin_count": len(bins),
                    "nonmissing_bin_count": sum(1 for item in bins if not item.get("is_missing")),
                    "missing_bin": any(bool(item.get("is_missing")) for item in bins),
                    "iv": iv,
                    "smoothed_bin_count": (final_artifact or {}).get("smoothed_bin_count"),
                    "monotonic_correction_count": len(corrections),
                    "corrections": corrections,
                    "bins": bins,
                    "category_to_group": (final_artifact or bin_artifact).get("category_to_group", {}),
                    "cache": cache_report["by_column"].get(name, {}),
                    "cache_keys": {
                        "content": self._key_bundle.get("content", {}).get(name),
                        "binning_params": self._key_bundle.get("binning_params", {}).get(name),
                        "direction_priority": self._key_bundle.get("direction_priority", {}).get(name),
                        "binning": self._key_bundle.get("binning", {}).get(name),
                        "woe": self._key_bundle.get("woe", {}).get(name),
                        "monotonic": self._key_bundle.get("monotonic", {}).get(name),
                    },
                }
            )

        evaluation_payload = stage_payloads.get("evaluation", {})
        evaluation_metrics = evaluation_payload.get("metrics") if evaluation_payload else None
        snapshot_version = int(manifest.get("global_version", 0))
        recovery = dict(self._recovery or {})
        recovery.setdefault("recovered", False)
        recovery.setdefault("refused", False)
        recovery.setdefault("resume_version", snapshot_version)
        recovery.setdefault("resumed_from_stage", "none")
        if interruption_stage:
            recovery["interruption_stage"] = interruption_stage
            recovery["interrupted"] = True

        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "status": status,
            "business_codes": sorted({item.code for item in self._failures}),
            "snapshot_version": snapshot_version,
            "recovery_point_version": int(recovery.get("resume_version", 0)),
            "input": {
                "config_fingerprint": config_fp,
                "data_fingerprint": data_fp,
                "rows": len(self._dataset_target_from_payload(stage_payloads) or []),
                "features": list(config.features),
            },
            "snapshot": {
                "version": snapshot_version,
                "recovery_point_version": int(recovery.get("resume_version", 0)),
                "last_completed_stage": manifest.get("last_completed_stage"),
                "stage_versions": {
                    stage: manifest["stage_versions"][stage]
                    for stage in sorted(manifest.get("stage_versions", {}))
                },
            },
            "recovery": recovery,
            "columns": columns_report,
            "evaluation": evaluation_metrics,
            "cache": cache_report,
            "failures": [item.to_dict() for item in sort_failures(self._failures)],
        }

    def _dataset_target_from_payload(self, stage_payloads: dict[str, dict[str, Any]]) -> list[Any] | None:
        load_payload = stage_payloads.get("load", {})
        target = load_payload.get("target")
        return target if isinstance(target, list) else None

    def _write_report(self, report: dict[str, Any], report_path: str | Path | None) -> None:
        self.last_report = report
        path = Path(report_path) if report_path is not None else self.state_dir / "last_report.json"
        atomic_write_json(path, report, indent=2)
