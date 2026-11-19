from __future__ import annotations

import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .base import BasePackage, load_base_package
from .canonical import (
    atomic_write_bytes,
    atomic_write_json,
    read_json,
    relative_path,
    sha256_file,
    stable_artifact_name,
)
from .compatibility import (
    ExistingAdapter,
    build_compatibility_table,
    compatibility_hash,
)
from .config import PeftConfig, load_config
from .errors import (
    E_ADAPTER_CONFIG,
    E_TASK_INVALID,
    E_TRAIN_FAILED,
    BusinessError,
    Failure,
    failure_sort_key,
)
from .merge import MergeResult, merge_adapters, write_merged_package
from .runtime import ShutdownController
from .samples import (
    InstructionSample,
    load_samples,
    sample_set_hash,
    samples_for_layer,
)
from .snapshot import SnapshotStore
from .tasks import TaskSpec, load_tasks, task_order_key, task_set_hash
from .training import save_adapter, train_adapter, validate_adapter_payload


REPORT_FORMAT = "peft-lora-report-v1"


@dataclass
class PipelineResult:
    exit_code: int
    status: str
    report: dict[str, Any]
    report_path: Path


def _failure_dict(failure: Failure) -> dict[str, Any]:
    return failure.as_dict()


def _sorted_failures(failures: list[Failure]) -> list[dict[str, Any]]:
    values = [failure.as_dict() for failure in failures]
    values.sort(key=failure_sort_key)
    return values


def _business_codes(failures: list[Failure], compatibility: list[dict[str, Any]]) -> list[str]:
    codes = {failure.code for failure in failures}
    codes.update(
        entry["business_code"]
        for entry in compatibility
        if isinstance(entry.get("business_code"), str) and entry["business_code"]
    )
    return sorted(codes)


class Pipeline:
    def __init__(
        self,
        samples_directory: str | Path,
        workspace: str | Path,
        *,
        base_directory: str | Path | None = None,
        controller: ShutdownController | None = None,
    ) -> None:
        self.samples_directory = Path(samples_directory).resolve()
        self.workspace = Path(workspace).resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.base_override = Path(base_directory).resolve() if base_directory else None
        self.controller = controller or ShutdownController()
        self.report_path = self.workspace / "report.json"
        self.current_stage: str | None = None

    def _base_path(self, config: PeftConfig) -> Path:
        if self.base_override is not None:
            return self.base_override
        configured = Path(config.base_dir)
        return configured if configured.is_absolute() else self.samples_directory / configured

    def _base_report_path(self, base: BasePackage) -> str:
        try:
            return base.path.relative_to(self.samples_directory).as_posix()
        except ValueError:
            return base.path.name

    def _write_report(self, report: dict[str, Any]) -> None:
        atomic_write_json(self.report_path, report)

    def _early_failure(
        self,
        failures: list[Failure],
        *,
        config: PeftConfig | None = None,
        base: BasePackage | None = None,
        tasks: list[TaskSpec] | None = None,
        compatibility: list[dict[str, Any]] | None = None,
    ) -> PipelineResult:
        compatibility = compatibility or []
        failures = sorted(failures, key=lambda item: failure_sort_key(item))
        codes = _business_codes(failures, compatibility)
        report = {
            "format": REPORT_FORMAT,
            "status": "failed",
            "exit_code": 1,
            "business_codes": codes,
            "failures": _sorted_failures(failures),
            "base": (
                {
                    "path": self._base_report_path(base),
                    "fingerprint": base.fingerprint,
                    "layer_fingerprints": dict(sorted(base.layer_fingerprints.items())),
                    "layer_count": len(base.layers),
                }
                if base is not None
                else None
            ),
            "config": (
                {"hash": config.hash, **config.as_dict()} if config is not None else None
            ),
            "tasks": [task.as_dict() for task in (tasks or [])],
            "compatibility": compatibility,
            "compatibility_hash": compatibility_hash(compatibility) if compatibility else None,
            "reuse": [],
            "retrain": [],
            "training_executed": [],
            "merge": None,
            "snapshot": None,
            "recovery": {"started_from_version": 0, "resumed": False},
        }
        self._write_report(report)
        return PipelineResult(1, "failed", report, self.report_path)

    def _graceful_stop(
        self,
        *,
        store: SnapshotStore,
        identity: Mapping[str, Any],
        load_payload: Mapping[str, Any],
        failures: list[Failure],
        compatibility: list[dict[str, Any]],
        training_index: Mapping[str, Any] | None,
    ) -> PipelineResult:
        stage_versions = {
            stage: int(reference["version"])
            for stage, reference in (store.current or {}).get("stages", {}).items()
            if isinstance(reference, dict) and isinstance(reference.get("version"), int)
        }
        report = {
            "format": REPORT_FORMAT,
            "status": "graceful_stop",
            "exit_code": 0,
            "business_codes": _business_codes(failures, compatibility),
            "failures": _sorted_failures(failures),
            "base": {
                "path": load_payload.get("base_path"),
                "fingerprint": identity["base_fingerprint"],
                "layer_fingerprints": load_payload.get("layer_fingerprints", {}),
            },
            "config": {
                "hash": identity["config_hash"],
                "snapshot_commit_interval": load_payload.get("snapshot_commit_interval"),
            },
            "tasks": load_payload.get("tasks", []),
            "compatibility": compatibility,
            "compatibility_hash": load_payload.get("compatibility_hash"),
            "reuse": load_payload.get("reuse", []),
            "retrain": load_payload.get("retrain", []),
            "training_executed": (
                training_index.get("training_executed", []) if training_index else []
            ),
            "merge": None,
            "snapshot": {
                "version": int((store.current or {}).get("global_version", 0)),
                "stage_versions": stage_versions,
                "config_hash": identity["config_hash"],
                "base_fingerprint": identity["base_fingerprint"],
                "compatibility_hash": load_payload.get("compatibility_hash"),
            },
            "recovery": store.recovery_info(identity),
            "shutdown": {
                "requested": True,
                "signal": self.controller.signal_name(),
                "completed_stage": self.current_stage,
            },
        }
        self._write_report(report)
        return PipelineResult(0, "graceful_stop", report, self.report_path)

    def _load_training_index(
        self,
        store: SnapshotStore,
        identity: Mapping[str, Any],
        tasks_by_name: Mapping[str, TaskSpec],
        base: BasePackage,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, Failure], set[str], bool]:
        index = store.current_training_adapters(identity)
        if not index:
            return {}, {}, set(), False
        stage_path = store.current_stage_path("training")
        payloads: dict[str, dict[str, Any]] = {}
        failures: dict[str, Failure] = {}
        executed: set[str] = set()
        for result in index.get("results", []):
            if not isinstance(result, dict):
                continue
            task_name = result.get("task_name")
            task = tasks_by_name.get(task_name)
            if task is None:
                continue
            if result.get("status") == "FAILURE":
                failure = result.get("failure")
                if isinstance(failure, dict):
                    failures[task_name] = Failure(
                        code=failure.get("code", E_TRAIN_FAILED),
                        message=failure.get("message", "previous training attempt failed"),
                        task_name=task_name,
                        target_layer=task.target_layer,
                        details=failure.get("details") or {},
                    )
                else:
                    failures[task_name] = Failure(
                        E_TRAIN_FAILED,
                        "previous training attempt failed",
                        task_name=task_name,
                        target_layer=task.target_layer,
                    )
                continue
            if result.get("status") != "SUCCESS" or stage_path is None:
                continue
            relative = result.get("file")
            if not isinstance(relative, str):
                continue
            try:
                adapter_path = (stage_path / relative).resolve()
                adapter_path.relative_to(stage_path.resolve())
                payload = read_json(adapter_path)
                layer = base.layers[task.target_layer]
                validate_adapter_payload(
                    payload,
                    expected_task=task,
                    expected_layer_shape=(
                        int(layer.weight.shape[0]),
                        int(layer.weight.shape[1]),
                    ),
                )
            except (OSError, ValueError, TypeError, BusinessError):
                continue
            payloads[task_name] = payload
            if result.get("executed", True):
                executed.add(task_name)
        return payloads, failures, executed, bool(index.get("complete"))

    def _commit_training_stage(
        self,
        store: SnapshotStore,
        identity: Mapping[str, Any],
        config: PeftConfig,
        compatibility: list[dict[str, Any]],
        expected_tasks: list[TaskSpec],
        successes: Mapping[str, dict[str, Any]],
        failures_by_task: Mapping[str, Failure],
        executed: set[str],
        complete: bool,
    ) -> dict[str, Any]:
        def populate(temp: Path, _version: int) -> dict[str, Any]:
            adapter_entries: list[dict[str, Any]] = []
            results: list[dict[str, Any]] = []
            for task in expected_tasks:
                if task.task_name in successes:
                    payload = successes[task.task_name]
                    filename = stable_artifact_name(task.task_name, task.target_layer, ".json")
                    relative = f"adapters/{filename}"
                    save_adapter(temp / relative, payload)
                    adapter_entries.append(
                        {
                            "task_name": task.task_name,
                            "target_layer": task.target_layer,
                            "file": relative,
                            "content_hash": payload["content_hash"],
                        }
                    )
                    results.append(
                        {
                            "task_name": task.task_name,
                            "target_layer": task.target_layer,
                            "status": "SUCCESS",
                            "file": relative,
                            "executed": task.task_name in executed,
                        }
                    )
                elif task.task_name in failures_by_task:
                    failure = failures_by_task[task.task_name]
                    results.append(
                        {
                            "task_name": task.task_name,
                            "target_layer": task.target_layer,
                            "status": "FAILURE",
                            "failure": failure.as_dict(),
                        }
                    )
            index = {
                "format": "peft-lora-training-index-v1",
                "complete": complete,
                "expected_task_names": [task.task_name for task in expected_tasks],
                "results": results,
                "adapters": adapter_entries,
                "training_executed": sorted(executed),
                **dict(identity),
            }
            atomic_write_json(temp / "index.json", index)
            self.controller.test_hold("training")
            return {"complete": complete, "adapter_count": len(adapter_entries)}

        reference = store.commit_stage(
            "training",
            identity,
            populate,
            compatibility_hash=compatibility_hash(compatibility),
            snapshot_commit_interval=config.snapshot_commit_interval,
            replace_stages=False,
            stage_metadata={"complete": complete},
        )
        if not complete:
            self.controller.test_hold_after("training")
        return reference

    def _run_training(
        self,
        *,
        store: SnapshotStore,
        identity: Mapping[str, Any],
        config: PeftConfig,
        base: BasePackage,
        tasks: list[TaskSpec],
        samples: list[InstructionSample],
        compatibility: list[dict[str, Any]],
        existing: Mapping[str, ExistingAdapter],
    ) -> tuple[
        dict[str, dict[str, Any]],
        dict[str, Failure],
        set[str],
        dict[str, Any] | None,
    ]:
        tasks_by_name = {task.task_name: task for task in tasks}
        prior_payloads, prior_failures, prior_executed, prior_complete = self._load_training_index(
            store, identity, tasks_by_name, base
        )
        compatibility_by_task = {entry["task_name"]: entry for entry in compatibility}
        to_train = [
            task
            for task in sorted(tasks, key=task_order_key)
            if compatibility_by_task[task.task_name]["action"] == "TRAIN"
        ]
        reused: dict[str, dict[str, Any]] = {}
        for task in tasks:
            entry = compatibility_by_task[task.task_name]
            if entry["action"] != "REUSE":
                continue
            candidate = existing.get(task.task_name)
            if candidate is None:
                prior_failures.setdefault(
                    task.task_name,
                    Failure(
                        E_TRAIN_FAILED,
                        "previously reusable adapter is no longer present",
                        task_name=task.task_name,
                        target_layer=task.target_layer,
                    ),
                )
                continue
            layer = base.layers[task.target_layer]
            try:
                validate_adapter_payload(
                    candidate.payload,
                    expected_task=task,
                    expected_layer_shape=(
                        int(layer.weight.shape[0]),
                        int(layer.weight.shape[1]),
                    ),
                )
            except BusinessError:
                prior_failures.setdefault(
                    task.task_name,
                    Failure(
                        E_ADAPTER_CONFIG,
                        "previously reusable adapter failed validation",
                        task_name=task.task_name,
                        target_layer=task.target_layer,
                    ),
                )
                continue
            reused[task.task_name] = candidate.payload

        succeeded = dict(prior_payloads)
        failures_by_task = dict(prior_failures)
        executed = set(prior_executed)
        required_names = {task.task_name for task in to_train}
        remaining = [task for task in to_train if task.task_name not in succeeded and task.task_name not in failures_by_task]
        if prior_complete and not remaining:
            index = store.current_training_adapters(identity)
            all_payloads = {**reused, **succeeded}
            return all_payloads, failures_by_task, executed, index

        if remaining:
            worker_count = max(1, min(config.concurrency, len(remaining)))
            with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="lora-train") as executor:
                futures: dict[str, Future[dict[str, Any]]] = {}
                for task in remaining:
                    layer = base.layers[task.target_layer]
                    layer_samples = samples_for_layer(samples, task.target_layer)
                    futures[task.task_name] = executor.submit(
                        train_adapter,
                        task,
                        layer,
                        layer_samples,
                        config,
                        base.fingerprint,
                        base.layer_fingerprints[task.target_layer],
                    )
                for processed, task in enumerate(remaining, start=1):
                    try:
                        payload = futures[task.task_name].result()
                        succeeded[task.task_name] = payload
                        executed.add(task.task_name)
                    except BusinessError as exc:
                        failures_by_task[task.task_name] = Failure(
                            exc.code,
                            exc.message,
                            task_name=task.task_name,
                            target_layer=task.target_layer,
                            details=exc.details,
                        )
                    except Exception as exc:  # pragma: no cover - defensive worker boundary
                        failures_by_task[task.task_name] = Failure(
                            E_TRAIN_FAILED,
                            f"unexpected training failure: {exc}",
                            task_name=task.task_name,
                            target_layer=task.target_layer,
                        )
                    completed = set(succeeded) | set(failures_by_task)
                    all_done = required_names.issubset(completed)
                    checkpoint_due = (
                        processed % config.snapshot_commit_interval == 0
                        or processed == len(remaining)
                    )
                    if checkpoint_due:
                        self._commit_training_stage(
                            store,
                            identity,
                            config,
                            compatibility,
                            to_train,
                            succeeded,
                            failures_by_task,
                            executed,
                            all_done,
                        )

        current_index = store.current_training_adapters(identity)
        if not current_index or not current_index.get("complete"):
            self._commit_training_stage(
                store,
                identity,
                config,
                compatibility,
                to_train,
                succeeded,
                failures_by_task,
                executed,
                required_names.issubset(set(succeeded) | set(failures_by_task)),
            )
            current_index = store.current_training_adapters(identity)
        all_payloads = {**reused, **succeeded}
        return all_payloads, failures_by_task, executed, current_index

    def run(self) -> PipelineResult:
        config_path = self.samples_directory / "peft_config.yaml"
        try:
            config = load_config(config_path)
        except BusinessError as exc:
            return self._early_failure([exc.as_failure()])

        base_path = self._base_path(config)
        base: BasePackage | None = None
        base_failure: Failure | None = None
        try:
            base = load_base_package(base_path)
        except BusinessError as exc:
            base_failure = exc.as_failure()

        tasks_path = self.samples_directory / "tasks.json"
        tasks, task_failures = load_tasks(
            tasks_path,
            config,
            {name for name in base.layers} if base is not None else None,
        )
        samples, sample_failures = load_samples(self.samples_directory / "instructions.jsonl")
        failures: list[Failure] = [*task_failures, *sample_failures]
        if base_failure is not None:
            failures.append(base_failure)
            return self._early_failure(failures, config=config, tasks=tasks)

        assert base is not None
        valid_sample_layers = set(base.layers) & set(config.target_layer_whitelist)
        valid_samples: list[InstructionSample] = []
        for sample in samples:
            if sample.target_layer in valid_sample_layers:
                valid_samples.append(sample)
            else:
                failures.append(
                    Failure(
                        E_TASK_INVALID,
                        "sample target layer is not in the base package and PEFT whitelist",
                        target_layer=sample.target_layer,
                        details={"sample_id": sample.sample_id},
                    )
                )
        samples = valid_samples
        try:
            store = SnapshotStore(self.workspace)
        except BusinessError as exc:
            failures.append(exc.as_failure())
            return self._early_failure(failures, config=config, base=base, tasks=tasks)

        identity = {
            "base_fingerprint": base.fingerprint,
            "config_hash": config.hash,
            "task_set_hash": task_set_hash(tasks),
            "sample_set_hash": sample_set_hash(samples),
        }
        initial_recovery = store.recovery_info(identity)
        existing = store.discover_adapters()

        compatibility: list[dict[str, Any]] | None = None
        load_payload: dict[str, Any] | None = None
        identity_matches = store.identity_matches(identity)
        if identity_matches:
            load_stage_path = store.current_stage_path("load")
            if load_stage_path is not None:
                try:
                    candidate = read_json(load_stage_path / "load.json")
                    if (
                        isinstance(candidate, dict)
                        and candidate.get("complete") is True
                        and candidate.get("base_fingerprint") == base.fingerprint
                        and candidate.get("config_hash") == config.hash
                        and candidate.get("task_set_hash") == identity["task_set_hash"]
                        and candidate.get("sample_set_hash") == identity["sample_set_hash"]
                    ):
                        load_payload = candidate
                        compatibility = candidate.get("compatibility", [])
                except (OSError, ValueError, TypeError):
                    load_payload = None

        self.current_stage = "load"
        if compatibility is None:
            compatibility = build_compatibility_table(tasks, base, existing)
            load_payload = {
                "format": "peft-lora-load-stage-v1",
                "complete": True,
                "base_path": self._base_report_path(base),
                "base_fingerprint": base.fingerprint,
                "layer_fingerprints": dict(sorted(base.layer_fingerprints.items())),
                "config_hash": config.hash,
                "task_set_hash": identity["task_set_hash"],
                "sample_set_hash": identity["sample_set_hash"],
                "snapshot_commit_interval": config.snapshot_commit_interval,
                "tasks": [task.as_dict() for task in tasks],
                "compatibility": compatibility,
                "compatibility_hash": compatibility_hash(compatibility),
                "reuse": sorted(
                    entry["task_name"] for entry in compatibility if entry["action"] == "REUSE"
                ),
                "retrain": sorted(
                    entry["task_name"] for entry in compatibility if entry["action"] == "TRAIN"
                ),
            }

            def populate_load(temp: Path, _version: int) -> dict[str, Any]:
                atomic_write_json(temp / "load.json", load_payload)
                self.controller.test_hold("load")
                return {"complete": True}

            store.commit_stage(
                "load",
                identity,
                populate_load,
                compatibility_hash=load_payload["compatibility_hash"],
                snapshot_commit_interval=config.snapshot_commit_interval,
                replace_stages=True,
                stage_metadata={"complete": True},
            )

        assert load_payload is not None
        selected_compatibility = load_payload["compatibility"]
        selected_hash = load_payload["compatibility_hash"]
        self.current_stage = "load"
        if self.controller.requested:
            return self._graceful_stop(
                store=store,
                identity=identity,
                load_payload=load_payload,
                failures=failures,
                compatibility=selected_compatibility,
                training_index=None,
            )

        self.current_stage = "training"
        all_payloads, training_failures, executed, training_index = self._run_training(
            store=store,
            identity=identity,
            config=config,
            base=base,
            tasks=tasks,
            samples=samples,
            compatibility=selected_compatibility,
            existing=existing,
        )
        failures.extend(training_failures.values())
        if self.controller.requested:
            return self._graceful_stop(
                store=store,
                identity=identity,
                load_payload=load_payload,
                failures=failures,
                compatibility=selected_compatibility,
                training_index=training_index,
            )

        self.current_stage = "merge"
        merge_payload: dict[str, Any]
        merge_source: Path | None = None
        previous_merge_path = store.current_stage_path("merge")
        merge_reused = False
        if identity_matches and previous_merge_path is not None:
            try:
                candidate = read_json(previous_merge_path / "merge.json")
                if (
                    isinstance(candidate, dict)
                    and candidate.get("complete") is True
                    and candidate.get("status") == "OK"
                    and (previous_merge_path / "merged" / "manifest.json").is_file()
                ):
                    merge_payload = candidate
                    merge_source = previous_merge_path / "merged"
                    merge_reused = True
            except (OSError, ValueError, TypeError):
                pass

        if not merge_reused:
            merge_tasks = [task for task in tasks if task.task_name in all_payloads]
            merge_result: MergeResult | None = None
            merge_error: BusinessError | None = None
            try:
                merge_result = merge_adapters(base, merge_tasks, all_payloads, config)
            except BusinessError as exc:
                merge_error = exc

            if merge_error is not None:
                failures.append(
                    Failure(
                        merge_error.code,
                        merge_error.message,
                        details={
                            **merge_error.details,
                            "previous_artifact_preserved": previous_merge_path is not None,
                        },
                    )
                )
                merge_payload = {
                    "format": "peft-lora-merge-stage-v1",
                    "complete": True,
                    "status": "ERROR",
                    "business_code": merge_error.code,
                    "message": merge_error.message,
                    "details": merge_error.details,
                    "artifact": None,
                    "previous_artifact": (
                        relative_path(previous_merge_path, self.workspace)
                        if previous_merge_path is not None
                        else None
                    ),
                    "merged_tasks": [],
                    "pending_tasks": [],
                    "budget": None,
                    "restore_max_abs_error": None,
                }

                def populate_merge_error(temp: Path, _version: int) -> dict[str, Any]:
                    atomic_write_json(temp / "merge.json", merge_payload)
                    self.controller.test_hold("merge")
                    return {"status": "ERROR"}

                store.commit_stage(
                    "merge",
                    identity,
                    populate_merge_error,
                    compatibility_hash=selected_hash,
                    snapshot_commit_interval=config.snapshot_commit_interval,
                    replace_stages=False,
                    stage_metadata={"status": "ERROR"},
                )
            else:
                assert merge_result is not None
                for error in merge_result.business_errors:
                    failures.append(
                        Failure(
                            error["code"],
                            error["message"],
                            task_name=error.get("task_name"),
                            target_layer=error.get("target_layer"),
                            details=error.get("details"),
                        )
                    )
                merge_payload = {
                    "format": "peft-lora-merge-stage-v1",
                    "complete": True,
                    "status": "OK",
                    "artifact": "merged",
                    "merged_tasks": merge_result.merged_tasks,
                    "pending_tasks": merge_result.pending_tasks,
                    "budget": merge_result.budget.summary(),
                    "restore_max_abs_error": merge_result.restore_max_abs_error,
                    "restore_per_layer": merge_result.restore_errors,
                    "quantization": {
                        "scale": merge_result.quantization_scale,
                        "step": merge_result.quantization_step,
                        "allowed_restore_error": merge_result.allowed_restore_error,
                    },
                }

                def populate_merge(temp: Path, _version: int) -> dict[str, Any]:
                    write_merged_package(temp / "merged", merge_result, base)
                    atomic_write_json(temp / "merge.json", merge_payload)
                    self.controller.test_hold("merge")
                    return {"status": "OK"}

                reference = store.commit_stage(
                    "merge",
                    identity,
                    populate_merge,
                    compatibility_hash=selected_hash,
                    snapshot_commit_interval=config.snapshot_commit_interval,
                    replace_stages=False,
                    stage_metadata={"status": "OK"},
                )
                merge_source = store.stage_path("merge", int(reference["global_version"])) / "merged"

        if self.controller.requested:
            return self._graceful_stop(
                store=store,
                identity=identity,
                load_payload=load_payload,
                failures=failures,
                compatibility=selected_compatibility,
                training_index=training_index,
            )

        self.current_stage = "export"
        compatibility_by_task = {entry["task_name"]: entry for entry in selected_compatibility}
        merge_ok = merge_payload.get("status") == "OK" and merge_source is not None
        if merge_ok:
            if (merge_source / "manifest.json").is_file():
                merge_manifest = read_json(merge_source / "manifest.json")
            else:
                merge_manifest = {"artifact": str(merge_source)}
        else:
            merge_manifest = None

        def populate_export(temp: Path, version: int) -> dict[str, Any]:
            adapter_exports: dict[str, str] = {}
            for task in sorted(tasks, key=task_order_key):
                payload = all_payloads.get(task.task_name)
                if payload is None:
                    continue
                filename = stable_artifact_name(task.task_name, task.target_layer, ".json")
                relative = f"adapters/{filename}"
                save_adapter(temp / relative, payload)
                adapter_exports[task.task_name] = (
                    f"snapshot/stages/export/v{version:08d}/{relative}"
                )
            merged_export = f"snapshot/stages/export/v{version:08d}/merged"
            if merge_ok and merge_source is not None:
                shutil.copytree(merge_source, temp / "merged")

            stages_for_report = {
                stage: int(reference["version"])
                for stage, reference in (store.current or {}).get("stages", {}).items()
                if isinstance(reference, dict) and isinstance(reference.get("version"), int)
            }
            stages_for_report["export"] = version
            task_reports: list[dict[str, Any]] = []
            reuse_names: list[str] = []
            retrain_names: list[str] = []
            adapter_entries: list[dict[str, Any]] = []
            for task in sorted(tasks, key=task_order_key):
                compatibility_entry = compatibility_by_task[task.task_name]
                if compatibility_entry["action"] == "REUSE":
                    reuse_names.append(task.task_name)
                else:
                    retrain_names.append(task.task_name)
                payload = all_payloads.get(task.task_name)
                entry: dict[str, Any] = {
                    "task_name": task.task_name,
                    "target_layer": task.target_layer,
                    "rank": task.rank,
                    "scale": task.scale,
                    "stable_key": task.stable_key,
                    "compatibility": compatibility_entry,
                    "adapter_export": adapter_exports.get(task.task_name),
                    "training": None,
                }
                if payload is not None:
                    entry["training"] = {
                        "source": (
                            "REUSED_EXPORT"
                            if compatibility_entry["action"] == "REUSE"
                            else "TRAINED"
                        ),
                        "loss_trajectory": payload.get("loss_trajectory", []),
                        "final_loss": payload.get("final_loss"),
                        "adapter_base_fingerprint": payload.get("base_fingerprint"),
                        "adapter_layer_fingerprint": payload.get(
                            "target_layer_fingerprint"
                        ),
                        "content_hash": payload.get("content_hash"),
                    }
                    adapter_entries.append(
                        {
                            "task_name": task.task_name,
                            "target_layer": task.target_layer,
                            "file": adapter_exports[task.task_name],
                            "content_hash": payload.get("content_hash"),
                        }
                    )
                else:
                    entry["training"] = {"source": "FAILED"}
                task_reports.append(entry)

            sorted_failures = _sorted_failures(failures)
            report = {
                "format": REPORT_FORMAT,
                "status": "success" if not sorted_failures else "partial",
                "exit_code": 0 if not sorted_failures else 1,
                "business_codes": _business_codes(failures, selected_compatibility),
                "failures": sorted_failures,
                "base": {
                    "path": self._base_report_path(base),
                    "fingerprint": base.fingerprint,
                    "layer_fingerprints": dict(sorted(base.layer_fingerprints.items())),
                    "layer_count": len(base.layers),
                },
                "config": {"hash": config.hash, **config.as_dict()},
                "tasks": task_reports,
                "compatibility": selected_compatibility,
                "compatibility_hash": selected_hash,
                "reuse": sorted(reuse_names),
                "retrain": sorted(retrain_names),
                "training_executed": sorted(executed),
                "merge": {
                    "status": merge_payload.get("status"),
                    "artifact": merged_export if merge_ok else None,
                    "manifest": merge_manifest,
                    "merged_tasks": merge_payload.get("merged_tasks", []),
                    "pending_tasks": merge_payload.get("pending_tasks", []),
                    "restore_max_abs_error": merge_payload.get("restore_max_abs_error"),
                    "restore_per_layer": merge_payload.get("restore_per_layer", {}),
                    "quantization": merge_payload.get("quantization"),
                    "budget": merge_payload.get("budget"),
                    "error": merge_payload.get("business_code"),
                    "message": merge_payload.get("message"),
                    "previous_artifact": merge_payload.get("previous_artifact"),
                },
                "snapshot": {
                    "version": version,
                    "stage_versions": stages_for_report,
                    "config_hash": config.hash,
                    "base_fingerprint": base.fingerprint,
                    "compatibility_hash": selected_hash,
                    "snapshot_commit_interval": config.snapshot_commit_interval,
                },
                "recovery": initial_recovery,
                "shutdown": {
                    "requested": self.controller.requested,
                    "signal": self.controller.signal_name(),
                    "completed_stage": "export" if not self.controller.requested else "export",
                },
            }
            atomic_write_json(temp / "report.json", report)
            index = {
                "format": "peft-lora-export-index-v1",
                "complete": True,
                "adapters": adapter_entries,
                "merged": merged_export if merge_ok else None,
                "report": "report.json",
                **dict(identity),
            }
            atomic_write_json(temp / "index.json", index)
            self.controller.test_hold("export")
            return {"report": report}

        committed = store.commit_stage(
            "export",
            identity,
            populate_export,
            compatibility_hash=selected_hash,
            snapshot_commit_interval=config.snapshot_commit_interval,
            replace_stages=False,
            stage_metadata={"complete": True},
        )
        version = int(committed["global_version"])
        export_stage = store.stage_path("export", version)
        report = read_json(export_stage / "report.json")
        atomic_write_bytes(self.report_path, (export_stage / "report.json").read_bytes())
        exit_code = int(report.get("exit_code", 1))
        status = str(report.get("status", "failed"))
        return PipelineResult(exit_code, status, report, self.report_path)
