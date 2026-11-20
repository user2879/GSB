"""工作台运行器：加载初始输入、恢复判定、逐批执行、全量对照诊断。"""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from typing import Any, Callable

from .batches import BatchPlan, apply_batch, load_batch_plan, validate_runtime_weights
from .errors import E_CONFIG_MISMATCH, E_CONFIG_INVALID, Failure, OK
from .loader import (
    load_config,
    load_score_params,
    load_seeds,
    load_slot_sources,
    load_vocab,
)
from .models import WorkspaceState
from .pipeline import Pipeline
from .snapshot import SnapshotStore
from .utils import canonical_json, fingerprint


@dataclass
class RunResult:
    business_code: str
    report: dict[str, Any] | None
    failures: list[Failure]
    recovery_batch_no: int | None
    stage_artifacts: dict[str, dict[str, Any]]
    shadow_report: dict[str, Any] | None = None


def _load_initial_state(
    sample_dir: str, failures: list[Failure]
) -> tuple[WorkspaceState | None, dict[str, str]]:
    seeds = load_seeds(os.path.join(sample_dir, "seeds.jsonl"), failures)
    sources = load_slot_sources(
        os.path.join(sample_dir, "slot_sources.jsonl"), failures
    )
    vocab = load_vocab(os.path.join(sample_dir, "vocab.json"), failures)
    config = load_config(os.path.join(sample_dir, "build_config.json"), failures)
    params = load_score_params(
        os.path.join(sample_dir, "score_params.json"), failures
    )
    if config is None or params is None:
        return None, {}
    state = WorkspaceState(
        seeds=seeds,
        slot_sources=sources,
        config=config,
        score_params=params,
        vocab=vocab,
    )
    _cross_validate(state, failures)
    return state, state.fingerprints()


def _cross_validate(state: WorkspaceState, failures: list[Failure]) -> None:
    config = state.config
    for task in config.task_types:
        task_seeds = [seed for seed in state.seeds.values() if seed.task_type == task]
        if not task_seeds:
            failures.append(
                Failure(
                    E_CONFIG_INVALID,
                    "seeds.jsonl",
                    task,
                    f"no seed template for task type {task}",
                )
            )
            continue
        template_ids = {seed.template_id for seed in task_seeds}
        unknown = set(config.template_weights[task]) - template_ids
        if unknown:
            failures.append(
                Failure(
                    E_CONFIG_INVALID,
                    "build_config.json",
                    task,
                    f"template weights reference unknown templates {sorted(unknown)}",
                )
            )
        unweighted = template_ids - set(config.template_weights[task])
        if unweighted:
            failures.append(
                Failure(
                    E_CONFIG_INVALID,
                    "build_config.json",
                    task,
                    f"templates missing weights {sorted(unweighted)}",
                )
            )
        slot_names = {
            (source.task_type, source.slot_name)
            for source in state.slot_sources.values()
            if source.task_type == task
        }
        for seed in task_seeds:
            for slot in seed.slots:
                if (task, slot) not in slot_names:
                    failures.append(
                        Failure(
                            E_CONFIG_INVALID,
                            "slot_sources.jsonl",
                            seed.seed_id,
                            f"no slot source for {task}/{slot}",
                        )
                    )
        source_ids = {
            source.source_id
            for source in state.slot_sources.values()
            if source.task_type == task
        }
        unknown_sources = set(config.slot_weights[task]) - source_ids
        if unknown_sources:
            failures.append(
                Failure(
                    E_CONFIG_INVALID,
                    "build_config.json",
                    task,
                    f"slot weights reference unknown sources {sorted(unknown_sources)}",
                )
            )


def _write_json(path: str, payload: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(payload))


def _build_full(
    final_state: WorkspaceState,
    base_fingerprint: str,
    batch_no: int,
    failures: list[Failure],
    fault_hook: Callable[[str, int], None] | None,
    temp_root: str,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """从空白快照目录按给定最终输入一次性全量构造（影子运行）。"""
    shadow_store = SnapshotStore(os.path.join(temp_root, "shadow_snapshot"))
    pipeline = Pipeline(
        copy.deepcopy(final_state),
        shadow_store,
        base_fingerprint,
        failures=[],
        fault_hook=fault_hook,
    )
    report = pipeline.execute_batch(batch_no, previous={}, checkpoint=False)
    artifacts = {
        "sampling": shadow_store.read_stage("sampling"),
        "scoring": shadow_store.read_stage("scoring"),
        "dedup": shadow_store.read_stage("dedup"),
    }
    return report, artifacts


def compare_artifacts(
    incremental: dict[str, dict[str, Any]],
    full: dict[str, dict[str, Any]],
    source_batch: int | str = "unknown",
) -> list[dict[str, Any]]:
    """逐字段比对增量与全量产物，返回差异清单（空列表表示完全一致）。"""
    differences: list[dict[str, Any]] = []
    for stage in ("sampling", "scoring", "dedup"):
        left = incremental.get(stage)
        right = full.get(stage)
        if canonical_json(left) != canonical_json(right):
            differences.extend(
                _diff_objects(stage, left, right, "$", source_batch)
            )
    return differences


def _diff_objects(
    stage: str, left: Any, right: Any, path: str, source_batch: int | str
) -> list[dict[str, Any]]:
    differences: list[dict[str, Any]] = []
    if type(left) is not type(right):
        differences.append({"stage": stage, "field": path, "incremental": left,
                            "full": right, "source_batch": source_batch})
        return differences
    if isinstance(left, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                differences.append(
                    {"stage": stage, "field": f"{path}.{key}",
                     "incremental": left.get(key), "full": right.get(key),
                     "source_batch": source_batch}
                )
            else:
                differences.extend(
                    _diff_objects(
                        stage, left[key], right[key],
                        f"{path}.{key}", source_batch,
                    )
                )
    elif isinstance(left, list):
        if len(left) != len(right):
            differences.append(
                {"stage": stage, "field": f"{path}.length",
                 "incremental": len(left), "full": len(right),
                 "source_batch": source_batch}
            )
        for index, (a, b) in enumerate(zip(left, right)):
            differences.extend(
                _diff_objects(
                    stage, a, b, f"{path}[{index}]", source_batch
                )
            )
    else:
        if left != right:
            differences.append(
                {"stage": stage, "field": path, "incremental": left,
                 "full": right, "source_batch": source_batch}
            )
    return differences


def run_workbench(
    sample_dir: str,
    output_dir: str,
    up_to_batch: int | None = None,
    shadow_check: bool = True,
    fault_hook: Callable[[str, int], None] | None = None,
) -> RunResult:
    failures: list[Failure] = []
    initial_state, initial_fps = _load_initial_state(sample_dir, failures)
    if initial_state is None:
        return RunResult(
            business_code=E_CONFIG_INVALID,
            report=None,
            failures=failures,
            recovery_batch_no=None,
            stage_artifacts={},
        )
    base_fingerprint = fingerprint(initial_fps)
    plans = load_batch_plan(os.path.join(sample_dir, "batch_plan.json"), failures)

    snapshot_dir = os.path.join(output_dir, "snapshot")
    store = SnapshotStore(snapshot_dir)
    store.cleanup_temp_dirs()

    final_state = copy.deepcopy(initial_state)
    selected_plans: list[BatchPlan] = []
    for plan in plans:
        if up_to_batch is not None and plan.batch_no > up_to_batch:
            break
        final_state = apply_batch(final_state, plan)
        validate_runtime_weights(final_state, failures, plan.batch_no)
        selected_plans.append(plan)

    manifest, recovery_failure = store.resume_check(
        base_fingerprint, final_state.combined_fingerprint()
    )
    if recovery_failure is not None:
        failures.append(recovery_failure)
        return RunResult(
            business_code=E_CONFIG_MISMATCH,
            report=None,
            failures=failures,
            recovery_batch_no=None,
            stage_artifacts={},
        )

    recovery_batch_no = manifest.completed_batch_no if manifest else None
    previous: dict[str, dict[str, Any]] = {}
    if manifest is not None:
        for stage in ("sampling", "scoring", "dedup"):
            payload = store.read_stage(stage)
            if payload is not None:
                previous[stage] = payload

    pipeline = Pipeline(
        final_state,
        store,
        base_fingerprint,
        failures=failures,
        fault_hook=fault_hook,
    )
    pipeline.manifest = manifest
    if recovery_batch_no is not None:
        pipeline._recovered_from = recovery_batch_no  # type: ignore[attr-defined]

    target_batch_no = selected_plans[-1].batch_no if selected_plans else 0
    interval = final_state.config.snapshot_interval_batches
    report: dict[str, Any] | None = None
    all_stages_present = all(stage in previous for stage in ("sampling", "scoring", "dedup"))
    if (
        manifest is None
        or manifest.combined_fingerprint != final_state.combined_fingerprint()
        or not all_stages_present
    ):
        checkpoint = target_batch_no != 0 and (target_batch_no % interval == 0)
        report = pipeline.execute_batch(target_batch_no, previous, checkpoint)
    else:
        report = pipeline.last_report or _rebuild_report_from_snapshot(
            pipeline, target_batch_no, previous, checkpoint=False
        )

    stage_artifacts = {
        stage: store.read_stage(stage)
        for stage in ("sampling", "scoring", "dedup")
    }

    shadow_report = None
    if shadow_check and report is not None:
        shadow_dir = os.path.join(output_dir, "shadow")
        shadow_report, shadow_artifacts = _build_full(
            final_state,
            base_fingerprint,
            target_batch_no,
            failures,
            fault_hook=None,
            temp_root=shadow_dir,
        )
        differences = compare_artifacts(
            stage_artifacts, shadow_artifacts, source_batch=target_batch_no
        )
        report["diagnostics"]["field_differences"] = differences
        report["diagnostics"]["shadow_fingerprints"] = shadow_report["fingerprints"]
        report["diagnostics"]["shadow_match"] = not differences
        _write_json(
            os.path.join(output_dir, "shadow_report.json"), shadow_report
        )

    if report is not None:
        _write_json(os.path.join(output_dir, "report.json"), report)
        _write_json(
            os.path.join(output_dir, "candidates.json"),
            stage_artifacts["sampling"]["candidates"],
        )
        _write_json(
            os.path.join(output_dir, "scores.json"),
            stage_artifacts["scoring"]["scores"],
        )
        _write_json(
            os.path.join(output_dir, "dedup_groups.json"),
            stage_artifacts["dedup"]["results"],
        )

    return RunResult(
        business_code=report["business_code"] if report else OK,
        report=report,
        failures=failures,
        recovery_batch_no=recovery_batch_no,
        stage_artifacts=stage_artifacts,
        shadow_report=shadow_report,
    )


def _rebuild_report_from_snapshot(
    pipeline: Pipeline,
    batch_no: int,
    previous: dict[str, dict[str, Any]],
    checkpoint: bool,
) -> dict[str, Any]:
    sampling = previous["sampling"]
    scoring = previous["scoring"]
    dedup = previous["dedup"]
    empty_stats = {
        "new_candidate_ids": [],
        "changed_candidate_ids": [],
        "reused_candidate_ids": [c["candidate_id"] for c in sampling["candidates"]],
        "removed_candidate_ids": [],
        "reasons": {},
    }
    score_status = {
        record["candidate_id"]: "reused" for record in scoring["scores"]
    }
    dedup_status: dict[str, str] = {}
    for result in dedup["results"]:
        for candidate_id in result["retained_ids"]:
            dedup_status[candidate_id] = "reused"
        for removal in result["removals"]:
            dedup_status[removal["candidate_id"]] = "reused"
    pipeline.manifest = pipeline.store.read_manifest()
    return pipeline.build_report(
        batch_no, sampling, scoring, dedup,
        empty_stats, score_status, dedup_status, checkpoint,
    )
