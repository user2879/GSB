from __future__ import annotations

import copy
import re
import tempfile
from pathlib import Path
from typing import Any

from .cache import CacheStore
from .commands import CommandState, initial_command_state
from .config import RecognitionConfig
from .pipeline import Pipeline
from .snapshot import SnapshotStore, snapshot_stage_docs


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return cleaned[:80] or "command"


def _snapshot_stages(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    keys = {
        image["name"]: {
            stage: details["cache_key"]
            for stage, details in image["stages"].items()
        }
        for image in report.get("images", [])
    }
    pending_images = {
        image["name"] for image in report.get("images", []) if image.get("pending_match_queue")
    }
    return snapshot_stage_docs(
        report.get("images", []),
        keys,
        consumed_points=int(report.get("budget", {}).get("consumed", 0)),
        pending_images=pending_images,
    )


def _write_report(path: Path, report: dict[str, Any]) -> None:
    from .canonical import atomic_write_json

    atomic_write_json(path, report)


def _commit_snapshot(
    store: SnapshotStore,
    report: dict[str, Any],
    *,
    global_version: int,
    command_index: int,
    command_id: str,
    config: RecognitionConfig,
) -> tuple[int, dict[str, Any]]:
    version = global_version + 1
    result = store.commit(
        global_version=version,
        command_index=command_index,
        command_id=command_id,
        stages=_snapshot_stages(report),
        config_hash=config.config_hash,
    )
    return version, result


def run_command_sequence(
    *,
    data_dir: Path,
    config: RecognitionConfig,
    commands: list[dict[str, Any]],
    base_dir: Path,
    work_dir: Path,
) -> dict[str, Any]:
    work_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = work_dir / "reports"
    cache_dir = work_dir / "cache"
    snapshot_dir = work_dir / "snapshots"
    reports_dir.mkdir(parents=True, exist_ok=True)
    state = initial_command_state(data_dir, config, base_dir)
    cache = CacheStore(cache_dir)
    snapshot_store = SnapshotStore(snapshot_dir, config.config_hash)
    report_index: list[dict[str, Any]] = []
    global_version = 0

    initial_report = Pipeline(config, cache, respect_faults=True).run(
        state.ordered_sources(), mode="incremental", snapshot_version=0, command=None
    )
    global_version, commit = _commit_snapshot(
        snapshot_store,
        initial_report,
        global_version=global_version,
        command_index=0,
        command_id="initial",
        config=config,
    )
    initial_report["snapshot_version"] = global_version
    initial_report["snapshot_commit"] = commit
    initial_path = reports_dir / "step_000_initial.json"
    _write_report(initial_path, initial_report)
    report_index.append(
        {
            "step": 0,
            "command_id": "initial",
            "report": str(initial_path),
            "snapshot_version": global_version,
        }
    )

    for command_index, command in enumerate(commands, start=1):
        effect = state.apply(command)
        report = Pipeline(state.config, cache, respect_faults=True).run(
            state.ordered_sources(),
            mode="incremental",
            snapshot_version=global_version,
            command={"index": command_index, **command, "effect": effect},
        )
        if command_index % state.config.snapshot_commit_interval_commands == 0:
            global_version, commit = _commit_snapshot(
                snapshot_store,
                report,
                global_version=global_version,
                command_index=command_index,
                command_id=str(command["id"]),
                config=state.config,
            )
            report["snapshot_commit"] = commit
        report["snapshot_version"] = global_version
        path = reports_dir / f"step_{command_index:03d}_{_safe_id(str(command['id']))}.json"
        _write_report(path, report)
        report_index.append(
            {
                "step": command_index,
                "command_id": command["id"],
                "command_type": command["type"],
                "effect": effect,
                "report": str(path),
                "snapshot_version": global_version,
            }
        )

    final_report = report_index[-1]
    index = {
        "schema": "layout-command-run-v1",
        "data_dir": str(data_dir),
        "config_hash": state.config.config_hash,
        "initial_config_hash": config.config_hash,
        "command_count": len(commands),
        "snapshot_version": global_version,
        "reports": report_index,
        "final_report": final_report,
    }
    _write_report(work_dir / "index.json", index)
    return index


def _cache_state_for_stage(image: dict[str, Any], stage: str) -> str:
    return str(image.get("stages", {}).get(stage, {}).get("state", "missing"))


def _expected_stage_states(
    previous_state: CommandState,
    current_state: CommandState,
    command: dict[str, Any] | None,
) -> dict[str, dict[str, str]]:
    previous_config = previous_state.config
    current_config = current_state.config
    detection_changed = previous_config.detection_params != current_config.detection_params
    line_changed = previous_config.line_params != current_config.line_params
    segmentation_changed = (
        previous_config.segmentation_params != current_config.segmentation_params
    )
    recognition_changed = (
        previous_config.recognition_params != current_config.recognition_params
    )
    budget_changed = (
        previous_config.char_match_point_budget != current_config.char_match_point_budget
    )
    expected: dict[str, dict[str, str]] = {}
    for name in sorted(current_state.sources):
        if name not in previous_state.sources:
            expected[name] = {
                "detection": "recomputed",
                "line_aggregation": "recomputed",
                "segmentation": "recomputed",
                "recognition": "recomputed",
            }
            continue
        if command and command.get("type") == "replace_image" and command.get("payload", {}).get("name") == name:
            expected[name] = {
                "detection": "recomputed",
                "line_aggregation": "recomputed",
                "segmentation": "recomputed",
                "recognition": "recomputed",
            }
            continue
        if detection_changed:
            expected[name] = {
                "detection": "recomputed",
                "line_aggregation": "recomputed",
                "segmentation": "recomputed",
                "recognition": "recomputed",
            }
        elif line_changed:
            expected[name] = {
                "detection": "reused",
                "line_aggregation": "recomputed",
                "segmentation": "recomputed",
                "recognition": "recomputed",
            }
        elif segmentation_changed:
            expected[name] = {
                "detection": "reused",
                "line_aggregation": "reused",
                "segmentation": "recomputed",
                "recognition": "recomputed",
            }
        elif recognition_changed:
            expected[name] = {
                "detection": "reused",
                "line_aggregation": "reused",
                "segmentation": "reused",
                "recognition": "recomputed",
            }
        elif budget_changed:
            expected[name] = {
                "detection": "reused",
                "line_aggregation": "reused",
                "segmentation": "reused",
                "recognition": "scheduler_recompute_or_pending",
            }
        else:
            expected[name] = {
                "detection": "reused",
                "line_aggregation": "reused",
                "segmentation": "reused",
                "recognition": "reused",
            }
    return expected


def _check_invalidation(
    previous_state: CommandState,
    current_state: CommandState,
    command: dict[str, Any],
    report: dict[str, Any],
) -> list[dict[str, Any]]:
    expected = _expected_stage_states(previous_state, current_state, command)
    mismatches: list[dict[str, Any]] = []
    for image in report.get("images", []):
        name = image["name"]
        for stage, expected_state in expected.get(name, {}).items():
            actual = _cache_state_for_stage(image, stage)
            if stage == "recognition" and expected_state == "scheduler_recompute_or_pending":
                okay = actual in {
                    "reused",
                    "recomputed",
                    "pending",
                    "reused_pending",
                }
            elif stage == "recognition" and expected_state == "recomputed":
                okay = actual in {"recomputed", "pending", "reused_pending"}
            elif stage == "recognition" and expected_state == "reused":
                okay = actual in {"reused", "reused_pending"}
            else:
                okay = actual == expected_state
            if not okay:
                mismatches.append(
                    {
                        "image": name,
                        "stage": stage,
                        "expected": expected_state,
                        "actual": actual,
                    }
                )
    return mismatches
