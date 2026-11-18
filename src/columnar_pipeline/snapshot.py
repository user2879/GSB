from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .canonical import atomic_write_json, canonical_json, sha256_obj, utc_now
from .errors import PipelineError, replay_error
from .model import Dataset, RunConfig


SNAPSHOT_FORMAT = 1


def build_snapshot(
    *,
    run_id: str,
    dataset: Dataset,
    config: RunConfig,
    result_json: str,
    result_digest: str,
    ledger_digest: str,
    task_count: int,
    split_count: int,
    data_path: str,
) -> dict[str, Any]:
    core = {
        "snapshot_format": SNAPSHOT_FORMAT,
        "run_id": run_id,
        "query_id": dataset.query.query_id,
        "data_digest": dataset.data_digest,
        "table_digests": dataset.table_digests(),
        "table_versions": dataset.table_versions(),
        "budget_version": config.budget_version,
        "budget_digest": config.budget_digest,
        "config_digest": config.config_digest,
        "split_threshold": config.split_threshold,
        "split_parts": config.split_parts,
        "row_group_size": config.row_group_size,
        "parallelism": config.parallelism,
        "memory_budget_bytes": config.memory_budget_bytes,
        "effective_memory_budget_bytes": config.effective_memory_budget(),
        "split_strategy": config.split_strategy,
        "skew_safety_limit": config.skew_safety_limit,
        "config": config.to_json_dict(),
        "injection": dataset.injection,
        "ledger_digest": ledger_digest,
        "result_json": result_json,
        "result_digest": result_digest,
        "task_count": task_count,
        "split_count": split_count,
        "data_path": data_path,
        "created_at": utc_now(),
    }
    snapshot_id = sha256_obj(core)
    return {"snapshot_id": snapshot_id, **core}


def snapshot_path(snapshot_dir: str | Path, run_id: str) -> Path:
    return Path(snapshot_dir) / f"{run_id}.json"


def write_snapshot(snapshot_dir: str | Path, snapshot: Mapping[str, Any]) -> Path:
    path = snapshot_path(snapshot_dir, str(snapshot["run_id"]))
    atomic_write_json(path, dict(snapshot), newline=True)
    return path


def load_snapshot(path: str | Path) -> dict[str, Any]:
    snapshot_file = Path(path)
    try:
        raw = json.loads(snapshot_file.read_text(encoding="utf-8"))
    except OSError as error:
        raise replay_error(
            f"cannot read snapshot: {error}",
            task_id="<snapshot>",
            field_name="snapshot_file",
            details={"path": str(snapshot_file)},
        ) from error
    except json.JSONDecodeError as error:
        raise replay_error(
            f"snapshot is not valid JSON: {error.msg}",
            task_id="<snapshot>",
            field_name="snapshot_json",
            details={"path": str(snapshot_file), "line": error.lineno, "column": error.colno},
        ) from error
    if not isinstance(raw, Mapping):
        raise replay_error(
            "snapshot root must be a JSON object",
            task_id="<snapshot>",
            field_name="snapshot_json",
        )
    snapshot = dict(raw)
    if snapshot.get("snapshot_format") != SNAPSHOT_FORMAT:
        raise replay_error(
            "unsupported snapshot format",
            task_id="<snapshot>",
            field_name="snapshot_format",
            details={"expected": SNAPSHOT_FORMAT, "actual": snapshot.get("snapshot_format")},
        )
    required = {
        "snapshot_id",
        "run_id",
        "query_id",
        "data_digest",
        "table_digests",
        "table_versions",
        "budget_version",
        "config_digest",
        "split_threshold",
        "split_parts",
        "ledger_digest",
        "result_json",
        "result_digest",
        "config",
    }
    missing = sorted(required - set(snapshot))
    if missing:
        raise replay_error(
            "snapshot is missing required fields",
            task_id="<snapshot>",
            field_name=missing[0],
            details={"missing": missing},
        )
    core = {key: value for key, value in snapshot.items() if key != "snapshot_id"}
    expected_id = sha256_obj(core)
    if expected_id != snapshot["snapshot_id"]:
        raise replay_error(
            "snapshot id does not match snapshot content",
            task_id="<snapshot>",
            field_name="snapshot_id",
            details={"expected": expected_id, "actual": snapshot["snapshot_id"]},
        )
    return snapshot


def compare_snapshot(
    snapshot: Mapping[str, Any],
    dataset: Dataset,
    config: RunConfig,
) -> dict[str, Any]:
    reasons: list[str] = []
    if snapshot["data_digest"] != dataset.data_digest:
        reasons.append("data_digest")
    if dict(snapshot["table_digests"]) != dataset.table_digests():
        reasons.append("table_digests")
    if dict(snapshot["table_versions"]) != dataset.table_versions():
        reasons.append("table_versions")
    if snapshot["budget_version"] != config.budget_version:
        reasons.append("budget_version")
    if snapshot.get("budget_digest") not in (None, config.budget_digest):
        reasons.append("budget_digest")
    if snapshot["split_threshold"] != config.split_threshold:
        reasons.append("split_threshold")
    if snapshot["split_parts"] != config.split_parts:
        reasons.append("split_parts")
    if snapshot.get("row_group_size") != config.row_group_size:
        reasons.append("row_group_size")
    if snapshot.get("parallelism") != config.parallelism:
        reasons.append("parallelism")
    if snapshot.get("effective_memory_budget_bytes") != config.effective_memory_budget():
        reasons.append("memory_budget")
    if snapshot["config_digest"] != config.config_digest:
        reasons.append("config_digest")
    if snapshot.get("query_id") != dataset.query.query_id:
        reasons.append("query_id")
    return {
        "compatible": not reasons,
        "reasons": reasons,
    }


def snapshot_config(snapshot: Mapping[str, Any]) -> RunConfig:
    try:
        return RunConfig.from_json_dict(snapshot["config"])
    except PipelineError:
        raise
    except Exception as error:
        raise replay_error(
            "snapshot contains an invalid run configuration",
            task_id="<snapshot>",
            field_name="config",
            details={"exception": str(error)},
        ) from error
