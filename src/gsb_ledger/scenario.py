"""Apply a deterministic command sequence and compare incremental/full ledgers."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .config import load_config, load_requests
from .engine import LedgerEngine
from .generator import generate_worker_rows, rows_to_ndjson
from .util import atomic_write_json, atomic_write_text, read_json


def _copy_project(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns(".state", "__pycache__", "*.pyc", "artifacts"),
    )


def _write_config(path: Path, raw: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(raw, ensure_ascii=False, indent=2) + "\n")


def _find_receipt_source(config: dict[str, Any], worker_id: str, version: str) -> str:
    for source in config.get("receipt_sources", []):
        if (
            source.get("default_worker_id") == worker_id
            and source.get("format", version) == version
        ):
            return str(source["path"])
    return f"receipts/{worker_id}.{version}.ndjson"


def _apply_mutation(
    project: Path,
    command: dict[str, Any],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    command_type = command.get("type")
    config_path = project / "config.json"
    requests_path = project / "requests.ndjson"
    generated_rows: list[dict[str, Any]] = []
    if command_type == "replace_receipt":
        source = project / str(command["source"])
        target = project / str(command["target"])
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        return None, generated_rows
    if command_type == "append_batch":
        config = load_config(config_path)
        requests = load_requests(requests_path, batch_size=config.batch_size)
        worker_id = str(command["worker_id"])
        param_version = int(command.get("param_version", config.worker_by_id[worker_id].param_version))
        version = str(command.get("receipt_version", "r3"))
        batch_id = str(command["batch_id"])
        request_ids = [str(value) for value in command.get("request_ids", [])]
        runtime_rows = generate_worker_rows(
            config,
            requests,
            worker_id=worker_id,
            version=version,
            param_version=param_version,
            batch_size=max(1, len(request_ids)),
            request_ids=request_ids,
            batch_prefix=batch_id,
            duration_ms=command.get("duration_ms"),
        )
        rows = runtime_rows
        # The generator appends a numeric suffix; one request_ids group is
        # emitted here, so force the caller-provided stable batch id.
        for row in rows:
            row["batch_id"] = batch_id
        covered_refs = command.get("covered_refs", {})
        if version == "r3" and isinstance(covered_refs, dict):
            for row in rows:
                row["covered_refs"] = [
                    str(value) for value in covered_refs.get(row["request_id"], [])
                ]
        target_path = command.get("target")
        if target_path is None:
            raw_config = read_json(config_path)
            target_path = _find_receipt_source(raw_config, worker_id, version)
        target = project / str(target_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        existing = target.read_text(encoding="utf-8") if target.exists() else ""
        if existing and not existing.endswith("\n"):
            existing += "\n"
        atomic_write_text(target, existing + rows_to_ndjson(rows))
        generated_rows.extend(rows)
        return None, generated_rows
    if command_type == "switch_param_version":
        raw = read_json(config_path)
        worker_id = str(command["worker_id"])
        version = int(command["param_version"])
        keep_old = bool(command.get("keep_old_merge_entry", True))
        for worker in raw.get("workers", []):
            if worker.get("worker_id") == worker_id:
                old_version = int(worker.get("param_version", 0))
                worker["param_version"] = version
                known = {
                    int(value)
                    for value in worker.get("known_param_versions", [old_version])
                }
                known.add(old_version)
                known.add(version)
                worker["known_param_versions"] = sorted(known)
                break
        else:
            raise ValueError(f"unknown worker {worker_id!r}")
        merge_rule = raw.setdefault("merge_rule", {})
        entries = list(merge_rule.get("entries", []))
        if not keep_old:
            entries = [entry for entry in entries if entry.get("worker_id") != worker_id]
        if not any(
            entry.get("worker_id") == worker_id and entry.get("param_version") == version
            for entry in entries
        ):
            entries.append({"worker_id": worker_id, "param_version": version})
        merge_rule["entries"] = entries
        _write_config(config_path, raw)
        return raw, generated_rows
    if command_type in {"set_batch_size", "set_fixed_point_bits", "set_throughput"}:
        raw = read_json(config_path)
        field = {
            "set_batch_size": "batch_size",
            "set_fixed_point_bits": "fixed_point_bits",
            "set_throughput": "throughput_ops_per_ms",
        }[command_type]
        raw[field] = int(command[field])
        _write_config(config_path, raw)
        return raw, generated_rows
    if command_type == "set_merge_rule":
        raw = read_json(config_path)
        merge_rule = raw.setdefault("merge_rule", {})
        merge_rule["entries"] = [
            {
                "worker_id": str(entry["worker_id"]),
                "param_version": int(entry["param_version"]),
            }
            for entry in command.get("entries", [])
        ]
        if "tie_break" in command:
            merge_rule["tie_break"] = list(command["tie_break"])
        _write_config(config_path, raw)
        return raw, generated_rows
    if command_type in {"run_incremental", "build"}:
        return None, generated_rows
    raise ValueError(f"unknown scenario command {command_type!r}")


def _ledger_diff_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if type(left) is not type(right):
        return [prefix or "<root>"]
    if isinstance(left, dict):
        result: list[str] = []
        for key in sorted(set(left) | set(right)):
            child = f"{prefix}.{key}" if prefix else key
            if key not in left or key not in right:
                result.append(child)
            else:
                result.extend(_ledger_diff_paths(left[key], right[key], child))
        return result
    if isinstance(left, list):
        result = []
        if len(left) != len(right):
            result.append(f"{prefix}.length")
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            result.extend(_ledger_diff_paths(left_item, right_item, f"{prefix}[{index}]"))
        return result
    return [] if left == right else [prefix or "<root>"]


def run_scenario(
    *,
    source_project: str | Path,
    work_dir: str | Path,
    commands_path: str | Path,
    out_dir: str | Path,
) -> dict[str, Any]:
    source = Path(source_project).resolve()
    work = Path(work_dir).resolve()
    out = Path(out_dir).resolve()
    _copy_project(source, work)
    out.mkdir(parents=True, exist_ok=True)
    commands_raw = read_json(Path(commands_path))
    commands = commands_raw.get("commands", []) if isinstance(commands_raw, dict) else []
    cursor = 0
    incremental_runs: list[dict[str, Any]] = []
    generated_rows: list[dict[str, Any]] = []
    engine = LedgerEngine(work, state_dir=".state")
    for command in commands:
        _, rows = _apply_mutation(work, command)
        generated_rows.extend(rows)
        cursor += 1
        engine = LedgerEngine(work, state_dir=".state", command_cursor=cursor)
        result = engine.build(resume=True, save_snapshot=True)
        incremental_runs.append(
            {
                "cursor": cursor,
                "command": command.get("type"),
                "ledger_version": result.ledger_version,
                "global_version": result.build_info["global_version"],
                "resumed": result.build_info["resumed"],
            }
        )
    final_incremental = engine.build(resume=True, save_snapshot=True)
    full_engine = LedgerEngine(work, state_dir=".state-full", command_cursor=cursor)
    final_full = full_engine.build(resume=False, save_snapshot=True)
    atomic_write_json(out / "incremental-ledger.json", final_incremental.ledger)
    atomic_write_json(out / "full-ledger.json", final_full.ledger)
    differences = _ledger_diff_paths(final_incremental.ledger, final_full.ledger)
    report = {
        "schema_version": 1,
        "commands": cursor,
        "incremental_runs": incremental_runs,
        "incremental_ledger_version": final_incremental.ledger_version,
        "full_ledger_version": final_full.ledger_version,
        "identical": not differences,
        "differences": differences,
        "summary": final_incremental.ledger["summary"],
        "invalidation": final_incremental.ledger["invalidation"],
        "failures": final_incremental.ledger["failures"],
    }
    atomic_write_json(out / "comparison.json", report)
    return report


def compare_current_with_full(
    *,
    project_dir: str | Path,
    incremental_state: str | Path,
    full_state: str | Path,
) -> dict[str, Any]:
    project = Path(project_dir).resolve()
    incremental = LedgerEngine(project, state_dir=incremental_state).build(resume=True)
    full = LedgerEngine(project, state_dir=full_state).build(resume=False)
    differences = _ledger_diff_paths(incremental.ledger, full.ledger)
    return {
        "identical": not differences,
        "differences": differences,
        "incremental_ledger_version": incremental.ledger_version,
        "full_ledger_version": full.ledger_version,
    }
