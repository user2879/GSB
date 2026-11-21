from __future__ import annotations

import copy
import json
import os
import shutil
import uuid
from pathlib import Path
from typing import Any

from .canonical import canonical_json, sha256_value
from .diagnostics import run_injection_diagnostics
from .engine import (
    Engine,
    EngineOptions,
    canonical_sentence_payload,
)
from .loader import InputBundle, load_state
from .model import Failure, STABLE_SCHEMA_VERSION, stable_failures
from .snapshot import SnapshotManager


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}-{uuid.uuid4().hex}.tmp"
    with temporary.open("wb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _write_json(path: Path, value: Any) -> None:
    _atomic_write(
        path,
        (
            json.dumps(
                value,
                ensure_ascii=False,
                sort_keys=True,
                indent=2,
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8"),
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "".join(
        canonical_json(row) + "\n" for row in rows
    ).encode("utf-8")
    _atomic_write(path, content)


def _snapshot_failure(reason: str, manifest: dict[str, Any] | None) -> Failure:
    return Failure(
        code="E_SNAPSHOT_STALE",
        phase="snapshot_recovery",
        message=f"snapshot not resumed: {reason}",
        file="manifest.json",
        active=False,
        details={
            "reason": reason,
            "manifest_global_version": (
                manifest.get("global_version") if manifest else None
            ),
            "recovered_by_fresh_rebuild": True,
        },
        order=(0, 0, "E_SNAPSHOT_STALE", ""),
    )


def _state_failure_projection(failures: list[Failure]) -> list[dict[str, Any]]:
    excluded = {
        "E_EDIT_INVALID",
        "E_SNAPSHOT_STALE",
        "E_INJECTION_UNKNOWN",
    }
    projected = [
        {
            "code": failure.code,
            "phase": failure.phase,
            "file": failure.file,
            "line": failure.line,
            "sentence_id": failure.sentence_id,
            "entry_key": failure.entry_key,
            "message": failure.message,
            "details": failure.details,
        }
        for failure in stable_failures(failures)
        if failure.active and failure.code not in excluded
    ]
    return projected


def _merge_load_failures(engine: Engine, bundle: InputBundle) -> None:
    seen = {
        (
            failure.code,
            failure.sentence_id,
            failure.entry_key,
            failure.file,
            failure.line,
        )
        for failure in engine.failures
    }
    for failure in bundle.sorted_failures():
        if not failure.phase.startswith("load_") or not failure.active:
            continue
        key = (
            failure.code,
            failure.sentence_id,
            failure.entry_key,
            failure.file,
            failure.line,
        )
        if key not in seen:
            engine.failures.append(failure)
            seen.add(key)
    engine.failures = stable_failures(engine.failures)


def _compare_with_full(
    incremental: Engine,
    full: Engine,
) -> dict[str, Any]:
    incremental_rows = {
        row["id"]: canonical_sentence_payload(row)
        for row in incremental.sentence_rows()
    }
    full_rows = {
        row["id"]: canonical_sentence_payload(row) for row in full.sentence_rows()
    }
    sentence_ids_equal = sorted(incremental_rows) == sorted(full_rows)
    sentence_fields_equal = sentence_ids_equal and all(
        incremental_rows[key] == full_rows[key] for key in incremental_rows
    )

    incremental_active = _state_failure_projection(incremental.active_failures())
    full_active = _state_failure_projection(full.active_failures())
    active_failures_equal = incremental_active == full_active

    incremental_ledger = incremental.canonical_ledger_rows()
    full_ledger = full.canonical_ledger_rows()
    ledger_order_equal = [row["stable_key"] for row in incremental_ledger] == [
        row["stable_key"] for row in full_ledger
    ]
    ledger_static_fields_equal = incremental_ledger == full_ledger

    mismatched_sentences = []
    if sentence_ids_equal:
        mismatched_sentences = [
            key for key in sorted(incremental_rows) if incremental_rows[key] != full_rows[key]
        ]
    mismatch_details = {}
    for key in mismatched_sentences:
        left = incremental_rows[key]
        right = full_rows[key]
        mismatch_details[key] = {
            field: {"incremental": left.get(field), "full": right.get(field)}
            for field in sorted(set(left) | set(right))
            if left.get(field) != right.get(field)
        }
    return {
        "sentence_payloads_equal": sentence_fields_equal,
        "sentence_order_equal": sentence_ids_equal,
        "mismatched_sentence_ids": mismatched_sentences,
        "mismatch_details": mismatch_details,
        "active_failures_equal": active_failures_equal,
        "active_failure_count_incremental": len(incremental_active),
        "active_failure_count_full": len(full_active),
        "ledger_order_equal": ledger_order_equal,
        "ledger_static_fields_equal": ledger_static_fields_equal,
        "ledger_static_sha256_incremental": sha256_value(incremental_ledger),
        "ledger_static_sha256_full": sha256_value(full_ledger),
        "runtime_reuse_accounting_note": (
            "Incremental and full runs intentionally differ only in runtime reuse/"
            "recompute counters and charged points; canonical translation payloads, "
            "active failures and static ledger fields/order must match."
        ),
        "all_required_parity_checks_pass": (
            sentence_fields_equal
            and active_failures_equal
            and ledger_order_equal
            and ledger_static_fields_equal
        ),
    }


def run_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    drain: bool = True,
    compare_full: bool = True,
    reset: bool = False,
) -> dict[str, Any]:
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if reset and output_path.exists():
        for child in output_path.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    output_path.mkdir(parents=True, exist_ok=True)

    bundle = load_state(input_path)
    injection_report = run_injection_diagnostics(bundle)

    if bundle.state is None:
        result = {
            "schema_version": STABLE_SCHEMA_VERSION,
            "mode": "incremental",
            "status": "INVALID_INPUT",
            "summary": {
                "sentence_count": 0,
                "active_failure_count": sum(
                    1 for failure in bundle.failures if failure.active
                ),
            },
            "sentences": [],
            "ledger": [],
            "canonical_ledger": [],
            "failures": [failure.to_dict() for failure in bundle.sorted_failures()],
            "diagnostics": injection_report,
            "parity": None,
        }
        _write_json(output_path / "result.json", result)
        _write_jsonl(output_path / "sentences.jsonl", [])
        _write_jsonl(output_path / "ledger.jsonl", [])
        _write_jsonl(output_path / "canonical_ledger.jsonl", [])
        _write_json(
            output_path / "failures.json",
            [failure.to_dict() for failure in bundle.sorted_failures()],
        )
        _write_json(output_path / "diagnostics.json", {"injections": injection_report})
        _write_json(output_path / "parity.json", {"ran": False, "status": "INVALID_INPUT"})
        return result

    manager = SnapshotManager(output_path)
    manifest = manager.read_manifest()
    load_result = manager.veto_if_stale(
        manifest,
        input_hashes=bundle.input_hashes,
        commands=bundle.commands,
    )
    snapshot_stale = load_result.stale
    stale_reason = load_result.stale_reason
    snapshot_failure: Failure | None = None
    engine: Engine | None = None
    command_cursor = 0
    resumed = False
    if manifest is None:
        engine = Engine(bundle.state, failures=bundle.sorted_failures())
    elif load_result.stale:
        snapshot_failure = _snapshot_failure(
            load_result.stale_reason or "snapshot validation failed",
            load_result.manifest,
        )
        stale_reason = load_result.stale_reason or "snapshot validation failed"
        engine = Engine(
            copy.deepcopy(bundle.state),
            failures=bundle.sorted_failures() + [snapshot_failure],
        )
    else:
        command_cursor = load_result.command_cursor
        engine = manager.restore(load_result)
        if engine is not None:
            _merge_load_failures(engine, bundle)
            resumed = True
        else:
            snapshot_stale = True
            stale_reason = "artifact restoration failed"
            snapshot_failure = _snapshot_failure(
                "artifact restoration failed", load_result.manifest
            )
            engine = Engine(
                copy.deepcopy(bundle.state),
                failures=bundle.sorted_failures() + [snapshot_failure],
            )
            command_cursor = 0

    assert engine is not None
    meta = {"cursor": command_cursor}

    def snapshot_callback(current_engine: Engine) -> None:
        manager.commit(
            current_engine,
            input_hashes=bundle.input_hashes,
            commands=bundle.commands,
            command_cursor=meta["cursor"],
        )

    engine.options.snapshot_callback = snapshot_callback
    engine._sync_records()

    if not resumed:
        engine.run_to_idle()
        meta["cursor"] = 0
    for index in range(command_cursor, len(bundle.commands)):
        engine.apply_command(bundle.commands[index], index)
        meta["cursor"] = index + 1
        engine.run_cycle()
    if drain:
        engine.run_to_idle()
    meta["cursor"] = len(bundle.commands)
    manager.commit(
        engine,
        input_hashes=bundle.input_hashes,
        commands=bundle.commands,
        command_cursor=len(bundle.commands),
    )

    parity: dict[str, Any] | None = None
    full_engine: Engine | None = None
    if compare_full:
        full_failures = [
            failure
            for failure in bundle.sorted_failures()
            if failure.phase.startswith("load_")
        ]
        full_engine = Engine(
            copy.deepcopy(engine.state),
            failures=full_failures,
            options=EngineOptions(snapshot_callback=None, max_cycles=10_000),
        )
        full_engine.run_to_idle()
        parity = _compare_with_full(engine, full_engine)

    rows = engine.sentence_rows()
    ledger_rows = engine.ledger_rows()
    canonical_ledger_rows = engine.canonical_ledger_rows()
    failures = [failure.to_dict() for failure in engine.all_failures()]
    active_failures = [
        failure.to_dict() for failure in engine.active_failures()
    ]
    diagnostics = {
        "schema_version": STABLE_SCHEMA_VERSION,
        "injections": injection_report,
        "snapshot": {
            "manifest_present_at_start": manifest is not None,
            "resumed": resumed,
            "stale": snapshot_stale,
            "stale_code": "E_SNAPSHOT_STALE" if snapshot_stale else None,
            "stale_reason": stale_reason if snapshot_stale else None,
            "command_cursor_at_start": command_cursor,
            "command_cursor_final": len(bundle.commands),
            "manifest_version": manager.read_manifest().get("global_version")
            if manager.read_manifest()
            else None,
        },
    }
    result = {
        "schema_version": STABLE_SCHEMA_VERSION,
        "mode": "incremental",
        "status": "OK",
        "input": {
            "directory": str(input_path),
            "input_hashes": bundle.input_hashes,
            "command_stream_hash": bundle.command_stream_hash,
            "command_count": len(bundle.commands),
        },
        "summary": engine.summary(),
        "sentences": rows,
        "ledger": ledger_rows,
        "canonical_ledger": canonical_ledger_rows,
        "runtime_ledger": {
            "total_charged_points": engine.total_charged_points,
            "batch_cumulative_points": {
                str(key): value
                for key, value in sorted(engine.batch_cumulative_points.items())
            },
            "events": engine.ledger_events,
            "runs": [report.to_dict() for report in engine.run_reports],
        },
        "failures": failures,
        "active_failures": active_failures,
        "diagnostics": diagnostics,
        "snapshot_manifest": manager.read_manifest(),
        "parity": parity,
    }
    _write_json(output_path / "result.json", result)
    _write_jsonl(output_path / "sentences.jsonl", rows)
    _write_jsonl(output_path / "ledger.jsonl", ledger_rows)
    _write_jsonl(output_path / "canonical_ledger.jsonl", canonical_ledger_rows)
    _write_json(output_path / "failures.json", failures)
    _write_json(output_path / "active_failures.json", active_failures)
    _write_json(output_path / "diagnostics.json", diagnostics)
    _write_json(
        output_path / "parity.json",
        parity or {"ran": False},
    )
    return result
