"""End-to-end deterministic demo and lease reclamation evidence."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .errors import BusinessError, E_LEASE_EXPIRED, E_LEASE_GENERATION
from .lease import LocalExecutor, ManualClock
from .renderer import Renderer, RenderResult
from .samples import create_sample_workspace
from .util import atomic_write_text, canonical_json, digest_obj
from .workspace import Workspace, load_commands


def lease_recovery_demo(params: dict[str, Any] | None = None) -> dict[str, Any]:
    params = {
        "ttl_ms": 250,
        "initial_now_ms": 1000,
        "advance_ms_before_renew": 150,
        "advance_ms_to_expiry": 300,
        "replacement_ttl_ms": 400,
        **(params or {}),
    }
    clock = ManualClock(int(params["initial_now_ms"]))
    executor = LocalExecutor(clock=clock)
    lease = executor.acquire([0, 1, 2], int(params["ttl_ms"]))
    initial_generation = lease.generation
    occupied_after_acquire = executor.occupied_units()

    clock.advance(int(params["advance_ms_before_renew"]))
    renewed = executor.renew(lease.lease_id, lease.generation, int(params["ttl_ms"]))
    generation_after_renew = executor.generation

    generation_mismatch = None
    try:
        executor.renew(lease.lease_id, lease.generation + 99, int(params["ttl_ms"]))
    except BusinessError as exc:
        generation_mismatch = {"code": exc.code, "generation_after_reject": executor.generation,
                               "occupied_units_after_reject": executor.occupied_units()}

    clock.advance(int(params["advance_ms_to_expiry"]))
    expired_events = [event.as_dict() for event in executor.reclaim_expired()]
    occupied_after_expiry = executor.occupied_units()
    generation_after_expiry = executor.generation

    replacement = executor.acquire([0, 1], int(params["replacement_ttl_ms"]))
    replacement_occupied = executor.occupied_units()
    wrong_release = None
    try:
        executor.release(replacement.lease_id, replacement.generation + 5)
    except BusinessError as exc:
        wrong_release = {"code": exc.code, "generation_after_reject": executor.generation,
                         "occupied_units_after_reject": executor.occupied_units()}
    released = executor.release(replacement.lease_id, replacement.generation)
    final_occupied = executor.occupied_units()
    expectations = {
        "renew_preserves_generation": renewed.generation == initial_generation
        and generation_after_renew == initial_generation,
        "generation_mismatch_rejected": generation_mismatch is not None
        and generation_mismatch["code"] == E_LEASE_GENERATION,
        "generation_mismatch_does_not_advance": generation_mismatch is not None
        and generation_mismatch["generation_after_reject"] == initial_generation,
        "expiry_emits_code": any(event["code"] == E_LEASE_EXPIRED for event in expired_events),
        "expiry_really_reclaims": occupied_after_expiry == [],
        "replacement_generation_increments": replacement.generation > initial_generation,
        "wrong_release_rejected": wrong_release is not None
        and wrong_release["code"] == E_LEASE_GENERATION,
        "release_reclaims_units": final_occupied == [],
    }
    return {
        "ok": all(expectations.values()),
        "expectations": expectations,
        "initial_generation": initial_generation,
        "occupied_after_acquire": occupied_after_acquire,
        "generation_mismatch": generation_mismatch,
        "expired_events": expired_events,
        "occupied_after_expiry": occupied_after_expiry,
        "generation_after_expiry": generation_after_expiry,
        "replacement_generation": replacement.generation,
        "replacement_occupied": replacement_occupied,
        "wrong_release": wrong_release,
        "released": released,
        "final_occupied": final_occupied,
        "events": executor.events(),
    }


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise BusinessError("E_LEDGER_JSON", "invalid ledger line",
                                {"path": str(path), "line": line_number}) from exc
    return entries


def _summarize_run(ledger_path: Path, run_id: str) -> dict[str, Any]:
    entries = [entry for entry in _read_ledger(ledger_path) if entry.get("run_id") == run_id]
    tasks = [entry for entry in entries if entry.get("event") == "task"]
    outcomes: dict[str, int] = {}
    basis: dict[str, int] = {}
    for task in tasks:
        outcome = str(task.get("outcome"))
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        for item in task.get("invalidation_basis", []):
            basis[str(item)] = basis.get(str(item), 0) + 1
    batches = [entry for entry in entries if entry.get("event") == "scheduler_batch"]
    return {
        "tasks": len(tasks),
        "outcomes": dict(sorted(outcomes.items())),
        "invalidation_basis": dict(sorted(basis.items())),
        "batches": len(batches),
    }


def _frames_equal(incremental: RenderResult, full: RenderResult) -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {"frame_pixels": True, "png_files": True, "blocks": True,
                               "task_keys": incremental.task_keys == full.task_keys,
                               "scheduler_selected_order": [batch["selected"] for batch in incremental.scheduler_batches]
                               == [batch["selected"] for batch in full.scheduler_batches],
                               "scheduler_pending_order": [batch["pending"] for batch in incremental.scheduler_batches]
                               == [batch["pending"] for batch in full.scheduler_batches]}
    if set(incremental.frames) != set(full.frames):
        details["frame_pixels"] = False
        return False, details
    for frame in sorted(incremental.frames):
        left = incremental.frames[frame]
        right = full.frames[frame]
        if (left["pixel_digest"] != right["pixel_digest"]
                or left["png_digest"] != right["png_digest"]
                or len(left["blocks"]) != len(right["blocks"])):
            details["frame_pixels"] = False
        if incremental.output_dir and full.output_dir:
            left_png = Path(incremental.output_dir) / "frames" / f"frame_{frame:04d}.png"
            right_png = Path(full.output_dir) / "frames" / f"frame_{frame:04d}.png"
            if not left_png.exists() or not right_png.exists() or left_png.read_bytes() != right_png.read_bytes():
                details["png_files"] = False
        for left_block, right_block in zip(left["blocks"], right["blocks"]):
            if (left_block["row_start"] != right_block["row_start"]
                    or left_block["row_end"] != right_block["row_end"]
                    or left_block["cache_key"] != right_block["cache_key"]
                    or left_block["pixel_digest"] != right_block["pixel_digest"]
                    or left_block["png_digest"] != right_block["png_digest"]):
                details["blocks"] = False
    details["all"] = all(value for value in details.values())
    return bool(details["all"]), details


def run_demo(root: str | Path = "samples", *, regenerate: bool = True) -> dict[str, Any]:
    sample_root = Path(root).resolve()
    if regenerate:
        create_sample_workspace(sample_root)
    demo_root = (sample_root / "demo").resolve()
    if demo_root.exists():
        shutil.rmtree(demo_root)
    demo_root.mkdir(parents=True, exist_ok=True)
    commands = load_commands(sample_root / "commands.json")
    incremental_cache = demo_root / "incremental_cache"
    full_cache = demo_root / "full_cache"
    incremental_ledger = demo_root / "ledger" / "incremental.jsonl"
    full_ledger = demo_root / "ledger" / "full.jsonl"
    incremental_out = demo_root / "incremental_output"
    full_out = demo_root / "full_output"
    workspace = Workspace.load(sample_root / "config.json")
    incremental = Renderer(workspace, cache_dir=incremental_cache, ledger_path=incremental_ledger)
    edit_reports: list[dict[str, Any]] = []
    try:
        initial = incremental.run(output_dir=incremental_out)
        edit_reports.append({
            "edit": "initial",
            "result": initial.as_dict(),
            "ledger": _summarize_run(incremental_ledger, initial.run_id),
        })
        for index, command in enumerate(commands, start=1):
            workspace = workspace.apply_command(command)
            incremental.workspace = workspace
            result = incremental.run(output_dir=incremental_out)
            edit_reports.append({
                "edit": index,
                "command": command,
                "result": result.as_dict(),
                "ledger": _summarize_run(incremental_ledger, result.run_id),
            })
        final_incremental = result
        final_ledger = _summarize_run(incremental_ledger, final_incremental.run_id)

        full_workspace = Workspace.load(sample_root / "config.json")
        for command in commands:
            full_workspace = full_workspace.apply_command(command)
        full_renderer = Renderer(full_workspace, cache_dir=full_cache, ledger_path=full_ledger)
        try:
            final_full = full_renderer.run(output_dir=full_out)
        finally:
            full_renderer.close()
    finally:
        incremental.close()

    equal, comparison = _frames_equal(final_incremental, final_full)
    lease = lease_recovery_demo(_read_lease_params(sample_root / "lease.json"))
    report = {
        "ok": bool(equal and lease["ok"] and final_incremental.complete and final_full.complete),
        "sample_root": str(sample_root),
        "demo_root": str(demo_root),
        "initial": edit_reports[0],
        "edits": edit_reports[1:],
        "final_incremental": final_incremental.as_dict(),
        "final_full": final_full.as_dict(),
        "final_incremental_ledger": final_ledger,
        "comparison": comparison,
        "lease_recovery": lease,
        "incremental_cache_keys_sha256": digest_obj(final_incremental.task_keys),
        "full_cache_keys_sha256": digest_obj(final_full.task_keys),
        "max_observed_workers_overall": max(
            item["result"]["max_observed_workers"] for item in edit_reports
        ),
    }
    atomic_write_text(demo_root / "demo_report.json", json.dumps(report, indent=2, sort_keys=True) + "\n")
    atomic_write_text(demo_root / "results.jsonl",
                      "".join(canonical_json(item) + "\n" for item in edit_reports))
    return report


def _read_lease_params(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise BusinessError("E_LEASE_CONFIG", "lease params must be an object", {"path": str(path)})
    return raw
