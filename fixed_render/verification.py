from __future__ import annotations

import json
import os
import random
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .config import RenderConfig
from .diagnostics import run_diagnostics
from .errors import InputInvalid
from .hashing import canonical_json
from .inputs import load_command_stream, load_png_condition, load_conditions
from .lease import LeaseClient, LeaseServer
from .samples import generate_all_samples
from .storage import atomic_write_json
from .workbench import CommandSession


def run_verification(root: Path, report_path: Path) -> dict[str, Any]:
    root = root.resolve()
    generate_all_samples(root)
    base_config = RenderConfig.load(root / "configs" / "render.yaml")
    checks: list[dict[str, Any]] = []

    validation = verify_invalid_inputs(base_config)
    checks.append({"name": "whole_file_input_validation", **validation})

    preview_startup = verify_preview_and_startup(root, base_config)
    checks.append(preview_startup)

    canonical_commands = load_command_stream(root / "commands" / "edit_sequence.yaml")
    canonical = verify_sequence_equivalence(
        root,
        base_config,
        canonical_commands,
        name="canonical_nine_edits",
        workspace_root=root / "runs" / "verification",
    )
    checks.append(canonical)

    random_cases: list[dict[str, Any]] = []
    rng = random.Random(987654321)
    for case_index in range(5):
        commands = make_random_commands(rng, case_index)
        result = verify_sequence_equivalence(
            root,
            base_config,
            commands,
            name=f"random_sequence_{case_index:02d}",
            workspace_root=root / "runs" / "verification",
        )
        random_cases.append(result)
        checks.append(result)

    concurrency = verify_concurrency(root, base_config)
    checks.append(concurrency)

    kill_recovery = verify_sigkill_recovery(root, base_config)
    checks.append(kill_recovery)

    diagnostics_report_path = root / "artifacts" / "diagnostic_report.json"
    diagnostics = run_diagnostics(root, diagnostics_report_path)
    checks.append(
        {
            "name": "fault_injection_diagnostics",
            "ok": diagnostics["summary"]["failed_cases"] == 0,
            "hit": diagnostics["summary"]["hit_injections"],
            "total": diagnostics["summary"]["total_injections"],
            "report": str(diagnostics_report_path),
        }
    )

    ok = all(bool(check.get("ok")) for check in checks)
    report = {
        "report_type": "fixed_render_verification",
        "ok": ok,
        "summary": {
            "ok": ok,
            "check_count": len(checks),
            "passed_count": sum(1 for check in checks if check.get("ok")),
            "pixel_equivalence_cases": 1 + len(random_cases),
            "lease_real_reclamation": diagnostics["lease_reclamation"]["slot_released"],
            "sigkill_no_exit_report": kill_recovery.get("no_exit_report_after_sigkill"),
            "concurrent_cases": concurrency.get("workers"),
            "diagnostic_report": str(diagnostics_report_path),
        },
        "checks": checks,
        "diagnostic_summary": diagnostics["summary"],
    }
    report_path = Path(report_path)
    if not report_path.is_absolute():
        report_path = root / report_path
    atomic_write_json(report_path, report)
    return report


def verify_sequence_equivalence(
    root: Path,
    base_config: RenderConfig,
    commands: list[dict[str, Any]],
    name: str,
    workspace_root: Path,
) -> dict[str, Any]:
    incremental_ws = workspace_root / f"{name}_incremental"
    full_ws = workspace_root / f"{name}_full_blank"
    _reset_dir(incremental_ws)
    _reset_dir(full_ws)

    started = time.perf_counter()
    with CommandSession(root, incremental_ws, base_config) as session:
        session.validate_command_stream(commands)
        session.render_current(ledger_name="00_baseline")
        step_reports = []
        last_result = None
        for index, command in enumerate(commands, 1):
            result = session.apply_and_render(command, ledger_name=f"{index:02d}_{command['op']}")
            last_result = result
            step_reports.append(
                {
                    "index": index,
                    "op": command["op"],
                    "recompute_order": result["manifest"]["recompute_order"],
                    "reuse_order": result["manifest"]["reuse_order"],
                    "preview_code": result["preview"]["code"],
                }
            )
        final_config = session.config
        incremental_manifest = session.engine.snapshots.current()
        incremental_ledger = last_result["ledger"]

    with CommandSession(root, full_ws, final_config) as full_session:
        full_result = full_session.render_current(ledger_name="full_blank")
        full_manifest = full_result["manifest"]

    comparison = compare_manifests(
        incremental_manifest,
        incremental_ws,
        full_manifest,
        full_ws,
    )
    canonical_order = incremental_manifest["stage_order"]
    full_order = full_manifest["recompute_order"]
    order_ok = full_order == canonical_order
    incremental_recompute_order_ok = _is_subsequence(
        incremental_manifest["recompute_order"], canonical_order
    )
    incremental_reuse_order_ok = _is_subsequence(
        incremental_manifest["reuse_order"], canonical_order
    )
    entry_order_ok = [entry["node_id"] for entry in incremental_ledger["entries"]] == canonical_order
    ok = (
        comparison["ok"]
        and order_ok
        and incremental_recompute_order_ok
        and incremental_reuse_order_ok
        and entry_order_ok
        and not incremental_manifest["failed_order"]
    )
    return {
        "name": name,
        "ok": ok,
        "command_count": len(commands),
        "incremental_workspace": str(incremental_ws),
        "full_workspace": str(full_ws),
        "pixel_frame_count": comparison["frame_count"],
        "pixel_fingerprints": comparison["pixel_fingerprints"],
        "differences": comparison["differences"],
        "stage_summary_mismatches": comparison["stage_summary_mismatches"],
        "canonical_order_matches_full_recompute": order_ok,
        "incremental_recompute_relative_order_matches": incremental_recompute_order_ok,
        "incremental_reuse_relative_order_matches": incremental_reuse_order_ok,
        "ledger_entry_order_matches": entry_order_ok,
        "step_reports": step_reports,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
    }


def compare_manifests(
    incremental_manifest: dict[str, Any],
    incremental_ws: Path,
    full_manifest: dict[str, Any],
    full_ws: Path,
) -> dict[str, Any]:
    differences: list[str] = []
    summary_mismatches: list[str] = []
    lease_key_differences: list[str] = []
    if incremental_manifest["stage_order"] != full_manifest["stage_order"]:
        differences.append("stage_order")

    for node_id in incremental_manifest["stage_order"]:
        left = incremental_manifest["nodes"].get(node_id, {})
        right = full_manifest["nodes"].get(node_id, {})
        if left.get("content_fingerprint") != right.get("content_fingerprint"):
            summary_mismatches.append(f"{node_id}:content_fingerprint")
        if left.get("content_key") != right.get("content_key"):
            summary_mismatches.append(f"{node_id}:content_key")
        if left.get("cache_key") != right.get("cache_key"):
            lease_key_differences.append(f"{node_id}:cache_key")
        if canonical_json(left.get("trace", {})) != canonical_json(right.get("trace", {})):
            summary_mismatches.append(f"{node_id}:trace")

    fingerprints: dict[str, str] = {}
    frame_count = len(incremental_manifest["frames"])
    for frame in incremental_manifest["frames"]:
        index = frame["frame_index"]
        rel_path = frame.get("png_path") or f"frames/frame_{index:02d}.png"
        left_path = incremental_ws / rel_path
        right_path = full_ws / rel_path
        if not left_path.exists() or not right_path.exists():
            differences.append(f"frame_{index}:missing_png")
            continue
        left_pixels = np.asarray(Image.open(left_path).convert("RGB"))
        right_pixels = np.asarray(Image.open(right_path).convert("RGB"))
        if left_pixels.shape != right_pixels.shape or not np.array_equal(left_pixels, right_pixels):
            differences.append(f"frame_{index}:pixels")
        fingerprints[str(index)] = frame.get("pixel_fingerprint")

        left_sidecar = json.loads((incremental_ws / "frames" / f"frame_{index:02d}.json").read_text())
        right_sidecar = json.loads((full_ws / "frames" / f"frame_{index:02d}.json").read_text())
        if left_sidecar["condition_fingerprint"] != right_sidecar["condition_fingerprint"]:
            summary_mismatches.append(f"frame_{index}:condition_fingerprint")
        if left_sidecar["steps"] != right_sidecar["steps"]:
            summary_mismatches.append(f"frame_{index}:step_summary")

    return {
        "ok": not differences and not summary_mismatches,
        "frame_count": frame_count,
        "pixel_fingerprints": fingerprints,
        "differences": differences,
        "stage_summary_mismatches": summary_mismatches,
        "lease_cache_key_differences": lease_key_differences,
    }


def verify_invalid_inputs(config: RenderConfig) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []

    def expect_field(name: str, callback: Any, field_fragment: str) -> None:
        try:
            callback()
            cases.append({"case": name, "ok": False, "reason": "did not reject"})
        except InputInvalid as exc:
            fields = " ".join(exc.fields.keys()) + " " + str(exc.field_name or "")
            cases.append(
                {
                    "case": name,
                    "ok": field_fragment in fields,
                    "code": exc.code,
                    "fields": exc.fields,
                }
            )

    expect_field(
        "unknown_prompt_term",
        lambda: load_conditions(config.with_changes({"prompt_tokens": ("not_controlled",)})),
        "conditions.prompt_tokens",
    )
    expect_field(
        "text_dimension_mismatch",
        lambda: load_conditions(config.with_changes({"text_dim": 999})),
        "text_dim",
    )
    expect_field(
        "tile_height_not_divisor",
        lambda: RenderConfig.from_mapping(
            {**config.to_mapping(), "tile_height": 7}, base_dir=config.base_dir
        ),
        "tile_height",
    )
    expect_field(
        "quant_enum_invalid",
        lambda: RenderConfig.from_mapping(
            {**config.to_mapping(), "fixed_frac_bits": 77}, base_dir=config.base_dir
        ),
        "fixed_frac_bits",
    )

    with tempfile.TemporaryDirectory() as temporary:
        bad_png = Path(temporary) / "bad.png"
        bad_png.write_bytes(b"not a png")
        expect_field(
            "png_bad_signature",
            lambda: load_png_condition(bad_png, config.image_width, config.image_height),
            "png_signature",
        )
        small_png = Path(temporary) / "small.png"
        Image.new("RGB", (31, 31), (1, 2, 3)).save(small_png)
        expect_field(
            "png_wrong_resolution",
            lambda: load_png_condition(small_png, config.image_width, config.image_height),
            "image_resolution",
        )

        motion = Path(temporary) / "motion.csv"
        motion.write_text(
            "block_row,block_col,dx_q8,dy_q8\n0,0,1,1\n", encoding="utf-8"
        )
        expect_field(
            "motion_block_count",
            lambda: load_conditions(config.with_changes({"motion_path": str(motion)})),
            "motion_block_count",
        )

    return {"ok": all(case.get("ok") for case in cases), "cases": cases}


def verify_preview_and_startup(root: Path, base_config: RenderConfig) -> dict[str, Any]:
    fresh_ws = root / "runs" / "verification" / "preview_fresh"
    stale_ws = root / "runs" / "verification" / "preview_stale_budget_zero"
    _reset_dir(fresh_ws)
    _reset_dir(stale_ws)
    command = {"op": "set_prompt", "tokens": ["entity_grid", "attr_yellow", "quality_sharp"]}

    fresh_config = base_config.with_changes({"preview_budget_ms": 60000})
    with CommandSession(root, fresh_ws, fresh_config) as session:
        bundle = load_conditions(session.config)
        before = session.engine.startup_decision(session.config, bundle)["decision"]
        session.render_current(ledger_name="baseline")
        resume = session.engine.startup_decision(
            session.config, load_conditions(session.config)
        )["decision"]
        result = session.apply_and_render(command, ledger_name="preview_command")
        fresh_preview = result["preview"]
        changed_config = session.config.with_changes({"denoise_steps": session.config.denoise_steps + 1})
        replan = session.engine.startup_decision(
            changed_config, load_conditions(changed_config)
        )["decision"]

    stale_config = base_config.with_changes({"preview_budget_ms": 0})
    with CommandSession(root, stale_ws, stale_config) as session:
        session.render_current(ledger_name="baseline")
        stale_result = session.apply_and_render(command, ledger_name="stale_preview_command")
        stale_preview = stale_result["preview"]

    ok = (
        before == "fresh_build"
        and resume == "resume"
        and replan == "replan_with_content_cache"
        and fresh_preview["code"] == "E_OK"
        and fresh_preview["fresh"] is True
        and fresh_preview["frame_fingerprints"] == result["manifest"]["frame_fingerprints"]
        and stale_preview["code"] == "E_PREVIEW_STALE"
        and stale_preview["based_on_version"] == 1
        and stale_preview["latest_version"] == 2
    )
    return {
        "name": "preview_budget_and_startup_decision",
        "ok": ok,
        "startup_before": before,
        "startup_same_inputs": resume,
        "startup_changed_inputs": replan,
        "fresh_preview": fresh_preview,
        "zero_budget_preview": stale_preview,
    }


def verify_concurrency(root: Path, config: RenderConfig, workers: int = 4) -> dict[str, Any]:
    work_root = root / "runs" / "verification" / "concurrency"
    _reset_dir(work_root)
    with LeaseServer("127.0.0.1", 0, max_slots=workers) as server:
        client = LeaseClient("127.0.0.1", server.port)

        def one(index: int) -> dict[str, Any]:
            workspace = work_root / f"worker_{index}"
            local_config = config.with_changes({"seed": config.seed + index})
            # Different seeds intentionally produce independent pixels; compare each to blank.
            record = client.acquire(
                lease_key=f"concurrent-{index}",
                scopes=["all-stages"],
                ttl_ms=30000,
            )
            lease_info = {
                "lease_id": record["lease_id"],
                "generation": record["generation"],
                "expires_ms": record["expires_ms"],
            }
            from .engine import RenderEngine

            engine = RenderEngine(workspace)
            bundle = load_conditions(local_config)
            result = engine.render(local_config, bundle, client, lease_info, ledger_name="concurrent")
            blank_ws = work_root / f"blank_{index}"
            with CommandSession(root, blank_ws, local_config) as blank_session:
                blank = blank_session.render_current(ledger_name="blank")["manifest"]
            comparison = compare_manifests(result["manifest"], workspace, blank, blank_ws)
            return {"worker": index, "ok": result["manifest"]["ok"] and comparison["ok"], **comparison}

        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(one, range(workers)))
    return {
        "name": "shared_loopback_executor_concurrency",
        "ok": all(item["ok"] for item in results),
        "workers": workers,
        "results": [{"worker": item["worker"], "ok": item["ok"], "differences": item["differences"]} for item in results],
    }


def verify_sigkill_recovery(root: Path, config: RenderConfig) -> dict[str, Any]:
    workspace = root / "runs" / "verification" / "sigkill_recovery"
    _reset_dir(workspace)
    ready_file = workspace / "worker_ready.json"
    exit_report = workspace / "exit_report.json"
    with LeaseServer("127.0.0.1", 0, max_slots=1) as server:
        client = LeaseClient("127.0.0.1", server.port)
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "fixed_render.worker_demo",
                "--host",
                "127.0.0.1",
                "--port",
                str(server.port),
                "--workspace",
                str(workspace),
                "--ready-file",
                str(ready_file),
                "--exit-report",
                str(exit_report),
                "--ttl-ms",
                "100",
            ],
            cwd=str(root),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        ready: dict[str, Any] | None = None
        deadline = time.time() + 8
        while time.time() < deadline:
            if ready_file.exists():
                ready = json.loads(ready_file.read_text())
                break
            if process.poll() is not None:
                stdout, stderr = process.communicate(timeout=1)
                return {"ok": False, "reason": "worker exited before ready", "stdout": stdout, "stderr": stderr}
            time.sleep(0.02)
        if ready is None:
            process.kill()
            return {"ok": False, "reason": "worker ready timeout"}

        reaped = False
        deadline = time.time() + 3
        snapshot = {}
        while time.time() < deadline:
            snapshot = client.snapshot()
            if snapshot["used_slots"] == 0:
                reaped = True
                break
            time.sleep(0.02)
        expired_code = None
        try:
            client.authorize(ready["lease_id"], int(ready["generation"]))
        except Exception as exc:
            expired_code = getattr(exc, "code", type(exc).__name__)

        process.send_signal(signal.SIGKILL)
        return_code = process.wait(timeout=5)
        stdout, stderr = process.communicate(timeout=1)
        no_report = not exit_report.exists()
        persisted = (workspace / "cache" / "objects").exists()

    with CommandSession(root, workspace / "resume", config) as resume_session:
        resumed = resume_session.render_current(ledger_name="resumed_after_sigkill")
        resumed_ok = resumed["manifest"]["ok"]

    expected_ws = workspace / "expected_full"
    with CommandSession(root, expected_ws, config) as expected_session:
        expected = expected_session.render_current(ledger_name="expected_full")["manifest"]
    comparison = compare_manifests(
        resumed["manifest"], workspace / "resume", expected, expected_ws
    )
    return {
        "name": "sigkill_and_lease_reclamation_recovery",
        "ok": (
            reaped
            and expired_code == "E_LEASE_EXPIRED"
            and return_code == -signal.SIGKILL
            and no_report
            and persisted
            and resumed_ok
            and comparison["ok"]
        ),
        "slot_reaped_while_worker_unrenewed": reaped,
        "expired_authorize_code": expired_code,
        "sigkill_return_code": return_code,
        "no_exit_report_after_sigkill": no_report,
        "persisted_artifacts_present": persisted,
        "resume_ok": resumed_ok,
        "pixel_equivalence_after_resume": comparison["ok"],
        "differences": comparison["differences"] + comparison["stage_summary_mismatches"],
        "worker_stderr_tail": stderr[-500:],
    }


def make_random_commands(rng: random.Random, case_index: int) -> list[dict[str, Any]]:
    vocab = [
        "entity_cube",
        "entity_sphere",
        "entity_grid",
        "attr_red",
        "attr_blue",
        "attr_green",
        "attr_bright",
        "attr_dark",
        "attr_smooth",
        "attr_rough",
        "motion_pan",
        "motion_zoom",
        "quality_sharp",
        "quality_soft",
    ]
    commands: list[dict[str, Any]] = []
    for index in range(6):
        choice = rng.randrange(8)
        if choice == 0:
            commands.append({"op": "set_prompt", "tokens": rng.sample(vocab, rng.randint(2, 5))})
        elif choice == 1:
            commands.append({"op": "set_image", "path": f"samples/images/condition_{rng.choice(['a', 'b'])}.png"})
        elif choice == 2:
            commands.append({"op": "delete_image"})
        elif choice == 3:
            commands.append(
                {
                    "op": "tune_motion",
                    "block_row": rng.randrange(8),
                    "block_col": rng.randrange(8),
                    "dx_q8": rng.randint(-1024, 1024),
                    "dy_q8": rng.randint(-1024, 1024),
                }
            )
        elif choice == 4:
            commands.append(
                {
                    "op": "set_weights",
                    "weights": {
                        "text": rng.randint(1, 5),
                        "image": rng.randint(1, 5),
                        "motion": rng.randint(1, 5),
                    },
                }
            )
        elif choice == 5:
            commands.append({"op": "set_denoise_steps", "value": rng.randint(1, 6)})
        elif choice == 6:
            commands.append({"op": "set_tile_height", "value": rng.choice([4, 8, 16, 32])})
        else:
            commands.append(
                {
                    "op": "set_quant",
                    "frac_bits": rng.choice([4, 8, 12]),
                    "mode": rng.choice(["round_half_up", "floor", "round_half_even"]),
                }
            )
    return commands


def _reset_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.parent.mkdir(parents=True, exist_ok=True)


def _is_subsequence(values: list[str], reference: list[str]) -> bool:
    position = 0
    for value in values:
        try:
            position = reference.index(value, position) + 1
        except ValueError:
            return False
    return True
