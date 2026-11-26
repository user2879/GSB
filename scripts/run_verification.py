#!/usr/bin/env python3
"""Run the reproducible local acceptance demonstration and write reports."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from mmfm.canonical import atomic_write_json, atomic_write_text, load_json
from mmfm.config import load_render_config, validate_prompt
from mmfm.errors import BusinessError
from mmfm.exchange import StageRecord, serialize_v1, v1_to_v2, v2_to_v1
from mmfm.png import decode_png_rgb
from mmfm.renderer import Renderer
from mmfm.samplegen import generate_samples
from mmfm.snapshot import SnapshotStore
from mmfm.workspace import Workspace, load_workspace


def clear_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def slim_result(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "code": result["code"],
        "generation": result["generation"],
        "global_version": result["global_version"],
        "max_workers": result["max_workers"],
        "max_observed_concurrency": result["max_observed_concurrency"],
        "ledger_summary": result["ledger_summary"],
        "budget": result["budget"],
        "pending_ids": [item["logical_id"] for item in result["pending"]],
        "frames": [
            {
                "frame": frame["frame"],
                "status": frame["status"],
                "pixel_digest": frame.get("pixel_digest"),
                "png_sha256": frame.get("png_sha256"),
                "chunks": frame.get("chunks", []),
            }
            for frame in result["frames"]
        ],
        "chunk_completion_order": result["chunk_completion_order"],
    }


def expect_code(code: str, function, *args, **kwargs) -> str:
    try:
        function(*args, **kwargs)
    except BusinessError as exc:
        return exc.code
    return f"EXPECTED_{code}_BUT_SUCCEEDED"


def exchange_check(config) -> dict[str, Any]:
    v2 = load_json(ROOT / "samples" / "exchange" / "denoise_v2.json")
    v1_text = (ROOT / "samples" / "exchange" / "denoise_v1.txt").read_text(encoding="utf-8")
    sidecar = load_json(ROOT / "samples" / "exchange" / "denoise_v1.sidecar.json")
    profiles = {name: profile.fraction_bits for name, profile in config.quant_profiles.items()}
    restored = v1_to_v2(
        v1_text,
        sidecar=sidecar,
        noise_code_map=config.noise_schemes,
        profile_bits=profiles,
    )
    fields_equal = all(
        left["guidance"] == right["guidance"]
        and left["refs"] == right["refs"]
        and left["noise_scheme"] == right["noise_scheme"]
        and left["quant_profile"] == right["quant_profile"]
        for left, right in zip(v2["records"], restored["records"])
    )
    regenerated_v1, regenerated_sidecar = v2_to_v1(
        v2,
        noise_schemes=config.noise_schemes,
        profiles=profiles,
        allow_missing=False,
    )
    return {
        "v1": str(ROOT / "samples" / "exchange" / "denoise_v1.txt"),
        "v2": str(ROOT / "samples" / "exchange" / "denoise_v2.json"),
        "v1_matches_regeneration": v1_text == regenerated_v1,
        "sidecar_matches_regeneration": sidecar == regenerated_sidecar,
        "round_trip_fields_exact": fields_equal,
        "sidecar_restored": restored["provenance"]["sidecar_restored"],
        "sidecar_hash_verified": restored["provenance"]["sidecar_hash_verified"],
        "residuals": sidecar["downgrade_residuals"],
    }


def validation_check(workspace, repo_root: Path) -> dict[str, Any]:
    png = (repo_root / "samples" / "images" / "geometry_a.png").read_bytes()
    return {
        "unknown_term": expect_code(
            "E_UNKNOWN_TERM",
            validate_prompt,
            {"entities": ["unknown"], "attributes": []},
            workspace.conditions.vocab,
        ),
        "bad_png_dimension": expect_code(
            "E_PNG_DIMENSION",
            decode_png_rgb,
            png,
            width=47,
            height=32,
        ),
        "v1_width_overflow": expect_code(
            "E_V1_WIDTH",
            serialize_v1,
            [StageRecord(stage="denoise", frame=10000, guidance_permille=1000, noise_code=1, quant_bits=8)],
        ),
        "unknown_edit_command": expect_code(
            "E_UNKNOWN_COMMAND",
            workspace.apply_commands,
            [{"op": "delete_snapshot"}],
        ),
    }


def frame_summary(result: dict[str, Any]) -> list[dict[str, Any]]:
    return slim_result(result)["frames"]


def all_equal(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> bool:
    return all(
        first["pixel_digest"] == second["pixel_digest"]
        and first["png_sha256"] == second["png_sha256"]
        for first, second in zip(left, right)
    )


def main() -> int:
    samples = generate_samples(ROOT)
    workspace = load_workspace(ROOT / "configs" / "render.json", ROOT / "samples" / "conditions.json")
    config = load_render_config(ROOT / "configs" / "render.json")
    verification_root = ROOT / "snapshots" / "verification"
    clear_directory(verification_root)
    reports_root = ROOT / "reports" / "verification"
    clear_directory(reports_root)

    full = Renderer(workspace, verification_root / "full", budget_override=100).render()
    full_frames = frame_summary(full)

    incremental_workspace = workspace
    incremental_initial = Renderer(
        incremental_workspace, verification_root / "incremental", budget_override=100
    ).render()
    edited_workspace, applied = incremental_workspace.apply_commands(
        ROOT / "samples" / "edits" / "selective_image.json"
    )
    incremental_edited = Renderer(
        edited_workspace, verification_root / "incremental", budget_override=100
    ).render()
    incremental_denoise = [
        event for event in incremental_edited["ledger_events"] if event["stage"] == "denoise"
    ]

    chunk_workspace = workspace
    chunk_initial = Renderer(chunk_workspace, verification_root / "chunks", budget_override=100).render()
    chunk_edited_workspace, _ = chunk_workspace.apply_commands(ROOT / "samples" / "edits" / "chunk_rows.json")
    chunk_edited = Renderer(
        chunk_edited_workspace, verification_root / "chunks", budget_override=100
    ).render()

    budget_limited = Renderer(
        workspace, verification_root / "budget", budget_override=2
    ).render()
    budget_resumed = Renderer(
        workspace, verification_root / "budget", budget_override=100
    ).render()

    serial_workspace, _ = workspace.apply_commands([{"op": "set_max_workers", "value": 1}])
    parallel_workspace, _ = workspace.apply_commands([{"op": "set_max_workers", "value": 4}])
    old_task_delay = os.environ.get("MMFM_TASK_DELAY_MS")
    os.environ["MMFM_TASK_DELAY_MS"] = "3"
    try:
        serial = Renderer(serial_workspace, verification_root / "serial", budget_override=5).render()
        parallel = Renderer(parallel_workspace, verification_root / "parallel", budget_override=5).render()
        serial_full = Renderer(
            serial_workspace, verification_root / "serial_full", budget_override=100
        ).render()
        parallel_full = Renderer(
            parallel_workspace, verification_root / "parallel_full", budget_override=100
        ).render()
    finally:
        if old_task_delay is None:
            os.environ.pop("MMFM_TASK_DELAY_MS", None)
        else:
            os.environ["MMFM_TASK_DELAY_MS"] = old_task_delay

    crash_snapshot = verification_root / "crash"
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(ROOT / "src")
    environment["MMFM_FAULT_AFTER_COMMIT"] = "3"
    crash = subprocess.run(
        [
            sys.executable,
            "-m",
            "mmfm.cli",
            "render",
            "--config",
            str(ROOT / "configs" / "render.json"),
            "--conditions",
            str(ROOT / "samples" / "conditions.json"),
            "--snapshot-dir",
            str(crash_snapshot),
            "--budget",
            "100",
            "--report",
            str(reports_root / "crash_before_resume.json"),
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    crash_store = SnapshotStore(crash_snapshot)
    crash_committed = crash_store.load_committed()
    if crash_committed is None:
        raise RuntimeError("fault injection did not leave a committed snapshot")
    status_process = subprocess.run(
        [
            sys.executable,
            "-m",
            "mmfm.cli",
            "status",
            "--snapshot-dir",
            str(crash_snapshot),
        ],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=False,
    )
    status_payload = json.loads(status_process.stdout)
    crash_workspace = Workspace.from_serialised(crash_committed.state["workspace"])
    crash_resumed = Renderer(
        crash_workspace, crash_snapshot, budget_override=100
    ).render()

    exchange = exchange_check(config)
    validation = validation_check(workspace, ROOT)
    test_process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-o", "addopts="],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
        check=False,
    )
    test_output = (test_process.stdout + test_process.stderr).strip().splitlines()
    report = {
        "ok": True,
        "environment": {
            "python": sys.version,
            "numpy": __import__("numpy").__version__,
            "tkinter": _tk_status(),
        },
        "samples": samples,
        "exchange": exchange,
        "validation_codes": validation,
        "full_render": slim_result(full),
        "incremental": {
            "applied_commands": applied,
            "initial_frame_digests": [frame["pixel_digest"] for frame in incremental_initial["frames"]],
            "edited_frame_digests": [frame["pixel_digest"] for frame in incremental_edited["frames"]],
            "unaffected_frame_1_reused": all(
                event["event"] == "reuse_hit"
                for event in incremental_denoise
                if event["frame"] == 1
            ),
            "changed_frame_2_recomputed_steps": sum(
                event["event"] == "recompute"
                for event in incremental_denoise
                if event["frame"] == 2
            ),
            "changed_frame_3_recomputed_steps": sum(
                event["event"] == "recompute"
                for event in incremental_denoise
                if event["frame"] == 3
            ),
            "recomputed_denoise_points": incremental_edited["budget"]["batch_spent"],
            "ledger_events": incremental_edited["ledger_events"],
        },
        "chunk_repartition": {
            "initial_chunks_per_frame": [len(frame["chunks"]) for frame in chunk_initial["frames"]],
            "edited_chunks_per_frame": [len(frame["chunks"]) for frame in chunk_edited["frames"]],
            "edited_chunk_ranges": [
                [chunk["row_range"] for chunk in frame["chunks"]]
                for frame in chunk_edited["frames"]
            ],
            "denoise_recompute_points": chunk_edited["budget"]["batch_spent"],
            "pixels_unchanged": all_equal(chunk_initial["frames"], chunk_edited["frames"]),
            "completion_order": chunk_edited["chunk_completion_order"],
        },
        "budget_recovery": {
            "limited_code": budget_limited["code"],
            "limited_pending": [item["logical_id"] for item in budget_limited["pending"]],
            "limited_spent": budget_limited["budget"]["batch_spent"],
            "resume_code": budget_resumed["code"],
            "resume_spent": budget_resumed["budget"]["batch_spent"],
            "matches_full": all_equal(budget_resumed["frames"], full_frames),
        },
        "concurrency": {
            "serial_observed": serial["max_observed_concurrency"],
            "parallel_observed": parallel["max_observed_concurrency"],
            "configured_max_workers": parallel["max_workers"],
            "observation_delay_ms": 3,
            "budget_sequence_equal": serial["budget"]["deduction_sequence"]
            == parallel["budget"]["deduction_sequence"],
            "pending_equal": serial["pending"] == parallel["pending"],
            "completed_frames_equal": all_equal(
                [frame for frame in serial["frames"] if frame["status"] == "complete"],
                [frame for frame in parallel["frames"] if frame["status"] == "complete"],
            ),
            "full_frames_equal": serial_full["frames"] == parallel_full["frames"],
            "full_ledger_events_equal": serial_full["ledger_events"] == parallel_full["ledger_events"],
            "full_cursor_equal": serial_full["cursor"] == parallel_full["cursor"],
            "full_budget_sequence_equal": serial_full["budget"]["deduction_sequence"]
            == parallel_full["budget"]["deduction_sequence"],
        },
        "crash_recovery": {
            "returncode": crash.returncode,
            "committed_generation_after_kill": crash_committed.generation,
            "pending_before_resume": len(crash_committed.state.get("pending", [])),
            "status_after_kill": {
                "exit_code": status_process.returncode,
                "code": status_payload.get("code"),
            },
            "resumed_code": crash_resumed["code"],
            "matches_full": all_equal(crash_resumed["frames"], full_frames),
            "resumed_frame_digests": [frame["pixel_digest"] for frame in crash_resumed["frames"]],
        },
        "tests": {
            "returncode": test_process.returncode,
            "summary": test_output[-1] if test_output else "",
        },
    }
    report["ok"] = all(
        [
            report["exchange"]["round_trip_fields_exact"],
            report["incremental"]["unaffected_frame_1_reused"],
            report["incremental"]["recomputed_denoise_points"] == 6,
            report["chunk_repartition"]["pixels_unchanged"],
            report["chunk_repartition"]["denoise_recompute_points"] == 0,
            report["budget_recovery"]["matches_full"],
            report["concurrency"]["budget_sequence_equal"],
            report["concurrency"]["pending_equal"],
            report["concurrency"]["completed_frames_equal"],
            report["concurrency"]["full_frames_equal"],
            report["concurrency"]["full_ledger_events_equal"],
            report["concurrency"]["full_cursor_equal"],
            report["concurrency"]["full_budget_sequence_equal"],
            report["concurrency"]["parallel_observed"] <= report["concurrency"]["configured_max_workers"],
            report["crash_recovery"]["returncode"] == 137,
            report["crash_recovery"]["matches_full"],
            report["crash_recovery"]["status_after_kill"]["exit_code"] == 4,
            report["tests"]["returncode"] == 0,
            all(code.startswith("E_") for code in report["validation_codes"].values()),
        ]
    )
    atomic_write_json(ROOT / "reports" / "verification.json", report)
    atomic_write_text(ROOT / "reports" / "verification.md", render_markdown(report))
    print(json.dumps({"ok": report["ok"], "report": str(ROOT / "reports" / "verification.json")}, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


def _tk_status() -> dict[str, Any]:
    try:
        import tkinter  # type: ignore

        return {"available": True, "version": tkinter.TkVersion}
    except Exception as exc:
        return {"available": False, "error": str(exc), "required_for_core_render": False}


def render_markdown(report: dict[str, Any]) -> str:
    full = report["full_render"]
    incremental = report["incremental"]
    chunks = report["chunk_repartition"]
    budget = report["budget_recovery"]
    concurrency = report["concurrency"]
    crash = report["crash_recovery"]
    lines = [
        "# 本地验收结果",
        "",
        f"总体状态：**{'PASS' if report['ok'] else 'FAIL'}**",
        "",
        "## 环境与样本",
        f"- Python：`{report['environment']['python'].splitlines()[0]}`",
        f"- numpy：`{report['environment']['numpy']}`",
        f"- tkinter：{report['environment']['tkinter']}",
        f"- 帧数：{len(full['frames'])}，去噪步：{full['budget']['batch_spent']}，分块行：{len(full['frames'][0]['chunks'])}",
        "",
        "## 双版本交换",
        f"- v1 重生成一致：`{report['exchange']['v1_matches_regeneration']}`",
        f"- sidecar 重生成一致：`{report['exchange']['sidecar_matches_regeneration']}`",
        f"- v2→v1→v2 字段无损：`{report['exchange']['round_trip_fields_exact']}`",
        "",
        "## 增量失效",
        f"- 编辑：`{incremental['applied_commands']}`",
        f"- 未受影响帧 1 全部复用：`{incremental['unaffected_frame_1_reused']}`",
        f"- 帧 2 重算去噪步：`{incremental['changed_frame_2_recomputed_steps']}`",
        f"- 帧 3 重算去噪步：`{incremental['changed_frame_3_recomputed_steps']}`",
        f"- 本批重算点：`{incremental['recomputed_denoise_points']}`",
        "",
        "## 分块行重划分",
        f"- 编辑前每帧分块数：`{chunks['initial_chunks_per_frame']}`",
        f"- 编辑后每帧分块数：`{chunks['edited_chunks_per_frame']}`",
        f"- 新行区间：`{chunks['edited_chunk_ranges'][0]}`",
        f"- 去噪重算点：`{chunks['denoise_recompute_points']}`",
        f"- 像素不变：`{chunks['pixels_unchanged']}`",
        f"- 规范化完成顺序：`{chunks['completion_order']}`",
        "",
        "## 预算与恢复",
        f"- 低预算代码：`{budget['limited_code']}`，消耗 `{budget['limited_spent']}`，待办 `{len(budget['limited_pending'])}`",
        f"- 恢复代码：`{budget['resume_code']}`，本批消耗 `{budget['resume_spent']}`",
        f"- 恢复像素与全量一致：`{budget['matches_full']}`",
        f"- 强杀返回码：`{crash['returncode']}`；重启前已提交代：`{crash['committed_generation_after_kill']}`",
        f"- 强杀后状态：`{crash['status_after_kill']}`",
        f"- 恢复像素与全量一致：`{crash['matches_full']}`",
        "",
        "## 并发一致性",
        f"- 观察并发：串行 `{concurrency['serial_observed']}` / 并行 `{concurrency['parallel_observed']}`，上限 `{concurrency['configured_max_workers']}`",
        f"- 预算扣减序列一致：`{concurrency['budget_sequence_equal']}`",
        f"- 待办队列一致：`{concurrency['pending_equal']}`",
        f"- 已完成帧摘要一致：`{concurrency['completed_frames_equal']}`",
        f"- 全量帧/分块摘要一致：`{concurrency['full_frames_equal']}`",
        f"- 全量账本事件一致：`{concurrency['full_ledger_events_equal']}`",
        f"- 全量游标与预算序列一致：`{concurrency['full_cursor_equal']}` / `{concurrency['full_budget_sequence_equal']}`",
        "",
        "## 校验错误码",
    ]
    for name, code in report["validation_codes"].items():
        lines.append(f"- `{name}` → `{code}`")
    lines.extend(
        [
            "",
            f"- pytest：`{report['tests']['summary']}`",
            "",
            "机器可读明细见 `reports/verification.json`。",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
