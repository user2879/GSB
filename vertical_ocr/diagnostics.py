from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .codes import FIELD_TRUNCATED, MATCH_BUDGET_PENDING, SNAPSHOT_STALE
from .config import canonical_hash, default_config
from .exchange import (
    downgrade_v3_to_v1,
    parse_v1,
    parse_v2,
    parse_v3,
    upgrade_v1_to_v3,
)
from .generator import apply_plan_commands, generate_workspace, load_plan
from .injection import run_injection_suite, write_injection_config
from .ledger import JsonlLedger, SidecarStore, atomic_write_json
from .model import MISSING, ExchangeRecord, is_missing
from .pipeline import Pipeline


def _check(checks: list[dict[str, Any]], name: str, ok: bool, detail: Any = "") -> bool:
    checks.append({"name": name, "ok": bool(ok), "detail": detail})
    return bool(ok)


def prepare_workspace(root: Path, config: dict[str, Any] | None = None) -> dict[str, Any]:
    if root.exists():
        shutil.rmtree(root)
    generate_workspace(root)
    cfg = config or default_config()
    atomic_write_json(root / "config" / "config.json", cfg)
    commands = {
        "schema": "vertical-ocr-commands/v1",
        "commands": [
            {"op": "replace_image", "image": "page_001.png", "seed": 319777},
            {"op": "set_keypoint_version", "image": "page_001.png", "block_id": "p1-c0-b0",
             "version_no": 9, "version_name": "kpv-demo-v9", "annotator": "demo",
             "annotated_ms": 1763300000000, "x": 630, "y": 128},
        ],
    }
    atomic_write_json(root / "config" / "commands.json", commands)
    write_injection_config(root)
    return cfg


def _cache_entries(root: Path) -> dict[str, Any]:
    return json.loads((root / "ledger" / "cache_store.json").read_text(encoding="utf-8"))


def _truth_accuracy(root: Path) -> dict[str, Any]:
    truth = json.loads((root / "input" / "layout_truth.json").read_text(encoding="utf-8"))
    expected = {
        (image["image"], block["block_id"]): block["char"]
        for image in truth["images"]
        for column in image["columns"]
        for block in column["blocks"]
    }
    cache = _cache_entries(root)
    correct = total = 0
    bad = []
    for entry in cache["entries"].values():
        if entry.get("stage") != "template_matching":
            continue
        product = entry["product"]
        scope = entry["scope"]
        image = scope.split(":", 2)[1]
        block_id = product.get("truth_block_id")
        if (image, block_id) not in expected:
            continue
        total += 1
        if product.get("recognized_char") == expected[(image, block_id)]:
            correct += 1
        else:
            bad.append({"image": image, "block_id": block_id,
                        "expected": expected[(image, block_id)],
                        "actual": product.get("recognized_char")})
    return {"correct": correct, "total": total, "bad": bad}


def _stage_events(events: list[dict[str, Any]], stage: str, status: str | None = None) -> list[dict[str, Any]]:
    return [e for e in events if e.get("stage") == stage and (status is None or e.get("status") == status)]


def conversion_self_check(base: Path, checks: list[dict[str, Any]]) -> None:
    conv_dir = base / "conversion-check"
    conv_dir.mkdir(parents=True, exist_ok=True)
    ledger = JsonlLedger(conv_dir / "conversion_ledger.jsonl")
    sidecars = SidecarStore(conv_dir / "sidecar_store.json", ledger)
    rec = ExchangeRecord(
        image_no=7,
        char_no=9,
        bbox=[10.2, 20.8, 110.25, 220.75],
        confidence=0.123456,
        reading_order="rtl_top_to_bottom",
        char="天",
        keypoint_version="kpv-v9",
        keypoint_x=60,
        keypoint_y=120,
        annotation_version="ann-demo",
        source_version="v3",
    )
    v1, entries = downgrade_v3_to_v1([rec], sidecars, ledger)
    parsed_v1 = parse_v1(v1)
    restored = upgrade_v1_to_v3(parsed_v1, sidecars, ledger)[0]
    v3 = "\n".join([]) if False else None
    parse_v3(__import__("vertical_ocr.exchange", fromlist=["to_v3"]).to_v3([restored]))
    _check(checks, "float_bbox_sidecar_restore",
           abs(float(restored.bbox[2]) - 110.25) < 1e-9 and restored.keypoint_x == 60,
           {"bbox": restored.bbox, "sidecar_entries": len(entries)})
    _check(checks, "confidence_half_up_and_restore",
           parsed_v1[0].source_values["v1_fields"]["confidence_permille"] == 123
           and abs(float(restored.confidence) - 0.123456) < 1e-12,
           {"permille": 123, "restored": restored.confidence})

    for version, text, field in [
        ("v1", "0001bad", "line_1"),
        ("v2", json.dumps({"records": [{"image_no": 1}]}), "char_no"),
        ("v3", "# vertical-ocr/v3\nimage_no=1\n", "char_no"),
    ]:
        try:
            if version == "v1":
                parse_v1(text)
            elif version == "v2":
                parse_v2(text)
            else:
                parse_v3(text)
            _check(checks, f"{version}_field_truncated_{field}", False, "no exception")
        except Exception as exc:
            _check(checks, f"{version}_field_truncated_{field}",
                   getattr(exc, "code", None) == FIELD_TRUNCATED and getattr(exc, "field", "").startswith(field.split("_")[0]) or field in str(getattr(exc, "field", "")),
                   {"code": getattr(exc, "code", None), "field": getattr(exc, "field", None)})


def run_acceptance(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    checks: list[dict[str, Any]] = []

    baseline = root / "baseline"
    cfg = prepare_workspace(baseline)
    first = Pipeline(baseline, cfg).run()
    _check(checks, "baseline_54_receipts", first["receipt_counts"]["generated_records"] == 54
           and first["receipt_counts"]["v3_receipts"] == 54, first["receipt_counts"])
    _check(checks, "baseline_no_stage_failures", not first["failures"], first["failures"][:2])
    accuracy = _truth_accuracy(baseline)
    _check(checks, "baseline_recognition_accuracy_54_54", accuracy["correct"] == 54, accuracy)

    cache = _cache_entries(baseline)
    fixed_ok = True
    fixed_detail = []
    for entry in cache["entries"].values():
        if entry.get("stage") != "template_matching":
            continue
        p = entry["product"]
        residual = abs(float(p["quantization_residual"]))
        if residual > 0.5 * float(p["quantization_step"]) + 1e-12:
            fixed_ok = False
            fixed_detail.append(p.get("truth_block_id"))
    _check(checks, "fixed_point_quant_step_and_residual", fixed_ok, {"bad": fixed_detail[:5]})

    second = Pipeline(baseline, cfg).run()
    _check(checks, "same_version_full_cache_hit", second["cache_counts"]["hit"] == 123
           and second["cache_counts"]["recomputed"] == 0, second["cache_counts"])
    _check(checks, "snapshot_fresh_on_second_run", second["snapshot"]["fresh"] is True,
           second["snapshot"]["reason"])

    conversion_self_check(root, checks)

    plan_path = baseline / "input" / "sample_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    plan["global_version"] = 2
    atomic_write_json(plan_path, plan)
    kp_path = baseline / "input" / "keypoint_versions.json"
    kp_data = json.loads(kp_path.read_text(encoding="utf-8"))
    kp_data["submissions"].append({
        "image": "page_001.png", "block_id": "p1-c0-b0", "version_no": 99,
        "version_name": "kpv-acceptance-99", "annotated_ms": 1763400000000,
        "annotator": "z_final", "x": 630, "y": 128,
    })
    atomic_write_json(kp_path, kp_data)
    kp_run = Pipeline(baseline, cfg).run()
    match_hits = len(_stage_events(kp_run["cache_events"], "template_matching", "hit"))
    reading_recompute = len(_stage_events(kp_run["cache_events"], "reading_assembly", "recomputed"))
    _check(checks, "keypoint_switch_reuses_all_matching", match_hits == 54, {"match_hits": match_hits})
    _check(checks, "keypoint_switch_recomputes_reading", reading_recompute == 3,
           {"reading_recompute": reading_recompute})
    _check(checks, "keypoint_switch_reports_snapshot_stale",
           kp_run["business_code_counts"].get(SNAPSHOT_STALE) == 1, kp_run["business_code_counts"])
    first_entry = next(e for img in json.loads((baseline / "snapshots" / "current.json").read_text()) and []) if False else None
    kp_report = json.loads((baseline / "reports" / "last_report.json").read_text())
    page1 = next(img for img in kp_report["images"] if img["image"] == "page_001.png")
    _check(checks, "keypoint_version_99_authority",
           page1["reading_sequence"][0]["character_id"] == "p1-c0-b0",
           page1["reading_sequence"][0])

    cfg_split = json.loads((baseline / "config" / "config.json").read_text())
    cfg_split["split_penalty"] = 0.91
    atomic_write_json(baseline / "config" / "config.json", cfg_split)
    split_run = Pipeline(baseline, cfg_split).run()
    _check(checks, "split_penalty_normal_blocks_all_reused",
           split_run["cache_counts"]["recomputed"] == 0 and split_run["cache_counts"]["hit"] == 123,
           split_run["cache_counts"])

    image_root = root / "image-edit"
    image_cfg = prepare_workspace(image_root)
    Pipeline(image_root, image_cfg).run()
    image_plan = load_plan(image_root)
    apply_plan_commands(image_plan, [{"op": "replace_image", "image": "page_002.png", "seed": 8675309}])
    generate_workspace(image_root, image_plan)
    image_run = Pipeline(image_root, image_cfg).run()
    recomputed_match = _stage_events(image_run["cache_events"], "template_matching", "recomputed")
    hit_match = _stage_events(image_run["cache_events"], "template_matching", "hit")
    _check(checks, "replace_single_image_invalidates_only_image",
           len(recomputed_match) == 18 and len(hit_match) == 36
           and all("page_002" in e["scope"] for e in recomputed_match)
           and all("page_002" not in e["scope"] for e in hit_match),
           {"recomputed": len(recomputed_match), "hit": len(hit_match)})

    structure_root = root / "structure"
    structure_cfg = prepare_workspace(structure_root)
    Pipeline(structure_root, structure_cfg).run()
    structure_plan = load_plan(structure_root)
    apply_plan_commands(structure_plan, [{
        "op": "add_column", "image": "page_001.png", "column_id": "p1-col-3",
        "x_center": 150, "chars": ["上", "下", "山", "人", "大", "天"],
    }])
    generate_workspace(structure_root, structure_plan)
    add_run = Pipeline(structure_root, structure_cfg).run()
    add_match_recompute = _stage_events(add_run["cache_events"], "template_matching", "recomputed")
    add_match_hit = _stage_events(add_run["cache_events"], "template_matching", "hit")
    _check(checks, "add_column_reuses_existing_characters",
           len(add_match_recompute) == 6 and len(add_match_hit) == 54,
           {"recomputed": len(add_match_recompute), "hit": len(add_match_hit)})

    structure_plan = load_plan(structure_root)
    apply_plan_commands(structure_plan, [{"op": "remove_column", "image": "page_001.png", "column_id": "p1-col-3"}])
    generate_workspace(structure_root, structure_plan)
    remove_run = Pipeline(structure_root, structure_cfg).run()
    _check(checks, "remove_column_reuses_remaining_characters",
           len(_stage_events(remove_run["cache_events"], "template_matching", "hit")) == 54
           and len(_stage_events(remove_run["cache_events"], "template_matching", "recomputed")) == 0,
           remove_run["cache_counts"])

    budget_root = root / "budget"
    budget_cfg = prepare_workspace(budget_root)
    budget_first = Pipeline(budget_root, budget_cfg).run(batch_budget=70_000)
    first_pending = budget_first["pending"][0]["work_id"] if budget_first["pending"] else None
    _check(checks, "budget_first_batch_has_pending",
           budget_first["budget"]["pending_count"] > 0
           and budget_first["business_code_counts"].get(MATCH_BUDGET_PENDING) == budget_first["budget"]["pending_count"],
           budget_first["budget"])
    budget_second = Pipeline(budget_root, budget_cfg).run(batch_budget=10_000_000)
    _check(checks, "budget_second_batch_prioritizes_pending",
           budget_second["budget"]["pending_count"] == 0
           and budget_second["budget"]["admitted_order"]
           and budget_second["budget"]["admitted_order"][0] == first_pending,
           {"first_pending": first_pending, "admitted_first": budget_second["budget"]["admitted_order"][:3]})

    kill_root = root / "kill-recovery"
    kill_cfg = prepare_workspace(kill_root)
    Pipeline(kill_root, kill_cfg).run()
    kill_plan = load_plan(kill_root)
    kill_plan["global_version"] = 2
    atomic_write_json(kill_root / "input" / "sample_plan.json", kill_plan)
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parent.parent)}
    proc = subprocess.run(
        [sys.executable, "-m", "vertical_ocr.cli", "run", "--root", str(kill_root), "--kill-before-commit"],
        cwd=str(Path(__file__).resolve().parent.parent),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    pointer_before_kill = json.loads((kill_root / "snapshots" / "current.json").read_text())
    manifest_before_path = Path(pointer_before_kill["manifest_path"])
    if not manifest_before_path.exists():
        manifest_before_path = kill_root / "snapshots" / pointer_before_kill["manifest_path"]
    _check(checks, "sigkill_during_commit_returns_signal", proc.returncode == -9,
           {"returncode": proc.returncode, "stderr": proc.stderr[-300:]})
    current_manifest_before = json.loads(manifest_before_path.read_text())
    _check(checks, "last_committed_snapshot_remains_v1",
           current_manifest_before["global_version"] == "1", current_manifest_before["snapshot_id"])
    recovery_run = Pipeline(kill_root, kill_cfg).run()
    _check(checks, "restart_reports_snapshot_stale_and_rebuilds",
           recovery_run["snapshot"]["reason"] == "global_version_mismatch"
           and recovery_run["business_code_counts"].get(SNAPSHOT_STALE) == 1
           and recovery_run["global_version"] == "2",
           {"reason": recovery_run["snapshot"]["reason"], "version": recovery_run["global_version"]})

    injections = run_injection_suite(root / "injections", baseline)
    _check(checks, "all_five_injections_hit_business_codes", injections["all_hits"],
           [{"id": r["injection_id"], "hit": r["hit"], "observed": r["observed_code"]} for r in injections["results"]])

    ok = all(c["ok"] for c in checks)
    summary = {
        "schema": "vertical-ocr-acceptance/v1",
        "ok": ok,
        "passed": sum(1 for c in checks if c["ok"]),
        "total": len(checks),
        "checks": checks,
        "injections": injections,
        "monotonic_elapsed_ms_main_report": first["elapsed_ms"],
    }
    atomic_write_json(root / "acceptance_report.json", summary)
    lines = ["# Acceptance Result", "", f"overall: {'PASS' if ok else 'FAIL'}", "",
             f"passed: {summary['passed']}/{summary['total']}", "", "| Check | Result | Detail |", "|---|---:|---|"]
    for check in checks:
        detail = json.dumps(check["detail"], ensure_ascii=False)
        if len(detail) > 180:
            detail = detail[:177] + "..."
        lines.append(f"| `{check['name']}` | {'PASS' if check['ok'] else 'FAIL'} | `{detail.replace('|', '/')}` |")
    (root / "acceptance_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
