from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from .incremental import IncrementalStore
from .ledger import (
    downgrade_g2_to_g1,
    parse_g1_text,
    parse_g2_text,
    serialize_g1_book,
    upgrade_g1_to_g2,
)
from .loading import load_input, parse_config, parse_weights
from .pipeline import run_pipeline, semantic_comparison_report
from .flip import attribute_verdict_flip, compare_run_verdicts
from .util import canonical_json


ROOT = Path(__file__).resolve().parents[1]


def _same_semantic(left: dict[str, Any], right: dict[str, Any]) -> tuple[bool, str | None]:
    a = semantic_comparison_report(left)
    b = semantic_comparison_report(right)
    text_a = canonical_json(a)
    text_b = canonical_json(b)
    if text_a == text_b:
        return True, None
    return False, _first_difference(a, b)


def _first_difference(left: Any, right: Any, path: str = "$", depth: int = 0) -> str:
    if depth > 5:
        return path
    if type(left) is not type(right):
        return f"{path}: type {type(left).__name__}!={type(right).__name__}"
    if isinstance(left, dict) and isinstance(right, dict):
        for key in sorted(set(left) | set(right)):
            if key not in left or key not in right:
                return f"{path}.{key}: missing"
            diff = _first_difference(left[key], right[key], f"{path}.{key}", depth + 1)
            if diff:
                return diff
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return f"{path}: length {len(left)}!={len(right)}"
        for index, (a, b) in enumerate(zip(left, right)):
            diff = _first_difference(a, b, f"{path}[{index}]", depth + 1)
            if diff:
                return diff
    elif left != right:
        return f"{path}: {left!r} != {right!r}"
    return ""


def write_sample_ledgers(root: Path) -> dict[str, Any]:
    samples = root / "samples"
    out_dir = samples / "ledgers"
    out_dir.mkdir(parents=True, exist_ok=True)
    loaded = load_input(samples / "preferences.jsonl", samples / "annotators.yaml", samples / "config.yaml")
    if loaded.config is None:
        raise AssertionError([i.to_dict() for i in loaded.issues])
    serial = run_pipeline(loaded.records, loaded.weights, loaded.config, list(loaded.issues), workers=1)
    threaded = run_pipeline(loaded.records, loaded.weights, loaded.config, list(loaded.issues), workers=4)
    equal, diff = _same_semantic(serial["report"], threaded["report"])
    if not equal:
        raise AssertionError(f"thread mismatch: {diff}")

    book = serial["book"]
    g2_l2 = book.to_g2_json("L2")
    g2_l1 = book.to_g2_json("L1")
    (out_dir / "ledger.L2.g2.json").write_text(g2_l2, encoding="utf-8")
    (out_dir / "ledger.L1.g2.json").write_text(g2_l1, encoding="utf-8")
    (out_dir / "ledger.g2.json").write_text(g2_l2, encoding="utf-8")

    downgraded, sidecar = downgrade_g2_to_g1(book)
    g1_l2 = serialize_g1_book(downgraded, "L2")
    g1_l1 = serialize_g1_book(downgraded, "L1")
    (out_dir / "ledger.L2.g1.txt").write_text(g1_l2, encoding="utf-8")
    (out_dir / "ledger.L1.g1.txt").write_text(g1_l1, encoding="utf-8")
    (out_dir / "ledger.g1.txt").write_text(g1_l2, encoding="utf-8")
    (out_dir / "ledger.g1.sidecar.json").write_text(
        json.dumps(sidecar, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )

    g1_book, g1_issues = parse_g1_text(g1_l2)
    if g1_issues:
        raise AssertionError([i.to_dict() for i in g1_issues])
    upgraded = upgrade_g1_to_g2(g1_book, sidecar)

    g2_book, g2_issues = parse_g2_text(g2_l2)
    if g2_book is None or g2_issues:
        raise AssertionError([i.to_dict() for i in g2_issues])
    entries_before = [entry.to_dict() for entry in g2_book.sorted_entries()]
    entries_after = [entry.to_dict() for entry in upgraded.sorted_entries()]
    roundtrip_equal = entries_before == entries_after

    reparsed_sidecar = json.loads((out_dir / "ledger.g1.sidecar.json").read_text(encoding="utf-8"))
    reparsed_g1, _ = parse_g1_text((out_dir / "ledger.g1.txt").read_text(encoding="utf-8"))
    reparsed_up = upgrade_g1_to_g2(reparsed_g1, reparsed_sidecar)
    file_roundtrip_equal = [e.to_dict() for e in reparsed_up.sorted_entries()] == entries_before

    l1_book, l1_issues = parse_g2_text(g2_l1)
    l1_core_equal = False
    if l1_book is not None and not l1_issues:
        l1_core_equal = [e.core_key() for e in l1_book.sorted_entries()] == [e.core_key() for e in g2_book.sorted_entries()]

    return {
        "ok": roundtrip_equal and file_roundtrip_equal and l1_core_equal and equal,
        "entry_count": len(entries_before),
        "g1_line_width": 77,
        "field_roundtrip_equal": roundtrip_equal,
        "file_roundtrip_equal": file_roundtrip_equal,
        "l1_l2_core_equal": l1_core_equal,
        "serial_threads_equal": equal,
        "thread_mismatch": diff,
        "paths": sorted(str(path.relative_to(root)) for path in out_dir.iterdir()),
    }


def check_incremental(root: Path) -> dict[str, Any]:
    samples = root / "samples"
    store = IncrementalStore.from_loaded_files(
        samples / "preferences.jsonl", samples / "annotators.yaml", samples / "config.yaml"
    )
    store.run(workers=1, save=False)
    commands = yaml.safe_load((samples / "edits.yaml").read_text(encoding="utf-8"))
    steps: list[dict[str, Any]] = []
    all_ok = True
    precision_ok = True
    precision_notes: list[str] = []

    for index, command in enumerate(commands):
        store.apply_command(command)
        incremental_result = store.run(workers=3, save=False)
        assert incremental_result["report"] is not None
        oracle = IncrementalStore(store.raw_records, store.raw_weights, store.raw_config)
        oracle_result = oracle.run(workers=1, save=False)
        equal, diff = _same_semantic(store.last_report, oracle_result["report"])
        all_ok &= equal

        invalidation = store.last_report.get("invalidation", [])
        reuse = store.last_report.get("reuse_ledger", [])
        stages = sorted({event["stage"] for event in invalidation})
        step_info = {
            "index": index,
            "op": command.get("op", command.get("type")),
            "semantic_equal": equal,
            "difference": diff,
            "invalidation_count": len(invalidation),
            "invalidation_stages": stages,
            "reuse_count": len(reuse),
            "cross_session": any(event["crosses_session_boundary"] for event in invalidation),
        }
        steps.append(step_info)
        if not equal:
            break

        if index == 0:
            invalid_sessions = {session for event in invalidation for session in event["sessions"]}
            unaffected_reuse = any("S002" in json.dumps(event) or "S003" in json.dumps(event) for event in reuse)
            expected = invalid_sessions <= {"S001"} and unaffected_reuse
            precision_ok &= expected
            precision_notes.append(f"replace_record_precise={expected}")
        if index == 1:
            cross = any(event["stage"] == "window" and event["crosses_session_boundary"] for event in invalidation)
            precision_ok &= cross
            precision_notes.append(f"weight_change_crosses_sessions={cross}")
        if step_info["op"] == "set_window_lengths":
            window_reuse = sum(1 for event in reuse if event["stage"] == "window")
            precision_ok &= window_reuse > 0
            precision_notes.append(f"unchanged_window_length_reuse={window_reuse}")
        if step_info["op"] == "set_partition_boundaries":
            window_reuse = sum(1 for event in reuse if event["stage"] == "window")
            precision_ok &= window_reuse > 0
            precision_notes.append(f"boundary_edit_window_reuse={window_reuse}")
        if step_info["op"] == "set_quant":
            feature_reuse = sum(1 for event in reuse if event["stage"] == "feature")
            precision_ok &= feature_reuse > 0
            precision_notes.append(f"quant_edit_bucket_reuse={feature_reuse}")
        if step_info["op"] in {"add_annotator", "remove_annotator"}:
            unused = command.get("annotator_id") in {"ann_temp", "ann_unused"}
            if unused:
                precision_ok &= len(invalidation) == 0
                precision_notes.append(f"unused_annotator_edit_zero_invalidation_step={index}")

    return {"ok": all_ok and precision_ok, "all_steps_equal": all_ok, "precision_ok": precision_ok, "precision_notes": precision_notes, "steps": steps}


def check_budget_and_snapshot(root: Path) -> dict[str, Any]:
    samples = root / "samples"
    budget_demo_root = root / "artifacts" / "budget_state"
    if budget_demo_root.exists():
        shutil.rmtree(budget_demo_root)
    budget_demo_root.mkdir(parents=True)
    state_root = budget_demo_root / "state"
    store = IncrementalStore.from_loaded_files(
        samples / "preferences.jsonl", samples / "annotators.yaml", samples / "config.yaml", root=state_root
    )
    budget_result = store.run(budget=128, workers=1)
    budget_report = budget_result["report"]
    budget_codes = [issue["code"] for issue in budget_report["issues"]]
    budget_ok = "E_WINDOW_BUDGET_PENDING" in budget_codes
    pending_count = len(budget_report["scheduler"]["pending_task_ids"])
    budget_thread_store = IncrementalStore.from_loaded_files(
        samples / "preferences.jsonl", samples / "annotators.yaml", samples / "config.yaml"
    )
    threaded_budget = budget_thread_store.run(budget=128, workers=4, save=False)
    budget_order_equal, budget_order_diff = _same_semantic(budget_report, threaded_budget["report"])

    recovered = IncrementalStore.recover(state_root)
    recovery_ok = recovered is not None and recovered.version == 1
    assert recovered is not None
    resume_result = recovered.run(budget=100_000, workers=2)
    oracle_store = IncrementalStore.from_loaded_files(
        samples / "preferences.jsonl", samples / "annotators.yaml", samples / "config.yaml"
    )
    oracle = oracle_store.run(budget=100_000, workers=1)
    resume_equal, resume_diff = _same_semantic(resume_result["report"], oracle["report"])

    quota_store = IncrementalStore.from_loaded_files(
        samples / "preferences.jsonl", samples / "annotators.yaml", samples / "config.yaml"
    )
    quota_result = quota_store.run(budget=100_000, session_quota=200, workers=1)
    quota_codes = [issue["code"] for issue in quota_result["report"]["issues"]]
    quota_ok = "E_SESSION_QUOTA_EXCEEDED" in quota_codes

    crash_demo_root = root / "artifacts" / "crash_state"
    if crash_demo_root.exists():
        shutil.rmtree(crash_demo_root)
    crash_demo_root.mkdir(parents=True)
    crash_state = crash_demo_root / "state"
    crash_store = IncrementalStore.from_loaded_files(
        samples / "preferences.jsonl", samples / "annotators.yaml", samples / "config.yaml", root=crash_state
    )
    crash_store.run(save=True)
    command = [
        sys.executable,
        "-m",
        "preference_ledger.cli",
        "crash",
        "--state",
        str(crash_state),
        "--stage",
        "03_scores.json",
    ]
    proc = subprocess.run(command, cwd=root, capture_output=True, text=True)
    killed_ok = proc.returncode == -9
    after_kill = IncrementalStore.recover(crash_state)
    recover_after_kill_ok = after_kill is not None and after_kill.version == 1
    final_result = after_kill.run(workers=1) if after_kill is not None else None
    crash_equal = False
    crash_diff = "no committed snapshot"
    if final_result is not None:
        crash_equal, crash_diff = _same_semantic(final_result["report"], oracle["report"])

    return {
        "ok": budget_ok and budget_order_equal and recovery_ok and resume_equal and quota_ok and killed_ok and recover_after_kill_ok and crash_equal,
        "budget_pending_code": budget_ok,
        "pending_count": pending_count,
        "budget_thread_order_equal": budget_order_equal,
        "budget_thread_difference": budget_order_diff,
        "snapshot_recovery": recovery_ok,
        "resume_matches_full": resume_equal,
        "resume_difference": resume_diff,
        "quota_code": quota_ok,
        "quota_codes": sorted(set(quota_codes)),
        "sigkill_returncode": proc.returncode,
        "sigkill_observed": killed_ok,
        "recover_after_sigkill": recover_after_kill_ok,
        "crash_rebuild_matches_full": crash_equal,
        "crash_difference": crash_diff,
        "state_root": "artifacts/budget_state/state",
        "crash_state": "artifacts/crash_state/state",
    }


def check_faults(root: Path) -> dict[str, Any]:
    samples = root / "samples"
    injections_doc = yaml.safe_load((samples / "fault_injections.yaml").read_text(encoding="utf-8"))
    base_config = yaml.safe_load((samples / "config.yaml").read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    hit_codes: set[str] = set()

    for injection in injections_doc["injections"]:
        injection_id = injection["id"]
        expected = injection["code"]
        issues = []
        if injection["kind"] in {"unknown_version", "truncated_g1_line"}:
            _, issues = parse_g1_text(injection["line"], injection_id=injection_id)
        elif injection["kind"] in {"weight_out_of_range", "non_finite_weight"}:
            _, issues = parse_weights({"weights": {injection["annotator_id"]: injection["weight_bp"]}})
        elif injection["kind"] == "window_lengths_not_increasing":
            raw = dict(base_config)
            raw["window_lengths"] = injection["window_lengths"]
            _, issues = parse_config(raw)
        issues = [replace(issue, injection_id=injection_id) for issue in issues]
        codes = [issue.code for issue in issues]
        injection_ids = [issue.injection_id for issue in issues if issue.injection_id]
        hit = expected in codes and injection_id in injection_ids
        if hit:
            hit_codes.add(expected)
        results.append(
            {
                "id": injection_id,
                "expected_code": expected,
                "observed_codes": codes,
                "stage": issues[0].stage if issues else "",
                "injection_id_attached": injection_id in injection_ids,
                "hit": hit,
            }
        )

    loaded = load_input(samples / "preferences.jsonl", samples / "annotators.yaml", samples / "config.yaml")
    healthy_count = len(loaded.records)
    healthy_ok = not loaded.issues and healthy_count == 21
    required = {
        "E_VERSION_UNKNOWN",
        "E_FIELD_TRUNCATED",
        "E_ANNOTATOR_WEIGHT_RANGE",
        "E_WINDOW_MONOTONIC",
        "E_NON_FINITE",
    }
    return {
        "ok": all(item["hit"] for item in results) and hit_codes == required and healthy_ok,
        "healthy_records_processed": healthy_count,
        "healthy_ok": healthy_ok,
        "required_codes_hit": sorted(hit_codes),
        "injections": results,
    }


def check_flip_attribution(root: Path) -> dict[str, Any]:
    def item(scope: str, verdict: int, numerator: int, denominator: int = 1) -> dict[str, Any]:
        return {
            "scope_id": scope,
            "session_id": "SFLIP",
            "kind": "window",
            "turns": [1],
            "verdict_code": verdict,
            "mean_permille_exact_num": numerator,
            "mean_permille_exact_den": denominator,
        }

    score_a = {("SFLIP", 1): {"quantized_scaled": 0, "scale": 10000}}
    score_b_changed = {("SFLIP", 1): {"quantized_scaled": 1, "scale": 10000}}
    score_b_same = {("SFLIP", 1): {"quantized_scaled": 0, "scale": 10000}}

    quant = attribute_verdict_flip(item("q", 0, 0), item("q", 1, 1), score_a, score_b_changed)
    accumulation = attribute_verdict_flip(item("w", 0, 0), item("w", 1, 1), score_a, score_b_same)
    threshold = attribute_verdict_flip(item("t", 0, 0), item("t", 1, 0), score_a, score_b_same)
    synthetic_sources = [quant["source"], accumulation["source"], threshold["source"]]
    synthetic_ok = synthetic_sources == ["score_quantization", "weighted_accumulation", "threshold_comparison"]

    samples = root / "samples"
    loaded = load_input(samples / "preferences.jsonl", samples / "annotators.yaml", samples / "config.yaml")
    coarse_config = replace(loaded.config, quant_scheme="Q_COARSE")
    fine_config = replace(loaded.config, quant_scheme="Q_FINE")
    coarse = run_pipeline(loaded.records, loaded.weights, coarse_config)
    fine = run_pipeline(loaded.records, loaded.weights, fine_config)
    sample_flips = compare_run_verdicts(coarse, fine)
    return {
        "ok": synthetic_ok,
        "synthetic_sources": synthetic_sources,
        "sample_coarse_vs_fine_flip_count": len(sample_flips),
        "sample_flips": sample_flips[:20],
    }


def run_verification(root: str | Path = ROOT) -> dict[str, Any]:
    root = Path(root)
    report = {
        "ok": False,
        "ledger_roundtrip": write_sample_ledgers(root),
        "incremental": check_incremental(root),
        "budget_snapshot": check_budget_and_snapshot(root),
        "faults": check_faults(root),
        "flip_attribution": check_flip_attribution(root),
    }
    report["ok"] = all(
        report[name].get("ok")
        for name in ("ledger_roundtrip", "incremental", "budget_snapshot", "faults", "flip_attribution")
    )
    artifacts = root / "artifacts"
    artifacts.mkdir(exist_ok=True)
    output = artifacts / "verification_report.json"
    output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return report
