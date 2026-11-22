from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any

from . import Workbench
from .errors import SimulatedCrash
from .io import write_json_atomic
from .samplegen import generate_samples
from .snapshot import SnapshotStore


SEMANTIC_FIELDS = (
    "semantic_digest",
    "failures",
    "pending",
    "budget",
    "boundary_ties",
    "mode_differences",
    "arcs",
    "trees",
    "injections",
)


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _report_fields_equal(left: dict[str, Any], right: dict[str, Any]) -> dict[str, bool]:
    return {field: left[field] == right[field] for field in SEMANTIC_FIELDS}


def _cache_ledger(report: dict[str, Any], sid: str) -> dict[str, Any] | None:
    for sentence in report["sentences"]:
        if sentence["sentence_id"] == sid:
            return sentence.get("cache_ledger")
    return None


def run_verification(
    samples_dir: str | Path,
    out_dir: str | Path,
    *,
    seed: int = 20261123,
    keep_work: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    samples = Path(samples_dir).resolve()
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if not (samples / "sentences.jsonl").exists():
        generate_samples(samples, seed=seed, count=10)

    work = out / "verification-work"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    project = work / "project"
    shutil.copytree(samples, project)

    checks: list[dict[str, Any]] = []

    def check(name: str, passed: bool, **details: Any) -> None:
        checks.append({"name": name, "passed": bool(passed), **details})

    baseline_wb = Workbench(project, snapshot_dir=work / "baseline-snapshot")
    baseline = baseline_wb.run(fresh=True)
    write_json_atomic(out / "demo_report.json", baseline)
    check(
        "baseline_complete",
        all(item["status"] == "OK" for item in baseline["sentences"]),
        sentence_count=len(baseline["sentences"]),
        failure_count=len(baseline["failures"]),
    )

    rerun = baseline_wb.run()
    all_reused = all(
        item["cache_ledger"]["recomputed_count"] == 0
        for item in rerun["sentences"]
        if item["status"] == "OK"
    )
    check(
        "no_change_reuses_every_transition",
        rerun["semantic_digest"] == baseline["semantic_digest"] and all_reused,
        reused_steps=sum(
            item["cache_ledger"]["reused_count"]
            for item in rerun["sentences"]
            if item["status"] == "OK"
        ),
    )

    serial_project = work / "serial-project"
    shutil.copytree(samples, serial_project)
    serial_wb = Workbench(serial_project, snapshot_dir=work / "serial-snapshot")
    serial_wb.set_config("max_threads", 1)
    serial = serial_wb.run(fresh=True)
    serial_fields = _report_fields_equal(baseline, serial)
    check(
        "serial_equals_parallel",
        all(serial_fields.values()),
        fields=serial_fields,
        serial_threads=1,
        parallel_threads=baseline["run_profile"]["max_threads"],
    )

    edit_rows: list[dict[str, Any]] = []
    s001 = next(item for item in baseline["sentences"] if item["sentence_id"] == "S001")

    def change_used_weight() -> None:
        assert previous_incremental is not None
        sentence = next(
            item for item in previous_incremental["sentences"] if item["sentence_id"] == "S001"
        )
        usage: dict[str, list[int]] = {}
        for step in sentence["steps"]:
            for cell in step["cache"]["weight_cells"]:
                usage.setdefault(cell, []).append(step["step_index"])
        cell = max((max(indexes), key) for key, indexes in usage.items())[1]
        action, raw_index = cell.rsplit(":", 1)
        model = _json_load(project / "model.json")
        old_value = model["action_weights"][action][int(raw_index)]
        new_value = old_value + 0.5 if old_value >= 0 else old_value - 0.5
        baseline_wb.set_action_weight(action, int(raw_index), new_value)

    operations: list[tuple[str, str, Any]] = [
        (
            "replace_sentence",
            "S001",
            lambda: baseline_wb.replace_sentence(
                "S001",
                list(s001["words"]),
                [("a" if index == len(s001["pos_tags"]) - 1 else value) for index, value in enumerate(s001["pos_tags"])],
            ),
        ),
        (
            "add_sentence",
            "S011",
            lambda: baseline_wb.add_sentence(
                "S011",
                list(s001["words"]),
                list(s001["pos_tags"]),
            ),
        ),
        ("delete_sentence", "S002", lambda: baseline_wb.delete_sentence("S002")),
        (
            "constraint_change",
            "S001",
            lambda: baseline_wb.set_constraint("forbid_left_dependent_pos", ["d"]),
        ),
        (
            "weight_change",
            "S001",
            change_used_weight,
        ),
        ("mode_switch", "S001", lambda: baseline_wb.set_mode("exhaustive")),
    ]
    last_incremental: dict[str, Any] | None = None
    previous_incremental: dict[str, Any] | None = None
    for index, (name, target_sid, operation) in enumerate(operations):
        operation()
        incremental = baseline_wb.run()
        full = Workbench(
            project,
            snapshot_dir=work / f"sequence-full-{index}",
        ).run(fresh=True)
        fields = _report_fields_equal(incremental, full)
        ledger = _cache_ledger(incremental, target_sid)
        row = {
            "sequence": index + 1,
            "operation": name,
            "target_sentence_id": target_sid,
            "semantic_equal": all(fields.values()),
            "field_equality": fields,
            "target_cache_ledger": ledger,
            "incremental_digest": incremental["semantic_digest"],
            "full_digest": full["semantic_digest"],
            "mode_difference_count": incremental["mode_differences"]["different_sentence_count"],
        }
        if name == "mode_switch" and previous_incremental is not None:
            previous_actions = {
                item["sentence_id"]: item.get("action_sequence", [])
                for item in previous_incremental["sentences"]
                if item["status"] == "OK"
            }
            changed_sentences = [
                item["sentence_id"]
                for item in incremental["sentences"]
                if item["status"] == "OK"
                and previous_actions.get(item["sentence_id"]) != item.get("action_sequence", [])
            ]
            row["mode_switch_changed_sentences"] = changed_sentences
            check(
                "mode_switch_changes_selected_action_sequence",
                bool(changed_sentences),
                changed_sentences=changed_sentences,
                evaluated_transition_difference_count=incremental["mode_differences"].get(
                    "different_evaluated_transition_count"
                ),
            )
        edit_rows.append(row)
        check(
            f"edit_sequence_{index + 1}_{name}",
            all(fields.values()),
            target_sentence_id=target_sid,
            first_invalid_step=(ledger or {}).get("first_invalid_step"),
            reused_count=(ledger or {}).get("reused_count"),
            recomputed_count=(ledger or {}).get("recomputed_count"),
        )
        if name == "constraint_change" and ledger is not None:
            check(
                "constraint_change_uses_feasible_set_invalidation",
                ledger["reason"] == "CONSTRAINT_CHANGED" and ledger["first_invalid_step"] is not None,
                first_invalid_step=ledger["first_invalid_step"],
                reused_count=ledger["reused_count"],
                recomputed_count=ledger["recomputed_count"],
            )
        if name == "weight_change" and ledger is not None:
            check(
                "weight_change_uses_step_level_invalidation",
                ledger["reason"] == "MODEL_WEIGHT_CHANGED"
                and ledger["first_invalid_step"] is not None,
                first_invalid_step=ledger["first_invalid_step"],
                reused_count=ledger["reused_count"],
                recomputed_count=ledger["recomputed_count"],
            )
        previous_incremental = incremental
        last_incremental = incremental
    write_json_atomic(out / "edit_sequence.json", {"edits": edit_rows})
    assert last_incremental is not None
    write_json_atomic(out / "final_incremental_report.json", last_incremental)

    injection_project = work / "injection-project"
    shutil.copytree(samples, injection_project)
    injection_wb = Workbench(
        injection_project,
        snapshot_dir=work / "injection-snapshot",
        injection_file="injections_demo.json",
    )
    injection_report = injection_wb.run(fresh=True)
    injection_rerun = injection_wb.run()
    injection_codes = {
        failure["sentence_id"]: failure["code"]
        for failure in injection_report["failures"]
    }
    injection_hits = {
        item["injection_id"]: item["business_code"] for item in injection_report["injections"]
    }
    expected_injection_codes = {
        "S002": "E_SENT_INVALID",
        "S003": "E_FEATURE_UNDECLARED",
        "S004": "E_MODEL_INVALID",
        "S005": "E_CONFIG_INVALID",
    }
    check(
        "fault_injection_mapping_and_isolation",
        all(injection_codes.get(sid) == code for sid, code in expected_injection_codes.items())
        and injection_report["semantic_digest"] == injection_rerun["semantic_digest"]
        and any(item["status"] == "OK" for item in injection_report["sentences"]),
        failure_codes=injection_codes,
        injection_hits=injection_hits,
    )
    write_json_atomic(out / "injection_report.json", injection_report)

    budget_project = work / "budget-project"
    shutil.copytree(samples, budget_project)
    budget_wb = Workbench(budget_project, snapshot_dir=work / "budget-snapshot")
    budget_wb.set_config("batch_parse_point_budget", 6)
    budget_report = budget_wb.run(fresh=True)
    check(
        "budget_pending_and_next_batch_priority",
        bool(budget_report["pending"])
        and len(budget_report["budget"]["batches"]) > 1
        and all(item["code"] == "E_PARSE_BUDGET_PENDING" for item in budget_report["pending"]),
        pending=[item["sentence_id"] for item in budget_report["pending"]],
        batch_count=len(budget_report["budget"]["batches"]),
    )
    write_json_atomic(out / "budget_report.json", budget_report)

    recovery_project = work / "recovery-project"
    shutil.copytree(samples, recovery_project)
    recovery_wb = Workbench(recovery_project, snapshot_dir=work / "recovery-reference")
    recovery_wb.set_config("max_threads", 1)
    reference = recovery_wb.run(fresh=True)
    crash_wb = Workbench(recovery_project, snapshot_dir=work / "recovery-snapshot")
    crashed = False
    try:
        crash_wb.run(fresh=True, crash_after_sentences=4, crash_partial_steps=2)
    except SimulatedCrash:
        crashed = True
    store = SnapshotStore(work / "recovery-snapshot")
    manifest = store.latest_manifest()
    assert manifest is not None
    partial_sids = [
        sid
        for sid, cursor in manifest["sentence_cursors"].items()
        if not cursor["completed"] and cursor["steps"] == 2
    ]
    resumed = crash_wb.run(
        resume=True,
        expected_version=int(manifest["global_version"]),
    )
    resumed_fields = _report_fields_equal(reference, resumed)
    resumed_partial_ledgers = {
        sid: _cache_ledger(resumed, sid)
        for sid in partial_sids
    }
    check(
        "snapshot_sentence_and_step_cursor_recovery",
        crashed
        and bool(partial_sids)
        and all(resumed_fields.values())
        and all(
            ledger is not None and ledger["reason"] == "RESUME_CURSOR" and ledger["reused_count"] >= 2
            for ledger in resumed_partial_ledgers.values()
        ),
        crash_manifest_version=manifest["global_version"],
        completed_before_resume=manifest["completed_sentences"],
        partial_sids=partial_sids,
        partial_ledgers=resumed_partial_ledgers,
        fields=resumed_fields,
    )
    write_json_atomic(out / "resume_report.json", resumed)
    write_json_atomic(
        out / "resume_manifest.json",
        {
            "crashed_manifest": manifest,
            "resumed_snapshot": resumed["snapshot"],
            "semantic_digest": resumed["semantic_digest"],
        },
    )

    old_version = int(resumed["snapshot"]["global_version"])
    recovery_wb.set_mode("exhaustive")
    stale_report = recovery_wb.run(resume=True, expected_version=old_version)
    stale_full = Workbench(
        recovery_project,
        snapshot_dir=work / "stale-full-snapshot",
    ).run(fresh=True)
    stale_fields = _report_fields_equal(stale_report, stale_full)
    check(
        "stale_snapshot_reinitialization",
        stale_report.get("snapshot_stale", {}).get("code") == "E_SNAPSHOT_STALE"
        and all(stale_fields.values()),
        stale_event=stale_report.get("snapshot_stale"),
        fields=stale_fields,
    )
    write_json_atomic(out / "stale_report.json", stale_report)

    summary = {
        "all_checks_passed": all(item["passed"] for item in checks),
        "check_count": len(checks),
        "passed_count": sum(1 for item in checks if item["passed"]),
        "seed": seed,
        "sample_dir": str(samples),
        "artifact_dir": str(out),
        "semantic_digest": baseline["semantic_digest"],
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    verification = {
        "schema_version": 1,
        "summary": summary,
        "checks": checks,
        "baseline_snapshot": baseline["snapshot"],
        "provenance_note": (
            "Parsing semantics, failures, arcs, trees and mode summaries are compared field-by-field. "
            "Cache reuse/recompute ledgers are intentionally execution provenance and therefore differ "
            "between an incremental run and a blank full rebuild; each ledger is checked separately."
        ),
    }
    write_json_atomic(out / "verification_report.json", verification)
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    return verification


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Run the deterministic acceptance verification")
    parser.add_argument("--samples", default="samples")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--keep-work", action="store_true")
    args = parser.parse_args()
    result = run_verification(args.samples, args.out, seed=args.seed, keep_work=args.keep_work)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["summary"]["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
