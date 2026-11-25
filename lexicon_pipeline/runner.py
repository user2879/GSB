"""Local first-version delivery orchestration."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .commands import apply_command_stream
from .config import load_config
from .formats import parse_jv1, parse_jv2
from .hashing import canonical_json
from .pipeline import comparison_view, run_state_once
from .snapshot import SnapshotManager
from .validation import load_and_validate_inputs


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def check_sample_formats(root: Path) -> dict[str, Any]:
    samples = root / "samples"
    valid_jv1 = parse_jv1((samples / "jv1/sample_receipts.jv1").read_text(encoding="utf-8"))
    bad_jv1 = parse_jv1((samples / "jv1/truncated.jv1").read_text(encoding="utf-8"))
    valid_jv2_text = (samples / "jv2/sample_receipts.jv2.json").read_text(encoding="utf-8")
    invalid_jv2_text = (samples / "jv2/invalid.jv2.json").read_text(encoding="utf-8")
    valid_jv2 = parse_jv2(valid_jv2_text)
    invalid_jv2 = parse_jv2(invalid_jv2_text)
    return {
        "valid_jv1_records": len(valid_jv1.records),
        "valid_jv1_issue_count": len(valid_jv1.issues),
        "malformed_jv1_issue_count": len(bad_jv1.issues),
        "malformed_jv1_issues": [issue.as_dict() for issue in bad_jv1.issues],
        "valid_jv2_records": len(valid_jv2.records),
        "valid_jv2_issue_count": len(valid_jv2.issues),
        "invalid_jv2_issue_count": len(invalid_jv2.issues),
        "invalid_jv2_issues": [issue.as_dict() for issue in invalid_jv2.issues],
        "expected_codes_found": {
            "E_FIELD_TRUNCATED": any(issue.code == "E_FIELD_TRUNCATED" for issue in bad_jv1.issues),
            "E_FORMAT_INVALID": any(issue.code == "E_FORMAT_INVALID" for issue in invalid_jv2.issues),
        },
    }


def _business_code_counts(*runs: dict[str, Any], extra_codes: list[str] | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}

    def add(code: str | None) -> None:
        if code:
            counts[code] = counts.get(code, 0) + 1

    for code in extra_codes or []:
        add(code)
    for run in runs:
        for batch in run["scheduler"]["batches"]:
            for item in batch["pending"]:
                add(item["code"])
        for item in run["scheduler"]["failures"]:
            add(item["code"])
        for item in run.get("failures", []):
            add(item.get("code"))
        for entry in run.get("ledger_entries", []):
            if entry.get("entry_type") == "command_failure":
                add(entry.get("code"))
    return dict(sorted(counts.items()))


def _cache_counts(run: dict[str, Any]) -> dict[str, Any]:
    events = run["cache_events"]
    by_stage: dict[str, dict[str, int]] = {}
    for event in events:
        stage = event["stage"]
        by_stage.setdefault(stage, {"hit": 0, "miss": 0})
        by_stage[stage]["hit" if event["hit"] else "miss"] += 1
    return {
        "hit_total": sum(1 for event in events if event["hit"]),
        "miss_total": sum(1 for event in events if not event["hit"]),
        "by_stage": by_stage,
    }


def precise_reuse_demo(base_state: dict[str, Any], artifacts: Path) -> dict[str, Any]:
    demo_cache = artifacts / "pipeline-state" / "frequency-only-reuse-cache.json"
    if demo_cache.exists():
        demo_cache.unlink()
    baseline = run_state_once(base_state, demo_cache, None, "serial", "reuse-baseline")
    edited = json.loads(json.dumps(base_state))
    entry = next(item for item in edited["lexicon"]["entries"] if item["id"] == "E010")
    entry["frequency"] = int(entry["frequency"]) + 37
    entry["version"] = int(entry["version"]) + 1
    incremental = run_state_once(edited, demo_cache, baseline["manifest"], "serial", "frequency-edit-incremental")

    full_cache = artifacts / "pipeline-state" / "frequency-only-full-cache.json"
    if full_cache.exists():
        full_cache.unlink()
    full = run_state_once(edited, full_cache, baseline["manifest"], "serial", "frequency-edit-full")
    equal = canonical_json(comparison_view(incremental)) == canonical_json(comparison_view(full))
    return {
        "edited_entry_id": entry["id"],
        "edited_frequency": entry["frequency"],
        "incremental_cache": _cache_counts(incremental),
        "invalidation_count": len(incremental["impact"]["invalidations"]),
        "reuse_plan_count": len(incremental["impact"]["reuses"]),
        "changed_entry_ids": incremental["impact"]["changed_entry_ids"],
        "incremental_vs_full_equal": equal,
        "sample_invalidations": incremental["impact"]["invalidations"],
        "sample_reuses": incremental["impact"]["reuses"],
    }


def execute_local_delivery(root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root).resolve()
    artifacts = root_path / "artifacts"
    receipts_dir = artifacts / "receipts"
    state_dir = artifacts / "pipeline-state"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)

    config = load_config(root_path / "configs/local.json")
    inputs = load_and_validate_inputs(root_path / "samples")
    base_state: dict[str, Any] = {**inputs, "config": config, "exchange_version": "jv2"}

    main_cache = root_path / config["cache_path"]
    if main_cache.exists():
        main_cache.unlink()
    baseline = run_state_once(base_state, main_cache, None, "concurrent", "baseline")

    applied = apply_command_stream(base_state, inputs["commands"])
    final_state = applied["state"]
    incremental = run_state_once(
        final_state,
        main_cache,
        baseline["manifest"],
        "concurrent",
        "command-incremental",
        applied["command_failures"],
    )

    full_cache = state_dir / "blank-full-cache.json"
    serial_cache = state_dir / "serial-equivalence-cache.json"
    parallel_cache = state_dir / "parallel-equivalence-cache.json"
    for path in (full_cache, serial_cache, parallel_cache):
        if path.exists():
            path.unlink()

    full_serial = run_state_once(
        final_state,
        full_cache,
        baseline["manifest"],
        "serial",
        "blank-full-serial",
        applied["command_failures"],
    )
    serial_equiv = run_state_once(
        final_state,
        serial_cache,
        baseline["manifest"],
        "serial",
        "serial-equivalence",
        applied["command_failures"],
    )
    parallel_equiv = run_state_once(
        final_state,
        parallel_cache,
        baseline["manifest"],
        "concurrent",
        "parallel-equivalence",
        applied["command_failures"],
    )

    incremental_view = comparison_view(incremental)
    full_view = comparison_view(full_serial)
    serial_view = comparison_view(serial_equiv)
    parallel_view = comparison_view(parallel_equiv)
    normalized_cache_events = lambda run: [
        {key: value for key, value in event.items() if key != "run_id"}
        for event in run["cache_events"]
    ]
    equivalence = {
        "incremental_vs_blank_full_equal": canonical_json(incremental_view) == canonical_json(full_view),
        "serial_vs_concurrent_equal": canonical_json(serial_view) == canonical_json(parallel_view),
        "serial_vs_concurrent_cache_events_equal": canonical_json(normalized_cache_events(serial_equiv))
        == canonical_json(normalized_cache_events(parallel_equiv)),
        "incremental_roundtrip_ok": incremental["roundtrip"]["matches"]["ok"],
        "serial_pool": serial_equiv["scheduler"]["pool_metrics"],
        "parallel_pool": parallel_equiv["scheduler"]["pool_metrics"],
    }

    reuse_demo = precise_reuse_demo(base_state, artifacts)
    sample_formats = check_sample_formats(root_path)

    (receipts_dir / "final.jv1").write_text(incremental["jv1"]["text"], encoding="utf-8")
    _write_json(receipts_dir / "final.jv2.json", incremental["jv2"])
    _write_json(receipts_dir / "final.sidecar.json", incremental["sidecar"])
    _write_jsonl(
        root_path / config["ledger_path"],
        baseline["ledger_entries"] + incremental["ledger_entries"],
    )
    _write_json(state_dir / "final-manifest.json", incremental["manifest"])
    _write_json(state_dir / "command-result.json", {
        "applied_commands": applied["applied_commands"],
        "command_failures": applied["command_failures"],
        "revision": applied["revision"],
    })

    snapshot_manager = SnapshotManager(root_path / config["snapshot_dir"])
    startup_decision = snapshot_manager.startup_decision(config["config_hash"], inputs["lexicon"]["revision"])
    snapshot_manifest = snapshot_manager.commit(
        stages={
            "01_load_validation": {"status": "ok", "sample_format_checks": sample_formats},
            "02_segmentation": [_stage_doc(item, "segmentation") for item in incremental["doc_results"]],
            "03_tagging": [_stage_doc(item, "tagging") for item in incremental["doc_results"]],
            "04_receipts": {
                "jv1_text": incremental["jv1"]["text"],
                "jv2": incremental["jv2"],
                "sidecar": incremental["sidecar"],
                "roundtrip": incremental["roundtrip"],
            },
        },
        identity={
            "config_hash": incremental["config"]["config_hash"],
            "exchange_version": incremental["exchange_version"],
            "effective_fraction_bits": incremental["config"]["effective_fraction_bits"],
            "quantization_tier": incremental["config"]["quantization_tier"],
        },
        pending=incremental["scheduler"]["pending"],
        cursor={
            "completed_doc_ids": [item["doc_id"] for item in incremental["doc_results"]],
            "completed_doc_seq": [item["doc_seq"] for item in incremental["doc_results"]],
        },
        consumed_budget=incremental["scheduler"]["cumulative_consumed_budget"],
        lexicon_revision=incremental["lexicon_revision"],
    )
    missing_snapshot_check = SnapshotManager(artifacts / "missing-snapshot-check").startup_decision(
        incremental["config"]["config_hash"], incremental["lexicon_revision"]
    )

    report = {
        "report_id": "diagnostic-318",
        "seed": 3182026,
        "equivalence": equivalence,
        "receipt_counts": {
            "jv1_lines": len(incremental["jv1"]["records"]),
            "jv2_receipts": len(incremental["jv2"]["receipts"]),
            "sidecar_records": len(incremental["sidecar"]["records"]),
        },
        "receipt_counts_by_version": {
            "jv1": {"schema_version": "1.0", "count": len(incremental["jv1"]["records"])},
            "jv2": {"schema_version": incremental["jv2"]["version"], "count": len(incremental["jv2"]["receipts"])},
        },
        "downgrade_entries": len(incremental["sidecar"]["downgrade_entries"]),
        "boundary_fragments": len(incremental["boundary_fragments"]),
        "boundary_fragment_list": incremental["boundary_fragments"],
        "invalidation_entries": len(incremental["impact"]["invalidations"]),
        "reuse_plan_entries": len(incremental["impact"]["reuses"]),
        "incremental_cache": _cache_counts(incremental),
        "baseline_cache": _cache_counts(baseline),
        "precise_reuse_demo": reuse_demo,
        "pending_now": incremental["scheduler"]["pending"],
        "pending_batches": [
            {"batch_id": batch["batch_id"], "pending": batch["pending"]}
            for batch in incremental["scheduler"]["batches"]
            if batch["pending"]
        ],
        "failures": incremental["failures"],
        "command_failures": applied["command_failures"],
        "business_code_counts": _ledger_business_code_counts(
            incremental["ledger_entries"],
            extra_codes=[missing_snapshot_check.get("code"), startup_decision.get("code")],
        ),
        "fixed_point": {
            "fraction_bits": incremental["config"]["effective_fraction_bits"],
            "scale": incremental["config"]["fixed_point_scale"],
            "high_precision_bits": incremental["config"]["high_precision_bits"],
            "high_precision_scale": incremental["config"]["high_precision_scale"],
            "quantization_tier": incremental["config"]["quantization_tier"],
            "config_hash": incremental["config"]["config_hash"],
        },
        "sample_format_checks": sample_formats,
        "snapshot": {
            "manifest": snapshot_manifest,
            "startup_decision_before_run": startup_decision,
            "missing_state_check": missing_snapshot_check,
        },
        "scheduler_batches": [
            {
                "batch_id": batch["batch_id"],
                "tasks": [task["doc_id"] for task in batch["tasks"]],
                "pending": [item["doc_id"] for item in batch["pending"]],
                "consumed": batch["consumed"],
                "remaining": batch["remaining"],
            }
            for batch in incremental["scheduler"]["batches"]
        ],
    }
    _write_json(root_path / config["report_path"], report)
    _write_markdown_report(root_path / "artifacts/diagnostic-report.md", report)

    if not (
        equivalence["incremental_vs_blank_full_equal"]
        and equivalence["serial_vs_concurrent_equal"]
        and equivalence["serial_vs_concurrent_cache_events_equal"]
        and reuse_demo["incremental_vs_full_equal"]
        and sample_formats["expected_codes_found"]["E_FIELD_TRUNCATED"]
        and sample_formats["expected_codes_found"]["E_FORMAT_INVALID"]
    ):
        raise AssertionError(json.dumps({
            "equivalence": equivalence,
            "reuse_equal": reuse_demo["incremental_vs_full_equal"],
            "sample_formats": sample_formats["expected_codes_found"],
        }, ensure_ascii=False, indent=2))
    return report


def _stage_doc(doc_result: dict[str, Any], key: str) -> dict[str, Any]:
    value = doc_result[key]
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, default=str))


def _ledger_business_code_counts(entries: list[dict[str, Any]], extra_codes: list[str] | None = None) -> dict[str, int]:
    counts: dict[str, int] = {}

    def add(code: str | None) -> None:
        if code:
            counts[code] = counts.get(code, 0) + 1

    for code in extra_codes or []:
        add(code)
    for entry in entries:
        add(entry.get("code"))
    return dict(sorted(counts.items()))


def _write_markdown_report(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "# 本地诊断报告",
        "",
        f"- 种子：`{report['seed']}`",
        f"- jv1 行数：{report['receipt_counts']['jv1_lines']}",
        f"- jv2 回执数：{report['receipt_counts']['jv2_receipts']}",
        f"- 降级条目：{report['downgrade_entries']}",
        f"- 边界片段：{report['boundary_fragments']}",
        f"- 失效条目：{report['invalidation_entries']}",
        f"- 复用计划条目：{report['reuse_plan_entries']}",
        f"- 增量缓存命中：{report['incremental_cache']['hit_total']}；未命中：{report['incremental_cache']['miss_total']}",
        f"- 精确复用演示命中：{report['precise_reuse_demo']['incremental_cache']['hit_total']}",
        f"- 增量/全量一致：`{report['equivalence']['incremental_vs_blank_full_equal']}`",
        f"- 串行/并发一致：`{report['equivalence']['serial_vs_concurrent_equal']}`",
        f"- jv1/jv2 往返零失配：`{report['equivalence']['incremental_roundtrip_ok']}`",
        "",
        "## 业务码计数",
        "",
    ]
    for code, count in report["business_code_counts"].items():
        lines.append(f"- `{code}`: {count}")
    lines.extend(["", "## 批次", ""])
    for batch in report["scheduler_batches"]:
        lines.append(f"- 批次 {batch['batch_id']}: 任务 {batch['tasks']}，待办 {batch['pending']}，消耗 {batch['consumed']}，剩余 {batch['remaining']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
