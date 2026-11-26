"""Local deterministic acceptance workflow and artifact writer."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .commands import apply_command, validate_command_stream
from .engine import LabEngine
from .generator import default_config, initialize_workspace, make_flip_attribution_cases
from .hashing import canonical_json, fingerprint
from .injections import run_injections
from .snapshot import SnapshotStore
from .tk_bootstrap import tk_status
from .validation import load_json, load_jsonl, validate_workspace


def write_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False, allow_nan=False) + "\n", encoding="utf-8")
    return path


def ledger_signature(report: dict[str, Any]) -> list[tuple[str, str]]:
    return [(entry["stable_key"], entry["cache_key"]) for entry in report["ledger"]]


def commit_engine_snapshot(store: SnapshotStore, engine: LabEngine) -> dict[str, Any]:
    head = store.head_meta()
    expected = fingerprint(head) if head else None
    meta = store.commit(
        {
            "report": engine.last_report,
            "cache": engine.cache,
            "task_index": engine.task_index,
            "pending": engine.pending,
            "cumulative_consumed": engine.cumulative_consumed,
        },
        global_version=engine.global_version,
        config_hash=engine.config_hash(),
        expected_head_fingerprint=expected,
    )
    store.prune_commits(keep=1)
    return meta


def run_verification(root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root)
    paths = {
        "config": root_path / "configs/lab_config.json",
        "samples": root_path / "data/samples.jsonl",
        "commands": root_path / "commands/edit_commands.jsonl",
        "injections": root_path / "injections/injections.json",
        "report_dir": root_path / "artifacts/verification",
    }
    initialize_workspace(root_path)

    config, config_errors = load_json(paths["config"])
    samples, sample_parse_errors = load_jsonl(paths["samples"])
    command_rows, command_parse_errors = load_jsonl(paths["commands"])
    injection_doc, injection_load_errors = load_json(paths["injections"])
    injections = injection_doc.get("injections", []) if isinstance(injection_doc, dict) else []

    validation_errors = validate_workspace(config, samples)
    command_errors = validate_command_stream(command_rows, config, samples)
    all_load_errors = config_errors + sample_parse_errors + command_parse_errors + injection_load_errors

    snapshot_store = SnapshotStore(root_path / "artifacts/snapshot-state")
    incremental = LabEngine(config, samples, global_version=1)
    initial_report = incremental.run(parallel=True)
    initial_snapshot = commit_engine_snapshot(snapshot_store, incremental)
    ledger_series = [
        {"stage": "initial", "command_id": "INIT", "global_version": 1, "ledger": initial_report["ledger"]}
    ]

    step_summaries = []
    for command in command_rows:
        manifest = incremental.apply_command(command)
        report = incremental.run(manifest, parallel=True)
        snapshot_meta = commit_engine_snapshot(snapshot_store, incremental)
        ledger_series.append(
            {
                "stage": "after_command",
                "command_id": command.get("command_id"),
                "op": command.get("op"),
                "global_version": report["global_version"],
                "ledger": report["ledger"],
            }
        )
        step_summaries.append(
            {
                "command_id": command.get("command_id"),
                "op": command.get("op"),
                "global_version": report["global_version"],
                "complete": report["complete"],
                "cache_stats": report["cache_stats"],
                "pending_sample_ids": report["pending_sample_ids"],
                "snapshot_commit": snapshot_meta.get("commit_name"),
            }
        )

    incremental_semantic = incremental.semantic_report()
    incremental_ledger = incremental.last_report["ledger"]

    final_config = copy.deepcopy(config)
    final_samples = copy.deepcopy(samples)
    for command in command_rows:
        final_config, final_samples, _ = apply_command(final_config, final_samples, command)

    blank_serial = LabEngine(final_config, final_samples, global_version=incremental.global_version)
    blank_serial_report = blank_serial.run(parallel=False)
    blank_serial_semantic = blank_serial.semantic_report()

    blank_parallel = LabEngine(final_config, final_samples, global_version=incremental.global_version)
    blank_parallel_report = blank_parallel.run(parallel=True)
    blank_parallel_semantic = blank_parallel.semantic_report()

    incremental_vs_blank = canonical_json(incremental_semantic) == canonical_json(blank_serial_semantic)
    serial_vs_parallel = canonical_json(blank_serial_semantic) == canonical_json(blank_parallel_semantic)
    inc_ledger_signature = ledger_signature(incremental.last_report)
    blank_ledger_signature = ledger_signature(blank_serial_report)
    ledger_order_equal = [item[0] for item in inc_ledger_signature] == [item[0] for item in blank_ledger_signature]
    ledger_cache_keys_equal = inc_ledger_signature == blank_ledger_signature

    budget = _verify_budget(config, samples, root_path / "artifacts/budget-state")
    injection_report = run_injections(config, samples, injections)
    flip_attribution = _verify_flip_attribution()
    recovery = snapshot_store.recovery_decision(incremental.global_version, incremental.config_hash())
    tk = tk_status(create_root=False)

    comparison = {
        "incremental_vs_blank_field_equal": incremental_vs_blank,
        "serial_vs_parallel_field_equal": serial_vs_parallel,
        "ledger_stable_key_order_equal": ledger_order_equal,
        "ledger_cache_key_signature_equal": ledger_cache_keys_equal,
        "incremental_ledger_entries": len(incremental_ledger),
        "blank_ledger_entries": len(blank_serial_report["ledger"]),
        "incremental_status_counts": _status_counts(incremental_ledger),
        "blank_status_counts": _status_counts(blank_serial_report["ledger"]),
    }

    summary = {
        "ok": bool(
            not validation_errors
            and not command_errors
            and not all_load_errors
            and incremental_vs_blank
            and serial_vs_parallel
            and ledger_order_equal
            and ledger_cache_keys_equal
            and budget["resume_compare_equal"]
            and injection_report["all_hit"]
            and flip_attribution["all_expected_sources_hit"]
            and recovery["use_snapshot"]
        ),
        "initial_validation_error_count": len(validation_errors),
        "command_validation_error_count": len(command_errors),
        "load_error_count": len(all_load_errors),
        "comparison": comparison,
        "budget": {key: value for key, value in budget.items() if key != "reports"},
        "injection": {"all_hit": injection_report["all_hit"], "hit_count": injection_report["hit_count"], "total": injection_report["total"]},
        "flip_attribution": {
            "all_expected_sources_hit": flip_attribution["all_expected_sources_hit"],
            "cases": [
                {"case_id": item["case_id"], "expected_source": item["expected_source"], "hit": item["hit"]}
                for item in flip_attribution["cases"]
            ],
        },
        "snapshot_recovery": recovery,
        "tkinter": tk,
        "steps": step_summaries,
        "initial_snapshot": initial_snapshot,
    }

    report_dir = paths["report_dir"]
    write_json(report_dir / "validation_report.json", {"workspace_errors": validation_errors, "command_errors": command_errors, "load_errors": all_load_errors})
    write_json(report_dir / "incremental_semantic_report.json", incremental_semantic)
    write_json(report_dir / "blank_serial_semantic_report.json", blank_serial_semantic)
    write_json(report_dir / "blank_parallel_semantic_report.json", blank_parallel_semantic)
    write_json(report_dir / "incremental_ledger.json", incremental_ledger)
    write_json(report_dir / "command_ledger_series.json", {"series": ledger_series})
    write_json(report_dir / "comparison.json", comparison)
    write_json(report_dir / "budget_report.json", budget["reports"])
    write_json(report_dir / "injection_report.json", injection_report)
    write_json(report_dir / "flip_attribution_report.json", flip_attribution)
    write_json(report_dir / "snapshot_recovery.json", recovery)
    write_json(report_dir / "verification_summary.json", summary)
    return summary


def _status_counts(ledger: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in ledger:
        status = entry.get("cache_status", "unknown")
        counts[status] = counts.get(status, 0) + 1
    return dict(sorted(counts.items()))


def _verify_budget(config: dict[str, Any], samples: list[dict[str, Any]], snapshot_root: Path) -> dict[str, Any]:
    budget_config = copy.deepcopy(config)
    budget_config["point_budget"] = 140
    budget_config["sample_quota"] = 80
    engine = LabEngine(budget_config, copy.deepcopy(samples), global_version=1)
    first = engine.run(parallel=False)
    first_pending = list(first["pending_sample_ids"])
    first_admitted = list(first["newly_admitted_sample_ids"])

    store = SnapshotStore(snapshot_root)
    first_snapshot = commit_engine_snapshot(store, engine)

    second = engine.run({}, parallel=False)
    second_pending = list(second["pending_sample_ids"])
    second_admitted = list(second["newly_admitted_sample_ids"])
    second_snapshot = commit_engine_snapshot(store, engine)

    full_config = copy.deepcopy(budget_config)
    full_config["point_budget"] = 1000
    full_engine = LabEngine(full_config, copy.deepcopy(samples), global_version=1)
    full_report = full_engine.run(parallel=False)
    resumed_semantic = engine.semantic_report()
    full_semantic = full_engine.semantic_report()
    resume_compare_equal = canonical_json(_core_outputs(resumed_semantic)) == canonical_json(
        _core_outputs(full_semantic)
    )

    quota_config = copy.deepcopy(config)
    quota_config["point_budget"] = 1000
    quota_config["sample_quota"] = 20
    quota_engine = LabEngine(quota_config, copy.deepcopy(samples), global_version=1)
    quota_report = quota_engine.run(parallel=False)

    return {
        "first_admitted": first_admitted,
        "first_pending": first_pending,
        "second_admitted": second_admitted,
        "second_pending": second_pending,
        "resume_compare_equal": resume_compare_equal,
        "quota_exceeded": quota_report["quota_exceeded_sample_ids"],
        "quota_code_count": quota_report["code_counts"].get("E_POINT_QUOTA_EXCEEDED", 0),
        "pending_code_count_first": first["code_counts"].get("E_BUDGET_PENDING", 0),
        "snapshots": {"first": first_snapshot.get("commit_name"), "second": second_snapshot.get("commit_name")},
        "reports": {"first": first, "second": second, "quota": quota_report, "full_reference": full_report},
    }


def _core_outputs(semantic: dict[str, Any]) -> dict[str, Any]:
    return {
        "sample_ids": semantic.get("sample_ids"),
        "error_records": semantic.get("error_records"),
        "scheme_stage_matrix": semantic.get("scheme_stage_matrix"),
        "stats_records": semantic.get("stats_records"),
        "extrapolation_flips": semantic.get("extrapolation_flips"),
        "dimension_outputs": semantic.get("dimension_outputs"),
        "row_outputs": semantic.get("row_outputs"),
    }


def _verify_flip_attribution() -> dict[str, Any]:
    results = []
    for case in make_flip_attribution_cases():
        config = default_config()
        config["schemes"] = [scheme for scheme in config["schemes"] if scheme["name"] == "rope_standard"]
        config["train_length"] = 0
        config["target_length"] = 1000
        config["extrapolation_bins"] = [100000]
        config["fixed_point"].update(
            {
                "signed_bits": 64,
                "fraction_bits": int(case["fraction_bits"]),
                "accumulator_bits": int(case["accumulator_bits"]),
                "quant_level": "nearest",
            }
        )
        engine = LabEngine(config, [case["sample"]])
        report = engine.run(parallel=False)
        flips = report["extrapolation_flips"]
        hit = any(flip.get("source") == case["expected_source"] for flip in flips)
        results.append(
            {
                "case_id": case["case_id"],
                "expected_source": case["expected_source"],
                "hit": hit,
                "flip_count": len(flips),
                "flips": flips,
                "code_counts": report["code_counts"],
            }
        )
    return {"all_expected_sources_hit": all(item["hit"] for item in results), "cases": results}
