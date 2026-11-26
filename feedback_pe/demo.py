from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path
from typing import Any

from .commands import apply_command, load_command_stream
from .config import load_config
from .engine import Workspace
from .generator import generate_project_files
from .hashing import canonical_json
from .analysis import build_error_trace
from .pipeline import compute_encoding_and_scores
from .reporting import build_report
from .runner import fresh_full_run, run_batch
from .samples import load_samples_jsonl


def write_json(path: str | Path, value: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _business_equal(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return canonical_json(left["business"]) == canonical_json(right["business"])


def run_demo(root: str | Path = ".") -> dict[str, Any]:
    root = Path(root)
    paths = generate_project_files(root)
    config, config_errors = load_config(paths["config"])
    if config_errors:
        raise RuntimeError(f"invalid generated config: {config_errors}")
    loaded = load_samples_jsonl(paths["samples"], int(config["dimension"]))
    write_json(root / "out" / "load_validation.json", {
        "accepted_count": len(loaded.samples),
        "failure_count": len(loaded.failures),
        "failures": loaded.failures,
        "code_counts": loaded.code_counts,
    })

    unlimited_config = copy.deepcopy(config)
    unlimited_config["batch_point_budget"] = 10**9
    unlimited_config["sample_point_quota"] = 10**9
    full = fresh_full_run(root / "state" / "full", unlimited_config,
                          copy.deepcopy(loaded.samples), loaded.failures)
    write_json(root / "out" / "full_report.json", full["report"])

    inc_config = copy.deepcopy(config)
    samples_by_id = {item["id"]: copy.deepcopy(item) for item in loaded.samples}
    commands, command_failures = load_command_stream(paths["commands"])
    incremental_root = root / "state" / "incremental"
    _reset_directory(incremental_root)
    workspace = Workspace(incremental_root)
    command_results: list[dict[str, Any]] = []
    last_summary = None
    for command in commands:
        try:
            result = apply_command(inc_config, samples_by_id, command)
            result["status"] = "applied"
            command_results.append(result)
            if result["run_batch"]:
                last_summary = run_batch(
                    workspace, inc_config, list(samples_by_id.values()), loaded.failures,
                    batch_budget=result.get("batch_budget"),
                    sample_quota=result.get("sample_quota"),
                )
        except Exception as exc:  # demo stream is valid; preserve unexpected business errors
            command_results.append({
                "op": command.get("op"),
                "line_number": command.get("_line_number"),
                "status": "error",
                "code": getattr(exc, "code", "E_COMMAND"),
                "field": getattr(exc, "field", "$"),
                "message": str(exc),
            })

    incremental_report = build_report(
        inc_config, list(samples_by_id.values()), loaded.failures, workspace, last_summary,
    )
    write_json(root / "out" / "incremental_report.json", incremental_report)
    write_json(root / "out" / "command_results.json", {
        "command_failures": command_failures,
        "results": command_results,
    })
    write_json(root / "state" / "ui_state.json", {
        "config": inc_config,
        "samples": list(samples_by_id.values()),
        "failures": loaded.failures,
        "workspace": "state/incremental",
        "report": "out/incremental_report.json",
    })

    final_full = fresh_full_run(
        root / "state" / "full_from_final",
        copy.deepcopy(inc_config),
        copy.deepcopy(list(samples_by_id.values())),
        loaded.failures,
    )
    equivalence = {
        "canonical_business_fields_equal": _business_equal(
            incremental_report, final_full["report"],
        ),
        "incremental_task_count": len(incremental_report["business"]["task_manifest"]),
        "full_task_count": len(final_full["report"]["business"]["task_manifest"]),
        "incremental_flip_count": len(incremental_report["business"]["flip_attribution"]),
        "full_flip_count": len(final_full["report"]["business"]["flip_attribution"]),
        "comparison": "canonical JSON with sorted keys and stable float repr",
    }
    write_json(root / "out" / "equivalence.json", equivalence)

    inc_manifest_keys = [
        item["sort_key"] for item in incremental_report["business"]["task_manifest"]
    ]
    full_manifest_keys = [
        item["sort_key"] for item in final_full["report"]["business"]["task_manifest"]
    ]
    latest = incremental_report.get("execution", {}).get("latest_run") or {}
    inc_reuse_keys = sorted(item["sort_key"] for item in latest.get("reuse", []))
    inc_recompute_keys = sorted(item["sort_key"] for item in latest.get("recompute", []))
    full_recompute_keys = sorted(
        item["sort_key"] for item in final_full["summary"]["recompute"]
    )
    list_equivalence = {
        "canonical_business_fields_equal": equivalence["canonical_business_fields_equal"],
        "task_manifest_order_equal": inc_manifest_keys == full_manifest_keys,
        "failure_order_equal": [
            item["sample_id"] for item in incremental_report["business"]["failures"]
        ] == [
            item["sample_id"] for item in final_full["report"]["business"]["failures"]
        ],
        "incremental_reuse_count": len(inc_reuse_keys),
        "incremental_recompute_count": len(inc_recompute_keys),
        "incremental_reuse_recompute_union_matches_manifest": sorted(
            inc_reuse_keys + inc_recompute_keys,
        ) == inc_manifest_keys,
        "blank_full_reuse_count": 0,
        "blank_full_recompute_count": len(full_recompute_keys),
        "blank_full_recompute_matches_manifest": full_recompute_keys == full_manifest_keys,
        "note": (
            "A blank cache has no historical hits by definition; stable identities, "
            "order, union coverage, and all payload fields are compared here."
        ),
    }
    write_json(root / "out" / "reuse_recompute_equivalence.json", list_equivalence)

    concurrency = _concurrency_check(root, config, loaded.samples, loaded.failures)
    write_json(root / "out" / "concurrency_equivalence.json", concurrency)

    budget = _budget_checks(root, config, loaded.samples, loaded.failures)
    write_json(root / "out" / "budget_demo.json", budget)

    invalidation = _invalidation_checks(root, config, loaded.samples, loaded.failures)
    write_json(root / "out" / "invalidation_demo.json", invalidation)

    recovery = _recovery_check(root, incremental_root)
    write_json(root / "out" / "recovery.json", recovery)

    flip_summary = {
        "flip_attribution": full["report"]["business"]["flip_attribution"],
        "cross_attribution": full["report"]["business"]["cross_attribution"],
    }
    write_json(root / "out" / "flip_attribution.json", flip_summary)
    stage_profiles = _stage_profile_check(config, loaded.samples)
    write_json(root / "out" / "stage_attribution_profiles.json", stage_profiles)

    summary = {
        "load": {
            "accepted": len(loaded.samples),
            "failures": len(loaded.failures),
            "code_counts": loaded.code_counts,
        },
        "equivalence": equivalence,
        "concurrency": concurrency,
        "budget": {
            "budget_batch_1_pending": budget["batch_budget"]["batch_1"]["pending_count"],
            "budget_batch_2_completed": budget["batch_budget"]["batch_2"]["completed_count"],
            "quota_pending": budget["sample_quota"]["batch_1"]["pending_count"],
            "quota_recovered": budget["sample_quota"]["batch_2"]["pending_count"] == 0,
        },
        "invalidation": invalidation["assertions"],
        "list_equivalence": list_equivalence,
        "recovery": recovery,
        "stage_profiles": stage_profiles["profile_counts"],
        "outputs": [
            "out/load_validation.json", "out/full_report.json",
            "out/incremental_report.json", "out/equivalence.json",
            "out/reuse_recompute_equivalence.json",
            "out/concurrency_equivalence.json", "out/budget_demo.json",
            "out/invalidation_demo.json", "out/recovery.json",
            "out/flip_attribution.json",
            "out/stage_attribution_profiles.json",
        ],
    }
    write_json(root / "out" / "demo_summary.json", summary)
    return summary


def _reset_directory(path: Path) -> None:
    import shutil
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _concurrency_check(root: Path, config: dict[str, Any],
                       samples: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    run_config = copy.deepcopy(config)
    run_config["workers"] = 4
    run_config["batch_point_budget"] = 10**9
    run_config["sample_point_quota"] = 10**9
    serial_root = root / "state" / "concurrency_serial"
    thread_root = root / "state" / "concurrency_thread"
    _reset_directory(serial_root)
    _reset_directory(thread_root)
    serial_ws = Workspace(serial_root)
    thread_ws = Workspace(thread_root)
    serial_summary = run_batch(serial_ws, run_config, samples, failures, parallel=False)
    thread_summary = run_batch(thread_ws, run_config, samples, failures, parallel=True)
    serial_report = build_report(run_config, samples, failures, serial_ws, serial_summary)
    thread_report = build_report(run_config, samples, failures, thread_ws, thread_summary)
    equal = _business_equal(serial_report, thread_report)
    return {
        "same_config_workers": 4,
        "canonical_business_fields_equal": equal,
        "serial_executed_task_count": serial_summary["scheduler"]["executed_task_count"],
        "thread_executed_task_count": thread_summary["scheduler"]["executed_task_count"],
    }


def _budget_checks(root: Path, config: dict[str, Any], samples: list[dict[str, Any]],
                   failures: list[dict[str, Any]]) -> dict[str, Any]:
    budget_config = copy.deepcopy(config)
    budget_config["workers"] = 1
    budget_root = root / "state" / "budget"
    _reset_directory(budget_root)
    ws = Workspace(budget_root)
    first = run_batch(ws, budget_config, samples, failures,
                       batch_budget=104, sample_quota=80)
    second = run_batch(ws, budget_config, samples, failures,
                       batch_budget=104, sample_quota=80)

    quota_config = copy.deepcopy(config)
    quota_config["workers"] = 1
    quota_root = root / "state" / "quota"
    _reset_directory(quota_root)
    quota_ws = Workspace(quota_root)
    quota_first = run_batch(quota_ws, quota_config, samples, failures,
                            batch_budget=10**9, sample_quota=80)
    quota_second = run_batch(quota_ws, quota_config, samples, failures,
                             batch_budget=10**9, sample_quota=128)
    return {
        "batch_budget": {
            "budget_points": 104,
            "batch_1": _public_batch(first["scheduler"]),
            "batch_2": _public_batch(second["scheduler"]),
        },
        "sample_quota": {
            "quota_points_first": 80,
            "quota_points_second": 128,
            "batch_1": _public_batch(quota_first["scheduler"]),
            "batch_2": _public_batch(quota_second["scheduler"]),
        },
    }


def _public_batch(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        key: batch[key]
        for key in (
            "batch_id", "budget_total", "sample_point_quota", "spent_points",
            "remaining_points", "ready_sample_ids", "pending_sample_ids",
            "completed_count", "pending_count", "pending_entries",
        )
    }


def _invalidation_checks(root: Path, config: dict[str, Any], samples: list[dict[str, Any]],
                         failures: list[dict[str, Any]]) -> dict[str, Any]:
    inv_config = copy.deepcopy(config)
    inv_config["batch_point_budget"] = 10**9
    inv_config["sample_point_quota"] = 10**9
    inv_root = root / "state" / "invalidation"
    _reset_directory(inv_root)
    ws = Workspace(inv_root)
    run_batch(ws, inv_config, samples, failures)

    length_command = {"op": "set_lengths", "train_length": 20, "target_length": 64}
    length_samples = {item["id"]: item for item in copy.deepcopy(samples)}
    apply_command(inv_config, length_samples, length_command)
    length_run = run_batch(ws, inv_config, list(length_samples.values()), failures)
    length_reuse_stages = sorted({item["stage"] for item in length_run["reuse"]})
    length_recompute_stages = sorted({item["stage"] for item in length_run["recompute"]})
    length_ok = length_reuse_stages == ["position", "score"] and set(
        length_recompute_stages,
    ) == {"stats", "error"}

    base_command = {
        "op": "set_base_freq", "scheme": "sinusoidal", "base_freq": 4321.0,
    }
    apply_command(inv_config, length_samples, base_command)
    base_run = run_batch(ws, inv_config, list(length_samples.values()), failures)
    sinusoidal_recomputed = [
        item for item in base_run["recompute"] if item["scheme"] == "sinusoidal"
    ]
    other_reuse = [
        item for item in base_run["reuse"] if item["scheme"] != "sinusoidal"
    ]
    base_ok = (
        {item["stage"] for item in sinusoidal_recomputed}
        == {"position", "score", "stats", "error"}
        and len(other_reuse) > 0
        and all(item["scheme"] != "sinusoidal" for item in other_reuse)
    )
    return {
        "length_change": {
            "reused_stages": length_reuse_stages,
            "recomputed_stages": length_recompute_stages,
            "invalidation_count": len(length_run["invalidations"]),
            "ok": length_ok,
        },
        "base_freq_change": {
            "sinusoidal_recomputed_task_count": len(sinusoidal_recomputed),
            "other_scheme_reuse_count": len(other_reuse),
            "sample_basis": [
                record["basis"] for record in base_run["invalidations"]
                if record["scheme"] == "sinusoidal" and record["stage"] == "position"
            ][:1],
            "ok": base_ok,
        },
        "assertions": {
            "length_only_invalidates_stats_error": length_ok,
            "base_freq_only_invalidates_named_scheme": base_ok,
        },
    }


def _recovery_check(root: Path, incremental_root: Path) -> dict[str, Any]:
    killed_dir = Path(tempfile.mkdtemp(prefix="tx-killed-demo-", dir=incremental_root))
    (killed_dir / "partial.json").write_text('{"partial": true}\n', encoding="utf-8")
    reloaded = Workspace(incremental_root)
    return {
        "recovery": reloaded.recovery,
        "global_version": reloaded.global_version,
        "killed_temp_directory_ignored": killed_dir.name in {
            path.name for path in incremental_root.glob("tx-killed-demo-*")
        },
        "current_pointer_exists": (incremental_root / "CURRENT.json").exists(),
    }


def _stage_profile_check(config: dict[str, Any],
                         samples: list[dict[str, Any]]) -> dict[str, Any]:
    profiles = {
        "position_first": {
            "position_rel": 1.0e-6, "position_abs": 1.0e-12,
            "score_rel": 1.0e9, "score_abs": 1.0e9,
            "weight_rel": 1.0e9, "weight_abs": 1.0e9,
        },
        "score_first": {
            "position_rel": 1.0e9, "position_abs": 1.0e9,
            "score_rel": 1.0e-12, "score_abs": 1.0e-12,
            "weight_rel": 1.0e9, "weight_abs": 1.0e9,
        },
        "weight_first": {
            "position_rel": 1.0e9, "position_abs": 1.0e9,
            "score_rel": 1.0e9, "score_abs": 1.0e9,
            "weight_rel": 1.0e-12, "weight_abs": 1.0e-12,
        },
    }
    scheme = config["schemes"][0]
    result: dict[str, Any] = {"profile_counts": {}, "examples": {}}
    for profile, tolerances in profiles.items():
        profile_config = copy.deepcopy(config)
        profile_config["tolerances"].update(tolerances)
        counts = {"position": 0, "score": 0, "weight": 0, "none": 0}
        zero_reference_traces = 0
        example = None
        for sample in samples:
            bundle = compute_encoding_and_scores(
                sample, scheme, int(config["frac_bits"]), str(config["quant_tier"]),
            )
            trace = build_error_trace(sample, scheme, profile_config, bundle)
            counts[trace["first_exceeded_stage"] or "none"] += 1
            zero_reference_traces += trace["stages"]["position"][
                "absolute_zero_reference_count"
            ]
            if example is None and trace["first_exceeded_stage"] is not None:
                example = {
                    "sample_id": sample["id"],
                    "first_exceeded_stage": trace["first_exceeded_stage"],
                    "first_exceedance": trace["first_exceedance"],
                }
        result["profile_counts"][profile] = counts
        result["examples"][profile] = example
        result.setdefault("zero_reference_branch_count", zero_reference_traces)
    result["all_three_stages_observable"] = all(
        result["profile_counts"][name][name.split("_")[0]] > 0
        for name in profiles
    )
    return result


def main() -> None:
    summary = run_demo(Path.cwd())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
