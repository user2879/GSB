#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gsb.config import validate_config  # noqa: E402
from gsb.edits import apply_commands  # noqa: E402
from gsb.engine import run_pipeline  # noqa: E402
from gsb.equivalence import canonical_state, diff_objects, execution_plan  # noqa: E402
from gsb.errors import E_INPUT_INVALID, PipelineError  # noqa: E402
from gsb.fixtures import base_batch_files, base_config, command_stream, write_fixture_tree  # noqa: E402
from gsb.samples import load_sample_set  # noqa: E402
from gsb.store import SnapshotStore  # noqa: E402


CheckResult = dict[str, Any]


def write_input(root: Path, raw_config: dict[str, Any], batches: dict[str, list[dict[str, Any]]]) -> dict[str, Path]:
    config_path = root / "configs" / "pipeline.json"
    sample_dir = root / "data" / "batches"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(raw_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for filename in sorted(batches):
        (sample_dir / filename).write_text(
            "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in batches[filename]) + "\n",
            encoding="utf-8",
        )
    return {"config": config_path, "samples": sample_dir}


def load_and_run(
    root: Path,
    raw_config: dict[str, Any],
    batches: dict[str, list[dict[str, Any]]],
    *,
    state_name: str = "state",
    budget: int | None = None,
    workers: int | None = 1,
) -> dict[str, Any]:
    paths = write_input(root / "input", raw_config, batches)
    config = validate_config(raw_config)
    sample_set = load_sample_set(paths["samples"], config)
    return run_pipeline(
        config,
        sample_set,
        root / state_name,
        budget=budget,
        workers=workers,
    )


class Reporter:
    def __init__(self) -> None:
        self.checks: list[CheckResult] = []

    def check(self, name: str, func: Callable[[], dict[str, Any] | None]) -> None:
        try:
            details = func() or {}
            self.checks.append({"name": name, "ok": True, **details})
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001 - verification records every failure
            tb = traceback.format_exc(limit=4)
            self.checks.append({"name": name, "ok": False, "error": str(exc), "traceback": tb})
            print(f"FAIL {name}: {exc}")


def assert_equal_canonical(left: dict[str, Any], right: dict[str, Any], label: str) -> None:
    diffs = diff_objects(canonical_state(left), canonical_state(right))
    if diffs:
        raise AssertionError(f"{label} canonical mismatch: {diffs[:5]}")


def expect_input_error(func: Callable[[], Any], field_fragment: str) -> PipelineError:
    try:
        func()
    except PipelineError as exc:
        if exc.code != E_INPUT_INVALID:
            raise AssertionError(f"expected {E_INPUT_INVALID}, got {exc.code}: {exc.to_dict()}") from exc
        if exc.field is None or field_fragment not in exc.field:
            raise AssertionError(f"expected field containing {field_fragment!r}, got {exc.field}") from exc
        return exc
    raise AssertionError(f"expected {E_INPUT_INVALID} with field {field_fragment}")


def check_input_validation() -> dict[str, Any]:
    details: list[dict[str, Any]] = []

    def config_case(name: str, mutate: Callable[[dict[str, Any]], None], fragment: str) -> None:
        raw = base_config()
        mutate(raw)
        err = expect_input_error(lambda: validate_config(raw), fragment)
        details.append({"case": name, "field": err.field, "batch_id": err.batch_id})

    config_case("negative_weight", lambda c: c["domains"][0].__setitem__("sampling_weight", -0.1), "sampling_weight")
    config_case("nonfinite_weight", lambda c: c["domains"][1].__setitem__("sampling_weight", float("nan")), "sampling_weight")
    config_case(
        "threshold_not_integer",
        lambda c: c["dedup"].__setitem__("similarity_threshold_permille", 800.5),
        "similarity_threshold_permille",
    )
    config_case("threshold_out_of_range", lambda c: c["dedup"].__setitem__("similarity_threshold_permille", 1001), "threshold")
    config_case("shard_size_zero", lambda c: c["dedup"].__setitem__("shard_size", 0), "shard_size")
    config_case("bad_decimal_width", lambda c: c["fixed_point"].__setitem__("decimal_width", 7), "decimal_width")
    config_case("bad_quant_tier", lambda c: c["fixed_point"].__setitem__("quantization_tier", "noisy"), "quantization_tier")
    config_case(
        "targets_not_sum_one",
        lambda c: c["domains"][0].__setitem__("target_ratio", 0.39),
        "target_ratio",
    )

    with tempfile.TemporaryDirectory(prefix="gsb-invalid-") as tmp:
        root = Path(tmp)

        def sample_case(name: str, mutate: Callable[[dict[str, list[dict[str, Any]]]], None], fragment: str) -> None:
            batches = base_batch_files()
            mutate(batches)
            paths = write_input(root / name, base_config(), batches)
            config = validate_config(base_config())
            err = expect_input_error(lambda: load_sample_set(paths["samples"], config), fragment)
            details.append({"case": name, "field": err.field, "batch_id": err.batch_id})

        def duplicate(batches: dict[str, list[dict[str, Any]]]) -> None:
            batches["alpha-batch-001.jsonl"][1]["sample_id"] = "alpha-s01"

        def negative_count(batches: dict[str, list[dict[str, Any]]]) -> None:
            for row in batches["alpha-batch-000.jsonl"]:
                if row.get("sample_id") == "alpha-s02":
                    row["token_counts"] = {"bad": -2}

        def bad_length_width(batches: dict[str, list[dict[str, Any]]]) -> None:
            for row in batches["alpha-batch-000.jsonl"]:
                if row.get("sample_id") == "alpha-s03":
                    row["length_distribution"].pop()

        def bad_bitmap_width(batches: dict[str, list[dict[str, Any]]]) -> None:
            for row in batches["beta-batch-000.jsonl"]:
                if row.get("sample_id") == "beta-s03":
                    row["missing_bitmap"].append(0)

        def undeclared_domain(batches: dict[str, list[dict[str, Any]]]) -> None:
            for row in batches["gamma-batch-000.jsonl"]:
                if row.get("sample_id") == "gamma-s03":
                    row["domain"] = "mars"

        sample_case("duplicate_sample_id", duplicate, "sample_id")
        sample_case("negative_token_count", negative_count, "token_counts")
        sample_case("length_width_mismatch", bad_length_width, "length_distribution")
        sample_case("bitmap_width_mismatch", bad_bitmap_width, "missing_bitmap")
        sample_case("undeclared_domain", undeclared_domain, "domain")

    return {"invalid_cases": details, "count": len(details)}


def check_incremental_sequence() -> dict[str, Any]:
    commands = command_stream()
    with tempfile.TemporaryDirectory(prefix="gsb-incremental-") as tmp:
        root = Path(tmp)
        inc_root = root / "incremental"
        inc_cfg = base_config()
        inc_batches = base_batch_files()
        inc_result = load_and_run(inc_root, inc_cfg, inc_batches, workers=2)
        versions = [inc_result["manifest"]["global_version"]]

        for index, command in enumerate(commands, start=1):
            inc_cfg, inc_batches = apply_commands(inc_cfg, inc_batches, [command])
            inc_result = load_and_run(inc_root, inc_cfg, inc_batches, workers=(index % 4) + 1)
            versions.append(inc_result["manifest"]["global_version"])

            full_cfg, full_batches = apply_commands(base_config(), base_batch_files(), commands[:index])
            full_result = load_and_run(root / f"full-{index}", full_cfg, full_batches, workers=1)
            assert_equal_canonical(inc_result, full_result, f"prefix-{index}")

        return {"prefixes_verified": len(commands), "versions": versions}


def check_precise_invalidation() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gsb-invalidate-") as tmp:
        root = Path(tmp)
        base_result = load_and_run(root, base_config(), base_batch_files(), workers=1)
        weight_cfg, _ = apply_commands(base_config(), base_batch_files(), [command_stream()[0]])
        weight_result = load_and_run(root, weight_cfg, base_batch_files(), workers=2)

        entries = weight_result["ledger"]["entries"]
        cleaning = {e["subject_id"]: e["disposition"] for e in entries if e["stage"] == "cleaning"}
        shards = {e["subject_id"]: e["disposition"] for e in entries if e["stage"] == "dedup_shard"}
        sampling = next(e for e in entries if e["stage"] == "sampling")
        summary = next(e for e in entries if e["stage"] == "summary")

        if not all(value == "reused" for value in cleaning.values()):
            raise AssertionError("weight edit must reuse all cleaning conclusions")
        alpha_shards = {k: v for k, v in shards.items() if "shard-alpha" in k}
        other_shards = {k: v for k, v in shards.items() if "shard-alpha" not in k}
        if not alpha_shards or any(v != "recomputed" for v in alpha_shards.values()):
            raise AssertionError(f"alpha shards should recompute: {alpha_shards}")
        if len(other_shards) != 4 or any(v != "reused" for v in other_shards.values()):
            raise AssertionError(f"non-alpha shards should reuse: {other_shards}")
        if sampling["disposition"] != "recomputed" or summary["disposition"] != "recomputed":
            raise AssertionError("sampling and summary must recompute after weight edit")

        base_order = base_result["sampling"]["global_order"]
        weight_order = weight_result["sampling"]["global_order"]
        if base_order == weight_order:
            raise AssertionError("weight edit did not change sampling order")
        base_dev = [d["deviation_fp"] for d in base_result["summary"]["domains"]]
        weight_dev = [d["deviation_fp"] for d in weight_result["summary"]["domains"]]
        if base_dev == weight_dev:
            raise AssertionError("weight edit did not change deviation conclusions")

        replace_root = root / "replace"
        load_and_run(replace_root, base_config(), base_batch_files(), workers=1)
        _, replaced_batches = apply_commands(base_config(), base_batch_files(), [command_stream()[1]])
        replace_result = load_and_run(replace_root, base_config(), replaced_batches, workers=2)
        r_entries = replace_result["ledger"]["entries"]
        r_clean = {e["subject_id"]: e["disposition"] for e in r_entries if e["stage"] == "cleaning"}
        r_shards = {e["subject_id"]: e["disposition"] for e in r_entries if e["stage"] == "dedup_shard"}
        if r_clean.get("alpha-s05") != "recomputed":
            raise AssertionError("replaced sample cleaning record should recompute")
        if any(k != "alpha-s05" and v != "reused" for k, v in r_clean.items()):
            raise AssertionError("only replaced sample cleaning should recompute")
        if r_shards.get("shard-alpha-0000") != "recomputed":
            raise AssertionError("containing alpha shard should recompute")
        unaffected = [k for k, v in r_shards.items() if k != "shard-alpha-0000"]
        if len(unaffected) != 5 or any(r_shards[k] != "reused" for k in unaffected):
            raise AssertionError(f"other five shards should reuse: {r_shards}")

        full_weight = load_and_run(root / "weight-full", weight_cfg, base_batch_files(), state_name="blank-state")
        full_replace = load_and_run(root / "replace-full", base_config(), replaced_batches, state_name="blank-state")
        assert_equal_canonical(weight_result, full_weight, "weight")
        assert_equal_canonical(replace_result, full_replace, "replace")

        return {
            "weight_alpha_shards_recomputed": sorted(alpha_shards),
            "weight_unaffected_shards_reused": sorted(other_shards),
            "replacement_recomputed_shards": ["shard-alpha-0000"],
            "sampling_changed": base_order != weight_order,
        }


def check_budget_concurrency() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gsb-budget-") as tmp:
        root = Path(tmp)
        reference = load_and_run(root / "reference", base_config(), base_batch_files(), workers=1)
        total_cost = reference["budget"]["cumulative_consumed_units"]
        if reference["manifest"]["status"] != "complete":
            raise AssertionError("reference did not complete")

        budget_cases = 0
        for budget in range(total_cost):
            case_root = root / f"budget-{budget:03d}"
            first = load_and_run(
                case_root,
                base_config(),
                base_batch_files(),
                budget=budget,
                workers=(budget % 4) + 1,
            )
            if first["manifest"]["status"] != E_INPUT_INVALID and first["manifest"]["status"] != "E_BUDGET_PENDING":
                if first["manifest"]["status"] != "E_BUDGET_PENDING":
                    raise AssertionError(f"budget {budget}: expected pending, got {first['manifest']['status']}")
            resumed = load_and_run(
                case_root,
                base_config(),
                base_batch_files(),
                budget=10**9,
                workers=((budget + 1) % 4) + 1,
            )
            if resumed["manifest"]["status"] != "complete":
                raise AssertionError(f"budget {budget}: resume failed {resumed['manifest']['status']}")
            assert_equal_canonical(reference, resumed, f"budget-{budget}")
            if execution_plan(reference) != execution_plan(resumed):
                diffs = diff_objects(execution_plan(reference), execution_plan(resumed))
                raise AssertionError(f"budget {budget}: execution plan mismatch {diffs[:3]}")
            budget_cases += 1

        worker_cases = []
        for workers in range(1, 5):
            result = load_and_run(root / f"workers-{workers}", base_config(), base_batch_files(), workers=workers)
            assert_equal_canonical(reference, result, f"workers-{workers}")
            if execution_plan(reference) != execution_plan(result):
                raise AssertionError(f"workers {workers}: execution plan mismatch")
            max_clean = result["budget"]["scheduler_stats"]["cleaning_max_active"]
            max_dedup = result["budget"]["scheduler_stats"]["dedup_max_active"]
            if max_clean > workers or max_dedup > workers:
                raise AssertionError("observed concurrency exceeded requested worker limit")
            worker_cases.append({"workers": workers, "cleaning_max": max_clean, "dedup_max": max_dedup})

        return {"budget_cut_points": budget_cases, "total_work_units": total_cost, "worker_cases": worker_cases}


def check_warm_budget_plan() -> dict[str, Any]:
    cut_points = [0, 1, 15, 16, 17, 22, 23, 24, 25]
    with tempfile.TemporaryDirectory(prefix="gsb-warm-budget-") as tmp:
        root = Path(tmp)
        weight_cfg, _ = apply_commands(base_config(), base_batch_files(), [command_stream()[0]])
        ref_root = root / "reference"
        load_and_run(ref_root, base_config(), base_batch_files(), workers=1)
        reference = load_and_run(ref_root, weight_cfg, base_batch_files(), workers=1)

        for budget in cut_points:
            case_root = root / f"warm-{budget}"
            load_and_run(case_root, base_config(), base_batch_files(), workers=1)
            first = load_and_run(case_root, weight_cfg, base_batch_files(), budget=budget, workers=2)
            if first["manifest"]["status"] == "E_BUDGET_PENDING":
                final = load_and_run(case_root, weight_cfg, base_batch_files(), budget=10**9, workers=3)
            else:
                final = first
            assert_equal_canonical(reference, final, f"warm-budget-{budget}")
            if execution_plan(reference) != execution_plan(final):
                diffs = diff_objects(execution_plan(reference), execution_plan(final))
                raise AssertionError(f"warm budget {budget}: plan mismatch {diffs[:4]}")
        return {"warm_budget_cut_points": cut_points}


def check_random_edit_sequences() -> dict[str, Any]:
    sequence_count = 12
    max_length = 5
    weight_choices = (0.1, 0.2, 0.5, 0.9, 1.5, 2.5)
    threshold_choices = (700, 750, 800, 850, 900)
    shard_choices = (4, 5, 6, 8)
    token_choices = (0, 3, 5, 8)

    with tempfile.TemporaryDirectory(prefix="gsb-random-edits-") as tmp:
        root = Path(tmp)
        verified_lengths: list[int] = []
        for sequence_index in range(sequence_count):
            rng = random.Random(9127 + sequence_index)
            commands = []
            for command_index in range(rng.randint(2, max_length)):
                command_id = f"random-{sequence_index}-{command_index}"
                kind = rng.randrange(4)
                if kind == 0:
                    commands.append(
                        {
                            "command_id": command_id,
                            "op": "set_weight",
                            "domain": rng.choice(("alpha", "beta", "gamma")),
                            "weight": rng.choice(weight_choices),
                        }
                    )
                elif kind == 1:
                    commands.append(
                        {
                            "command_id": command_id,
                            "op": "set_threshold",
                            "threshold_permille": rng.choice(threshold_choices),
                        }
                    )
                elif kind == 2:
                    commands.append(
                        {
                            "command_id": command_id,
                            "op": "set_shard_size",
                            "shard_size": rng.choice(shard_choices),
                        }
                    )
                else:
                    commands.append(
                        {
                            "command_id": command_id,
                            "op": "set_min_tokens",
                            "min_total_token_count": rng.choice(token_choices),
                        }
                    )

            seq_root = root / f"seq-{sequence_index:02d}"
            raw_config = base_config()
            batches = base_batch_files()
            load_and_run(seq_root, raw_config, batches, workers=1)
            for command_index, command in enumerate(commands):
                raw_config, batches = apply_commands(raw_config, batches, [command])
                result = load_and_run(
                    seq_root,
                    raw_config,
                    batches,
                    workers=(command_index % 4) + 1,
                )
                if result["manifest"]["status"] != "complete":
                    raise AssertionError(
                        f"random sequence {sequence_index} did not complete: {result['manifest']['status']}"
                    )

            final_config, final_batches = apply_commands(base_config(), base_batch_files(), commands)
            full = load_and_run(root / f"full-{sequence_index:02d}", final_config, final_batches, workers=1)
            incremental = SnapshotStore(seq_root / "state").load_current()
            assert_equal_canonical(incremental, full, f"random-sequence-{sequence_index}")
            verified_lengths.append(len(commands))

        return {"sequences": sequence_count, "lengths": verified_lengths}


def check_business_codes() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gsb-codes-") as tmp:
        root = Path(tmp)
        quota_cfg = base_config()
        quota_cfg["domains"][0]["quota"] = 3
        quota_result = load_and_run(root / "quota", quota_cfg, base_batch_files())
        if quota_result["manifest"]["status"] != "E_DOMAIN_QUOTA_EXCEEDED":
            raise AssertionError("quota code not returned")
        if quota_result["error"]["error_code"] != "E_DOMAIN_QUOTA_EXCEEDED":
            raise AssertionError("quota error payload code mismatch")

        pending_result = load_and_run(root / "pending", base_config(), base_batch_files(), budget=0)
        if pending_result["manifest"]["business_code"] != "E_BUDGET_PENDING":
            raise AssertionError("budget pending code not returned")
        if not pending_result["queue"]["items"] or pending_result["queue"]["next_priority_subject"] is None:
            raise AssertionError("pending queue missing first subject")

        limit_cfg = base_config()
        limit_cfg["budgets"]["shard_comparison_limit"] = 1
        limit_result = load_and_run(root / "limit", limit_cfg, base_batch_files(), budget=10**9)
        if limit_result["manifest"]["status"] != "E_SHARD_COMPARISON_LIMIT":
            raise AssertionError(f"comparison limit code not returned: {limit_result['manifest']['status']}")

        store = SnapshotStore(root / "closeout")
        report_path = store.write_closeout_report({"ok": True, "reason": "unit-check", "handled_exit": True})
        if not report_path.exists():
            raise AssertionError("closeout report was not written")

        return {
            "quota_field": quota_result["error"]["field"],
            "quota_batch": quota_result["error"]["batch_id"],
            "pending_first": pending_result["queue"]["next_priority_subject"],
            "comparison_limit_status": limit_result["manifest"]["status"],
        }


def check_crash_recovery() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="gsb-crash-") as tmp:
        root = Path(tmp)
        fixture_root = root / "fixture"
        write_fixture_tree(fixture_root)
        env = {**os.environ, "PYTHONPATH": str(ROOT)}
        base_cmd = [sys.executable, "-m", "gsb.cli", "--root-help"]  # replaced below

        def run_cli(cwd_root: Path, *extra: str, check: bool = True) -> subprocess.CompletedProcess[str]:
            cmd = [sys.executable, "-m", "gsb.cli", "run", *extra]
            result = subprocess.run(
                cmd,
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            if check and result.returncode != 0:
                raise AssertionError(f"CLI failed {result.returncode}: {result.stderr}\n{result.stdout}")
            return result

        config = fixture_root / "configs" / "pipeline.json"
        samples = fixture_root / "data" / "batches"
        state = fixture_root / "state"
        run_cli(fixture_root, "--config", str(config), "--samples", str(samples), "--state", str(state))

        edited_cfg, _ = apply_commands(base_config(), base_batch_files(), [command_stream()[0]])
        edited_root = fixture_root / "edited"
        edited_paths = write_input(edited_root, edited_cfg, base_batch_files())
        crashed = run_cli(
            fixture_root,
            "--config",
            str(edited_paths["config"]),
            "--samples",
            str(edited_paths["samples"]),
            "--state",
            str(state),
            "--crash-before-commit",
            check=False,
        )
        if crashed.returncode not in (-9, 137):
            raise AssertionError(f"expected SIGKILL-like 137/-9, got {crashed.returncode}: {crashed.stderr}")
        current = SnapshotStore(state).load_current()
        if current["manifest"]["global_version"] != 1:
            raise AssertionError("SIGKILL changed committed current snapshot")

        recovered = run_cli(
            fixture_root,
            "--config",
            str(edited_paths["config"]),
            "--samples",
            str(edited_paths["samples"]),
            "--state",
            str(state),
            "--workers",
            "2",
        )
        recovered_json = json.loads(recovered.stdout)
        if not recovered_json["ok"]:
            raise AssertionError(f"recovery CLI did not complete: {recovered.stdout} {recovered.stderr}")

        full_root = root / "full"
        full_paths = write_input(full_root / "input", edited_cfg, base_batch_files())
        full_state = full_root / "state-full"
        run_cli(
            full_root,
            "--config",
            str(full_paths["config"]),
            "--samples",
            str(full_paths["samples"]),
            "--state",
            str(full_state),
            "--workers",
            "1",
        )
        recovered_snapshot = SnapshotStore(state).load_current()
        full_snapshot = SnapshotStore(full_state).load_current()
        assert_equal_canonical(recovered_snapshot, full_snapshot, "crash-recovery")

        closeout = subprocess.run(
            [sys.executable, "-m", "gsb.cli", "write-closeout", "--state", str(state)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        closeout_json = json.loads(closeout.stdout)
        if not closeout_json["ok"]:
            raise AssertionError("closeout command failed")

        return {"kill_returncode": crashed.returncode, "recovered_global_version": recovered_json["global_version"]}


def check_optional_tk() -> dict[str, Any]:
    try:
        import tkinter  # noqa: F401

        tk_available = True
        error = None
    except Exception as exc:  # noqa: BLE001
        tk_available = False
        error = f"{type(exc).__name__}: {exc}"
    return {"tkinter_importable": tk_available, "note": error or "python3-tk is available"}


def main() -> int:
    reporter = Reporter()
    reporter.check("input_validation_business_codes", check_input_validation)
    reporter.check("incremental_matches_full_rebuild", check_incremental_sequence)
    reporter.check("random_edit_sequences", check_random_edit_sequences)
    reporter.check("precise_fingerprint_invalidation", check_precise_invalidation)
    reporter.check("budget_cutpoints_and_concurrency", check_budget_concurrency)
    reporter.check("warm_budget_reuse_plan_stability", check_warm_budget_plan)
    reporter.check("business_codes_quota_pending_limit", check_business_codes)
    reporter.check("sigkill_recovery_matches_full", check_crash_recovery)
    reporter.check("optional_tkinter_diagnostic", check_optional_tk)

    total = len(reporter.checks)
    passed = sum(1 for check in reporter.checks if check["ok"])
    report = {
        "ok": passed == total,
        "passed": passed,
        "total": total,
        "checks": reporter.checks,
    }
    artifacts = ROOT / "artifacts"
    artifacts.mkdir(exist_ok=True)
    report_path = artifacts / "acceptance_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport written: {report_path}")
    print(f"{passed}/{total} acceptance groups passed")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
