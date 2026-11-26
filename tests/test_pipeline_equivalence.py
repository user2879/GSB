from __future__ import annotations

import copy
import random
import tempfile
from decimal import Decimal
from pathlib import Path

from fixed_lab.config import load_config
from fixed_lab.dataset import apply_commands, dataset_from_records, load_command_stream, load_samples_jsonl
from fixed_lab.hashing import canonical_dumps
from fixed_lab.pipeline import run_pipeline
from fixed_lab.pathing import path_comparison_index
from fixed_lab.solvers import solve_ridge_path_level


ROOT = Path(__file__).resolve().parents[1]


def base_config():
    config = load_config(ROOT / "configs/lab.json")
    for path in config["paths"].values():
        path["alphas"] = [Decimal("1"), Decimal("8"), Decimal("32")]
        path["solver"]["max_iterations"] = 180
    config["scheduler"].update({"worker_count": 2, "mac_budget_per_batch": 80, "single_column_quota": 40})
    return config


def base_dataset(config):
    records, errors = load_samples_jsonl(ROOT / "data/samples.jsonl", config["columns"])
    assert not errors
    return dataset_from_records(config["columns"], records)


def test_ridge_lasso_pipeline_produces_deterministic_paths() -> None:
    result = run_pipeline(base_dataset(base_config()), base_config(), cache_dir=None, worker_count=1)
    report = result["report"]
    assert report["status"] == "OK"
    assert len(report["paths"]["ridge"]["levels"][0]["coefficients"]) == 6
    assert all(level["converged"] for level in report["paths"]["lasso"]["levels"])
    assert any(pair["category"] == "near_collinear" for pair in report["correlation"]["pairs"])
    audit = result["trace"]["order_audits"]["lasso:1"]
    assert any(item["order_effect_observed"] for item in audit["pair_effects"])
    assert any(batch["deferred"] for batch in report["accumulation"]["budget_plan"])


def test_ridge_low_pivot_fallback_is_recorded() -> None:
    result = solve_ridge_path_level(
        [[100]],
        [1],
        type("P", (), {"accum_scale": 10_000, "scale": 100})(),
        Decimal("1"),
        {
            "pivot_abs_threshold": 10**30,
            "fallback_factor": Decimal("0.5"),
            "max_fallbacks": 3,
        },
    )
    assert result["fallback_count"] == 3
    assert len(result["attempts"]) == 4
    assert result["status"] == "E_SOLVER_LOW_PIVOT"


def test_path_remains_comparable_when_alpha_set_changes() -> None:
    config = base_config()
    dataset = base_dataset(config)
    first = run_pipeline(dataset, copy.deepcopy(config), cache_dir=None, worker_count=1)

    changed = copy.deepcopy(config)
    changed["paths"]["lasso"]["alphas"] = [Decimal("2"), Decimal("8")]
    changed["paths"]["ridge"]["alphas"] = [Decimal("2"), Decimal("8")]
    second = run_pipeline(dataset, changed, cache_dir=None, worker_count=1)

    for path_name in ["ridge", "lasso"]:
        before = path_comparison_index(first["report"]["paths"][path_name])
        after = path_comparison_index(second["report"]["paths"][path_name])
        assert "8" in before and "8" in after
        assert before["8"]["coefficients_fixed_raw"] == after["8"]["coefficients_fixed_raw"]
        assert second["report"]["paths"][path_name]["alpha_keys"] == ["2", "8"]


def test_demo_edits_incremental_matches_blank_rebuild() -> None:
    config = base_config()
    dataset = base_dataset(config)
    commands, parse_errors = load_command_stream(ROOT / "edits/demo_edits.jsonl")
    assert not parse_errors

    cache_dir = tempfile.mkdtemp(prefix="lab-inc-")
    initial = run_pipeline(dataset, copy.deepcopy(config), cache_dir=cache_dir, worker_count=2)
    previous = {entry["artifact_id"]: entry for entry in initial["ledger"]["entries"]}

    inc_dataset, inc_config, command_results, command_errors = apply_commands(dataset, copy.deepcopy(config), commands)
    assert not command_errors
    incremental = run_pipeline(
        inc_dataset,
        inc_config,
        cache_dir=cache_dir,
        previous_lineage=previous,
        worker_count=2,
        command_results=command_results,
    )

    full_dataset, full_config, full_results, full_errors = apply_commands(dataset, copy.deepcopy(config), commands)
    assert not full_errors
    full = run_pipeline(
        full_dataset,
        full_config,
        cache_dir=tempfile.mkdtemp(prefix="lab-full-"),
        worker_count=2,
        command_results=full_results,
    )

    assert canonical_dumps(incremental["report"]) == canonical_dumps(full["report"])
    assert [entry["artifact_id"] for entry in incremental["report"]["lineage_checklist"]] == [
        entry["artifact_id"] for entry in full["report"]["lineage_checklist"]
    ]


def test_stepwise_incremental_demo_commands_match_rebuild(tmp_path: Path) -> None:
    config = base_config()
    dataset = base_dataset(config)
    commands, parse_errors = load_command_stream(ROOT / "edits/demo_edits.jsonl")
    assert not parse_errors

    cache_dir = tmp_path / "step-cache"
    current_dataset = dataset
    current_config = copy.deepcopy(config)
    initial = run_pipeline(current_dataset, current_config, cache_dir=cache_dir, worker_count=2)
    previous = {entry["artifact_id"]: entry for entry in initial["ledger"]["entries"]}
    cumulative_results = []

    for command_index, command in enumerate(commands):
        current_dataset, current_config, results, errors = apply_commands(
            current_dataset, current_config, [command]
        )
        assert not errors, errors
        for result in results:
            result["index"] = command_index
        cumulative_results.extend(results)
        step = run_pipeline(
            current_dataset,
            current_config,
            cache_dir=cache_dir,
            previous_lineage=previous,
            worker_count=2,
            command_results=cumulative_results,
        )
        previous = {entry["artifact_id"]: entry for entry in step["ledger"]["entries"]}

    full_dataset, full_config, full_results, full_errors = apply_commands(
        dataset, copy.deepcopy(config), commands
    )
    assert not full_errors
    full = run_pipeline(full_dataset, full_config, cache_dir=None, worker_count=1, command_results=full_results)
    assert canonical_dumps(step["report"]) == canonical_dumps(full["report"])


def test_thread_pool_matches_serial_field_by_field() -> None:
    config = base_config()
    dataset = base_dataset(config)
    commands, _ = load_command_stream(ROOT / "edits/demo_edits.jsonl")
    edited, edited_config, results, errors = apply_commands(dataset, copy.deepcopy(config), commands)
    assert not errors
    serial = run_pipeline(edited, edited_config, cache_dir=None, worker_count=1, command_results=results)
    parallel = run_pipeline(edited, copy.deepcopy(edited_config), cache_dir=None, worker_count=4, command_results=results)
    assert canonical_dumps(serial["report"]) == canonical_dumps(parallel["report"])


def test_sample_and_column_cache_invalidation_is_precise() -> None:
    config = base_config()
    dataset = base_dataset(config)
    cache_dir = tempfile.mkdtemp(prefix="lab-cache-")
    initial = run_pipeline(dataset, copy.deepcopy(config), cache_dir=cache_dir, worker_count=1)
    previous = {entry["artifact_id"]: entry for entry in initial["ledger"]["entries"]}

    replacement = [{
        "op": "replace_sample",
        "id": "S0003",
        "sample": {"id": "S0003", "seed": 7, "features": [0.01] * 6, "target": 0.2, "weight": 1.0},
    }]
    edited, edited_config, _, errors = apply_commands(dataset, copy.deepcopy(config), replacement)
    assert not errors
    result = run_pipeline(edited, edited_config, cache_dir=cache_dir, previous_lineage=previous, worker_count=1)
    mac = [entry for entry in result["ledger"]["entries"] if entry["stage"] == "accumulation_mac"]
    assert sum(entry["status"] == "rebuilt" for entry in mac) == 27
    assert sum(entry["status"] == "reused" for entry in mac) == 17 * 27

    previous = {entry["artifact_id"]: entry for entry in result["ledger"]["entries"]}
    column_edit = [{"op": "update_column", "name": "base_signal", "values": {"S0002": 0.333333}}]
    edited2, config2, _, errors = apply_commands(edited, edited_config, column_edit)
    assert not errors
    result2 = run_pipeline(edited2, config2, cache_dir=cache_dir, previous_lineage=previous, worker_count=1)
    mac2 = [entry for entry in result2["ledger"]["entries"] if entry["stage"] == "accumulation_mac"]
    rebuilt = [entry["artifact_id"] for entry in mac2 if entry["status"] == "rebuilt"]
    assert len(rebuilt) == 7
    assert all("sample=S0002" in artifact for artifact in rebuilt)


def _random_command_sequence(seed: int, columns: list[str], ids: list[str]) -> list[dict]:
    rng = random.Random(seed)
    commands: list[dict] = []
    current_columns = list(columns)
    current_ids = list(ids)
    next_id = 1000
    for step in range(12):
        choices = ["replace", "update_column", "add_column", "alphas", "fixed"]
        if len(current_ids) > 6:
            choices.append("delete_sample")
        if len(current_columns) > 2:
            choices.append("delete_column")
        op = rng.choice(choices)
        if op == "replace":
            sid = rng.choice(current_ids)
            commands.append({"op": "replace_sample", "id": sid, "sample": {
                "id": sid, "seed": seed * 100 + step,
                "features": [round(rng.uniform(-2, 2), 6) for _ in current_columns],
                "target": round(rng.uniform(-3, 3), 6), "weight": round(rng.uniform(0.5, 1.5), 3),
            }})
        elif op == "update_column":
            commands.append({"op": "update_column", "name": rng.choice(current_columns),
                             "values": {rng.choice(current_ids): round(rng.uniform(-2, 2), 6)}})
        elif op == "add_column":
            name = f"added_{seed}_{step}"
            commands.append({"op": "add_column", "name": name, "value": round(rng.uniform(-1, 1), 6)})
            current_columns.append(name)
        elif op == "delete_column":
            name = rng.choice(current_columns)
            commands.append({"op": "delete_column", "name": name})
            current_columns.remove(name)
        elif op == "delete_sample":
            sid = rng.choice(current_ids)
            commands.append({"op": "delete_sample", "id": sid})
            current_ids.remove(sid)
        elif op == "alphas":
            alpha = rng.choice([1, 2, 4])
            commands.append({"op": "set_alphas", "path": rng.choice(["ridge", "lasso"]),
                             "alphas": [alpha, alpha * 4, alpha * 16]})
        else:
            decimal_places = rng.choice([5, 6, 8])
            commands.append({"op": "set_fixed_point", "decimal_places": decimal_places,
                             "quant_level": rng.choice([0, 1, min(2, decimal_places)])})
    return commands


def test_multiple_arbitrary_edit_sequences_match_rebuild() -> None:
    config = base_config()
    dataset = base_dataset(config)
    for seed in [11, 29, 47]:
        commands = _random_command_sequence(seed, config["columns"], dataset.ids)
        cache_dir = tempfile.mkdtemp(prefix=f"lab-random-{seed}-")
        initial = run_pipeline(dataset, copy.deepcopy(config), cache_dir=cache_dir, worker_count=2)
        previous = {entry["artifact_id"]: entry for entry in initial["ledger"]["entries"]}

        inc_dataset, inc_config, results, errors = apply_commands(dataset, copy.deepcopy(config), commands)
        assert not errors, errors
        incremental = run_pipeline(inc_dataset, inc_config, cache_dir=cache_dir,
                                   previous_lineage=previous, worker_count=2, command_results=results)

        full_dataset, full_config, full_results, full_errors = apply_commands(dataset, copy.deepcopy(config), commands)
        assert not full_errors
        full = run_pipeline(full_dataset, full_config, cache_dir=None, worker_count=2, command_results=full_results)
        assert canonical_dumps(incremental["report"]) == canonical_dumps(full["report"]), f"seed {seed}"
