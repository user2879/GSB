from __future__ import annotations

import json
from pathlib import Path

from instruction_factory.constants import E_CONFIG_INVALID, E_SEED_INVALID, E_WEIGHT_INVALID
from instruction_factory.loader import load_inputs
from instruction_factory.runner import write_validation_report

from .conftest import DATA_DIR, read_json, seed_lines, write_json


def test_valid_sample_loads_all_inputs() -> None:
    inputs = load_inputs(DATA_DIR)
    assert not inputs.failures.fatal
    assert inputs.failures.items == []
    assert len(inputs.seeds_by_id) == 8
    assert len(inputs.config["task_types"]) == 3
    assert len(inputs.plan["batches"]) == 8
    assert inputs.plan_templates["rewrite"]["rewrite.short"]["instruction_template"].startswith("请把")


def test_seed_errors_have_business_codes_and_line_locations(sample_dir: Path) -> None:
    seeds_path = sample_dir / "seeds.jsonl"
    lines = seed_lines(seeds_path)
    lines.append(lines[0])
    lines.append(
        json.dumps(
            {
                "seed_id": "seed-unknown-task",
                "task_type": "unknown.task",
                "template_variant_id": "unknown.template",
                "instruction_template": "处理“{topic}”。",
                "slots": ["topic"],
                "reference_answer": "处理{topic}。",
            },
            ensure_ascii=False,
        )
    )
    seeds_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    inputs = load_inputs(sample_dir)
    codes = [failure.code for failure in inputs.failures.items]
    assert E_SEED_INVALID in codes
    duplicate = next(failure for failure in inputs.failures.items if "duplicate seed_id" in failure.message)
    assert duplicate.code == E_SEED_INVALID
    assert duplicate.file.endswith("seeds.jsonl")
    assert duplicate.line == 9
    assert duplicate.field == "seed_id"
    unknown_task = next(failure for failure in inputs.failures.items if "not present" in failure.message)
    assert unknown_task.code == E_SEED_INVALID
    assert unknown_task.line == 10
    assert unknown_task.field == "task_type"


def test_config_and_weight_errors_have_field_paths(sample_dir: Path) -> None:
    config_path = sample_dir / "config.json"
    config = read_json(config_path)
    config["task_types"]["classify"]["target_count"] = 0
    config["task_types"]["classify"]["template_weights"]["classify.intent"] = -1
    config["near_duplicate_threshold"] = 1.2
    config["fixed_decimal_places"] = 9
    config["max_concurrency"] = 99
    config["global_score_budget"] = 0
    write_json(config_path, config)

    inputs = load_inputs(sample_dir)
    codes = {failure.code for failure in inputs.failures.items}
    assert E_CONFIG_INVALID in codes
    assert E_WEIGHT_INVALID in codes
    fields = {failure.field for failure in inputs.failures.items if failure.field}
    assert "task_types.classify.target_count" in fields
    assert "task_types.classify.template_weights.classify.intent" in fields
    assert "near_duplicate_threshold" in fields
    assert "fixed_decimal_places" in fields
    assert "max_concurrency" in fields
    assert "global_score_budget" in fields
    assert inputs.failures.fatal


def test_plan_weight_error_is_reported(sample_dir: Path) -> None:
    plan_path = sample_dir / "plan.json"
    plan = read_json(plan_path)
    plan["batches"][4]["operations"][0]["weight"] = -0.1
    write_json(plan_path, plan)

    inputs = load_inputs(sample_dir)
    failures = [failure for failure in inputs.failures.items if failure.code == E_WEIGHT_INVALID]
    assert failures
    assert any("operations[0].weight" in (failure.field or "") for failure in failures)
    assert inputs.failures.fatal


def test_deep_validation_rejects_plan_that_removes_last_template(sample_dir: Path, tmp_path: Path) -> None:
    plan_path = sample_dir / "plan.json"
    plan = read_json(plan_path)
    plan["batches"].append(
        {
            "batch_id": "b09-delete-last-rewrite-template",
            "operations": [
                {
                    "op": "set_template_weights",
                    "task_type": "rewrite",
                    "template_weights": {"rewrite.short": 1.0},
                },
                {"op": "delete_template", "task_type": "rewrite", "variant_id": "rewrite.short"},
            ],
        }
    )
    write_json(plan_path, plan)
    inputs = load_inputs(sample_dir)
    assert not inputs.failures.fatal
    report = write_validation_report(inputs, tmp_path / "out")
    assert report["valid"] is False
    assert E_WEIGHT_INVALID in report["failure_codes"]
