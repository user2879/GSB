from __future__ import annotations

import json
import shutil
from dataclasses import replace
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from multimodal_fusion.cache import BoundedBlockCache
from multimodal_fusion.config import load_config
from multimodal_fusion.diagnostics import compare_precisions
from multimodal_fusion.features import extract_visual
from multimodal_fusion.fusion import fuse_and_decide
from multimodal_fusion.errors import ConfigError
from multimodal_fusion.inputs import load_inputs
from multimodal_fusion.pipeline import run_pipeline
from multimodal_fusion.text import encode_text


ROOT = Path(__file__).resolve().parents[1]


def copy_example(tmp_path: Path) -> tuple[Path, Path]:
    input_dir = tmp_path / "input"
    shutil.copytree(ROOT / "examples" / "input", input_dir)
    config_path = tmp_path / "config.yaml"
    shutil.copyfile(ROOT / "examples" / "config.yaml", config_path)
    return input_dir, config_path


def edit_yaml(path: Path, mutator) -> None:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    mutator(value)
    path.write_text(yaml.safe_dump(value, sort_keys=True), encoding="utf-8")


def rewrite_descriptions(input_dir: Path, mutator) -> None:
    path = input_dir / "texts.jsonl"
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    mutator(entries)
    path.write_text(
        "\n".join(json.dumps(entry, sort_keys=True) for entry in entries) + "\n",
        encoding="utf-8",
    )


def run_tmp(tmp_path: Path, input_dir: Path, config_path: Path, tag: str = "report") -> dict:
    return run_pipeline(
        config_path=config_path,
        input_dir=input_dir,
        state_dir=tmp_path / f"state-{tag}",
        report_path=tmp_path / f"{tag}.json",
    )


def core_fields(report: dict) -> dict:
    return {
        "global_version": report["global_version"],
        "config_hash": report["config_hash"],
        "visual_config_hash": report["visual_config_hash"],
        "text_config_hash": report["text_config_hash"],
        "fusion_config_hash": report["fusion_config_hash"],
        "quantization": report["quantization"],
        "fusion": report["fusion"],
        "budgets": report["budgets"],
        "samples": report["samples"],
        "boundary_samples": report["boundary_samples"],
        "failure_list": report["failure_list"],
        "summary": report["summary"],
        "stage_versions": report["snapshot"]["stage_versions"],
    }


def test_example_report_snapshot_and_restart(tmp_path: Path) -> None:
    input_dir, config_path = copy_example(tmp_path)
    state_dir = tmp_path / "state"
    report_path = tmp_path / "report.json"
    first = run_pipeline(config_path, input_dir, state_dir, report_path)

    assert first["summary"] == {
        "input_samples": 6,
        "valid_samples": 6,
        "invalid_samples": 0,
        "scored_samples": 6,
        "pending_samples": 0,
        "positive_samples": 3,
        "negative_samples": 3,
        "decision_flips": 1,
    }
    assert first["boundary_samples"] == ["sample-001"]
    assert first["restore"]["status"] == "fresh"
    assert first["failure_list"] == []
    assert report_path.is_file()
    assert json.loads(report_path.read_text(encoding="utf-8"))["global_version"] == first["global_version"]

    manifest = json.loads((state_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["global_version"] == first["global_version"]
    assert manifest["config_hash"] == first["config_hash"]
    assert manifest["quantization_tier"] == 1
    assert set(manifest["stage_versions"]) == {
        "feature_quantization",
        "text_encoding",
        "weighted_fusion",
        "threshold_decision",
    }

    junk = state_dir / "junk.txt"
    junk.write_text("stale", encoding="utf-8")
    second = run_pipeline(config_path, input_dir, state_dir, report_path)
    assert not junk.exists()
    assert second["restore"]["status"] == "resumed_same_version"
    assert sum(second["restore"]["reused_artifacts"].values()) > 0
    assert core_fields(second) == core_fields(first)


def test_full_rebuild_is_field_identical(tmp_path: Path) -> None:
    input_dir, config_path = copy_example(tmp_path)
    first = run_tmp(tmp_path, input_dir, config_path, "first")
    second = run_tmp(tmp_path, input_dir, config_path, "second")
    assert core_fields(first) == core_fields(second)


def test_invalid_samples_are_sorted_and_excluded(tmp_path: Path) -> None:
    input_dir, config_path = copy_example(tmp_path)
    (input_dir / "images" / "sample-002.png").write_bytes(b"not-a-png")
    duplicate = (input_dir / "texts.jsonl").read_text(encoding="utf-8").splitlines()[2]
    with (input_dir / "texts.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(duplicate + "\n")

    def break_sample_four(entries: list[dict]) -> None:
        for entry in entries:
            if entry["sample_id"] == "sample-004":
                entry["assertions"] = [{"entity": "", "attribute": "color", "value": "green"}]

    rewrite_descriptions(input_dir, break_sample_four)
    report = run_tmp(tmp_path, input_dir, config_path)

    assert report["summary"]["invalid_samples"] == 3
    assert report["summary"]["scored_samples"] == 3
    assert [item["sample_id"] for item in report["failure_list"]] == [
        "sample-002",
        "sample-003",
        "sample-004",
    ]
    assert all(item["code"] == "E_SAMPLE_INVALID" for item in report["failure_list"])
    assert all(item["details"] for item in report["failure_list"])
    sample_ids = [sample["sample_id"] for sample in report["samples"]]
    assert sample_ids == ["sample-001", "sample-005", "sample-006"]
    missing_text = next(sample for sample in report["samples"] if sample["sample_id"] == "sample-005")
    assert missing_text["text"]["fill_positions"] == list(range(16))


def test_block_divisibility_rejects_every_sample(tmp_path: Path) -> None:
    input_dir, config_path = copy_example(tmp_path)

    def change_block_size(config: dict) -> None:
        config["features"]["block_size"] = 7

    edit_yaml(config_path, change_block_size)
    report = run_tmp(tmp_path, input_dir, config_path)
    assert report["summary"]["scored_samples"] == 0
    assert report["summary"]["invalid_samples"] == 6
    assert len(report["failure_list"]) == 6
    assert all(
        any("does not divide" in detail for detail in item["details"])
        for item in report["failure_list"]
    )


def test_config_validation_is_strict(tmp_path: Path) -> None:
    _, config_path = copy_example(tmp_path)

    def invalid_word_bits(config: dict) -> None:
        config["quantization"]["word_bits"] = 8

    edit_yaml(config_path, invalid_word_bits)
    with pytest.raises(ConfigError, match="word_bits"):
        load_config(config_path)


def test_visual_and_text_encoding_are_deterministic(tmp_path: Path) -> None:
    input_dir, config_path = copy_example(tmp_path)
    config = load_config(config_path)
    samples = load_inputs(config, input_dir).samples
    by_id = {sample.sample_id: sample for sample in samples}

    cache_one: BoundedBlockCache = BoundedBlockCache(7)
    cache_two: BoundedBlockCache = BoundedBlockCache(7)
    visual_one = extract_visual(by_id["sample-001"], config, config.profile, cache_one)
    visual_two = extract_visual(by_id["sample-001"], config, config.profile, cache_two)
    assert visual_one.as_dict() == visual_two.as_dict()
    assert len(visual_one.blocks) == 64
    for block in visual_one.blocks:
        assert all(isinstance(value, int) for value in block["values"])
        assert block["residuals"] == [
            raw - quantized for raw, quantized in zip(block["raw"], block["values"])
        ]
        assert all(value % config.profile.quant_step == 0 for value in block["values"])

    text_one = encode_text(by_id["sample-001"], config, config.profile)
    text_two = encode_text(by_id["sample-001"], config, config.profile)
    assert text_one.as_dict() == text_two.as_dict()
    assert text_one.fill_positions == sorted(text_one.fill_positions)

    six_text = encode_text(by_id["sample-006"], config, config.profile)
    shape = next(source for source in six_text.merge_sources if source["entity"] == "shape")
    assert shape["assertion_count"] == 3
    assert shape["sources"] == sorted(shape["sources"], key=lambda item: (item["attribute"], item["value"]))

    missing_text = encode_text(by_id["sample-005"], config, config.profile)
    assert missing_text.fill_positions == list(range(config.text_dim))
    assert missing_text.merge_sources == []
    assert missing_text.as_dict() == encode_text(by_id["sample-005"], config, config.profile).as_dict()


def test_fusion_boundary_tie_rule_and_fixed_order(tmp_path: Path) -> None:
    _, config_path = copy_example(tmp_path)
    config = load_config(config_path)
    config = replace(config, threshold=Decimal("0.5"), tie_break="positive")
    zeros_visual = [0] * config.visual_dim
    zeros_text = [0] * config.text_dim
    positive = fuse_and_decide(zeros_visual, zeros_text, config, config.profile)
    assert positive.score_q == config.threshold_q(config.profile)
    assert positive.boundary is True
    assert positive.decision == "positive"
    assert positive.accumulation_order == [
        *(f"text:{index}" for index in range(config.text_dim)),
        *(f"visual:{index}" for index in range(config.visual_dim)),
    ]
    assert sum(item["score_contribution_q"] for item in positive.modality_contributions) == positive.score_q
    assert positive.modality_contributions[0]["exp_q"] == config.profile.one

    negative_config = replace(config, tie_break="negative")
    negative = fuse_and_decide(zeros_visual, zeros_text, negative_config, negative_config.profile)
    assert negative.boundary is True
    assert negative.decision == "negative"


def test_error_trace_distinguishes_quantization_and_fusion_rounding(tmp_path: Path) -> None:
    input_dir, config_path = copy_example(tmp_path)
    config = load_config(config_path)
    samples = load_inputs(config, input_dir).samples
    sample = next(value for value in samples if value.sample_id == "sample-001")
    coarse = replace(config, tier=0)
    cache: BoundedBlockCache = BoundedBlockCache(1000)
    low_visual = extract_visual(sample, coarse, coarse.profile, cache)
    low_text = encode_text(sample, coarse, coarse.profile)
    low_fusion_probe = fuse_and_decide(low_visual.vector, low_text.vector, coarse, coarse.profile)
    low_threshold = Decimal(low_fusion_probe.score_q) / Decimal(coarse.profile.one)
    coarse = replace(coarse, threshold=low_threshold, tolerance_q=0)
    high_visual = extract_visual(sample, coarse, coarse.diagnostic_profile, cache)
    high_text = encode_text(sample, coarse, coarse.diagnostic_profile)
    low_fusion = fuse_and_decide(low_visual.vector, low_text.vector, coarse, coarse.profile)
    high_fusion = fuse_and_decide(high_visual.vector, high_text.vector, coarse, coarse.diagnostic_profile)
    trace = compare_precisions(
        configured_visual=low_visual,
        configured_text=low_text,
        configured_fusion=low_fusion,
        higher_visual=high_visual,
        higher_text=high_text,
        higher_fusion=high_fusion,
        configured_profile=coarse.profile,
        higher_profile=coarse.diagnostic_profile,
        tolerance_q=0,
    )
    assert trace["decision_flip"] is True
    assert trace["first_exceeded_stage"] == "quantization"
    assert trace["flip_attribution"] == "quantization_truncation"
    assert len(trace["stages"]) == 4
    assert trace["stages"][0]["items"]

    report = run_tmp(tmp_path, input_dir, config_path, "rounding")
    boundary = next(value for value in report["samples"] if value["sample_id"] == "sample-001")
    assert boundary["boundary_sample"] is True
    assert boundary["error_trace"]["decision_flip"] is True
    assert boundary["error_trace"]["flip_attribution"] == "fusion_rounding_accumulation"


def test_budget_pending_is_stable_and_does_not_change_scores(tmp_path: Path) -> None:
    input_dir, config_path = copy_example(tmp_path)
    baseline = run_tmp(tmp_path, input_dir, config_path, "generous")
    baseline_scores = {sample["sample_id"]: sample["score_q"] for sample in baseline["samples"]}

    def tighten(config: dict) -> None:
        config["budgets"]["global_macs"] = 26000

    edit_yaml(config_path, tighten)
    constrained = run_tmp(tmp_path, input_dir, config_path, "tight")
    assert constrained["budgets"]["admitted_sample_ids"] == ["sample-001", "sample-002", "sample-005"]
    assert constrained["budgets"]["pending_sample_ids"] == ["sample-003", "sample-004", "sample-006"]
    assert [item["code"] for item in constrained["failure_list"]] == ["E_BUDGET_PENDING"] * 3
    for sample in constrained["samples"]:
        assert sample["score_q"] == baseline_scores[sample["sample_id"]]

    def batch_limit(config: dict) -> None:
        config["budgets"]["global_macs"] = 1000000
        config["budgets"]["batch_points"] = 1800

    edit_yaml(config_path, batch_limit)
    batch = run_tmp(tmp_path, input_dir, config_path, "batch")
    assert batch["budgets"]["admitted_sample_ids"] == ["sample-001"]
    assert all("batch_points" in item["details"] for item in batch["failure_list"])


def test_cache_eviction_recomputes_identical_block(tmp_path: Path) -> None:
    cache: BoundedBlockCache[int] = BoundedBlockCache(1)
    assert cache.get_or_compute("a", lambda: 11) == 11
    assert cache.get_or_compute("b", lambda: 22) == 22
    assert cache.get_or_compute("a", lambda: 11) == 11
    stats = cache.stats()
    assert stats.evictions == 2
    assert stats.recomputations_after_eviction == 1
    assert stats.resident_entries == 1


def test_nonstrict_empty_assertion_fields_use_deterministic_fill(tmp_path: Path) -> None:
    input_dir, config_path = copy_example(tmp_path)

    def allow_empty_fields(config: dict) -> None:
        config["input"]["strict_assertions"] = False

    edit_yaml(config_path, allow_empty_fields)

    def blank_field(entries: list[dict]) -> None:
        for entry in entries:
            if entry["sample_id"] == "sample-001":
                entry["assertions"][0]["value"] = ""

    rewrite_descriptions(input_dir, blank_field)
    first = run_tmp(tmp_path, input_dir, config_path, "nonstrict-first")
    second = run_tmp(tmp_path, input_dir, config_path, "nonstrict-second")
    first_sample = next(sample for sample in first["samples"] if sample["sample_id"] == "sample-001")
    second_sample = next(sample for sample in second["samples"] if sample["sample_id"] == "sample-001")
    assert first_sample["text"]["filled_fields"]
    assert first_sample["text"]["filled_fields"][0]["fill_token"] == "<missing:value>"
    assert first_sample["text"]["vector"] == second_sample["text"]["vector"]
    assert first_sample["text"]["fill_positions"] == second_sample["text"]["fill_positions"]


def test_threads_and_cache_capacity_do_not_change_numeric_results(tmp_path: Path) -> None:
    input_dir, config_path = copy_example(tmp_path)
    baseline = run_tmp(tmp_path, input_dir, config_path, "parallel")

    def serialize(config: dict) -> None:
        config["budgets"]["threads"] = 1
        config["budgets"]["cache_blocks"] = 1
        config["budgets"]["snapshot_interval"] = 1

    edit_yaml(config_path, serialize)
    serialized = run_tmp(tmp_path, input_dir, config_path, "serial")
    for field in (
        "quantization",
        "fusion",
        "samples",
        "boundary_samples",
        "failure_list",
        "summary",
    ):
        assert serialized[field] == baseline[field]
    assert serialized["snapshot"]["stage_versions"] == baseline["snapshot"]["stage_versions"]


def test_incremental_edit_sequence_matches_fresh_final_rebuild(tmp_path: Path) -> None:
    input_dir, config_path = copy_example(tmp_path)
    state_dir = tmp_path / "state-incremental"
    run_pipeline(config_path, input_dir, state_dir)

    shutil.copyfile(input_dir / "images" / "sample-004.png", input_dir / "images" / "sample-002.png")
    run_pipeline(config_path, input_dir, state_dir)

    def edit_assertions(entries: list[dict]) -> None:
        for entry in entries:
            if entry["sample_id"] == "sample-003":
                entry["assertions"] = [value for value in entry["assertions"] if value["attribute"] != "orientation"]
            if entry["sample_id"] == "sample-006":
                entry["assertions"].append({"entity": "scene", "attribute": "motion", "value": "still"})

    rewrite_descriptions(input_dir, edit_assertions)
    after_text_edit = run_pipeline(config_path, input_dir, state_dir)
    assert after_text_edit["restore"]["status"] == "changed_version_rebuild"
    assert sum(after_text_edit["restore"]["reused_artifacts"].values()) > 0

    def change_fusion(config: dict) -> None:
        config["fusion"]["weights"] = {"text": 0.55, "visual": 0.45}
        config["fusion"]["temperature"] = 0.9
        config["quantization"]["tier"] = 2

    edit_yaml(config_path, change_fusion)
    incremental = run_pipeline(config_path, input_dir, state_dir)
    assert incremental["restore"]["status"] == "changed_version_rebuild"

    fresh_input = tmp_path / "fresh-input"
    shutil.copytree(input_dir, fresh_input)
    fresh_config = tmp_path / "fresh-config.yaml"
    shutil.copyfile(config_path, fresh_config)
    fresh = run_pipeline(
        fresh_config,
        fresh_input,
        tmp_path / "state-fresh",
        tmp_path / "fresh-report.json",
    )
    assert core_fields(incremental) == core_fields(fresh)
    assert incremental["restore"]["status"] != fresh["restore"]["status"]
