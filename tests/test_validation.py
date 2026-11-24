from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from routing_workbench.config import ConfigValidationError, parse_config
from routing_workbench.io_layer import load_config, load_utterances, parse_utterance, validate_utterances
from routing_workbench.sample_generator import sample_config, sample_utterances


def test_sample_loads_and_all_records_validate(sample_dir: Path) -> None:
    config, config_failures = load_config(sample_dir / "routing.yaml", sample_dir / "vocabulary.txt")
    assert config is not None
    assert config_failures == []
    utterances, load_failures = load_utterances(sample_dir / "utterances.jsonl")
    assert load_failures == []
    valid, validation_failures = validate_utterances(utterances, config)
    assert validation_failures == []
    assert len(valid) == 13
    assert {utterance.domain_truth for utterance in valid.values()} == {"account", "billing"}


def test_dataset_rejects_turn_gap_duplicate_pair_and_out_of_vocab(sample_dir: Path) -> None:
    config, _ = load_config(sample_dir / "routing.yaml", sample_dir / "vocabulary.txt")
    assert config is not None
    utterances, _ = load_utterances(sample_dir / "utterances.jsonl")
    by_id = {utterance.utterance_id: utterance for utterance in utterances}
    mutated = list(utterances)
    mutated[1] = parse_utterance({**by_id["S001-U2"].raw, "turn": 9}, 2)
    gap_valid, gap_failures = validate_utterances(mutated, config)
    assert "S001-U2" not in gap_valid
    assert any(failure.field == "turn" for failure in gap_failures)

    duplicate = list(utterances)
    duplicate[1] = parse_utterance({**by_id["S001-U2"].raw, "turn": 1}, 2)
    duplicate_valid, duplicate_failures = validate_utterances(duplicate, config)
    assert "S001-U2" not in duplicate_valid
    assert any(failure.field == "session_id+turn" for failure in duplicate_failures)

    out_raw = copy.deepcopy(by_id["S001-U3"].raw)
    out_raw["tokens"] = list(out_raw["tokens"]) + ["未登记故障词"]
    out_of_vocab = parse_utterance(out_raw, 3)
    oov_valid, oov_failures = validate_utterances([*utterances[:2], out_of_vocab, *utterances[3:]], config)
    assert "S001-U3" not in oov_valid
    assert any(failure.field == "tokens" for failure in oov_failures)


def _invalid_config_mutators():
    def nonfinite(raw):
        raw["domains"]["account"]["triggers"][0]["weight"] = math.inf

    def threshold(raw):
        raw["thresholds"]["domain"] = "1.0001"

    def duplicate_trigger(raw):
        raw["domains"]["account"]["triggers"].append({"term": "账号", "weight": 1})

    def duplicate_negator(raw):
        raw["negation_rules"][0]["negators"].append("没有")

    def undeclared_slot(raw):
        raw["slot_patterns"]["phone_after_label"]["slot"] = "ghost"

    def concurrency(raw):
        raw["thread_pool_size"] = raw["max_thread_pool_size"] + 1

    def budget(raw):
        raw["batch_point_budget"] = 0

    def quantization(raw):
        raw["quantization_levels"] = 0
        raw["quantization_scale"] = 0

    return [
        pytest.param(nonfinite, id="nonfinite-trigger-weight"),
        pytest.param(threshold, id="threshold-range"),
        pytest.param(duplicate_trigger, id="duplicate-trigger"),
        pytest.param(duplicate_negator, id="duplicate-negator"),
        pytest.param(undeclared_slot, id="undeclared-slot"),
        pytest.param(concurrency, id="concurrency-cap"),
        pytest.param(budget, id="budget-positive"),
        pytest.param(quantization, id="quantization-positive"),
    ]


@pytest.mark.parametrize("mutator", _invalid_config_mutators())
def test_config_validation_rejects_invalid_inputs(mutator) -> None:
    raw = copy.deepcopy(sample_config())
    mutator(raw)
    with pytest.raises(ConfigValidationError):
        parse_config(raw)


def test_malformed_jsonl_is_a_failure_not_a_crash(sample_dir: Path, tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    good_line = json.dumps(sample_utterances()[0], ensure_ascii=False)
    bad.write_text(good_line + "\n{not-json\n" + json.dumps(sample_utterances()[1], ensure_ascii=False) + "\n", encoding="utf-8")
    utterances, failures = load_utterances(bad)
    assert len(utterances) == 2
    assert len(failures) == 1
    assert failures[0].business_code == "E_ROUTE_INVALID"
    assert failures[0].utterance_id.startswith("__line_")

