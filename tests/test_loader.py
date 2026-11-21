from __future__ import annotations

import json
from pathlib import Path

import yaml

from mtcomp.loader import load_state
from mtcomp.samplegen import write_sample


def _write_inputs(tmp_path: Path, *, params: dict | None = None, config: dict | None = None):
    base_config = {
        "ngram_order": 2,
        "max_sentences_per_batch": 2,
        "thread_pool_concurrency": 2,
        "decode_points_per_batch": 20,
        "snapshot_commit_interval_batches": 1,
        "max_sentence_tokens": 8,
        "candidate_limit": 4,
        "controlled_vocabulary": ["a", "b", "c"],
    }
    if config:
        base_config.update(config)
    base_params = {
        "alignment_weight": 1.0,
        "order_weight": 0.8,
        "length_penalty": 0.1,
        "fusion_temperature": 1.0,
        "confidence_threshold": 0.5,
        "fixed_point_bits": 12,
        "quantization_tier": 16,
        "allowed_fixed_bits": [8, 12, 16],
        "allowed_quantization_tiers": [1, 2, 4, 8, 16],
    }
    if params:
        base_params.update(params)
    (tmp_path / "compile.yaml").write_text(
        yaml.safe_dump(base_config, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )
    (tmp_path / "params.json").write_text(
        json.dumps(base_params, ensure_ascii=False), encoding="utf-8"
    )
    return base_config, base_params


def test_loader_isolates_invalid_rows_and_reports_stable_codes(tmp_path: Path):
    _write_inputs(tmp_path)
    (tmp_path / "sentences.jsonl").write_text(
        "\n".join(
            [
                '{"id":"S1","tokens":"a b","batch":0}',
                '{"id":"S1","tokens":"b c","batch":0}',
                '{"id":"S2","tokens":"c","batch":0}',
                '{"id":"S3","tokens":"a","batch":0}',
                '{"id":"S4","tokens":"outside","batch":1}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "lexicon.tsv").write_text(
        "\n".join(
            [
                "a x 1 0.5",
                "a x 2 0.5",
                "b y 0 0.5",
                "c z 1 1.5",
                "malformed line",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (tmp_path / "edits.json").write_text("[]\n", encoding="utf-8")

    bundle = load_state(tmp_path)
    assert bundle.state is not None
    assert set(bundle.state.sentences) == {"S1", "S2"}
    codes = [failure.code for failure in bundle.failures]
    assert codes.count("E_SENT_INVALID") == 3  # duplicate, batch overflow, vocabulary
    assert codes.count("E_LEX_INVALID") == 4
    assert "E_CONFIG_INVALID" not in codes

    ordered = bundle.sorted_failures()
    assert [item.order[:2] for item in ordered] == sorted(
        item.order[:2] for item in ordered
    )
    assert all(item.order for item in ordered)

    lex_failures = [item for item in ordered if item.code == "E_LEX_INVALID"]
    assert [item.line for item in lex_failures] == [2, 3, 4, 5]


def test_loader_rejects_nonfinite_and_disallowed_fixed_point_enums(tmp_path: Path):
    _write_inputs(
        tmp_path,
        params={"alignment_weight": float("nan"), "fixed_point_bits": 10, "quantization_tier": 3},
    )
    (tmp_path / "sentences.jsonl").write_text(
        '{"id":"S1","tokens":"a","batch":0}\n', encoding="utf-8"
    )
    (tmp_path / "lexicon.tsv").write_text("a x 1 0.5\n", encoding="utf-8")
    (tmp_path / "edits.json").write_text("[]\n", encoding="utf-8")

    bundle = load_state(tmp_path)
    assert bundle.state is None
    config_failures = [
        item for item in bundle.failures if item.code == "E_CONFIG_INVALID"
    ]
    messages = " | ".join(item.message for item in config_failures)
    assert "alignment_weight must be finite" in messages
    assert "fixed_point_bits must be one of" in messages
    assert "quantization_tier must be one of" in messages


def test_generated_sample_loads_without_failures(tmp_path: Path):
    write_sample(tmp_path / "sample")
    bundle = load_state(tmp_path / "sample")
    assert bundle.state is not None
    assert not bundle.failures
    assert len(bundle.state.sentences) == 6
    assert len(bundle.commands) == 8
    assert bundle.injection_data is not None
    assert len(bundle.injection_data["injections"]) == 3
