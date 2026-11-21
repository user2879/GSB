from __future__ import annotations

import json

import pytest

from ner_workbench.config import load_config
from ner_workbench.errors import E_PARAM_SHAPE_MISMATCH, WorkbenchError
from ner_workbench.generator import generate_dataset
from ner_workbench.loader import load_corpus
from ner_workbench.params import load_parameter_package


def test_corpus_validation_codes_and_continue(tmp_path):
    samples = tmp_path / "samples"
    generate_dataset(samples, seed=41, sample_count=8)
    original_lines = (samples / "corpus.jsonl").read_text(encoding="utf-8").splitlines()
    base = json.loads(original_lines[0])
    corrupt: list[dict] = []

    empty = dict(base)
    empty["sample_id"] = "BAD-EMPTY"
    empty["chars"] = ""
    empty["labels"] = ""
    empty["entities"] = []
    empty["entity_types"] = []
    corrupt.append(empty)

    mismatch = dict(base)
    mismatch["sample_id"] = "BAD-LENGTH"
    mismatch["labels"] = " ".join(str(base["labels"]).split()[:-1])
    mismatch["entities"] = []
    mismatch["entity_types"] = []
    corrupt.append(mismatch)

    unknown = dict(base)
    unknown["sample_id"] = "BAD-LABEL"
    tags = str(base["labels"]).split()
    tags[0] = "Z-人物"
    unknown["labels"] = " ".join(tags)
    unknown["entities"] = []
    unknown["entity_types"] = []
    corrupt.append(unknown)

    duplicate = dict(base)
    corrupt.append(duplicate)

    with (samples / "corpus.jsonl").open("a", encoding="utf-8") as handle:
        for item in corrupt:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    result = load_corpus(samples / "corpus.jsonl")
    assert len(result.samples) == 8
    codes = {failure["code"] for failure in result.failures}
    assert codes == {
        "E_CORPUS_INVALID",
        "E_LABEL_LENGTH_MISMATCH",
        "E_LABEL_UNKNOWN",
    }
    assert any(item["sample_id"] == "BAD-EMPTY" for item in result.failures)
    assert any(item["sample_id"] == "BAD-LENGTH" for item in result.failures)
    assert any(item["sample_id"] == "BAD-LABEL" for item in result.failures)
    assert [item["sample_id"] for item in result.failures] == sorted(
        item["sample_id"] for item in result.failures
    )


def test_parameter_shape_mismatch_and_transition_shape(tmp_path):
    samples = tmp_path / "samples"
    generate_dataset(samples, seed=42, sample_count=6)
    cfg = load_config(samples / "inference.yaml")

    payload = json.loads((samples / "model_params.json").read_text(encoding="utf-8"))
    first_char = sorted(payload["char_embeddings"])[0]
    payload["char_embeddings"][first_char] = payload["char_embeddings"][first_char][:-1]
    bad_embedding = tmp_path / "bad_embedding.json"
    bad_embedding.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(WorkbenchError) as exc_info:
        load_parameter_package(bad_embedding, cfg)
    assert exc_info.value.code == E_PARAM_SHAPE_MISMATCH

    payload = json.loads((samples / "model_params.json").read_text(encoding="utf-8"))
    payload["transition_weights"][0] = payload["transition_weights"][0][:-1]
    bad_transition = tmp_path / "bad_transition.json"
    bad_transition.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(WorkbenchError) as exc_info:
        load_parameter_package(bad_transition, cfg)
    assert exc_info.value.code == E_PARAM_SHAPE_MISMATCH
