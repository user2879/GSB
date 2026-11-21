from __future__ import annotations

from ner_workbench.config import load_config
from ner_workbench.decoder import verify_path_score, viterbi_decode
from ner_workbench.entities import merge_entities
from ner_workbench.features import hash_collision_report, score_sample_once
from ner_workbench.generator import generate_dataset
from ner_workbench.loader import load_corpus
from ner_workbench.params import all_quantized_values, load_parameter_package


def test_emission_is_bitwise_deterministic_and_quantized(tmp_path):
    samples_dir = tmp_path / "samples"
    generate_dataset(samples_dir, seed=73, sample_count=6)
    cfg = load_config(samples_dir / "inference.yaml")
    params = load_parameter_package(samples_dir / "model_params.json", cfg)
    samples = load_corpus(samples_dir / "corpus.jsonl").samples
    sample = samples[0]

    first = score_sample_once(sample, params)
    second = score_sample_once(sample, params)
    assert first == second
    assert all(isinstance(value, int) for row in first for value in row)
    assert all(value % params.quant_step == 0 for value in all_quantized_values(params))

    collisions = hash_collision_report(samples, params)
    assert collisions["collision_events"] > 0
    assert collisions["colliding_bucket_count"] > 0


def test_viterbi_tie_breaks_by_label_index_and_verifies_score():
    labels = ["O", "B-人物", "S-人物"]
    emissions = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    transitions = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    starts = [0, 0, 0]
    decoded = viterbi_decode(emissions, labels, transitions, starts)
    assert decoded.labels == ("O", "O", "O")
    assert decoded.path_score == 0
    assert verify_path_score(decoded.labels, labels, emissions, transitions, starts) == decoded.path_score


def test_entity_merge_repairs_and_lexicographic_type_tie():
    chars = ["甲", "乙", "丙", "丁", "戊", "己", "庚"]
    tags = ["I-人物", "B-地点", "I-机构", "O", "E-时间", "S-地点", "B-机构"]
    merged = merge_entities("SMP-X", chars, tags)
    codes = [repair.code for repair in merged.repairs]
    assert "I_WITHOUT_B_PROMOTED_B" in codes
    assert "B_INTERRUPTED_BY_B" in codes
    assert "TYPE_JUMP_ON_I" in codes
    assert "B_INTERRUPTED_BY_O" in codes
    assert "E_WITHOUT_B_PROMOTED_S" in codes
    assert "UNCLOSED_B_CLOSED_AT_END" in codes
    assert "TYPE_TIE_LEXICOGRAPHIC" in codes
    assert [entity.start for entity in merged.entities] == sorted(entity.start for entity in merged.entities)
    conflict = next(entity for entity in merged.entities if entity.start == 1)
    assert conflict.candidate_types == tuple(sorted(("地点", "机构")))
    assert conflict.entity_type == sorted(("地点", "机构"))[0]
