import json

import pytest

from qu.errors import (
    E_DUPLICATE_QUERY_ID,
    E_EMPTY_QUERY,
    E_INVALID_JSON,
    E_INVALID_WEIGHT,
    E_RULE_DANGLING,
)
from qu.loader import (
    load_bundle,
    load_entity_dict,
    load_intent_vocab,
    load_rewrite_rules,
    load_samples,
)


def _write(path, lines):
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_sample_loader_failure_codes(tmp_path):
    path = tmp_path / "q.jsonl"
    _write(path, [
        '{"query_id": "q1", "query": ',
        '{"query_id": "q2", "query": "   "}',
        '{"query_id": "q3", "query": "!!!"}',
        '{"query_id": "q4", "query": "手机"}',
        '{"query_id": "q4", "query": "耳机"}',
    ])
    samples, failures = load_samples(str(path))
    codes = [item["code"] for item in failures]
    assert E_INVALID_JSON in codes
    assert E_EMPTY_QUERY in codes
    assert E_DUPLICATE_QUERY_ID in codes
    assert [sample.query_id for sample in samples] == ["q4"]


def test_infinite_weight_rejected(tmp_path):
    vocab = tmp_path / "v.jsonl"
    _write(vocab, [
        '{"intent": "a", "term": "手机", "weight": Infinity}',
        '{"intent": "a", "term": "耳机", "weight": 1.0}',
    ])
    entries, failures = load_intent_vocab(str(vocab))
    assert failures[0]["code"] == E_INVALID_WEIGHT
    assert len(entries) == 1

    entities = tmp_path / "e.jsonl"
    _write(entities, [
        '{"entity_type": "brand", "term": "星河", "entity_id": "b1",'
        ' "weight": NaN}',
    ])
    entries, failures = load_entity_dict(str(entities))
    assert failures[0]["code"] == E_INVALID_WEIGHT
    assert entries == []


def test_dangling_rule_detected(tmp_path):
    rules = tmp_path / "r.jsonl"
    _write(rules, [
        '{"rule_id": "r1", "type": "synonym", "trigger_term": "未知词",'
        ' "replacement": "x", "ref_terms": ["未知词"]}',
        '{"rule_id": "r2", "type": "synonym", "trigger_term": "手机",'
        ' "replacement": "电话", "ref_terms": ["手机"]}',
    ])
    rules_out, failures = load_rewrite_rules(str(rules), ["手机"])
    assert failures[0]["code"] == E_RULE_DANGLING
    assert [rule.rule_id for rule in rules_out] == ["r2"]


def test_generated_bundle_loads_with_clean_tables(generated):
    bundle = load_bundle(generated["samples"], generated["vocab"],
                         generated["entities"], generated["rules"])
    assert bundle.samples and bundle.vocab and bundle.entities and bundle.rules
    assert bundle.failures == []
    assert set(bundle.table_fingerprints) == {
        "samples", "sample_content", "vocab", "entities", "rules"}


def test_invalid_companion_file_reports_all_codes(generated):
    _samples, failures = load_samples(generated["invalid_samples"])
    codes = {item["code"] for item in failures}
    assert {E_INVALID_JSON, E_EMPTY_QUERY, E_DUPLICATE_QUERY_ID} <= codes
