from __future__ import annotations

import copy
import json
import threading
from dataclasses import replace
from pathlib import Path

import pytest

from ner_review.config import QuantConfig, RuntimeConfig
from ner_review.coordinator import CommandCoordinator
from ner_review.decoder import viterbi_decode
from ner_review.engine import IncrementalEngine
from ner_review.entities import merge_entities, repair_bieso
from ner_review.features import emission_scores
from ner_review.labels import LABELS
from ner_review.snapshot import SnapshotStore
from ner_review.synth import write_demo_bundle
from ner_review.validation import load_command_stream, load_injection_manifest, load_package, parse_corpus


@pytest.fixture()
def bundle(tmp_path: Path):
    config = RuntimeConfig(seed=335, corpus_size=6)
    paths = write_demo_bundle(tmp_path / "data", config)
    package_result = load_package(paths["package"])
    assert package_result["ok"], package_result["failures"]
    corpus = parse_corpus(paths["corpus"])
    assert not corpus["failures"]
    engine = IncrementalEngine.from_bundle(paths["corpus"], paths["lexicon"], package_result, config)
    engine.run_incremental()
    return {"paths": paths, "config": config, "package": package_result["package"], "engine": engine}


def full_parity_engine(engine: IncrementalEngine) -> IncrementalEngine:
    fresh = IncrementalEngine(
        package=copy.deepcopy(engine.package),
        config=replace(engine.config),
        samples=copy.deepcopy(list(engine.samples.values())),
        lexicon=copy.deepcopy(engine.lexicon_list),
        corpus_failures=copy.deepcopy(engine.corpus_failures),
        lexicon_failures=copy.deepcopy(engine.lexicon_failures),
    )
    fresh.run_incremental()
    return fresh


def test_generated_bundle_is_valid(bundle):
    result = bundle["engine"].authoritative_result()
    assert result["sample_count"] == 6
    assert all(sample["path_total"] == sample["path_total"] for sample in result["samples"])
    assert result["live_cache_keys"]


def test_validation_business_codes_are_isolated():
    lines = [
        json.dumps({"id": "ok", "char_sequence": "北 京", "label_sequence": "B-LOC E-LOC"}, ensure_ascii=False),
        json.dumps({"id": "bad-length", "char_sequence": "北 京", "label_sequence": "B-LOC"}, ensure_ascii=False),
        json.dumps({"id": "bad-tag", "char_sequence": "北 京", "label_sequence": "B-X E-X"}, ensure_ascii=False),
        json.dumps({"id": "ok", "char_sequence": "上 海", "label_sequence": "B-LOC E-LOC"}, ensure_ascii=False),
    ]
    result = parse_corpus(lines)
    assert len(result["samples"]) == 1
    codes = [item["code"] for item in result["failures"]]
    assert "NER-E1002" in codes
    assert "NER-E1003" in codes
    assert "NER-E1004" in codes


def test_package_faults_have_fixed_codes(tmp_path: Path, bundle):
    bad_dim = tmp_path / "dim.json"
    data = copy.deepcopy(bundle["package"])
    data["feature_dim"] = 0
    bad_dim.write_text(json.dumps(data), encoding="utf-8")
    assert load_package(bad_dim)["failures"][0]["code"] == "NER-E2002"

    oob = tmp_path / "oob.json"
    data = copy.deepcopy(bundle["package"])
    data["feature_dim"] = 999
    oob.write_text(json.dumps(data), encoding="utf-8")
    assert load_package(oob)["failures"][0]["code"] == "NER-E2004"

    nonfinite = tmp_path / "nan.json"
    data = copy.deepcopy(bundle["package"])
    data["emission_weights"][0][0][0] = float("nan")
    nonfinite.write_text(json.dumps(data, allow_nan=True), encoding="utf-8")
    assert load_package(nonfinite)["failures"][0]["code"] == "NER-E2005"

    truncated = tmp_path / "truncated.json"
    truncated.write_text(Path(bundle["paths"]["package"]).read_text(encoding="utf-8")[:80], encoding="utf-8")
    assert load_package(truncated)["failures"][0]["code"] == "NER-E2001"


def test_command_and_injection_manifest_loaders(bundle):
    commands = load_command_stream({"commands": [{"op": "set_quant"}, "bad"]})
    assert len(commands["commands"]) == 2
    assert commands["failures"][0]["code"] == "NER-E3001"

    manifest = load_injection_manifest(bundle["paths"]["injections"])
    assert not manifest["failures"]
    assert len(manifest["injections"]) == 6

    bad_manifest = load_injection_manifest({"injections": [{"id": "x", "kind": "nope"}]})
    assert bad_manifest["failures"][0]["code"] == "NER-E3004"


def test_quant_width_and_level_change_integers(bundle):
    engine = bundle["engine"]
    sample_id = engine.sample_ids[0]
    feature_key = engine.stage_index[sample_id]["features"]
    features = engine.cache[feature_key]["payload"]
    narrow = emission_scores(features, engine.package, QuantConfig(decimal_bits=8, quant_level=1))
    wide = emission_scores(features, engine.package, QuantConfig(decimal_bits=12, quant_level=1))
    coarse = emission_scores(features, engine.package, QuantConfig(decimal_bits=12, quant_level=0))
    assert narrow["integer_scores"] != wide["integer_scores"]
    assert narrow["scores"] != wide["scores"]
    assert coarse["integer_scores"] != wide["integer_scores"]
    trace = narrow["quantization_trace"][0]
    assert trace["quantization_step"] == "1/2^8"
    assert trace["coordinates"][0]["h_direction"] in {"up", "down", "exact"}
    assert set(trace["labels"][0]["weight_directions"]) <= {"u", "d", "e"}
    assert trace["labels"][0]["term_integers"]


def test_viterbi_is_deterministic_and_records_ties():
    package = {
        "feature_dim": 2,
        "bucket_count": 2,
        "model_version": "tie-test",
        "transition_weights": {
            "start": [0.0] * len(LABELS),
            "end": [0.0] * len(LABELS),
            "matrix": [[0.0] * len(LABELS) for _ in LABELS],
        },
    }
    emission = {"sample_id": "tie", "labels": list(LABELS), "scores": [[0.0] * len(LABELS) for _ in range(3)]}
    first = viterbi_decode(emission, package)
    second = viterbi_decode(emission, package)
    assert first == second
    assert first["labels"][0] == LABELS[0]
    assert first["tie_positions"]
    assert all(item["selected_label"] == LABELS[0] for item in first["tie_positions"])


def test_repair_and_suppression_are_recorded():
    repaired = repair_bieso(["I-PER", "O", "B-LOC", "I-ORG", "O", "E-TIME"])
    codes = {item["rule_code"] for item in repaired["repairs"]}
    assert "NER-W4101" in codes
    assert {"NER-W4103", "NER-W4104", "NER-W4105"} & codes

    sample = {"id": "X1", "chars": list("北京")}
    candidate_scores = [[0.0] * len(LABELS), [0.0] * len(LABELS)]
    viterbi_result = {"labels": ["B-PER", "E-PER"], "candidate_scores": candidate_scores}
    lexicon = [{"id": "lex-loc-beijing", "text": "北京", "type": "LOC"}]
    merged = merge_entities(sample, viterbi_result, lexicon)
    assert merged["entities"][0]["type"] == "LOC"
    assert merged["suppressions"]
    assert merged["suppressions"][0]["candidate_type"] == "PER"


def test_precise_sample_and_stage_invalidation(bundle):
    engine = bundle["engine"]
    target = engine.sample_ids[1]
    result = engine.apply_command(
        {
            "op": "replace_sample",
            "sample": {
                "id": target,
                "char_sequence": "苏 州 我 去",
                "label_sequence": "B-LOC E-LOC O O",
            },
        }
    )
    assert result["affected_samples"] == [target]
    run = engine.run_incremental()
    recomputed_samples = {item["sample_id"] for item in run["recomputed"]}
    assert recomputed_samples == {target}
    assert len([item for item in run["recomputed"] if item["sample_id"] == target]) == 4

    add = engine.apply_command({"op": "upsert_lexicon", "entry": {"id": "lex-suzhou", "text": "苏州", "type": "LOC"}})
    assert add["affected_samples"] == [target]
    run = engine.run_incremental()
    assert {item["sample_id"] for item in run["recomputed"]} == {target}

    transition = engine.apply_command({"op": "set_transition_weight", "from": "O", "to": "O", "value": 0.91})
    assert len(transition["affected_samples"]) == len(engine.samples)
    run = engine.run_incremental()
    recomputed_stages = {(item["sample_id"], item["stage"]) for item in run["recomputed"]}
    reused_stages = {(item["sample_id"], item["stage"]) for item in run["reused"]}
    for sample_id in engine.sample_ids:
        assert (sample_id, "viterbi") in recomputed_stages
        assert (sample_id, "merge") in recomputed_stages
        assert (sample_id, "features") in reused_stages
        assert (sample_id, "emission") in reused_stages


def test_incremental_matches_blank_full_after_command_sequence(bundle):
    engine = bundle["engine"]
    sample_id = engine.sample_ids[0]
    target_id = engine.sample_ids[1]
    entry_id = engine.lexicon_list[0]["id"]
    commands = [
        {"op": "upsert_lexicon", "entry": {"id": "lex-test-custom", "text": "苏州", "type": "LOC"}},
        {"op": "replace_sample", "sample": {"id": target_id, "char_sequence": "苏 州 你 好", "label_sequence": "B-LOC E-LOC O O"}},
        {"op": "set_emission_weight", "label": "B-LOC", "feature_index": 0, "bucket": 1, "value": 0.33},
        {"op": "set_transition_weight", "from": "B-LOC", "to": "E-LOC", "value": 2.25},
        {"op": "set_quant", "decimal_bits": 8, "quant_level": 2},
        {"op": "upsert_lexicon", "entry": {"id": entry_id, "text": "北京", "type": "ORG"}},
        {"op": "remove_lexicon", "entry_id": "lex-test-custom"},
        {"op": "delete_sample", "sample_id": sample_id},
        {"op": "not-a-real-op"},
        {"op": "set_quant", "decimal_bits": 99},
    ]
    for command in commands:
        engine.apply_command(command)
        engine.run_incremental()

    assert len(engine.command_failures) == 2
    fresh = full_parity_engine(engine)
    assert engine.authoritative_result() == fresh.authoritative_result()


def test_snapshot_roundtrip_matches_authoritative_result(tmp_path: Path, bundle):
    engine = bundle["engine"]
    engine.apply_command({"op": "set_quant", "decimal_bits": 9, "quant_level": 0})
    engine.run_incremental()
    expected = engine.authoritative_result()
    store = SnapshotStore(tmp_path / "state")
    store.save(engine)
    loaded = store.load()
    assert loaded["ok"]
    restored = loaded["engine"]
    restored.run_incremental()
    assert restored.authoritative_result() == expected


def test_snapshot_version_and_corruption_diagnostics(tmp_path: Path, bundle):
    store = SnapshotStore(tmp_path / "state")
    store.save(bundle["engine"])
    pointer_path = store.current_pointer
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    manifest_path = tmp_path / "state" / pointer["snapshot_dir"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["snapshot_format_version"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    pointer["manifest_hash"] = __import__("hashlib").sha256(manifest_path.read_bytes()).hexdigest()
    pointer_path.write_text(json.dumps(pointer), encoding="utf-8")
    assert store.load()["reason_code"] == "NER-E4001"

    store2 = SnapshotStore(tmp_path / "state2")
    store2.save(bundle["engine"])
    state_file = tmp_path / "state2" / json.loads(store2.current_pointer.read_text())["snapshot_dir"] / "state.json"
    data = json.loads(state_file.read_text(encoding="utf-8"))
    data["generation"] = 123456
    state_file.write_text(json.dumps(data), encoding="utf-8")
    assert store2.load()["reason_code"] == "NER-E4002"


def test_budget_coordinator_returns_stale_then_fresh(bundle):
    engine = bundle["engine"]
    gate = threading.Event()

    def hook(_engine, _command):
        assert gate.wait(2.0)

    coordinator = CommandCoordinator(engine, budget_ms=1, work_hook=hook)
    try:
        response = coordinator.submit({"op": "set_transition_weight", "from": "O", "to": "O", "value": 0.44}, budget_ms=5)
        assert response["code"] == "NER-E3002"
        assert response["stale_generation"] == 0
        gate.set()
        final = coordinator.wait_for_latest(timeout=3.0)
        assert final["status"] == "fresh"
        assert final["summary"]["generation"] == 1
    finally:
        gate.set()
        coordinator.shutdown()
