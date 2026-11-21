import json

import pytest

from pe_diag.cache import make_cache_key, sample_fingerprint
from pe_diag.config import Config, load_injections, load_json, load_samples
from pe_diag.errors import E_SNAPSHOT_STALE
from pe_diag.pipeline import DiagnosticPipeline
from pe_diag.snapshot import FORMAT_VERSION, SnapshotStore

FIXTURE = "/tmp/smoke"


def _load():
    cfg = Config.load(FIXTURE + "/position_encoding_config.json")
    samples = load_samples(FIXTURE + "/attention_samples.jsonl")
    scale = load_json(FIXTURE + "/scale_descriptor.json")
    injections = load_injections(FIXTURE + "/injection_manifest.json")
    return cfg, samples, scale, injections


def test_cache_key_changes_with_each_factor(small_config, sample):
    decision = {"scheme": "rope", "active_dim": 16, "dim_range": [0, 16],
                "block_method": "single", "layer_precision": "float64",
                "scheme_priority": ["rope"]}
    base = make_cache_key("score", "rope", small_config, sample,
                          "float64", decision)
    # precision change
    assert base != make_cache_key("score", "rope", small_config, sample,
                                  "float32", decision)
    # stage change
    assert base != make_cache_key("encode", "rope", small_config, sample,
                                  "float64", decision)
    # base-frequency change alters the scheme-parameter fingerprint
    cfg2 = small_config.with_overrides(base=5000.0)
    assert base != make_cache_key("score", "rope", cfg2, sample,
                                  "float64", decision)
    # sample content change alters the sample fingerprint
    import dataclasses
    changed = dataclasses.replace(
        sample, query_features=(0.0,) + sample.query_features[1:])
    assert sample_fingerprint(changed) != sample_fingerprint(sample)


def test_snapshot_stale_on_format_bump(tmp_path):
    hashes = {"config_hash": "x", "injection_hash": "y", "scale_hash": "z",
              "samples_hash": "w"}
    store = SnapshotStore(tmp_path / "snap", hashes)
    assert store.status == "fresh"
    # corrupt the format version
    manifest = tmp_path / "snap" / "generation-0" / "manifest.json"
    doc = json.loads(manifest.read_text())
    doc["format_version"] = FORMAT_VERSION + 1
    manifest.write_text(json.dumps(doc))
    store2 = SnapshotStore(tmp_path / "snap", hashes)
    assert store2.status == E_SNAPSHOT_STALE
    # a new generation directory is created; the old one is untouched
    assert (tmp_path / "snap" / "generation-0").exists()
    assert (tmp_path / "snap" / "generation-1").exists()


def test_atomic_object_commit_and_resume(tmp_path):
    hashes = {"config_hash": "c", "injection_hash": "i", "scale_hash": "s",
              "samples_hash": "h"}
    store = SnapshotStore(tmp_path / "snap", hashes)
    store.commit("score", {"k1": {"a": 1}}, ["rope|s|score"])
    assert store.has_entry("score", "k1")
    assert store.load_object("score", "k1") == {"a": 1}
    # resume: same hashes -> resumed, entry still readable
    store_b = SnapshotStore(tmp_path / "snap", hashes)
    assert store_b.status == "resumed"
    assert store_b.load_object("score", "k1") == {"a": 1}


def test_pipeline_failure_isolation(tmp_path):
    cfg, samples, scale, injections = _load()
    pipe = DiagnosticPipeline(cfg, samples, injections, scale,
                              str(tmp_path / "snap"))
    report = pipe.run(parallel=False)
    codes = set(report["failure_codes"])
    assert {
        "E_NONFINITE_INPUT", "E_KEY_POS_OOR",
        "E_BASE_ZERO", "E_SCALE_INVALID"} <= codes
    # every failure carries the manifest injection id
    for failure in report["failures"]:
        assert failure["injection_id"].startswith("INJ-")
    # failures are excluded from statistics but other samples still run
    stats = report["tier_reports"]["small"]["layered_statistics"]["schemes"]
    total_scored = sum(
        len(node["per_sample"]) for node in stats.values())
    assert total_scored > 0
