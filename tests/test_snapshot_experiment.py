import json

import pytest

from qu.config import PipelineConfig
from qu.errors import E_CONFIG_MISMATCH, ConfigMismatchError
from qu.experiment import ExperimentRunner, TierSpec
from qu.loader import load_bundle
from qu.snapshot import SnapshotStore, cache_key, scope_fingerprint


def _spec(generated, tier_id="tiny"):
    return TierSpec(tier_id, generated["samples"], generated["vocab"],
                    generated["entities"], generated["rules"])


def test_snapshot_roundtrip_and_config_mismatch(tmp_path):
    store = SnapshotStore(str(tmp_path / "store"), "cfg-1")
    store.save("tiny", {"records": {"k": {"query_id": "q1"}},
                        "tier_summary": {"tier_id": "tiny"}})
    payload = store.load("tiny")
    assert payload["metadata"]["config_fingerprint"] == "cfg-1"

    other = SnapshotStore(str(tmp_path / "store"), "cfg-2")
    with pytest.raises(ConfigMismatchError) as exc:
        other.load("tiny")
    assert exc.value.code == E_CONFIG_MISMATCH


def test_cache_key_components_change_independently():
    scope = scope_fingerprint("v", "s", "tier-a")
    base = cache_key("q", scope)
    assert cache_key("q2", scope) != base
    assert cache_key("q", scope) == base
    assert scope_fingerprint("v2", "s", "tier-a") != scope
    assert scope_fingerprint("v", "s2", "tier-a") != scope
    assert scope_fingerprint("v", "s", "tier-b") != scope


def test_incremental_reuse_invalidates_only_changed(generated, tmp_path):
    config = PipelineConfig(exhaustive_verify_sample=32)
    workdir = str(tmp_path / "exp")
    runner = ExperimentRunner(config, workdir)
    bundle = load_bundle(generated["samples"], generated["vocab"],
                         generated["entities"], generated["rules"])
    first = runner.run_tier(_spec(generated), bundle)
    assert first["cache"]["hits"] == 0
    assert first["cache"]["misses"] == len(bundle.samples)

    # Second run with identical inputs: every record is reused.
    runner2 = ExperimentRunner(config, workdir)
    second = runner2.run_tier(_spec(generated), bundle)
    assert second["cache"]["hits"] == len(bundle.samples)
    assert second["cache"]["misses"] == 0
    first_records = json.load(
        open(f"{workdir}/snapshot_tiny.json", encoding="utf-8"))["records"]
    second_records = json.load(
        open(f"{workdir}/snapshot_tiny.json", encoding="utf-8"))["records"]
    assert first_records == second_records

    # Replace one query: only that sample is recomputed.
    changed = bundle.samples[0]
    changed.query = changed.query + " 新增词"
    runner3 = ExperimentRunner(config, workdir)
    third = runner3.run_tier(_spec(generated), bundle)
    assert third["cache"]["misses"] == 1
    assert third["cache"]["hits"] == len(bundle.samples) - 1
    assert third["cache"]["invalidations"] == 1


def test_full_report_contract(generated, tmp_path):
    config = PipelineConfig(exhaustive_verify_sample=64, max_workers=3)
    runner = ExperimentRunner(config, str(tmp_path / "report"))
    bundle = load_bundle(generated["samples"], generated["vocab"],
                         generated["entities"], generated["rules"])
    report = runner.run_experiment([_spec(generated)],
                                   lambda _spec: bundle)
    tier = report["tiers"][0]
    required = {
        "scale_metrics", "strategy", "timing_stats", "equivalence",
        "concurrency_comparison", "cache", "failures", "snapshot_version",
        "work_stats",
    }
    assert required <= set(tier)
    assert report["all_tiers_match_exhaustive"] is True
    assert report["all_concurrency_comparisons_equal"] is True
    for stat in ("avg_latency_ms", "p50_latency_ms", "p90_latency_ms",
                 "p99_latency_ms"):
        assert tier["timing_stats"][stat] >= 0.0
    assert tier["equivalence"]["equal_to_exhaustive"] is True
    assert tier["equivalence"]["verified_count"] == 64
    # Report file exists and parses.
    on_disk = json.load(
        open(f"{runner.workdir}/experiment_report.json", encoding="utf-8"))
    assert on_disk["config_fingerprint"] == config.fingerprint()


def test_graceful_stop_finishes_tier_snapshot(generated, tmp_path):
    config = PipelineConfig(exhaustive_verify_sample=16)
    runner = ExperimentRunner(config, str(tmp_path / "stop"))
    runner.request_stop()
    bundle = load_bundle(generated["samples"], generated["vocab"],
                         generated["entities"], generated["rules"])
    report = runner.run_experiment([_spec(generated)], lambda _: bundle)
    assert report["interrupted_graceful_stop"] is True
    # The current tier snapshot is still committed and recoverable.
    store = SnapshotStore(runner.workdir, config.fingerprint())
    assert store.load("tiny") is not None
