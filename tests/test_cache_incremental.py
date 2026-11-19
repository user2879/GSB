from __future__ import annotations

from feature_diag.config import load_experiment
from feature_diag.evaluation import run_experiment


def _entries_by(ledger, **filters):
    output = []
    for entry in ledger["entries"]:
        if all(entry.get(key) == value for key, value in filters.items()):
            output.append(entry)
    return output


def test_pca_parameter_change_only_invalidates_pca_and_downstream(make_experiment, tmp_path) -> None:
    original = load_experiment(make_experiment(pca_components=3, cache_dir="shared-cache", snapshot_dir="snap-a"))
    first = run_experiment(
        original,
        output_dir=tmp_path / "first",
        cache_dir=tmp_path / "shared-cache",
        snapshot_dir=tmp_path / "snap-a",
        max_workers=3,
    )
    assert first.report is not None

    changed = load_experiment(make_experiment(pca_components=2, cache_dir="shared-cache", snapshot_dir="snap-b"))
    second = run_experiment(
        changed,
        output_dir=tmp_path / "second",
        cache_dir=tmp_path / "shared-cache",
        snapshot_dir=tmp_path / "snap-b",
        max_workers=3,
    )
    assert second.report is not None
    ledger = second.report["cache_ledger"]

    crossing_entries = _entries_by(ledger, artifact_kind="crossing")
    standard_entries = _entries_by(ledger, artifact_kind="standardization")
    pca_reductions = _entries_by(ledger, artifact_kind="reduction", path="pca")
    rp_reductions = _entries_by(ledger, artifact_kind="reduction", path="random_projection")
    variance_reductions = _entries_by(ledger, artifact_kind="reduction", path="variance_filter")
    pca_metrics = _entries_by(ledger, artifact_kind="metrics", path="pca")
    rp_metrics = _entries_by(ledger, artifact_kind="metrics", path="random_projection")
    variance_metrics = _entries_by(ledger, artifact_kind="metrics", path="variance_filter")

    assert crossing_entries and all(entry["status"] == "hit" for entry in crossing_entries)
    assert standard_entries and all(entry["status"] == "hit" for entry in standard_entries)
    assert pca_reductions and all(entry["status"] == "invalidated" for entry in pca_reductions)
    assert all(entry["previous_key"] for entry in pca_reductions)
    assert rp_reductions and all(entry["status"] == "hit" for entry in rp_reductions)
    assert variance_reductions and all(entry["status"] == "hit" for entry in variance_reductions)
    assert pca_metrics and all(entry["status"] == "invalidated" for entry in pca_metrics)
    assert rp_metrics and all(entry["status"] == "hit" for entry in rp_metrics)
    assert variance_metrics and all(entry["status"] == "hit" for entry in variance_metrics)


def test_cardinality_budget_change_invalidates_crossing_and_all_downstream(make_experiment, tmp_path) -> None:
    first = load_experiment(make_experiment(cache_dir="budget-cache", snapshot_dir="snap-1"))
    run_experiment(
        first,
        output_dir=tmp_path / "b1",
        cache_dir=tmp_path / "budget-cache",
        snapshot_dir=tmp_path / "snap-1",
        max_workers=3,
    )
    second = load_experiment(make_experiment(cache_dir="budget-cache", snapshot_dir="snap-2"))
    scenario = second.scenarios[0]
    scenario.cardinality_budget = scenario.cardinality_budget + 1
    second.config_hash = "changed-budget-" + second.config_hash
    result = run_experiment(
        second,
        output_dir=tmp_path / "b2",
        cache_dir=tmp_path / "budget-cache",
        snapshot_dir=tmp_path / "snap-2",
        max_workers=3,
    )
    assert result.report is not None
    entries = result.report["cache_ledger"]["entries"]
    crossing = [entry for entry in entries if entry["artifact_kind"] == "crossing"]
    assert crossing and all(entry["status"] == "invalidated" for entry in crossing)
    assert any(entry["artifact_kind"] == "metrics" and entry["status"] == "invalidated" for entry in entries)


def test_sample_size_tier_change_reuses_unchanged_tier_and_retires_old_tier(make_experiment, tmp_path) -> None:
    first = load_experiment(
        make_experiment(cache_dir="size-cache", snapshot_dir="size-snap-1", sample_sizes=(64, 96))
    )
    run_experiment(
        first,
        output_dir=tmp_path / "s1",
        cache_dir=tmp_path / "size-cache",
        snapshot_dir=tmp_path / "size-snap-1",
        max_workers=3,
    )
    second = load_experiment(
        make_experiment(cache_dir="size-cache", snapshot_dir="size-snap-2", sample_sizes=(64, 128))
    )
    result = run_experiment(
        second,
        output_dir=tmp_path / "s2",
        cache_dir=tmp_path / "size-cache",
        snapshot_dir=tmp_path / "size-snap-2",
        max_workers=3,
    )
    assert result.report is not None
    entries = result.report["cache_ledger"]["entries"]
    unchanged = [entry for entry in entries if entry["sample_size"] == 64]
    new_tier = [entry for entry in entries if entry["sample_size"] == 128]
    old_tier = [entry for entry in entries if entry["sample_size"] == 96]
    assert unchanged and all(entry["status"] == "hit" for entry in unchanged)
    assert new_tier and all(entry["status"] in {"miss", "invalidated"} for entry in new_tier)
    assert old_tier and all(entry["status"] == "retired" for entry in old_tier)
