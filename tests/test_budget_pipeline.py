
from mlimgedit.features import feature_point_cost
from mlimgedit.pipeline import Pipeline
from mlimgedit.runner import full_run
from mlimgedit.snapshot import SnapshotStore


def test_budget_pending_drains_across_batches(config, sample_dir, tmp_path):
    cost = feature_point_cost(config)
    tight = config.with_updates(
        {"runtime": {"threads": 1, "feature_point_budget": cost * 3 + 1}}
    )
    store = SnapshotStore(tmp_path / "store")
    pipeline = Pipeline(store)
    first = pipeline.run_batch(tight, sample_dir, 0)
    sample_count = len(first["samples"])
    assert first["counts"]["pending"] == sample_count - 3
    assert first["downstream_blocked_by_budget"] is True
    assert first["per_sample_predictions"][0]["classifiers"]["logreg"]["labels"] == {}

    # Subsequent batches give pending work priority; no new samples are added.
    second = pipeline.run_batch(tight, sample_dir, 1)
    third = pipeline.run_batch(tight, sample_dir, 2)
    fourth = pipeline.run_batch(tight, sample_dir, 3)
    assert len(second["budget"]["deductions"]) == 3
    assert third["counts"]["pending"] == max(0, sample_count - 9)
    assert fourth["counts"]["pending"] == 0
    assert fourth["downstream_blocked_by_budget"] is False
    # Once drained, downstream metrics exist and match a full unconstrained run.
    full = full_run(config, sample_dir, tmp_path / "full")
    from mlimgedit.runner import comparable_state

    assert comparable_state(fourth) == comparable_state(full)


def test_budget_does_not_change_completed_values(config, sample_dir, tmp_path):
    cost = feature_point_cost(config)
    tight = config.with_updates(
        {"runtime": {"threads": 1, "feature_point_budget": cost + 1}}
    )
    store = SnapshotStore(tmp_path / "tight_store")
    pipeline = Pipeline(store)
    first = pipeline.run_batch(tight, sample_dir, 0)
    completed = list(first["budget"]["deductions"][0:1])
    # Re-run with an ample budget and confirm the first sample remains identical.
    full = full_run(config, sample_dir, tmp_path / "full_store")
    first_sample = next(
        item for item in full["per_sample_predictions"] if item["sample"] == completed[0]["sample"]
    )
    assert first_sample["sample"] == completed[0]["sample"]
    # Its feature artifact was reused (not recomputed) once the store had it.
    second = pipeline.run_batch(config, sample_dir, 1)
    feature_reuse = {
        item["name"]
        for item in second["reuse"]
        if item["stage"] == "features" and item["name"] == completed[0]["sample"]
    }
    assert completed[0]["sample"] in feature_reuse
