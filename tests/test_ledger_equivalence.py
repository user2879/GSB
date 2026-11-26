from arg_review.diagnostics import run_edit_equivalence
from arg_review.engine import ReviewEngine, projections_equal
from arg_review.samplegen import generate_sample_tree
from arg_review.state import apply_command, load_workspace


def test_repeat_run_reuses_all_fingerprinted_artifacts(tmp_path):
    root = tmp_path / "samples"
    generate_sample_tree(root)
    state = load_workspace(root)
    state_dir = tmp_path / "ledger"
    first = ReviewEngine(state, state_dir).run()
    second = ReviewEngine(state, state_dir).run()
    assert projections_equal(first["projection"], second["projection"])
    assert second["report"]["ledger_summary"]["reuse_count"] == 43
    assert second["report"]["ledger_summary"]["recomputed_count"] == 0


def test_role_weight_change_invalidates_only_role_downstream(tmp_path):
    root = tmp_path / "samples"
    generate_sample_tree(root)
    state_dir = tmp_path / "ledger"
    ReviewEngine(load_workspace(root), state_dir).run()

    changed = load_workspace(root)
    changed.config["role_weights_permille"]["host"] = 990
    incremental = ReviewEngine(changed, state_dir).run()
    full = ReviewEngine(changed, tmp_path / "full").run()

    assert projections_equal(incremental["projection"], full["projection"])
    recomputed = incremental["ledger_lists"]["restricted_recompute_set"]
    assert "arg:E005:E005-A1" in recomputed
    assert "decision:E005" in recomputed
    assert any(item.startswith("merge:") for item in recomputed)
    assert "trigger:E001" in incremental["ledger_lists"]["reuse_artifact_ids"]
    assert "arg:E001:E001-A1" in incremental["ledger_lists"]["reuse_artifact_ids"]
    assert incremental["report"]["ledger_summary"]["reuse_count"] == 40


def test_vocabulary_removal_skips_only_referencing_event(tmp_path):
    root = tmp_path / "samples"
    generate_sample_tree(root)
    state_dir = tmp_path / "ledger"
    ReviewEngine(load_workspace(root), state_dir).run()

    changed = load_workspace(root)
    apply_command(changed, {"op": "update_vocabulary", "remove_terms": ["monday"]})
    result = ReviewEngine(changed, state_dir).run()
    decision_ids = result["projection"]["orders"]["decision_ids"]
    assert "decision:E001" not in decision_ids
    assert "decision:E002" in decision_ids
    assert result["report"]["business_code_counts"]["E_SCHEMA_INVALID"] >= 1
    assert len(result["ledger_lists"]["reuse_hits"]) >= 35


def test_command_sequence_incremental_matches_blank_full_every_step(tmp_path):
    root = tmp_path / "samples"
    paths = generate_sample_tree(root)
    report = run_edit_equivalence(
        root, paths["commands"], tmp_path / "equivalence"
    )
    assert report["command_count"] == 13
    assert report["completed_steps"] == 13
    assert report["all_steps_equal"] is True
    for step in report["steps"]:
        assert step["equal"] is True
        assert step["incremental_projection_fingerprint"] == (
            step["full_projection_fingerprint"]
        )
