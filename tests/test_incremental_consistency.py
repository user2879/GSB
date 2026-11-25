from __future__ import annotations

from cross_coding.pipeline import comparison_view, run_pipeline
from cross_coding.replay import apply_edit


EDIT_SEQUENCE = [
    {"type": "add_pair", "pair": ["a", "c"]},
    {"type": "set_max_order", "value": 4},
    {"type": "set_cardinality_budget", "value": 6},
    {"type": "set_point_budget", "value": 1800},
    {"type": "set_encoding_width", "value": 16},
    {"type": "set_fixed_point_bits", "value": 24},
    {"type": "set_quantization_levels", "value": 32},
    {"type": "replace_column", "column": "a", "replacement_seed": 7788},
    {"type": "remove_pair", "pair": ["c", "d"]},
    {"type": "set_point_budget", "value": 10_000_000},
    {"type": "set_encoding_width", "value": 8},
    {"type": "set_max_order", "value": 2},
    {"type": "set_cardinality_budget", "value": 18},
]


def test_arbitrary_edit_sequence_matches_blank_full_rebuild(bundle, tmp_path) -> None:
    incremental_bundle = bundle
    full_bundle = bundle
    incremental_state = tmp_path / "incremental"

    for cursor, edit in enumerate(EDIT_SEQUENCE, start=1):
        incremental_bundle = apply_edit(incremental_bundle, edit)
        full_bundle = apply_edit(full_bundle, edit)

        incremental_report = run_pipeline(
            incremental_bundle,
            str(incremental_state),
            command_cursor=cursor,
            parallel=True,
        )
        full_report = run_pipeline(
            full_bundle,
            str(tmp_path / f"blank-{cursor}"),
            command_cursor=cursor,
            parallel=False,
        )

        incremental_view = comparison_view(incremental_report)
        full_view = comparison_view(full_report)
        assert incremental_view == full_view, f"comparison mismatch after edit {cursor}: {edit}"


def test_only_referenced_columns_invalidate_coding_cache(feature_doc, policy_doc, tmp_path) -> None:
    policy_doc["candidate_pairs"] = [["a", "b"], ["c", "d"]]
    policy_doc["max_order"] = 2
    from cross_coding.config import load_bundle_from_docs

    bundle, errors = load_bundle_from_docs(feature_doc, policy_doc)
    assert errors.ok
    state = tmp_path / "precise"
    first = run_pipeline(bundle, str(state))
    assert first["status"] == "completed"

    changed = apply_edit(bundle, {"type": "replace_column", "column": "a", "replacement_seed": 31337})
    second = run_pipeline(changed, str(state))
    coding = {
        entry["canonical_key"]: entry
        for entry in second["ledger"]["entries"]
        if entry["stage"] == "coding"
    }
    assert coding["a"]["result"] == "recomputed"
    assert coding["a|b"]["result"] == "recomputed"
    assert any(reason.startswith("column_content_changed:a") for reason in coding["a|b"]["basis"])
    assert coding["c"]["result"] == "reuse_hit"
    assert coding["d"]["result"] == "reuse_hit"
    assert coding["c|d"]["result"] == "reuse_hit"


def test_pair_addition_reuses_unaffected_combination(feature_doc, policy_doc, tmp_path) -> None:
    policy_doc["candidate_pairs"] = [["a", "b"], ["c", "d"]]
    policy_doc["max_order"] = 2
    from cross_coding.config import load_bundle_from_docs

    bundle, errors = load_bundle_from_docs(feature_doc, policy_doc)
    assert errors.ok
    state = tmp_path / "pair-cache"
    run_pipeline(bundle, str(state))

    changed = apply_edit(bundle, {"type": "add_pair", "pair": ["a", "c"]})
    report = run_pipeline(changed, str(state))
    coding = {
        entry["canonical_key"]: entry["result"]
        for entry in report["ledger"]["entries"]
        if entry["stage"] == "coding"
    }
    assert coding["a|b"] == "reuse_hit"
    assert coding["c|d"] == "reuse_hit"
    assert coding["a|c"] == "recomputed"
