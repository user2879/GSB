from arg_review.engine import ReviewEngine
from arg_review.samplegen import generate_sample_tree
from arg_review.state import load_workspace
from arg_review.validation import validate_revision_chain, validate_workspace


def test_generated_sample_chain_and_main_events_are_valid(tmp_path):
    sample_root = tmp_path / "samples"
    generate_sample_tree(sample_root)
    state = load_workspace(sample_root)
    chain = validate_revision_chain(state.revisions_doc)
    assert chain["valid"] is True
    result = ReviewEngine(state, tmp_path / "state").run()
    assert result["report"]["status"] == "ok"
    assert result["report"]["stage_completion"]["validation"][
        "completed_valid_events"
    ] == 7


def test_invalid_entries_are_skipped_with_same_business_code(tmp_path):
    sample_root = tmp_path / "samples"
    generate_sample_tree(sample_root)
    state = load_workspace(sample_root, include_invalid=True)
    validation = validate_workspace(
        state.events,
        state.vocab_terms,
        state.revisions_doc,
        state.active_revision_id,
        parse_errors=state.parse_errors,
    )
    by_id = {
        item.get("event_id"): item for item in validation["invalid_entries"]
    }
    assert validation["revision_chain"]["valid"] is True
    assert {"X001", "X002", "X003", "X004", "X005"} <= set(by_id)
    assert any(item.get("line_no") for item in validation["invalid_entries"])
    for item in validation["invalid_entries"]:
        assert item["business_code"] == "E_SCHEMA_INVALID"
    assert any("unknown_event_type" in e for e in by_id["X001"]["errors"])
    assert any("offset_out_of_range" in e for e in by_id["X002"]["errors"])
    assert any("confidence_not_permille" in e for e in by_id["X003"]["errors"])
    assert any("relation_target_arg_missing" in e for e in by_id["X004"]["errors"])
    assert any("role_not_in_set" in e for e in by_id["X005"]["errors"])

    result = ReviewEngine(state, tmp_path / "runtime").run()
    assert result["report"]["business_code_counts"]["E_SCHEMA_INVALID"] >= 6
    assert len(result["stage_outputs"]["adjudication"]["decisions"]) == 7


def test_broken_revision_chain_is_detected():
    doc = {
        "active_revision_id": "r2",
        "revisions": [
            {"revision_id": "r1", "parent_revision": None, "version": 1},
            {"revision_id": "r2", "parent_revision": "missing", "version": 2},
        ],
    }
    result = validate_revision_chain(doc)
    assert result["valid"] is False
    assert any("revision_parent_missing" in error for error in result["errors"])


def test_revision_cycle_and_version_decrease_detected():
    doc = {
        "active_revision_id": "r2",
        "revisions": [
            {"revision_id": "r1", "parent_revision": "r2", "version": 2},
            {"revision_id": "r2", "parent_revision": "r1", "version": 1},
        ],
    }
    result = validate_revision_chain(doc)
    assert result["valid"] is False
    assert any("cycle" in error for error in result["errors"])
    assert any("version_decreases" in error for error in result["errors"])
