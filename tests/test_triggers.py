from arg_review.samplegen import schema_revisions
from arg_review.triggers import match_event_triggers


def test_longest_trigger_wins_and_short_trigger_is_suppressed():
    doc = schema_revisions()
    revision = next(r for r in doc["revisions"] if r["revision_id"] == "rev-3")
    event = {
        "event_id": "E-TRG",
        "event_type": "acquisition",
        "tokens": "northwind group completes acquisition of meridian".split(),
    }
    result = match_event_triggers(
        event, revision["event_types"]["acquisition"], revision
    )
    assert [m["term"] for m in result["matches"]] == ["completes acquisition"]
    assert len(result["suppressed"]) == 1
    suppressed = result["suppressed"][0]
    assert suppressed["term"] == "acquisition"
    assert suppressed["suppressed_by"] == result["matches"][0]["match_id"]
    assert suppressed["suppression_reason"] == "overlapping_longer_or_prior_trigger"


def test_same_character_length_uses_term_lexicographic_order():
    revision = {
        "revision_id": "lex",
        "version": 1,
        "event_types": {
            "t": {"trigger_terms": ["zz aa", "aa bb"], "roles": []}
        },
    }
    event = {"event_id": "E-LEX", "event_type": "t", "tokens": ["zz", "aa", "bb"]}
    result = match_event_triggers(
        event, revision["event_types"]["t"], revision
    )
    assert [m["term"] for m in result["matches"]] == ["aa bb"]
    assert result["suppressed"][0]["term"] == "zz aa"


def test_rollback_changes_available_trigger_rule():
    doc = schema_revisions()
    rev2 = next(r for r in doc["revisions"] if r["revision_id"] == "rev-2")
    rev3 = next(r for r in doc["revisions"] if r["revision_id"] == "rev-3")
    event = {
        "event_id": "E-ROLL",
        "event_type": "meeting",
        "tokens": "civic group hosts shanghai meeting q3".split(),
    }
    old = match_event_triggers(event, rev2["event_types"]["meeting"], rev2)
    assert any(m["term"] == "shanghai meeting" for m in old["matches"])
    assert any(s["term"] == "meeting" for s in old["suppressed"])

    rolled_back = match_event_triggers(event, rev3["event_types"]["meeting"], rev3)
    terms = {m["term"] for m in rolled_back["matches"]}
    assert "shanghai meeting" not in terms
    assert "meeting" in terms
    assert rolled_back["schema_revision_id"] == "rev-3"
