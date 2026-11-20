from qu.entity import EntityMatcher
from qu.loader import EntityEntry


def _entry(entity_type, term, entity_id, weight=1.0):
    return EntityEntry(entity_type, term, entity_id, weight)


def test_maximum_match_and_suppression():
    entries = [
        _entry("brand", "星河", "b1"),
        _entry("category", "星河手机", "c1"),
        _entry("attribute", "手机", "a1"),
    ]
    matcher = EntityMatcher(entries, ("brand", "category", "attribute"))
    result = matcher.match("星河手机新款", "exhaustive")
    terms = [(slot["term"], slot["start"], slot["end"]) for slot in result.slots]
    # Longest span wins; shorter overlapping candidates are suppressed.
    assert ("星河手机", 0, 4) in terms
    assert ("新款", 0, 1) not in terms
    suppressed_terms = {item["term"] for item in result.suppressed}
    assert {"星河", "手机"} <= suppressed_terms
    assert {item["rule"] for item in result.resolutions} == {"longest_match"}


def test_tie_resolution_priority_then_position():
    entries = [
        _entry("attribute", "苹果", "a1"),
        _entry("brand", "苹果", "b1"),
    ]
    matcher = EntityMatcher(entries, ("brand", "category", "attribute"))
    result = matcher.match("苹果手机", "exhaustive")
    assert len(result.slots) == 1
    assert result.slots[0]["entity_type"] == "brand"
    assert result.resolutions[0]["rule"] == "dict_priority"


def test_three_modes_identical_results_but_different_work():
    entries = [
        _entry("brand", f"{chr(0x4e00 + (i % 64))}牌{i:04d}号", f"b{i}")
        for i in range(400)
    ] + [
        _entry("category", f"{chr(0x4e00 + 1)}牌0001号手机", "c1"),
        _entry("attribute", "红色", "attr1"),
    ]
    matcher = EntityMatcher(entries, ("brand", "category", "attribute"))
    text = f"红色 {chr(0x4e00 + 1)}牌0001号手机 {chr(0x4e00 + 2)}牌0002号"
    outcomes = {mode: matcher.match(text, mode)
                for mode in ("exhaustive", "inverted", "prefix_group")}
    reference = outcomes["exhaustive"]
    for mode in ("inverted", "prefix_group"):
        other = outcomes[mode]
        assert other.slots == reference.slots
        assert other.suppressed == reference.suppressed
        assert other.resolutions == reference.resolutions
        assert other.match_checks < reference.match_checks, (
            f"{mode} must perform fewer probes than exhaustive")


def test_generated_dictionary_modes_equivalent(generated):
    from qu.loader import load_entity_dict

    entries, failures = load_entity_dict(generated["entities"])
    assert failures == []
    matcher = EntityMatcher(entries, ("brand", "category", "attribute"))
    text = "朱雀 平板 防水 新款"
    reference = matcher.match(text, "exhaustive")
    for mode in ("inverted", "prefix_group"):
        outcome = matcher.match(text, mode)
        assert outcome.slots == reference.slots
        assert outcome.suppressed == reference.suppressed
