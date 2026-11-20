import pytest

from qu.errors import EmptyQueryError
from qu.normalization import normalize_query


def test_width_case_whitespace_punctuation():
    result = normalize_query("ＡＢＣ  Ｈｅｌｌｏ!!  世界")
    # Full-width folded, punctuation stripped, spaces merged.
    assert "  " not in result.normalized
    assert "!" not in result.normalized
    assert "abc" in result.normalized
    stages = {item["stage"] for item in result.stage_fingerprints}
    assert {"width_fold", "case_fold", "whitespace_merge",
            "punctuation_strip"}.issubset(stages)
    assert result.modifications, "modification list must be populated"


def test_alignment_maps_to_original_offsets():
    result = normalize_query("Ａ 手机")
    # Every normalized character points to a valid original index.
    assert len(result.alignment) == len(result.normalized)
    assert all(0 <= idx < len("Ａ 手机") for idx in result.alignment)


@pytest.mark.parametrize("raw", ["", "   ", "!!!", "？？ ...", "***"])
def test_empty_and_punctuation_only_rejected(raw):
    with pytest.raises(EmptyQueryError):
        normalize_query(raw)


def test_deterministic_fingerprints():
    first = normalize_query("Hello  世界!!")
    second = normalize_query("Hello  世界!!")
    assert first.stage_fingerprints == second.stage_fingerprints
    assert first.normalized == second.normalized
    assert first.modifications == second.modifications
