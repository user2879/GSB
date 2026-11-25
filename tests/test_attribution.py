from __future__ import annotations

from vscan.attribution import batch_attribution, integer_contribution_ppm


def _assert_no_float(value):
    if isinstance(value, dict):
        for item in value.values():
            _assert_no_float(item)
    elif isinstance(value, list):
        for item in value:
            _assert_no_float(item)
    else:
        assert not isinstance(value, float)


def test_batch_attribution_exact_integers_and_components():
    result = batch_attribution(
        rows=10,
        all_column_count=4,
        decode_column_count=3,
        predicate_column_count=1,
        passed_rows=4,
        predicate_leaf_count=2,
        decoded_cell_bytes=9,
        tolerance_ppm=2,
    )
    assert result["uncropped_decoded_bytes"] == 360
    assert result["pruned_full_scan_decoded_bytes"] == 270
    assert result["actual_decoded_bytes"] == 162
    assert result["bytes"] == {
        "column_pruning": 90,
        "predicate_pushdown": 108,
        "batch_partition": 0,
        "total": 198,
    }
    assert result["candidate_rows_before"] == 10
    assert result["candidate_rows_after"] == 4
    assert result["comparisons"]["before"] == 20
    assert result["comparisons"]["after"] == 8
    assert result["comparisons"]["total_saved"] == 12
    _assert_no_float(result)


def test_two_calibers_are_stable_and_tolerance_marked():
    result = integer_contribution_ppm(
        [("column_pruning", 90), ("predicate_pushdown", 108), ("batch_partition", 0)],
        tolerance_ppm=2,
    )
    floor = {item["name"]: item["ppm"] for item in result["calibers"]["floor"]}
    largest = {item["name"]: item["ppm"] for item in result["calibers"]["largest_remainder"]}
    assert sum(floor.values()) <= 1_000_000
    assert sum(largest.values()) == 1_000_000
    assert result["selected_caliber"] == "floor"
    assert result["selected"] == result["calibers"]["floor"]
    assert result["tolerance_affected"] is True
    assert result["max_caliber_difference_ppm"] <= 2


def test_zero_denominator_is_deterministic():
    result = integer_contribution_ppm([("a", 0), ("b", 0), ("c", 0)], 2)
    assert [item["ppm"] for item in result["selected"]] == [0, 0, 0]
    assert result["tolerance_affected"] is False
