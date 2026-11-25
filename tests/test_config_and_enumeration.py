from __future__ import annotations

from pathlib import Path

from cross_coding.combos import enumerate_combinations
from cross_coding.config import load_bundle, load_bundle_from_docs


ROOT = Path(__file__).resolve().parents[1]


def test_sample_configs_are_valid() -> None:
    bundle, errors = load_bundle(
        ROOT / "config" / "features.yaml",
        ROOT / "config" / "policy.yaml",
    )
    assert errors.ok
    assert bundle is not None
    assert bundle["table"]["row_count"] == 72
    assert len(bundle["table"]["columns"]["city"]["missing_bitmap"]) == 72


def test_validation_collects_stable_config_errors() -> None:
    feature_doc = {
        "seed": "not-int",
        "row_count": 2,
        "columns": [
            {"name": "same", "type": "categorical", "categories": ["a"]},
            {"name": "same", "type": "categorical", "categories": ["b"]},
            {"name": "num", "type": "numeric", "min": 9, "max": 1, "values": [0, 500]},
        ],
    }
    policy_doc = {
        "candidate_pairs": [["same", "missing"], ["same", "num"]],
        "max_order": 0,
        "cardinality_budget": -1,
        "encoding_point_budget": -2,
        "encoding_width_bits": 7,
        "fixed_point_bits": 13,
        "quantization_levels": 17,
    }
    bundle, report = load_bundle_from_docs(feature_doc, policy_doc)
    assert bundle is None
    assert report.errors
    assert all(item["code"] == "E_CONFIG_INVALID" for item in report.errors)
    paths = [item["path"] for item in report.errors]
    assert paths == sorted(paths, key=lambda p: (not p.startswith("feature_table"), p)) or len(paths) > 5
    assert any(item["detail_code"] == "E_UNKNOWN_COLUMN" for item in report.errors)
    assert any(item["detail_code"] == "E_REFERENCE_NOT_CATEGORICAL" for item in report.errors)
    assert any(item["detail_code"] == "E_WIDTH_NOT_ALLOWED" for item in report.errors)
    assert any(item["detail_code"] == "E_FIXED_POINT_NOT_ALLOWED" for item in report.errors)


def test_independent_injection_configs_return_expected_codes() -> None:
    cases = {
        "value_out_of_range": "E_VALUE_OUT_OF_RANGE",
        "unknown_pair": "E_UNKNOWN_COLUMN",
        "invalid_width": "E_WIDTH_NOT_ALLOWED",
    }
    for case, detail_code in cases.items():
        _bundle, report = load_bundle(
            ROOT / "config" / "injections" / case / "features.yaml",
            ROOT / "config" / "injections" / case / "policy.yaml",
        )
        assert not report.ok, case
        assert all(item["code"] == "E_CONFIG_INVALID" for item in report.errors)
        assert any(item["detail_code"] == detail_code for item in report.errors), case


def test_enumeration_order_and_connected_expansion(bundle) -> None:
    combos = enumerate_combinations(bundle["policy"], bundle["table"])
    keys = [item["canonical_key"] for item in combos]
    assert keys == [
        "a",
        "b",
        "c",
        "d",
        "a|b",
        "b|c",
        "c|d",
        "a|b|c",
        "b|c|d",
    ]
    by_key = {item["canonical_key"]: item for item in combos}
    assert [pair[0] + "|" + pair[1] for pair in by_key["a|b|c"]["reusable_pairs"]] == ["a|b", "b|c"]
    assert by_key["a|b|c"]["task_sort_keys"]["coding"]


def test_column_and_pair_changes_change_fingerprints(bundle) -> None:
    before = {item["canonical_key"]: item for item in enumerate_combinations(bundle["policy"], bundle["table"])}
    bundle["table"]["columns"]["a"]["fingerprint"] = "col:changed"
    after = {item["canonical_key"]: item for item in enumerate_combinations(bundle["policy"], bundle["table"])}
    assert after["a|b"]["content_fingerprint"] != before["a|b"]["content_fingerprint"]
    assert after["b|c"]["content_fingerprint"] == before["b|c"]["content_fingerprint"]

    bundle["policy"]["candidate_pairs"].append(["a", "c"])
    pair_after = {item["canonical_key"]: item for item in enumerate_combinations(bundle["policy"], bundle["table"])}
    assert "a|c" in pair_after
    assert pair_after["a|b|c"]["content_fingerprint"] != before["a|b|c"]["content_fingerprint"]
