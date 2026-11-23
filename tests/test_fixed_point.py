from fractions import Fraction

from rerank_lab.canonical import fixed_from_real, round_half_away
from rerank_lab.models import (
    FeatureCatalog,
    FeatureGroup,
    LabConfig,
    RawCandidate,
)
from rerank_lab.normalize import compute_normalization
from rerank_lab.scoring import fuse_request


def _candidate(item: str, dense: list[int]) -> RawCandidate:
    return RawCandidate(
        request_id="r",
        item_id=item,
        features={f"dense.{index}": Fraction(value) for index, value in enumerate(dense)},
        recall_scores={},
        dense_features=tuple(Fraction(value) for value in dense),
        sparse_features={},
        line=1,
        source_record={},
    )


def _catalog(threshold: Fraction = Fraction(1)) -> FeatureCatalog:
    return FeatureCatalog(
        feature_order=("dense.0", "dense.1"),
        groups=(FeatureGroup("dense", ("dense.0", "dense.1"), False),),
        similarity_features=("dense.0", "dense.1"),
        similarity_threshold=threshold,
        raw={"feature_order": ["dense.0", "dense.1"]},
    )


def _config(**changes):
    values = dict(
        feature_weights=(Fraction(1), Fraction(1)),
        normalization="minmax",
        fusion_temperature=Fraction(1),
        score_min=Fraction(-2),
        score_max=Fraction(2),
        fixed_decimal_places=4,
        quantization_step=1,
        rerank_window_size=2,
        diversity_penalty=Fraction(0),
        concurrency=2,
        global_mac_budget=1000,
        per_request_quota=1000,
        cache_residency_budget=32,
        snapshot_commit_interval_commands=1,
    )
    values.update(changes)
    return LabConfig(**values)


def test_minmax_zero_denominator_rule_and_contribution_sum():
    candidates = (_candidate("b", [5, 1]), _candidate("a", [5, 3]))
    catalog = _catalog()
    config = _config(feature_weights=(Fraction(1), Fraction(2)))
    normalization = compute_normalization(candidates, catalog, "minmax", 4, 1)
    assert {
        (item["feature"], item["rule"])
        for item in normalization["zero_denominators"]
    } == {("dense.0", "zero_denominator_to_zero")}
    assert normalization["normalized_fixed"]["a"] == [0, 10000]
    assert normalization["normalized_fixed"]["b"] == [0, 0]

    fusion = fuse_request(candidates, normalization, catalog, config)
    for score in fusion["candidates"].values():
        assert sum(c["contribution_fixed"] for c in score["contributions"]) == score[
            "raw_score_fixed"
        ]
    assert fusion["fusion_order"] == ["a", "b"]


def test_zscore_and_clamp_are_deterministic():
    candidates = (
        _candidate("a", [1, 2]),
        _candidate("b", [3, 4]),
        _candidate("c", [5, 6]),
    )
    catalog = _catalog()
    config = _config(
        normalization="zscore",
        feature_weights=(Fraction(10), Fraction(10)),
        score_min=Fraction(-0.1),
        score_max=Fraction(0.1),
    )
    first = compute_normalization(candidates, catalog, "zscore", 4, 1)
    second = compute_normalization(candidates, catalog, "zscore", 4, 1)
    assert first == second
    fusion = fuse_request(candidates, first, catalog, config)
    assert {score["score_fixed"] for score in fusion["candidates"].values()} == {
        -1000,
        0,
        1000,
    }


def test_round_half_away_and_quantisation_step():
    assert round_half_away(Fraction(1, 2)) == 1
    assert round_half_away(Fraction(-1, 2)) == -1
    assert fixed_from_real(Fraction(1, 2), 3, 2) == 250
