from fractions import Fraction

from rerank_lab.canonical import fixed_from_real
from rerank_lab.models import (
    FeatureCatalog,
    FeatureGroup,
    LabConfig,
    RawCandidate,
)
from rerank_lab.rerank import rerank_request


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


def _catalog() -> FeatureCatalog:
    return FeatureCatalog(
        feature_order=("dense.0", "dense.1"),
        groups=(FeatureGroup("dense", ("dense.0", "dense.1"), False),),
        similarity_features=("dense.0", "dense.1"),
        similarity_threshold=Fraction(1),
        raw={},
    )


def _config(**changes):
    base = dict(
        feature_weights=(Fraction(1), Fraction(1)),
        normalization="minmax",
        fusion_temperature=Fraction(1),
        score_min=Fraction(0),
        score_max=Fraction(1),
        fixed_decimal_places=4,
        quantization_step=1,
        rerank_window_size=3,
        diversity_penalty=Fraction("0.1"),
        concurrency=2,
        global_mac_budget=1000,
        per_request_quota=1000,
        cache_residency_budget=32,
        snapshot_commit_interval_commands=1,
        similarity_threshold=Fraction(1),
    )
    base.update(changes)
    return LabConfig(**base)


def _run(scores, normalized, *, config=None):
    config = config or _config()
    candidates = tuple(_candidate(item, [1, 0]) for item in scores)
    fusion = {
        "fusion_order": list(scores),
        "candidates": {
            item: {"score_fixed": score} for item, score in scores.items()
        },
        "score_min_fixed": fixed_from_real(
            config.score_min, config.fixed_decimal_places, config.quantization_step
        ),
        "score_max_fixed": fixed_from_real(
            config.score_max, config.fixed_decimal_places, config.quantization_step
        ),
    }
    return rerank_request(
        candidates,
        fusion,
        {"normalized_fixed": normalized},
        _catalog(),
        config,
    )


def test_exact_threshold_reduction_and_item_id_tie_break():
    result = _run(
        {"a": 10000, "b": 9000, "c": 8000},
        {"a": [1000, 0], "b": [1000, 0], "c": [0, 1000]},
    )
    assert result["order"][:3] == ["a", "b", "c"]
    assert len(result["reductions"]) == 1
    event = result["reductions"][0]
    assert event["item_id"] == "b"
    assert event["reduction_fixed"] == 1000
    assert event["selected_too_close"][0]["meets_threshold"] is True
    assert result["scores_after_rerank"]["b"] == result["scores_after_rerank"]["c"] == 8000


def test_reduction_never_crosses_lower_bound():
    config = _config(
        score_min=Fraction("0.2"),
        diversity_penalty=Fraction("0.1"),
        rerank_window_size=2,
    )
    result = _run(
        {"a": 10000, "b": 2500},
        {"a": [1000, 0], "b": [1000, 0]},
        config=config,
    )
    event = result["reductions"][0]
    assert event["new_score_fixed"] == 2000
    assert event["reduction_fixed"] == 500
    assert result["scores_after_rerank"]["b"] == 2000


def test_lower_score_item_id_wins_after_equal_reduction():
    result = _run(
        {"z": 10000, "b": 9500, "a": 9500},
        {"z": [1000, 0], "a": [0, 1000], "b": [0, 1000]},
    )
    assert result["rounds"][0]["selected_item_id"] == "z"
    assert result["rounds"][1]["selected_item_id"] == "a"
