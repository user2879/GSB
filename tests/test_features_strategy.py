from __future__ import annotations

from pref_reward_lab.config import LabConfig
from pref_reward_lab.features import FeatureParams, extract_pair_features
from pref_reward_lab.models import RawPair
from pref_reward_lab.sample import default_config
from pref_reward_lab.strategy import select_strategy


def cfg(**updates):
    value = default_config()
    value.update(updates)
    return LabConfig.from_dict(value)


def pair() -> RawPair:
    return RawPair(
        pair_id="pair-000001",
        prompt="Amber  anchor",
        chosen="Cedar circle cobalt",
        rejected="Delta drift ember",
        annotator_id="ann-01",
        timestamp="2026-01-01T00:00:00Z",
    )


def test_feature_extraction_is_deterministic_and_normalized() -> None:
    config = cfg()
    params = FeatureParams.from_config(
        config, precision_tier="balanced", hash_buckets=257, feature_dim=48
    )
    first = extract_pair_features(pair(), params)
    second = extract_pair_features(pair(), params)
    assert first.content_dict() == second.content_dict()
    assert first.buckets == second.buckets
    assert len(first.buckets) <= 257
    assert first.precision_fingerprint == second.precision_fingerprint

    unnormalized = extract_pair_features(
        pair(),
        FeatureParams.from_config(
            config, precision_tier="balanced", hash_buckets=257, feature_dim=48
        )
        .__class__(
            tokenizer_version=params.tokenizer_version,
            precision_tier=params.precision_tier,
            hash_buckets=params.hash_buckets,
            feature_dim=params.feature_dim,
            normalize_text=False,
        ),
    )
    assert unnormalized.precision_fingerprint != first.precision_fingerprint


def test_strategy_force_changes_real_bucket_space_and_precision() -> None:
    config = cfg(forced_strategy_tier="small")
    pairs = [pair()]
    small = select_strategy(pairs, config)
    large = select_strategy(pairs, config.with_updates({"forced_strategy_tier": "large"}))
    assert small.effective_hash_buckets == 128
    assert large.effective_hash_buckets == 512
    assert small.precision_tier == "compact"
    assert large.precision_tier == "fine"
    assert small.selected_tier != large.selected_tier
    assert small.to_dict() != large.to_dict()
    assert small.metrics["pair_count"] == 1

    small_features = extract_pair_features(
        pair(),
        FeatureParams.from_config(
            config,
            precision_tier=small.precision_tier,
            hash_buckets=small.effective_hash_buckets,
        ),
    )
    large_features = extract_pair_features(
        pair(),
        FeatureParams.from_config(
            config,
            precision_tier=large.precision_tier,
            hash_buckets=large.effective_hash_buckets,
        ),
    )
    assert small_features.buckets != large_features.buckets
