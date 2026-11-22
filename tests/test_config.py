from __future__ import annotations

from iidx.config import AppConfig, config_from_mapping


def test_default_config_is_valid() -> None:
    config, failures = config_from_mapping(AppConfig().to_dict())
    assert failures == []
    assert config.thread_pool_size <= config.concurrency_quota
    assert list(config.compression_switch_points) == sorted(set(config.compression_switch_points))


def test_config_collects_business_codes_and_uses_safe_fallbacks() -> None:
    config, failures = config_from_mapping(
        {
            "chunk_size": 0,
            "batch_max_docs": 0,
            "thread_pool_size": 9,
            "concurrency_quota": 4,
            "compression_switch_points": [5, 5],
            "quantization_tier": 99,
            "compression_tier": "invalid",
        }
    )
    codes = {item["code"] for item in failures}
    assert "E_CONFIG_VALUE_RANGE" in codes
    assert "E_CONFIG_CONCURRENCY_EXCEEDS_QUOTA" in codes
    assert "E_CONFIG_SWITCH_POINTS_NOT_INCREASING" in codes
    assert "E_CONFIG_QUANTIZATION_TIER" in codes
    assert "E_CONFIG_COMPRESSION_TIER" in codes
    assert config.thread_pool_size == 4
    assert config.chunk_size >= 1
    assert config.batch_max_docs >= 1


def test_auto_tier_switch_points_toggle_deterministically() -> None:
    config, failures = config_from_mapping({"compression_tier": "auto", "compression_switch_points": [2, 5]})
    assert failures == []
    assert config.selected_tier(0, 100) == (1, "auto_initial", None)
    assert config.selected_tier(2, 100) == (2, "auto_toggle", 2)
    assert config.selected_tier(5, 100) == (1, "auto_toggle", 5)
