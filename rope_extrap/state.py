from __future__ import annotations

from typing import Any

from .loader import parse_position_config, parse_sample_object
from .models import AttentionSample, PositionEncodingConfig


def state_from_sequence_result(
    result: dict[str, Any],
) -> tuple[PositionEncodingConfig | None, dict[str, AttentionSample], list[str]]:
    config, config_failures = parse_position_config(result.get("current_config"))
    if config is None:
        return None, {}, [failure.message for failure in config_failures]
    samples: dict[str, AttentionSample] = {}
    errors: list[str] = []
    for sample_id, raw in sorted(result.get("current_samples", {}).items()):
        sample, sample_failures = parse_sample_object(
            raw,
            dimension=config.dimension,
            target_length=config.extrapolation_target_length,
            feature_line=int(raw.get("feature_line", 0)),
            source_ref=str(raw.get("source_ref", "sequence-result")),
        )
        if sample is None:
            errors.extend(failure.message for failure in sample_failures)
        else:
            samples[sample.sample_id] = sample
    return config, samples, errors

