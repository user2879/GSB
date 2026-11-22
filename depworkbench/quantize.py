from __future__ import annotations

import math

from .canonical import quantize_decimal, sha256_json
from .errors import ValidationError
from .models import ACTION_NAMES, Config, Model, QuantizedModel


def quantize_model(model: Model, config: Config) -> QuantizedModel:
    weights: dict[str, tuple[int, ...]] = {}
    for action in ACTION_NAMES:
        values = model.action_weights[action]
        if not all(math.isfinite(value) for value in values):
            raise ValidationError("E_MODEL_INVALID", f"action weight for {action} is not finite")
        weights[action] = tuple(
            quantize_decimal(
                value,
                config.fixed_point_decimals,
                config.quantization_level,
            )
            for value in values
        )
    payload = {
        "weights": {action: list(weights[action]) for action in ACTION_NAMES},
        "feature_dim": model.feature_dim,
        "num_buckets": model.num_buckets,
        "quantization": config.quantization_fingerprint,
    }
    return QuantizedModel(weights=weights, fingerprint=sha256_json(payload))

