"""JSONL sample loading and immutable per-task copies."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .errors import ConfigError
from .util import hash_object


def _hashable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return {"__nonfinite_float__": "nan"}
        return {"__nonfinite_float__": "inf" if value > 0 else "-inf"}
    if isinstance(value, dict):
        return {str(key): _hashable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_hashable(item) for item in value]
    if isinstance(value, tuple):
        return [_hashable(item) for item in value]
    return value


def _sample_id(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError("each sample requires a non-empty string sample_id", {"value": value})
    return value


def _position(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field} must be an integer position", {"value": value})
    return value


def _features(value: Any, field: str) -> list[float]:
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{field} must be a non-empty numeric list", {"value": value})
    result: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ConfigError(f"{field} contains a non-numeric value", {"value": item})
        result.append(float(item))
    return result


@dataclass
class Sample:
    sample_id: str
    query_position: int
    query_features: list[float]
    key_positions: list[int]
    key_features: list[list[float]]
    raw: dict[str, Any]
    fingerprint: str

    def copy(self) -> "Sample":
        return Sample(
            sample_id=self.sample_id,
            query_position=self.query_position,
            query_features=list(self.query_features),
            key_positions=list(self.key_positions),
            key_features=[list(row) for row in self.key_features],
            raw=copy.deepcopy(self.raw),
            fingerprint=self.fingerprint,
        )

    @property
    def query_array(self) -> np.ndarray:
        return np.asarray(self.query_features, dtype=np.float64)

    @property
    def key_array(self) -> np.ndarray:
        return np.asarray(self.key_features, dtype=np.float64)


def parse_sample(raw: Any, line_number: int | None = None) -> Sample:
    if not isinstance(raw, dict):
        raise ConfigError("sample line must be a JSON object", {"line_number": line_number})
    sample_id = _sample_id(raw.get("sample_id", raw.get("id")))
    query_position = _position(raw.get("query_position"), "query_position")
    query_features = _features(raw.get("query_features", raw.get("query_feature")), "query_features")

    if "key_positions" in raw or "key_features" in raw:
        if "key_positions" not in raw or "key_features" not in raw:
            raise ConfigError(
                "key_positions and key_features must be provided together",
                {"sample_id": sample_id, "line_number": line_number},
            )
        key_positions_raw = raw["key_positions"]
        key_features_raw = raw["key_features"]
        if not isinstance(key_positions_raw, list) or not isinstance(key_features_raw, list):
            raise ConfigError(
                "key_positions and key_features must be lists",
                {"sample_id": sample_id, "line_number": line_number},
            )
        key_positions = [
            _position(value, f"key_positions[{index}]")
            for index, value in enumerate(key_positions_raw)
        ]
        key_features = [
            _features(value, f"key_features[{index}]")
            for index, value in enumerate(key_features_raw)
        ]
    else:
        keys = raw.get("keys")
        if not isinstance(keys, list) or not keys:
            raise ConfigError(
                "sample requires key_positions/key_features or a non-empty keys array",
                {"sample_id": sample_id, "line_number": line_number},
            )
        key_positions = []
        key_features = []
        for index, item in enumerate(keys):
            if not isinstance(item, dict):
                raise ConfigError(
                    "keys entries must be objects",
                    {"sample_id": sample_id, "index": index},
                )
            key_positions.append(_position(item.get("position"), f"keys[{index}].position"))
            key_features.append(_features(item.get("features"), f"keys[{index}].features"))

    if len(key_positions) != len(key_features):
        raise ConfigError(
            "key_positions and key_features lengths differ",
            {
                "sample_id": sample_id,
                "key_position_count": len(key_positions),
                "key_feature_count": len(key_features),
            },
        )
    if not key_positions:
        raise ConfigError("sample must contain at least one key", {"sample_id": sample_id})
    key_widths = {len(row) for row in key_features}
    if len(key_widths) != 1:
        raise ConfigError(
            "all key feature rows must have the same dimension",
            {"sample_id": sample_id, "key_widths": sorted(key_widths)},
        )
    if len(query_features) != next(iter(key_widths)):
        raise ConfigError(
            "query and key feature dimensions differ",
            {
                "sample_id": sample_id,
                "query_dimension": len(query_features),
                "key_dimension": next(iter(key_widths)),
            },
        )
    return Sample(
        sample_id=sample_id,
        query_position=query_position,
        query_features=query_features,
        key_positions=key_positions,
        key_features=key_features,
        raw=copy.deepcopy(raw),
        fingerprint=hash_object({"sample_id": sample_id, "content": _hashable(raw)}),
    )


def load_samples(path: str | Path) -> tuple[list[Sample], str]:
    samples: list[Sample] = []
    seen: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConfigError(
                    "invalid JSONL sample",
                    {"line_number": line_number, "error": str(exc)},
                ) from exc
            sample = parse_sample(raw, line_number)
            if sample.sample_id in seen:
                raise ConfigError(
                    "duplicate sample_id",
                    {"sample_id": sample.sample_id, "line_number": line_number},
                )
            seen.add(sample.sample_id)
            samples.append(sample)
    if not samples:
        raise ConfigError("sample file contains no samples", {"path": str(path)})
    samples.sort(key=lambda item: item.sample_id)
    samples_hash = hash_object(
        [_hashable(sample.raw) for sample in sorted(samples, key=lambda item: item.sample_id)]
    )
    return samples, samples_hash
