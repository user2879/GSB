"""Weight package JSONL loading and structural validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from quantcalib.errors import (
    E_GROUP_GRANULARITY,
    E_LAYER_DUPLICATE,
    E_SHAPE_INVALID,
    E_VERSION_ORDER,
    E_WEIGHT_NONFINITE,
    Issue,
)
from quantcalib.models import WeightLayer


def load_weight_pack(path: str | Path) -> tuple[list[WeightLayer], list[Issue]]:
    issues: list[Issue] = []
    layers: list[WeightLayer] = []
    seen: set[str] = set()
    last_version: int | None = None

    with open(path, "r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            record = json.loads(raw)
            name = str(record["layer"])

            if name in seen:
                issues.append(
                    Issue(E_LAYER_DUPLICATE, f"duplicate layer name {name!r}", name,
                          {"line": line_no})
                )
                continue
            seen.add(name)

            shape = tuple(int(d) for d in record["shape"])
            if not shape or any(d <= 0 for d in shape):
                issues.append(
                    Issue(E_SHAPE_INVALID, "shape dimensions must be positive integers",
                          name, {"shape": list(shape)})
                )
                continue

            weights = np.asarray(record["weights"], dtype=np.float64).reshape(shape)
            if not np.all(np.isfinite(weights)):
                bad = int(np.count_nonzero(~np.isfinite(weights)))
                issues.append(
                    Issue(E_WEIGHT_NONFINITE, "weights contain non-finite elements",
                          name, {"nonfinite_count": bad})
                )
                continue

            version = int(record["content_version"])
            if last_version is not None and version < last_version:
                issues.append(
                    Issue(E_VERSION_ORDER,
                          "content_version must be monotonically non-decreasing",
                          name, {"previous": last_version, "current": version})
                )
            last_version = version

            granularity = int(record["group_granularity"])
            if granularity <= 0 or weights.size % granularity != 0:
                issues.append(
                    Issue(E_GROUP_GRANULARITY,
                          "group_granularity must divide the total element count",
                          name, {"granularity": granularity, "size": int(weights.size)})
                )
                continue

            layers.append(WeightLayer(name, shape, weights, version, granularity))

    return layers, issues


def write_weight_pack(path: str | Path, layers: list[WeightLayer]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for layer in layers:
            handle.write(json.dumps(layer.to_record(), sort_keys=True))
            handle.write("\n")
