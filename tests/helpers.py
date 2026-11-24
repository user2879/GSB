from __future__ import annotations

from pathlib import Path

import numpy as np

from matgen.models import LayerArtifact


ROOT = Path(__file__).resolve().parents[1]
BASIC = ROOT / "samples" / "basic"
INJECTED = ROOT / "samples" / "injected"


def layer_artifact(
    index: int,
    values: np.ndarray,
    *,
    key: str | None = None,
    scale: int = 256,
) -> LayerArtifact:
    quantized = np.asarray(values, dtype=np.int64)
    return LayerArtifact(
        index=index,
        key=key or f"layer-{index}",
        texture_key="texture",
        upstream_layer_key="ROOT" if index == 0 else f"layer-{index - 1}",
        frequency=0.1,
        amplitude=0.1,
        unquantized=quantized.copy(),
        quantized=quantized.copy(),
        residual=np.zeros_like(quantized),
        scale=scale,
        quantization_step=2,
    )
