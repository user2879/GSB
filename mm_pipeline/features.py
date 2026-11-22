"""Deterministic visual and text feature extraction (pure fixed-point).

Visual path: each grid region is reduced to six exact integer channel stats,
then projected by a fixed integer matrix whose entries derive solely from the
config fingerprint and the feature dimensions. Text path: every assertion maps
to an integer vector via a config-seeded RNG plus a content hash; regions with
no evidence are backfilled with a deterministic default vector and the backfill
is explicitly recorded.
"""

from __future__ import annotations

import io
import numpy as np
from PIL import Image

from .config import PipelineConfig, STAGE_FEATURE
from .errors import PipelineError, E_IMAGE_FORMAT
from .fixedpoint import scale, quant_shift
from .hashing import hash_json, cache_key, combine
from .models import (
    Sample,
    RegionFeature,
    Relation,
)


# ---------------------------------------------------------------------------
# Deterministic, config-fingerprint derived helpers.
# ---------------------------------------------------------------------------
def _rng(config: PipelineConfig, namespace: str) -> np.random.Generator:
    return np.random.default_rng(config.rng_seed(namespace))


def projection_matrix(config: PipelineConfig) -> np.ndarray:
    """Fixed (stats=6 x visual_dim) integer projection matrix."""
    rng = _rng(config, "visual-projection")
    # Values in [-8, 7], fully determined by the config fingerprint.
    return rng.integers(
        -8, 8, size=(6, config.visual_dim), dtype=np.int64, endpoint=True
    )


def default_text_vector(config: PipelineConfig) -> list[int]:
    """Deterministic backfill vector derived from the config fingerprint."""
    rng = _rng(config, "text-default")
    s = scale(config.frac_bits)
    raw = rng.integers(0, 1 << 30, size=config.text_dim, dtype=np.int64)
    # Keep the default modest in magnitude: map into [0, S/4).
    return [int(x % (s // 4)) for x in raw]


def assertion_vector(
    assertion, config: PipelineConfig, rng: np.random.Generator
) -> list[int]:
    """Map one assertion to a fixed-point integer text vector."""
    s = scale(config.frac_bits)
    content_digest = hash_json(
        {
            "assertion_id": assertion.assertion_id,
            "entity": assertion.entity,
            "attribute": assertion.attribute,
            # Canonicalize the numeric value through its exact repr.
            "value": repr(assertion.value),
        },
        prefix="assertion-vector",
    )
    content_seed = int(content_digest[:16], 16)
    content_rng = np.random.default_rng(content_seed)
    base = rng.integers(-(s // 8), s // 8, size=config.text_dim, dtype=np.int64)
    modulation = content_rng.integers(
        -(s // 16), s // 16, size=config.text_dim, dtype=np.int64
    )
    numeric = int((assertion.value - int(assertion.value)) * s)
    vector = base + modulation + numeric
    return [int(x) for x in vector]


# ---------------------------------------------------------------------------
# Visual region statistics.
# ---------------------------------------------------------------------------
def _image_array(sample: Sample) -> np.ndarray:
    try:
        with Image.open(io.BytesIO(sample.image_bytes)) as img:
            arr = np.asarray(img.convert("RGB"), dtype=np.int64)
    except OSError as exc:
        raise PipelineError(
            E_IMAGE_FORMAT,
            f"cannot decode image pixels: {exc}",
            stage=STAGE_FEATURE,
            sample=sample.sample_id,
        ) from exc
    return arr


def region_stats(sample: Sample, region: int, config: PipelineConfig) -> list[int]:
    """Six exact integer stats for one grid region (R/G/B mean + R/G/B max)."""
    arr = _image_array(sample)
    height, width, _ = arr.shape
    col = region % config.grid_cols
    row = region // config.grid_cols
    x0 = col * width // config.grid_cols
    x1 = (col + 1) * width // config.grid_cols
    y0 = row * height // config.grid_rows
    y1 = (row + 1) * height // config.grid_rows
    block = arr[y0:y1, x0:x1, :]
    if block.size == 0:
        return [0] * 6
    shift = quant_shift(config.quant_level)
    means = block.mean(axis=(0, 1)).astype(np.int64)  # floor means
    maxes = block.max(axis=(0, 1)).astype(np.int64)
    stats = np.concatenate([means, maxes])
    stats = np.right_shift(stats, shift)
    return [int(x) for x in stats]


def visual_vector(
    sample: Sample, region: int, config: PipelineConfig, proj: np.ndarray
) -> list[int]:
    stats = np.array(region_stats(sample, region, config), dtype=np.int64)
    projected = stats @ proj  # integer matrix product
    s = scale(config.frac_bits)
    clipped = np.clip(projected, 0, s)
    return [int(x) for x in clipped]


# ---------------------------------------------------------------------------
# Per-region feature assembly.
# ---------------------------------------------------------------------------
def feature_cache_key(
    sample: Sample, region: int, config: PipelineConfig
) -> str:
    return cache_key(
        "feature",
        combine(
            sample.sample_id,
            sample.image_hash,
            sample.assertion_hash,
            sample.relation_hash,
            region,
            config.fingerprint(),
            config.quant_level,
            config.frac_bits,
        ),
    )


def _relations_for_region(sample: Sample, region: int) -> list[Relation]:
    return sorted(
        (r for r in sample.relations if r.region == region),
        key=lambda r: r.relation_id,
    )


def extract_region_feature(
    sample: Sample, region: int, config: PipelineConfig
) -> RegionFeature:
    proj = projection_matrix(config)
    vis = visual_vector(sample, region, config, proj)

    rng = _rng(config, "text-base")
    s = scale(config.frac_bits)
    assertion_by_id = {a.assertion_id: a for a in sample.assertions}
    region_relations = _relations_for_region(sample, region)

    text_acc = [0] * config.text_dim
    referenced_assertions: list[str] = []
    backfilled: list[str] = []

    if not region_relations:
        default_vec = default_text_vector(config)
        text_acc = list(default_vec)
        backfilled.append(f"region:{region}")
    else:
        for relation in region_relations:
            assertion = assertion_by_id[relation.assertion_id]
            vec = assertion_vector(assertion, config, rng)
            # Per-relation raw weight only; conflict/supplement semantics are
            # applied explicitly in the fixed-point fusion stage.
            weight_fixed = int(round(relation.weight * s))
            scaled = [(v * weight_fixed) >> config.frac_bits for v in vec]
            text_acc = [acc + val for acc, val in zip(text_acc, scaled)]
            referenced_assertions.append(assertion.assertion_id)

    key = feature_cache_key(sample, region, config)
    return RegionFeature(
        sample_id=sample.sample_id,
        region=region,
        visual_vector=vis,
        text_vector=text_acc,
        backfilled_assertions=sorted(set(backfilled)),
        referenced_assertions=sorted(set(referenced_assertions)),
        referenced_regions=[region],
        cache_key=key,
    )
