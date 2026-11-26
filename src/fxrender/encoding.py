"""Fixed-point condition encoders and the ordered fusion kernel.

All accumulation uses explicit Python integer loops in documented order
(condition type name ascending, then dimension index ascending) so results
never depend on numpy reduction order or any global random state.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import RenderConfig
from .fixedpoint import FpSpec, quantize, fixed_mul, fixed_add, clamp_int
from .fingerprint import fingerprint


@dataclass
class EncodedCondition:
    name: str
    vector: list[int]
    hits: list[int] = field(default_factory=list)
    detail: dict = field(default_factory=dict)
    content_fp: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "vector": self.vector, "hits": self.hits,
                "detail": self.detail, "content_fp": self.content_fp}


def encode_text(terms: list[str], vocabulary: list[str],
                cfg: RenderConfig) -> EncodedCondition:
    spec = cfg.fp_spec
    if len(vocabulary) != cfg.text_dim:
        raise ValueError(f"E_DIM_MISMATCH: vocabulary {len(vocabulary)} != "
                         f"text_dim {cfg.text_dim}")
    counts = [0] * cfg.text_dim
    index_of = {term: i for i, term in enumerate(vocabulary)}
    hits: list[int] = []
    for term in terms:
        idx = index_of[term]
        counts[idx] += 1
        hits.append(idx)
    total = len(terms)
    vector: list[int] = []
    for dim in range(cfg.text_dim):
        normalized = counts[dim] / total if total else 0.0
        vector.append(quantize(normalized, spec))
    hits_sorted = sorted(set(hits))
    content = fingerprint(
        {"terms": terms, "counts": counts, "spec": spec.key_material()},
        domain="text-encode")
    return EncodedCondition("text", vector, hits_sorted,
                            {"counts": counts, "total": total}, content)


def encode_image(pixels: np.ndarray, cfg: RenderConfig) -> EncodedCondition:
    """Tile color means plus horizontal/vertical gradient magnitudes."""
    spec = cfg.fp_spec
    height, width, _channels = pixels.shape
    if (height, width) != (cfg.image_height, cfg.image_width):
        raise ValueError("E_DIM_MISMATCH: image size")
    th = cfg.tile_height
    tw = cfg.tile_width
    raw: list[float] = []
    tile_means: list[list[float]] = []
    # Explicit row-major tile order, no vectorized reduction default reliance.
    for tr in range(cfg.tile_rows):
        row_means = []
        for tc in range(cfg.tile_cols):
            red_sum = green_sum = blue_sum = 0
            count = 0
            for y in range(tr * th, (tr + 1) * th):
                for x in range(tc * tw, (tc + 1) * tw):
                    r, g, b = (int(pixels[y, x, 0]), int(pixels[y, x, 1]),
                               int(pixels[y, x, 2]))
                    red_sum += r
                    green_sum += g
                    blue_sum += b
                    count += 1
            mean_rgb = (red_sum / count, green_sum / count, blue_sum / count)
            row_means.append(sum(mean_rgb) / 3.0 / 255.0)
        tile_means.append(row_means)
    # Gradient statistics per tile row and column
    for tr in range(cfg.tile_rows):
        grad_sum = 0
        grad_count = 0
        for y in range(tr * th, min((tr + 1) * th, height - 1)):
            for x in range(width - 1):
                grad_sum += abs(int(pixels[y, x + 1, 0]) - int(pixels[y, x, 0]))
                grad_count += 1
        raw.append((grad_sum / grad_count / 255.0) if grad_count else 0.0)
    for tc in range(cfg.tile_cols):
        grad_sum = 0
        grad_count = 0
        for x in range(tc * tw, min((tc + 1) * tw, width - 1)):
            for y in range(height - 1):
                grad_sum += abs(int(pixels[y + 1, x, 2]) - int(pixels[y, x, 2]))
                grad_count += 1
        raw.append((grad_sum / grad_count / 255.0) if grad_count else 0.0)
    flat_means = [value for row in tile_means for value in row]
    raw = flat_means + raw
    if len(raw) < cfg.image_dim:
        raw.extend([0.0] * (cfg.image_dim - len(raw)))
    raw = raw[:cfg.image_dim]
    vector = [quantize(value, spec) for value in raw]
    content = fingerprint(
        {"pixels": pixels.tobytes(), "spec": spec.key_material()},
        domain="image-encode")
    return EncodedCondition("image", vector, [],
                            {"tile_means": tile_means}, content)


def encode_motion(motion: np.ndarray, cfg: RenderConfig) -> EncodedCondition:
    spec = cfg.fp_spec
    if motion.shape != (cfg.tile_rows, cfg.tile_cols, 2):
        raise ValueError("E_MOTION_SHAPE: motion table shape")
    flat: list[int] = []
    # Interleave dx/dy in explicit row-major cell order.
    for tr in range(cfg.tile_rows):
        for tc in range(cfg.tile_cols):
            flat.append(int(motion[tr, tc, 0]))
            flat.append(int(motion[tr, tc, 1]))
    max_abs = max((abs(v) for v in flat), default=0)
    denom = max_abs if max_abs > 0 else 1
    normalized = [v / denom for v in flat]
    if len(normalized) < cfg.motion_dim:
        normalized.extend([0.0] * (cfg.motion_dim - len(normalized)))
    normalized = normalized[:cfg.motion_dim]
    vector = [quantize(value, spec) for value in normalized]
    hits = [i for i, value in enumerate(flat) if value != 0]
    content = fingerprint(
        {"motion": flat, "spec": spec.key_material()},
        domain="motion-encode")
    return EncodedCondition("motion", vector, hits,
                            {"max_abs": max_abs}, content)


@dataclass
class FusionTrace:
    order: list[str]
    per_step: list[dict] = field(default_factory=list)
    residuals: list[dict] = field(default_factory=list)
    content_fp: str = ""


@dataclass
class FusedCondition:
    vector: list[int]
    trace: FusionTrace
    weights_fp: dict[str, int]
    param_fp: str


def fuse_conditions(conditions: list[EncodedCondition],
                    cfg: RenderConfig) -> FusedCondition:
    spec = cfg.fp_spec
    ordered = sorted(conditions, key=lambda item: item.name)  # name ascending
    names = [item.name for item in ordered]
    dim = max((len(item.vector) for item in ordered), default=0)
    dim = max(dim, cfg.latent_dim)
    weights_fp = {
        name: quantize(float(cfg.weights.get(name, 0.0)), spec)
        for name in names
    }
    accumulator = [0] * dim
    trace = FusionTrace(order=names)
    for cond in ordered:
        padded = list(cond.vector) + [0] * (dim - len(cond.vector))
        weight = weights_fp[cond.name]
        for dim_index in range(dim):  # dimension index ascending, explicit
            product, mul_residual = fixed_mul(padded[dim_index], weight, spec)
            before = accumulator[dim_index]
            accumulator[dim_index] = fixed_add(before, product, spec)
            trace.per_step.append({
                "condition": cond.name,
                "dim": dim_index,
                "weight_fp": weight,
                "operand_fp": padded[dim_index],
                "product_fp": product,
                "sum_fp": accumulator[dim_index],
                "quant_step": 1 << spec.frac_bits,
            })
            if mul_residual != 0:
                trace.residuals.append({
                    "condition": cond.name, "dim": dim_index,
                    "residual": mul_residual, "stage": "multiply"})
    content = fingerprint(
        {"order": names, "vector": accumulator,
         "weights": weights_fp, "spec": spec.key_material()},
        domain="fusion")
    trace.content_fp = content
    param = fingerprint(
        {"weights": cfg.weights, "spec": spec.key_material()},
        domain="fusion-params")
    return FusedCondition(accumulator, trace, weights_fp, param)
