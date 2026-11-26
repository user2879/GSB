"""Deterministic condition encoders (business rule 2).

Each encoder returns an :class:`EncodedCondition`: a fixed point vector of
length ``config.condition_dim``, an integer coverage percentage (0..100) and a
list of auditable quantization traces.  Everything is computed with integers.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image

from .config import RenderConfig
from .fixedpoint import (
    QuantTrace, q_div_round, q_from_int, q_scale, quantize,
)


@dataclass
class EncodedCondition:
    condition: str
    frame: int
    vector: list[int]                 # Q fixed point ints
    coverage: int                     # 0..100
    traces: list[QuantTrace] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "condition": self.condition,
            "frame": self.frame,
            "vector": self.vector,
            "coverage": self.coverage,
            "traces": [t.to_dict() for t in self.traces],
        }


def _normalize_counts(counts: list[int], total: int, frac_bits: int,
                      qmode: str, op: str) -> tuple[list[int], list[QuantTrace]]:
    """counts / total mapped to Q with explicit rounding and residuals."""
    scale = q_scale(frac_bits)
    out, traces = [], []
    for dim, count in enumerate(counts):
        raw = count * scale
        value = q_div_round(raw, total, qmode) if total else 0
        residual = raw - value * total
        traces.append(QuantTrace(f"{op}[{dim}]", 0, frac_bits, qmode,
                                 total, raw, value, residual))
        out.append(value)
    return out, traces


def encode_text(frame: int, terms: list[str],
                config: RenderConfig) -> EncodedCondition:
    counts = [0] * config.condition_dim
    # Controlled vocab order defines the bag-of-words layout.  When the vocab is
    # shorter than the condition dim the trailing slots stay zero.
    for slot, vocab_term in enumerate(config.vocab[:config.condition_dim]):
        counts[slot] = sum(1 for term in terms if term == vocab_term)
    total = sum(counts)
    vector, traces = _normalize_counts(counts, total if total else 1,
                                       config.frac_bits, config.qmode,
                                       "text_bow")
    coverage = q_div_round(sum(1 for c in counts if c) * 100,
                           config.condition_dim, "trunc")
    return EncodedCondition("text", frame, vector, coverage, traces)


def encode_image(frame: int, image: Image.Image,
                 config: RenderConfig) -> EncodedCondition:
    width, height = config.width, config.height
    tile_h = height // config.tile_rows
    gray = image.convert("L") if image.mode != "L" else image
    pixels = list(gray.getdata())
    dim = config.condition_dim
    half = dim // 2
    means = [0] * half
    grads = [0] * half
    # Block (tile row) color means first, then gradient statistics.  Tile rows
    # and condition slots are decoupled: every tile row maps deterministically
    # onto a slot so the encoder works for any tile_rows/dim combination.
    for tile in range(config.tile_rows):
        slot = tile % half
        row0 = tile * tile_h
        row1 = min(height, row0 + tile_h)
        block = [pixels[r * width + c] for r in range(row0, row1)
                 for c in range(width)]
        total = sum(block)
        tile_mean = q_div_round(total, len(block), "trunc")  # 0..255 integer
        means[slot] = tile_mean
        # Gradient: mean absolute horizontal difference inside the block.
        diff_sum = 0
        diff_count = 0
        for r in range(row0, row1):
            base = r * width
            for c in range(width - 1):
                diff_sum += abs(pixels[base + c + 1] - pixels[base + c])
                diff_count += 1
        grads[slot] = q_div_round(diff_sum, diff_count if diff_count else 1,
                                  "trunc")
    raw_features = means + grads
    # Scale 0..255 integer features into the Q domain, recording quantization.
    vector: list[int] = []
    traces: list[QuantTrace] = []
    for dim_index, feature in enumerate(raw_features):
        qv = q_from_int(feature, config.frac_bits)
        traces.append(QuantTrace(f"image_scale[{dim_index}]", 0,
                                 config.frac_bits, config.qmode, 1,
                                 feature, qv, 0))
        vector.append(qv)
    nonzero = sum(1 for v in vector if v != 0)
    coverage = q_div_round(nonzero * 100, dim, "trunc")
    return EncodedCondition("image", frame, vector, coverage, traces)


def encode_motion(frame: int, motion: dict,
                  config: RenderConfig) -> EncodedCondition:
    dim = config.condition_dim
    tiles = motion["tiles"]
    raw = [0] * dim
    # Explicit per-tile vectors win; otherwise distribute (vx, vy) across the
    # dimension deterministically by alternating slots per tile row.
    for row, tile in enumerate(tiles):
        vector = tile.get("vector")
        if vector is not None:
            for d in range(dim):
                raw[d] += int(vector[d])
        else:
            raw[(2 * row) % dim] += int(tile["vx"])
            raw[(2 * row + 1) % dim] += int(tile["vy"])
    scale = q_scale(config.frac_bits)
    q_vector: list[int] = []
    traces: list[QuantTrace] = []
    for d, value in enumerate(raw):
        qv, trace = quantize(q_from_int(value, config.frac_bits),
                             config.frac_bits, config.frac_bits,
                             config.qmode, f"motion_load[{d}]")
        q_vector.append(qv)
        traces.append(trace)
    nonzero_tiles = sum(1 for t in tiles if int(t["vx"]) != 0 or int(t["vy"]) != 0)
    coverage = int(motion.get("frame_cover",
                              q_div_round(nonzero_tiles * 100, len(tiles), "trunc")))
    coverage = max(0, min(100, coverage))
    return EncodedCondition("motion", frame, q_vector, coverage, traces)
