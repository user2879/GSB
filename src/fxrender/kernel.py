"""Latent initialization, step-wise denoising and tile-row decoding kernels.

Every numeric step runs in fixed-point integers with an explicit operation
order. Decoding is partitioned by tile rows and a read-set guard makes any
attempt to read latent elements belonging to a neighbouring tile row fail
with E_TILE_BORROW instead of silently leaking pixels.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from .codes import BizCode
from .config import RenderConfig
from .encoding import FusedCondition
from .fixedpoint import FpSpec, quantize, fixed_mul, fixed_add, clamp_int
from .formats import NOISE_CODE_TO_SCHEME
from .rng import DeterministicRNG


class TileBorrowError(RuntimeError):
    def __init__(self, tile_row: int, requested: int, allowed: range) -> None:
        super().__init__(
            f"{BizCode.E_TILE_BORROW.value}: tile_row={tile_row} requested "
            f"latent[{requested}] allowed={allowed.start}..{allowed.stop - 1}")
        self.code = BizCode.E_TILE_BORROW
        self.tile_row = tile_row
        self.requested = requested


class TileAccessGuard:
    """Restricts latent reads to the owning tile row's element range."""

    def __init__(self, latent: list[int], tile_row: int, tile_rows: int,
                 latent_dim: int) -> None:
        self._latent = latent
        self.tile_row = tile_row
        per_row = max(1, latent_dim // tile_rows)
        self.allowed = range(tile_row * per_row,
                             min(latent_dim, (tile_row + 1) * per_row))
        self.reads: list[int] = []

    def read(self, index: int) -> int:
        if index not in self.allowed:
            raise TileBorrowError(self.tile_row, index, self.allowed)
        self.reads.append(index)
        return self._latent[index]


def build_noise_table(cfg: RenderConfig, frame_id: int,
                      fused_fp: str) -> list[list[float]]:
    scheme = NOISE_CODE_TO_SCHEME[cfg.noise_code]
    rng = DeterministicRNG(
        f"noise|v{cfg.exchange_version}|f{frame_id}|{fused_fp}|s{cfg.seed}")
    table: list[list[float]] = []
    for _step in range(cfg.denoise_steps):
        row = [rng.noise(scheme) for _ in range(cfg.latent_dim)]
        table.append(row)
    return table


def init_latent(cfg: RenderConfig, frame_id: int,
                fused: FusedCondition) -> list[int]:
    spec = cfg.fp_spec
    cfg_fp = hashlib.blake2b(
        repr(cfg.config_fingerprint_material()).encode(),
        digest_size=16).hexdigest()
    rng = DeterministicRNG(
        f"latent|{fused.trace.content_fp}|s{cfg.seed}|{cfg_fp}|f{frame_id}")
    latent = [0] * cfg.latent_dim
    for index in range(cfg.latent_dim):
        latent[index] = quantize(rng.next_gaussian() * 0.25, spec)
    # Fold the fused condition fingerprint in once, in dimension order.
    for index in range(cfg.latent_dim):
        seed_byte = int(fused.trace.content_fp[index % 32], 16)
        latent[index] = fixed_add(
            latent[index], quantize(seed_byte / 15.0 - 0.5, spec), spec)
    return latent


def guidance_schedule(cfg: RenderConfig, step: int) -> int:
    """Permille guidance that gently increases across denoising steps."""
    if cfg.denoise_steps <= 1:
        return cfg.guidance_permille
    delta = (1000 - cfg.guidance_permille) * step // (cfg.denoise_steps - 1)
    return cfg.guidance_permille + delta // 4


@dataclass
class StepResult:
    step: int
    latent: list[int]
    delta: list[int]
    guidance_permille: int
    noise_scale_fp: int
    condition_scale_fp: int
    cost: int
    merged: bool


def denoise_step(cfg: RenderConfig, frame_id: int, step: int,
                 latent: list[int], fused: FusedCondition,
                 noise_row: list[float], merged: bool = False) -> StepResult:
    spec = cfg.fp_spec
    guidance_permille = guidance_schedule(cfg, step)
    guidance_fp = quantize(guidance_permille / 1000.0, spec)
    noise_scale = quantize(0.50 / (step + 1), spec)
    condition_scale = quantize(0.35, spec)
    delta = [0] * cfg.latent_dim
    next_latent = list(latent)
    for index in range(cfg.latent_dim):
        # Fixed operation order: noise first, then guided condition.
        noise_fp = quantize(noise_row[index], spec)
        scaled_noise, noise_residual = fixed_mul(noise_fp, noise_scale, spec)
        condition_fp = (fused.vector[index]
                        if index < len(fused.vector) else 0)
        guided, guide_residual = fixed_mul(condition_fp, guidance_fp, spec)
        scaled_condition, cond_residual = fixed_mul(
            guided, condition_scale, spec)
        update = fixed_add(scaled_noise, scaled_condition, spec)
        delta[index] = update
        next_latent[index] = clamp_int(latent[index] + update, spec)
    cost = cfg.points_per_step * (1 if not merged else 1)
    return StepResult(step, next_latent, delta, guidance_permille,
                      noise_scale, condition_scale, cost, merged)


def _projection_row(cfg: RenderConfig, index: int, pixel_index: int) -> int:
    """Deterministic linear projection coefficient in fixed-point [-1, 1]."""
    spec = cfg.fp_spec
    rng = DeterministicRNG(f"proj|{index}|{pixel_index}|s{cfg.seed}")
    return quantize(rng.next_signed() * 0.5, spec)


def decode_tile_row(cfg: RenderConfig, frame_id: int, tile_row: int,
                    latent: list[int]) -> tuple[bytes, str, list[int]]:
    """Decode one tile row's PNG strip using only its own latent slice."""
    spec = cfg.fp_spec
    guard = TileAccessGuard(latent, tile_row, cfg.tile_rows, cfg.latent_dim)
    th = cfg.tile_height
    tw = cfg.tile_width
    strip = np.zeros((th, cfg.image_width, 3), dtype=np.uint8)
    for y in range(th):
        for x in range(cfg.image_width):
            for channel in range(3):
                pixel_index = y * cfg.image_width * 3 + x * 3 + channel
                accum = 0
                for latent_index in guard.allowed:
                    coeff = _projection_row(cfg, latent_index, pixel_index)
                    value = guard.read(latent_index)
                    product, _ = fixed_mul(value, coeff, spec)
                    accum = fixed_add(accum, product, spec)
                mapped = accum // (spec.scale) + 128
                if mapped < 0:
                    mapped = 0
                elif mapped > 255:
                    mapped = 255
                strip[y, x, channel] = np.uint8(mapped)
    from PIL import Image
    import io as _io
    buffer = _io.BytesIO()
    Image.fromarray(strip).save(buffer, format="PNG")
    png_bytes = buffer.getvalue()
    digest = hashlib.blake2b(png_bytes, digest_size=16).hexdigest()
    return png_bytes, digest, sorted(guard.reads)
