"""Quantized per-channel style lookup and retention decisions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fixed import blend_fixed, to_u8_index
from .util import array_fingerprint


@dataclass
class StylePatchDecision:
    image: np.ndarray
    mapped_pixels: int
    retained_mask_repair_pixels: int
    retained_source_pixels: int
    decision_fingerprint: str
    lut_fingerprint: str


class StyleMapper:
    def __init__(self, style_rgb: np.ndarray, fraction_bits: int, style_fingerprint: str):
        self.style_rgb = np.asarray(style_rgb, dtype=np.uint8)
        self.fraction_bits = int(fraction_bits)
        self.scale = 1 << self.fraction_bits
        self.style_fingerprint = style_fingerprint
        self._sorted = np.sort(self.style_rgb.reshape(-1, 3), axis=0)
        self._lut_cache: dict[int, tuple[np.ndarray, str]] = {}

    def lookup_table(self, quant_levels: int) -> tuple[np.ndarray, str]:
        quant_levels = int(quant_levels)
        cached = self._lut_cache.get(quant_levels)
        if cached is not None:
            return cached
        sample_count = self._sorted.shape[0]
        lut = np.empty((256, 3), dtype=np.int64)
        for value in range(256):
            quant_bin = min(quant_levels - 1, (value * quant_levels) // 256)
            source_index = min(
                sample_count - 1,
                ((2 * quant_bin + 1) * sample_count) // (2 * quant_levels),
            )
            lut[value] = self._sorted[source_index].astype(np.int64) << self.fraction_bits
        fingerprint = array_fingerprint(lut)
        result = (lut, fingerprint)
        self._lut_cache[quant_levels] = result
        return result

    def prebuild(self, quant_levels_by_layer: list[int]) -> dict[int, tuple[np.ndarray, str]]:
        return {level: self.lookup_table(levels) for level, levels in enumerate(quant_levels_by_layer)}

    def map_patch(
        self,
        repaired: np.ndarray,
        mask: np.ndarray,
        lut: np.ndarray,
        lut_fingerprint: str,
        weights_q: tuple[int, int, int],
        style_inside: str,
        style_outside: str,
        retained_source_mask: np.ndarray,
    ) -> StylePatchDecision:
        repaired = np.asarray(repaired, dtype=np.int64)
        mask = np.asarray(mask, dtype=bool)
        indices = to_u8_index(repaired, self.fraction_bits)
        mapped = lut[indices, np.arange(3, dtype=np.int64)[None, None, :]]
        styled = np.empty_like(repaired)
        for channel, weight in enumerate(weights_q):
            styled[..., channel] = blend_fixed(
                repaired[..., channel], mapped[..., channel], int(weight), self.scale
            )

        apply_inside = style_inside == "mapped"
        apply_outside = style_outside == "mapped"
        apply_style = np.where(mask, apply_inside, apply_outside)
        image = np.where(apply_style[..., None], styled, repaired)

        retained_source = retained_source_mask & ~apply_style
        retained_mask_repair = mask & ~apply_style
        decision = np.zeros(mask.shape, dtype=np.uint8)
        decision[~apply_style & ~retained_source] = 1
        decision[apply_style] = 2
        decision[retained_source] = 3
        decision_fingerprint = array_fingerprint(decision)
        return StylePatchDecision(
            image=image,
            mapped_pixels=int(apply_style.sum()),
            retained_mask_repair_pixels=int(retained_mask_repair.sum()),
            retained_source_pixels=int(retained_source.sum()),
            decision_fingerprint=decision_fingerprint,
            lut_fingerprint=lut_fingerprint,
        )

