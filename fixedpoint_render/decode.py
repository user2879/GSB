"""Frame decoding and tile-row PNG output (business rule 5).

Pixels are produced by a deterministic integer linear projection from the final
latent vector.  Each tile row only reads pixel rows it owns; the explicit
:class:`TileBoundaryGuard` raises ``E_TILE_BOUNDARY`` with frame and tile row
locators on any cross-row read attempt.
"""
from __future__ import annotations

import io
from dataclasses import dataclass, field

from PIL import Image

from .config import RenderConfig
from .errors import E_TILE_BOUNDARY, RenderError
from .fingerprints import fingerprint
from .fixedpoint import q_div_round, q_from_int, q_scale, q_to_uint8_clamp
from .latent import FrameLatent
from .rng import DeterministicRandom


class TileBoundaryGuard:
    """Tracks the pixel-row range owned by each tile row."""

    def __init__(self, frame: int, width: int, height: int, tile_rows: int):
        self.frame = frame
        self.width = width
        self.height = height
        self.tile_h = height // tile_rows
        self.ranges = {
            tile: (tile * self.tile_h,
                   (tile + 1) * self.tile_h if tile < tile_rows - 1 else height)
            for tile in range(tile_rows)
        }

    def check_read(self, tile: int, row: int) -> None:
        lo, hi = self.ranges[tile]
        if not (lo <= row < hi):
            raise RenderError(
                E_TILE_BOUNDARY,
                f"frame_{self.frame:02d}",
                f"tile_row={tile}",
                f"illegal read of pixel row {row}; owned range is [{lo},{hi})",
            )


def projection_matrix(config: RenderConfig) -> list[list[int]]:
    """Deterministic integer projection weights (in Q), per pixel channel.

    Weights are small signed Q values (around +-1/8 of a unit) so the sum over
    ``latent_dim`` slots lands in a resolvable mid range instead of saturating
    the 0..255 clamp.
    """
    rng = DeterministicRandom(
        "projection|" + fingerprint(config.canonical_json(), "proj-config"))
    rows = config.channels
    small_scale = q_scale(config.frac_bits) // 8
    matrix = [[rng.next_signed(small_scale)
               for _ in range(config.latent_dim)] for _ in range(rows)]
    return matrix


@dataclass
class TileRowResult:
    frame: int
    tile: int
    row_start: int
    row_end: int
    png_bytes: bytes
    pixels: list[list[list[int]]]   # rows x width x channels (uint8)
    rounding_traces: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "frame": self.frame, "tile": self.tile,
            "row_start": self.row_start, "row_end": self.row_end,
            "png_bytes_len": len(self.png_bytes),
            "pixel_fp": fingerprint(self.pixels, f"tile-pixels-f{self.frame}-t{self.tile}"),
            "rounding_traces": self.rounding_traces,
        }


def _project_pixel(frame: int, row: int, col: int, latent: list[int],
                   matrix: list[list[int]], config: RenderConfig,
                   guard: TileBoundaryGuard,
                   traces: list[dict]) -> list[int]:
    scale = q_scale(config.frac_bits)
    channels = []
    for channel in range(config.channels):
        acc = 0
        for slot in range(config.latent_dim):
            weight = matrix[channel][slot]
            product = q_div_round(weight * latent[slot], scale, config.qmode)
            acc += product
        # Add a deterministic row/column offset so adjacent pixels differ.
        bias = q_from_int(128, config.frac_bits)
        offset = q_from_int((row * 7 + col * 13 + channel * 17) % 31,
                            config.frac_bits)
        raw = acc + bias + offset
        pixel, trace = q_to_uint8_clamp(raw, config.frac_bits, config.qmode)
        channels.append(pixel)
        traces.append({"frame": frame, "row": row, "col": col,
                       "channel": channel, **trace.to_dict()})
    return channels


def decode_tile_row(frame: int, tile: int, latent: FrameLatent,
                    matrix: list[list[int]], config: RenderConfig,
                    guard: TileBoundaryGuard | None = None) -> TileRowResult:
    if guard is None:
        guard = TileBoundaryGuard(frame, config.width, config.height,
                                  config.tile_rows)
    row_start, row_end = guard.ranges[tile]
    mode = config.pixel_mode
    rows: list[list[list[int]]] = []
    traces: list[dict] = []
    for row in range(row_start, row_end):
        guard.check_read(tile, row)  # boundary enforcement
        pixel_row = []
        for col in range(config.width):
            pixel_row.append(_project_pixel(frame, row, col, latent.final,
                                            matrix, config, guard, traces))
        rows.append(pixel_row)
    flat = bytearray()
    for pixel_row in rows:
        for pixel in pixel_row:
            flat.extend(pixel)
    image = Image.frombytes(mode, (config.width, row_end - row_start),
                            bytes(flat))
    buffer = io.BytesIO()
    # Fixed compression settings keep PNG bytes deterministic.
    image.save(buffer, format="PNG", compress_level=6, optimize=False)
    return TileRowResult(frame=frame, tile=tile, row_start=row_start,
                         row_end=row_end, png_bytes=buffer.getvalue(),
                         pixels=rows, rounding_traces=traces)


def assemble_frame(tiles: list[TileRowResult],
                   config: RenderConfig) -> bytes:
    """Stitch tile rows in ascending order into one full-frame PNG."""
    ordered = sorted(tiles, key=lambda t: t.tile)
    full = Image.new(config.pixel_mode, (config.width, config.height))
    for tile in ordered:
        tile_img = Image.open(io.BytesIO(tile.png_bytes))
        full.paste(tile_img, (0, tile.row_start))
    buffer = io.BytesIO()
    full.save(buffer, format="PNG", compress_level=6, optimize=False)
    return buffer.getvalue()
