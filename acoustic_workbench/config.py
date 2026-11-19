"""Deterministic room, channel, and reconstruction defaults."""

from __future__ import annotations

import math
from dataclasses import dataclass


SAMPLE_RATE = 48_000
DURATION_SECONDS = 0.45
CHANNEL_COUNT = 4
DEFAULT_NOISE_SEED = 20261119
DEFAULT_ROOM = {"width": 6.0, "depth": 5.0, "height": 3.0}
DEFAULT_LISTENER = [3.0, 2.5, 1.5]
DEFAULT_SOURCES = {
    "s0": [1.2, 1.0, 1.2],
    "s1": [4.8, 3.8, 1.4],
}
DEFAULT_WALL_ALPHAS = {
    "xmin": 0.18,
    "xmax": 0.22,
    "ymin": 0.16,
    "ymax": 0.20,
    "floor": 0.24,
    "ceiling": 0.26,
}
WALL_IDS = tuple(DEFAULT_WALL_ALPHAS)

DEFAULT_SAMPLE_DENSITY = 128
DEFAULT_SH_ORDER = 3
MAX_SH_ORDER = 8
MIN_SAMPLE_DENSITY = 32
MAX_SAMPLE_DENSITY = 512

DEFAULT_GRID_WIDTH = 256
DEFAULT_GRID_HEIGHT = 128
DEFAULT_TILE_ROWS = 4
DEFAULT_TILE_COLS = 8
PROJECTION_BOUNDS = (-180.0, 180.0, -90.0, 90.0)


@dataclass(frozen=True)
class ChannelSpec:
    channel_id: int
    source_id: str
    azimuth_deg: float
    elevation_deg: float
    aperture_deg: float = 68.0
    support_deg: float = 72.0


CHANNEL_SPECS = (
    ChannelSpec(0, "s0", 0.0, 10.0),
    ChannelSpec(1, "s0", 90.0, 10.0),
    ChannelSpec(2, "s1", 180.0, -10.0),
    ChannelSpec(3, "s1", 270.0, -10.0),
)

SOURCE_TO_CHANNELS = {
    source_id: tuple(spec.channel_id for spec in CHANNEL_SPECS if spec.source_id == source_id)
    for source_id in DEFAULT_SOURCES
}

MAX_ORDER_BOUNDS_DECLARATION = "order <= min(8, floor(sqrt(sample_density)) - 1)"
SPHERICAL_COVERAGE_DECLARATION = "full sphere, deterministic Fibonacci sampling"


def sph_order_bound(sample_density: int) -> int:
    return min(MAX_SH_ORDER, math.isqrt(sample_density) - 1)


def channel_spec(channel_id: int) -> ChannelSpec:
    return CHANNEL_SPECS[channel_id]
