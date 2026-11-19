"""Declarative configuration for the whole workbench.

All measurement convention parameters, tolerances and UI budget values live
here so that a report can be independently recomputed from the exact same
constants. Nothing in this module reads wall-clock time or environment state.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

# ---- Audio format -------------------------------------------------------
SAMPLE_RATE = 48_000
BIT_DEPTH = 24
CHANNEL_COUNT = 6
# ITU-R BS.1770 channel order L, R, C, LFE, Ls, Rs.
CHANNEL_LABELS = ("L", "R", "C", "LFE", "Ls", "Rs")
# Gains applied after K-weighting (LFE carries no contribution, weight 0).
CHANNEL_WEIGHTS_DB = (0.0, 0.0, 0.0, -float("inf"), 1.5, 1.5)

# ---- Block bounds (samples and milliseconds) ----------------------------
MIN_BLOCK_MS = 10
MAX_BLOCK_MS = 200
DEFAULT_BLOCK_MS = 100
# Internal control hop of the compressor and the measurement grid. Keeping the
# grid independent of the audio block length is what makes the offline and
# streaming paths converge for any block size.
CONTROL_HOP_MS = 10

# ---- ITU-R BS.1770 measurement convention -------------------------------
INTEGRATION_WINDOW_MS = 400
WINDOW_HOP_MS = 100
ABSOLUTE_GATE_LKFS = -70.0
RELATIVE_GATE_OFFSET_LU = -10.0
LRA_LOW_PERCENTILE = 10.0
LRA_HIGH_PERCENTILE = 95.0
LRA_ABSOLUTE_THRESHOLD_LKFS = -70.0
TRUE_PEAK_UPSAMPLE = 4

# ---- Declared consistency tolerances ------------------------------------
# Streaming vs offline integrated loudness tolerance (LU / dB).
LOUDNESS_TOLERANCE_LU = 0.1
# Streaming vs offline true-peak tolerance (dBTP).
TRUE_PEAK_TOLERANCE_DB = 0.05
# Streaming vs offline compressor envelope max absolute difference (dB).
ENVELOPE_TOLERANCE_DB = 1e-9
# Maximum gain jump allowed across any block boundary (dB).
MAX_BOUNDARY_JUMP_DB = 1e-6

# ---- Interaction budget -------------------------------------------------
# A 100 ms block period is the nominal UI scheduling period.
UI_BLOCK_PERIOD_MS = 100
# The meter/table refresh must finish within this budget per block.
UI_REFRESH_BUDGET_MS = 40.0
# Consecutive budget overruns that promote the condition to E_BUDGET_EXCEEDED.
BUDGET_OVERRUN_LIMIT = 3

# ---- Deterministic material ---------------------------------------------
DEFAULT_SEED = 20240517


@dataclass(frozen=True)
class CompressorConfig:
    """Log-domain dynamic range compressor parameters."""

    threshold_dbfs: float = -20.0
    ratio: float = 3.0
    attack_ms: float = 10.0
    release_ms: float = 120.0
    knee_db: float = 2.0
    makeup_gain_db: float = 0.0

    def __post_init__(self) -> None:
        if self.ratio < 1.0:
            raise ValueError("compressor ratio must be >= 1.0")
        if self.attack_ms <= 0.0 or self.release_ms <= 0.0:
            raise ValueError("attack/release time constants must be positive")


@dataclass(frozen=True)
class WorkbenchConfig:
    sample_rate: int = SAMPLE_RATE
    bit_depth: int = BIT_DEPTH
    channel_count: int = CHANNEL_COUNT
    seed: int = DEFAULT_SEED
    block_ms: int = DEFAULT_BLOCK_MS
    integration_window_ms: int = INTEGRATION_WINDOW_MS
    window_hop_ms: int = WINDOW_HOP_MS
    absolute_gate_lkfs: float = ABSOLUTE_GATE_LKFS
    relative_gate_offset_lu: float = RELATIVE_GATE_OFFSET_LU
    true_peak_upsample: int = TRUE_PEAK_UPSAMPLE
    ui_refresh_budget_ms: float = UI_REFRESH_BUDGET_MS
    loudness_tolerance_lu: float = LOUDNESS_TOLERANCE_LU
    true_peak_tolerance_db: float = TRUE_PEAK_TOLERANCE_DB
    envelope_tolerance_db: float = ENVELOPE_TOLERANCE_DB
    compressor: CompressorConfig = field(default_factory=CompressorConfig)

    def to_dict(self) -> dict:
        data = asdict(self)
        return data

    @property
    def block_samples(self) -> int:
        return ms_to_samples(self.block_ms, self.sample_rate)


def ms_to_samples(ms: float, sample_rate: int = SAMPLE_RATE) -> int:
    return int(round(ms * sample_rate / 1000.0))


def samples_to_seconds(n: int, sample_rate: int = SAMPLE_RATE) -> float:
    return n / sample_rate


def validate_block_ms(block_ms: int) -> None:
    """Raise E_BLOCK_BOUNDS for block lengths outside the declared range."""
    from .errors import WorkbenchError, E_BLOCK_BOUNDS

    if not isinstance(block_ms, (int, float)) or isinstance(block_ms, bool):
        raise WorkbenchError(E_BLOCK_BOUNDS, f"block_ms must be numeric, got {block_ms!r}")
    if block_ms < MIN_BLOCK_MS or block_ms > MAX_BLOCK_MS:
        raise WorkbenchError(
            E_BLOCK_BOUNDS,
            f"block_ms={block_ms} outside [{MIN_BLOCK_MS}, {MAX_BLOCK_MS}] ms",
        )
    if block_ms % CONTROL_HOP_MS != 0:
        raise WorkbenchError(
            E_BLOCK_BOUNDS,
            f"block_ms={block_ms} must be a multiple of {CONTROL_HOP_MS} ms",
        )


def validate_audio_format(sample_rate: int, channels: int, bit_depth: int = BIT_DEPTH) -> None:
    """Raise E_AUDIO_FORMAT when a stream does not match the declared format."""
    from .errors import WorkbenchError, E_AUDIO_FORMAT

    if sample_rate != SAMPLE_RATE:
        raise WorkbenchError(
            E_AUDIO_FORMAT, f"sample_rate={sample_rate}, expected {SAMPLE_RATE}"
        )
    if channels != CHANNEL_COUNT:
        raise WorkbenchError(
            E_AUDIO_FORMAT, f"channels={channels}, expected {CHANNEL_COUNT}"
        )
    if bit_depth != BIT_DEPTH:
        raise WorkbenchError(
            E_AUDIO_FORMAT, f"bit_depth={bit_depth}, expected {BIT_DEPTH}"
        )
