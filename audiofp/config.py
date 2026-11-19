"""Fingerprint and matching configuration with an explicit version contract."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, replace
from typing import Any


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class FingerprintConfig:
    """All values that define the fingerprint encoding space.

    ``dt_quantum`` changes the hash encoding but not the extracted peaks.  This
    distinction is what makes a narrow, auditable index migration possible:
    indexes whose extraction fields are identical may be re-hashed, while an
    STFT or peak-picking change requires full re-extraction.
    """

    version: str = "fp-v2"
    sample_rate: int = 11025
    n_fft: int = 2048
    hop_length: int = 256
    window: str = "hann"
    min_freq: float = 90.0
    max_freq: float = 5000.0
    band_count: int = 48
    peak_density: float = 34.0
    neighborhood_frames: int = 2
    neighborhood_bands: int = 2
    target_min_frames: int = 30
    target_max_frames: int = 140
    max_pairs_per_peak: int = 5
    dt_quantum: int = 4
    encoding_space: str = "logband-pair-dt-v2"
    match_anchor_tolerance_frames: int = 2
    format_hit_threshold: float = 0.55
    dedup_score_threshold: float = 0.42

    def __post_init__(self) -> None:
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.n_fft <= 0 or self.hop_length <= 0:
            raise ValueError("n_fft and hop_length must be positive")
        if not 0 <= self.min_freq < self.max_freq <= self.sample_rate / 2:
            raise ValueError("frequency bounds must lie in (0, Nyquist]")
        if self.band_count < 4:
            raise ValueError("band_count must be at least 4")
        if self.peak_density <= 0:
            raise ValueError("peak_density must be positive")
        if self.neighborhood_frames < 0 or self.neighborhood_bands < 0:
            raise ValueError("neighborhood sizes must be non-negative")
        if not 0 <= self.target_min_frames < self.target_max_frames:
            raise ValueError("target zone must be a non-empty increasing range")
        if self.max_pairs_per_peak <= 0:
            raise ValueError("max_pairs_per_peak must be positive")
        if self.dt_quantum <= 0:
            raise ValueError("dt_quantum must be positive")
        if not 0 < self.format_hit_threshold <= 1:
            raise ValueError("format_hit_threshold must be in (0, 1]")

    @property
    def frames_per_second(self) -> float:
        return self.sample_rate / self.hop_length

    @property
    def frame_duration(self) -> float:
        return self.hop_length / self.sample_rate

    @property
    def time_tolerance_seconds(self) -> float:
        return self.match_anchor_tolerance_frames * self.frame_duration

    def extraction_dict(self) -> dict[str, Any]:
        """Fields that affect the peak sequence and therefore hashes."""

        return {
            "version": self.version,
            "sample_rate": self.sample_rate,
            "n_fft": self.n_fft,
            "hop_length": self.hop_length,
            "window": self.window,
            "min_freq": self.min_freq,
            "max_freq": self.max_freq,
            "band_count": self.band_count,
            "peak_density": self.peak_density,
            "neighborhood_frames": self.neighborhood_frames,
            "neighborhood_bands": self.neighborhood_bands,
            "target_min_frames": self.target_min_frames,
            "target_max_frames": self.target_max_frames,
            "max_pairs_per_peak": self.max_pairs_per_peak,
            "dt_quantum": self.dt_quantum,
            "encoding_space": self.encoding_space,
        }

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "FingerprintConfig":
        allowed = {item.name for item in fields(cls)}
        return cls(**{key: item for key, item in value.items() if key in allowed})

    def digest(self) -> str:
        payload = _canonical_json(self.extraction_dict()).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def short_digest(self) -> str:
        return self.digest()[:16]

    def canonical_json(self) -> str:
        return _canonical_json(self.to_dict())

    def migration_compatible_from(self, old: "FingerprintConfig") -> bool:
        """Whether an old index can be re-hashed without re-reading audio."""

        keys = (
            "sample_rate",
            "n_fft",
            "hop_length",
            "window",
            "min_freq",
            "max_freq",
            "band_count",
            "peak_density",
            "neighborhood_frames",
            "neighborhood_bands",
            "target_min_frames",
            "target_max_frames",
            "max_pairs_per_peak",
        )
        return all(getattr(self, key) == getattr(old, key) for key in keys)

    def hash_payload(
        self,
        band1: int,
        band2: int,
        delta_frames: int,
    ) -> str:
        """Canonical, independently reproducible hash pre-image."""

        dt_bucket = int(round(delta_frames / self.dt_quantum))
        return f"{self.encoding_space}|{band1:02d}|{band2:02d}|{dt_bucket:04d}"

    def hash_for(self, band1: int, band2: int, delta_frames: int) -> str:
        payload = self.hash_payload(band1, band2, delta_frames).encode("ascii")
        return hashlib.blake2b(payload, digest_size=8).hexdigest()

    def band_edges(self) -> list[float]:
        ratio = self.max_freq / self.min_freq
        return [self.min_freq * (ratio ** (index / self.band_count)) for index in range(self.band_count + 1)]

    def band_index_for_frequency(self, frequency: float) -> int:
        if frequency <= self.min_freq:
            return 0
        if frequency >= self.max_freq:
            return self.band_count - 1
        position = math.log(frequency / self.min_freq) / math.log(self.max_freq / self.min_freq)
        return min(self.band_count - 1, max(0, int(position * self.band_count)))


DEFAULT_CONFIG = FingerprintConfig()

# Same extraction parameters as fp-v2, but a different delta-time quantization
# and encoding-space identifier.  This is the deliberately migratable case.
LEGACY_CONFIG = replace(
    DEFAULT_CONFIG,
    version="fp-v1",
    dt_quantum=8,
    encoding_space="logband-pair-dt-v1",
)

INDEX_SCHEMA_VERSION = 1


def config_from_json(payload: str) -> FingerprintConfig:
    return FingerprintConfig.from_dict(json.loads(payload))
