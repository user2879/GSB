from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

from .canonical import digest_object
from .errors import WorkbenchError


SAMPLE_RATE = 44_100


@dataclass(frozen=True)
class FingerprintConfig:
    window_samples: int = 2048
    hop_samples: int = 512
    peak_density: float = 24.0
    band_edges_hz: tuple[float, ...] = (
        80.0,
        250.0,
        800.0,
        2000.0,
        5000.0,
        10000.0,
    )
    neighborhood_freq_bins: int = 4
    neighborhood_time_frames: int = 2
    fanout: int = 6
    target_zone_frames: int = 32
    amplitude_floor_db: float = -72.0

    def __post_init__(self) -> None:
        if not 256 <= self.window_samples <= 16_384:
            raise WorkbenchError(
                "E_CONFIG_RANGE",
                "window_samples must be between 256 and 16384",
                {"window_samples": self.window_samples},
            )
        if not 32 <= self.hop_samples <= self.window_samples:
            raise WorkbenchError(
                "E_CONFIG_RANGE",
                "hop_samples must be between 32 and window_samples",
                {"hop_samples": self.hop_samples},
            )
        if not 2.0 <= self.peak_density <= 250.0:
            raise WorkbenchError(
                "E_CONFIG_RANGE",
                "peak_density must be between 2 and 250 peaks/second",
                {"peak_density": self.peak_density},
            )
        edges = tuple(float(edge) for edge in self.band_edges_hz)
        if len(edges) < 2 or len(edges) > 16:
            raise WorkbenchError(
                "E_CONFIG_RANGE",
                "band_edges_hz must contain 2..16 boundaries",
                {"band_edges_hz": edges},
            )
        if any(left >= right for left, right in zip(edges, edges[1:])):
            raise WorkbenchError(
                "E_CONFIG_RANGE",
                "band_edges_hz must be strictly increasing",
                {"band_edges_hz": edges},
            )
        if edges[0] < 20.0 or edges[-1] > SAMPLE_RATE / 2:
            raise WorkbenchError(
                "E_CONFIG_RANGE",
                "band_edges_hz must stay inside [20, Nyquist]",
                {"band_edges_hz": edges},
            )
        if not 1 <= self.neighborhood_freq_bins <= 32:
            raise WorkbenchError("E_CONFIG_RANGE", "invalid frequency neighborhood")
        if not 0 <= self.neighborhood_time_frames <= 16:
            raise WorkbenchError("E_CONFIG_RANGE", "invalid time neighborhood")
        if not 1 <= self.fanout <= 32:
            raise WorkbenchError("E_CONFIG_RANGE", "fanout must be 1..32")
        if not 1 <= self.target_zone_frames <= 256:
            raise WorkbenchError("E_CONFIG_RANGE", "target_zone_frames must be 1..256")
        if not -120.0 <= self.amplitude_floor_db <= -6.0:
            raise WorkbenchError("E_CONFIG_RANGE", "invalid amplitude_floor_db")
        object.__setattr__(self, "band_edges_hz", edges)

    @property
    def digest(self) -> str:
        return digest_object(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["band_edges_hz"] = list(self.band_edges_hz)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FingerprintConfig":
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise WorkbenchError(
                "E_CONFIG_RANGE",
                "unknown fingerprint configuration keys",
                {"unknown": unknown},
            )
        payload = dict(data)
        if "band_edges_hz" in payload:
            payload["band_edges_hz"] = tuple(payload["band_edges_hz"])
        return cls(**payload)


@dataclass(frozen=True)
class BudgetConfig:
    query_threads: int = 2
    partition_memory_mb: float = 8.0
    query_time_budget_ms: int = 80
    index_partition_memory_mb: float = 8.0
    max_candidates: int = 50

    def __post_init__(self) -> None:
        if not 1 <= self.query_threads <= 64:
            raise WorkbenchError(
                "E_BUDGET_DENIED",
                "query_threads must be between 1 and 64",
                {"query_threads": self.query_threads},
            )
        if self.partition_memory_mb < 0.05:
            raise WorkbenchError(
                "E_BUDGET_DENIED",
                "partition_memory_mb is below the hard minimum",
                {"partition_memory_mb": self.partition_memory_mb, "minimum": 0.05},
            )
        if self.query_time_budget_ms < 1:
            raise WorkbenchError(
                "E_BUDGET_DENIED",
                "query_time_budget_ms must be at least 1",
                {"query_time_budget_ms": self.query_time_budget_ms},
            )
        if self.index_partition_memory_mb < 0.01:
            raise WorkbenchError(
                "E_BUDGET_DENIED",
                "index_partition_memory_mb is below the hard minimum",
                {"index_partition_memory_mb": self.index_partition_memory_mb, "minimum": 0.01},
            )
        if not 1 <= self.max_candidates <= 10_000:
            raise WorkbenchError("E_BUDGET_DENIED", "max_candidates must be 1..10000")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BudgetConfig":
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise WorkbenchError("E_BUDGET_DENIED", "unknown budget keys", {"unknown": unknown})
        return cls(**data)
