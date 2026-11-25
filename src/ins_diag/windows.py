from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import EffectiveConfig
from .util import fingerprint, sha256_bytes
from .waveform import Dataset


@dataclass(frozen=True)
class WindowRef:
    window_index: int
    device_id: str
    channel_id: str
    start: int
    end: int
    sample_rate_hz: float
    samples: np.ndarray
    present: np.ndarray

    @property
    def window_id(self) -> str:
        return f"{self.device_id}:{self.channel_id}:{self.start:06d}"

    @property
    def sample_count(self) -> int:
        return self.end - self.start

    def content_fingerprint(self) -> str:
        values = np.ascontiguousarray(self.samples, dtype="<f8").tobytes(order="C")
        present = np.ascontiguousarray(self.present, dtype=np.uint8).tobytes(order="C")
        return sha256_bytes(
            b"|".join(
                [
                    self.window_id.encode("utf-8"),
                    str(self.end).encode("ascii"),
                    str(self.sample_rate_hz).encode("ascii"),
                    values,
                    present,
                ]
            )
        )

    def to_dict(self) -> dict:
        return {
            "window_index": self.window_index,
            "window_id": self.window_id,
            "device_id": self.device_id,
            "channel_id": self.channel_id,
            "start": self.start,
            "end": self.end,
            "sample_count": self.sample_count,
            "sample_rate_hz": self.sample_rate_hz,
            "content_fingerprint": self.content_fingerprint(),
        }


def build_windows(dataset: Dataset, config: EffectiveConfig) -> list[WindowRef]:
    windows: list[WindowRef] = []
    for channel in dataset.channels:
        if channel.samples.size < config.window_length:
            continue
        for start in range(0, channel.samples.size - config.window_length + 1, config.window_shift):
            end = start + config.window_length
            windows.append(
                WindowRef(
                    window_index=len(windows),
                    device_id=channel.device_id,
                    channel_id=channel.channel_id,
                    start=start,
                    end=end,
                    sample_rate_hz=channel.sample_rate_hz,
                    samples=channel.samples[start:end],
                    present=channel.present[start:end],
                )
            )
    return windows


def window_set_fingerprint(windows: list[WindowRef]) -> str:
    return fingerprint(
        [
            {
                "window_id": window.window_id,
                "content": window.content_fingerprint(),
            }
            for window in sorted(windows, key=lambda item: item.window_index)
        ]
    )

