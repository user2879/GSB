"""Acoustic analysis configuration, canonical hashing and audio fingerprints."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def canonical_json(obj) -> str:
    """Deterministic JSON text used for hashing and snapshot manifests."""
    return json.dumps(
        obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        allow_nan=False,
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint_json(obj) -> str:
    return sha256_text(canonical_json(obj))


def fingerprint_pcm(pcm: np.ndarray) -> str:
    return sha256_bytes(np.asarray(pcm, dtype="<i2").tobytes())


@dataclass(frozen=True)
class PipelineConfig:
    sample_rate: int
    frame_length_ms: float
    frame_shift_ms: float
    mel_bands: int
    mel_fmax: float
    cepstrum_order: int
    f0_min_hz: float
    f0_max_hz: float
    max_frames_per_segment: int
    frac_bits: int
    quant_levels: int
    workers: int
    snapshot_commit_interval: int
    crossfade_ms: float
    timbre_table: dict

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        required = [
            "sample_rate", "frame_length_ms", "frame_shift_ms", "mel_bands",
            "mel_fmax", "cepstrum_order", "f0_min_hz", "f0_max_hz",
            "max_frames_per_segment", "frac_bits", "quant_levels", "workers",
            "snapshot_commit_interval", "crossfade_ms", "timbre_table",
        ]
        missing = [k for k in required if k not in data]
        if missing:
            raise ValueError(f"config missing keys: {missing}")
        if data["workers"] < 1:
            raise ValueError("workers must be >= 1")
        if data["quant_levels"] < 1:
            raise ValueError("quant_levels must be >= 1")
        return cls(**{k: data[k] for k in required})

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    def to_dict(self) -> dict:
        return {
            "sample_rate": self.sample_rate,
            "frame_length_ms": self.frame_length_ms,
            "frame_shift_ms": self.frame_shift_ms,
            "mel_bands": self.mel_bands,
            "mel_fmax": self.mel_fmax,
            "cepstrum_order": self.cepstrum_order,
            "f0_min_hz": self.f0_min_hz,
            "f0_max_hz": self.f0_max_hz,
            "max_frames_per_segment": self.max_frames_per_segment,
            "frac_bits": self.frac_bits,
            "quant_levels": self.quant_levels,
            "workers": self.workers,
            "snapshot_commit_interval": self.snapshot_commit_interval,
            "crossfade_ms": self.crossfade_ms,
            "timbre_table": self.timbre_table,
        }

    @property
    def frame_length(self) -> int:
        return int(round(self.frame_length_ms * self.sample_rate / 1000.0))

    @property
    def frame_shift(self) -> int:
        return int(round(self.frame_shift_ms * self.sample_rate / 1000.0))

    @property
    def crossfade_frames(self) -> int:
        return max(0, int(round(self.crossfade_ms / self.frame_shift_ms)))

    def fingerprint(self) -> str:
        return fingerprint_json(self.to_dict())

    def structural_fingerprint(self) -> str:
        """Hash of structural analysis settings, excluding runtime knobs.

        ``quant_levels`` is an exchange/quantization parameter deliberately
        excluded: changing it must invalidate cache keys (which embed it) but
        not the snapshot resume gate, so unaffected stages still reuse.
        """
        payload = self.to_dict()
        payload.pop("quant_levels", None)
        return fingerprint_json(payload)

    def code_to_name(self, code: int) -> str | None:
        for name, entry in self.timbre_table.items():
            if int(entry["code"]) == int(code):
                return name
        return None

    def name_to_code(self, name: str) -> int | None:
        entry = self.timbre_table.get(name)
        return None if entry is None else int(entry["code"])

    def timbre_entry(self, name: str) -> dict | None:
        return self.timbre_table.get(name)
