"""Configuration and pair-manifest loading with scope-level fingerprints."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .util import fingerprint, read_json

REQUIRED_FIELDS = (
    "frame_length_ms",
    "frame_shift_ms",
    "mel_bands",
    "mel_fmax",
    "cepstrum_order",
    "embedding_dim",
    "quant_levels",
    "fixed_point_bits",
    "similarity_threshold",
    "tie_rule",
    "min_valid_frames",
    "silence_energy_db",
    "quantile_points",
    "quantile_interpolation",
    "feature_sig_figs",
    "max_workers",
    "snapshot_commit_interval",
    "tolerance",
)

FEATURE_FIELDS = (
    "frame_length_ms",
    "frame_shift_ms",
    "mel_bands",
    "mel_fmax",
    "cepstrum_order",
    "feature_sig_figs",
)
QUANT_FIELDS = ("quant_levels",)
EMBED_FIELDS = ("embedding_dim",)
STATS_FIELDS = (
    "silence_energy_db",
    "min_valid_frames",
    "quantile_points",
    "quantile_interpolation",
)
SIM_FIELDS = ("tie_rule", "similarity_threshold")
RUNTIME_FIELDS = ("max_workers", "snapshot_commit_interval")

_TIE_RULES = ("same_speaker", "different_speaker")
_INTERP_RULES = ("linear", "lower", "higher", "nearest", "midpoint")


@dataclass(frozen=True)
class PipelineConfig:
    raw: dict[str, Any]
    config_hash: str
    scope_hashes: dict[str, str]

    @property
    def frame_length_ms(self) -> float:
        return float(self.raw["frame_length_ms"])

    @property
    def frame_shift_ms(self) -> float:
        return float(self.raw["frame_shift_ms"])

    @property
    def mel_bands(self) -> int:
        return int(self.raw["mel_bands"])

    @property
    def mel_fmax(self) -> float:
        return float(self.raw["mel_fmax"])

    @property
    def cepstrum_order(self) -> int:
        return int(self.raw["cepstrum_order"])

    @property
    def embedding_dim(self) -> int:
        return int(self.raw["embedding_dim"])

    @property
    def quant_levels(self) -> int:
        return int(self.raw["quant_levels"])

    @property
    def fixed_point_bits(self) -> int:
        return int(self.raw["fixed_point_bits"])

    @property
    def similarity_threshold(self) -> float:
        return float(self.raw["similarity_threshold"])

    @property
    def tie_rule(self) -> str:
        return str(self.raw["tie_rule"])

    @property
    def min_valid_frames(self) -> int:
        return int(self.raw["min_valid_frames"])

    @property
    def silence_energy_db(self) -> float:
        return float(self.raw["silence_energy_db"])

    @property
    def quantile_points(self) -> list[float]:
        return [float(x) for x in self.raw["quantile_points"]]

    @property
    def quantile_interpolation(self) -> str:
        return str(self.raw["quantile_interpolation"])

    @property
    def feature_sig_figs(self) -> int:
        return int(self.raw["feature_sig_figs"])

    @property
    def max_workers(self) -> int:
        return int(self.raw["max_workers"])

    @property
    def snapshot_commit_interval(self) -> int:
        return int(self.raw["snapshot_commit_interval"])

    @property
    def tolerance(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.raw["tolerance"].items()}

    def scope_hash(self, scope: str) -> str:
        return self.scope_hashes[scope]


def load_config(path: str) -> PipelineConfig:
    raw = read_json(path)
    return build_config(raw)


def build_config(raw: dict[str, Any]) -> PipelineConfig:
    missing = [name for name in REQUIRED_FIELDS if name not in raw]
    if missing:
        raise ValueError(f"missing config fields: {missing}")
    if float(raw["frame_length_ms"]) <= 0 or float(raw["frame_shift_ms"]) <= 0:
        raise ValueError("frame lengths must be positive")
    if int(raw["quant_levels"]) < 2:
        raise ValueError("quant_levels must be >= 2")
    if int(raw["fixed_point_bits"]) < 4:
        raise ValueError("fixed_point_bits must be >= 4")
    if not 0.0 <= float(raw["similarity_threshold"]) <= 1.0:
        raise ValueError("similarity_threshold must lie within [0, 1]")
    if str(raw["tie_rule"]) not in _TIE_RULES:
        raise ValueError(f"tie_rule must be one of {_TIE_RULES}")
    if str(raw["quantile_interpolation"]) not in _INTERP_RULES:
        raise ValueError(f"quantile_interpolation must be one of {_INTERP_RULES}")
    if int(raw["max_workers"]) < 1 or int(raw["snapshot_commit_interval"]) < 1:
        raise ValueError("max_workers and snapshot_commit_interval must be >= 1")
    expected_tol = {"quantization", "aggregation", "embedding", "similarity"}
    if set(raw["tolerance"]) != expected_tol:
        raise ValueError(f"tolerance must contain exactly {sorted(expected_tol)}")
    for point in raw["quantile_points"]:
        if not 0.0 <= float(point) <= 1.0:
            raise ValueError("quantile points must lie within [0, 1]")

    scope_sources = {
        "feature": {name: raw[name] for name in FEATURE_FIELDS},
        "stats": {name: raw[name] for name in STATS_FIELDS},
        "quant": {name: raw[name] for name in QUANT_FIELDS},
        "embedding": {name: raw[name] for name in EMBED_FIELDS},
        "similarity": {name: raw[name] for name in SIM_FIELDS},
    }
    scope_hashes = {scope: fingerprint(payload) for scope, payload in scope_sources.items()}
    return PipelineConfig(raw=dict(raw), config_hash=fingerprint(raw), scope_hashes=scope_hashes)


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    sample_a: str
    sample_b: str
    relation: str  # "same_speaker" | "different_speaker"


@dataclass(frozen=True)
class PairManifest:
    speaker_of: dict[str, str]
    pairs: list[PairRecord]

    @property
    def sample_ids(self) -> list[str]:
        return sorted(self.speaker_of)


def load_pair_manifest(path: str) -> PairManifest:
    data = read_json(path)
    speaker_of: dict[str, str] = {}
    for entry in data["samples"]:
        sample_id = str(entry["sample_id"])
        if sample_id in speaker_of:
            raise ValueError(f"duplicate sample_id in pair manifest: {sample_id}")
        speaker_of[sample_id] = str(entry["speaker_id"])
    pairs: list[PairRecord] = []
    seen: set[str] = set()
    for entry in data["pairs"]:
        pair_id = str(entry["pair_id"])
        if pair_id in seen:
            raise ValueError(f"duplicate pair_id: {pair_id}")
        seen.add(pair_id)
        relation = str(entry["relation"])
        if relation not in ("same_speaker", "different_speaker"):
            raise ValueError(f"invalid relation for {pair_id}: {relation}")
        pairs.append(
            PairRecord(
                pair_id=pair_id,
                sample_a=str(entry["sample_a"]),
                sample_b=str(entry["sample_b"]),
                relation=relation,
            )
        )
    for pair in pairs:
        for sample_id in (pair.sample_a, pair.sample_b):
            if sample_id not in speaker_of:
                raise ValueError(f"pair {pair.pair_id} references unknown sample {sample_id}")
    return PairManifest(speaker_of=speaker_of, pairs=pairs)
