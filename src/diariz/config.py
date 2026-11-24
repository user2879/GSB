"""Logging configuration parsing and strict validation.

All public values are plain JSON-friendly Python objects so that the
configuration fingerprint is reproducible across runs and machines.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict

from .errors import ConfigInvalidError

SAMPLE_RATE = 16000
ALLOWED_WORD_BITS = (8, 16, 32)
ALLOWED_FRAC_BITS = (4, 6, 8, 10, 12, 14)
ALLOWED_BOUNDARY_RULES = ("frame_edge", "window_midpoint", "margin_weighted")
MAX_CONCURRENCY = 16
# Ordered low -> high precision quantization tiers (requirement 7).
QUANT_TIERS = (8, 16, 32)


@dataclass(frozen=True)
class DiarConfig:
    window_ms: int
    hop_ms: int
    embed_dim: int
    threshold: float
    frac_bits: int
    word_bits: int
    concurrency: int
    min_segment_ms: int
    min_silence_ms: int
    boundary_rule: str
    tolerance: float
    frame_ms: int = 25
    frame_hop_ms: int = 10

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def fingerprint(self) -> str:
        return fingerprint_config(self.to_dict())

    def projection_fp(self) -> str:
        return projection_fingerprint(self.to_dict())

    def next_tier_word_bits(self) -> int:
        idx = QUANT_TIERS.index(self.word_bits)
        return QUANT_TIERS[min(idx + 1, len(QUANT_TIERS) - 1)]

    def with_quant(self, word_bits: int, frac_bits: int) -> "DiarConfig":
        data = self.to_dict()
        data["word_bits"] = word_bits
        data["frac_bits"] = frac_bits
        return DiarConfig(**data)


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ConfigInvalidError(message, {"reason": message})


def fingerprint_config(cfg: Dict[str, Any]) -> str:
    """Deterministic SHA-256 fingerprint independent of key order."""
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


PROJECTION_FIELDS = ("embed_dim", "frame_ms", "frame_hop_ms", "window_ms",
                     "hop_ms")


def projection_fingerprint(cfg: Dict[str, Any]) -> str:
    """Fingerprint covering only values that change the embedding projection."""
    subset = {key: cfg[key] for key in PROJECTION_FIELDS if key in cfg}
    return fingerprint_config(subset)


def load_config(path: str) -> DiarConfig:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        raise ConfigInvalidError(f"config not found: {path}")
    except json.JSONDecodeError as exc:
        raise ConfigInvalidError(f"config is not valid JSON: {exc}")
    if not isinstance(raw, dict):
        raise ConfigInvalidError("config root must be a JSON object")
    return parse_config(raw)


def parse_config(raw: Dict[str, Any]) -> DiarConfig:
    def int_field(name: str) -> int:
        if name not in raw or not isinstance(raw[name], int) or isinstance(raw[name], bool):
            raise ConfigInvalidError(f"{name} must be an integer")
        return int(raw[name])

    def float_field(name: str) -> float:
        if name not in raw or not isinstance(raw[name], (int, float)) or isinstance(raw[name], bool):
            raise ConfigInvalidError(f"{name} must be a number")
        return float(raw[name])

    window_ms = int_field("window_ms")
    hop_ms = int_field("hop_ms")
    embed_dim = int_field("embed_dim")
    threshold = float_field("threshold")
    frac_bits = int_field("frac_bits")
    word_bits = int_field("word_bits")
    concurrency = int_field("concurrency")
    min_segment_ms = int_field("min_segment_ms")
    min_silence_ms = int_field("min_silence_ms")
    frame_ms = int(raw.get("frame_ms", 25))
    frame_hop_ms = int(raw.get("frame_hop_ms", 10))
    tolerance = float(raw.get("tolerance", 1e-3))
    boundary_rule = str(raw.get("boundary_rule", "margin_weighted"))

    _require(window_ms > 0, "window_ms must be positive")
    _require(hop_ms > 0, "hop_ms must be positive")
    _require(hop_ms <= window_ms, "hop_ms must not exceed window_ms")
    _require(embed_dim > 0 and embed_dim % 2 == 0, "embed_dim must be a positive even number")
    _require(0.0 <= threshold <= 1.0, "threshold must be within [0, 1]")
    _require(word_bits in ALLOWED_WORD_BITS,
             f"word_bits must be one of {ALLOWED_WORD_BITS}")
    _require(frac_bits in ALLOWED_FRAC_BITS,
             f"frac_bits must be one of {ALLOWED_FRAC_BITS}")
    _require(frac_bits < word_bits - 1,
             "frac_bits must leave at least one integer/sign bit")
    _require(1 <= concurrency <= MAX_CONCURRENCY,
             f"concurrency must be within [1, {MAX_CONCURRENCY}]")
    _require(min_segment_ms >= 0, "min_segment_ms must be non-negative")
    _require(min_silence_ms >= 0, "min_silence_ms must be non-negative")
    _require(boundary_rule in ALLOWED_BOUNDARY_RULES,
             f"boundary_rule must be one of {ALLOWED_BOUNDARY_RULES}")
    _require(tolerance > 0, "tolerance must be positive")
    _require(frame_ms > 0 and 0 < frame_hop_ms <= frame_ms,
             "frame_ms/frame_hop_ms invalid")

    return DiarConfig(
        window_ms=window_ms, hop_ms=hop_ms, embed_dim=embed_dim,
        threshold=threshold, frac_bits=frac_bits, word_bits=word_bits,
        concurrency=concurrency, min_segment_ms=min_segment_ms,
        min_silence_ms=min_silence_ms, boundary_rule=boundary_rule,
        tolerance=tolerance, frame_ms=frame_ms, frame_hop_ms=frame_hop_ms,
    )


def default_config() -> DiarConfig:
    return DiarConfig(
        window_ms=200, hop_ms=100, embed_dim=16, threshold=0.90,
        frac_bits=10, word_bits=16, concurrency=4, min_segment_ms=120,
        min_silence_ms=150, boundary_rule="margin_weighted", tolerance=1e-3,
    )


def write_config(path: str, cfg: DiarConfig) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(cfg.to_dict(), handle, indent=2, sort_keys=True)
        handle.write("\n")
