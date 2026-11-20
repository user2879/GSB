"""Deterministic pipeline configuration and fingerprint helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Dict


def stable_json(obj) -> str:
    """Canonical JSON encoding used for fingerprints and cache keys."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def fingerprint(obj) -> str:
    """SHA-256 fingerprint of a JSON-serialisable object."""
    return hashlib.sha256(stable_json(obj).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PipelineConfig:
    """All tunable parameters. Frozen + hashable for deterministic runs."""

    seed: int = 20261120
    position_decay: float = 0.9
    rule_bonus_weight: float = 1.0
    max_rewrites: int = 8
    max_workers: int = 4
    latency_budget_ms: float = 25.0
    exhaustive_verify_sample: int = 256
    dict_priority: tuple = ("brand", "category", "attribute")
    strategy_thresholds: Dict[str, int] = field(
        default_factory=lambda: {
            "inverted_dict_entries": 4000,
            "prefix_dict_entries": 12000,
            "inverted_vocab_entries": 4000,
            "cache_queries": 5000,
        }
    )

    def fingerprint(self) -> str:
        return fingerprint(asdict(self))


DEFAULT_CONFIG = PipelineConfig()
