from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Any, Iterable

from .canonical import sha256_json
from .encoding import scheme_parameter_fingerprint
from .models import AttentionSample, CacheEntry, PositionEncodingConfig


@dataclass(frozen=True)
class StageKeys:
    position_encoding: str
    attention_score: str
    error_trace: str

    def as_dict(self) -> dict[str, str]:
        return {
            "position_encoding": self.position_encoding,
            "attention_score": self.attention_score,
            "error_trace": self.error_trace,
        }


def build_stage_keys(config: PositionEncodingConfig, sample: AttentionSample) -> StageKeys:
    """Build content-addressed keys for all three diagnostic stages.

    Training context length affects the interval statistics and reference
    positions, but not the raw position encoding.  This is deliberate: when
    only the context length changes, the encoding stage is reused while score
    and trace stages are recomputed.
    """

    scheme_fingerprint = scheme_parameter_fingerprint(config)
    sample_fingerprint = sample.content_fingerprint()
    common = {
        "scheme": config.scheme,
        "scheme_parameters": scheme_fingerprint,
        "sample_content": sample_fingerprint,
        "precision": config.precision,
        "dimension": config.dimension,
        "dimension_chunk_size": config.dimension_chunk_size,
        "pruning_policy": config.pruning_policy,
        "pruning_tolerance": config.pruning_tolerance,
    }
    encoding_key = sha256_json({"stage": "position_encoding", **common})
    score_key = sha256_json(
        {
            "stage": "attention_score",
            **common,
            "position_encoding_key": encoding_key,
            "training_context_length": config.training_context_length,
            "extrapolation_target_length": config.extrapolation_target_length,
        }
    )
    trace_key = sha256_json(
        {
            "stage": "error_trace",
            **common,
            "attention_score_key": score_key,
            "training_context_length": config.training_context_length,
            "extrapolation_target_length": config.extrapolation_target_length,
            "error_tolerance": config.error_tolerance,
        }
    )
    return StageKeys(
        position_encoding=encoding_key,
        attention_score=score_key,
        error_trace=trace_key,
    )


class ResultCache:
    """In-memory content-addressed cache.

    A cache hit is only possible for an exact key match.  The engine therefore
    cannot accidentally reuse a result with a different scheme, precision,
    pruning mask or sample content.
    """

    def __init__(self) -> None:
        self.entries: dict[tuple[str, str], CacheEntry] = {}
        self.recompute_counts: dict[str, int] = {
            "position_encoding": 0,
            "attention_score": 0,
            "error_trace": 0,
        }
        self._lock = threading.RLock()

    def get(self, stage: str, cache_key: str) -> CacheEntry | None:
        with self._lock:
            return self.entries.get((stage, cache_key))

    def put(self, entry: CacheEntry) -> None:
        with self._lock:
            self.entries[(entry.stage, entry.cache_key)] = entry

    def mark_recomputed(self, stage: str) -> None:
        with self._lock:
            self.recompute_counts[stage] = self.recompute_counts.get(stage, 0) + 1

    def current_entries(self, keys: Iterable[tuple[str, str]]) -> list[CacheEntry]:
        with self._lock:
            return [self.entries[key] for key in keys if key in self.entries]

    def all_entries(self) -> list[CacheEntry]:
        with self._lock:
            return [self.entries[key] for key in sorted(self.entries)]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "entry_count": len(self.entries),
                "recompute_counts": dict(sorted(self.recompute_counts.items())),
            }
