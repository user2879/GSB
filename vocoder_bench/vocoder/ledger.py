"""Content-fingerprint cache ledger and precise invalidation bookkeeping."""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

from .hashing import hash_mix

STAGES = ("phase", "harmonic", "overlap_add", "metric")


def stage_cache_key(stage: str, utterance_id: str, feature_fp: str,
                    config_fp: str, phase_profile_fp: str,
                    frame_index: int | None = None) -> str:
    frame_part = "all" if frame_index is None else f"f{frame_index}"
    return hash_mix(stage, utterance_id, feature_fp, config_fp,
                    phase_profile_fp, frame_part)


@dataclass
class ReuseRecord:
    utterance_id: str
    stage: str
    cache_key: str
    frame_index: int | None
    reused: bool


@dataclass
class Ledger:
    # cache_key -> payload location metadata
    entries: dict[str, dict] = field(default_factory=dict)
    hits: list[ReuseRecord] = field(default_factory=list)
    # per-utterance referenced feature frame fingerprints
    frame_sets: dict[str, list[str]] = field(default_factory=dict)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def register_frames(self, utterance_id: str, frame_fingerprints: list[str]) -> None:
        with self._lock:
            self.frame_sets[utterance_id] = list(frame_fingerprints)

    def reset_hits(self) -> None:
        with self._lock:
            self.hits = []

    def lookup(self, stage: str, utterance_id: str, cache_key: str,
               frame_index: int | None = None, record: bool = True) -> dict | None:
        with self._lock:
            payload = self.entries.get(cache_key)
            if record:
                self.hits.append(ReuseRecord(
                    utterance_id=utterance_id, stage=stage, cache_key=cache_key,
                    frame_index=frame_index, reused=payload is not None,
                ))
            return payload

    def record_hit(self, stage: str, utterance_id: str, cache_key: str,
                   frame_index: int | None, reused: bool) -> None:
        with self._lock:
            self.hits.append(ReuseRecord(
                utterance_id=utterance_id, stage=stage, cache_key=cache_key,
                frame_index=frame_index, reused=reused,
            ))

    def store(self, cache_key: str, payload: dict) -> None:
        with self._lock:
            self.entries[cache_key] = payload

    def counters(self) -> dict:
        with self._lock:
            return self._counters_locked()

    def _counters_locked(self) -> dict:
        total = len(self.hits)
        reused = sum(1 for hit in self.hits if hit.reused)
        by_stage = {}
        for stage in STAGES:
            stage_hits = [hit for hit in self.hits if hit.stage == stage]
            by_stage[stage] = {
                "lookups": len(stage_hits),
                "reused": sum(1 for hit in stage_hits if hit.reused),
                "recomputed": sum(1 for hit in stage_hits if not hit.reused),
            }
        return {"total_lookups": total, "total_reused": reused,
                "total_recomputed": total - reused, "by_stage": by_stage}

    def hit_evidence(self) -> list[dict]:
        with self._lock:
            return self._hit_evidence_locked()

    def _hit_evidence_locked(self) -> list[dict]:
        return [
            {
                "utterance_id": hit.utterance_id,
                "stage": hit.stage,
                "frame_index": hit.frame_index,
                "cache_key": hit.cache_key,
                "reused": hit.reused,
            }
            for hit in self.hits
        ]

    def save(self, path: str | Path) -> None:
        with self._lock:
            Path(path).write_text(json.dumps({
            "entries": self.entries,
            "frame_sets": self.frame_sets,
            "counters": self.counters(),
        }, indent=2, sort_keys=True), encoding="utf-8")
