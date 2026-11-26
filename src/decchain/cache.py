"""Incremental conclusion cache with selective invalidation.

A cached conclusion is keyed by the request content fingerprint, config
fingerprint, sequence id and step index.  Edits invalidate only conclusions
whose inputs actually changed; unaffected steps are served from cache and
recorded as cache hits.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .config import DecodeConfig
from .models import Request
from .sampling import SamplingConclusion


def make_cache_key(step_fp: str, config_fp: str,
                   seq_id: int, step_index: int) -> str:
    mat = f"{step_fp}|{config_fp}|{int(seq_id)}|{int(step_index)}"
    return hashlib.sha256(mat.encode("utf-8")).hexdigest()


@dataclass
class CacheEvent:
    request_id: str
    seq_id: int
    step_index: int
    kind: str           # HIT / RECOMPUTE / INVALIDATE
    cache_key: str


class ConclusionCache:
    def __init__(self) -> None:
        self._store: dict[str, SamplingConclusion] = {}
        self.events: list[CacheEvent] = []
        self.invalidated: list[CacheEvent] = []

    def key_for(self, request: Request, seq_id: int, step_index: int,
                config: DecodeConfig) -> str:
        step_fp = request.sequence_map()[seq_id].step_fingerprint(step_index)
        return make_cache_key(step_fp, config.fingerprint(),
                              seq_id, step_index)

    def peek(self, request: Request, seq_id: int, step_index: int,
             config: DecodeConfig) -> SamplingConclusion | None:
        key = self.key_for(request, seq_id, step_index, config)
        return self._store.get(key)

    def get(self, request: Request, seq_id: int, step_index: int,
            config: DecodeConfig) -> SamplingConclusion | None:
        key = self.key_for(request, seq_id, step_index, config)
        conclusion = self._store.get(key)
        if conclusion is not None:
            self.events.append(CacheEvent(
                request.request_id, seq_id, step_index, "HIT", key))
        return conclusion

    def put(self, request: Request, seq_id: int, step_index: int,
            config: DecodeConfig, conclusion: SamplingConclusion) -> None:
        key = self.key_for(request, seq_id, step_index, config)
        self._store[key] = conclusion
        self.events.append(CacheEvent(
            request.request_id, seq_id, step_index, "RECOMPUTE", key))

    def invalidate_step(self, request_id: str, content_fp_before: str,
                        config_fp: str, seq_id: int, step_index: int) -> None:
        key = make_cache_key(content_fp_before, config_fp, seq_id, step_index)
        if key in self._store:
            del self._store[key]
            event = CacheEvent(request_id, seq_id, step_index,
                               "INVALIDATE", key)
            self.invalidated.append(event)
            self.events.append(event)

    def invalidate_request(self, request_id: str, content_fp_before: str,
                           config: DecodeConfig,
                           seq_steps: list[tuple[int, int]]) -> None:
        for seq_id, step_index in seq_steps:
            self.invalidate_step(request_id, content_fp_before,
                                 config.fingerprint(), seq_id, step_index)

    def invalidate_all_config(self, old_config_fp: str) -> None:
        for key in list(self._store):
            # Cache keys embed the config fingerprint as the second segment.
            if old_config_fp in key:
                del self._store[key]


    def replace_config_fingerprint(self, old_fp: str, new_fp: str) -> None:
        """Rewrite embedded config fingerprints; stale entries are removed.

        Content-only conclusions remain valid if the new config still maps to
        the same material, otherwise the rewritten key simply will not match a
        future lookup and is therefore dropped on first access.
        """
        rebuilt = {}
        for key, conclusion in list(self._store.items()):
            new_key = make_cache_key(conclusion.content_fingerprint, new_fp,
                                     conclusion.seq_id, conclusion.step_index)
            if new_key != key:
                self.events.append(CacheEvent(
                    conclusion.request_id, conclusion.seq_id,
                    conclusion.step_index, "INVALIDATE", new_key))
                self.invalidated.append(CacheEvent(
                    conclusion.request_id, conclusion.seq_id,
                    conclusion.step_index, "INVALIDATE", new_key))
            rebuilt[new_key] = conclusion
        self._store = rebuilt

    def hits(self) -> list[CacheEvent]:
        return [e for e in self.events if e.kind == "HIT"]

    def recomputes(self) -> list[CacheEvent]:
        return [e for e in self.events if e.kind == "RECOMPUTE"]

    def reset_events(self) -> None:
        self.events = []

    def __len__(self) -> int:
        return len(self._store)
