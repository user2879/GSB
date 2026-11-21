"""Content-fingerprint cache with precise invalidation (requirement 5).

Every stage product (normalization statistics, quantized integers,
weighted sums, clip verdicts) is stored under a cache key derived from
the content fingerprint of its upstream products plus the fingerprint of
the parameters that govern the stage.  Each entry records its reference
set (the upstream keys it was derived from).

After a command mutates inputs or parameters, recomputing a key either
hits (product reused verbatim, never recomputed) or misses (product is
recomputed and stored).  Because keys are pure functions of content, an
incremental run and a from-scratch full run with the same final inputs
produce byte-identical products (requirement 7).
"""

from __future__ import annotations

from .fingerprint import fingerprint


class StageCache:
    def __init__(self):
        self._entries = {}

    def key(self, stage, turn_id, upstream_fps, params):
        return fingerprint({
            "stage": stage,
            "turn_id": turn_id,
            "upstream": upstream_fps,
            "params": params,
        })

    def get(self, key):
        return self._entries.get(key)

    def put(self, key, stage, turn_id, value, refs):
        self._entries[key] = {
            "key": key,
            "stage": stage,
            "turn_id": turn_id,
            "value": value,
            "refs": sorted(refs),
        }
        return self._entries[key]

    def lookup_or(self, stage, turn_id, upstream_fps, params, compute):
        """Return (value, reused, key, refs).

        Reused products are returned verbatim and never recomputed.
        """
        key = self.key(stage, turn_id, upstream_fps, params)
        entry = self.get(key)
        if entry is not None:
            return entry["value"], True, key, entry["refs"]
        value = compute()
        refs = sorted(upstream_fps)
        self.put(key, stage, turn_id, value, refs)
        return value, False, key, refs

    def stats(self):
        return {"entries": len(self._entries)}

    def dump(self):
        return self._entries
