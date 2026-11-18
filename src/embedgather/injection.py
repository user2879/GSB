"""Fault injection driven by a local JSON injection config.

Supported injection types:
  - nonfinite_vector      (stage validate)  -> E_EMBED_INVALID
  - id_out_of_range       (stage gather)    -> E_ID_MISSING
  - stale_snapshot_layout (stage snapshot)  -> E_SNAPSHOT_MISMATCH
  - zero_quota            (stage schedule)  -> E_POINT_BUDGET_EXHAUSTED

Every applied injection is recorded with its id and mapped business code;
a single fault never interrupts other requests.
"""

import json

from .errors import (E_EMBED_INVALID, E_ID_MISSING, E_POINT_BUDGET_EXHAUSTED,
                     E_SNAPSHOT_MISMATCH)

TYPE_TO_CODE = {
    "nonfinite_vector": E_EMBED_INVALID,
    "id_out_of_range": E_ID_MISSING,
    "stale_snapshot_layout": E_SNAPSHOT_MISMATCH,
    "zero_quota": E_POINT_BUDGET_EXHAUSTED,
}


class Injector:
    def __init__(self, injections=()):
        self.injections = list(injections)
        self.hits = []

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(data.get("injections", []))

    @classmethod
    def null(cls):
        return cls(())

    def _record(self, inj, detail):
        self.hits.append({
            "injection_id": inj.get("id"),
            "type": inj.get("type"),
            "stage": inj.get("stage"),
            "business_code": TYPE_TO_CODE[inj["type"]],
            "detail": detail,
        })

    # ------------------------------------------------------------ validate
    def corrupt_vector(self, word_id, vector):
        for inj in self.injections:
            if (inj.get("type") == "nonfinite_vector"
                    and inj.get("word_id") == word_id):
                if isinstance(vector, list) and vector:
                    corrupted = list(vector)
                    corrupted[0] = float("nan")
                    self._record(inj, {"word_id": word_id})
                    return corrupted
        return vector

    # -------------------------------------------------------------- gather
    def extra_ids_for(self, request_index):
        extra = []
        for inj in self.injections:
            if (inj.get("type") == "id_out_of_range"
                    and inj.get("request_index") == request_index):
                wid = inj.get("out_of_range_id", 2**31 - 1)
                extra.append(wid)
                self._record(inj, {"request_index": request_index,
                                   "word_id": wid})
        return extra

    # ------------------------------------------------------------ schedule
    def quota_for(self, request_index, quota):
        for inj in self.injections:
            if (inj.get("type") == "zero_quota"
                    and inj.get("request_index") == request_index):
                self._record(inj, {"request_index": request_index})
                return 0
        return quota

    # ------------------------------------------------------------ snapshot
    def tampered_manifest(self, manifest):
        """Return a tampered copy of the manifest simulating a stale
        snapshot reused after the block layout changed."""
        for inj in self.injections:
            if inj.get("type") == "stale_snapshot_layout":
                tampered = dict(manifest)
                tampered["layout_fp"] = "0" * 64
                self._record(inj, {"field": "layout_fp"})
                return tampered
        return None
