"""Auditable, reproducibility ledger.

The ledger records, in deterministic order:

* every cache **reuse hit** with the key and the dependency that matched,
* every **invalidation** with the precise reason (which segment/window/quant
  level caused it),
* every **recompute** and the stage it belongs to,
* every validation/injection **failure**,
* the authoritative summaries (stale summaries are explicitly *not* recorded).

It is an append-only JSONL-style document plus an in-memory index, so the
history of every conclusion flip can be traced back to a segment, a window and
a fixed-point quantization level.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .errors import FailureRecord


@dataclass
class Ledger:
    path: str
    events: List[Dict[str, Any]] = field(default_factory=list)

    def load(self) -> "Ledger":
        self.events = []
        if os.path.isfile(self.path):
            with open(self.path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        self.events.append(json.loads(line))
        return self

    def append(self, kind: str, **fields: Any) -> Dict[str, Any]:
        event = {"seq": len(self.events), "kind": kind}
        event.update(fields)
        self.events.append(event)
        return event

    def record_reuse(self, stage: str, key: str, reason: str,
                     item_id: str) -> None:
        self.append("reuse_hit", stage=stage, key=key, reason=reason,
                    item_id=item_id)

    def record_invalidation(self, stage: str, key: str, reason: str,
                            item_id: str, caused_by: Optional[str] = None,
                            quant_level: Optional[str] = None) -> None:
        event = {"stage": stage, "key": key, "reason": reason,
                 "item_id": item_id}
        if caused_by is not None:
            event["caused_by_segment"] = caused_by
        if quant_level is not None:
            event["quant_level"] = quant_level
        self.append("invalidation", **event)

    def record_recompute(self, stage: str, key: str, item_id: str,
                         quant_level: Optional[str] = None) -> None:
        event = {"stage": stage, "key": key, "item_id": item_id}
        if quant_level is not None:
            event["quant_level"] = quant_level
        self.append("recompute", **event)

    def record_failure(self, failure: FailureRecord) -> None:
        self.append("failure", **failure.to_dict())

    def record_summary(self, summary: Dict[str, Any]) -> None:
        # Only authoritative (fresh) summaries reach the ledger.
        self.append("authoritative_summary", summary=summary)

    def record_region_move(self, region_id: str, old: Dict[str, Any],
                           new: Dict[str, Any], caused_by: str) -> None:
        self.append("region_move", region_id=region_id, old=old, new=new,
                    caused_by_segment=caused_by)

    def record_event(self, kind: str, **fields: Any) -> None:
        self.append(kind, **fields)

    def by_kind(self, kind: str) -> List[Dict[str, Any]]:
        return [e for e in self.events if e["kind"] == kind]

    def flush(self) -> None:
        directory = os.path.dirname(os.path.abspath(self.path)) or "."
        os.makedirs(directory, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".ledger-", suffix=".jsonl",
                                   dir=directory)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                for event in self.events:
                    handle.write(json.dumps(
                        event, sort_keys=True, separators=(",", ":"),
                        ensure_ascii=False))
                    handle.write("\n")
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def reset(self) -> None:
        self.events = []
        if os.path.exists(self.path):
            os.remove(self.path)
