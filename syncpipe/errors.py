"""Business error codes and structured error records.

Every error carries:
  * code        -- one of the stable E_* codes below
  * version     -- authoritative version at the time of detection (or null
                   when no version has ever been produced)
  * tick        -- tick of the triggering event (or null)
  * event_loc   -- peer_id / local_seq of the triggering event (or null)
  * reason      -- short deterministic human readable explanation

Illegal ticks and peer sequence breaks share E_PEER_GAP but carry distinct
``reason`` values, keeping the seven mandated codes stable while preserving
precise diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional, Tuple


E_BASE_VERSION = "E_BASE_VERSION"
E_PEER_GAP = "E_PEER_GAP"
E_SNAPSHOT_HASH = "E_SNAPSHOT_HASH"
E_LOG_TRUNCATED = "E_LOG_TRUNCATED"
E_VERSION_DIVERGE = "E_VERSION_DIVERGE"
E_CONFLICT_UNRESOLVED = "E_CONFLICT_UNRESOLVED"
E_CHAIN_BROKEN = "E_CHAIN_BROKEN"

ALL_CODES = (
    E_BASE_VERSION,
    E_PEER_GAP,
    E_SNAPSHOT_HASH,
    E_LOG_TRUNCATED,
    E_VERSION_DIVERGE,
    E_CONFLICT_UNRESOLVED,
    E_CHAIN_BROKEN,
)


@dataclass(frozen=True)
class PipelineError(Exception):
    code: str
    reason: str
    version: Optional[int] = None
    tick: Optional[int] = None
    event_loc: Optional[Tuple[str, int]] = None

    def __str__(self) -> str:  # pragma: no cover - trivial
        loc = f" event={self.event_loc[0]}#{self.event_loc[1]}" if self.event_loc else ""
        ver = f" version={self.version}" if self.version is not None else ""
        tick = f" tick={self.tick}" if self.tick is not None else ""
        return f"{self.code}{ver}{tick}{loc}: {self.reason}"

    def to_record(self) -> Dict[str, Any]:
        record = asdict(self)
        if self.event_loc is not None:
            record["event_loc"] = {
                "peer_id": self.event_loc[0],
                "local_seq": self.event_loc[1],
            }
        return record


def make_error(
    code: str,
    reason: str,
    version: Optional[int] = None,
    tick: Optional[int] = None,
    event_loc: Optional[Tuple[str, int]] = None,
) -> Dict[str, Any]:
    return PipelineError(code, reason, version, tick, event_loc).to_record()
