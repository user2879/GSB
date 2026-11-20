"""Business error codes and failure records.

All failures are collected into an ordered failure list; parsing and
execution never abort early on a single failure.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional

# --- business codes -------------------------------------------------------
E_CONFIG_INVALID = "E_CONFIG_INVALID"
E_CLUSTER_ZERO = "E_CLUSTER_ZERO"
E_BUDGET_INVALID = "E_BUDGET_INVALID"
E_CAND_BUDGET_EXHAUSTED = "E_CAND_BUDGET_EXHAUSTED"
E_ITER_BUDGET_PENDING = "E_ITER_BUDGET_PENDING"
E_SNAPSHOT_MISMATCH = "E_SNAPSHOT_MISMATCH"
E_INJECTED_FAULT = "E_INJECTED_FAULT"
E_CANDIDATE_FAULT = "E_CANDIDATE_FAULT"


@dataclass
class Failure:
    code: str
    message: str
    stage: Optional[str] = None
    candidate_key: Optional[str] = None
    context: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class FailureLog:
    """Ordered, deterministic failure collection."""

    def __init__(self) -> None:
        self._items: list[Failure] = []

    def add(self, code: str, message: str, stage: Optional[str] = None,
            candidate_key: Optional[str] = None, **context: Any) -> Failure:
        rec = Failure(code=code, message=message, stage=stage,
                      candidate_key=candidate_key, context=context)
        self._items.append(rec)
        return rec

    def extend(self, others: list[Failure]) -> None:
        self._items.extend(others)

    def to_list(self) -> list[dict]:
        return [f.to_dict() for f in self._items]

    def codes(self) -> list[str]:
        return [f.code for f in self._items]

    def __len__(self) -> int:
        return len(self._items)
