"""Reviewable diagnostic report assembly."""

from __future__ import annotations

import time
from collections import Counter
from dataclasses import dataclass, field, asdict

from .errors import BizCode, BizError


@dataclass
class RunReport:
    started_at: float
    finished_at: float = 0.0
    receipt_counts: dict[str, int] = field(default_factory=dict)
    boundary_token_count: int = 0
    invalidated_count: int = 0
    cache_hit_count: int = 0
    recompute_count: int = 0
    todos: list[dict] = field(default_factory=list)
    biz_code_counts: dict[str, int] = field(default_factory=dict)
    injection_hits: list[dict] = field(default_factory=list)
    attribution_rows: list[dict] = field(default_factory=list)
    request_count: int = 0
    failed_requests: list[str] = field(default_factory=list)

    @property
    def elapsed_seconds(self) -> float:
        return round(self.finished_at - self.started_at, 6)

    def add_failures(self, failures: list[BizError]) -> None:
        counter = Counter(BizCode.name(f.code) for f in failures)
        for name, count in counter.items():
            self.biz_code_counts[name] = self.biz_code_counts.get(name, 0) + count
        for failure in failures:
            if failure.injection_id:
                self.injection_hits.append(failure.to_dict())

    def finalize(self) -> dict:
        self.finished_at = self.finished_at or time.perf_counter()
        data = asdict(self)
        data["elapsed_seconds"] = self.elapsed_seconds
        return data
