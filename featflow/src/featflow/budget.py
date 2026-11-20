"""Two-layer budget scheduling with an auditable ledger.

All budget decisions and ledger appends happen on the orchestrating thread in a
stable order (topological rank then feature name).  Worker threads only execute
pure computations, so concurrency level never changes the ledger sequence.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Callable

from .common import (
    CODE_FEAT_BUDGET_EXHAUSTED,
    CODE_GLOBAL_BUDGET_EXHAUSTED,
    Failure,
)
from .models import FeatureDef


@dataclass(frozen=True)
class LedgerEntry:
    seq: int
    point_seq: int
    topo_rank: int
    feature: str
    charge: int
    global_before: int
    global_after: int          # -1 means unlimited
    feature_used: int
    feature_quota: int
    decision: str              # charged | skip_aborted | skip_global | skip_upstream
    reason: str = ""


class BudgetScheduler:
    def __init__(self, global_budget: int, feature_quota: int,
                 concurrency: int) -> None:
        self.global_budget = global_budget          # -1 = unlimited
        self.global_remaining = global_budget
        self.feature_quota = feature_quota
        self.feature_used: dict[str, int] = {}
        self.aborted: dict[str, Failure] = {}
        self.concurrency = max(1, concurrency)
        self.ledger: list[LedgerEntry] = []
        self._seq = 0
        self.pending: list[tuple[int, str]] = []

    def is_aborted(self, feature: str) -> bool:
        return feature in self.aborted

    def plan(self, point_seq: int, feat: FeatureDef,
             upstream_aborted: bool) -> tuple[str, int]:
        """Decide the fate of one feature task. Returns (decision, charge)."""
        self._seq += 1  # continues from the restored snapshot sequence
        used = self.feature_used.get(feat.name, 0)
        global_before = self.global_remaining

        if upstream_aborted or self.is_aborted(feat.name):
            reason = self.aborted[feat.name].code if feat.name in self.aborted else "upstream_aborted"
            entry = LedgerEntry(self._seq, point_seq, feat.topo_rank, feat.name, 0,
                                global_before, self.global_remaining, used,
                                self.feature_quota, "skip_upstream", reason)
            self.ledger.append(entry)
            if upstream_aborted and feat.name not in self.aborted:
                self.aborted[feat.name] = Failure(
                    "feature", feat.name, CODE_FEAT_BUDGET_EXHAUSTED,
                    "upstream feature aborted")
            return "skip_upstream", 0

        if self.feature_quota >= 0 and used >= self.feature_quota:
            if feat.name not in self.aborted:
                self.aborted[feat.name] = Failure(
                    "feature", feat.name, CODE_FEAT_BUDGET_EXHAUSTED,
                    f"per-feature quota {self.feature_quota} exhausted")
            self.ledger.append(LedgerEntry(
                self._seq, point_seq, feat.topo_rank, feat.name, 0,
                global_before, self.global_remaining, used,
                self.feature_quota, "skip_aborted", CODE_FEAT_BUDGET_EXHAUSTED))
            self.pending.append((point_seq, feat.name))
            return "skip_aborted", 0

        if self.global_budget >= 0 and self.global_remaining <= 0:
            if feat.name not in self.aborted:
                self.aborted[feat.name] = Failure(
                    "feature", feat.name, CODE_GLOBAL_BUDGET_EXHAUSTED,
                    f"global budget {self.global_budget} exhausted")
            self.ledger.append(LedgerEntry(
                self._seq, point_seq, feat.topo_rank, feat.name, 0,
                global_before, self.global_remaining, used,
                self.feature_quota, "skip_global", CODE_GLOBAL_BUDGET_EXHAUSTED))
            self.pending.append((point_seq, feat.name))
            return "skip_global", 0

        charge = 1
        if self.global_budget >= 0:
            self.global_remaining -= charge
        self.feature_used[feat.name] = used + charge
        self.ledger.append(LedgerEntry(
            self._seq, point_seq, feat.topo_rank, feat.name, charge,
            global_before, self.global_remaining, used + charge,
            self.feature_quota, "charged"))
        return "charged", charge

    def run_level(self, tasks: list[tuple[FeatureDef, Callable[[], object]]]) -> list[object | None]:
        """Run a topological level under the concurrency cap; preserve input order."""
        if not tasks:
            return []
        if self.concurrency == 1 or len(tasks) == 1:
            return [fn() for _, fn in tasks]
        with ThreadPoolExecutor(max_workers=min(self.concurrency, len(tasks))) as pool:
            futures = [pool.submit(fn) for _, fn in tasks]
            return [f.result() for f in futures]


def _ledger_from_dict(row: dict) -> "LedgerEntry":
    return LedgerEntry(
        seq=row["seq"], point_seq=row["point_seq"], topo_rank=row["topo_rank"],
        feature=row["feature"], charge=row["charge"],
        global_before=row["global_before"], global_after=row["global_after"],
        feature_used=row["feature_used"], feature_quota=row["feature_quota"],
        decision=row["decision"], reason=row.get("reason", ""))
