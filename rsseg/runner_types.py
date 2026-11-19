"""Shared runtime data structures for incremental runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from .config import SegmentationConfig
from .errors import FailureRecord
from .injection import InjectionPlan
from .tiles import TileIndex


@dataclass
class RunInputs:
    sample_dir: Path
    work_dir: Path
    config: SegmentationConfig
    index: TileIndex
    index_path: Path
    patches_dir: Path | None = None
    injections: InjectionPlan | None = None


@dataclass
class RunReport:
    run_mode: str
    config_hash: str
    index_version: str
    requested_tiles: list[tuple[int, int]] = field(default_factory=list)
    recomputed_tiles: list[tuple[int, int]] = field(default_factory=list)
    reused_tiles: list[tuple[int, int]] = field(default_factory=list)
    recomputed_fusions: list[str] = field(default_factory=list)
    reused_fusions: list[str] = field(default_factory=list)
    pending_tiles: list[tuple[int, int]] = field(default_factory=list)
    pending_fusions: list[str] = field(default_factory=list)
    failures: list[FailureRecord] = field(default_factory=list)
    hits: list[str] = field(default_factory=list)
    snapshot_version: int | None = None
    budget: int = 0
    budget_used: int = 0
    signaled: bool = False

    def failure_dicts(self) -> list[dict]:
        return [record.to_dict() for record in sorted(self.failures, key=lambda r: r.sort_key)]

    def to_dict(self) -> dict:
        return {
            "run_mode": self.run_mode,
            "config_hash": self.config_hash,
            "index_version": self.index_version,
            "requested_tiles": [list(k) for k in self.requested_tiles],
            "recomputed_tiles": [list(k) for k in self.recomputed_tiles],
            "reused_tiles": [list(k) for k in self.reused_tiles],
            "recomputed_fusions": self.recomputed_fusions,
            "reused_fusions": self.reused_fusions,
            "pending_tiles": [list(k) for k in self.pending_tiles],
            "pending_fusions": self.pending_fusions,
            "failures": self.failure_dicts(),
            "injection_hits": sorted(set(self.hits)),
            "snapshot_version": self.snapshot_version,
            "budget": self.budget,
            "budget_used": self.budget_used,
            "signaled": self.signaled,
        }
