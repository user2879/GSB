"""End-to-end profiling session orchestration.

The session wires together strict sample loading, batched ledger commits,
adaptive dual-path folding, stable ranking/context, fixed-point error
annotation, versioned snapshots and replay verification.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import folding as fold_mod
from .errors import LedgerError, RangeError
from .fixedpoint import QuantParams, build_share_report
from .folding import (
    DEFAULT_MAX_TOTAL_WEIGHT,
    DEFAULT_MAX_UNIQUE_STACKS,
    FoldResult,
)
from .ledger import Ledger, RecoveryInfo
from .ranking import FrameContext, FrameRank, build_contexts, rank_frames
from .samples import (
    Sample,
    iter_samples,
    make_interval,
    sample_stream_digest,
)
from .snapshot import (
    build_snapshot,
    read_snapshot,
    should_replace,
    verify_replay,
    write_snapshot,
)


class ForcedKill(RuntimeError):
    """Raised by the crash demo *after* the configured batches committed."""


@dataclass
class SessionConfig:
    sample_path: str
    ledger_path: str
    snapshot_dir: str
    start_ms: Optional[int] = None
    end_ms: Optional[int] = None
    scale: int = 10_000
    top_k: int = 10
    batch_size: int = 1_000
    max_unique_stacks: int = DEFAULT_MAX_UNIQUE_STACKS
    max_total_weight: int = DEFAULT_MAX_TOTAL_WEIGHT
    force_path: Optional[str] = None
    interval_version: int = 1

    def session_id(self) -> str:
        basis = json.dumps(
            {
                "sample": str(Path(self.sample_path).resolve()),
                "interval": [self.start_ms, self.end_ms],
                "interval_version": self.interval_version,
                "scale": self.scale,
                "top_k": self.top_k,
                "thresholds": [
                    self.max_unique_stacks, self.max_total_weight],
                "force_path": self.force_path,
            },
            sort_keys=True,
        )
        return "sess-" + hashlib.sha256(basis.encode()).hexdigest()[:16]


def _batch_deltas(samples: Sequence[Sample]) -> dict:
    stack_deltas: Dict[Tuple[str, ...], int] = {}
    self_deltas: Dict[str, int] = {}
    cumulative_deltas: Dict[str, int] = {}
    weight_added = 0
    for sample in samples:
        frames = sample.frames
        weight = sample.weight
        stack_deltas[frames] = stack_deltas.get(frames, 0) + weight
        self_deltas[frames[-1]] = self_deltas.get(frames[-1], 0) + weight
        for frame in set(frames):
            cumulative_deltas[frame] = \
                cumulative_deltas.get(frame, 0) + weight
        weight_added += weight
    return {
        "stack_deltas": stack_deltas,
        "self_deltas": self_deltas,
        "cumulative_deltas": cumulative_deltas,
        "weight_added": weight_added,
    }


class ProfileSession:
    def __init__(self, config: SessionConfig) -> None:
        self.config = config
        self.interval = make_interval(config.start_ms, config.end_ms)
        if config.top_k <= 0:
            raise RangeError("top_k must be positive", field="top_k")
        if config.batch_size <= 0:
            raise RangeError("batch_size must be positive",
                             field="batch_size")
        self.quant = QuantParams(config.scale)
        self.session_id = config.session_id()
        self.ledger: Optional[Ledger] = None
        self.result: Optional[FoldResult] = None
        self.ranks: List[FrameRank] = []
        self.contexts: Dict[str, FrameContext] = {}
        self.report = None
        self.snapshot_record: Optional[dict] = None

    # ------------------------------------------------------------- lifecycle
    def _open_ledger(self, sample_digest: str) -> Ledger:
        path = Path(self.config.ledger_path)
        if path.exists():
            probe = Ledger(path)
            existing = probe.get_meta("session_id")
            if existing is not None and existing != self.session_id:
                probe.close()
                # Parameters/session changed: whole-state replacement.
                for suffix in ("", "-wal", "-shm"):
                    stale = Path(str(path) + suffix)
                    if stale.exists():
                        stale.unlink()
            else:
                return probe
        ledger = Ledger(path)
        ledger.init_session(
            self.session_id,
            self.interval.canonical(),
            self.config.interval_version,
            sample_digest,
            extra={
                "scale": self.config.scale,
                "top_k": self.config.top_k,
                "batch_size": self.config.batch_size,
            },
        )
        return ledger

    def _window_samples_from(self, min_line: int = 0) -> List[Sample]:
        out: List[Sample] = []
        for sample in iter_samples(self.config.sample_path):
            if sample.line_number <= min_line:
                continue
            if self.interval.contains(sample.timestamp_ms):
                out.append(sample)
        return out

    # ------------------------------------------------------------------- run
    def run(self, crash_after_batches: Optional[int] = None) -> dict:
        """Process the window in batches with atomic per-batch commits.

        ``crash_after_batches`` simulates ``SIGKILL``: the named number of
        batches commit durably, then :class:`ForcedKill` is raised and no
        further state is written.
        """

        digest = sample_stream_digest(
            self.config.sample_path, self.interval)
        self.ledger = self._open_ledger(digest)
        samples = self._window_samples_from()
        batch_size = self.config.batch_size

        start_seq = len(self.ledger.list_batches())
        committed_weight = (
            self.ledger.recovery_info().committed_weight if start_seq else 0)
        batch_seq = start_seq
        for offset in range(0, len(samples), batch_size):
            batch = samples[offset:offset + batch_size]
            deltas = _batch_deltas(batch)
            committed_weight += deltas["weight_added"]
            batch_seq += 1
            self.ledger.commit_batch(
                batch_seq=batch_seq,
                start_line=batch[0].line_number,
                end_line=batch[-1].line_number,
                sample_count=len(batch),
                weight_added=deltas["weight_added"],
                stack_deltas=deltas["stack_deltas"],
                frame_self_deltas=deltas["self_deltas"],
                frame_cumulative_deltas=deltas["cumulative_deltas"],
                committed_total_weight=committed_weight,
            )
            if (crash_after_batches is not None
                    and batch_seq - start_seq >= crash_after_batches):
                raise ForcedKill(
                    f"simulated kill after {batch_seq - start_seq} "
                    "committed batch(es); later samples are unknown"
                )

        return self._finalize(digest)

    def _finalize(self, digest: str) -> dict:
        assert self.ledger is not None
        # Reconstruct full input ordering for the adaptive engine selection.
        samples = self._window_samples_from()
        self.result = fold_mod.fold_adaptive(
            samples,
            max_unique_stacks=self.config.max_unique_stacks,
            max_total_weight=self.config.max_total_weight,
            force_path=self.config.force_path,
        )
        self.ranks = rank_frames(self.result, self.config.top_k)
        self.contexts = build_contexts(self.result, self.ranks)
        self.report = build_share_report(
            self.result, self.ranks, self.quant)
        self.ledger.replace_results(
            self.result, self.ranks, self.contexts, self.report,
            path_choice={
                "path_used": self.result.path_used,
                "path_reason": self.result.path_reason,
                "unique_stack_threshold":
                    self.result.unique_stack_threshold,
                "total_weight_threshold":
                    self.result.total_weight_threshold,
            },
        )
        snapshot = build_snapshot(
            session_id=self.session_id,
            sample_path=self.config.sample_path,
            interval=self.interval,
            interval_version=self.config.interval_version,
            result=self.result,
            ranks=self.ranks,
            contexts=self.contexts,
            report=self.report,
            force_path=self.config.force_path,
        )
        existing = read_snapshot(self.config.snapshot_dir)
        replaced, replacement_reason = should_replace(
            existing,
            sample_digest=digest,
            interval_version=self.config.interval_version,
        )
        write_snapshot(self.config.snapshot_dir, snapshot)
        self.snapshot_record = snapshot
        return {
            "session_id": self.session_id,
            "sample_digest": digest,
            "snapshot_replaced": replaced,
            "replacement_reason": replacement_reason,
            "batches": len(self.ledger.list_batches()),
            "path_used": self.result.path_used,
        }

    # -------------------------------------------------------------- recovery
    def recover(self) -> RecoveryInfo:
        """Inspect durable state; uncommitted-tail samples are unknown.

        Recovery only trusts fully committed batches. The rerun resumes from
        the line after the last committed cursor, so the incomplete batch is
        reprocessed rather than silently assumed.
        """

        if self.ledger is None:
            digest = sample_stream_digest(
                self.config.sample_path, self.interval)
            self.ledger = self._open_ledger(digest)
        info = self.ledger.recovery_info()
        return info

    def resume(self) -> dict:
        """Resume after a kill: rerun unknown tail samples, then finalize."""

        assert self.ledger is not None
        info = self.ledger.recovery_info()
        digest = sample_stream_digest(
            self.config.sample_path, self.interval)
        remaining = self._window_samples_from(info.last_committed_line)
        batch_size = self.config.batch_size
        batch_seq = info.recoverable_batches
        committed_weight = info.committed_weight
        for offset in range(0, len(remaining), batch_size):
            batch = remaining[offset:offset + batch_size]
            deltas = _batch_deltas(batch)
            committed_weight += deltas["weight_added"]
            batch_seq += 1
            self.ledger.commit_batch(
                batch_seq=batch_seq,
                start_line=batch[0].line_number,
                end_line=batch[-1].line_number,
                sample_count=len(batch),
                weight_added=deltas["weight_added"],
                stack_deltas=deltas["stack_deltas"],
                frame_self_deltas=deltas["self_deltas"],
                frame_cumulative_deltas=deltas["cumulative_deltas"],
                committed_total_weight=committed_weight,
            )
        return self._finalize(digest)

    # ---------------------------------------------------------------- replay
    def replay(self) -> dict:
        """Reload samples using recorded parameters and verify conclusions."""

        recorded = read_snapshot(
            self.config.snapshot_dir, verify_digest=False)
        if recorded is None:
            raise LedgerError("no snapshot available for replay")
        samples = self._window_samples_from()
        recomputed = fold_mod.fold_adaptive(
            samples,
            max_unique_stacks=recorded["result"]
            ["unique_stack_threshold"],
            max_total_weight=recorded["result"]
            ["total_weight_threshold"],
            force_path=recorded.get("force_path"),
        )
        recomputed_ranks = rank_frames(
            recomputed, len(recorded["ranks"]))
        outcome = verify_replay(recorded, recomputed, recomputed_ranks)
        return {
            "matched": outcome.matched,
            "checked_frames": outcome.checked_frames,
            "recorded_path": recorded["result"]["path_used"],
            "recomputed_path": recomputed.path_used,
        }

    def close(self) -> None:
        if self.ledger is not None:
            self.ledger.close()
            self.ledger = None
