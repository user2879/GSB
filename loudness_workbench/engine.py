"""Workbench engine orchestrating streaming metering, compression and snapshots.

The engine owns the persistent measurement/compressor state. Changing the
audio block length only changes how many samples are fed per call; filter
state, integration-window residuals and compressor smoothing all survive.
Snapshots are immutable and versioned, and every UI-visible number comes from
an accepted snapshot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .bs1770 import StreamingLoudnessMeter
from .budget import BudgetMonitor
from .compressor import LogDomainCompressor
from .config import (
    ABSOLUTE_GATE_LKFS,
    RELATIVE_GATE_OFFSET_LU,
    WorkbenchConfig,
    validate_block_ms,
)
from .errors import E_AUDIO_FORMAT, WorkbenchError
from .offline import envelope_hash
from .snapshot import Snapshot, SnapshotStore


@dataclass
class BlockResult:
    chunk_index: int
    block_ms: int
    start_sample: int
    end_sample: int
    window_blocks_completed: int
    latest_loudness_lkfs: float
    true_peak_dbtp: float
    compressor_gain_db: float
    painted: bool


class PlaybackEngine:
    def __init__(self, config: WorkbenchConfig, clock=None) -> None:
        self.config = config
        self.block_ms = config.block_ms
        self.mode = "realtime"
        self.meter = StreamingLoudnessMeter(config.sample_rate)
        self.compressor = LogDomainCompressor(config.compressor, config.sample_rate)
        self.budget = BudgetMonitor(config.ui_refresh_budget_ms, clock=clock)
        self.store = SnapshotStore()
        self._chunk_index = 0
        self._position = 0
        self.segment_targets: dict[int, float] = {}
        self._freeze_snapshot("initial")

    def adjust_segment_target(self, segment_id: int, target_lkfs: float) -> Snapshot:
        """Record an interactive target adjustment for a labelled segment.

        The adjustment is part of the immutable snapshot history so it can be
        replayed deterministically; already accumulated measurement state is
        not cleared.
        """
        if not (-80.0 <= target_lkfs <= 0.0):
            raise WorkbenchError(
                E_AUDIO_FORMAT,
                f"segment target {target_lkfs} LKFS outside [-80, 0]",
            )
        self.segment_targets[int(segment_id)] = float(target_lkfs)
        return self._freeze_snapshot(
            "target_adjustment",
            adjustment={"segment_id": int(segment_id),
                        "target_lkfs": float(target_lkfs)},
        )

    # ------------------------------------------------------------------
    def validate_signal(self, signal: np.ndarray) -> None:
        if signal.ndim != 2 or signal.shape[1] != self.config.channel_count:
            raise WorkbenchError(
                E_AUDIO_FORMAT,
                f"signal channels={0 if signal.ndim != 2 else signal.shape[1]}, "
                f"expected {self.config.channel_count}",
            )

    def set_block_ms(self, block_ms: int) -> None:
        """Change chunk size without resetting any measurement state."""
        validate_block_ms(block_ms)
        self.block_ms = int(block_ms)
        self._freeze_snapshot("block_length_change")

    def set_mode(self, mode: str) -> Snapshot:
        if mode not in ("realtime", "offline"):
            raise WorkbenchError(E_AUDIO_FORMAT, f"unknown mode {mode!r}")
        self.mode = mode
        return self._freeze_snapshot("mode_switch")

    # ------------------------------------------------------------------
    def process_block(self, chunk: np.ndarray) -> tuple[BlockResult, Snapshot]:
        self.validate_signal(chunk)
        start = self._position
        self.budget.mark_start()
        window_blocks = self.meter.process(chunk)
        self.compressor.process(chunk)
        partial = self.meter.finalize(
            ABSOLUTE_GATE_LKFS, RELATIVE_GATE_OFFSET_LU
        )
        latest = window_blocks[-1].loudness_lkfs if window_blocks else (
            self.meter.blocks[-1].loudness_lkfs if self.meter.blocks else float("-inf")
        )
        self._position += chunk.shape[0]
        result = BlockResult(
            chunk_index=self._chunk_index,
            block_ms=self.block_ms,
            start_sample=start,
            end_sample=self._position,
            window_blocks_completed=len(window_blocks),
            latest_loudness_lkfs=latest,
            true_peak_dbtp=self.meter.true_peak.peak_dbtp(),
            compressor_gain_db=self.compressor.envelope_db,
            painted=False,
        )
        result.painted = self.budget.mark_end()
        self._chunk_index += 1
        snapshot = self._freeze_snapshot("block", partial=partial, result=result)
        return result, snapshot

    # ------------------------------------------------------------------
    def remeasure_from_snapshot(self, extra_signal: np.ndarray | None = None) -> Snapshot:
        """Re-measurement continues from the frozen snapshot rather than zeroing.

        The snapshot's accumulated meter state is reinstalled (proving the
        lineage), then any additional samples are appended. With no extra
        signal the new snapshot simply re-derives the same conclusions under a
        new version number.
        """
        base = self.store.current
        if base is None or "meter_state" not in base.payload:
            raise WorkbenchError(
                E_AUDIO_FORMAT, "no measurement snapshot available to re-measure from"
            )
        self.meter.set_state(base.payload["meter_state"])
        self.compressor.set_state(base.payload["compressor_state"])
        if extra_signal is not None:
            self.validate_signal(extra_signal)
            self.meter.process(extra_signal)
            self.compressor.process(extra_signal)
            self._position += extra_signal.shape[0]
        partial = self.meter.finalize(ABSOLUTE_GATE_LKFS, RELATIVE_GATE_OFFSET_LU)
        return self._freeze_snapshot("remeasure", partial=partial)

    # ------------------------------------------------------------------
    def current_conclusions(self) -> dict:
        gated = self.meter.finalize(ABSOLUTE_GATE_LKFS, RELATIVE_GATE_OFFSET_LU)
        return {
            "integrated_lkfs": gated.integrated_lkfs,
            "loudness_range_lu": gated.lra_lu,
            "true_peak_dbtp": self.meter.true_peak.peak_dbtp(),
            "blocks_total": gated.blocks_total,
            "blocks_below_absolute": gated.blocks_below_absolute,
            "blocks_below_relative": gated.blocks_below_relative,
            "blocks_kept": gated.blocks_kept,
        }

    def gain_envelope(self) -> np.ndarray:
        return self.compressor.gain_envelope()

    def _freeze_snapshot(self, reason: str, partial=None,
                         result: BlockResult | None = None,
                         adjustment: dict | None = None) -> Snapshot:
        gated = partial or self.meter.finalize(
            ABSOLUTE_GATE_LKFS, RELATIVE_GATE_OFFSET_LU
        )
        payload = {
            "reason": reason,
            "mode": self.mode,
            "block_ms": self.block_ms,
            "position_sample": self._position,
            "position_s": round(self._position / self.config.sample_rate, 6),
            "integrated_lkfs": gated.integrated_lkfs,
            "relative_gate_lkfs": gated.relative_gate_lkfs,
            "loudness_range_lu": gated.lra_lu,
            "true_peak_dbtp": self.meter.true_peak.peak_dbtp(),
            "blocks_total": gated.blocks_total,
            "blocks_below_absolute": gated.blocks_below_absolute,
            "blocks_below_relative": gated.blocks_below_relative,
            "blocks_kept": gated.blocks_kept,
            "compressor_gain_db": self.compressor.envelope_db,
            "envelope_hash": envelope_hash(self.compressor.gain_envelope()),
            "budget": self.budget.to_dict(),
            "segment_targets": dict(sorted(self.segment_targets.items())),
            "meter_state": self.meter.get_state(),
            "compressor_state": self.compressor.get_state(),
        }
        if adjustment is not None:
            payload["adjustment"] = adjustment
        if result is not None:
            payload["last_block"] = {
                "chunk_index": result.chunk_index,
                "start_sample": result.start_sample,
                "end_sample": result.end_sample,
                "painted": result.painted,
            }
        snapshot = Snapshot(
            version=self.store.next_version(),
            mode=self.mode,
            start_sample=0,
            end_sample=self._position,
            sample_rate=self.config.sample_rate,
            block_ms=self.block_ms,
            payload=payload,
        )
        return self.store.commit(snapshot)
