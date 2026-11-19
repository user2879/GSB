"""Repeatable fixed-sequence demonstration of the whole workbench.

The sequence is fully scripted: synthesise -> stream with block size 50 ms ->
switch block size to 100 ms mid-stream -> switch mode -> re-measure from a
frozen snapshot -> offline scan -> offline compression -> explicit tolerance
verdict. A scripted clock injects deterministic budget overruns so that the
degradation path is exercised without any wall-clock dependence.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np

from .audio_io import read_wav
from .config import WorkbenchConfig, ms_to_samples
from .engine import PlaybackEngine
from .errors import WorkbenchError
from .offline import (
    assert_conclusions_match,
    envelope_hash,
    run_offline_analysis,
    run_offline_compression,
    write_report,
)
from .snapshot import Snapshot, SnapshotStore
from .synth import render_material


class ScriptedClock:
    """Deterministic millisecond clock used to provoke budget degradation."""

    def __init__(self, schedule: list[float]) -> None:
        self._schedule = list(schedule)
        self._index = 0

    def __call__(self) -> float:
        value = self._schedule[min(self._index, len(self._schedule) - 1)]
        self._index += 1
        return value


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_demo_config() -> WorkbenchConfig:
    return WorkbenchConfig(block_ms=50)


def run_demo(output_dir: str, seed: int | None = None) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    config = build_demo_config()
    seed = config.seed if seed is None else seed
    wav_path = os.path.join(output_dir, "programme_5p1_48k_24bit.wav")
    label_path = os.path.join(output_dir, "labels.json")

    # Step 0: deterministic material generation.
    manifest = render_material(wav_path, label_path, seed)
    signal, sample_rate, channels = read_wav(wav_path)

    # Scripted elapsed times per block: two mark_start/mark_end pairs per block.
    # Blocks 3 and 4 overrun the 40 ms budget but stay below the 3-strike limit.
    block_count_50 = int(np.ceil(signal.shape[0] / ms_to_samples(50, sample_rate)))
    schedule: list[float] = []
    ticks = 0.0
    for block in range(block_count_50 + 8):
        start_tick = ticks
        ticks += 1.0
        end_tick = ticks
        if block in (3, 4):
            end_tick = start_tick + 55.0  # 55 ms > 40 ms budget
        schedule.extend([start_tick, end_tick])
    clock = ScriptedClock(schedule)

    engine = PlaybackEngine(config, clock=clock)
    streaming_blocks: list[dict] = []

    def _consume(block_ms: int, sample_limit: int | None = None) -> None:
        engine.set_block_ms(block_ms)
        size = ms_to_samples(block_ms, sample_rate)
        consumed = 0
        while engine._position < signal.shape[0]:
            if sample_limit is not None and consumed >= sample_limit:
                break
            nxt = min(size, signal.shape[0] - engine._position)
            chunk = signal[engine._position : engine._position + nxt]
            result, snapshot = engine.process_block(chunk)
            streaming_blocks.append(
                {
                    "chunk_index": result.chunk_index,
                    "block_ms": result.block_ms,
                    "start_sample": result.start_sample,
                    "end_sample": result.end_sample,
                    "painted": result.painted,
                    "snapshot_version": snapshot.version,
                    "loudness_lkfs": _safe(result.latest_loudness_lkfs),
                    "true_peak_dbtp": _safe(result.true_peak_dbtp),
                    "compressor_gain_db": round(result.compressor_gain_db, 9),
                }
            )
            consumed += nxt

    # Step 1: stream half the material in 50 ms blocks.
    half = signal.shape[0] // 2
    _consume(50, sample_limit=half)
    # Step 2: switch block length to 100 ms and finish the stream.
    _consume(100)
    # Step 3: mode switch freezes a snapshot.
    mode_snapshot = engine.set_mode("offline")
    # Step 3b: interactive segment-target adjustment (snapshotted, no reset).
    target_snapshot = engine.adjust_segment_target(segment_id=1, target_lkfs=-21.5)
    # Step 4: re-measure from the frozen snapshot (no state reset).
    remeasure_snapshot = engine.remeasure_from_snapshot()

    streaming_conclusions = engine.current_conclusions()
    stream_envelope = engine.gain_envelope()

    # Step 5: independent offline full scan in one call.
    offline = run_offline_analysis(signal, sample_rate)
    # Step 6: offline compression of the identical signal.
    processed, offline_compressor = run_offline_compression(signal, config.compressor, sample_rate)
    offline_envelope = offline_compressor.gain_envelope()

    # The streaming compressor was fed only complete control frames just like
    # the offline one (both consume the same hop grid), so lengths align.
    usable = min(stream_envelope.size, offline_envelope.size)
    verdict = assert_conclusions_match(
        streaming_conclusions,
        offline,
        stream_envelope[:usable],
        offline_envelope[:usable],
    )

    # Step 7: stale snapshot roll-back must be rejected.
    stale_rejected = _attempt_stale_rollback(engine.store)

    # Step 8: boundary-continuity evidence of the compressor envelope.
    boundary_samples = [
        rec["end_sample"] for rec in streaming_blocks if rec["end_sample"] < signal.shape[0]
    ]
    boundary_gains_stream = [
        engine.compressor.gain_at_sample(sample) for sample in boundary_samples
    ]
    boundary_gains_offline = [
        offline_compressor.gain_at_sample(sample) for sample in boundary_samples
    ]
    boundary_diffs = np.abs(
        np.asarray(boundary_gains_stream) - np.asarray(boundary_gains_offline)
    )
    increments = offline_compressor.frame_increments()
    continuity = {
        "block_boundary_count": len(boundary_samples),
        "max_stream_vs_offline_boundary_gain_diff_db": float(
            np.max(boundary_diffs) if boundary_diffs.size else 0.0
        ),
        "max_natural_frame_increment_db": float(
            np.max(increments) if increments.size else 0.0
        ),
        "max_allowed_boundary_diff_db": 1e-9,
        "pass": bool(
            (boundary_diffs.size == 0)
            or np.max(boundary_diffs) <= 1e-9
        ),
    }

    report = {
        "material_manifest": manifest,
        "wav_sha256": _file_sha256(wav_path),
        "labels_sha256": _file_sha256(label_path),
        "config": config.to_dict(),
        "streaming": {
            "conclusions": _stringify(streaming_conclusions),
            "block_size_schedule": [50, 100],
            "block_records": streaming_blocks,
            "mode_switch_snapshot_version": mode_snapshot.version,
            "mode_switch_coverage_s": list(mode_snapshot.coverage_s),
            "target_adjustment_snapshot_version": target_snapshot.version,
            "segment_targets": target_snapshot.payload["segment_targets"],
            "remeasure_snapshot_version": remeasure_snapshot.version,
            "envelope_hash": envelope_hash(stream_envelope),
        },
        "offline": {
            k: v for k, v in offline.items() if k != "meter_state"
        },
        "offline_envelope_hash": envelope_hash(offline_envelope),
        "compressor_continuity": continuity,
        "tolerance_verdict": verdict,
        "budget": engine.budget.to_dict(),
        "stale_snapshot_rejected": stale_rejected,
        "processed_signal_sha256": hashlib.sha256(
            np.asarray(processed, dtype=np.float64).tobytes()
        ).hexdigest(),
    }
    report_path = os.path.join(output_dir, "verification_report.json")
    write_report(report_path, report)
    return report


def _attempt_stale_rollback(store: SnapshotStore) -> dict:
    current = store.current
    stale = Snapshot(
        version=current.version - 1,
        mode="realtime",
        start_sample=0,
        end_sample=0,
        sample_rate=current.sample_rate,
        block_ms=current.block_ms,
        payload={"reason": "stale_rollback_attempt"},
    )
    try:
        store.commit(stale)
    except WorkbenchError as exc:
        return {"rejected": True, "code": exc.code, "detail": exc.detail}
    return {"rejected": False}


def _safe(value: float):
    if np.isneginf(value) or np.isnan(value):
        return None
    return float(value)


def _stringify(data: dict) -> dict:
    return {k: (_safe(v) if isinstance(v, float) else v) for k, v in data.items()}
