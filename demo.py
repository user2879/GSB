"""End-to-end deterministic speaker-diarization example.

Run with:  python demo.py
Everything is generated locally; no network or external service is used.
"""
from __future__ import annotations

import os
import shutil
import sys
from dataclasses import asdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from diariz.audioio import read_wav
from diariz.cache import CacheStore, atomic_write_json
from diariz.config import (DiarConfig, default_config, write_config)
from diariz.errorprop import higher_tier_config, propagate
from diariz.faults import run_fault_injections
from diariz.features import plan_windows
from diariz.pipeline import process_audio, stable_audio_key
from diariz.snapshot import SnapshotManager
from diariz.synth import SynthSpec, synthesize, write_silence

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLES = os.path.join(ROOT, "samples")
OUTPUT = os.path.join(ROOT, "output")


def _result_to_json(result: dict) -> dict:
    payload = {
        "audio": result["audio"],
        "config_fingerprint": result["config_fingerprint"],
        "window_count": result["window_count"],
        "windows": [asdict(w) for w in result["windows"]],
        "embeddings": [
            {
                "index": e.index,
                "content_key": e.content_key,
                "accumulation_order": list(e.accumulation_order),
                "embedding": [float(v) for v in e.embedding],
            } for e in result["embeddings"]
        ],
        "quantized": [asdict(q) for q in result["quantized"]],
        "clustering": {
            "assignments": result["clustering"].assignments,
            "clusters": {cid: asdict(cluster)
                         for cid, cluster in result["clustering"].clusters.items()},
            "comparisons": [asdict(c) for c in result["clustering"].comparisons],
            "boundary_windows": result["clustering"].boundary_windows,
            "merges": [asdict(m) for m in result["clustering"].merges],
            "window_margin_q16": result["clustering"].window_margin_q16,
            "threshold_q16": result["clustering"].threshold_q16,
        },
        "segmentation": {
            "segments": [asdict(s) for s in result["segmentation"].segments],
            "dropped_short": result["segmentation"].dropped_short,
            "merged_gaps": result["segmentation"].merged_gaps,
            "frame_votes": [asdict(v)
                            for v in result["segmentation"].frame_votes],
        },
        "reuse_report": result["reuse_report"],
        "traces": result["traces"],
    }
    return payload


def main() -> int:
    if os.path.exists(OUTPUT):
        shutil.rmtree(OUTPUT)
    os.makedirs(SAMPLES, exist_ok=True)
    os.makedirs(OUTPUT, exist_ok=True)

    cfg = default_config()
    config_path = os.path.join(SAMPLES, "diar_config.json")
    write_config(config_path, cfg)

    # Deterministic synthetic long-form audio (3 speakers, pauses + overlap).
    spec = SynthSpec(duration_s=6.0, speaker_count=3, seed=20240601)
    wav_path = os.path.join(SAMPLES, "synthetic_3spk.wav")
    truth = synthesize(spec, wav_path)
    silence_path = os.path.join(SAMPLES, "synthetic_silence.wav")
    write_silence(silence_path, duration_s=2.5)
    atomic_write_json(os.path.join(OUTPUT, "synthesis_truth.json"),
                      {"segments": truth.segments, "overlaps": truth.overlaps,
                       "pcm_fingerprint": truth.pcm_fingerprint})

    wave = read_wav(wav_path)
    audio_key = stable_audio_key(wav_path)
    work_root = os.path.join(OUTPUT, "work")
    os.makedirs(work_root, exist_ok=True)
    cache = CacheStore(os.path.join(work_root, "shared_cache"))

    # Baseline run.
    base = process_audio(wav_path, cfg, work_root, cache=cache)
    atomic_write_json(os.path.join(OUTPUT, "baseline_result.json"),
                      _result_to_json(base))

    # Determinism rerun: identical window offsets and embeddings expected.
    rerun = process_audio(wav_path, cfg, work_root,
                          cache=CacheStore(os.path.join(work_root, "cache_rerun")))
    same_offsets = ([(w.start_sample, w.end_sample) for w in base["windows"]]
                    == [(w.start_sample, w.end_sample) for w in rerun["windows"]])
    same_embedding = all(
        np.array_equal(a.embedding, b.embedding)
        for a, b in zip(base["embeddings"], rerun["embeddings"]))

    # Single-window audio replacement: only dependent comparisons/segments move.
    windows = plan_windows(len(wave.samples), cfg)
    target_window = len(windows) // 2
    rng = np.random.default_rng(99)
    win_len = windows[target_window].end_sample - windows[target_window].start_sample
    replacement = rng.normal(0.0, 0.25, win_len)
    replaced = process_audio(wav_path, cfg, work_root, cache=cache,
                             window_audio_overrides={target_window: replacement})
    atomic_write_json(os.path.join(OUTPUT, "single_window_replace.json"),
                      _result_to_json(replaced))

    # Threshold change: quant embeddings reused, comparisons recomputed.
    threshold_cfg = DiarConfig(**{**cfg.to_dict(), "threshold": 0.80})
    threshold_run = process_audio(wav_path, threshold_cfg, work_root, cache=cache)
    atomic_write_json(os.path.join(OUTPUT, "threshold_change.json"),
                      _result_to_json(threshold_run))

    # Bit-width change: quantization and downstream stages recompute.
    width_cfg = cfg.with_quant(word_bits=32, frac_bits=12)
    width_run = process_audio(wav_path, width_cfg, work_root, cache=cache)

    # Two-tier error propagation and flip attribution.
    high_cfg = higher_tier_config(cfg)
    high_run = process_audio(wav_path, high_cfg, work_root,
                             cache=CacheStore(os.path.join(work_root, "cache_high")))
    diffs, attribution = propagate(
        {"quantized": base["quantized"], "clustering": base["clustering"],
         "segmentation": base["segmentation"]},
        {"quantized": high_run["quantized"], "clustering": high_run["clustering"],
         "segmentation": high_run["segmentation"]},
        cfg)
    atomic_write_json(os.path.join(OUTPUT, "error_propagation.json"), {
        "base_word_bits": cfg.word_bits,
        "high_word_bits": high_cfg.word_bits,
        "high_frac_bits": high_cfg.frac_bits,
        "window_diffs": [asdict(d) for d in diffs],
        "attribution": attribution,
    })

    # Snapshot commit and fault injection.
    snapshot = SnapshotManager(os.path.join(OUTPUT, "snapshots"))
    snapshot.begin()
    snapshot.stage_artifact("embed", f"{audio_key}.json",
                            [e.content_key for e in base["embeddings"]])
    manifest = snapshot.commit(cfg.fingerprint(), [audio_key],
                               {"embed": 1, "quant": 1, "compare": 1,
                                "segment": 1})
    fault_report = run_fault_injections(wav_path, wave.samples, cfg, work_root)
    atomic_write_json(os.path.join(OUTPUT, "fault_injection_report.json"),
                      fault_report)

    # Concurrency consistency: serial vs configured thread pool.
    serial = process_audio(wav_path, cfg, work_root,
                           cache=CacheStore(os.path.join(work_root, "cache_serial")))
    consistency = {
        "window_offsets_match": same_offsets,
        "embeddings_match": bool(same_embedding),
        "serial_assignments_match":
            serial["clustering"].assignments == base["clustering"].assignments,
        "serial_segments_match":
            [asdict(s) for s in serial["segmentation"].segments]
            == [asdict(s) for s in base["segmentation"].segments],
        "budget_sequence_embed":
            base["traces"]["embed"]["budget_deduction_order"]
            == serial["traces"]["embed"]["budget_deduction_order"],
    }

    summary = {
        "config": cfg.to_dict(),
        "config_path": config_path,
        "audio": {"path": wav_path, "samples": len(wave.samples),
                  "windows": base["window_count"],
                  "truth_segments": len(truth.segments),
                  "truth_overlaps": len(truth.overlaps)},
        "clusters": {cid: cluster.members
                     for cid, cluster in base["clustering"].clusters.items()},
        "boundary_windows": base["clustering"].boundary_windows,
        "segments": [asdict(s) for s in base["segmentation"].segments],
        "single_window_replacement": {
            "window": target_window,
            "reused": replaced["reuse_report"]["reused"],
            "recomputed": replaced["reuse_report"]["recomputed"],
        },
        "threshold_change_reuse": threshold_run["reuse_report"],
        "width_change_reuse": width_run["reuse_report"],
        "error_attribution": attribution["attributed_reasons"],
        "faults": fault_report,
        "consistency": consistency,
        "snapshot_manifest": manifest,
    }
    atomic_write_json(os.path.join(OUTPUT, "summary.json"), summary)

    print("Diarization demo finished.")
    print(f"  windows={base['window_count']} "
          f"clusters={len(base['clustering'].clusters)} "
          f"segments={len(base['segmentation'].segments)}")
    print(f"  determinism: offsets={same_offsets} embeddings={same_embedding}")
    print(f"  fault hits: "
          f"{[f['id'] for f in fault_report if f['hit']]}")
    print(f"  output directory: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
