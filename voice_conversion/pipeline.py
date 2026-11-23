"""End-to-end VC pipeline: load, analyze, map, synthesize, assemble, report."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .assembly import assemble_segments
from .audio import read_wav, write_wav
from .config import PipelineConfig, fingerprint_json, fingerprint_pcm
from .dsp import analyze_segment, map_segment, synthesize_segment
from .errors import FailureLog
from .executor import run_tasks
from .ledger import (LedgerEntry, ReuseLedger, analysis_key, assembly_key,
                     mapping_key, segment_content_fingerprint, synthesis_key)
from .snapshot import SnapshotStore
from .v1 import parse_v1
from .v2 import parse_v2


def load_exchange(path: str | Path, version: str, config: PipelineConfig,
                  failures: FailureLog):
    if version == "v1":
        return parse_v1(path, config, failures)
    if version == "v2":
        return parse_v2(path, config, failures)
    raise ValueError(f"unknown exchange version {version!r}")


def _analysis_worker(pcm, seg, config, failures, store, keys):
    a_key = keys[seg.index]["analysis"]
    if store.has_product("analysis", a_key):
        return store.read_json("analysis", a_key), a_key, True
    result = analyze_segment(pcm, seg, config, failures)
    if result is None:
        return None, a_key, False
    store.write_staged_json("analysis", a_key, result)
    return result, a_key, False


def _mapping_worker(analysis, seg, config, store, keys):
    m_key = keys[seg.index]["mapping"]
    if store.has_product("mapping", m_key):
        return store.read_json("mapping", m_key), m_key, True
    result = map_segment(analysis, seg, config)
    store.write_staged_json("mapping", m_key, result)
    return result, m_key, False


def _synthesis_worker(analysis, mapping, seg, config, store, keys):
    s_key = keys[seg.index]["synthesis"]
    product = store.product_path("synthesis", s_key, ".pcm")
    if product.exists():
        return np.frombuffer(product.read_bytes(), dtype="<i2").copy(), s_key, True
    pcm = synthesize_segment(analysis, mapping, seg, config)
    store.write_staged_bytes("synthesis", s_key, pcm.tobytes())
    return pcm, s_key, False


def run_pipeline(*, exchange_path: str | Path, version: str,
                 source_wav: str | Path, config: PipelineConfig,
                 snapshot_dir: str | Path, output_wav: str | Path,
                 full_rebuild: bool = False,
                 max_workers: int | None = None) -> dict:
    """Execute the pipeline and return the deterministic diagnostic report."""
    failures = FailureLog()
    store = SnapshotStore(snapshot_dir)
    if full_rebuild:
        store.reset()
    else:
        store.ensure()

    # Stage 1: load + validate the exchange file and source audio.
    segments = load_exchange(exchange_path, version, config, failures)
    source_pcm, sample_rate = read_wav(source_wav)
    if sample_rate != config.sample_rate:
        raise ValueError("source WAV sample rate does not match config")
    audio_fp = fingerprint_pcm(source_pcm)
    exchange_fp = fingerprint_json(
        [segment_content_fingerprint(s) for s in segments])
    config_fp = config.fingerprint()
    structural_fp = config.structural_fingerprint()
    manifest = store.verify_resume(structural_fp, exchange_fp,
                                   audio_fp, failures)

    # v1 workspaces may carry a downgrade sidecar from a prior v2 -> v1 step.
    sidecar_stats: dict = {}
    if version == "v1":
        sidecar_path = Path(exchange_path).parent / "active.sidecar.json"
        if sidecar_path.exists():
            import json
            doc = json.loads(sidecar_path.read_text(encoding="utf-8"))
            entries = doc.get("segments", [])
            sidecar_stats = {
                "schema": doc.get("schema", ""),
                "segment_count": len(entries),
                "with_start_residual": sum(
                    1 for e in entries if e.get("start_residual")),
                "with_end_residual": sum(
                    1 for e in entries if e.get("end_residual")),
                "with_discarded_passthrough": sum(
                    1 for e in entries if e.get("discarded_passthrough")),
            }

    ledger = ReuseLedger(Path(snapshot_dir) / "ledger.json")
    if manifest is not None:
        if not ledger.load():
            raise RuntimeError("resumable manifest without reuse ledger")

    # Precompute every stage key from content fingerprints.
    keys: dict[int, dict] = {}
    for seg in segments:
        a_key = analysis_key(seg, config, audio_fp)
        m_key = mapping_key(seg, config, a_key)
        s_key = synthesis_key(seg, config, m_key)
        keys[seg.index] = {
            "analysis": a_key, "mapping": m_key, "synthesis": s_key,
        }

    # The execution width must never change content fingerprints or hashes.
    worker_count = config.workers if max_workers is None \
        else min(config.workers, max_workers)

    # Stage 2: acoustic analysis (bounded thread pool, stable ordering).
    analysis_tasks = [
        (seg.index, "analysis", seg) for seg in segments
    ]

    def analysis_task(seg_index, _stage, seg):
        return _analysis_worker(source_pcm, seg, config, failures, store, keys)

    analysis_results = run_tasks(analysis_tasks, analysis_task, worker_count)
    store.commit_stage("analysis")

    # Stage 3: fixed-point timbre mapping.
    valid_pairs = [
        (seg, result[0], result[2])
        for seg, result in zip(segments, analysis_results)
        if result is not None and result[0] is not None
    ]
    mapping_tasks = [(seg.index, "mapping", analysis, seg)
                     for seg, analysis, _a_reused in valid_pairs]

    def mapping_task(seg_index, _stage, analysis, seg):
        return _mapping_worker(analysis, seg, config, store, keys)

    mapping_results = run_tasks(mapping_tasks, mapping_task, worker_count)
    store.commit_stage("mapping")

    # Stage 4: per-segment synthesis, strictly isolated per segment.
    synth_tasks = [(seg.index, "synthesis", analysis, mapping, seg)
                   for (seg, analysis, _a_reused), (mapping, _key, _reused)
                   in zip(valid_pairs, mapping_results)]

    def synthesis_task(seg_index, _stage, analysis, mapping, seg):
        return _synthesis_worker(analysis, mapping, seg, config, store, keys)

    synth_results = run_tasks(synth_tasks, synthesis_task, worker_count)
    store.commit_stage("synthesis")

    # Reuse ledger and per-segment report rows.
    ledger.prune({seg.index for seg in segments})
    segment_rows, segment_pcms, frame_starts, synth_fps = [], {}, {}, []
    reused = recomputed = 0
    for (seg, analysis, a_reused), (mapping, m_key, m_reused), \
            (pcm, s_key, s_reused) in zip(
                valid_pairs, mapping_results, synth_results):
        a_key = keys[seg.index]["analysis"]
        previous = ledger.entries.get(seg.index)
        # Segment-level reuse means the final per-segment PCM cache key is
        # unchanged; upstream stages may still have been replayed internally.
        was_reused = previous is not None and \
            previous.synthesis_key == s_key and s_reused
        entry = LedgerEntry(
            segment=seg.index,
            analysis_key=a_key,
            mapping_key=m_key,
            synthesis_key=s_key,
            source_frame_set=list(analysis["source_frame_set"]),
            parameter_set=list(mapping["parameter_set"]),
            content_fingerprint=segment_content_fingerprint(seg),
        )
        ledger.remember(entry)
        # Periodic durable checkpoints let a restart resume a long run.
        processed = len(segment_rows) + 1
        if processed % max(1, config.snapshot_commit_interval) == 0:
            ledger.save()
        reused += int(was_reused)
        recomputed += int(not was_reused)
        segment_pcms[seg.index] = pcm
        frame_starts[seg.index] = int(round(seg.start_frame))
        synth_fps.append(s_key)
        segment_rows.append({
            "segment": seg.index,
            "frame_count": analysis["frame_count"],
            "timbre": seg.timbre,
            "intensity": seg.intensity,
            "mapping": {
                "intensity_q": mapping["intensity_q"],
                "f0_shift_q": mapping["f0_shift_q"],
                "formant_shift_q": mapping["formant_shift_q"],
                "bend_q": mapping["bend_q"],
                "gain_factor_q": mapping["gain_factor_q"],
                "mapped_f0_q": mapping["mapped_f0_q"],
            },
            "cache_keys": keys[seg.index],
            "reused": was_reused,
            "stage_reuse": {
                "analysis": a_reused,
                "mapping": m_reused,
                "synthesis": s_reused,
            },
            "source_frame_set": entry.source_frame_set,
            "mean_f0_hz": analysis["mean_f0_hz"],
            "mean_rms": analysis["mean_rms"],
        })

    # Stage 5: total assembly with deterministic crossfades.
    assembled = assemble_segments(segment_pcms, frame_starts, config)
    final_key = assembly_key(
        [seg for seg, _analysis, _a_reused in valid_pairs], config, synth_fps)
    output_fp = write_wav(output_wav, assembled, config.sample_rate)
    ledger.assembly_key = final_key
    ledger.save()

    manifest_doc = {
        "config_hash": structural_fp,
        "full_config_hash": config_fp,
        "exchange_fingerprint": exchange_fp,
        "audio_fingerprint": audio_fp,
        "output_fingerprint": output_fp,
        "assembly_key": final_key,
        "stage_versions": {
            "load": exchange_fp,
            "analysis": fingerprint_json(
                sorted((r[1] for r in analysis_results if r[0] is not None))),
            "mapping": fingerprint_json(sorted(r[1] for r in mapping_results)),
            "synthesis": fingerprint_json(sorted(synth_fps)),
            "assembly": final_key,
        },
    }
    store.publish_manifest(manifest_doc)

    return {
        "exchange_version": version,
        "segment_count": len(segments),
        "valid_segment_count": len(valid_pairs),
        "output_fingerprint": output_fp,
        "assembly_key": final_key,
        "reused_segments": reused,
        "recomputed_segments": recomputed,
        "reuse_invalidation": {
            "reused": [row["segment"] for row in segment_rows if row["reused"]],
            "recomputed": [row["segment"] for row in segment_rows
                           if not row["reused"]],
        },
        "segments": segment_rows,
        "failures": failures.to_list(),
        "failure_count": len(failures),
        "sidecar_downgrade": sidecar_stats,
        "fingerprints": {
            "config": structural_fp,
            "full_config": config_fp,
            "exchange": exchange_fp,
            "audio": audio_fp,
            "output": output_fp,
        },
    }
