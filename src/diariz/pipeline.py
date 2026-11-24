"""End-to-end diarization pipeline orchestration."""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import cache as cache_mod
from .audioio import read_wav, truncate_wav, write_wav
from .clustering import incrementally_cluster
from .concurrency import Task, run_tasks
from .config import DiarConfig, SAMPLE_RATE
from .errors import AudioFormatError, ConfigInvalidError
from .features import (WindowEmbedding, WindowInfo, compute_frame_features,
                       embed_window, fit_scaler, plan_windows,
                       projection_matrix, raw_feature_dim, window_content_key,
                       _window_raw)
from .quantize import (QuantizedWindow, integer_similarity,
                       quantize_embedding, threshold_to_q16)
from .segmentation import build_segments, vote_frames
from .snapshot import SnapshotManager


@dataclass
class AudioOutcome:
    audio: str
    ok: bool
    error_code: str = ""
    message: str = ""
    window_count: int = 0
    cluster_count: int = 0
    segment_count: int = 0


def stable_audio_key(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def _serialize_embedding(item: WindowEmbedding) -> Dict[str, Any]:
    # Content-only payload: window index/position are rebound by the caller so
    # that distinct windows sharing identical content can reuse one cache entry.
    return {
        "content_key": item.content_key,
        "accumulation_order": list(item.accumulation_order),
        "raw_feature": list(item.raw_feature),
        "embedding": [float(v) for v in item.embedding],
    }


def _bind_embedding(payload: Dict[str, Any], info: WindowInfo) -> WindowEmbedding:
    return WindowEmbedding(
        index=info.index, info=info, content_key=payload["content_key"],
        accumulation_order=tuple(payload["accumulation_order"]),
        raw_feature=tuple(payload["raw_feature"]),
        embedding=np.asarray(payload["embedding"], dtype=np.float64),
    )


def _serialize_quant(item: QuantizedWindow) -> Dict[str, Any]:
    # Content-only payload: window index is rebound by the caller so multiple
    # windows with identical quantized content can share one persistent entry.
    return {
        "word_bits": item.word_bits,
        "frac_bits": item.frac_bits,
        "step": item.step,
        "integers": list(item.integers),
        "residuals": list(item.residuals),
        "dequantized": list(item.dequantized),
        "max_residual_dim": item.max_residual_dim,
        "max_residual_abs": item.max_residual_abs,
        "quant_key": item.quant_key,
    }


def _bind_quant(payload: Dict[str, Any], index: int) -> QuantizedWindow:
    return QuantizedWindow(
        index=index, word_bits=payload["word_bits"],
        frac_bits=payload["frac_bits"], step=payload["step"],
        integers=tuple(payload["integers"]),
        residuals=tuple(payload["residuals"]),
        dequantized=tuple(payload["dequantized"]),
        max_residual_dim=payload["max_residual_dim"],
        max_residual_abs=payload["max_residual_abs"],
        quant_key=payload["quant_key"],
    )


def _load_or_compute(cache: cache_mod.CacheStore, stage: str, key: str,
                     factory) -> Any:
    cached = cache.get(stage, key)
    if cached is not None:
        return cached, True
    payload = factory()
    cache.put(stage, key, payload)
    return payload, False


def process_audio(path: str, cfg: DiarConfig, work_root: str,
                  cache: Optional[cache_mod.CacheStore] = None,
                  window_audio_overrides: Optional[Dict[int, np.ndarray]] = None,
                  ) -> Dict[str, Any]:
    audio_key = stable_audio_key(path)
    wave = read_wav(path)
    samples = list(wave.samples)
    if window_audio_overrides:
        hop = int(round(cfg.hop_ms * SAMPLE_RATE / 1000.0))
        for wid, replacement in window_audio_overrides.items():
            start = wid * hop
            block = np.clip(np.asarray(replacement, dtype=np.float64), -1.0, 1.0)
            end = min(len(samples), start + block.size)
            samples[start:end] = block[:end - start].tolist()

    windows = plan_windows(len(samples), cfg)
    frame_features = compute_frame_features(samples, cfg)
    projection = projection_matrix(raw_feature_dim(), cfg)
    config_fp = cfg.fingerprint()
    proj_fp = cfg.projection_fp()
    own_cache = cache is None
    if own_cache:
        cache = cache_mod.CacheStore(os.path.join(work_root, "cache_root", audio_key))
    # Each invocation gets an isolated reuse ledger while the persistent
    # key/value store itself is shared across runs.
    cache.report = cache_mod.ReuseReport()

    raw_by_window: Dict[int, np.ndarray] = {}
    for info in windows:
        order = sorted(info.frame_indices)
        if order:
            raw_by_window[info.index] = _window_raw(frame_features, order)
        else:
            raw_by_window[info.index] = np.zeros(raw_feature_dim(),
                                                  dtype=np.float64)
    scaler = fit_scaler([raw_by_window[w.index] for w in windows])

    embed_tasks = [Task.make(audio_key, w.index, cache_mod.STAGE_EMBED, w)
                   for w in windows]

    def embed_worker(task: Task) -> Dict[str, Any]:
        info: WindowInfo = task.payload
        content_key = window_content_key(samples, info)
        e_key = cache_mod.embed_key(content_key, proj_fp)
        payload, _ = _load_or_compute(
            cache, cache_mod.STAGE_EMBED, e_key,
            lambda: _serialize_embedding(
                embed_window(raw_by_window[info.index], scaler, projection,
                             info, content_key)))
        return payload

    embed_results, embed_trace = run_tasks(
        embed_tasks, embed_worker, cfg.concurrency,
        budget=max(1, len(embed_tasks) * 2))
    embeddings: List[WindowEmbedding] = [
        _bind_embedding(embed_results[t.sort_key], t.payload)
        for t in sorted(embed_tasks, key=lambda t: t.sort_key)
    ]

    # Stage 3: fixed point quantization.
    quantized: List[QuantizedWindow] = []
    quant_keys: Dict[int, str] = {}
    for embedding in embeddings:
        e_key = cache_mod.embed_key(embedding.content_key, proj_fp)
        q_key = cache_mod.quant_key(e_key, config_fp, cfg.word_bits,
                                    cfg.frac_bits)
        quant_keys[embedding.index] = q_key
        payload, _ = _load_or_compute(
            cache, cache_mod.STAGE_QUANT, q_key,
            lambda emb=embedding, key=q_key: _serialize_quant(
                quantize_embedding(emb.index, emb.embedding, cfg, key)))
        quantized.append(_bind_quant(payload, embedding.index))

    # Stage 4: pairwise similarity comparisons, concurrently.
    threshold_q16 = threshold_to_q16(cfg.threshold)
    compare_tasks: List[Task] = []
    pair_index: Dict[Tuple[int, int], str] = {}
    window_ids = [q.index for q in sorted(quantized, key=lambda q: q.index)]
    id_to_quant = {q.index: q for q in quantized}
    for pos_left in range(len(window_ids)):
        for pos_right in range(pos_left + 1, len(window_ids)):
            left = window_ids[pos_left]
            right = window_ids[pos_right]
            lk = quant_keys[left]
            rk = quant_keys[right]
            c_key = cache_mod.compare_key(lk, rk, threshold_q16)
            pair_index[(left, right)] = c_key
            compare_tasks.append(Task.make(
                audio_key, right, f"{cache_mod.STAGE_COMPARE}-{left:06d}",
                (left, right)))

    def compare_worker(task: Task) -> Dict[str, Any]:
        left, right = task.payload
        c_key = pair_index[(left, right)]
        left_q = id_to_quant[left]
        right_q = id_to_quant[right]
        left_key = quant_keys[left]
        right_key = quant_keys[right]

        def compute():
            sim = integer_similarity(left_q.integers, right_q.integers)
            return {
                "left_window": left, "right_window": right,
                "left_quant_key": left_key,
                "right_quant_key": right_key,
                "similarity_q16": sim, "threshold_q16": threshold_q16,
                "accepted": sim >= threshold_q16,
                "boundary": sim == threshold_q16,
            }

        payload, _ = _load_or_compute(
            cache, cache_mod.STAGE_COMPARE, c_key, compute)
        return payload

    compare_budget = max(1, len(compare_tasks) * 2)
    compare_results, compare_trace = run_tasks(
        compare_tasks, compare_worker, cfg.concurrency, budget=compare_budget)
    pairwise: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for task in sorted(compare_tasks, key=lambda t: t.sort_key):
        payload = compare_results[task.sort_key]
        pairwise[(payload["left_window"], payload["right_window"])] = payload

    # Incremental clustering stays serial; ordering is deterministic by design.
    clustering = incrementally_cluster(quantized, cfg, quant_keys)
    # Annotate comparisons with persistent keys.
    for comparison in clustering.comparisons:
        pair = (min(comparison.left_window, comparison.right_window),
                max(comparison.left_window, comparison.right_window))
        comparison.compare_key = pair_index.get(pair, "")

    # Stage 5: frame votes and segmentation.
    votes = vote_frames(windows, len(frame_features), clustering, cfg)
    segmentation = build_segments(votes, windows, clustering, cfg, len(samples))
    compare_keys = sorted({c.compare_key for c in clustering.comparisons
                           if c.compare_key})
    seg_key = cache_mod.segment_key(compare_keys, config_fp, cfg.word_bits,
                                    cfg.frac_bits)
    seg_payload, _ = _load_or_compute(
        cache, cache_mod.STAGE_SEGMENT, seg_key,
        lambda: {
            "segments": [asdict(seg) for seg in segmentation.segments],
            "dropped_short": segmentation.dropped_short,
            "merged_gaps": segmentation.merged_gaps,
            "frame_votes": [asdict(v) for v in segmentation.frame_votes],
        })

    if own_cache:
        cache.flush_index()

    return {
        "audio": audio_key,
        "path": path,
        "config_fingerprint": config_fp,
        "window_count": len(windows),
        "windows": windows,
        "embeddings": embeddings,
        "quantized": quantized,
        "quant_keys": quant_keys,
        "pairwise": pairwise,
        "clustering": clustering,
        "segmentation": segmentation,
        "reuse_report": cache.report.to_dict(),
        "traces": {
            "embed": asdict(embed_trace),
            "compare": asdict(compare_trace),
        },
        "segment_key": seg_key,
    }


def discover_wavs(input_dir: str) -> List[str]:
    if not os.path.isdir(input_dir):
        return []
    return sorted(os.path.join(input_dir, name)
                  for name in os.listdir(input_dir)
                  if name.lower().endswith(".wav"))


def run_batch(input_dir: str, cfg: DiarConfig, work_root: str,
              snapshot: Optional[SnapshotManager] = None) -> Dict[str, Any]:
    paths = discover_wavs(input_dir)
    cache = cache_mod.CacheStore(os.path.join(work_root, "shared_cache"))
    outcomes: List[AudioOutcome] = []
    detailed: Dict[str, Any] = {}
    for path in paths:
        key = stable_audio_key(path)
        try:
            result = process_audio(path, cfg, work_root, cache=cache)
            outcomes.append(AudioOutcome(
                audio=key, ok=True,
                window_count=result["window_count"],
                cluster_count=len(result["clustering"].clusters),
                segment_count=len(result["segmentation"].segments),
            ))
            detailed[key] = result
        except AudioFormatError as exc:
            outcomes.append(AudioOutcome(
                audio=key, ok=False, error_code=exc.code, message=exc.message))
        except ConfigInvalidError:
            raise
        except Exception as exc:  # isolate per-audio failures
            outcomes.append(AudioOutcome(
                audio=key, ok=False, error_code="E_INTERNAL", message=str(exc)))
    cache.flush_index()

    if snapshot is not None:
        snapshot.begin()
        for key, result in detailed.items():
            snapshot.stage_artifact(
                "embed", f"{key}.json",
                [_serialize_embedding(e) for e in result["embeddings"]])
        manifest = snapshot.commit(
            config_hash=cfg.fingerprint(),
            audio_cursor=[o.audio for o in outcomes if o.ok],
            artifact_versions={stage: 1 for stage in cache_mod.STAGES})
    else:
        manifest = {}

    return {
        "outcomes": [asdict(o) for o in outcomes],
        "detailed": detailed,
        "cache_report": cache.report.to_dict(),
        "manifest": manifest,
    }


def write_fault_injection_cases(fault_dir: str, base_samples: Sequence[float],
                                valid_path: str) -> List[Dict[str, Any]]:
    """Materialize realistic fault inputs and describe each injection id."""
    os.makedirs(fault_dir, exist_ok=True)
    cases = []

    # INJ-01: valid WAV container, but a non-finite value is injected after
    # decode (int16 PCM cannot itself encode NaN/Inf). The pipeline loader is
    # re-run through a dedicated decoder hook to hit the E_AUDIO_FORMAT path.
    nonfinite_path = os.path.join(fault_dir, "inj01_nonfinite.wav")
    write_wav(nonfinite_path, base_samples)
    cases.append({"id": "INJ-01", "type": "non_finite_sample",
                  "path": nonfinite_path, "note": "post-decode NaN injection"})

    # INJ-02: all-silence audio.
    silence_path = os.path.join(fault_dir, "inj02_silence.wav")
    write_wav(silence_path, [0.0] * len(base_samples))
    cases.append({"id": "INJ-02", "type": "all_silence", "path": silence_path})

    # INJ-03: out-of-range threshold (separate config object).
    cases.append({"id": "INJ-03", "type": "threshold_out_of_range",
                  "path": valid_path, "threshold": 1.7})

    # INJ-04: truncated data block while header keeps declared size.
    truncated_path = os.path.join(fault_dir, "inj04_truncated.wav")
    write_wav(truncated_path, base_samples)
    truncate_wav(truncated_path, os.path.getsize(truncated_path) - 256)
    cases.append({"id": "INJ-04", "type": "truncated_data",
                  "path": truncated_path})
    return cases
