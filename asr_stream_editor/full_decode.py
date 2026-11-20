"""Independent one-shot full decode used as the reference oracle.

This module deliberately does not reuse the incremental engine: it runs the
same feature, Viterbi, merge and stability math once over the whole clip in a
single thread, so end-to-end equality tests compare two independent paths.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .audio import AudioClip, plan_chunks, plan_frames
from .config import AcousticConfig, PhonemeMap, StreamConfig
from .stability import evaluate_stability, split_tokens
from .streaming import ChunkResult, decode_one_chunk, merge_chunks, re_score_tokens


def full_decode(clip: AudioClip, samples, phoneme_map: PhonemeMap,
                acoustic: AcousticConfig, stream: StreamConfig,
                overlap_ms: int, min_consistent_frames: int,
                backoff_scale: float = 1.0) -> Dict[str, Any]:
    effective = StreamConfig(
        chunk_ms=stream.chunk_ms, overlap_ms=overlap_ms,
        min_consistent_frames=min_consistent_frames,
        redecode_budget_ms=stream.redecode_budget_ms,
        concurrency=1, concurrency_limit=stream.concurrency_limit,
        snapshot_commit_interval_chunks=stream.snapshot_commit_interval_chunks,
        precision=stream.precision,
        simulated_task_cost_ms=stream.simulated_task_cost_ms)
    chunks = plan_chunks(samples.shape[0], effective.chunk_ms, effective.overlap_ms)
    results: List[ChunkResult] = []
    failures: List[Dict[str, Any]] = []
    for chunk in chunks:
        try:
            results.append(decode_one_chunk(samples, chunk, acoustic, phoneme_map,
                                            effective, backoff_scale))
        except Exception as exc:
            from .errors import E_DECODE_FAILED, FailureRecord
            failures.append(FailureRecord(
                source=clip.source, stage="decode", code=E_DECODE_FAILED,
                exception_type=type(exc).__name__, message=str(exc),
                chunk_index=chunk.index).to_dict())
    frame_plan = plan_frames(samples.shape[0], acoustic.frame_length_ms,
                             acoustic.frame_shift_ms)
    merged = merge_chunks(results, phoneme_map.words, frame_plan.frame_shift,
                          samples.shape[0], stream.significant_digits)
    merged.tokens = re_score_tokens(merged.tokens, results, phoneme_map.words,
                                    stream.significant_digits)
    stability = evaluate_stability(results, merged.n_frames, min_consistent_frames)
    prefix, tail = split_tokens(merged.tokens, stability.stable_until_frame)
    return {
        "source": clip.source,
        "audio_fingerprint": clip.fingerprint,
        "transcript": [token.to_dict(acoustic.frame_shift_ms)
                       for token in merged.tokens],
        "text": "".join(token.word for token in merged.tokens),
        "frame_labels": list(merged.frame_labels),
        "frame_owner": list(merged.frame_owner),
        "stable_prefix": [token.to_dict(acoustic.frame_shift_ms) for token in prefix],
        "stable_prefix_text": "".join(token.word for token in prefix),
        "mutable_tail": [token.to_dict(acoustic.frame_shift_ms) for token in tail],
        "stable_until_frame": stability.stable_until_frame,
        "boundary_runs": stability.boundary_runs,
        "alignment_conflicts": list(merged.conflicts),
        "failures": sorted(failures, key=lambda item: (item["chunk_index"],
                                                       item["stage"], item["code"])),
        "chunk_count": len(chunks),
    }
