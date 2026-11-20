"""Chunk streaming: per-chunk decode, overlap time alignment and merging."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .audio import ChunkPlan, SAMPLE_RATE
from .config import AcousticConfig, PhonemeMap, StreamConfig, round_sig
from .decoder import WordToken, build_transition_matrix, labels_to_tokens, viterbi_decode
from .features import FrameFeatures, compute_features


def frame_index_of_sample(sample: int, frame_shift: int) -> int:
    return int(sample // frame_shift)


def chunk_frame_span(chunk: ChunkPlan, frame_shift: int,
                     max_frame: Optional[int] = None) -> Tuple[int, int]:
    """Return [first_frame, last_frame_inclusive) global frame indices of a chunk."""
    first = frame_index_of_sample(chunk.start_sample, frame_shift)
    last_exclusive = frame_index_of_sample(chunk.end_sample - 1, frame_shift) + 1
    if max_frame is not None:
        last_exclusive = min(last_exclusive, max_frame)
    return first, last_exclusive


@dataclass
class ChunkResult:
    index: int
    chunk: ChunkPlan
    features: Optional[FrameFeatures] = None
    frame_span: Tuple[int, int] = (0, 0)
    labels: Optional[List[int]] = None
    tokens: List[WordToken] = field(default_factory=list)
    failed: bool = False
    failure_code: Optional[str] = None
    failure_message: Optional[str] = None


def decode_one_chunk(clip_samples: np.ndarray, chunk: ChunkPlan, acoustic: AcousticConfig,
                     phoneme_map: PhonemeMap, stream: StreamConfig,
                     backoff_scale: float) -> ChunkResult:
    part = clip_samples[chunk.start_sample:chunk.end_sample]
    features = compute_features(part, acoustic, phoneme_map, stream.significant_digits)
    frame_shift = features.plan.frame_shift
    first_frame, last_exclusive = chunk_frame_span(
        chunk, frame_shift,
        max_frame=frame_index_of_sample(clip_samples.shape[0] - 1, frame_shift) + 1)
    transition = build_transition_matrix(acoustic, phoneme_map.words, backoff_scale)
    labels = viterbi_decode(features.emissions, phoneme_map.words, transition)
    tokens = labels_to_tokens(labels, features.emissions, phoneme_map.words,
                              first_frame, stream.significant_digits)
    return ChunkResult(index=chunk.index, chunk=chunk, features=features,
                       frame_span=(first_frame, last_exclusive), labels=labels, tokens=tokens)


@dataclass
class MergedTranscript:
    frame_labels: List[int]
    frame_owner: List[int]
    tokens: List[WordToken]
    conflicts: List[Dict[str, Any]]
    n_frames: int


def merge_chunks(results: Sequence[ChunkResult], words: Sequence[str],
                 frame_shift: int, total_samples: int,
                 significant_digits: int) -> MergedTranscript:
    """Align chunk labels on global frame timestamps; earliest chunk owns a frame."""
    n_frames = frame_index_of_sample(max(total_samples - 1, 0), frame_shift) + 1
    frame_labels = [-1] * n_frames
    frame_owner = [-1] * n_frames
    conflicts: List[Dict[str, Any]] = []
    for result in sorted(results, key=lambda item: item.index):
        if result.failed or result.labels is None:
            continue
        first_frame, last_exclusive = result.frame_span
        for local_index, label in enumerate(result.labels):
            global_frame = first_frame + local_index
            if global_frame >= n_frames:
                break
            if frame_labels[global_frame] == -1:
                frame_labels[global_frame] = label
                frame_owner[global_frame] = result.index
            elif frame_labels[global_frame] != label:
                conflicts.append({
                    "frame_index": global_frame,
                    "start_ms": global_frame * frame_shift * 1000 // SAMPLE_RATE,
                    "owner_chunk": frame_owner[global_frame],
                    "conflicting_chunk": result.index,
                    "kept_word": words[frame_labels[global_frame]],
                    "dropped_word": words[label],
                })
    valid = [label for label in frame_labels if label != -1]
    if valid:
        for index, label in enumerate(frame_labels):
            if label == -1:
                frame_labels[index] = valid[0]
                frame_owner[index] = frame_owner[index] if frame_owner[index] != -1 else -1
    tokens = _collapse_frames(frame_labels, words, significant_digits)
    conflicts.sort(key=lambda item: (item["frame_index"], item["conflicting_chunk"]))
    return MergedTranscript(frame_labels=frame_labels, frame_owner=frame_owner,
                            tokens=tokens, conflicts=conflicts, n_frames=n_frames)


def _collapse_frames(frame_labels: List[int], words: Sequence[str],
                     significant_digits: int) -> List[WordToken]:
    tokens: List[WordToken] = []
    if not frame_labels:
        return tokens
    run_start = 0
    for pos in range(1, len(frame_labels) + 1):
        if pos == len(frame_labels) or frame_labels[pos] != frame_labels[run_start]:
            word_index = frame_labels[run_start]
            if word_index >= 0:
                tokens.append(WordToken(
                    word=words[word_index],
                    start_frame=run_start,
                    end_frame=pos - 1,
                    score=float(round_sig(1.0, significant_digits)),
                ))
            run_start = pos
    return tokens


def re_score_tokens(tokens: List[WordToken], chunk_results: Sequence[ChunkResult],
                    words: Sequence[str], significant_digits: int) -> List[WordToken]:
    """Recompute each token's mean frame emission score from owning chunk emissions."""
    word_index = {word: idx for idx, word in enumerate(words)}
    by_index = {result.index: result for result in chunk_results}
    rescored: List[WordToken] = []
    for token in tokens:
        values: List[float] = []
        for frame in range(token.start_frame, token.end_frame + 1):
            owner = _owner_for_frame(chunk_results, frame)
            if owner is None:
                continue
            result = by_index[owner]
            first_frame = result.frame_span[0]
            local_frame = frame - first_frame
            if 0 <= local_frame < result.features.emissions.shape[0]:
                values.append(float(result.features.emissions[local_frame, word_index[token.word]]))
        score = float(np.mean(values)) if values else token.score
        rescored.append(WordToken(word=token.word, start_frame=token.start_frame,
                                  end_frame=token.end_frame,
                                  score=float(round_sig(score, significant_digits))))
    return rescored


def _owner_for_frame(results: Sequence[ChunkResult], frame: int) -> Optional[int]:
    for result in sorted(results, key=lambda item: item.index):
        if result.failed or result.labels is None:
            continue
        first_frame, last_exclusive = result.frame_span
        if first_frame <= frame < last_exclusive:
            return result.index
    return None
