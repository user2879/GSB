"""Log-domain word-level Viterbi decoding with bigram LM and backoff."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np

from .config import AcousticConfig, round_sig


@dataclass(frozen=True)
class WordToken:
    word: str
    start_frame: int
    end_frame: int  # inclusive
    score: float    # mean frame emission logprob, rounded artifact value

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    def to_dict(self, frame_shift_ms: int) -> Dict[str, Any]:
        return {
            "word": self.word,
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "start_ms": self.start_frame * frame_shift_ms,
            "end_ms": (self.end_frame + 1) * frame_shift_ms,
            "score": self.score,
        }


def _safe_log(value: float) -> float:
    return float(np.log(value)) if value > 0.0 else -1e9


def build_transition_matrix(acoustic: AcousticConfig, words: Sequence[str],
                            backoff_scale: float) -> np.ndarray:
    """Log transition cost matrix transition[prev, next], shape (W, W)."""
    n_words = len(words)
    matrix = np.zeros((n_words, n_words), dtype=np.float64)
    for i, prev in enumerate(words):
        backoff_weight = min(1.0, max(0.0, acoustic.backoff.get(prev, 0.5) * backoff_scale))
        for j, nxt in enumerate(words):
            key = f"{prev}|{nxt}"
            if key in acoustic.bigram:
                matrix[i, j] = _safe_log(acoustic.bigram[key])
            else:
                matrix[i, j] = _safe_log(backoff_weight) + _safe_log(1.0 / n_words)
    return matrix


def viterbi_decode(emissions: np.ndarray, words: Sequence[str],
                   transition: np.ndarray) -> List[int]:
    """Return the best word index for every frame (frame labels)."""
    n_frames, n_words = emissions.shape
    if n_frames == 0:
        return []
    trellis = np.full((n_frames, n_words), -np.inf, dtype=np.float64)
    backpointer = np.zeros((n_frames, n_words), dtype=np.int64)
    # uniform initial state
    trellis[0] = emissions[0] + _safe_log(1.0 / n_words)
    for frame in range(1, n_frames):
        candidates = trellis[frame - 1][:, None] + transition
        best_prev = np.argmax(candidates, axis=0)
        backpointer[frame] = best_prev
        trellis[frame] = candidates[best_prev, np.arange(n_words)] + emissions[frame]
    labels = np.zeros(n_frames, dtype=np.int64)
    labels[-1] = int(np.argmax(trellis[-1]))
    for frame in range(n_frames - 1, 0, -1):
        labels[frame - 1] = backpointer[frame, labels[frame]]
    return [int(label) for label in labels]


def labels_to_tokens(labels: List[int], emissions: np.ndarray,
                     words: Sequence[str], frame_offset: int,
                     significant_digits: int) -> List[WordToken]:
    """Collapse consecutive identical labels into one word token."""
    if not labels:
        return []
    tokens: List[WordToken] = []
    run_start = 0
    for pos in range(1, len(labels) + 1):
        if pos == len(labels) or labels[pos] != labels[run_start]:
            word_index = labels[run_start]
            end = pos - 1
            mean_score = float(emissions[run_start:pos, word_index].mean())
            tokens.append(WordToken(
                word=words[word_index],
                start_frame=frame_offset + run_start,
                end_frame=frame_offset + end,
                score=float(round_sig(mean_score, significant_digits)),
            ))
            run_start = pos
    return tokens


def decode_chunk(emissions: np.ndarray, words: Sequence[str], acoustic: AcousticConfig,
                 frame_offset: int, significant_digits: int,
                 backoff_scale: float = 1.0) -> List[WordToken]:
    transition = build_transition_matrix(acoustic, words, backoff_scale)
    labels = viterbi_decode(emissions, words, transition)
    return labels_to_tokens(labels, emissions, words, frame_offset, significant_digits)
