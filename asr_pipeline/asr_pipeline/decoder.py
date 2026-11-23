"""Log-domain Viterbi decoding on the character lattice, budget-aware."""

import math

import numpy as np

from .errors import E_SHARD_BUDGET_EXHAUSTED, PipelineError

INIT_LOG = 0.0


def _transition(prev_char, next_char, score_params):
    bigram = score_params["bigram_probs"].get(prev_char)
    if bigram is not None and next_char in bigram:
        return math.log(bigram[next_char])
    unigram = score_params["unigram_probs"][next_char]
    return math.log(unigram) + score_params["backoff_weight"]


def lm_matrix(score_params):
    vocab = score_params["vocab"]
    size = len(vocab)
    matrix = np.zeros((size, size), dtype=np.float64)
    for i, prev_char in enumerate(vocab):
        for j, next_char in enumerate(vocab):
            matrix[i, j] = _transition(prev_char, next_char, score_params)
    return matrix


def min_points_per_frame(score_params):
    """Every frame evaluates one emission candidate per vocab character."""
    return len(score_params["vocab"])


def _checkpoint(trellis, backptr, next_frame, consumed):
    return {
        "next_frame": int(next_frame),
        "consumed": int(consumed),
        "trellis": trellis.tolist(),
        "backptr": backptr.tolist(),
    }


def run_viterbi(emissions, point_costs, score_params, allowance,
                checkpoint=None, injection=None, global_limited=False):
    """Decode frames under a hard point allowance.

    Returns the best path, per-word spans/scores, consumed points and either a
    resume checkpoint (when the allowance runs out) or ``None`` on completion.
    """
    vocab = score_params["vocab"]
    size = len(vocab)
    n_frames = emissions.shape[0]
    trans = lm_matrix(score_params)
    min_cost = int(point_costs.min()) if n_frames else size
    forced_small = bool(injection and injection.get("type") == "quota_below_min_points")
    if checkpoint is None and forced_small and allowance < min_cost:
        # Only the deliberately injected undersized quota is a hard failure.
        raise PipelineError(
            E_SHARD_BUDGET_EXHAUSTED,
            f"shard allowance {allowance} below minimum points per frame {min_cost}",
            stage="decode", injection_id=injection.get("id"),
        )

    if checkpoint is None:
        trellis = np.full((n_frames, size), -math.inf, dtype=np.float64)
        backptr = np.full((n_frames, size), -1, dtype=np.int64)
        start_frame = 0
        consumed = 0
        if allowance < min_cost:
            # A not-yet-started shard starved by the global budget stays
            # pending instead of becoming a permanent failure.
            empty = _checkpoint(trellis, backptr, 0, 0)
            return None, 0, 0, empty
    else:
        # Restore the full trellis/backpointer history so the final traceback
        # is byte-identical to a single uninterrupted Viterbi run.
        trellis = np.asarray(checkpoint["trellis"], dtype=np.float64)
        backptr = np.asarray(checkpoint["backptr"], dtype=np.int64)
        start_frame = int(checkpoint["next_frame"])
        consumed = int(checkpoint["consumed"])
        if start_frame >= n_frames:
            result, _ = _traceback(trellis, backptr, emissions, vocab, score_params)
            return result, 0, consumed, None
    # ``allowance`` is the incremental budget for this call; the cumulative
    # cap makes first-run and resume guards compare the same quantity.
    start_cost = consumed
    budget_cap = consumed + allowance
    if checkpoint is not None and budget_cap < consumed + int(point_costs[start_frame]):
        # Resumed shard with no room for its next frame: stay pending.
        return None, 0, consumed, _checkpoint(trellis, backptr, start_frame, consumed)

    pending = False
    frame_idx = start_frame
    while frame_idx < n_frames:
        cost = int(point_costs[frame_idx])
        # Hard per-frame guard: the shard stops before crossing the allowance,
        # so the global budget ledger can never go negative.
        if consumed + cost > budget_cap:
            pending = True
            break
        if frame_idx == 0:
            # Initial frame uses unigram priors whether this is a fresh run or
            # a resume from an empty (all -inf) checkpoint.
            trellis[0] = (
                np.asarray([math.log(score_params["unigram_probs"][c]) for c in vocab])
                + emissions[0]
            )
            backptr[0] = -1
        else:
            scores = trellis[frame_idx - 1][:, None] + trans + emissions[frame_idx]
            best_prev = np.argmax(scores, axis=0)
            backptr[frame_idx] = best_prev
            trellis[frame_idx] = scores[best_prev, np.arange(size)]
        consumed += cost
        frame_idx += 1

    if pending or (global_limited and frame_idx < n_frames):
        incremental = consumed - start_cost
        return None, incremental, consumed, _checkpoint(trellis, backptr, frame_idx, consumed)

    result, _ = _traceback(trellis, backptr, emissions, vocab, score_params)
    incremental = consumed - start_cost
    return result, incremental, consumed, None


def _traceback(trellis, backptr, emissions, vocab, score_params):
    n_frames = trellis.shape[0]
    if n_frames == 0:
        return {"words": [], "path": []}, []
    last_state = int(np.argmax(trellis[-1]))
    path_states = [0] * n_frames
    path_states[-1] = last_state
    state = last_state
    for frame_idx in range(n_frames - 1, 0, -1):
        state = int(backptr[frame_idx, state])
        path_states[frame_idx - 1] = state

    words = []
    start = 0
    for idx in range(1, n_frames + 1):
        if idx == n_frames or path_states[idx] != path_states[start]:
            char = vocab[path_states[start]]
            end_frame = idx - 1
            word_score = float(np.sum(emissions[start: idx, path_states[start]]))
            lm_score = 0.0
            if start > 0:
                lm_score = _transition(
                    vocab[path_states[start - 1]], char, score_params
                )
            words.append({
                "char": char,
                "frame_start": int(start),
                "frame_end": int(end_frame),
                "emission_score": word_score,
                "lm_score": float(lm_score),
                "total_score": float(word_score + lm_score),
            })
            start = idx
    return {"words": words,
            "path": [vocab[state] for state in path_states]}, path_states
