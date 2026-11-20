from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from .models import Candidate, DecodeConfig, DecodeResult, GridFrame, TokenDecision, Vocabulary


def _candidate_total(candidate: Candidate, previous: str, vocab: Vocabulary, config: DecodeConfig) -> float:
    return candidate.emission + config.lm_weight * vocab.bigram_score(
        previous,
        candidate.token,
        config.lm_backoff_score,
    )


def _rank_candidates(
    frame: GridFrame,
    previous: str,
    vocab: Vocabulary,
    config: DecodeConfig,
) -> List[Tuple[float, Candidate]]:
    ranked = [(_candidate_total(candidate, previous, vocab, config), candidate) for candidate in frame.candidates]
    ranked.sort(key=lambda item: (-item[0], item[1].token_id))
    return ranked


def greedy_decode_suffix(
    grid: Sequence[GridFrame],
    start: int,
    vocab: Vocabulary,
    config: DecodeConfig,
    previous_token: str,
) -> Tuple[TokenDecision, ...]:
    previous = previous_token
    decisions: List[TokenDecision] = []
    for frame in grid[start:]:
        ranked = _rank_candidates(frame, previous, vocab, config)
        best_score, best = ranked[0]
        if len(ranked) > 1:
            second_score = ranked[1][0]
        else:
            second_score = best_score - 1_000_000.0
        decisions.append(
            TokenDecision(
                global_frame=frame.global_frame,
                token_id=best.token_id,
                token=best.token,
                score=best_score,
                second_score=second_score,
                margin=best_score - second_score,
                smoothed=frame.smoothed,
                owner_shard=frame.owner_shard,
            )
        )
        previous = best.token
    return tuple(decisions)


def greedy_decode(grid: Sequence[GridFrame], vocab: Vocabulary, config: DecodeConfig) -> Tuple[TokenDecision, ...]:
    return greedy_decode_suffix(grid, 0, vocab, config, vocab.start_token)


def _viterbi_window_first_choice(
    grid: Sequence[GridFrame],
    start: int,
    end: int,
    previous_token: str,
    vocab: Vocabulary,
    config: DecodeConfig,
) -> Candidate:
    """Run Viterbi over an inclusive window and return the first-frame token."""

    first_frame = grid[start]
    first_states: Dict[int, float] = {}
    for candidate in first_frame.candidates:
        total = _candidate_total(candidate, previous_token, vocab, config)
        # If duplicate token ids cannot occur in a grid card, but min() makes
        # this robust to future extensions.
        current = first_states.get(candidate.token_id)
        if current is None or total > current:
            first_states[candidate.token_id] = total

    backpointers: List[Dict[int, int]] = []
    state_scores = first_states
    for position in range(start + 1, end + 1):
        frame = grid[position]
        new_scores: Dict[int, float] = {}
        back: Dict[int, int] = {}
        previous_frame_candidates = {item.token_id: item for item in grid[position - 1].candidates}
        for candidate in frame.candidates:
            best_prev_id: Optional[int] = None
            best_total = float("-inf")
            for prev_id, prev_score in state_scores.items():
                prev_token = previous_frame_candidates[prev_id].token
                total = prev_score + _candidate_total(candidate, prev_token, vocab, config)
                if total > best_total or (total == best_total and (best_prev_id is None or prev_id < best_prev_id)):
                    best_total = total
                    best_prev_id = prev_id
            if best_prev_id is not None:
                new_scores[candidate.token_id] = best_total
                back[candidate.token_id] = best_prev_id
        backpointers.append(back)
        state_scores = new_scores

    final_id = min(state_scores, key=lambda token_id: (-state_scores[token_id], token_id))
    # Walk backwards to the first frame.  backpointers[0] maps frame
    # start+1 -> start, etc.
    current = final_id
    for back in reversed(backpointers):
        current = back[current]
    return next(item for item in first_frame.candidates if item.token_id == current)


def lookahead_decode_suffix(
    grid: Sequence[GridFrame],
    start: int,
    vocab: Vocabulary,
    config: DecodeConfig,
    previous_token: str,
) -> Tuple[TokenDecision, ...]:
    decisions: List[TokenDecision] = []
    previous = previous_token
    for position in range(start, len(grid)):
        end = min(len(grid) - 1, position + max(1, config.lookahead_frames))
        chosen = _viterbi_window_first_choice(grid, position, end, previous, vocab, config)
        frame = grid[position]
        ranked = _rank_candidates(frame, previous, vocab, config)
        chosen_score = next(score for score, candidate in ranked if candidate.token_id == chosen.token_id)
        second_score = ranked[1][0] if len(ranked) > 1 else chosen_score - 1_000_000.0
        decisions.append(
            TokenDecision(
                global_frame=frame.global_frame,
                token_id=chosen.token_id,
                token=chosen.token,
                score=chosen_score,
                second_score=second_score,
                margin=chosen_score - second_score,
                smoothed=frame.smoothed,
                owner_shard=frame.owner_shard,
            )
        )
        previous = chosen.token
    return tuple(decisions)


def compute_stable_length(greedy: Sequence[TokenDecision], total_frames: int, config: DecodeConfig) -> int:
    if not greedy:
        return 0
    run = 0
    for decision in greedy:
        if decision.smoothed:
            break
        if decision.margin + 1e-12 < config.stable_margin:
            break
        run += 1
    if run < config.min_stable_frames:
        return 0
    # A frame is certified only when at least min_stable_frames consistent
    # frames are available from that point.  Consequently the newest N-1
    # frames remain in the mutable tail under every strategy.
    stable_limit = max(0, total_frames - config.min_stable_frames + 1)
    return min(run, stable_limit)


def decode_grid(grid: Sequence[GridFrame], vocab: Vocabulary, config: DecodeConfig) -> DecodeResult:
    greedy = greedy_decode(grid, vocab, config)
    stable_length = compute_stable_length(greedy, len(grid), config)

    if config.strategy == "greedy":
        active = greedy
    else:
        prefix = greedy[:stable_length]
        if stable_length == 0:
            previous = vocab.start_token
        else:
            previous = prefix[-1].token
        suffix = lookahead_decode_suffix(grid, stable_length, vocab, config, previous)
        active = tuple(prefix) + tuple(suffix)

    agreement = 0
    for left, right in zip(greedy, active):
        if left.token_id != right.token_id:
            break
        agreement += 1

    return DecodeResult(
        decisions=tuple(active),
        greedy_decisions=tuple(greedy),
        stable_length=stable_length,
        strategy=config.strategy,
        lookahead_frames=config.lookahead_frames,
        strategy_agreement_prefix=agreement,
    )


def decode_grid_reusing_prefix(
    grid: Sequence[GridFrame],
    vocab: Vocabulary,
    config: DecodeConfig,
    previous: DecodeResult,
) -> DecodeResult:
    """Decode only frames after a previously certified stable prefix."""

    stable_length = min(previous.stable_length, len(grid))
    stable_prefix = tuple(previous.decisions[:stable_length])
    greedy_prefix = tuple(previous.greedy_decisions[:stable_length])
    if stable_length:
        previous_token = greedy_prefix[-1].token
    else:
        previous_token = vocab.start_token
    greedy_suffix = greedy_decode_suffix(grid, stable_length, vocab, config, previous_token)
    greedy = tuple(greedy_prefix) + tuple(greedy_suffix)

    if config.strategy == "greedy":
        active = greedy
    else:
        lookahead_suffix = lookahead_decode_suffix(
            grid,
            stable_length,
            vocab,
            config,
            previous_token,
        )
        active = tuple(stable_prefix) + tuple(lookahead_suffix)

    agreement = 0
    for left, right in zip(greedy, active):
        if left.token_id != right.token_id:
            break
        agreement += 1
    return DecodeResult(
        decisions=tuple(active),
        greedy_decisions=tuple(greedy),
        stable_length=stable_length,
        strategy=config.strategy,
        lookahead_frames=config.lookahead_frames,
        strategy_agreement_prefix=agreement,
        reused_prefix_frames=stable_length,
        decode_mode="tail_reuse",
    )
