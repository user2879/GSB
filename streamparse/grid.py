from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .errors import E_INPUT_INVALID, failure
from .fingerprint import content_hash, round_score
from .models import (
    Candidate,
    Dataset,
    DecodeConfig,
    GridFrame,
    OwnedFrame,
    ShardBuild,
    ShardRecord,
    Vocabulary,
)


def effective_config_for_session(
    base: DecodeConfig,
    overrides: Optional[Dict[str, Any]],
    injection_config_overrides: Optional[Dict[str, Any]] = None,
) -> DecodeConfig:
    config = base.with_overrides(overrides)
    if injection_config_overrides:
        # Injection overrides are trusted test fixtures, but still go through
        # the same typed validation path.
        config = config.with_overrides(injection_config_overrides)
    return config


def build_shard(
    session_id: str,
    record: ShardRecord,
    rows: Dict[int, Tuple[float, ...]],
    config: DecodeConfig,
    vocab: Vocabulary,
    *,
    injection_id: Optional[str] = None,
    injected_non_finite: bool = False,
    injected_frame_delta: int = 0,
) -> ShardBuild:
    """Validate one shard and convert its local rows into global frame cells.

    This function is deliberately pure so that it can run in a worker thread.
    Any failure is returned in the ShardBuild and never escapes the shard.
    """

    start_frame = record.start_frame
    end_frame = record.end_frame + injected_frame_delta
    frame_count = end_frame - start_frame + 1

    def fail(message: str, stage: str = "score_validation", details: Optional[Dict[str, Any]] = None) -> ShardBuild:
        return ShardBuild(
            session_id=session_id,
            shard_seq=record.shard_seq,
            revision=record.revision,
            start_frame=record.start_frame,
            end_frame=record.end_frame,
            frames=[],
            failure=failure(
                E_INPUT_INVALID,
                stage,
                message,
                session_id=session_id,
                shard_seq=record.shard_seq,
                injection_id=injection_id,
                details=details or {},
            ),
        )

    if frame_count <= 0:
        return fail("shard end frame is before its start frame", "session_validation")
    expected_frames = set(range(frame_count))
    actual_frames = set(rows)
    if actual_frames != expected_frames:
        return fail(
            "shard frame count does not match score table rows",
            details={
                "start_frame": record.start_frame,
                "end_frame": end_frame,
                "expected_local_frames": sorted(expected_frames),
                "actual_local_frames": sorted(actual_frames),
            },
        )

    frames: List[Tuple[int, Tuple[float, ...], bool]] = []
    for local in range(frame_count):
        values = tuple(rows[local])
        if len(values) != len(vocab):
            return fail(
                "emission score dimension does not match vocabulary size",
                details={"frame_in_shard": local, "expected": len(vocab), "actual": len(values)},
            )
        if injected_non_finite and local == 0:
            changed = list(values)
            changed[0] = math.nan
            values = tuple(changed)
        if any(not math.isfinite(value) for value in values):
            return fail(
                "emission score matrix contains a non-finite value",
                details={"frame_in_shard": local},
            )
        has_candidate = any(value > config.candidate_missing_below for value in values)
        frames.append((start_frame + local, values, not has_candidate))

    return ShardBuild(
        session_id=session_id,
        shard_seq=record.shard_seq,
        revision=record.revision,
        start_frame=record.start_frame,
        end_frame=end_frame,
        frames=frames,
        failure=None,
    )


def resolve_ownership(
    session_id: str,
    builds: Sequence[ShardBuild],
) -> Tuple[List[OwnedFrame], List[Dict[str, Any]]]:
    """Apply later-shard-wins ownership and return current + historical conflict rows."""

    by_frame: Dict[int, List[Tuple[Tuple[int, int], ShardBuild, int, Tuple[float, ...]]]] = {}
    for build in builds:
        if build.failure is not None:
            continue
        authority = (build.shard_seq, build.revision)
        for global_frame, values, _smoothed in build.frames:
            local = global_frame - build.start_frame
            by_frame.setdefault(global_frame, []).append((authority, build, local, values))

    owned: List[OwnedFrame] = []
    conflicts: List[Dict[str, Any]] = []
    for global_frame in sorted(by_frame):
        contenders = sorted(by_frame[global_frame], key=lambda item: item[0], reverse=True)
        authority, winner, local, values = contenders[0]
        covered = []
        for loser_authority, loser, loser_local, _ in contenders[1:]:
            conflict = {
                "global_frame": global_frame,
                "winning_shard_seq": winner.shard_seq,
                "winning_revision": winner.revision,
                "winning_frame_in_shard": local,
                "overwritten_shard_seq": loser.shard_seq,
                "overwritten_revision": loser.revision,
                "overwritten_frame_in_shard": loser_local,
                "overwritten_global_frame": global_frame,
            }
            covered.append(conflict)
            conflicts.append(conflict)
        owned.append(
            OwnedFrame(
                global_frame=global_frame,
                shard_seq=winner.shard_seq,
                revision=winner.revision,
                frame_in_shard=local,
                scores=tuple(values),
                covered=tuple(covered),
            )
        )

    conflicts.sort(
        key=lambda item: (
            item["global_frame"],
            item["winning_shard_seq"],
            item["winning_revision"],
            item["overwritten_shard_seq"],
            item["overwritten_revision"],
        )
    )
    return owned, conflicts


def build_grid(owned_frames: Sequence[OwnedFrame], config: DecodeConfig, vocab: Vocabulary) -> List[GridFrame]:
    grid: List[GridFrame] = []
    for owned in owned_frames:
        candidates: List[Candidate] = []
        for token_id, value in enumerate(owned.scores):
            if value > config.candidate_missing_below:
                candidates.append(
                    Candidate(
                        token_id=token_id,
                        token=vocab.token(token_id),
                        emission=float(value),
                    )
                )
        smoothed = not candidates
        smoothed_token_id: Optional[int] = None
        if smoothed:
            smoothed_token_id = vocab.smoothing_token_id
            candidates = [
                Candidate(
                    token_id=smoothed_token_id,
                    token=vocab.smoothing_token,
                    emission=float(config.smoothing_score),
                )
            ]
        candidates.sort(key=lambda item: (-item.emission, item.token_id))
        grid.append(
            GridFrame(
                global_frame=owned.global_frame,
                owner_shard=owned.shard_seq,
                owner_revision=owned.revision,
                frame_in_shard=owned.frame_in_shard,
                candidates=tuple(candidates),
                smoothed=smoothed,
                smoothed_token_id=smoothed_token_id,
            )
        )
    return grid


def ownership_fingerprint(session_id: str, builds: Sequence[ShardBuild]) -> str:
    return content_hash(
        {
            "session_id": session_id,
            "shards": [
                {
                    "shard_seq": build.shard_seq,
                    "revision": build.revision,
                    "start_frame": build.start_frame,
                    "end_frame": build.end_frame,
                    "frames": [frame for frame, _, _ in build.frames],
                }
                for build in sorted(builds, key=lambda item: item.sort_key)
                if build.failure is None
            ],
        }
    )


def grid_fingerprint(
    session_id: str,
    ownership_fp: str,
    vocab: Vocabulary,
    config: DecodeConfig,
    owned_frames: Sequence[OwnedFrame],
    grid: Sequence[GridFrame],
) -> str:
    return content_hash(
        {
            "session_id": session_id,
            "ownership_fp": ownership_fp,
            "vocab_fp": vocab.source_fingerprint,
            "grid_params": config.grid_fingerprint_payload(),
            "cards": [
                {
                    "global_frame": item.global_frame,
                    "owner": [item.owner_shard, item.owner_revision],
                    "candidates": [
                        [candidate.token_id, round_score(candidate.emission)] for candidate in item.candidates
                    ],
                    "smoothed": item.smoothed,
                }
                for item in grid
            ],
        }
    )


def stable_fingerprint(
    grid_fp: str,
    vocab: Vocabulary,
    config: DecodeConfig,
    stable_decisions: Sequence[Any] = (),
) -> str:
    # grid_fp is intentionally not hashed: appending frames changes the whole
    # grid key, but must not invalidate an unchanged certified prefix.  The
    # prefix content itself and the vocab/min-stability inputs are the stable
    # cache identity.
    return content_hash(
        {
            "vocab_fp": vocab.source_fingerprint,
            "min_stable_frames": config.min_stable_frames,
            "stable_margin": config.stable_margin,
            "stable_decisions": [
                [
                    item.global_frame,
                    item.token_id,
                    item.token,
                    round_score(item.score),
                    item.owner_shard,
                ]
                for item in stable_decisions
            ],
        }
    )


def tail_fingerprint(
    stable_fp: str,
    grid_fp: str,
    config: DecodeConfig,
    tail_decisions: Sequence[Any] = (),
) -> str:
    return content_hash(
        {
            "stable_fp": stable_fp,
            "grid_fp": grid_fp,
            "strategy_params": config.strategy_fingerprint_payload(),
            "tail_decisions": [
                [item.global_frame, item.token_id, item.token, round_score(item.score), item.owner_shard]
                for item in tail_decisions
            ],
        }
    )


def summarize_grid(grid: Sequence[GridFrame]) -> Dict[str, Any]:
    smoothed = [item.global_frame for item in grid if item.smoothed]
    return {
        "frame_count": len(grid),
        "candidate_counts": {str(item.global_frame): len(item.candidates) for item in grid},
        "smoothed_frames": smoothed,
        "first_frame": grid[0].global_frame if grid else None,
        "last_frame": grid[-1].global_frame if grid else None,
    }
