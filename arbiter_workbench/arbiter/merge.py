"""Deterministic cross-environment merge for the three arbitration policies.

Every ordering decision uses explicit tuple keys; no set/dict iteration order
influents the output. Environment participation is always rendered in the
view's declared environment order, which is itself validated to be unique.
"""
from __future__ import annotations

from dataclasses import dataclass

from .fixedpoint import FixedPoint, normalize_weights, trunc_div_toward_zero
from .sample import Observation


@dataclass(frozen=True)
class EnvVote:
    env_id: str
    revision: int
    ts_ms: int
    q_value: int          # quantized head value
    raw_value: str
    weight: int
    chain_digest: str


@dataclass(frozen=True)
class MergeResult:
    key: str
    policy: str
    authority_q: int
    adopted_envs: tuple[str, ...]
    rejected_envs: tuple[str, ...]
    tie_break: str
    evidence: dict


def _head_per_env(
    env_order: tuple[str, ...],
    observations_by_env: dict[str, list[Observation]],
    fp: FixedPoint,
    weights: dict[str, int],
    digests: dict[str, dict[str, str]],
    key: str,
) -> list[EnvVote]:
    votes: list[EnvVote] = []
    for env_id in env_order:
        chain = [
            obs for obs in observations_by_env.get(env_id, ()) if obs.key == key
        ]
        if not chain:
            continue
        chain.sort(key=lambda o: (o.revision, o.ts_ms, o.instance_id))
        head = chain[-1]
        quantized = fp.quantize(head.raw_value)
        digest = ""
        if env_id in digests and key in digests[env_id]:
            digest = digests[env_id][key]["digest"]
        votes.append(
            EnvVote(
                env_id=env_id,
                revision=head.revision,
                ts_ms=head.ts_ms,
                q_value=quantized.q,
                raw_value=head.raw_value,
                weight=weights[env_id],
                chain_digest=digest,
            )
        )
    return votes


def merge_latest(votes: list[EnvVote], key: str) -> MergeResult:
    # Highest revision wins; deterministic tie break by ts, then env id.
    winner = max(votes, key=lambda v: (v.revision, v.ts_ms, v.env_id))
    adopted = tuple(
        v.env_id
        for v in sorted(votes, key=lambda x: x.env_id)
        if (v.revision, v.ts_ms) == (winner.revision, winner.ts_ms)
        and v.q_value == winner.q_value
    )
    rejected = tuple(v.env_id for v in sorted(votes, key=lambda x: x.env_id) if v.env_id not in adopted)
    return MergeResult(
        key=key,
        policy="latest",
        authority_q=winner.q_value,
        adopted_envs=adopted,
        rejected_envs=rejected,
        tie_break="revision>ts_ms>env_id",
        evidence={"winning_revision": winner.revision, "winning_ts_ms": winner.ts_ms},
    )


def merge_weighted_majority(
    votes: list[EnvVote], key: str, fp: FixedPoint
) -> MergeResult:
    # Group equal quantized values; accumulate declared integer weights.
    groups: dict[int, list[EnvVote]] = {}
    for vote in votes:
        groups.setdefault(vote.q_value, []).append(vote)
    weight_totals = {
        value: sum(v.weight for v in members) for value, members in groups.items()
    }
    # Winner: greatest total weight; stable tie break by value then first env.
    winner_value = max(
        sorted(groups),  # sorted values make tie resolution fully deterministic
        key=lambda value: (
            weight_totals[value],
            -min(v.revision for v in groups[value]),
        ),
    )
    winner_members = sorted(groups[winner_value], key=lambda v: v.env_id)
    adopted = tuple(v.env_id for v in winner_members)
    rejected = tuple(
        v.env_id
        for v in sorted(votes, key=lambda x: x.env_id)
        if v.env_id not in adopted
    )
    raw_weights = [v.weight for v in winner_members]
    normalized, _ = normalize_weights(raw_weights, fp)
    return MergeResult(
        key=key,
        policy="weighted_majority",
        authority_q=winner_value,
        adopted_envs=adopted,
        rejected_envs=rejected,
        tie_break="weight_sum>value_asc",
        evidence={
            "group_weights": {
                str(value): weight_totals[value] for value in sorted(groups)
            },
            "adopted_normalized_weights": dict(zip(adopted, normalized)),
        },
    )


def merge_interval_midpoint(
    votes: list[EnvVote], key: str, fp: FixedPoint
) -> MergeResult:
    values = sorted(v.q_value for v in votes)
    lower, upper = values[0], values[-1]
    if lower == upper:
        midpoint = lower
        adopted = tuple(sorted(v.env_id for v in votes))
        rejected: tuple[str, ...] = ()
    else:
        # Integer midpoint truncated toward zero; interval span parity recorded.
        midpoint = trunc_div_toward_zero(lower + upper, 2)
        # Adopted: environments whose value lies within [lower, upper] (all do);
        # endpoints are explicitly attributed so boundary ownership is auditable.
        adopted = tuple(sorted(v.env_id for v in votes))
        rejected = ()
    endpoint_envs = {
        "lower": tuple(sorted(v.env_id for v in votes if v.q_value == lower)),
        "upper": tuple(sorted(v.env_id for v in votes if v.q_value == upper)),
    }
    return MergeResult(
        key=key,
        policy="interval_midpoint",
        authority_q=midpoint,
        adopted_envs=adopted,
        rejected_envs=rejected,
        tie_break="min_max_midpoint_trunc_toward_zero",
        evidence={
            "lower_q": lower,
            "upper_q": upper,
            "span_even": ((upper - lower) % 2 == 0),
            "endpoint_envs": endpoint_envs,
        },
    )


def merge_key(
    policy: str,
    key: str,
    env_order: tuple[str, ...],
    observations_by_env: dict[str, list[Observation]],
    weights: dict[str, int],
    digests: dict[str, dict[str, str]],
    fp: FixedPoint,
) -> MergeResult | None:
    votes = _head_per_env(env_order, observations_by_env, fp, weights, digests, key)
    if not votes:
        return None
    if policy == "latest":
        return merge_latest(votes, key)
    if policy == "weighted_majority":
        return merge_weighted_majority(votes, key, fp)
    if policy == "interval_midpoint":
        return merge_interval_midpoint(votes, key, fp)
    raise ValueError(f"unknown policy {policy!r}")
