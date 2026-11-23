"""Deterministic multi-environment arbitration engine.

Order-independence guarantee
----------------------------
Every collection is reduced through explicit, total-order sorts
(observation identity, then seq desc, timestamp desc). Environment maps are
iterated in sorted id order. Consequently the same specification produces
byte-identical authority values, accepted sets and rejection reasons regardless
of declaration or traversal order.

Decision bounds (fixed-point q-units, see :mod:`fixedpoint`)
------------------------------------------------------------
* latest_wins             : ``B = 1q`` (one quantum)
* weighted_majority       : ``B = 2q`` (quantized input + division truncation)
* interval_intersection   : ``B = 2q`` (two endpoints combined)

Stable tie-break
----------------
When the fixed-point difference between two candidates is ``<= B`` the winner
is the candidate with the lexicographically smallest stable identity, and the
conclusion is flagged ``error_influenced`` with attribution
``error_indistinguishable``.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .errors import GridError
from .fixedpoint import (
    B_INTERVAL_Q,
    B_LATEST_Q,
    B_MAJORITY_Q,
    FixedPoint,
)
from .spec import (
    POLICY_INTERVAL,
    POLICY_LATEST,
    POLICY_MAJORITY,
    Observation,
    Specification,
    View,
)

REJECT_OUTDATED = "outdated_lower_seq"
REJECT_TIE = "stable_tie_break"
REJECT_MINORITY = "minority_candidate"
REJECT_OUTSIDE_INTERVAL = "outside_intersection"

ATTRIB_ERROR = "error_indistinguishable"
ATTRIB_POLICY = "policy_conflict"
ATTRIB_REJECT_RATIO = "high_rejected_ratio"
REJECTED_RATIO_THRESHOLD = 0.5


@dataclass
class ObservationVerdict:
    obs_id: str
    env_id: str
    seq: int
    timestamp_ms: int
    raw_value: object
    quantized_value: object          # int or [lo_q, hi_q]
    accepted: bool
    reject_reason: str | None = None


@dataclass
class KeyConclusion:
    view_id: str
    key: str
    policy: str
    authority_q: int | None
    authority_real: float | None
    bound_q: int
    adjacent_diff_q: int | None
    error_influenced: bool
    attribution: list[str]
    attribution_contributions: dict[str, str]
    accepted_obs_ids: list[str]
    rejected: list[ObservationVerdict]
    accepted_verdicts: list[ObservationVerdict]
    distribution_q: dict[str, list]
    conflict_detail: str | None = None


@dataclass
class ViewConclusion:
    view_id: str
    version: int
    keys: dict[str, KeyConclusion]
    conclusion_digest: str
    added_keys: list[str] = field(default_factory=list)
    removed_keys: list[str] = field(default_factory=list)
    changed_keys: list[str] = field(default_factory=list)
    is_stale: bool = False


def _quant_obs(fp: FixedPoint, obs: Observation):
    if obs.is_interval:
        lo, hi = obs.value
        lo_q, hi_q = fp.quantize(lo), fp.quantize(hi)
        if lo_q > hi_q:
            raise GridError(
                f"interval lo > hi after quantization for {obs.obs_id}",
                location=f"{obs.env_id}.observations[{obs.key}#{obs.seq}].value",
            )
        return [lo_q, hi_q]
    return fp.quantize(obs.value)


def _sorted_obs(observations: list[Observation]) -> list[Observation]:
    # Total order: seq desc, timestamp desc, identity asc (stable tie-break).
    return sorted(
        observations,
        key=lambda o: (-o.seq, -o.timestamp_ms, o.obs_id),
    )


def _latest_per_env(obs_list: list[Observation]) -> dict[str, Observation]:
    latest: dict[str, Observation] = {}
    for obs in _sorted_obs(obs_list):
        if obs.env_id not in latest:
            latest[obs.env_id] = obs
    return dict(sorted(latest.items()))


def _reject(obs: Observation, qval, reason: str) -> ObservationVerdict:
    return ObservationVerdict(
        obs_id=obs.obs_id, env_id=obs.env_id, seq=obs.seq,
        timestamp_ms=obs.timestamp_ms, raw_value=obs.value,
        quantized_value=qval, accepted=False, reject_reason=reason,
    )


def _accept(obs: Observation, qval) -> ObservationVerdict:
    return ObservationVerdict(
        obs_id=obs.obs_id, env_id=obs.env_id, seq=obs.seq,
        timestamp_ms=obs.timestamp_ms, raw_value=obs.value,
        quantized_value=qval, accepted=True,
    )


def _make_attribution(error_influenced: bool, conflict: bool,
                      rejected_count: int, total_count: int) -> tuple[list[str], dict[str, str]]:
    labels: list[str] = []
    contributions: dict[str, str] = {}
    if error_influenced:
        labels.append(ATTRIB_ERROR)
        contributions[ATTRIB_ERROR] = "candidate fixed-point diff within synthetic bound"
    if conflict:
        labels.append(ATTRIB_POLICY)
        contributions[ATTRIB_POLICY] = "policy could not merge candidates (empty intersection)"
    ratio = (rejected_count / total_count) if total_count else 0.0
    if total_count and ratio > REJECTED_RATIO_THRESHOLD:
        labels.append(ATTRIB_REJECT_RATIO)
        contributions[ATTRIB_REJECT_RATIO] = (
            f"rejected {rejected_count}/{total_count} observations "
            f"({ratio:.2f} > {REJECTED_RATIO_THRESHOLD:.2f})"
        )
    return labels, contributions


def _adjacent_diff(verdicts: list[ObservationVerdict], authority_q: int) -> int | None:
    scalar_vals: list[int] = []
    for verdict in verdicts:
        qv = verdict.quantized_value
        if isinstance(qv, int):
            scalar_vals.append(qv)
        else:
            scalar_vals.extend(qv)
    if not scalar_vals:
        return None
    nearest = min(scalar_vals, key=lambda q: (abs(q - authority_q), q))
    return abs(nearest - authority_q)


class ArbitrationEngine:
    def __init__(self, spec: Specification):
        # Constructing FixedPoint validates the grid range -> E_GRID.
        self.fp = FixedPoint(spec.scale_exponent)
        self.spec = spec

    # ------------------------------------------------------------------
    def _observations_for(self, view: View, key: str) -> list[Observation]:
        wanted = set(view.env_ids)
        found: list[Observation] = []
        for env_id in sorted(wanted):
            env = self.spec.environments[env_id]
            found.extend(o for o in env.observations if o.key == key)
        return found

    def _latest_wins(self, view: View, key: str, obs_list: list[Observation]) -> KeyConclusion:
        # Only the latest observation of each participating environment is eligible.
        obs_list = list(_latest_per_env(obs_list).values())
        ordered = _sorted_obs(obs_list)
        winner = ordered[0]
        winner_q = _quant_obs(self.fp, winner)
        if isinstance(winner_q, list):
            raise GridError(
                "latest_wins policy requires scalar observations",
                location=f"views[{view.view_id}].policies[{key}]",
            )
        verdicts = [_accept(winner, winner_q)]
        rejected: list[ObservationVerdict] = []
        for obs in ordered[1:]:
            rejected.append(_reject(obs, _quant_obs(self.fp, obs), REJECT_OUTDATED))
        # Error influence: the closest rejected candidate within the bound.
        scalar_rejected = [v for v in rejected if isinstance(v.quantized_value, int)]
        error_influenced = any(
            abs(v.quantized_value - winner_q) <= B_LATEST_Q for v in scalar_rejected
        )
        labels, contrib = _make_attribution(error_influenced, False, len(rejected), len(ordered))
        return self._conclusion(view, key, POLICY_LATEST, winner_q, B_LATEST_Q,
                                verdicts, rejected, labels, contrib, ordered)

    def _weighted_majority(self, view: View, key: str,
                           obs_list: list[Observation]) -> KeyConclusion:
        latest = _latest_per_env(obs_list)
        quantized: list[tuple[Observation, int]] = []
        for env_id, obs in latest.items():
            qv = _quant_obs(self.fp, obs)
            if isinstance(qv, list):
                raise GridError(
                    "weighted_majority policy requires scalar observations",
                    location=f"views[{view.view_id}].policies[{key}]",
                )
            quantized.append((obs, qv))
        # Group candidates by quantized value; candidate id is the value itself
        # plus the smallest contributing obs id for a total order.
        groups: dict[int, list[Observation]] = {}
        for obs, qv in quantized:
            groups.setdefault(qv, []).append(obs)
        weights = {env_id: self.spec.environments[env_id].weight for env_id in view.env_ids}
        candidate_rows = []
        for qv, members in groups.items():
            weight_sum = sum(weights[m.env_id] for m in members)
            identity = min(m.obs_id for m in members)
            candidate_rows.append((qv, weight_sum, identity, members))
        max_weight = max(row[1] for row in candidate_rows)
        leaders = [row for row in candidate_rows if row[1] == max_weight]
        # Indistinguishable: leaders whose fixed-point diff is inside the bound.
        leader_values = sorted(row[0] for row in leaders)
        error_influenced = any(
            abs(leader_values[i + 1] - leader_values[i]) <= B_MAJORITY_Q
            for i in range(len(leader_values) - 1)
        )
        # Stable winner: highest weight, then smallest q value, then identity.
        winner_qv, winner_weight, _, winner_members = sorted(
            candidate_rows, key=lambda r: (-r[1], r[0], r[2])
        )[0]
        winner_ids = {m.obs_id for m in winner_members}
        pairs = [(qv, weights[obs.env_id]) for obs, qv in quantized]
        authority_q = self.fp.weighted_mean(pairs)
        verdicts = [_accept(obs, qv) for obs, qv in quantized if obs.obs_id in winner_ids]
        rejected = [
            _reject(obs, qv, REJECT_MINORITY if not error_influenced else REJECT_TIE)
            for obs, qv in quantized if obs.obs_id not in winner_ids
        ]
        labels, contrib = _make_attribution(
            error_influenced, False, len(rejected), len(quantized)
        )
        all_obs = [obs for obs, _ in quantized]
        return self._conclusion(view, key, POLICY_MAJORITY, authority_q, B_MAJORITY_Q,
                                verdicts, rejected, labels, contrib, all_obs)

    def _interval_intersection(self, view: View, key: str,
                               obs_list: list[Observation]) -> KeyConclusion:
        latest = _latest_per_env(obs_list)
        intervals: list[tuple[Observation, list]] = []
        for env_id, obs in latest.items():
            qv = _quant_obs(self.fp, obs)
            if not isinstance(qv, list):
                qv = [qv, qv]  # scalar degenerates to a zero-width interval
            intervals.append((obs, qv))
        lo = max(qv[0] for _, qv in intervals)
        hi = min(qv[1] for _, qv in intervals)
        conflict = lo > hi
        verdicts: list[ObservationVerdict] = []
        rejected: list[ObservationVerdict] = []
        if conflict:
            authority_q = None
            for obs, qv in intervals:
                rejected.append(_reject(obs, qv, REJECT_OUTSIDE_INTERVAL))
        else:
            authority_q = self.fp.midpoint_floor(lo, hi)
            for obs, qv in intervals:
                if qv[0] <= authority_q <= qv[1]:
                    verdicts.append(_accept(obs, qv))
                else:
                    rejected.append(_reject(obs, qv, REJECT_OUTSIDE_INTERVAL))
        error_influenced = (not conflict) and (hi - lo) <= B_INTERVAL_Q
        labels, contrib = _make_attribution(
            error_influenced, conflict, len(rejected), len(intervals)
        )
        all_obs = [obs for obs, _ in intervals]
        conflict_detail = f"empty intersection [{lo}, {hi}]" if conflict else None
        return self._conclusion(view, key, POLICY_INTERVAL, authority_q, B_INTERVAL_Q,
                                verdicts, rejected, labels, contrib, all_obs,
                                conflict_detail=conflict_detail)

    # ------------------------------------------------------------------
    def _conclusion(self, view: View, key: str, policy: str, authority_q, bound_q,
                    verdicts, rejected, labels, contrib, all_obs,
                    conflict_detail=None) -> KeyConclusion:
        distribution: dict[str, list] = {}
        for obs in _sorted_obs(all_obs):
            qv = _quant_obs(self.fp, obs)
            distribution.setdefault(obs.env_id, []).append(
                {"obs_id": obs.obs_id, "seq": obs.seq, "ts": obs.timestamp_ms, "q": qv}
            )
        adjacent = _adjacent_diff(verdicts + rejected, authority_q) if authority_q is not None else None
        return KeyConclusion(
            view_id=view.view_id, key=key, policy=policy,
            authority_q=authority_q,
            authority_real=None if authority_q is None else self.fp.to_real(authority_q),
            bound_q=bound_q, adjacent_diff_q=adjacent,
            error_influenced=bool(labels and ATTRIB_ERROR in labels),
            attribution=labels, attribution_contributions=contrib,
            accepted_obs_ids=sorted(v.obs_id for v in verdicts),
            rejected=sorted(rejected, key=lambda v: v.obs_id),
            accepted_verdicts=sorted(verdicts, key=lambda v: v.obs_id),
            distribution_q={env_id: distribution[env_id] for env_id in sorted(distribution)},
            conflict_detail=conflict_detail,
        )

    def arbitrate_key(self, view: View, key: str) -> KeyConclusion:
        obs_list = self._observations_for(view, key)
        if not obs_list:
            raise GridError(
                f"no observations available for key '{key}' in view '{view.view_id}'",
                location=f"views[{view.view_id}].policies[{key}]",
            )
        policy = view.policies[key]
        if policy == POLICY_LATEST:
            return self._latest_wins(view, key, obs_list)
        if policy == POLICY_MAJORITY:
            return self._weighted_majority(view, key, obs_list)
        return self._interval_intersection(view, key, obs_list)
