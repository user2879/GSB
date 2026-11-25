"""Deterministic arbitration engine and the three merge policies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .determinism import digest_of
from .errors import FixedPointError
from .fixedpoint import FixedPoint
from .spec import Manifest, ViewSpec

CAUSE_ERROR_INDISTINGUISHABLE = "error_indistinguishable"
CAUSE_POLICY_CONFLICT = "policy_conflict"
CAUSE_REJECTED_RATIO_HIGH = "rejected_ratio_too_high"
CAUSE_NONE = "none"


@dataclass
class EnvChain:
    """Effective revision chain for one environment and one config key."""

    env_id: str
    key: str
    seq_low: int
    seq_high: int
    # Per-instance latest observation for this key after revision ordering.
    latest: dict[str, dict[str, Any]] = field(default_factory=dict)

    def representative(self) -> dict[str, Any] | None:
        """Deterministically chosen representative observation of the env."""
        if not self.latest:
            return None
        # Highest revision wins; ties resolve by instance id for determinism.
        return sorted(
            self.latest.values(),
            key=lambda o: (-int(o["seq"]), str(o["instance_id"])),
        )[0]


@dataclass
class KeyVerdict:
    key: str
    policy: str
    authority_q: int | None
    authority_text: str
    interval: list[int] | None
    adopted: tuple[str, ...]
    rejected: tuple[str, ...]
    error_bound_ulp: int
    aggregate_bound_ulp: int
    error_influenced: bool
    causes: tuple[str, ...]
    candidate_digest: str

    def to_record(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "policy": self.policy,
            "authority_q": self.authority_q,
            "authority_text": self.authority_text,
            "interval": self.interval,
            "adopted": list(self.adopted),
            "rejected": list(self.rejected),
            "error_bound_ulp": self.error_bound_ulp,
            "aggregate_bound_ulp": self.aggregate_bound_ulp,
            "error_influenced": self.error_influenced,
            "causes": list(self.causes),
            "candidate_digest": self.candidate_digest,
        }


@dataclass
class ViewResult:
    view_id: str
    view_version: int
    scale_power: int
    verdicts: dict[str, KeyVerdict]
    content_digest: str

    def ordered_verdicts(self) -> list[KeyVerdict]:
        return [self.verdicts[k] for k in sorted(self.verdicts)]

    def to_record(self) -> dict[str, Any]:
        return {
            "view_id": self.view_id,
            "view_version": self.view_version,
            "scale_power": self.scale_power,
            "verdicts": [v.to_record() for v in self.ordered_verdicts()],
            "content_digest": self.content_digest,
        }


class ArbitrationEngine:
    """Stateless deterministic arbitration over an observation store."""

    def __init__(self, manifest: Manifest) -> None:
        self.manifest = manifest
        self.fp = FixedPoint(manifest.scale_power)

    # -- environment chains ----------------------------------------------
    def build_chains(
        self, observations: list[dict[str, Any]], env_ids: tuple[str, ...], key: str
    ) -> list[EnvChain]:
        wanted_envs = set(env_ids)
        chains = {env_id: EnvChain(env_id, key, 10**12, -(10**12))
                  for env_id in env_ids}
        for obs in observations:
            if obs.get("key") != key or obs.get("env_id") not in wanted_envs:
                continue
            chain = chains[obs["env_id"]]
            seq = int(obs["seq"])
            chain.seq_low = min(chain.seq_low, seq)
            chain.seq_high = max(chain.seq_high, seq)
            current = chain.latest.get(obs["instance_id"])
            if current is None or int(obs["seq"]) >= int(current["seq"]):
                chain.latest[obs["instance_id"]] = obs
        return [chains[eid] for eid in sorted(wanted_envs)]

    def build_chain_for_env(
        self, observations: list[dict[str, Any]], env_id: str, key: str
    ) -> EnvChain:
        """Build one environment chain; safe to run inside a worker thread."""
        chain = EnvChain(env_id, key, 10**12, -(10**12))
        for obs in observations:
            if obs.get("key") != key or obs.get("env_id") != env_id:
                continue
            seq = int(obs["seq"])
            chain.seq_low = min(chain.seq_low, seq)
            chain.seq_high = max(chain.seq_high, seq)
            current = chain.latest.get(obs["instance_id"])
            if current is None or int(obs["seq"]) >= int(current["seq"]):
                chain.latest[obs["instance_id"]] = obs
        return chain

    def arbitrate_key(self, view: ViewSpec, key: str,
                      chains: list[EnvChain]) -> KeyVerdict:
        """Dispatch a single key to its declared deterministic policy."""
        policy = view.policies[key]
        if policy == "latest_revision":
            return self._latest(view, key, policy, chains)
        if policy == "weighted_majority":
            return self._weighted(view, key, policy, chains)
        return self._interval(view, key, policy, chains)

    # -- view arbitration -------------------------------------------------
    def arbitrate_view(
        self,
        view: ViewSpec,
        view_version: int,
        observations: list[dict[str, Any]],
        only_keys: tuple[str, ...] | None = None,
        previous: dict[str, KeyVerdict] | None = None,
    ) -> ViewResult:
        keys = sorted(view.policies) if only_keys is None else sorted(only_keys)
        verdicts: dict[str, KeyVerdict] = {}
        if previous:
            verdicts.update(previous)
        for key in keys:
            chains = self.build_chains(observations, view.env_refs, key)
            verdicts[key] = self.arbitrate_key(view, key, chains)
        ordered = [verdicts[k].to_record() for k in sorted(verdicts)]
        return ViewResult(
            view_id=view.view_id,
            view_version=view_version,
            scale_power=self.manifest.scale_power,
            verdicts=verdicts,
            content_digest=digest_of(
                {"view_id": view.view_id, "view_version": view_version,
                 "scale_power": self.manifest.scale_power, "verdicts": ordered}
            ),
        )

    # -- policy: latest revision -----------------------------------------
    def _latest(self, view: ViewSpec, key: str, policy: str,
                chains: list[EnvChain]) -> KeyVerdict:
        candidates: list[tuple[dict[str, Any], str]] = []
        for chain in chains:
            rep = chain.representative()
            if rep is not None:
                candidates.append((rep, chain.env_id))
        bound = self.fp.comparison_bound_ulp(view.tolerance)
        if not candidates:
            return self._empty_verdict(key, policy, view, bound, len(chains))
        quant = [(self.fp.quantize(rep["value"]).q, rep, env_id)
                 for rep, env_id in candidates]
        max_seq = max(int(rep["seq"]) for rep, _ in candidates)
        leaders = [(q, rep, env_id) for q, rep, env_id in quant
                   if int(rep["seq"]) == max_seq]
        # Stable tie-break: smallest quantized value, then env id.
        leaders.sort(key=lambda t: (t[0], t[2]))
        chosen_q, chosen_rep, chosen_env = leaders[0]
        adopted = tuple(sorted({env for _, rep, env in leaders
                                if self.fp.quantize(rep["value"]).q == chosen_q}))
        rejected = tuple(sorted(
            c.env_id for c in chains if c.env_id not in adopted
        ))
        error_influenced = any(
            abs(q - chosen_q) <= bound and q != chosen_q for q, _, _ in quant
        )
        causes = self._causes(error_influenced, False, adopted, rejected, view)
        return KeyVerdict(
            key=key, policy=policy, authority_q=chosen_q,
            authority_text=self.fp.dequantize(chosen_q), interval=None,
            adopted=adopted, rejected=rejected, error_bound_ulp=bound,
            aggregate_bound_ulp=self.fp.view_aggregate_bound_ulp(
                len(view.policies), view.tolerance),
            error_influenced=error_influenced, causes=causes,
            candidate_digest=digest_of(
                sorted((eid, q) for q, _, eid in quant)),
        )

    # -- policy: weighted majority ---------------------------------------
    def _weighted(self, view: ViewSpec, key: str, policy: str,
                  chains: list[EnvChain]) -> KeyVerdict:
        bound = self.fp.comparison_bound_ulp(view.tolerance)
        grouped: dict[int, dict[str, Any]] = {}
        for chain in chains:
            rep = chain.representative()
            if rep is None:
                continue
            q = self.fp.quantize(rep["value"]).q
            grouped.setdefault(q, {"envs": [], "weight": 0})
            grouped[q]["envs"].append(chain.env_id)
            grouped[q]["weight"] += self.manifest.env(chain.env_id).weight
        if not grouped:
            return self._empty_verdict(key, policy, view, bound, len(chains))
        ranked = sorted(
            grouped.items(),
            key=lambda kv: (-kv[1]["weight"], kv[0], tuple(sorted(kv[1]["envs"]))),
        )
        top_q, top_item = ranked[0]
        runner_up = ranked[1] if len(ranked) > 1 else None
        error_influenced = runner_up is not None and abs(
            runner_up[0] - top_q
        ) <= bound
        adopted = tuple(sorted(top_item["envs"]))
        rejected = tuple(sorted(
            c.env_id for c in chains
            if c.env_id not in adopted and c.representative() is not None
        ))
        conflict = runner_up is not None and runner_up[1]["weight"] == top_item["weight"]
        causes = self._causes(error_influenced, conflict, adopted, rejected, view)
        return KeyVerdict(
            key=key, policy=policy, authority_q=top_q,
            authority_text=self.fp.dequantize(top_q), interval=None,
            adopted=adopted, rejected=rejected, error_bound_ulp=bound,
            aggregate_bound_ulp=self.fp.view_aggregate_bound_ulp(
                len(view.policies), view.tolerance),
            error_influenced=error_influenced, causes=causes,
            candidate_digest=digest_of(
                sorted((q, item["weight"], tuple(sorted(item["envs"])))
                       for q, item in grouped.items())),
        )

    # -- policy: interval intersection midpoint --------------------------
    def _interval(self, view: ViewSpec, key: str, policy: str,
                  chains: list[EnvChain]) -> KeyVerdict:
        bound = self.fp.comparison_bound_ulp(view.tolerance)
        reps: list[tuple[list[int], str]] = []
        for chain in chains:
            rep = chain.representative()
            if rep is None:
                continue
            value = rep["value"]
            if not (isinstance(value, list) and len(value) == 2
                    and all(isinstance(v, int) for v in value)
                    and value[0] <= value[1]):
                raise FixedPointError(
                    f"interval policy requires [lo,hi] integers for {key}",
                    field=f"value.{key}",
                )
            reps.append((value, chain.env_id))
        if not reps:
            return self._empty_verdict(key, policy, view, bound, len(chains))
        lo = max(value[0] for value, _ in reps)
        hi = min(value[1] for value, _ in reps)
        conflict = lo > hi
        if conflict:
            # Deterministic fallback: collapse to the smallest lower bound's
            # midpoint and flag a policy conflict; nothing is adopted.
            value, env_id = sorted(reps, key=lambda t: (t[0][0], t[1]))[0]
            mid_q = self.fp.quantize((value[0] + value[1]) / 2).q
            adopted: tuple[str, ...] = ()
            rejected = tuple(sorted(eid for _, eid in reps))
            causes = self._causes(False, True, adopted, rejected, view)
            interval = [lo, hi]
        else:
            mid_q = self.fp.quantize((lo + hi) / 2).q
            adopted = tuple(sorted(
                env_id for value, env_id in reps if value[0] <= lo and hi <= value[1]
            ))
            rejected = tuple(sorted(
                env_id for _, env_id in reps if env_id not in adopted
            ))
            causes = self._causes(False, False, adopted, rejected, view)
            interval = [lo, hi]
        # Error influence: any two interval midpoints within the bound.
        mids = sorted(self.fp.quantize((v[0] + v[1]) / 2).q for v, _ in reps)
        error_influenced = any(
            abs(mids[i] - mids[i - 1]) <= bound and mids[i] != mids[i - 1]
            for i in range(1, len(mids))
        )
        if error_influenced and CAUSE_ERROR_INDISTINGUISHABLE not in causes:
            causes = tuple(sorted(set(causes) | {CAUSE_ERROR_INDISTINGUISHABLE}))
        return KeyVerdict(
            key=key, policy=policy, authority_q=mid_q,
            authority_text=self.fp.dequantize(mid_q), interval=interval,
            adopted=adopted, rejected=rejected, error_bound_ulp=bound,
            aggregate_bound_ulp=self.fp.view_aggregate_bound_ulp(
                len(view.policies), view.tolerance),
            error_influenced=error_influenced, causes=causes,
            candidate_digest=digest_of(
                sorted((eid, tuple(value)) for value, eid in reps)),
        )

    # -- helpers ----------------------------------------------------------
    def _causes(self, error_influenced: bool, conflict: bool,
                adopted: tuple[str, ...], rejected: tuple[str, ...],
                view: ViewSpec) -> tuple[str, ...]:
        causes: set[str] = set()
        if error_influenced:
            causes.add(CAUSE_ERROR_INDISTINGUISHABLE)
        if conflict:
            causes.add(CAUSE_POLICY_CONFLICT)
        total = len(adopted) + len(rejected)
        if total and len(rejected) / total > self.manifest.rejected_ratio_threshold:
            causes.add(CAUSE_REJECTED_RATIO_HIGH)
        return tuple(sorted(causes)) if causes else (CAUSE_NONE,)

    def _empty_verdict(self, key: str, policy: str, view: ViewSpec,
                       bound: int, chain_count: int) -> KeyVerdict:
        return KeyVerdict(
            key=key, policy=policy, authority_q=None, authority_text=None,
            interval=None, adopted=(), rejected=tuple(
                env.env_id for env in self.manifest.environments
                if env.env_id in view.env_refs),
            error_bound_ulp=bound,
            aggregate_bound_ulp=self.fp.view_aggregate_bound_ulp(
                len(view.policies), view.tolerance),
            error_influenced=False, causes=(CAUSE_NONE,),
            candidate_digest=digest_of([]),
        )
