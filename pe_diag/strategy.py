"""Scale-tier algorithmic trade-off selector.

A fixed, pre-registered policy table maps three candidate strategies onto the
four knobs that genuinely change the compute path:

* scheme priority order,
* active dimension range fed to scoring,
* scoring block method (single batch vs streaming online softmax),
* extrapolation layering precision.

Eligibility is a deterministic budget/memory test, so the three tiers map to
three genuinely different compute paths:

* ``S_FULL``   only fits when the full-dimension point load is inside both the
  batch-point budget and the memory ceiling (small tier);
* ``S_CHUNK``  fits when a 75%%-dimension batched path fits (medium tier);
* ``S_STREAM`` always fits because it holds O(active_dim) state per key and is
  the only eligible strategy on the large tier.

The selected strategy is the *lightest eligible* one (full > chunk > stream
preference when several fit).  Every candidate, its eligibility verdict and its
numeric score are retained in the audit record.
"""

from __future__ import annotations

from dataclasses import dataclass

POLICY_TABLE = {
    "S_FULL": {
        "scheme_priority": ["rope", "linear", "ntk"],
        "dim_fraction": 1.0,
        "block_method": "single",
        "layer_precision": "float64",
        # multiplier of the basic point/memory cost for this strategy
        "cost_weight": 1.0,
    },
    "S_CHUNK": {
        "scheme_priority": ["ntk", "rope", "linear"],
        "dim_fraction": 0.75,
        "block_method": "single",
        "layer_precision": "float64",
        "cost_weight": 0.6,
    },
    "S_STREAM": {
        "scheme_priority": ["ntk", "linear", "rope"],
        "dim_fraction": 0.5,
        "block_method": "stream",
        "layer_precision": "float32",
        "cost_weight": 0.35,
    },
}
PREFERENCE_ORDER = ["S_FULL", "S_CHUNK", "S_STREAM"]


@dataclass(frozen=True)
class TierMetrics:
    tier_id: str
    sample_count: int
    dim: int
    extrapolation_ratio: float
    train_len: int
    target_len: int
    batch_point_budget: int
    memory_limit_mb: int

    def to_dict(self) -> dict:
        return {
            "tier_id": self.tier_id,
            "sample_count": self.sample_count,
            "dim": self.dim,
            "extrapolation_ratio": self.extrapolation_ratio,
            "train_len": self.train_len,
            "target_len": self.target_len,
            "batch_point_budget": self.batch_point_budget,
            "memory_limit_mb": self.memory_limit_mb,
        }


def _even_dim(fraction: float, dim: int) -> int:
    active = int(round(dim * fraction))
    active = max(2, min(dim, active))
    if active % 2 == 1:
        active -= 1
    return active


def score_strategy(name: str, metrics: TierMetrics) -> dict:
    policy = POLICY_TABLE[name]
    active_dim = _even_dim(policy["dim_fraction"], metrics.dim)
    points = metrics.sample_count * active_dim
    weighted_points = points * policy["cost_weight"]
    # batched paths materialize rotated features for every sample in a
    # working batch; streaming only keeps one key at a time.
    items_per_batch = 1 if policy["block_method"] == "stream" else metrics.sample_count
    bytes_per_item = active_dim * 8
    mem_mb = (items_per_batch * bytes_per_item) / (1024.0 * 1024.0)
    within_point_budget = weighted_points <= metrics.batch_point_budget
    within_memory = mem_mb <= metrics.memory_limit_mb
    eligible = within_point_budget and within_memory

    # deterministic scalar score: lower is better; ineligible strategies are
    # pushed above every eligible one by a fixed large penalty
    load_ratio = weighted_points / max(1, metrics.batch_point_budget)
    mem_ratio = mem_mb / max(1e-9, metrics.memory_limit_mb)
    score = policy["cost_weight"] + load_ratio + mem_ratio
    if not eligible:
        score += 1000.0

    return {
        "strategy": name,
        "eligible": eligible,
        "score": score,
        "active_dim": active_dim,
        "dim_range": [0, active_dim],
        "block_method": policy["block_method"],
        "layer_precision": policy["layer_precision"],
        "scheme_priority": list(policy["scheme_priority"]),
        "factors": {
            "weighted_points": weighted_points,
            "points": points,
            "load_ratio": load_ratio,
            "memory_mb": mem_mb,
            "memory_ratio": mem_ratio,
            "within_point_budget": within_point_budget,
            "within_memory": within_memory,
            "cost_weight": policy["cost_weight"],
        },
    }


def select_strategy(metrics: TierMetrics) -> dict:
    scored = [score_strategy(name, metrics) for name in PREFERENCE_ORDER]
    eligible = [row for row in scored if row["eligible"]]
    pool = eligible if eligible else scored
    # lightest eligible by preference order (full > chunk > stream)
    order_index = {name: i for i, name in enumerate(PREFERENCE_ORDER)}
    best = min(pool, key=lambda row: (order_index[row["strategy"]],
                                      row["score"]))
    return {
        "tier_metrics": metrics.to_dict(),
        "candidate_scores": scored,
        "selected": best["strategy"],
        "selection_rule": "lightest_eligible_preference_full_chunk_stream",
        "decision": {
            "strategy": best["strategy"],
            "scheme_priority": best["scheme_priority"],
            "active_dim": best["active_dim"],
            "dim_range": best["dim_range"],
            "block_method": best["block_method"],
            "layer_precision": best["layer_precision"],
        },
    }
