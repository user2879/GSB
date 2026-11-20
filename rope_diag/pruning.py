"""Dimension pruning policy selection with an auditable decision record.

The selected policy genuinely changes the downstream compute path: its
``dim_range`` restricts the dimensions participating in scoring and its
``score_chunk`` forces chunked dot-product reduction.
"""
from __future__ import annotations


def score_candidate(policy: dict, config: dict) -> float:
    """Score a candidate policy against the requested tier.

    Higher is better: exact block-size match, tolerance headroom and wider
    dimension coverage all contribute.
    """
    score = 0.0
    if int(policy["dim_block_size"]) == int(config["dim_block_size"]):
        score += 100.0
    else:
        score -= abs(int(policy["dim_block_size"]) - int(config["dim_block_size"]))
    headroom = float(config["prune_tolerance"]) - float(policy["min_prune_tolerance"])
    if headroom < 0:
        return -1e9  # policy demands more tolerance than configured: reject
    score += headroom * 1000.0
    lo, hi = policy["dim_range"]
    score += (hi - lo) / max(int(config["dim"]), 1)
    return float(score)


def select_policy(policy_table: dict, config: dict, n_samples: int) -> tuple[dict, dict]:
    """Pick the active policy and build the auditable decision record."""
    extrapolation_ratio = float(config["target_len"]) / float(config["train_len"])
    candidates = []
    for policy in policy_table["policies"]:
        candidates.append(
            {
                "policy_id": policy["policy_id"],
                "score": score_candidate(policy, config),
            }
        )
    ranked = sorted(candidates, key=lambda c: (-c["score"], c["policy_id"]))
    chosen_id = ranked[0]["policy_id"]
    chosen = next(p for p in policy_table["policies"] if p["policy_id"] == chosen_id)
    record = {
        "n_samples_referenced": int(n_samples),
        "extrapolation_ratio": extrapolation_ratio,
        "dim_block_size": int(config["dim_block_size"]),
        "prune_tolerance": float(config["prune_tolerance"]),
        "candidate_scores": ranked,
        "selected_policy": chosen_id,
        "rationale": (
            f"policy {chosen_id} scores highest ({ranked[0]['score']:.6f}) for "
            f"dim_block_size={config['dim_block_size']} and "
            f"prune_tolerance={config['prune_tolerance']}; its dim_range and "
            f"score_chunk are applied to the scoring path"
        ),
    }
    return chosen, record
