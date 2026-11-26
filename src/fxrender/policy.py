"""Tiered policy table controlling real render-path decisions.

The policy is selected from frame count, tile rows and per-step cost. It
changes the denoise batch size, low-impact step merging and the tile-row
decode order, and therefore changes the point ledger and execution path.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Policy:
    tier: str
    batch_steps: int
    merge_low_impact: bool
    decode_order: str  # "topdown" or "outside-in"
    low_impact_every: int

    def as_dict(self) -> dict:
        return {
            "tier": self.tier,
            "batch_steps": self.batch_steps,
            "merge_low_impact": self.merge_low_impact,
            "decode_order": self.decode_order,
            "low_impact_every": self.low_impact_every,
        }


def select_tier(frame_count: int, cfg) -> str:
    if frame_count <= cfg.small_max_frames:
        return "small"
    if frame_count <= cfg.medium_max_frames:
        return "medium"
    return "large"


_POLICY_TABLE = {
    "small": Policy("small", batch_steps=4, merge_low_impact=True,
                    decode_order="topdown", low_impact_every=3),
    "medium": Policy("medium", batch_steps=3, merge_low_impact=True,
                     decode_order="outside-in", low_impact_every=2),
    "large": Policy("large", batch_steps=2, merge_low_impact=False,
                    decode_order="outside-in", low_impact_every=1),
}


def select_policy(frame_count: int, cfg) -> Policy:
    tier = select_tier(frame_count, cfg)
    policy = _POLICY_TABLE[tier]
    if not cfg.merge_low_impact:
        # Config can disable merging; other tier attributes still apply.
        policy = Policy(policy.tier, policy.batch_steps, False,
                        policy.decode_order, policy.low_impact_every)
    return policy


def tile_decode_order(tile_rows: int, policy: Policy) -> list[int]:
    if policy.decode_order == "topdown":
        return list(range(tile_rows))
    # outside-in: 0, n-1, 1, n-2, ... deterministic and stable
    order: list[int] = []
    left, right = 0, tile_rows - 1
    while left <= right:
        order.append(left)
        if left != right:
            order.append(right)
        left += 1
        right -= 1
    return order


def merged_step_groups(denoise_steps: int, policy: Policy) -> list[tuple[int, int, bool]]:
    """Return (step, group_id, merged_flag) in stable step order."""
    groups: list[tuple[int, int, bool]] = []
    if not policy.merge_low_impact:
        return [(step, step, False) for step in range(denoise_steps)]
    group_id = 0
    step = 0
    while step < denoise_steps:
        if (policy.low_impact_every > 1
                and step > 0
                and step % policy.low_impact_every == 0
                and step + 1 < denoise_steps):
            groups.append((step, group_id, True))
            groups.append((step + 1, group_id, True))
            step += 2
        else:
            groups.append((step, group_id, False))
            step += 1
        group_id += 1
    return groups
