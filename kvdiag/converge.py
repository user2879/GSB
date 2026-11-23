"""Convergence judgement between replayed index positions and lease expiry.

For every shard the replayed index low-water position is compared with the
current lease expiry position. A difference within the declared integer
tolerance converges; otherwise E_DIVERGE is reported with the shard id,
both positions and the difference. The judgement iterates shard ids in
sorted order and depends only on positions, so the conclusion is
independent of shard traversal and renewal processing order.
"""

from .errors import DivergeError


def judge_convergence(spec, replayed_states, lease_expiry_positions, logger=None,
                      shard_ids=None):
    """Return {shard_id: conclusion}; raise nothing — divergence is recorded.

    Each conclusion: {shard_id, status, index_pos, lease_expiry_pos, diff,
    basis}. Divergent shards carry status "diverged" and their basis holds
    the E_DIVERGE report fields.
    """
    conclusions = {}
    order = sorted(shard_ids) if shard_ids is not None else spec.shard_ids()
    for shard_id in order:  # sorted -> order independent
        shard = spec.shard(shard_id)
        state = replayed_states[shard_id]
        index_pos = state["index_pos_lo"]
        expiry_pos = lease_expiry_positions[shard_id]
        diff = index_pos - expiry_pos
        if abs(diff) <= shard.tolerance:
            status = "converged"
            basis = ("|index_pos %d - lease_expiry_pos %d| = %d <= tolerance %d"
                     % (index_pos, expiry_pos, abs(diff), shard.tolerance))
        else:
            status = "diverged"
            basis = ("E_DIVERGE shard=%s index_pos=%d lease_expiry_pos=%d diff=%d"
                     " exceeds tolerance %d"
                     % (shard_id, index_pos, expiry_pos, diff, shard.tolerance))
            if logger:
                logger.emit_error(DivergeError(
                    "index position diverges from lease expiry position",
                    shard_id=shard_id,
                    details={"index_pos": index_pos,
                             "lease_expiry_pos": expiry_pos,
                             "diff": diff,
                             "tolerance": shard.tolerance}))
        conclusions[shard_id] = {
            "shard_id": shard_id,
            "status": status,
            "index_pos": index_pos,
            "lease_expiry_pos": expiry_pos,
            "diff": diff,
            "basis": basis,
        }
    return conclusions


def assert_converged(conclusions):
    """Raise DivergeError for the first (sorted) divergent shard."""
    for shard_id in sorted(conclusions):
        c = conclusions[shard_id]
        if c["status"] != "converged":
            raise DivergeError(
                "index position diverges from lease expiry position",
                shard_id=shard_id,
                details={"index_pos": c["index_pos"],
                         "lease_expiry_pos": c["lease_expiry_pos"],
                         "diff": c["diff"]})
