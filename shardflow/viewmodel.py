"""Stable read-only diagnostic view model usable without a display."""

from .ledger import Ledger
from .spec import load_spec


def build_view_model(spec_path, ledger_path, events_path=None):
    spec = load_spec(spec_path, events_path)
    ledger = Ledger(ledger_path, replace_on_mismatch=False)
    outcomes = ledger.fetch_outcomes()
    latest = {}
    for row in outcomes:
        latest[row["shard_id"]] = row

    node_rows = []
    for node_id in spec.node_order:
        node = spec.nodes[node_id]
        lease_rows = [
            row for row in outcomes
            if row["node_id"] == node_id and row.get("lease") is not None
        ]
        node_rows.append(
            {
                "node_id": node_id,
                "host": node.host,
                "port": node.port,
                "concurrency": node.concurrency,
                "lease_limit": node.lease_limit,
                "capacity_version": node.capacity_version,
                "lease_event_count": len(lease_rows),
                "active_lease_count": 0,
                "resource_units_seen": sum(row["resource_units"]
                                           for row in lease_rows),
            }
        )

    shard_rows = []
    for shard_id in spec.topo_order:
        row = latest.get(shard_id)
        shard_rows.append(
            {
                "shard_id": shard_id,
                "node_id": spec.shards[shard_id].node_id,
                "depends_on": spec.shards[shard_id].depends_on,
                "status": row["status"] if row else "missing",
                "attempt": row["attempt"] if row else 0,
                "config_version": row["config_version"] if row else None,
                "capacity_version": row["capacity_version"] if row else None,
                "summary_digest": row["outcome_digest"] if row else None,
                "unknown": bool(row and row["status"] == "unknown"),
            }
        )

    invalidations = ledger.fetch_invalidations()
    snapshot_rows = ledger.fetch_snapshots()
    model = {
        "format_version": ledger.format_version(),
        "content_digest": ledger.content_digest(),
        "progress_cursor": ledger.progress_cursor(),
        "nodes": node_rows,
        "shards": shard_rows,
        "ready_or_queued": [
            row["shard_id"]
            for row in shard_rows
            if row["status"] in {"pending", "inflight", "unknown", "missing"}
        ],
        "invalidations": [
            {
                "revision": row["revision"],
                "shard_id": row["shard_id"],
                "reason": row["reason"],
                "basis": row["basis"],
            }
            for row in invalidations
        ],
        "unknown_shards": [
            row["shard_id"] for row in shard_rows if row["unknown"]
        ],
        "snapshots": snapshot_rows,
    }
    ledger.close()
    return model


def render_text(model):
    lines = [
        "Shardflow Diagnostics",
        "ledger format=%s digest=%s"
        % (model["format_version"], model["content_digest"]),
        "",
        "Nodes (stable order):",
    ]
    for row in model["nodes"]:
        lines.append(
            "  %s port=%d lease_events=%d active=%d/%d concurrency=%d capacity=v%d units=%d"
            % (
                row["node_id"],
                row["port"],
                row["lease_event_count"],
                row["active_lease_count"],
                row["lease_limit"],
                row["concurrency"],
                row["capacity_version"],
                row["resource_units_seen"],
            )
        )
    lines.append("")
    lines.append("Shards (topological order):")
    for row in model["shards"]:
        marker = " [UNKNOWN]" if row["unknown"] else ""
        lines.append(
            "  %s -> %s status=%s attempt=%d config=v%s%s"
            % (
                row["shard_id"],
                row["node_id"],
                row["status"],
                row["attempt"],
                row["config_version"],
                marker,
            )
        )
    lines.append("")
    lines.append("Invalidation propagation:")
    for row in model["invalidations"]:
        lines.append(
            "  rev=%s shard=%s reason=%s"
            % (row["revision"], row["shard_id"], row["reason"])
        )
    lines.append("")
    lines.append("Snapshots:")
    for row in model["snapshots"]:
        lines.append("  %s digest=%s" % (row["note"], row["content_digest"]))
    return "\n".join(lines)
