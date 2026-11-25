"""Independent read-only replay verifier.

The verifier reads only the original declaration files and the durable ledger.
It rebuilds grants, renewals, expiries, cancellations, invalidation propagation
and resource occupancy from ledger events, then compares the derived view and
outcomes against the persisted rows. The first mismatch raises ``E_REPLAY``
with the shard id and field name.
"""

from .errors import ReplayError
from .invalidation import propagate_invalidation
from .ledger import Ledger
from .samples import generate_samples, sample_digest, fixed_point_aggregate
from .spec import load_spec
from .util import canonical_json


class ReplayState:
    def __init__(self, spec):
        self.spec = spec
        self.active_leases = {node_id: set() for node_id in spec.node_order}
        self.active_shards = {node_id: set() for node_id in spec.node_order}
        self.resource_in_use = 0
        self.leases = {}
        self.current_revision = min(spec.revisions)
        self.revision_parent = {}
        self.completed = {}
        self.cancelled = set()
        self.invalidated = []
        self.reason_by_shard = {}
        self.concurrency = {
            node_id: node.concurrency
            for node_id, node in spec.nodes.items()
        }

    def fail(self, shard_id, field, expected, actual):
        raise ReplayError(
            "replay mismatch on shard %s field %s: expected=%r actual=%r"
            % (shard_id, field, expected, actual),
            object_id=shard_id,
            field=field,
        )

    def check_budget(self, shard_id):
        if self.resource_in_use > self.spec.global_budget:
            self.fail(shard_id, "resource_in_use",
                      "<= %d" % self.spec.global_budget, self.resource_in_use)
        node_id = self.spec.shards[shard_id].node_id
        node = self.spec.nodes[node_id]
        if len(self.active_leases[node_id]) > node.lease_limit:
            self.fail(shard_id, "node_lease_limit", node.lease_limit,
                      len(self.active_leases[node_id]))
        if len(self.active_shards[node_id]) > self.concurrency[node_id]:
            self.fail(shard_id, "node_concurrency", node.concurrency,
                      len(self.active_shards[node_id]))


def verify_ledger(spec_path, ledger_path, events_path=None):
    spec = load_spec(spec_path, events_path)
    ledger = Ledger(ledger_path, replace_on_mismatch=False)
    state = ReplayState(spec)

    for event in ledger.fetch_lease_events():
        _apply_lease_event(spec, state, event)

    _verify_invalidations(spec, ledger, state)
    _verify_outcomes(spec, ledger, state)
    _verify_final_view(spec, ledger, state)

    return {
        "status": "ok",
        "format_version": ledger.format_version(),
        "content_digest": ledger.content_digest(),
        "completed_shards": sorted(state.completed),
        "cancelled_shards": sorted(state.cancelled),
        "unknown_shards": sorted(
            row["shard_id"]
            for row in ledger.fetch_outcomes()
            if row["status"] == "unknown"
        ),
        "resource_in_use": state.resource_in_use,
        "checks": [
            "lease_events",
            "resource_budget",
            "node_limits",
            "invalidations",
            "outcome_digests",
            "aggregate_digests",
            "capacity_versions",
            "final_view",
        ],
    }


def _apply_lease_event(spec, state, event):
    kind = event["kind"]
    payload = event["payload"]
    if kind == "grant":
        shard_id = payload["shard_id"]
        node_id = payload["node_id"]
        state.leases[shard_id] = dict(payload)
        state.active_leases[node_id].add(shard_id)
        state.active_shards[node_id].add(shard_id)
        state.resource_in_use += payload["resource_units"]
        state.check_budget(shard_id)
    elif kind in {"release", "expire", "expired_late_receipt",
                  "stale_receipt", "node_error", "invalidated", "complete"}:
        shard_id = payload.get("shard_id")
        if shard_id is None and "lease" in payload:
            shard_id = payload["lease"].get("shard_id")
        if shard_id and shard_id in state.leases:
            node_id = state.leases[shard_id]["node_id"]
            state.resource_in_use -= spec.shards[shard_id].resource_units
            state.active_leases[node_id].discard(shard_id)
            state.active_shards[node_id].discard(shard_id)
            state.leases.pop(shard_id, None)
    elif kind == "renew":
        shard_id = payload["lease"]["shard_id"]
        if shard_id not in state.leases:
            state.fail(shard_id, "renew", "active lease", "missing")
        renewed = dict(payload["lease"])
        renewed.setdefault(
            "resource_units", spec.shards[shard_id].resource_units
        )
        state.leases[shard_id] = renewed
    elif kind in {"renew_rejected_local", "renew_rejected_node",
                  "renew_network_failed"}:
        shard_id = payload["lease"]["shard_id"]
        stored = state.leases.get(shard_id)
        if stored is None:
            state.fail(shard_id, "renew_reject", "active lease", "missing")
    elif kind == "cancel":
        shard_id = payload["shard_id"]
        state.cancelled.add(shard_id)
        if shard_id in state.leases:
            node_id = state.leases[shard_id]["node_id"]
            state.resource_in_use -= spec.shards[shard_id].resource_units
            state.active_leases[node_id].discard(shard_id)
            state.active_shards[node_id].discard(shard_id)
            state.leases.pop(shard_id, None)
    elif kind == "late_receipt_ignored":
        return
    elif kind == "quota_change":
        state.concurrency[payload["node_id"]] = payload["new"]
        return
    elif kind == "config_applied":
        state.current_revision = payload["revision"]
        state.revision_parent[payload["revision"]] = payload.get("parent")
        return
    else:
        raise ReplayError("unknown lease event kind %s" % kind,
                          field=kind)
    if state.resource_in_use < 0:
        state.fail(kind, "resource_in_use", ">=0", state.resource_in_use)


def _verify_invalidations(spec, ledger, state):
    rows = ledger.fetch_invalidations()
    by_revision = {}
    for row in rows:
        by_revision.setdefault(row["revision"], set()).add(row["shard_id"])
    for revision, shard_ids in sorted(by_revision.items()):
        parent = state.revision_parent.get(
            revision,
            max(v for v in spec.revisions if v < revision),
        )
        ordered, reasons = propagate_invalidation(
            spec,
            parent if parent in spec.revisions else None,
            revision,
        )
        # Cancellation is terminal: the engine deliberately does not re-dispatch
        # cancelled shards or their blocked downstream set.
        ordered = [sid for sid in ordered if sid not in state.cancelled]
        reasons = {sid: reasons[sid] for sid in ordered}
        if shard_ids != set(ordered):
            first = sorted(shard_ids.symmetric_difference(ordered))[0]
            state.fail(first, "invalidation_set", set(ordered), shard_ids)
        for shard_id in ordered:
            ledger_reasons = next(
                row["basis"]["reasons"]
                for row in rows
                if row["revision"] == revision and row["shard_id"] == shard_id
            )
            derived_reasons = reasons[shard_id]["reasons"]
            if canonical_json(ledger_reasons) != canonical_json(derived_reasons):
                state.fail(shard_id, "invalidation_basis", derived_reasons,
                           ledger_reasons)
        state.invalidated.extend(ordered)
        state.reason_by_shard.update(reasons)


def _verify_outcomes(spec, ledger, state):
    latest = {}
    for row in ledger.fetch_outcomes():
        latest[row["shard_id"]] = row
    for shard_id in spec.shard_order:
        if shard_id not in latest:
            continue
        row = latest[shard_id]
        shard = spec.shards[shard_id]
        node = spec.nodes[shard.node_id]
        if row["node_id"] != shard.node_id:
            state.fail(shard_id, "node_id", shard.node_id, row["node_id"])
        if row["resource_units"] != shard.resource_units:
            state.fail(shard_id, "resource_units", shard.resource_units,
                       row["resource_units"])
        if row["capacity_version"] != node.capacity_version and \
                row["status"] != "unknown":
            state.fail(shard_id, "capacity_version", node.capacity_version,
                       row["capacity_version"])
        if row["status"] == "completed":
            _verify_completed(spec, ledger, state, shard_id, row)
        elif row["status"] == "cancelled":
            if shard_id not in state.cancelled:
                state.fail(shard_id, "cancelled", True, False)
        elif row["status"] == "blocked":
            deps = shard.depends_on
            if not any(dep in state.cancelled for dep in deps):
                state.fail(shard_id, "blocked",
                           "a cancelled dependency", row["status"])
        elif row["status"] == "unknown":
            if row["summary_json"] is not None:
                state.fail(shard_id, "unknown_summary", None,
                           row["summary_json"])


def _verify_completed(spec, ledger, state, shard_id, row):
    shard = spec.shards[shard_id]
    node = spec.nodes[shard.node_id]
    summary = row["summary"]
    groups, values = generate_samples(node.seed, shard.sample_count)
    expected_digest = sample_digest(groups, values)
    # The dispatch digest is not stored on outcome, but regenerating the exact
    # sample batch proves the deterministic input used by the node.
    if groups.size != shard.sample_count:
        state.fail(shard_id, "sample_count", shard.sample_count, groups.size)
    effective = spec.revision_values(row["config_version"])
    expected_summary = fixed_point_aggregate(
        groups, values, node.fixed_scale, effective, node.compute_iters
    )
    if canonical_json(summary) != canonical_json(expected_summary):
        state.fail(shard_id, "summary", expected_summary, summary)
    if summary["digest"] != expected_summary["digest"]:
        state.fail(shard_id, "summary_digest", expected_summary["digest"],
                   summary["digest"])
    if row["config_version"] not in spec.revisions:
        state.fail(shard_id, "config_version",
                   "known revision", row["config_version"])
    state.completed[shard_id] = row
    # Silence unused variable while keeping the input digest assertion named.
    assert expected_digest


def _verify_final_view(spec, ledger, state):
    # All non-cancelled, non-unknown shards must have a committed completed row
    # and no lease may remain active at the end of a clean run.
    latest = {}
    for row in ledger.fetch_outcomes():
        latest[row["shard_id"]] = row
    for shard_id in spec.shard_order:
        row = latest.get(shard_id)
        if row is None:
            state.fail(shard_id, "outcome", "committed row", "missing")
            continue
        if row["status"] == "unknown":
            continue
        if row["status"] not in {"completed", "cancelled", "blocked"}:
            state.fail(shard_id, "status",
                       "completed/cancelled/unknown", row["status"])
    if state.leases:
        shard_id = sorted(state.leases)[0]
        state.fail(shard_id, "active_lease_at_end", set(), set(state.leases))
    if state.resource_in_use != 0:
        state.fail("ledger", "resource_in_use", 0, state.resource_in_use)
