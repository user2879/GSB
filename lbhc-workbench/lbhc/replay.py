"""Ledger consistency replay.

Re-advances the health state machine from the ledger's committed probe
sequence and compares, item by item, the per-instance state, transition
sequence and counters against what the ledger recorded. The first
divergence is reported as E_REPLAY with the instance id and field.
"""

from __future__ import annotations

from . import health as hm
from .errors import ReplayError


def replay_verify(ledger, spec):
    ledger.verify_chain()  # E_LEDGER surfaces first if tampered
    version = ledger.current_version()
    if version is None:
        raise ReplayError("ledger has no committed spec version",
                          path="meta.spec_version")
    if version != spec.version:
        raise ReplayError(
            f"spec version {spec.version} does not match ledger version "
            f"{version}; refusing to replay across versions",
            path="version")

    health_map = {}
    for pool in spec.pools:
        for b in pool.backends:
            health_map[b.id] = hm.initial_health(b.role)

    expected_transitions = []
    for probe in ledger.probes(version):
        pool, backend = spec.backend(probe["instance_id"])
        h = health_map[backend.id]
        t = hm.apply_outcome(h, probe["outcome"], probe["reason"],
                             pool.success_threshold,
                             pool.failure_threshold)
        if t:
            expected_transitions.append({
                "probe_seq": probe["seq"], "instance_id": backend.id,
                "from_state": t[0], "to_state": t[1],
                "reason_category": t[2]})

    actual = ledger.transitions(version)
    if len(actual) != len(expected_transitions):
        raise ReplayError(
            f"transition count mismatch: ledger has {len(actual)}, "
            f"replay produced {len(expected_transitions)}",
            path="transitions")
    for i, (exp, act) in enumerate(zip(expected_transitions, actual)):
        for fieldname in ("probe_seq", "instance_id", "from_state",
                          "to_state", "reason_category"):
            if exp[fieldname] != act[fieldname]:
                raise ReplayError(
                    f"transition #{i} field {fieldname!r} mismatch: "
                    f"ledger={act[fieldname]!r} replay={exp[fieldname]!r}",
                    path=f"transitions[{i}].{fieldname}",
                    detail={"instance_id": act["instance_id"],
                            "field": fieldname})

    view = {row["instance_id"]: row for row in ledger.view(version)}
    for iid, h in sorted(health_map.items()):
        row = view.get(iid)
        if row is None:
            raise ReplayError("instance missing from current view",
                              path=f"instance_view.{iid}",
                              detail={"instance_id": iid,
                                      "field": "instance_id"})
        checks = {
            "state": h.state,
            "consec_success": h.consec_success,
            "consec_failure": h.consec_failure,
            "last_reason": h.last_reason,
        }
        for fieldname, expect in checks.items():
            if row[fieldname] != expect:
                raise ReplayError(
                    f"instance {iid!r} field {fieldname!r} mismatch: "
                    f"ledger={row[fieldname]!r} replay={expect!r}",
                    path=f"instance_view.{iid}.{fieldname}",
                    detail={"instance_id": iid, "field": fieldname,
                            "ledger": row[fieldname], "replay": expect})
    return {"version": version,
            "probes_replayed": len(ledger.probes(version)),
            "transitions_verified": len(actual),
            "instances_verified": len(health_map)}
