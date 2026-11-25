from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from cloudshard.adjudicate import Adjudicator
from cloudshard.aggregate import aggregate_samples
from cloudshard.errors import LeaseError, SpecError, VerifyError
from cloudshard.invalidate import build_invalidation_plan
from cloudshard.ledger import Ledger
from cloudshard.ledger import monotonic_ms
from cloudshard.scheduler import DispatchEngine
from cloudshard.spec import parse_spec
from cloudshard.verifier import verify_ledger


def base_spec_dict(**overrides):
    data = {
        "format_version": 1,
        "global_budget_units": 6,
        "lease_ttl_ms": 1000,
        "nodes": [
            {"node_id": "n1", "host": "127.0.0.1", "port": 40101, "concurrency_quota": 2, "lease_limit": 2, "seed": 1, "capacity_version": 1},
            {"node_id": "n2", "host": "127.0.0.1", "port": 40102, "concurrency_quota": 2, "lease_limit": 2, "seed": 2, "capacity_version": 1},
        ],
        "shards": [
            {"shard_id": "a", "dependencies": [], "sample_rows": 24, "resource_units": 2, "priority": 1},
            {"shard_id": "b", "dependencies": ["a"], "sample_rows": 24, "resource_units": 2, "priority": 2},
            {"shard_id": "c", "dependencies": [], "sample_rows": 24, "resource_units": 2, "priority": 0},
        ],
        "revisions": [{"revision": 1, "parent_revision": None, "config": {}}],
        "events": [],
    }
    data.update(overrides)
    return data


def test_spec_reports_first_duplicate_port_location():
    data = base_spec_dict()
    data["nodes"][1]["port"] = data["nodes"][0]["port"]
    with pytest.raises(SpecError) as exc:
        parse_spec(data, path="spec.json")
    assert exc.value.code == "E_SPEC"
    assert exc.value.field == "port"
    assert "nodes[1].port" in (exc.value.path or "")


def test_dependency_cycle_is_espec():
    data = base_spec_dict()
    data["shards"][0]["dependencies"] = ["b"]
    with pytest.raises(SpecError) as exc:
        parse_spec(data)
    assert exc.value.field == "dependencies"
    assert "cycle" in str(exc.value.basis)


def test_revision_parent_cycle_and_monotonic():
    data = base_spec_dict()
    data["revisions"] = [
        {"revision": 2, "parent_revision": 1, "config": {}},
        {"revision": 1, "parent_revision": 2, "config": {}},
    ]
    with pytest.raises(SpecError) as exc:
        parse_spec(data)
    assert exc.value.code == "E_SPEC"


def test_adjudication_unique_and_batch_boundary_bytes():
    spec = parse_spec(
        base_spec_dict(
            global_budget_units=7,
            events=[
                {"seq": 1, "type": "lease_apply", "lease_id": "L1", "shard_id": "a", "node_id": "n1", "at_ms": 0, "ttl_ms": 10},
                {"seq": 2, "type": "lease_apply", "lease_id": "L2", "shard_id": "c", "node_id": "n2", "at_ms": 0, "ttl_ms": 10},
                {"seq": 3, "type": "lease_renew", "lease_id": "L1", "expected_version": 1, "at_ms": 4},
                {"seq": 4, "type": "lease_expire", "lease_id": "L1", "at_ms": 14},
                {"seq": 5, "type": "quota_change", "quota_version": 2, "node_id": "n1", "lease_limit": 1, "revoke_lease_ids": ["L1"], "at_ms": 15},
                {"seq": 6, "type": "lease_release", "lease_id": "L2", "at_ms": 16},
            ],
        )
    )
    full = Adjudicator.from_spec(spec).replay(spec.events).ownership_bytes()
    for batch_size in range(1, 7):
        adj = Adjudicator.from_spec(spec)
        for start in range(0, len(spec.events), batch_size):
            adj.replay(spec.events[start : start + batch_size])
        assert adj.ownership_bytes() == full
    result = json.loads(Adjudicator.from_spec(spec).replay(spec.events).result_dict()["ownership_sequence"].__str__().replace("'", '"')) if False else None
    adj = Adjudicator.from_spec(spec).replay(spec.events)
    reasons = [(r["lease_id"], r["reason"]) for r in adj.ownership_sequence if r["change"] == "reclaimed"]
    assert ("L1", "lease_expired") in reasons
    assert not any(r.get("reason") == "quota_revocation" and r.get("lease_id") == "L1" for r in adj.ownership_sequence)
    duplicate = [d for d in adj.decisions if d["outcome"] == "duplicate_reclamation_skipped"]
    assert duplicate and duplicate[0]["basis"]["lease_id"] == "L1"


def test_optimistic_renew_rejects_stale_version_without_extension():
    spec = parse_spec(base_spec_dict())
    adj = Adjudicator.from_spec(spec)
    adj.handle_event({"seq": 1, "type": "lease_apply", "lease_id": "L", "shard_id": "a", "node_id": "n1", "at_ms": 100, "ttl_ms": 50})
    adj.handle_event({"seq": 2, "type": "lease_renew", "lease_id": "L", "expected_version": 1, "at_ms": 120})
    expires = adj.claims["L"]["expires_at_ms"]
    with pytest.raises(LeaseError) as exc:
        adj.handle_event({"seq": 3, "type": "lease_renew", "lease_id": "L", "expected_version": 1, "at_ms": 125})
    assert exc.value.code == "E_LEASE"
    assert adj.claims["L"]["expires_at_ms"] == expires
    assert adj.claims["L"]["lease_version"] == 2


def test_renewal_count_limit():
    data = base_spec_dict()
    data["nodes"][0]["renewal_limit"] = 0
    spec = parse_spec(data)
    adj = Adjudicator.from_spec(spec)
    adj.handle_event({"seq": 1, "type": "lease_apply", "lease_id": "L", "shard_id": "a", "node_id": "n1", "at_ms": 0, "ttl_ms": 100})
    with pytest.raises(LeaseError):
        adj.handle_event({"seq": 2, "type": "lease_renew", "lease_id": "L", "expected_version": 1, "at_ms": 10})


def test_real_monotonic_expiry_reclaims_once():
    spec = parse_spec(base_spec_dict())
    adj = Adjudicator.from_spec(spec)
    start = monotonic_ms()
    adj.handle_event(
        {"seq": 1, "type": "lease_apply", "lease_id": "L", "shard_id": "a", "node_id": "n1", "at_ms": start, "ttl_ms": 20}
    )
    assert adj.claims["L"]["status"] == "active"
    while monotonic_ms() < start + 25:
        __import__("time").sleep(0.005)
    now = monotonic_ms()
    records = adj.handle_event({"seq": 2, "type": "expire_due", "at_ms": now})
    assert len(records) == 1 and records[0]["reason"] == "lease_expired"
    duplicate = adj.handle_event({"seq": 3, "type": "lease_expire", "lease_id": "L", "at_ms": now + 1})
    assert duplicate == []
    assert any(d["outcome"] == "expire_after_terminal_skipped" for d in adj.decisions)


def test_invalidation_propagates_to_descendants_and_node_refs():
    data = base_spec_dict()
    data["shards"][0]["node_refs"] = ["n1"]
    data["revisions"].append(
        {
            "revision": 2,
            "parent_revision": 1,
            "config": {"multiplier": 2},
            "changed_node_ids": ["n1"],
        }
    )
    spec = parse_spec(data)
    plan = build_invalidation_plan(spec, 2)
    assert plan.invalid_shard_ids == ("a", "b")
    assert plan.stable_topological_order.index("a") < plan.stable_topological_order.index("b")


def test_canceled_root_propagates_and_never_dispatches(tmp_path):
    data = base_spec_dict(
        events=[{"seq": 1, "type": "cancel_shard", "shard_id": "a", "at_ms": 1}]
    )
    spec = parse_spec(data)
    ledger = Ledger(tmp_path / "cancel.sqlite")
    release = threading.Event()
    release.set()
    engine = DispatchEngine(spec, ledger, dispatcher=BlockingDispatcher(release))
    report = engine.run_target(1, [s.shard_id for s in spec.shards])
    assert set(report["canceled_shard_ids"]) == {"a", "b"}
    assert "a" not in report["committed_revisions"]
    assert "b" not in report["committed_revisions"]
    assert report["committed_revisions"].get("c") == 1
    ledger.close()


class BlockingDispatcher:
    def __init__(self, release: threading.Event, entered: threading.Event | None = None):
        self.release = release
        self.entered = entered or threading.Event()
        self.active = 0
        self.max_active = 0
        self.lock = threading.Lock()

    def __call__(self, context):
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            self.entered.set()
        assert self.release.wait(timeout=5)
        node = context["node"]
        summary, digest = aggregate_samples(context["samples"], node, context["config"])
        with self.lock:
            self.active -= 1
        return {
            "receipt": {
                "type": "receipt",
                "node_id": node.node_id,
                "shard_id": context["shard"].shard_id,
                "revision": context["revision"],
                "summary": summary,
                "summary_digest": digest,
                "elapsed_ms": 0.1,
            }
        }


def _run_with_engine(spec_dict, tmp_path: Path):
    spec = parse_spec(spec_dict)
    ledger = Ledger(tmp_path / "ledger.sqlite")
    ledger.initialize_meta("digest", "spec")
    release = threading.Event()
    dispatcher = BlockingDispatcher(release)
    engine = DispatchEngine(spec, ledger, dispatcher=dispatcher)
    result: dict = {}

    def target() -> None:
        result["report"] = engine.run_target(1, [s.shard_id for s in spec.shards])

    thread = threading.Thread(target=target)
    thread.start()
    assert dispatcher.entered.wait(timeout=2)
    return spec, ledger, engine, release, thread, dispatcher, result


def test_global_budget_is_hard_constraint_and_records_wait(tmp_path):
    data = base_spec_dict(global_budget_units=5)
    for node in data["nodes"]:
        node["concurrency_quota"] = 10
        node["lease_limit"] = 10
    data["shards"] = [
        {"shard_id": "a", "dependencies": [], "sample_rows": 8, "resource_units": 2, "priority": 3},
        {"shard_id": "b", "dependencies": [], "sample_rows": 8, "resource_units": 2, "priority": 2},
        {"shard_id": "c", "dependencies": [], "sample_rows": 8, "resource_units": 2, "priority": 1},
    ]
    spec, ledger, engine, release, thread, dispatcher, result = _run_with_engine(data, tmp_path)
    for _ in range(200):
        if engine.adj._used_units() >= 4:
            break
        __import__("time").sleep(0.01)
    assert engine.adj._used_units() <= 5
    assert len([c for c in engine.adj.claims.values() if c["status"] == "active"]) == 2
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()
    ended = [w for w in result["report"]["wait_records"] if w["event"] == "wait_ended"]
    assert ended and all(w["duration_ms"] >= 0 for w in ended)
    ledger.close()


def test_node_concurrency_and_lease_limits_bound_active_claims(tmp_path):
    for field_name, value in (("concurrency_quota", 1), ("lease_limit", 1)):
        data = base_spec_dict(global_budget_units=100)
        data["nodes"] = [
            {"node_id": "n1", "host": "127.0.0.1", "port": 40201, "concurrency_quota": 10, "lease_limit": 10, "seed": 1, "capacity_version": 1}
        ]
        data["nodes"][0][field_name] = value
        data["shards"] = [
            {"shard_id": "a", "dependencies": [], "sample_rows": 8, "resource_units": 1, "priority": 2},
            {"shard_id": "b", "dependencies": [], "sample_rows": 8, "resource_units": 1, "priority": 1},
        ]
        case_path = tmp_path / field_name
        case_path.mkdir(exist_ok=True)
        spec, ledger, engine, release, thread, dispatcher, result = _run_with_engine(data, case_path)
        assert len(engine.adj._active_node_claims("n1")) <= 1
        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        ledger.close()
