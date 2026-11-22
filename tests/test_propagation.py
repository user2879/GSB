from __future__ import annotations

from driftwatch.propagation import derive_affected

from .conftest import make_agent, make_criteria


def test_agent_set_and_aggregation_version_propagation(snapshot_payload):
    from driftwatch.config import parse_snapshot_chain

    chain = parse_snapshot_chain(snapshot_payload)
    criteria = make_criteria()
    old = (
        make_agent("a1", 42001, hosts=("host-a",)),
        make_agent("a2", 42002, version="v1", hosts=("host-a", "host-b")),
    )
    new = (
        make_agent("a2", 42002, version="v2", hosts=("host-a", "host-b")),
        make_agent("a3", 42003, hosts=("host-b",)),
    )
    propagation = derive_affected(chain, chain, old, new, criteria, criteria)
    by_key = {(item.snapshot_seq, item.host_id, item.metric_id): item for item in propagation.affected}
    assert by_key[(1, "host-a", "cpu_quota")].reasons == (
        "agent_removed:a1",
        "aggregation_version_changed:a2:v1->v2",
    )
    assert "agent_removed:a1" in by_key[(2, "host-a", "cpu_quota")].reasons
    assert "aggregation_version_changed:a2:v1->v2" in by_key[(2, "host-b", "processes")].reasons
    assert "agent_added:a3" not in by_key[(2, "host-a", "cpu_quota")].reasons
    assert "agent_added:a3" in by_key[(2, "host-b", "cpu_quota")].reasons
    kinds = {item["kind"] for item in propagation.differences}
    assert {"agent_removed", "agent_added", "aggregation_version_changed"} <= kinds
    assert propagation.affected == sorted(
        propagation.affected, key=lambda item: (item.snapshot_seq, item.host_id, item.metric_id)
    )


def test_snapshot_added_marks_only_new_snapshot(snapshot_payload):
    from driftwatch.config import parse_snapshot_chain

    old_chain = parse_snapshot_chain(snapshot_payload)
    added = {
        "seq": 3,
        "parent_seq": 2,
        "hosts": {
            "host-c": {
                "expected": {"cpu_quota": 50000, "memory_pages": 32000, "processes": 300},
                "tolerance_bps": 700,
            }
        },
    }
    new_payload = {"snapshots": snapshot_payload["snapshots"] + [added]}
    new_chain = parse_snapshot_chain(new_payload)
    agents = (make_agent("a1", 42011, hosts=("host-a", "host-b", "host-c")),)
    propagation = derive_affected(old_chain, new_chain, agents, agents, make_criteria(), make_criteria())
    assert propagation.affected
    assert {item.snapshot_seq for item in propagation.affected} == {3}
    assert all("snapshot_added:3" in item.reasons for item in propagation.affected)
    assert any(item["kind"] == "snapshot_added" and item["snapshot_seq"] == 3 for item in propagation.differences)
