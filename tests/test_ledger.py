from __future__ import annotations

import sqlite3

from driftwatch.config import parse_agent_manifest, parse_snapshot_chain
from driftwatch.fixedpoint import judge_snapshot
from driftwatch.ledger import Ledger
from driftwatch.merge import merge_snapshot
from driftwatch.models import Observation

from .conftest import make_criteria


def _setup(payload):
    chain = parse_snapshot_chain(payload)
    criteria = make_criteria()
    agents = parse_agent_manifest(
        {
            "agents": [
                {
                    "agent_id": "a1",
                    "host": "127.0.0.1",
                    "port": 43001,
                    "seed": 1,
                    "aggregation": {"kind": "median", "window": 5, "version": "v1"},
                    "hosts": ["host-a"],
                },
                {
                    "agent_id": "a2",
                    "host": "127.0.0.1",
                    "port": 43002,
                    "seed": 2,
                    "aggregation": {"kind": "mean_floor", "window": 5, "version": "v1"},
                    "hosts": ["host-a"],
                },
            ]
        },
        chain,
        criteria,
    )
    return chain, agents, criteria


def _commit_snapshot(ledger: Ledger, generation: int, chain, snapshot_seq: int, agents, criteria) -> str:
    snapshot = chain.snapshots[snapshot_seq]
    planned = [
        {"snapshot_seq": snapshot_seq, "host_id": host, "metric_id": metric}
        for host in sorted(snapshot.hosts)
        for metric in ("cpu_quota", "memory_pages", "processes")
    ]
    batch_id = ledger.begin_batch(generation, snapshot_seq, planned)
    observations = [
        Observation(agent.agent_id, snapshot_seq, "host-a", metric, 1000 + index, agent.aggregation.version, 1)
        for index, metric in enumerate(("cpu_quota", "memory_pages", "processes"))
        for agent in agents
        if agent.covers("host-a", metric)
    ]
    merged = merge_snapshot(snapshot, observations, criteria.min_sources)
    drift = judge_snapshot(merged, snapshot.hosts, criteria)
    ledger.commit_batch(
        batch_id,
        generation=generation,
        snapshot_seq=snapshot_seq,
        observations=observations,
        merged_results=merged,
        drift_results=drift,
        attempts=[],
        cursor={"generation": generation, "snapshot_seq": snapshot_seq},
    )
    return batch_id


def test_uncommitted_batch_is_aborted_and_requires_recollection(tmp_path, snapshot_payload):
    chain, agents, criteria = _setup(snapshot_payload)
    db = tmp_path / "ledger.sqlite3"
    ledger = Ledger(db)
    first = ledger.reconcile_session(chain, agents, criteria)
    planned = [
        {"snapshot_seq": 1, "host_id": "host-a", "metric_id": metric}
        for metric in ("cpu_quota", "memory_pages", "processes")
    ]
    batch_id = ledger.begin_batch(first.generation, 1, planned)
    assert ledger.active_generation() == first.generation
    ledger.close()

    reopened = Ledger(db)
    assert reopened.reconcile_session(chain, agents, criteria).generation == first.generation
    recovered = reopened.recover_incomplete()
    assert [item["batch_id"] for item in recovered] == [batch_id] * 3
    unknown = reopened.required_recollection(first.generation)
    assert len(unknown) == 3
    assert all(item["required_recollect"] == 1 for item in unknown)
    with sqlite3.connect(db) as raw:
        status = raw.execute("SELECT status FROM batches WHERE batch_id=?", (batch_id,)).fetchone()[0]
    assert status == "ABORTED"
    assert reopened.committed_snapshot_seqs(first.generation) == ()

    committed = _commit_snapshot(reopened, first.generation, chain, 1, agents, criteria)
    assert committed
    assert reopened.required_recollection(first.generation) == []
    reopened.close()


def test_generation_replacement_is_whole_view_and_stale_is_explicit(tmp_path, snapshot_payload):
    chain, agents, criteria = _setup(snapshot_payload)
    db = tmp_path / "ledger.sqlite3"
    ledger = Ledger(db)
    first = ledger.reconcile_session(chain, agents, criteria)
    _commit_snapshot(ledger, first.generation, chain, 1, agents, criteria)
    assert len(ledger.session_snapshot(first.generation)["merged_view"]) == 6

    payload = {
        "agents": [
            {
                "agent_id": "a1",
                "host": "127.0.0.1",
                "port": 43001,
                "seed": 1,
                "aggregation": {"kind": "median", "window": 7, "version": "v2"},
                "hosts": ["host-a"],
            },
            {
                "agent_id": "a2",
                "host": "127.0.0.1",
                "port": 43002,
                "seed": 2,
                "aggregation": {"kind": "mean_floor", "window": 5, "version": "v1"},
                "hosts": ["host-a"],
            },
        ]
    }
    new_agents = parse_agent_manifest(payload, chain, criteria)
    second = ledger.reconcile_session(chain, new_agents, criteria)
    assert second.replaced is True
    assert second.generation != first.generation
    assert ledger.active_generation() == second.generation
    assert ledger.session_snapshot(second.generation)["merged_view"] == []
    assert any(item["kind"] == "aggregation_version_changed" for item in second.differences)
    assert {item.metric_id for item in second.affected} == {"cpu_quota", "memory_pages", "processes"}
    with sqlite3.connect(db) as raw:
        stale = raw.execute(
            "SELECT host_id, stale FROM merged_rows WHERE generation=? ORDER BY host_id, metric_id",
            (first.generation,),
        ).fetchall()
    assert all(int(row[1]) == 1 for row in stale if row[0] == "host-a")
    assert all(int(row[1]) == 0 for row in stale if row[0] == "host-b")
    ledger.close()


def test_session_digest_mismatch_forces_replacement(tmp_path, snapshot_payload):
    chain, agents, criteria = _setup(snapshot_payload)
    db = tmp_path / "ledger.sqlite3"
    ledger = Ledger(db)
    first = ledger.reconcile_session(chain, agents, criteria)
    _commit_snapshot(ledger, first.generation, chain, 1, agents, criteria)
    with sqlite3.connect(db) as raw:
        raw.execute(
            "UPDATE session_snapshots SET merged_view_json=? WHERE generation=?",
            ("[]", first.generation),
        )
        raw.commit()
    second = ledger.reconcile_session(chain, agents, criteria)
    assert second.replaced is True
    assert any(item["kind"] == "session_snapshot_mismatch" for item in second.differences)
    assert ledger.session_snapshot(second.generation)["merged_view"] == []
    ledger.close()
