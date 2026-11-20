import json

import pytest

from cfgdist.distributor import distribute, evaluate_convergence, NodeDispatchResult
from cfgdist.errors import BizError, ErrorCode
from cfgdist.manifest import manifest_to_json
from cfgdist.planner import build_plan, expected_digest_map
from cfgdist.runtime import NodePool
from cfgdist.session import SessionStore


def _prepare(tmp_path, v3_manifest, node_specs, faults=None):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(manifest_to_json(v3_manifest), encoding="utf-8")
    nodes = [dict(spec) for spec in node_specs]
    if faults:
        for spec in nodes:
            spec["faults"] = faults.get(spec["id"], {})
    plan = build_plan(v3_manifest, [spec["id"] for spec in nodes])
    return manifest_path, nodes, plan


def test_parallel_distribution_converges_and_persists_receipts(
    tmp_path, v3_manifest, node_specs
):
    _, nodes, plan = _prepare(tmp_path, v3_manifest, node_specs)
    db = tmp_path / "sessions.db"
    with NodePool(nodes) as pool:
        with SessionStore(db) as store:
            report = distribute(plan, nodes, store, max_concurrency=3)

    assert report["converged"] is True
    assert report["confirmed_nodes"] == ["node-a", "node-b", "node-c"]

    with SessionStore(db) as store:
        committed = store.latest_committed_version()
        assert committed["config_version"] == plan["config_version"]
        receipts = store.list_receipts(report["session_id"])
        assert {receipt["node"] for receipt in receipts} == {
            "node-a",
            "node-b",
            "node-c",
        }
        transitions = store.list_transitions(report["session_id"])
        assert [item["to_state"] for item in transitions] == [
            "prepared",
            "distributing",
            "committed",
        ]


def test_idempotent_resubmit_does_not_dispatch_again(
    tmp_path, v3_manifest, node_specs
):
    _, nodes, plan = _prepare(tmp_path, v3_manifest, node_specs)
    db = tmp_path / "sessions.db"
    with NodePool(nodes):
        with SessionStore(db) as store:
            first = distribute(plan, nodes, store)
            first_session = first["session_id"]
            receipt_count_after_first = len(store.list_receipts(first_session))

            with pytest.raises(BizError) as exc_info:
                distribute(plan, nodes, store)
            assert exc_info.value.code == ErrorCode.E_IDEMPOTENT

            # No second session and no additional receipts.
            rows = store.conn.execute("SELECT session_id FROM sessions").fetchall()
            assert [row["session_id"] for row in rows] == [first_session]
            assert len(store.list_receipts(first_session)) == receipt_count_after_first


def test_timeout_once_retries_with_distinct_retry_receipt(
    tmp_path, v3_manifest, node_specs
):
    faults = {"node-a": {"timeout_once": True}}
    _, nodes, plan = _prepare(tmp_path, v3_manifest, node_specs, faults=faults)
    db = tmp_path / "sessions.db"
    with NodePool(nodes):
        with SessionStore(db) as store:
            report = distribute(plan, nodes, store)

    node_a = report["results"]["node-a"]
    assert node_a["attempts"] == 2
    assert node_a["retry_used"] is True
    with SessionStore(db) as store:
        receipts = store.list_receipts(report["session_id"])
        node_a_receipts = [r for r in receipts if r["node"] == "node-a"]
    assert node_a_receipts[0]["ok"] == 0
    assert node_a_receipts[1]["ok"] == 1
    assert node_a_receipts[1]["retry"] == 1


def test_digest_mismatch_raises_e_converge_with_stable_deviations(
    tmp_path, v3_manifest, node_specs
):
    faults = {"node-b": {"digest_mismatch": True}}
    _, nodes, plan = _prepare(tmp_path, v3_manifest, node_specs, faults=faults)
    db = tmp_path / "sessions.db"
    expected = expected_digest_map(plan)
    with NodePool(nodes):
        with SessionStore(db) as store:
            with pytest.raises(BizError) as exc_info:
                distribute(plan, nodes, store)

    error = exc_info.value
    assert error.code == ErrorCode.E_CONVERGE
    deviation = error.details["deviations"][0]
    assert deviation["node"] == "node-b"
    assert deviation["expected_digest"] == expected["node-b"]
    assert deviation["actual_digest"] == "0" * 64


def test_convergence_verdict_is_order_independent(v3_manifest, node_specs):
    plan = build_plan(v3_manifest, [spec["id"] for spec in node_specs])
    expected = expected_digest_map(plan)

    def make_results(order):
        results = {}
        for node_id in order:
            results[node_id] = NodeDispatchResult(
                node=node_id,
                ok=True,
                attempts=1,
                last_receipt={"ok": True, "node": node_id, "digest": expected[node_id]},
            )
        return results

    verdict_one = evaluate_convergence(plan, make_results(["node-c", "node-a", "node-b"]))
    verdict_two = evaluate_convergence(plan, make_results(["node-b", "node-c", "node-a"]))
    assert verdict_one == verdict_two
    assert verdict_one["converged"] is True


def test_missing_receipt_reported_as_e_converge(v3_manifest, node_specs):
    plan = build_plan(v3_manifest, [spec["id"] for spec in node_specs])
    results = {
        spec["id"]: NodeDispatchResult(node=spec["id"], ok=True, last_receipt={
            "ok": True,
            "node": spec["id"],
            "digest": plan["slices"][i]["expected_digest"],
        })
        for i, spec in enumerate(node_specs[:2])
    }
    verdict = evaluate_convergence(plan, results)
    assert verdict["converged"] is False
    assert verdict["deviations"][0]["node"] == "node-c"
    assert verdict["deviations"][0]["reason"] == "no successful receipt"
