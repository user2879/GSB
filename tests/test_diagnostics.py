from __future__ import annotations

import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from shard_diag.demo import free_loopback_port, write_demo_workspace
from shard_diag.engine import DiagnosticEngine, compare_deterministic_outcomes
from shard_diag.errors import LeaseConflictError, ReplayConsistencyError, SpecError, SocketTimeoutError
from shard_diag.ledger import Ledger, load_or_replace_snapshot, now_monotonic_ns
from shard_diag.protocol import JsonLineConnection, assert_loopback_port_free, dispatch_job, renew_lease_over_tcp
from shard_diag.quota import choose_error_tie, compute_water, quota_change_scope, resolve_quota, candidate_water_decision
from shard_diag.runtime import NodeCluster
from shard_diag.samples import compute_loss_summary, generate_rows
from shard_diag.spec import load_spec_bundle
from shard_diag.workspace import Workspace, propagation_report


def make_workspace(tmp_path: Path, budget: int = 10) -> tuple[Path, object]:
    root = tmp_path / "ws"
    write_demo_workspace(root, global_budget_units=budget)
    Workspace(root).regenerate_sample_catalogs()
    bundle = load_spec_bundle(root)
    return root, bundle


def test_spec_reports_first_invalid_entry_and_field(tmp_path: Path) -> None:
    root, _ = make_workspace(tmp_path)
    nodes_path = root / "declarations" / "nodes.json"
    doc = json.loads(nodes_path.read_text())
    doc["nodes"][1]["node_id"] = doc["nodes"][0]["node_id"]
    nodes_path.write_text(json.dumps(doc))
    with pytest.raises(SpecError) as exc:
        load_spec_bundle(root)
    assert exc.value.code == "E_SPEC"
    assert exc.value.entry == "nodes[1]"
    assert exc.value.field == "node_id"


def test_quota_parent_must_exist_and_scale_supported(tmp_path: Path) -> None:
    root, _ = make_workspace(tmp_path)
    quota_path = root / "declarations" / "quota_versions.json"
    doc = json.loads(quota_path.read_text())
    doc["versions"][0]["session_scale"] = 7
    quota_path.write_text(json.dumps(doc))
    with pytest.raises(SpecError) as exc:
        load_spec_bundle(root)
    assert exc.value.field == "session_scale"


def test_port_and_socket_timeout_use_real_socket(tmp_path: Path) -> None:
    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = free_loopback_port()
    occupied.bind(("127.0.0.1", port))
    occupied.listen(1)
    try:
        with pytest.raises(Exception) as exc:
            assert_loopback_port_free(port)
        assert exc.value.code == "E_PORT"
    finally:
        occupied.close()

    blackhole = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    blackhole.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    timeout_port = free_loopback_port()
    blackhole.bind(("127.0.0.1", timeout_port))
    blackhole.listen(1)
    stop = threading.Event()

    def accept_forever() -> None:
        while not stop.is_set():
            try:
                conn, _ = blackhole.accept()
            except OSError:
                return
            time.sleep(1)
            conn.close()

    thread = threading.Thread(target=accept_forever, daemon=True)
    thread.start()
    try:
        with pytest.raises(SocketTimeoutError) as exc:
            JsonLineConnection.connect("127.0.0.1", timeout_port, 80).read()
        assert exc.value.code == "E_TIMEOUT"
    finally:
        stop.set()
        blackhole.close()


def test_deterministic_digest_across_batches_order_and_concurrency(tmp_path: Path) -> None:
    root, bundle = make_workspace(tmp_path)
    cluster = NodeCluster(root, bundle.nodes)
    try:
        cluster.start_all()
        node = bundle.node_by_id["node-a"]
        shard = bundle.shard_by_id["shard-a"]
        rows = generate_rows(node, shard, 1)
        expected = compute_loss_summary(node, "shard-a", 1, rows)["loss_digest"]

        for batch_size in (1, 2, 5, 999):
            receipt = dispatch_job(
                host=node["host"], port=node["port"], node_id=node["node_id"], shard=shard,
                quota_version=1, rows=rows, lease_version=1, batch_size=batch_size, timeout_ms=3000,
            )
            assert receipt["loss_summary"]["loss_digest"] == expected
            assert isinstance(receipt["monotonic_ms"], int)

        renewed = renew_lease_over_tcp(
            host=node["host"], port=node["port"], node_id=node["node_id"],
            shard_id="shard-a", expected_version=1, ttl_ms=1000, timeout_ms=1000,
        )
        assert renewed["lease_version"] == 2
        with pytest.raises(LeaseConflictError):
            renew_lease_over_tcp(
                host=node["host"], port=node["port"], node_id=node["node_id"],
                shard_id="shard-a", expected_version=1, ttl_ms=1000, timeout_ms=1000,
            )

        def concurrent_dispatch(index: int) -> str:
            receipt = dispatch_job(
                host=node["host"], port=node["port"], node_id=node["node_id"], shard=shard,
                quota_version=1, rows=rows, lease_version=100 + index, batch_size=(index % 4) + 1,
                timeout_ms=5000,
            )
            return receipt["loss_summary"]["loss_digest"]

        with ThreadPoolExecutor(max_workers=6) as pool:
            digests = list(pool.map(concurrent_dispatch, range(6)))
        assert set(digests) == {expected}

        for shard_id in reversed(bundle.topology_order if hasattr(bundle, "topology_order") else []):
            pass
        for shard in reversed(bundle.shards):
            node = bundle.node_by_id[shard["node_id"]]
            rows = generate_rows(node, shard, 1)
            expected_digest = compute_loss_summary(node, shard["shard_id"], 1, rows)["loss_digest"]
            receipt = dispatch_job(
                host=node["host"], port=node["port"], node_id=node["node_id"], shard=shard,
                quota_version=1, rows=rows, lease_version=7, batch_size=3, timeout_ms=3000,
            )
            assert receipt["loss_summary"]["loss_digest"] == expected_digest

    finally:
        cluster.stop_all()


def test_water_tie_uses_stable_topology_and_marks_error() -> None:
    bundle = load_spec_bundle(".")
    resolved = resolve_quota(bundle.quota_versions, 3)
    water = compute_water(bundle.nodes, resolved, {})
    ready = [bundle.shard_by_id["shard-a"], bundle.shard_by_id["shard-b"]]
    decisions = [
        candidate_water_decision(shard, bundle.node_by_id[shard["node_id"]], resolved, water, ("shard-a", "shard-b"))
        for shard in ready
    ]
    assert all(item["error_affected"] for item in decisions)
    assert choose_error_tie([item["shard_id"] for item in decisions], bundle.shards) == "shard-a"


def test_scheduler_invalidation_byte_preservation_and_full_rebuild(tmp_path: Path) -> None:
    root, bundle = make_workspace(tmp_path)
    db1 = root / "ledger" / "incremental.sqlite"
    ledger1 = Ledger(db1)
    engine1 = DiagnosticEngine(root, ledger1, bundle, quota_version=1)
    cluster = NodeCluster(root, bundle.nodes)
    try:
        cluster.start_all()
        report = engine1.run_all(cluster, batch_size=3, max_seconds=30)
        assert not report["failures"]
        assert len(report["committed_now"]) == 6
        assert report["replay"]["ok"]
        before = engine1.byte_records()

        Workspace(root).replace_sample("shard-a", 0, {"x": [4, -3, 2, 1], "y": 7})
        engine1.reload_declarations()
        propagation = propagation_report(engine1.bundle, changed_shard_ids={"shard-a"})
        assert propagation["affected_shards"] == ["shard-a", "shard-c", "shard-d", "shard-f"]
        assert propagation["unaffected_shards"] == ["shard-b", "shard-e"]
        engine1.mark_invalidated(propagation["affected_shards"], {"kind": "sample_replaced", "shard": "shard-a"})
        incremental = engine1.run_all(cluster, batch_size=2, max_seconds=30)
        assert not incremental["failures"]

        after = engine1.byte_records()
        for shard_id in propagation["unaffected_shards"]:
            assert before[shard_id] == after[shard_id]

        db2 = root / "ledger" / "full-rebuild.sqlite"
        ledger2 = Ledger(db2)
        engine2 = DiagnosticEngine(root, ledger2, quota_version=1)
        full = engine2.run_all(cluster, batch_size=5, max_seconds=30)
        assert not full["failures"]
        mismatches = compare_deterministic_outcomes(ledger1.committed_outcomes(), ledger2.committed_outcomes())
        assert mismatches == []
        assert ledger2.replay_and_verify()["ok"]
        ledger2.close()

        ledger1.conn.execute(
            "UPDATE shard_outcomes SET loss_digest='tampered' WHERE shard_id='shard-b'"
        )
        ledger1.conn.commit()
        with pytest.raises(ReplayConsistencyError) as exc:
            ledger1.replay_and_verify()
        assert exc.value.entry == "shard-b"
        assert exc.value.field == "loss_digest"
        ledger1.conn.rollback()
    finally:
        cluster.stop_all()
        ledger1.close()


def test_force_kill_recovery_and_lease_mutex(tmp_path: Path) -> None:
    root, bundle = make_workspace(tmp_path)
    db = root / "ledger" / "recovery.sqlite"
    ledger = Ledger(db)
    engine = DiagnosticEngine(root, ledger, bundle, quota_version=1)
    cluster = NodeCluster(root, bundle.nodes)
    try:
        cluster.start_node("node-a")
        shard_a = bundle.shard_by_id["shard-a"]
        limits = {"resource_units_cap": 6, "concurrency": 2, "lease_limit": 2}
        lease = ledger.grant_lease(
            shard=shard_a, quota_version=1, ttl_ms=30000, limits=limits, global_budget_units=10
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(engine._execute_granted, lease, batch_size=1, timeout_ms=5000, delay_ms=800)
            time.sleep(0.25)
            killed = cluster.kill_node("node-a")
            assert killed["returncode"] != 0
            with pytest.raises(Exception):
                future.result(timeout=5)

        with pytest.raises(LeaseConflictError):
            ledger.renew_lease(shard_id="shard-a", expected_version=999, ttl_ms=1, now_ns=now_monotonic_ns())

        recovery = ledger.recover_interrupted()
        assert recovery["unknown_shards"] == ["shard-a"]
        assert ledger.active_usage() == {}

        snapshot_info, replaced = load_or_replace_snapshot(root / "ledger" / "session_snapshot.json", ledger)
        ledger.save_snapshot_file(root / "ledger" / "session_snapshot.json", snapshot_version=1)
        snapshot_path = root / "ledger" / "session_snapshot.json"
        envelope = json.loads(snapshot_path.read_text())
        envelope["content_digest"] = "sha256:bad"
        snapshot_path.write_text(json.dumps(envelope))
        _, was_replaced = load_or_replace_snapshot(snapshot_path, ledger)
        assert was_replaced is True

        cluster.start_all()
        report = engine.run_all(cluster, batch_size=3, max_seconds=30)
        assert not report["failures"]
        assert len([s for s in report["readiness"].values() if s["status"] == "committed"]) == 6

        shard_b = bundle.shard_by_id["shard-b"]
        shard_d = bundle.shard_by_id["shard-d"]
        lease_b = ledger.grant_lease(shard=shard_b, quota_version=2, ttl_ms=100000, limits=limits, global_budget_units=10)
        lease_d = ledger.grant_lease(shard=shard_d, quota_version=1, ttl_ms=1, limits=limits, global_budget_units=10)
        ledger.conn.execute("UPDATE leases SET expires_monotonic_ns=? WHERE lease_id=?", (now_monotonic_ns() - 1, lease_b["lease_id"]))
        ledger.conn.execute("UPDATE leases SET expires_monotonic_ns=? WHERE lease_id=?", (now_monotonic_ns() - 1, lease_d["lease_id"]))
        ledger.conn.commit()
        reaped = ledger.reap_expired_and_generation(current_quota_version=2)
        assert "shard-d" in reaped["expired"]
        assert "shard-b" in reaped["expired"]
        assert reaped["generation"] == []
        assert ledger.active_usage() == {}
    finally:
        cluster.stop_all()
        ledger.close()


def test_generation_reaping_is_disjoint_from_expiry(tmp_path: Path) -> None:
    root, bundle = make_workspace(tmp_path)
    ledger = Ledger(root / "ledger" / "mutex.sqlite")
    limits = {"resource_units_cap": 6, "concurrency": 2, "lease_limit": 3}
    try:
        old_future = ledger.grant_lease(shard=bundle.shard_by_id["shard-a"], quota_version=1, ttl_ms=100000, limits=limits, global_budget_units=10)
        old_expired = ledger.grant_lease(shard=bundle.shard_by_id["shard-b"], quota_version=1, ttl_ms=1, limits=limits, global_budget_units=10)
        current_expired = ledger.grant_lease(shard=bundle.shard_by_id["shard-c"], quota_version=2, ttl_ms=1, limits=limits, global_budget_units=10)
        time.sleep(0.002)
        result = ledger.reap_expired_and_generation(current_quota_version=2)
        assert result["expired"] == ["shard-b", "shard-c"]
        assert result["generation"] == ["shard-a"]
        assert not (set(result["expired"]) & set(result["generation"]))
        assert ledger.active_usage() == {}
        with pytest.raises(LeaseConflictError):
            ledger.grant_lease(shard=bundle.shard_by_id["shard-a"], quota_version=2, ttl_ms=10, limits={**limits, "lease_limit": 1}, global_budget_units=0)
    finally:
        ledger.close()


def test_quota_version_change_and_activation_propagation() -> None:
    bundle = load_spec_bundle(".")
    scope = quota_change_scope(bundle.quota_by_version[2], bundle.quota_by_version[1])
    report = propagation_report(bundle, quota_change=scope)
    assert report["affected_shards"] == ["shard-c", "shard-f", "shard-missing"]
    assert all(step["basis"] for step in report["steps"])

    activated = propagation_report(bundle, quota_change=scope, active_quota_version_change=True)
    assert len(activated["affected_shards"]) == len(bundle.shards)
    assert any("activated_quota_version_binding" in step["basis"] for step in activated["steps"])


def test_unknown_upstream_freezes_descendants(tmp_path: Path) -> None:
    root, bundle = make_workspace(tmp_path)
    ledger = Ledger(root / "ledger" / "freeze.sqlite")
    try:
        ledger.conn.execute(
            "INSERT INTO shard_state(shard_id,status,current_lease_id,format_version,content_digest,updated_wall_ms) VALUES(?,?,NULL,1,'test',?)",
            ("shard-a", "unknown", int(time.time() * 1000)),
        )
        ledger.conn.commit()
        engine = DiagnosticEngine(root, ledger, bundle, quota_version=1)
        readiness = engine.evaluate_readiness()
        assert readiness["shard-a"]["status"] == "ready"
        assert readiness["shard-a"]["rerun_required"] is True
        assert readiness["shard-d"]["status"] == "frozen_upstream_failure"
        assert "shard-a" in readiness["shard-d"]["blocked_upstream"]
    finally:
        ledger.close()


def test_three_nodes_run_delayed_shards_in_parallel(tmp_path: Path) -> None:
    root, bundle = make_workspace(tmp_path)
    ledger = Ledger(root / "ledger" / "parallel.sqlite")
    cluster = NodeCluster(root, bundle.nodes)
    try:
        cluster.start_all()
        engine = DiagnosticEngine(root, ledger, bundle, quota_version=1)
        started = time.monotonic()
        report = engine.run_all(
            cluster,
            batch_size=99,
            max_seconds=20,
            delay_ms_by_shard={"shard-c": 500, "shard-d": 500, "shard-e": 500},
        )
        elapsed = time.monotonic() - started
        assert not report["failures"]
        assert elapsed < 2.5
        assert len(report["committed_now"]) == 6
    finally:
        cluster.stop_all()
        ledger.close()
