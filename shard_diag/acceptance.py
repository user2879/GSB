from __future__ import annotations

import json
import platform
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from .demo import free_loopback_port, write_demo_workspace
from .engine import DiagnosticEngine, compare_deterministic_outcomes
from .errors import ReplayConsistencyError, SocketTimeoutError
from .ledger import Ledger, load_or_replace_snapshot, now_monotonic_ns
from .protocol import JsonLineConnection, assert_loopback_port_free
from .quota import candidate_water_decision, choose_error_tie, compute_water, resolve_quota
from .runtime import NodeCluster
from .spec import load_spec_bundle
from .workspace import Workspace, propagation_report


def _unlink_patterns(paths: list[Path]) -> None:
    for path in paths:
        for suffix in ("", "-wal", "-shm"):
            candidate = Path(str(path) + suffix)
            if candidate.exists():
                candidate.unlink()


def _check(name: str, ok: bool, **details: Any) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), **details}


def _port_and_timeout_checks() -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    occupied = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    occupied.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    port = free_loopback_port()
    occupied.bind(("127.0.0.1", port))
    occupied.listen(1)
    try:
        try:
            assert_loopback_port_free(port)
            checks.append(_check("e_port", False))
        except Exception as exc:
            checks.append(_check("e_port", getattr(exc, "code", None) == "E_PORT", port=port))
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
            time.sleep(0.5)
            try:
                conn.close()
            except OSError:
                pass

    thread = threading.Thread(target=accept_forever, daemon=True)
    thread.start()
    try:
        try:
            JsonLineConnection.connect("127.0.0.1", timeout_port, 80).read()
            checks.append(_check("e_timeout_real_socket", False))
        except SocketTimeoutError as exc:
            checks.append(_check("e_timeout_real_socket", exc.code == "E_TIMEOUT", timeout_ms=80))
    finally:
        stop.set()
        blackhole.close()
    return checks


def _root_kill_recovery(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    bundle = load_spec_bundle(project_root)
    db = project_root / "ledger" / "acceptance.sqlite"
    _unlink_patterns([db])
    ledger = Ledger(db)
    engine = DiagnosticEngine(project_root, ledger, bundle, quota_version=3)
    cluster = NodeCluster(project_root, bundle.nodes)
    context: dict[str, Any] = {"database": str(db)}
    try:
        cluster.start_node("node-a")
        shard_a = bundle.shard_by_id["shard-a"]
        resolved = resolve_quota(bundle.quota_versions, 3)
        from .quota import effective_node_quota

        limits = effective_node_quota(bundle.node_by_id["node-a"], resolved)
        lease = ledger.grant_lease(
            shard=shard_a,
            quota_version=3,
            ttl_ms=30000,
            limits=limits,
            global_budget_units=int(resolved["global_budget_units"]),
        )
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(engine._execute_granted, lease, batch_size=1, timeout_ms=5000, delay_ms=800)
            time.sleep(0.25)
            killed = cluster.kill_node("node-a")
            context["killed"] = killed
            dispatch_failed = False
            try:
                future.result(timeout=5)
            except Exception:
                dispatch_failed = True
        checks.append(_check("sigkill_interrupted_dispatch", dispatch_failed, killed=killed))

        ledger.close()
        ledger = Ledger(db)
        engine = DiagnosticEngine(project_root, ledger, bundle, quota_version=3)
        recovery = ledger.recover_interrupted()
        checks.append(
            _check(
                "committed_only_recovery_after_reopen",
                recovery["unknown_shards"] == ["shard-a"] and ledger.active_usage() == {},
                recovery=recovery,
                ledger_reopened=str(db),
            )
        )

        cluster.start_all()
        report = engine.run_all(cluster, batch_size=3, max_seconds=30)
        committed = sorted(s for s, item in report["readiness"].items() if item["status"] == "committed")
        frozen = report["readiness"].get("shard-missing", {})
        checks.append(_check("all_six_runnable_committed", committed == [
            "shard-a", "shard-b", "shard-c", "shard-d", "shard-e", "shard-f"
        ], committed=committed, failures=report["failures"]))
        checks.append(_check(
            "missing_dependency_frozen",
            frozen.get("status") == "frozen_missing_dependency" and frozen.get("missing_upstream") == ["ghost-upstream"],
            frozen=frozen,
        ))
        checks.append(_check("replay_after_recovery", report["replay"]["ok"] is True, replay=report["replay"]))
        context["byte_records_after_recovery"] = engine.byte_records()
        return checks, context
    finally:
        cluster.stop_all()
        ledger.close()


def _mutation_and_rebuild(project_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    workspace_root = project_root / "artifacts" / "acceptance-workspace"
    write_demo_workspace(workspace_root, global_budget_units=10)
    Workspace(workspace_root).regenerate_sample_catalogs()
    bundle = load_spec_bundle(workspace_root)
    db1 = workspace_root / "incremental.sqlite"
    db2 = workspace_root / "full-rebuild.sqlite"
    mutex_db = workspace_root / "mutex.sqlite"
    _unlink_patterns([db1, db2, mutex_db])
    ledger1 = Ledger(db1)
    cluster = NodeCluster(workspace_root, bundle.nodes)
    context: dict[str, Any] = {"workspace": str(workspace_root)}
    try:
        cluster.start_all()
        engine1 = DiagnosticEngine(workspace_root, ledger1, bundle, quota_version=1)
        baseline = engine1.run_all(cluster, batch_size=3, max_seconds=30)
        checks.append(_check("baseline_full_run", not baseline["failures"], committed=baseline["committed_now"]))
        before = engine1.byte_records()

        Workspace(workspace_root).replace_sample("shard-a", 0, {"x": [4, -3, 2, 1], "y": 7})
        engine1.reload_declarations()
        propagation = propagation_report(engine1.bundle, changed_shard_ids={"shard-a"})
        expected_affected = ["shard-a", "shard-c", "shard-d", "shard-f"]
        checks.append(_check(
            "invalidation_propagation",
            propagation["affected_shards"] == expected_affected and propagation["unaffected_shards"] == ["shard-b", "shard-e"],
            propagation=propagation,
        ))
        engine1.mark_invalidated(propagation["affected_shards"], {"kind": "acceptance_sample_replace"})
        incremental = engine1.run_all(cluster, batch_size=2, max_seconds=30)
        checks.append(_check("only_invalidated_rerun", not incremental["failures"], committed_now=incremental["committed_now"]))
        after = engine1.byte_records()
        preserved = {sid: before.get(sid) == after.get(sid) for sid in propagation["unaffected_shards"]}
        checks.append(_check("unaffected_bytes_preserved", all(preserved.values()), preserved=preserved))

        snapshot_path = workspace_root / "snapshot.json"
        ledger1.save_snapshot_file(snapshot_path, snapshot_version=1)
        envelope = json.loads(snapshot_path.read_text(encoding="utf-8"))
        envelope["content_digest"] = "sha256:corrupt"
        snapshot_path.write_text(json.dumps(envelope), encoding="utf-8")
        _, replaced = load_or_replace_snapshot(snapshot_path, ledger1)
        checks.append(_check("snapshot_digest_full_replace", replaced is True))

        ledger2 = Ledger(db2)
        try:
            engine2 = DiagnosticEngine(workspace_root, ledger2, quota_version=1)
            full = engine2.run_all(cluster, batch_size=5, max_seconds=30)
            mismatches = compare_deterministic_outcomes(ledger1.committed_outcomes(), ledger2.committed_outcomes())
            checks.append(_check("blank_full_rebuild_field_match", not full["failures"] and mismatches == [], mismatch_count=len(mismatches), mismatches=mismatches[:5]))

            ledger2.conn.execute("UPDATE shard_outcomes SET loss_digest='tampered-for-e-replay' WHERE shard_id='shard-b'")
            ledger2.conn.commit()
            replay_error = None
            try:
                ledger2.replay_and_verify()
            except ReplayConsistencyError as exc:
                replay_error = exc.to_dict()
            checks.append(_check(
                "e_replay_locates_first_field",
                replay_error is not None and replay_error.get("entry") == "shard-b" and replay_error.get("field") == "loss_digest",
                error=replay_error,
            ))
        finally:
            ledger2.close()

        mutex_ledger = Ledger(mutex_db)
        try:
            limits = {"resource_units_cap": 6, "concurrency": 2, "lease_limit": 3}
            old_generation = mutex_ledger.grant_lease(shard=bundle.shard_by_id["shard-a"], quota_version=1, ttl_ms=100000, limits=limits, global_budget_units=10)
            expired_old = mutex_ledger.grant_lease(shard=bundle.shard_by_id["shard-b"], quota_version=1, ttl_ms=1, limits=limits, global_budget_units=10)
            expired_current = mutex_ledger.grant_lease(shard=bundle.shard_by_id["shard-c"], quota_version=2, ttl_ms=1, limits=limits, global_budget_units=10)
            time.sleep(0.01)
            reaped = mutex_ledger.reap_expired_and_generation(current_quota_version=2)
            checks.append(_check(
                "expiry_generation_mutex",
                reaped["expired"] == ["shard-b", "shard-c"] and reaped["generation"] == ["shard-a"],
                reaped=reaped,
            ))
        finally:
            mutex_ledger.close()
        context["byte_records"] = after
        return checks, context
    finally:
        cluster.stop_all()
        ledger1.close()


def run_acceptance(project_root: str | Path = ".", output: str | Path = "artifacts/acceptance-result.json") -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    output_path = Path(output)
    if not output_path.is_absolute():
        output_path = Path.cwd() / output_path
    checks: list[dict[str, Any]] = []
    checks.extend(_port_and_timeout_checks())

    bundle = load_spec_bundle(project_root)
    resolved = resolve_quota(bundle.quota_versions, max(v["version"] for v in bundle.quota_versions))
    water = compute_water(bundle.nodes, resolved, {})
    tie_decisions = [
        candidate_water_decision(bundle.shard_by_id[sid], bundle.node_by_id[bundle.shard_by_id[sid]["node_id"]], resolved, water, bundle and ("shard-a", "shard-b"))
        for sid in ("shard-a", "shard-b")
    ]
    tie_winner = choose_error_tie(["shard-a", "shard-b"], bundle.shards)
    checks.append(_check(
        "quota_water_tie_adjudication",
        all(item["error_affected"] for item in tie_decisions) and tie_winner == "shard-a",
        decisions=tie_decisions,
        winner=tie_winner,
        error_bounds=water["error_bounds_q"],
    ))

    root_checks, root_context = _root_kill_recovery(project_root)
    checks.extend(root_checks)
    mutation_checks, mutation_context = _mutation_and_rebuild(project_root)
    checks.extend(mutation_checks)

    try:
        import tkinter

        tk_info: dict[str, Any] = {"import": "ok", "tk_version": tkinter.TkVersion}
    except Exception as exc:
        tk_info = {"import": "failed", "error": repr(exc)}

    result = {
        "ok": all(item["ok"] for item in checks),
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "sqlite": __import__("sqlite3").sqlite_version,
            "tkinter": tk_info,
        },
        "check_count": len(checks),
        "passed_count": sum(1 for item in checks if item["ok"]),
        "checks": checks,
        "root_context": root_context,
        "mutation_context": mutation_context,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = output_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str), encoding="utf-8")
    tmp.replace(output_path)
    return result
