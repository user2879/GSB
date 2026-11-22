#!/usr/bin/env python3
"""Sequential, repeatable demonstrations for the Meshwork workbench.

Scenario 1: cold-start weighted parallel dispatch under dual budgets.
Scenario 2: circuit breaker trigger, real cooldown and half-open recovery.
Scenario 3: snapshot version repair, forced-termination recovery and replay.
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from meshwork.canonical import canonical_json  # noqa: E402
from meshwork.engine import MeshRuntime  # noqa: E402
from meshwork.errors import MeshError  # noqa: E402
from meshwork.ledger import Ledger  # noqa: E402
from meshwork.logging_utils import StructuredLogger  # noqa: E402
from meshwork.model import load_requests, load_spec  # noqa: E402
from meshwork.protocol import check_port_free  # noqa: E402
from meshwork.snapshot import SnapshotStore  # noqa: E402

SAMPLE_SPEC = ROOT / "samples" / "sample_spec.json"
ARTIFACTS = ROOT / "artifacts" / "demo"
LOG_PATH = ARTIFACTS / "demo.jsonl"


def print_section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def wait_for_ports_free(spec: Any, timeout: float = 4.0) -> None:
    deadline = time.monotonic() + timeout
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            for service in spec.ordered_services():
                for instance in service.instances:
                    check_port_free(instance.host, instance.port)
            return
        except MeshError as exc:
            last_error = exc
            time.sleep(0.1)
    raise RuntimeError(f"ports did not become free: {last_error}")


def scenario_cold_start(logger: StructuredLogger) -> dict[str, Any]:
    print_section("SCENARIO 1 · cold start, weighted selection, dual budgets")
    spec = load_spec(SAMPLE_SPEC)
    revision = spec.active_revision(1)
    requests = load_requests(ROOT / "samples" / "requests.json", spec.services.keys())
    ledger = Ledger(ARTIFACTS / "cold.db", reset=True, logger=logger)
    runtime = MeshRuntime(
        spec,
        revision,
        ledger,
        SnapshotStore(ARTIFACTS / "cold-snapshots", logger),
        logger,
        "demo-cold",
    )
    try:
        runtime.start()
        report = runtime.run_requests(requests)
        snapshot = runtime.write_snapshot("demo_cold_complete")
    finally:
        runtime.stop()
    budget = report["budget"]
    stats = report["stats"]["overall"]
    selection_preview = [
        {
            "request_id": row["request_id"],
            "attempt": row["attempt_no"],
            "instance": row["instance_id"],
            "select_seq": row["select_seq"],
            "outcome": row["outcome"],
        }
        for row in ledger.latest_attempts()[:10]
    ]
    result = {
        "request_count": report["request_count"],
        "successes": stats["successes"],
        "final_distribution": stats["final_distribution"],
        "budget": budget,
        "selection_preview": selection_preview,
        "fixed_stats": report["stats"],
        "snapshot": snapshot.path,
        "ledger_digest": report["ledger"]["content_digest"],
    }
    assert_true(stats["successes"] == report["request_count"], "cold scenario must succeed")
    assert_true(budget["peak_global"] <= spec.session.global_concurrency, "global budget exceeded")
    for service in spec.ordered_services():
        assert_true(
            budget["peak_service"][service.id] <= service.quota,
            f"service quota exceeded for {service.id}",
        )
    print(
        f"requests={report['request_count']} successes={stats['successes']} "
        f"peak_global={budget['peak_global']}/{spec.session.global_concurrency}"
    )
    print(
        "peak_service="
        + json.dumps(budget["peak_service"], sort_keys=True)
        + " quotas="
        + json.dumps({item.id: item.quota for item in spec.ordered_services()}, sort_keys=True)
    )
    print(
        "sample selections="
        + json.dumps(selection_preview[:5], ensure_ascii=False, sort_keys=True)
    )
    ledger.close()
    logger.info("demo_scenario_complete", scenario="cold_start", result="pass")
    return result


def scenario_circuit(logger: StructuredLogger) -> dict[str, Any]:
    print_section("SCENARIO 2 · circuit trigger, real cooldown, half-open recovery")
    spec = load_spec(SAMPLE_SPEC)
    revision = spec.active_revision(1)
    fail_requests = load_requests(
        ROOT / "samples" / "circuit_fail_requests.json", spec.services.keys()
    )
    ok_requests = load_requests(
        ROOT / "samples" / "circuit_recover_requests.json", spec.services.keys()
    )
    ledger = Ledger(ARTIFACTS / "circuit.db", reset=True, logger=logger)
    runtime = MeshRuntime(
        spec,
        revision,
        ledger,
        SnapshotStore(ARTIFACTS / "circuit-snapshots", logger),
        logger,
        "demo-circuit",
    )
    try:
        runtime.start()
        fail_report = runtime.run_requests(fail_requests)
        breaker = runtime.breakers.get("fragile-a")
        open_view = breaker.view()
        attempts_after_fail = ledger.latest_attempts()
        cool_down_ms = breaker.cooldown_remaining_ms()
        time.sleep((cool_down_ms + 35) / 1000.0)
        before_probe = breaker.view()
        ok_report = runtime.run_requests(ok_requests)
        closed_view = breaker.view()
        transitions = [
            {
                "previous": item.previous,
                "current": item.current,
                "reason": item.reason,
            }
            for item in breaker.transitions
        ]
        snapshot = runtime.write_snapshot("demo_circuit_complete")
    finally:
        runtime.stop()
    result = {
        "fail_result": fail_report["results"],
        "attempts_after_fail": [
            {
                "attempt_no": row["attempt_no"],
                "outcome": row["outcome"],
                "error_code": row["error_code"],
                "budget_used": bool(row["budget_used"]),
            }
            for row in attempts_after_fail
        ],
        "open_view": open_view,
        "before_probe": before_probe,
        "recover_result": ok_report["results"],
        "closed_view": closed_view,
        "transitions": transitions,
        "snapshot": snapshot.path,
    }
    assert_true(open_view["state"] == "OPEN", "breaker did not open")
    assert_true(
        any(row["outcome"] == "circuit_open" and not row["budget_used"] for row in attempts_after_fail),
        "no fast-fail attempt recorded",
    )
    assert_true(closed_view["state"] == "CLOSED", "breaker did not recover")
    assert_true(
        any(item["current"] == "HALF_OPEN" for item in transitions),
        "breaker never entered half-open",
    )
    print("attempts=" + json.dumps(result["attempts_after_fail"], sort_keys=True))
    print("open=" + json.dumps(open_view, sort_keys=True))
    print("after_cooldown=" + json.dumps(before_probe, sort_keys=True))
    print("recover=" + json.dumps(result["recover_result"], sort_keys=True))
    print("closed=" + json.dumps(closed_view, sort_keys=True))
    ledger.close()
    logger.info("demo_scenario_complete", scenario="circuit", result="pass")
    return result


def run_replay_snapshot_source(logger: StructuredLogger) -> dict[str, Any]:
    print_section("SCENARIO 3A · deterministic source session and snapshot restore")
    spec = load_spec(SAMPLE_SPEC)
    revision = spec.active_revision(1)
    requests = load_requests(ROOT / "samples" / "replay_requests.json", spec.services.keys())
    ledger_path = ARTIFACTS / "replay-source.db"
    snapshot_dir = ARTIFACTS / "replay-source-snapshots"
    ledger = Ledger(ledger_path, reset=True, logger=logger)
    runtime = MeshRuntime(
        spec,
        revision,
        ledger,
        SnapshotStore(snapshot_dir, logger),
        logger,
        "demo-replay-source",
    )
    try:
        runtime.start()
        source_report = runtime.run_requests(requests)
        source_snapshot = runtime.write_snapshot("demo_replay_source")
    finally:
        runtime.stop()
    ledger.close()

    reopened = Ledger(ledger_path, logger=logger)
    reopened_runtime = MeshRuntime(
        spec,
        revision,
        reopened,
        SnapshotStore(snapshot_dir, logger),
        logger,
        "demo-replay-reopen",
    )
    try:
        restore_validation = reopened_runtime.validate_or_replace_snapshot()
        assert_true(restore_validation["status"] == "valid", "valid snapshot was not restored")
        wrong_version = replace(
            reopened_runtime.snapshot_state(),
            route_revision=999,
            instance_version="bogus-instance-version",
        )
        wrong_envelope = reopened_runtime.snapshot_store.write(wrong_version)
        time.sleep(0.002)
        replacement_validation = reopened_runtime.validate_or_replace_snapshot()
        assert_true(replacement_validation["status"] == "replaced", "bad version not replaced")
        assert_true(
            replacement_validation["error"]["code"] == "E_SNAPSHOT",
            "bad snapshot did not report E_SNAPSHOT",
        )
        digest_copy = ARTIFACTS / "snapshot-digest-mismatch.json"
        envelope_data = json.loads(Path(wrong_envelope.path).read_text(encoding="utf-8"))
        envelope_data["payload"]["completed_cursor"] += 1
        digest_copy.write_text(canonical_json(envelope_data), encoding="utf-8")
        snapshot_error: dict[str, Any]
        try:
            reopened_runtime.snapshot_store.load(digest_copy)
            raise AssertionError("digest mismatch snapshot unexpectedly loaded")
        except MeshError as exc:
            snapshot_error = exc.to_dict()
            assert_true(exc.code == "E_SNAPSHOT", "expected E_SNAPSHOT")
        latest = reopened_runtime.snapshot_store.load_latest()
        assert_true(latest is not None, "replacement snapshot missing")
    finally:
        reopened.close()
    print(
        "restore="
        + restore_validation["status"]
        + " replacement="
        + replacement_validation["status"]
        + " replacement_diff="
        + json.dumps(replacement_validation["diff"], sort_keys=True)
    )
    print(
        "digest_mismatch_error="
        + snapshot_error["code"]
        + " latest_snapshot="
        + str(latest.path if latest else None)
    )
    return {
        "source_requests": source_report["request_count"],
        "source_successes": source_report["stats"]["overall"]["successes"],
        "source_snapshot": source_snapshot.path,
        "restore_validation": restore_validation,
        "replacement_validation": replacement_validation,
        "digest_mismatch_error": snapshot_error,
        "latest_snapshot": latest.path if latest else None,
        "ledger": str(ledger_path),
    }


def scenario_forced_termination(logger: StructuredLogger) -> dict[str, Any]:
    print_section("SCENARIO 3B · forced termination, unknown intents and ledger recovery")
    spec = load_spec(SAMPLE_SPEC)
    ledger_path = ARTIFACTS / "crash.db"
    snapshot_dir = ARTIFACTS / "crash-snapshots"
    command = [
        sys.executable,
        "-m",
        "meshwork.cli",
        "run",
        "--spec",
        str(SAMPLE_SPEC),
        "--requests",
        str(ROOT / "samples" / "crash_requests.json"),
        "--ledger",
        str(ledger_path),
        "--snapshot-dir",
        str(snapshot_dir),
        "--revision",
        "1",
        "--reset",
        "--session-id",
        "demo-crash",
        "--log-file",
        str(ARTIFACTS / "crash-child.jsonl"),
    ]
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    intents = 0
    for _ in range(250):
        time.sleep(0.02)
        if process.poll() is not None:
            break
        if ledger_path.exists():
            try:
                with sqlite3.connect(str(ledger_path), timeout=0.2) as connection:
                    intents = int(
                        connection.execute("SELECT COUNT(*) FROM attempt_intents").fetchone()[0]
                    )
            except sqlite3.Error:
                intents = 0
            if intents > 0:
                break
    if process.poll() is not None or intents <= 0:
        process.kill()
        process.communicate(timeout=5)
        raise RuntimeError("failed to catch the workbench in-flight before kill")
    process.kill()
    process.communicate(timeout=5)
    wait_for_ports_free(spec)
    recover_command = [
        sys.executable,
        "-m",
        "meshwork.cli",
        "recover",
        "--spec",
        str(SAMPLE_SPEC),
        "--ledger",
        str(ledger_path),
        "--snapshot-dir",
        str(snapshot_dir),
        "--log-file",
        str(ARTIFACTS / "crash-recover.jsonl"),
    ]
    recovered = subprocess.run(
        recover_command,
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=30,
    )
    if recovered.returncode != 0:
        raise RuntimeError(f"recovery failed: {recovered.stderr}")
    payload = json.loads(recovered.stdout)
    assert_true(payload["exit_report_available"] is False, "must not claim a pre-crash exit report")
    assert_true(payload["pre_crash_exit_report"] is None, "pre-crash report must be absent")
    assert_true(len(payload["recovery"]["unknown_attempts"]) >= 1, "no unknown attempt marked")
    assert_true(
        payload["recovery"]["state_source"] == "committed_ledger_attempts_only",
        "recovery state source is not ledger-only",
    )
    print(
        "killed_after_intents="
        + str(intents)
        + " unknown_attempts="
        + str(len(payload["recovery"]["unknown_attempts"]))
        + " rerun="
        + str(payload["rerun"]["request_count"])
        + " exit_report_available="
        + str(payload["exit_report_available"])
    )
    logger.info("demo_scenario_complete", scenario="forced_termination", result="pass")
    return {
        "killed_intents": intents,
        "exit_report_available": payload["exit_report_available"],
        "pre_crash_exit_report": payload["pre_crash_exit_report"],
        "recovery": payload["recovery"],
        "rerun_request_count": payload["rerun"]["request_count"],
        "ledger": str(ledger_path),
    }


def scenario_replay(logger: StructuredLogger) -> dict[str, Any]:
    print_section("SCENARIO 3C · full request-sequence replay and E_REPLAY locator")
    source = ARTIFACTS / "replay-source.db"
    replay_ledger = ARTIFACTS / "replay-actual.db"
    command = [
        sys.executable,
        "-m",
        "meshwork.cli",
        "replay",
        "--spec",
        str(SAMPLE_SPEC),
        "--source-ledger",
        str(source),
        "--replay-ledger",
        str(replay_ledger),
        "--snapshot-dir",
        str(ARTIFACTS / "replay-actual-snapshots"),
        "--log-file",
        str(ARTIFACTS / "replay.jsonl"),
    ]
    completed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=30)
    if completed.returncode != 0:
        raise RuntimeError(f"replay failed: {completed.stderr}")
    payload = json.loads(completed.stdout)
    comparison = payload["replay"]
    assert_true(comparison["status"] == "consistent", "replay was not consistent")
    print("replay=" + json.dumps(comparison, sort_keys=True))

    tampered_path = ARTIFACTS / "replay-tampered.db"
    shutil.copy2(replay_ledger, tampered_path)
    with sqlite3.connect(str(tampered_path)) as connection:
        connection.execute("UPDATE attempts SET outcome='error' WHERE attempt_id=1")
        connection.commit()
    from meshwork.replay import compare_ledgers

    expected = Ledger(source, logger=logger)
    tampered = Ledger(tampered_path, verify=False, logger=logger)
    try:
        try:
            compare_ledgers(expected, tampered)
            raise AssertionError("tampered replay unexpectedly compared equal")
        except MeshError as exc:
            assert_true(exc.code == "E_REPLAY", "expected E_REPLAY")
            replay_error = exc.to_dict()
    finally:
        expected.close()
        tampered.close()
    print("tamper_locator=" + json.dumps(replay_error, sort_keys=True))
    logger.info("demo_scenario_complete", scenario="replay", result="pass")
    return {
        "comparison": comparison,
        "tamper_locator": replay_error,
        "replay_ledger": str(replay_ledger),
    }


def main() -> int:
    if ARTIFACTS.exists():
        shutil.rmtree(ARTIFACTS)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    logger = StructuredLogger(LOG_PATH)
    logger.info(
        "demo_started",
        python=sys.version.split()[0],
        spec=str(SAMPLE_SPEC),
        scope="local_loopback_only",
    )
    spec = load_spec(SAMPLE_SPEC)
    wait_for_ports_free(spec)
    results: dict[str, Any] = {}
    try:
        results["scenario_1_cold_start"] = scenario_cold_start(logger)
        wait_for_ports_free(spec)
        results["scenario_2_circuit"] = scenario_circuit(logger)
        wait_for_ports_free(spec)
        results["scenario_3_snapshot"] = run_replay_snapshot_source(logger)
        wait_for_ports_free(spec)
        results["scenario_3_forced_termination"] = scenario_forced_termination(logger)
        wait_for_ports_free(spec)
        results["scenario_3_replay"] = scenario_replay(logger)
        results["status"] = "PASS"
        results["ports"] = [
            {
                "service": service.id,
                "instance": instance.id,
                "host": instance.host,
                "port": instance.port,
            }
            for service in spec.ordered_services()
            for instance in service.instances
        ]
        verification = ARTIFACTS / "verification.json"
        verification.write_text(
            json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        logger.info("demo_finished", result="PASS", verification=str(verification))
        print_section("DEMO RESULT")
        print(f"PASS · verification written to {verification}")
        return 0
    except Exception as exc:
        logger.error("demo_failed", code="E_DEMO", message=str(exc))
        print_section("DEMO RESULT")
        print(f"FAIL · {exc}")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
