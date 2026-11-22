from __future__ import annotations

import json
import os
import socket
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any

from configdist.canonical import canonical_json, content_digest
from configdist.errors import AppError
from configdist.evolution import evolve_spec_data
from configdist.ledger import Ledger
from configdist.orchestrator import NodeManager
from configdist.planner import compile_plan, plan_bytes
from configdist.scheduler import schedule_plan
from configdist.service import (
    execute_run,
    preview_ledger,
    recover_ledger,
    replay_restored_snapshot,
    restore_snapshot,
    verify_run,
)
from configdist.snapshot import load_snapshot
from configdist.validation import validate_spec


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample" / "spec.json"


def free_ports(count: int) -> list[int]:
    sockets: list[socket.socket] = []
    ports: list[int] = []
    try:
        while len(ports) < count:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            port = int(sock.getsockname()[1])
            if port not in ports:
                sockets.append(sock)
                ports.append(port)
    finally:
        for sock in sockets:
            sock.close()
    return ports


def sample_raw(ports: list[int] | None = None) -> dict[str, Any]:
    raw = json.loads(SAMPLE.read_text(encoding="utf-8"))
    if ports is not None:
        if len(ports) != len(raw["clusters"]):
            raise ValueError("one port per cluster is required")
        for cluster, port in zip(raw["clusters"], ports):
            cluster["port"] = port
    return raw


def single_cluster_raw(port: int) -> dict[str, Any]:
    raw = sample_raw([port, *free_ports(2)])
    raw["tasks"] = [raw["tasks"][0]]
    raw["tasks"][0]["target_clusters"] = ["edge-a"]
    raw["tasks"][0]["shards"] = 2
    return raw


class ValidationTests(unittest.TestCase):
    def test_first_error_and_cycle_classification(self) -> None:
        raw = sample_raw()
        raw["clusters"][0]["host"] = "localhost"
        raw["clusters"][1]["id"] = raw["clusters"][0]["id"]
        error = self._error(raw)
        self.assertEqual(error.code, "E_SPEC")
        self.assertEqual(error.field, "host")
        self.assertEqual(error.location, "$.clusters[0].host")

        raw = sample_raw()
        raw["tasks"][0]["shards"] = -1
        error = self._error(raw)
        self.assertEqual((error.code, error.field, error.location), ("E_SPEC", "shards", "$.tasks[0].shards"))

        raw = sample_raw()
        raw["tasks"][0]["config_keys"] = []
        error = self._error(raw)
        self.assertEqual((error.code, error.field), ("E_SPEC", "config_keys"))

        raw = sample_raw()
        raw["revisions"] = [item for item in raw["revisions"] if item["cluster_id"] != "edge-b"]
        base = next(item for item in raw["revisions"] if item["cluster_id"] == "edge-a" and item["version"] == 1)
        raw["revisions"].extend(
            [
                {"cluster_id": "edge-b", "version": 1, "parent": 2, "config": base["config"]},
                {"cluster_id": "edge-b", "version": 2, "parent": 1, "config": base["config"]},
            ]
        )
        error = self._error(raw)
        self.assertEqual(error.code, "E_CYCLE")
        self.assertEqual(error.field, "parent")

        raw = sample_raw()
        next(item for item in raw["revisions"] if item["cluster_id"] == "edge-a" and item["version"] == 2)["parent"] = 99
        error = self._error(raw)
        self.assertEqual((error.code, error.field), ("E_SPEC", "parent"))

    def _error(self, raw: dict[str, Any]) -> AppError:
        with self.assertRaises(AppError) as caught:
            validate_spec(raw)
        return caught.exception


class PlanTests(unittest.TestCase):
    def test_plan_is_byte_identical_under_edit_order_permutation(self) -> None:
        raw = sample_raw()
        spec = validate_spec(raw)
        expected = compile_plan(spec)
        permuted = json.loads(json.dumps(raw))
        permuted["clusters"].reverse()
        permuted["revisions"].reverse()
        permuted["tasks"].reverse()
        for task in permuted["tasks"]:
            task["config_keys"].reverse()
            task["target_clusters"].reverse()
        for cluster in permuted["clusters"]:
            cluster["nodes"].reverse()
        actual = compile_plan(validate_spec(permuted))
        self.assertEqual(plan_bytes(expected), plan_bytes(actual))
        self.assertEqual(expected["plan_digest"], actual["plan_digest"])
        self.assertEqual(
            [shard["shard_id"] for shard in expected["shards"]],
            [shard["shard_id"] for shard in actual["shards"]],
        )
        self.assertEqual(
            [shard["shard_id"] for shard in expected["shards"]],
            sorted(shard["shard_id"] for shard in expected["shards"]),
        )
        for shard in expected["shards"]:
            self.assertEqual(shard["config_keys"], sorted(shard["config_keys"]))
            self.assertEqual(shard["revision_version"], spec.current_revisions()[shard["cluster_id"]].version)


class SchedulerTests(unittest.TestCase):
    def test_triple_budget_is_invariant_at_every_start(self) -> None:
        spec = validate_spec(sample_raw())
        plan = compile_plan(spec)
        lock = threading.Lock()
        active_resource = 0
        active_cluster: dict[str, int] = {}
        active_task: dict[str, int] = {}
        peaks = {"resource": 0, "cluster": {}, "task": {}}

        def dispatch(shard: dict[str, Any]) -> dict[str, Any]:
            nonlocal active_resource
            with lock:
                active_resource += shard["resource_units"]
                active_cluster[shard["cluster_id"]] = active_cluster.get(shard["cluster_id"], 0) + 1
                active_task[shard["task_id"]] = active_task.get(shard["task_id"], 0) + 1
                self.assertLessEqual(active_resource, plan["global_resource_units"])
                self.assertLessEqual(active_cluster[shard["cluster_id"]], shard["cluster_quota"])
                self.assertLessEqual(active_task[shard["task_id"]], shard["max_parallel"])
                peaks["resource"] = max(peaks["resource"], active_resource)
                peaks["cluster"][shard["cluster_id"]] = max(
                    peaks["cluster"].get(shard["cluster_id"], 0), active_cluster[shard["cluster_id"]]
                )
                peaks["task"][shard["task_id"]] = max(
                    peaks["task"].get(shard["task_id"], 0), active_task[shard["task_id"]]
                )
            time.sleep(0.012)
            with lock:
                active_resource -= shard["resource_units"]
                active_cluster[shard["cluster_id"]] -= 1
                active_task[shard["task_id"]] -= 1
            return {
                "elapsed_ms": 1,
                "aggregate": {
                    "scan_rows": 1,
                    "byte_digest": shard["shard_id"],
                    "summary_digest": shard["shard_id"],
                    "groups": [],
                },
            }

        result = schedule_plan(plan, dispatch)
        self.assertEqual(result["counts"], {"total": 18, "completed": 18, "failed": 0, "blocked": 0})
        self.assertEqual(result["peak"]["resource_units"], 7)
        self.assertEqual(result["peak"]["cluster_concurrency"], {"edge-a": 2, "edge-b": 1, "edge-c": 2})
        self.assertEqual(result["peak"]["task_concurrency"], {"audit-policy": 2, "deploy-core": 3})
        for trace in result["trace"]:
            self.assertIsInstance(trace["wait_started_ms"], int)
            self.assertIsInstance(trace["wait_ended_ms"], int)
            self.assertLessEqual(trace["wait_started_ms"], trace["wait_ended_ms"])

    def test_zero_quota_and_oversized_shards_are_decidable_blocked(self) -> None:
        raw = sample_raw()
        raw["clusters"][0]["concurrency"] = 0
        raw["tasks"][0]["resource_units"] = raw["budgets"]["global_resource_units"] + 1
        plan = compile_plan(validate_spec(raw))
        result = schedule_plan(plan, lambda shard: self.fail("blocked shard must not dispatch"))
        self.assertGreater(result["counts"]["blocked"], 0)
        self.assertEqual(result["counts"]["completed"], 0)
        codes = {trace["error_code"] for trace in result["trace"] if trace["outcome"] == "blocked"}
        self.assertEqual(codes, {"E_BUDGET"})


class LoopbackAndLedgerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.dir = Path(self.temp.name)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_real_nodes_ledger_authoritative_verify_snapshot_replay_and_preview_error(self) -> None:
        ports = free_ports(3)
        spec = validate_spec(sample_raw(ports))
        plan = compile_plan(spec)
        ledger_path = self.dir / "ledger.sqlite3"
        snapshot_path = self.dir / "snapshot.json"
        result = execute_run(spec, plan, ledger_path, snapshot_path=snapshot_path)
        self.assertEqual(result["counts"], {"total": 18, "completed": 18, "failed": 0, "blocked": 0})
        self.assertEqual(result["peak"]["resource_units"], 7)
        with Ledger(ledger_path) as ledger:
            ledger.verify_integrity()
            self.assertEqual(ledger._meta_value("format_version"), "1")
            raw_version = ledger.connection.execute(
                "SELECT value FROM meta WHERE key = 'format_version'"
            ).fetchone()["value"]
            self.assertIsInstance(raw_version, int)
            preview = ledger.preview()
            self.assertEqual(preview["completed_cursor"], 18)
            self.assertEqual(preview["stale_shards"], 0)
            self.assertEqual(sum(group["progress"]["completed"] for group in preview["groups"]), 18)
        verified = verify_run(ledger_path, use_nodes=True)
        self.assertEqual(verified["records_checked"], 18)
        self.assertEqual(verified["node_replays"], 18)
        replayed = replay_restored_snapshot(snapshot_path)
        self.assertEqual(replayed["records_replayed"], 18)
        self.assertEqual(replayed["restore"]["mode"], "whole_replace")

        with Ledger(ledger_path) as ledger:
            first = next(iter(ledger.current_shards(include_failed=False)))
            aggregate = json.loads(first["aggregate_json"])
            aggregate["scan_rows"] += 1
            aggregate["byte_digest"] = "tampered"
            with ledger.connection:
                ledger.connection.execute(
                    "UPDATE shards SET aggregate_json = ? WHERE run_id = ? AND shard_id = ?",
                    (canonical_json(aggregate), first["run_id"], first["shard_id"]),
                )
                ledger._write_content_digest()
            ledger.verify_integrity()
        with self.assertRaises(AppError) as caught:
            verify_run(ledger_path, use_nodes=False)
        error = caught.exception
        self.assertEqual(error.code, "E_PREVIEW")
        self.assertEqual(error.details["cluster_id"], first["cluster_id"])
        self.assertEqual(error.details["task_id"], first["task_id"])
        self.assertEqual(error.details["shard_id"], first["shard_id"])
        self.assertEqual(error.details["field"], "scan_rows")
        self.assertIn("scan_rows", error.details["comparison"])
        self.assertIn("byte_digest", error.details["comparison"])

    def test_snapshot_digest_replay_first_mismatch_and_cycle_injection(self) -> None:
        ports = free_ports(3)
        spec = validate_spec(sample_raw(ports))
        plan = compile_plan(spec)
        ledger_path = self.dir / "ledger.sqlite3"
        snapshot_path = self.dir / "snapshot.json"
        execute_run(spec, plan, ledger_path, snapshot_path=snapshot_path)
        original = json.loads(snapshot_path.read_text(encoding="utf-8"))
        restored_with_different_versions = load_snapshot(
            snapshot_path,
            current_versions={"clusters": 9, "revisions": 9, "budgets": 9},
        )
        self.assertTrue(restored_with_different_versions["restore"]["version_changed"])
        self.assertEqual(restored_with_different_versions["restore"]["mode"], "whole_replace")
        self.assertFalse(restored_with_different_versions["restore"]["incremental_merge"])

        bad_digest_path = self.dir / "bad-digest.json"
        bad_digest = json.loads(json.dumps(original))
        bad_digest["content_digest"] = "0" * 64
        bad_digest_path.write_text(json.dumps(bad_digest), encoding="utf-8")
        with self.assertRaises(AppError) as caught:
            load_snapshot(bad_digest_path)
        self.assertEqual(caught.exception.code, "E_SNAPSHOT")

        bad_replay_path = self.dir / "bad-replay.json"
        bad_replay = json.loads(json.dumps(original))
        trace = sorted(
            (item for item in bad_replay["run"]["traces"] if item["outcome"] == "completed"),
            key=lambda item: (item["cluster_id"], item["task_id"], item["shard_index"], item["shard_id"]),
        )[0]
        trace["aggregate"]["scan_rows"] += 9
        unsigned = {key: value for key, value in bad_replay.items() if key != "content_digest"}
        bad_replay["content_digest"] = content_digest(unsigned)
        bad_replay_path.write_text(json.dumps(bad_replay), encoding="utf-8")
        with self.assertRaises(AppError) as caught:
            replay_restored_snapshot(bad_replay_path)
        error = caught.exception
        self.assertEqual(error.code, "E_REPLAY")
        self.assertEqual(error.details["shard_id"], trace["shard_id"])
        self.assertEqual(error.details["field"], "scan_rows")

        cycle_path = self.dir / "cycle.json"
        cycle = json.loads(json.dumps(original))
        cycle_revisions = [item for item in cycle["spec"]["revisions"] if item["cluster_id"] != "edge-b"]
        config = next(item["config"] for item in cycle["spec"]["revisions"] if item["cluster_id"] == "edge-b")
        cycle_revisions.extend(
            [
                {"cluster_id": "edge-b", "version": 1, "parent": 2, "config": config},
                {"cluster_id": "edge-b", "version": 2, "parent": 1, "config": config},
            ]
        )
        cycle["spec"]["revisions"] = cycle_revisions
        unsigned = {key: value for key, value in cycle.items() if key != "content_digest"}
        cycle["content_digest"] = content_digest(unsigned)
        cycle_path.write_text(json.dumps(cycle), encoding="utf-8")
        with self.assertRaises(AppError) as caught:
            restore_snapshot(cycle_path)
        self.assertEqual(caught.exception.code, "E_CYCLE")

    def test_ledger_missing_digest_is_not_silently_healed(self) -> None:
        ports = free_ports(3)
        spec = validate_spec(sample_raw(ports))
        ledger_path = self.dir / "ledger.sqlite3"
        execute_run(spec, compile_plan(spec), ledger_path)
        connection = sqlite3.connect(ledger_path)
        try:
            connection.execute("DELETE FROM meta WHERE key = 'content_digest'")
            connection.commit()
        finally:
            connection.close()
        with self.assertRaises(AppError) as caught:
            Ledger(ledger_path)
        self.assertEqual(caught.exception.code, "E_LEDGER")

    def test_generation_replacement_marks_old_conclusions_stale(self) -> None:
        ports = free_ports(3)
        raw = sample_raw(ports)
        base = validate_spec(raw)
        ledger_path = self.dir / "ledger.sqlite3"
        first = execute_run(base, compile_plan(base), ledger_path)
        self.assertEqual(first["counts"]["completed"], 18)
        evolved_raw = evolve_spec_data(
            raw,
            config_updates={"edge-a": {"mode": "canary", "region": "east", "policy": "strict"}},
            target_updates={"deploy-core": ["edge-a", "edge-c"]},
        )
        evolved = validate_spec(evolved_raw)
        second = execute_run(evolved, compile_plan(evolved), ledger_path)
        self.assertEqual(second["stale_shards_after"], 18)
        self.assertEqual(second["counts"]["completed"], 14)
        self.assertTrue(any("revision changed" in reason for reason in second["replacement"]["reasons"]))
        self.assertTrue(any("target clusters changed" in reason for reason in second["replacement"]["reasons"]))
        self.assertTrue(second["replacement"]["affected_shards"])
        self.assertGreaterEqual(
            len(second["replacement"]["affected_shards"]), len(second["replacement"]["changed_shards"])
        )
        preview = preview_ledger(ledger_path)
        self.assertEqual(preview["run_id"], second["run_id"])
        self.assertEqual(preview["stale_shards"], 18)
        self.assertEqual(sum(group["progress"]["completed"] for group in preview["groups"]), 14)
        self.assertNotIn(first["run_id"], json.dumps(preview))

    def test_port_and_handshake_errors(self) -> None:
        ports = free_ports(3)
        raw = sample_raw(ports)
        spec = validate_spec(raw)
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", ports[0]))
        listener.listen(1)
        try:
            with NodeManager(spec, handshake_timeout_ms=300) as nodes:
                with self.assertRaises(AppError) as caught:
                    nodes.start({"edge-a"})
            self.assertEqual(caught.exception.code, "E_PORT")
        finally:
            listener.close()

        stall_spec = validate_spec(single_cluster_raw(ports[0]))
        previous = os.environ.get("CONFIGDIST_TEST_STALL_HANDSHAKE")
        os.environ["CONFIGDIST_TEST_STALL_HANDSHAKE"] = "1"
        try:
            with NodeManager(stall_spec, handshake_timeout_ms=250) as nodes:
                with self.assertRaises(AppError) as caught:
                    nodes.start({"edge-a"})
            self.assertEqual(caught.exception.code, "E_HANDSHAKE")
        finally:
            if previous is None:
                os.environ.pop("CONFIGDIST_TEST_STALL_HANDSHAKE", None)
            else:
                os.environ["CONFIGDIST_TEST_STALL_HANDSHAKE"] = previous


class RecoveryTests(unittest.TestCase):
    def test_force_termination_recovers_only_committed_shards(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            ports = free_ports(3)
            spec_path = directory / "spec.json"
            ledger_path = directory / "ledger.sqlite3"
            spec_path.write_text(json.dumps(sample_raw(ports)), encoding="utf-8")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(ROOT)
            command = [
                sys.executable,
                "-m",
                "configdist",
                "run",
                "--spec",
                str(spec_path),
                "--ledger",
                str(ledger_path),
                "--crash-after-commits",
                "2",
                "--json",
            ]
            crashed = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=30)
            self.assertEqual(crashed.returncode, 17, crashed.stderr + crashed.stdout)
            time.sleep(0.5)
            recovered = recover_ledger(ledger_path)
            self.assertEqual(len(recovered["committed_shards"]), 2)
            self.assertEqual(len(recovered["unknown_shards"]), 16)
            self.assertEqual(recovered["requires_rerun"], sorted(recovered["unknown_shards"]))
            self.assertFalse(recovered["exit_report_generated"])
            preview = preview_ledger(ledger_path)
            self.assertEqual(preview["completed_cursor"], 2)
            self.assertEqual(sum(group["progress"]["total"] for group in preview["groups"]), 18)
            self.assertEqual(sum(group["progress"]["unknown"] for group in preview["groups"]), 16)

            rerun_spec = validate_spec(sample_raw(ports))
            rerun = execute_run(rerun_spec, compile_plan(rerun_spec), ledger_path)
            self.assertEqual(rerun["counts"]["completed"], 18)
            self.assertEqual(rerun["stale_shards_after"], 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
