from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
import unittest
from pathlib import Path

from meshdiag.canonical import pretty_json, read_json
from meshdiag.config import load_workspace
from meshdiag.engine import Engine
from meshdiag.errors import MeshError
from meshdiag.models import normalize_revision
from meshdiag.replay import replay_snapshot
from tests.support import (
    ROOT,
    LocalUpstream,
    TempWorkspace,
    service_rows,
    v3_rule,
)


class EngineIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.upstream = LocalUpstream(2)
        self.temp = TempWorkspace()

    def tearDown(self):
        self.upstream.close()
        self.temp.cleanup()

    def _workspace(self, rules, requests, **kwargs):
        path = self.temp.create(
            services=service_rows(self.upstream.ports, limit=1),
            revision={"revision": kwargs.pop("revision", 1), "rules": rules},
            requests=requests,
            global_concurrency=kwargs.pop("global_concurrency", 2),
            **kwargs,
        )
        return load_workspace(path)

    def test_real_timeout_retry_headers_circuit_and_ledger(self):
        weights = {f"svc-{index}": 1 if index == 0 else 0 for index in range(2)}
        rule = v3_rule(
            weights=weights,
            timeout_ms=60,
            retries=1,
            threshold=4,
            cooldown_ms=5000,
        )
        workspace = self._workspace(
            [rule],
            [
                {"id": "r-flaky", "path": "/svc/flaky"},
                {"id": "r-slow", "path": "/svc/slow"},
                {"id": "r-fail-1", "path": "/svc/fail"},
                {"id": "r-fail-2", "path": "/svc/fail"},
                {"id": "r-fail-3", "path": "/svc/fail"},
            ],
        )
        with Engine(workspace) as engine:
            summary = engine.run_round(round_id="round-real", snapshot_id="snap-real")
            view = engine.ledger.current_view()
        results = {result.request_id: result for result in summary.results}
        self.assertEqual(results["r-flaky"].outcome, "success")
        self.assertEqual(results["r-flaky"].attempt_count, 2)
        self.assertEqual(results["r-flaky"].instance_sequence, ("svc-0", "svc-0"))
        self.assertEqual(results["r-slow"].outcome, "timeout")
        self.assertEqual(results["r-slow"].reason, "socket_timeout")
        self.assertEqual(results["r-slow"].attempt_count, 2)
        self.assertGreaterEqual(results["r-slow"].traces[0].elapsed_ms, 55)
        self.assertLess(results["r-slow"].traces[0].elapsed_ms, 250)
        self.assertEqual(results["r-fail-1"].outcome, "failure")
        self.assertEqual(results["r-fail-2"].outcome, "circuit_open")
        self.assertEqual(results["r-fail-3"].outcome, "circuit_open")
        self.assertTrue(
            all(
                trace.outcome == "circuit_open"
                for trace in results["r-fail-2"].traces
            )
        )
        attempt_rows = self.upstream.state(
            self.upstream.ports[0], "/svc/flaky"
        )
        self.assertEqual(attempt_rows[0]["attempts"], [1, 2])
        self.assertEqual(view["round"]["status"], "completed")
        self.assertEqual(len(view["requests"]), 5)
        self.assertIn("circuit_open", {row["outcome"] for row in view["requests"]})
        self.assertTrue(
            all(row["instance_id"] == "svc-0" for row in view["attempts"])
        )

    def test_revision_upgrade_replaces_current_view_and_marks_history_stale(self):
        v1 = {
            "id": "route",
            "format_version": 1,
            "prefix": "/svc",
            "target_service": "svc",
        }
        workspace = self._workspace([v1], [{"id": "r", "path": "/svc/ok"}])
        with Engine(workspace) as engine:
            engine.run_round(round_id="round-v1", snapshot_id="snap-v1")
            v2_rule = v3_rule(weights={"svc-0": 2, "svc-1": 1}, retries=1)
            engine.run_round(
                round_id="round-v2",
                snapshot_id="snap-v2",
                revision_document={"revision": 2, "rules": [v2_rule]},
            )
            rounds = engine.ledger.list_rounds()
            view = engine.ledger.current_view()
            self.assertEqual(
                [(row["round_id"], row["stale"]) for row in rounds],
                [("round-v1", 1), ("round-v2", 0)],
            )
            self.assertEqual(view["round"]["round_id"], "round-v2")
            self.assertEqual(len(view["requests"]), 1)
            current_count = engine.ledger.connection.execute(
                "SELECT COUNT(*) AS count FROM current_requests"
            ).fetchone()["count"]
            self.assertEqual(current_count, 1)
            changed_same_version = dict(v2_rule)
            changed_same_version["timeout_ms"] = 999
            with self.assertRaises(MeshError) as context:
                engine.prepare_revision(
                    {"revision": 2, "rules": [changed_same_version]}
                )
            self.assertEqual(context.exception.code, "E_VERSION")

    def test_same_version_with_changed_instances_is_e_version(self):
        rule = v3_rule(weights={"svc-0": 1, "svc-1": 1})
        workspace = self._workspace([rule], [{"id": "r", "path": "/svc/ok"}])
        with Engine(workspace) as engine:
            engine.run_round(round_id="round-before", snapshot_id="snap-before")
        services = list(workspace.services)
        services[0] = dataclasses.replace(services[0], max_concurrency=2)
        changed_workspace = dataclasses.replace(workspace, services=tuple(services))
        with Engine(changed_workspace) as engine:
            with self.assertRaises(MeshError) as context:
                engine.prepare_revision()
        self.assertEqual(context.exception.code, "E_VERSION")
        self.assertIn(
            "current_instance_fingerprint",
            context.exception.diagnostics[0].extra,
        )

    def test_referenced_deletion_reports_referrers_before_normalization(self):
        rule_a = {
            "id": "a",
            "format_version": 1,
            "prefix": "/a",
            "target_service": "svc",
        }
        rule_b = v3_rule(
            rule_id="b",
            weights={"svc-0": 1, "svc-1": 0},
            prefix="/b",
            depends_on=["rule:a"],
        )
        workspace = self._workspace(
            [rule_a, rule_b], [{"id": "r", "path": "/a/ok"}]
        )
        with Engine(workspace) as engine:
            engine.run_round(round_id="round-ref", snapshot_id="snap-ref")
            with self.assertRaises(MeshError) as context:
                engine.prepare_revision(
                    {
                        "revision": 2,
                        "rules": [
                            {
                                **rule_b,
                                "depends_on": ["rule:a"],
                            }
                        ],
                    }
                )
        self.assertEqual(context.exception.code, "E_REF")
        self.assertEqual(
            context.exception.diagnostics[0].extra["referrers"], ["b"]
        )

    def test_completed_snapshot_replay_and_first_difference(self):
        rule = v3_rule(
            weights={"svc-0": 1, "svc-1": 0},
            retries=1,
            threshold=4,
        )
        workspace = self._workspace(
            [rule],
            [
                {"id": "a", "path": "/svc/ok"},
                {"id": "b", "path": "/svc/flaky"},
            ],
        )
        with Engine(workspace) as engine:
            summary = engine.run_round(round_id="round-replay", snapshot_id="snap-replay")
            self.upstream.reset()
            current = engine.load_workspace_revision()
            report = replay_snapshot(engine, "snap-replay", current_revision=current)
            self.assertEqual(report.requests_compared, 2)
            self.assertEqual(report.ledger_requests_compared, 2)
            self.assertEqual(report.ledger_attempts_compared, 3)
            path = Path(summary.snapshot_path)
            envelope = read_json(path)
            envelope["payload"]["completed_results"][0]["outcome"] = "failure"
            from meshdiag.canonical import sha256_json

            envelope["checksum"] = sha256_json(envelope["payload"])
            path.write_text(pretty_json(envelope) + "\n", encoding="utf-8")
            with self.assertRaises(MeshError) as context:
                replay_snapshot(engine, path, current_revision=current)
        self.assertEqual(context.exception.code, "E_REPLAY")
        diagnostic = context.exception.diagnostics[0]
        self.assertEqual(diagnostic.extra["request_id"], "a")
        self.assertEqual(diagnostic.extra["field"], "outcome")

    def test_instance_drift_replay_reports_specific_basis(self):
        rule = v3_rule(weights={"svc-0": 1, "svc-1": 1})
        workspace = self._workspace([rule], [{"id": "a", "path": "/svc/ok"}])
        with Engine(workspace) as engine:
            engine.run_round(round_id="round-drift", snapshot_id="snap-drift")
            current = engine.load_workspace_revision()
        services = list(workspace.services)
        instances = list(services[0].instances)
        instances[0] = dataclasses.replace(instances[0], port=instances[0].port + 1000)
        services[0] = dataclasses.replace(services[0], instances=tuple(instances))
        drifted = dataclasses.replace(workspace, services=tuple(services))
        with Engine(drifted) as engine:
            with self.assertRaises(MeshError) as context:
                replay_snapshot(engine, "snap-drift", current_revision=current)
        self.assertEqual(context.exception.code, "E_REPLAY")
        diagnostic = context.exception.diagnostics[0]
        self.assertEqual(diagnostic.extra["field"], "instance_set")
        self.assertIn("实例", diagnostic.extra["basis"])

    def test_snapshot_revision_mismatch_replaces_revision_graph_and_ledger(self):
        rule1 = v3_rule(weights={"svc-0": 1, "svc-1": 1})
        workspace1 = self._workspace(
            [rule1], [{"id": "a", "path": "/svc/ok"}], revision_name="revision-1.json"
        )
        with Engine(workspace1) as engine:
            engine.run_round(round_id="round-snap-r1", snapshot_id="snap-r1")
        rule2 = {
            **rule1,
            "timeout_ms": 77,
            "depends_on": ["service:svc"],
        }
        revision2_path = self.temp.write_json(
            "revision-2.json", {"revision": 2, "rules": [rule2]}
        )
        workspace2 = dataclasses.replace(workspace1, revision_path=revision2_path)
        with Engine(workspace2) as engine:
            before = engine.load_workspace_revision()
            report = engine.recover_snapshot("snap-r1")
            state = engine.ledger.current_state()
            active_revision, active_services = engine.load_active_revision_and_services()
        self.assertEqual(before.revision_version, 2)
        self.assertTrue(report.revision_replaced)
        self.assertTrue(report.dependency_graph_replaced)
        self.assertEqual(report.consistency, "SNAPSHOT_REVISION_MISMATCH")
        self.assertEqual(state["revision_version"], 1)
        self.assertEqual(active_revision.revision_version, 1)
        self.assertEqual(
            [instance.instance_id for instance in active_services[0].instances],
            ["svc-0", "svc-1"],
        )

    def test_forced_termination_recovery_marks_unknown_and_blocks_replay(self):
        rule = v3_rule(weights={"svc-0": 1, "svc-1": 0})
        workspace = self._workspace(
            [rule],
            [{"id": f"r-{index}", "path": "/svc/ok"} for index in range(4)],
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT)
        process = subprocess.run(
            [
                sys.executable,
                "-m",
                "meshdiag",
                "run",
                "--workspace",
                str(workspace.config_path),
                "--round-id",
                "round-crash",
                "--snapshot-id",
                "snap-crash",
                "--crash-after",
                "2",
            ],
            cwd=ROOT,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
        )
        self.assertEqual(process.returncode, 96)
        with Engine(workspace) as engine:
            payload = engine.snapshots.load("snap-crash")
            self.assertEqual(payload["status"], "running")
            self.assertEqual(len(payload["completed_results"]), 2)
            with self.assertRaises(MeshError) as replay_error:
                replay_snapshot(engine, "snap-crash")
            self.assertEqual(replay_error.exception.code, "E_REPLAY")
            report = engine.recover_snapshot("snap-crash")
            view = engine.ledger.current_view()
        self.assertEqual(report.snapshot_status, "running")
        self.assertEqual(report.consistency, "INTERRUPTED_RERUN_REQUIRED")
        self.assertEqual(len(report.completed_request_ids), 2)
        self.assertEqual(len(report.unknown_request_ids), 2)
        self.assertEqual(len(report.rerun_required), 2)
        unknown_rows = [row for row in view["requests"] if row["unknown"]]
        self.assertEqual(len(unknown_rows), 2)
        self.assertTrue(all(row["rerun_required"] for row in unknown_rows))
        self.assertEqual(view["round"]["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
