from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from cluster_diag.errors import DiagError, E_LEDGER, E_REPLAY, E_VIEW
from cluster_diag.runtime import DiagnosticRuntime
from cluster_diag.spec import load_spec
from cluster_diag.util import node_content_digest

from .support import ROOT, base_payload, child_env, free_ports, write_spec


class RuntimeAcceptanceTests(unittest.TestCase):
    def _spec(self, temp: Path) -> tuple[Path, object]:
        ports = free_ports(2)
        path = write_spec(temp / "spec.json", base_payload(ports))
        return path, load_spec(path)

    def test_incremental_view_whole_root_replacement_and_authoritative_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            path, spec = self._spec(temp)
            with DiagnosticRuntime(spec=spec, workdir=temp / "runtime", latency_budget_ms=100) as runtime:
                first = runtime.commit_next()
                self.assertEqual(first["root_revision"], 1)
                self.assertEqual({item["cluster_id"] for item in runtime.current_items()}, {"edge-a", "edge-b"})
                old_root = runtime.ledger.load_snapshot(1)
                old_snapshot_digest = old_root.snapshot_digest

                old_query = runtime.query("edge-a")
                self.assertEqual(old_query["root_revision"], 1)
                self.assertGreaterEqual(old_query["latency"]["elapsed_ms_x1000"], 0)

                second = runtime.commit_next()
                self.assertEqual(second["root_revision"], 2)
                switch = second["root_switch"]
                self.assertEqual(switch["old_root_revision"], 1)
                self.assertEqual(switch["new_root_revision"], 2)
                self.assertEqual(switch["affected_clusters"], ["edge-b"])
                self.assertEqual([item["cluster_id"] for item in switch["changed_clusters"]], ["edge-b"])
                stale_keys = {(item["cluster_id"], item["revision"]) for item in switch["stale_nodes"]}
                self.assertEqual(stale_keys, {("edge-a", 1), ("edge-b", 1)})
                self.assertEqual(runtime.latency_stats()["groups"], [])

                unchanged_a = runtime.view.item("edge-a")
                self.assertEqual(unchanged_a.node_revision, 1)
                self.assertEqual(unchanged_a.root_revision, 2)
                current_old = runtime.view.item("edge-b")
                self.assertEqual(current_old.node_revision, 2)

                third = runtime.commit_next()
                self.assertEqual(third["root_revision"], 3)
                self.assertEqual([item["cluster_id"] for item in third["root_switch"]["changed_clusters"]], ["edge-a"])
                self.assertEqual(runtime.view.item("edge-a").node_revision, 3)
                self.assertEqual(runtime.view.item("edge-b").node_revision, 2)

                old_again = runtime.ledger.load_snapshot(1)
                self.assertEqual(old_again.snapshot_digest, old_snapshot_digest)
                self.assertEqual(old_again.effective_nodes["edge-a"].revision, 1)

                query = runtime.query("edge-a")
                self.assertFalse(query["stale"])
                stats = runtime.latency_stats()
                self.assertTrue(stats["groups"])
                self.assertTrue(all(group["revision"] == 3 for group in stats["groups"] if group["kind"] == "ui_query"))

                authoritative = runtime.authoritative()
                self.assertTrue(authoritative["ok"])
                self.assertEqual(authoritative["root_revision"], 3)
                self.assertEqual(authoritative["recomputed_node_count"], 4)
                self.assertEqual(authoritative["clusters"], ["edge-a", "edge-b"])

                recovery = runtime.recover(replay=True)
                self.assertFalse(recovery["snapshot_replaced"])
                self.assertEqual(recovery["replay"]["replayed_node_count"], 4)
                self.assertEqual(recovery["unknown_uncommitted_nodes"], [])

    def test_reopen_marks_uncommitted_revisions_unknown_and_replay_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            path, spec = self._spec(temp)
            runtime_dir = temp / "runtime"
            with DiagnosticRuntime(spec=spec, workdir=runtime_dir) as runtime:
                runtime.commit_next()
                self.assertEqual(runtime.ledger.root_revision(), 1)
            with DiagnosticRuntime(spec=spec, workdir=runtime_dir) as reopened:
                self.assertEqual(
                    {(item["cluster_id"], item["revision"]) for item in reopened.unknown_uncommitted_nodes},
                    {("edge-b", 2), ("edge-a", 3)},
                )
                recovery = reopened.recover(replay=True)
                self.assertEqual(recovery["replay"]["root_revision"], 1)
                self.assertEqual(recovery["replay"]["replayed_node_count"], 2)
                self.assertEqual(len(recovery["unknown_uncommitted_nodes"]), 2)
                self.assertTrue(all(item["status"] == "unknown" for item in recovery["unknown_uncommitted_nodes"]))
                self.assertTrue(all(item["required_action"] == "rerun" for item in recovery["unknown_uncommitted_nodes"]))

    def test_snapshot_node_mismatch_replaces_whole_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            path, spec = self._spec(temp)
            runtime_dir = temp / "runtime"
            with DiagnosticRuntime(spec=spec, workdir=runtime_dir) as runtime:
                runtime.commit_next()
            snapshot_path = runtime_dir / "session_snapshot.json"
            document = json.loads(snapshot_path.read_text(encoding="utf-8"))
            document["node_digests"]["ghost|99"] = "sha256:" + "f" * 64
            snapshot_path.write_text(json.dumps(document), encoding="utf-8")
            with DiagnosticRuntime(spec=spec, workdir=runtime_dir) as reopened:
                self.assertIsNotNone(reopened.session_open)
                assert reopened.session_open is not None
                self.assertTrue(reopened.session_open.replaced)
                self.assertIn("node_digest_set_mismatch", reopened.session_open.reasons)
                self.assertEqual(len(reopened.current_items()), 2)
                self.assertIsNone(reopened.view.item("ghost"))
                repaired = json.loads(snapshot_path.read_text(encoding="utf-8"))
                self.assertEqual(set(repaired["node_digests"]), {"edge-a|1", "edge-b|1"})

    def test_authoritative_summary_mismatch_reports_first_e_view(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            path, spec = self._spec(temp)
            runtime_dir = temp / "runtime"
            with DiagnosticRuntime(spec=spec, workdir=runtime_dir) as runtime:
                runtime.commit_to("latest")
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["revisions"][1]["config"]["net.timeout_ms"] = 321
            write_spec(path, payload)
            changed_spec = load_spec(path)
            with DiagnosticRuntime(spec=changed_spec, workdir=runtime_dir) as runtime:
                with self.assertRaises(DiagError) as raised:
                    runtime.authoritative()
                self.assertEqual(raised.exception.code, E_VIEW)
                self.assertEqual(raised.exception.details["cluster_id"], "edge-b")
                self.assertEqual(raised.exception.details["revision"], 2)
                self.assertEqual(raised.exception.details["field"], "aggregate_summary")
                events = runtime.ledger.latency_events(bound_root_revision=3, kind="authoritative")
                self.assertEqual(events[-1].error_code, E_VIEW)

    def test_recovery_replay_mismatch_reports_e_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            path, spec = self._spec(temp)
            runtime_dir = temp / "runtime"
            with DiagnosticRuntime(spec=spec, workdir=runtime_dir) as runtime:
                runtime.commit_to("latest")
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["revisions"][1]["config"]["net.timeout_ms"] = 777
            write_spec(path, payload)
            changed_spec = load_spec(path)
            with DiagnosticRuntime(spec=changed_spec, workdir=runtime_dir) as runtime:
                with self.assertRaises(DiagError) as raised:
                    runtime.recover(replay=True)
                self.assertEqual(raised.exception.code, E_REPLAY)
                self.assertEqual(raised.exception.details["basis"], "aggregate_summary")
                self.assertEqual(raised.exception.details["cluster_id"], "edge-b")
                self.assertEqual(raised.exception.details["revision"], 2)

    def test_snapshot_latency_stats_mismatch_reports_e_replay(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            path, spec = self._spec(temp)
            runtime_dir = temp / "runtime"
            with DiagnosticRuntime(spec=spec, workdir=runtime_dir, latency_budget_ms=50) as runtime:
                runtime.commit_to("latest")
                runtime.query("edge-a")
            snapshot_path = runtime_dir / "session_snapshot.json"
            document = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertTrue(document["latency_stats"]["groups"])
            document["latency_stats"]["groups"][0]["mean_ms_x1000"] += 1
            snapshot_path.write_text(json.dumps(document), encoding="utf-8")
            with DiagnosticRuntime(spec=spec, workdir=runtime_dir) as runtime:
                with self.assertRaises(DiagError) as raised:
                    runtime.recover(replay=True)
                self.assertEqual(raised.exception.code, E_REPLAY)
                self.assertEqual(raised.exception.details["basis"], "latency_stats")

    def test_injected_ledger_corruption_is_reported_as_e_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            path, spec = self._spec(temp)
            runtime_dir = temp / "runtime"
            with DiagnosticRuntime(spec=spec, workdir=runtime_dir) as runtime:
                runtime.commit_to("latest")
            with self.assertRaises(DiagError) as raised:
                with DiagnosticRuntime(
                    spec=spec,
                    workdir=runtime_dir,
                    inject_ledger_corruption=True,
                ):
                    pass
            self.assertEqual(raised.exception.code, E_LEDGER)
            self.assertEqual(raised.exception.details["field"], "content_digest")

    def test_authoritative_parent_semantic_error_is_e_view_not_e_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            path, spec = self._spec(temp)
            runtime_dir = temp / "runtime"
            with DiagnosticRuntime(spec=spec, workdir=runtime_dir) as runtime:
                runtime.commit_to("latest")
            connection = sqlite3.connect(runtime_dir / "results.sqlite3")
            try:
                row = connection.execute(
                    "SELECT task_id, aggregate_summary FROM nodes WHERE cluster_id='edge-a' AND revision=3"
                ).fetchone()
                task_id, summary = row
                digest = node_content_digest(
                    task_id=task_id,
                    cluster_id="edge-a",
                    revision=3,
                    parent_revision=None,
                    aggregate_summary_canonical=summary,
                )
                connection.execute(
                    "UPDATE nodes SET parent_revision=NULL, content_digest=? WHERE cluster_id='edge-a' AND revision=3",
                    (digest,),
                )
                connection.commit()
            finally:
                connection.close()
            with DiagnosticRuntime(
                spec=spec,
                workdir=runtime_dir,
                defer_semantic_ledger_checks=True,
            ) as runtime:
                with self.assertRaises(DiagError) as raised:
                    runtime.authoritative()
                self.assertEqual(raised.exception.code, E_VIEW)
                self.assertEqual(raised.exception.details["field"], "parent_revision")
                self.assertEqual(raised.exception.details["cluster_id"], "edge-a")
                self.assertEqual(raised.exception.details["revision"], 3)

    def test_forced_process_exit_rolls_back_uncommitted_nodes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_name:
            temp = Path(temp_name)
            path, spec = self._spec(temp)
            runtime_dir = temp / "runtime"
            with DiagnosticRuntime(spec=spec, workdir=runtime_dir) as runtime:
                runtime.commit_next()
            ledger_path = runtime_dir / "results.sqlite3"
            script = r"""
import os, sqlite3, sys
connection = sqlite3.connect(sys.argv[1], isolation_level=None)
connection.execute("BEGIN IMMEDIATE")
connection.execute(
    "INSERT INTO commits(revision, parent_revision, committed_at_ns, snapshot_digest) VALUES(2,1,1,?)",
    ("sha256:" + "a"*64,),
)
connection.execute(
    "INSERT INTO nodes(cluster_id,revision,task_id,parent_revision,aggregate_summary,content_digest,completed_at_ns) VALUES(?,?,?,?,?,?,?)",
    ("edge-b", 2, "core-network", 1, "{}", "sha256:" + "b"*64, 1),
)
os._exit(9)
"""
            process = subprocess.run(
                [sys.executable, "-c", script, str(ledger_path)],
                cwd=ROOT,
                env=child_env(),
                timeout=10,
                check=False,
            )
            self.assertEqual(process.returncode, 9)
            with DiagnosticRuntime(spec=spec, workdir=runtime_dir) as runtime:
                self.assertEqual(runtime.ledger.root_revision(), 1)
                self.assertEqual(runtime.ledger.latest_committed_revision(), 1)
                self.assertIsNone(runtime.ledger.get_node("edge-b", 2))
                self.assertEqual(len(runtime.unknown_uncommitted_nodes), 2)
                self.assertTrue(all(item["required_action"] == "rerun" for item in runtime.unknown_uncommitted_nodes))


if __name__ == "__main__":
    unittest.main()
