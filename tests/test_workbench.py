from __future__ import annotations

import copy
import io
import json
import os
import shutil
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

from sched_diag.cli import main
from sched_diag.demo import run_demo
from sched_diag.errors import BusinessError, E_INCR, E_MANIFEST, E_REF, E_REPLAY, E_SNAPSHOT
from sched_diag.manifest import load_manifests
from sched_diag.model import Node, PluginConfig, Pod, Selector, Taint
from sched_diag.persistence import Persistence
from sched_diag.restore import replay_payload, restore_payload
from sched_diag.scheduler import WorkbenchSession
from sched_diag.util import canonical_json, sha256_digest


ROOT = Path(__file__).resolve().parent.parent
SAMPLES = ROOT / "samples"


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _make_writable(function: Any, path: str, _exc: Any) -> None:
    os.chmod(path, stat.S_IWRITE | stat.S_IREAD)
    function(path)


def cleanup_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path, onerror=_make_writable)


def sample_materials() -> tuple[list[Node], list[Pod], PluginConfig]:
    return load_manifests(SAMPLES / "nodes.json", SAMPLES / "pods.json", SAMPLES / "plugins.json")


def scheduled_sample() -> WorkbenchSession:
    session = WorkbenchSession(*sample_materials())
    session.full_schedule()
    return session


def reseal(payload: dict[str, Any]) -> dict[str, Any]:
    payload["contentDigest"] = sha256_digest(payload["content"])
    payload["envelopeDigest"] = sha256_digest(
        {key: value for key, value in payload.items() if key != "envelopeDigest"}
    )
    return payload


class ManifestValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="sched-diag-manifest-"))
        self.nodes = _read_json(SAMPLES / "nodes.json")
        self.pods = _read_json(SAMPLES / "pods.json")
        self.plugins = _read_json(SAMPLES / "plugins.json")

    def tearDown(self) -> None:
        cleanup_tree(self.temp)

    def _write(self, nodes: Any | None = None, pods: Any | None = None) -> tuple[Path, Path, Path]:
        nodes_path = self.temp / "nodes.json"
        pods_path = self.temp / "pods.json"
        plugins_path = self.temp / "plugins.json"
        _write_json(nodes_path, nodes if nodes is not None else self.nodes)
        _write_json(pods_path, pods if pods is not None else self.pods)
        _write_json(plugins_path, self.plugins)
        return nodes_path, pods_path, plugins_path

    def test_sample_is_valid(self) -> None:
        nodes, pods, plugins = load_manifests(*self._write())
        self.assertEqual(4, len(nodes))
        self.assertEqual(5, len(pods))
        self.assertEqual(10, plugins.total_weight())

    def test_duplicate_node_name_is_manifest_error_with_location(self) -> None:
        nodes = copy.deepcopy(self.nodes)
        nodes["nodes"][1]["name"] = nodes["nodes"][0]["name"]
        paths = self._write(nodes=nodes)
        with self.assertRaises(BusinessError) as raised:
            load_manifests(*paths)
        self.assertEqual(E_MANIFEST, raised.exception.code)
        self.assertIn("#/nodes/1/name", raised.exception.location or "")

    def test_invalid_taint_effect_is_manifest_error(self) -> None:
        nodes = copy.deepcopy(self.nodes)
        nodes["nodes"][1]["taints"][0]["effect"] = "Sometimes"
        paths = self._write(nodes=nodes)
        with self.assertRaises(BusinessError) as raised:
            load_manifests(*paths)
        self.assertEqual(E_MANIFEST, raised.exception.code)
        self.assertIn("taints/0/effect", raised.exception.location or "")

    def test_pod_request_above_single_node_maximum_is_rejected(self) -> None:
        pods = copy.deepcopy(self.pods)
        pods["pods"][0]["resources"]["cpu"] = 9
        paths = self._write(pods=pods)
        with self.assertRaises(BusinessError) as raised:
            load_manifests(*paths)
        self.assertEqual(E_MANIFEST, raised.exception.code)
        self.assertIn("resources/cpu", raised.exception.location or "")

    def test_unresolved_node_affinity_key_is_ref_error(self) -> None:
        nodes = copy.deepcopy(self.nodes)
        for node in nodes["nodes"]:
            node["labels"].pop("zone")
        paths = self._write(nodes=nodes)
        with self.assertRaises(BusinessError) as raised:
            load_manifests(*paths)
        self.assertEqual(E_REF, raised.exception.code)
        self.assertEqual("zone", raised.exception.details.get("key"))

    def test_unresolved_anti_affinity_key_is_ref_error(self) -> None:
        pods = copy.deepcopy(self.pods)
        pods["pods"][4]["antiAffinity"]["soft"][0]["selector"]["matchLabels"] = {"missing": "x"}
        paths = self._write(pods=pods)
        with self.assertRaises(BusinessError) as raised:
            load_manifests(*paths)
        self.assertEqual(E_REF, raised.exception.code)
        self.assertIn("antiAffinity", raised.exception.location or "")

    def test_invalid_label_key_is_manifest_error(self) -> None:
        nodes = copy.deepcopy(self.nodes)
        nodes["nodes"][0]["labels"]["bad key"] = "x"
        paths = self._write(nodes=nodes)
        with self.assertRaises(BusinessError) as raised:
            load_manifests(*paths)
        self.assertEqual(E_MANIFEST, raised.exception.code)
        self.assertIn("bad key", raised.exception.location or "")


class SchedulingTests(unittest.TestCase):
    def test_full_schedule_is_deterministic(self) -> None:
        first = scheduled_sample()
        second = scheduled_sample()
        self.assertEqual(canonical_json(first.results_payload()), canonical_json(second.results_payload()))
        self.assertEqual(first.content_digest(), second.content_digest())
        self.assertEqual(first.graph.signature(), second.graph.signature())

    def test_filter_order_and_specific_rejections(self) -> None:
        session = scheduled_sample()
        result = session.results["api-front-b"]
        self.assertEqual(
            ["node_affinity_hard", "anti_affinity_hard", "taint_toleration", "allocatable_resources"],
            result["filter_order"],
        )
        self.assertEqual(["node-east-b"], result["candidates"])
        rejections = {item["node"]: item for item in result["filtered"]}
        self.assertEqual(2, rejections["node-east-a"]["stage"])
        self.assertEqual("anti_affinity_hard", rejections["node-east-a"]["stage_name"])
        self.assertIn("api-front-a", rejections["node-east-a"]["reason"])
        self.assertEqual(1, rejections["node-west-a"]["stage"])

    def test_score_contributions_are_integer_and_additive(self) -> None:
        session = scheduled_sample()
        for result in session.results.values():
            self.assertEqual("floor_toward_zero_non_negative", result["truncation"])
            for score in result["scores"].values():
                for raw in score["raw"].values():
                    self.assertTrue(0 <= raw <= 100)
                self.assertEqual(sum(score["weighted_contrib_milli"].values()), score["total_milli"])
                self.assertTrue(0 <= score["total_milli"] <= 100000)
            self.assertGreaterEqual(result["error_bound_milli"], 1)

    def test_error_band_uses_stable_tie_break(self) -> None:
        nodes = [
            Node("node-a", {"zone": "z"}, 4, 4096),
            Node("node-b", {"zone": "z"}, 4, 4096),
        ]
        pod = Pod(
            "pod-tie",
            {"app": "tie"},
            1,
            512,
            node_affinity_hard=Selector(match_labels={"zone": "z"}),
        )
        plugins = PluginConfig(
            {"node_affinity": 1, "anti_affinity": 0, "resource_balance": 0, "image_locality": 0},
            version=1,
        )
        session = WorkbenchSession(nodes, [pod], plugins)
        session.full_schedule()
        result = session.results["pod-tie"]
        self.assertEqual("node-a", result["selected"])
        self.assertIn("error_band", result["tie_break"]["rule"])
        self.assertEqual(["node-a", "node-b"], result["tie_break"]["band"])

    def test_taint_rejection_is_after_affinity_checks(self) -> None:
        nodes = [Node("node-a", {"zone": "z"}, 4, 4096, taints=())]
        pod = Pod("pod-a", {"app": "a"}, 1, 256, node_affinity_hard=Selector(match_labels={"zone": "other"}))
        plugins = PluginConfig(
            {name: 1 for name in ("node_affinity", "anti_affinity", "resource_balance", "image_locality")}
        )
        session = WorkbenchSession(nodes, [pod], plugins)
        session.full_schedule()
        rejection = session.results["pod-a"]["filtered"][0]
        self.assertEqual(1, rejection["stage"])
        self.assertEqual("node_affinity_hard", rejection["stage_name"])


class DependencyAndIncrementalTests(unittest.TestCase):
    def test_graph_is_complete_and_acyclic(self) -> None:
        session = scheduled_sample()
        summary = session.graph.validate()
        self.assertGreater(summary["nodes"], summary["pod_conclusions"])
        self.assertGreater(summary["facts"], 0)
        self.assertGreater(summary["stats"], 0)

    def test_label_change_invalidates_only_dependent_pods(self) -> None:
        session = scheduled_sample()
        before = {pod.pod_id: canonical_json(session.results[pod.pod_id]) for pod in session.pods}
        versions = {pod.pod_id: session.results[pod.pod_id]["version"] for pod in session.pods}
        report = session.edit_node_label("node-west-a", "disk", "hdd")
        self.assertEqual(["cache-index", "batch-report"], report["pods"])
        self.assertEqual(2, len(session.last_recomputed_pods))
        self.assertEqual({"cache-index", "batch-report"}, set(session.last_recomputed_pods))
        for pod_id in ("probe-west", "api-front-a", "api-front-b"):
            self.assertEqual(before[pod_id], canonical_json(session.results[pod_id]))
            self.assertEqual(versions[pod_id], session.results[pod_id]["version"])
        self.assertTrue(session._unaffected_check["byte_identical"])
        self.assertEqual(3, session._unaffected_check["count"])
        self.assertTrue(session.verify_full_consistency()["ok"])

    def test_every_invalidation_step_has_trigger_source(self) -> None:
        session = scheduled_sample()
        report = session.edit_node_label("node-west-a", "disk", "hdd")
        self.assertEqual(report["roots"][0], report["steps"][0]["id"])
        for index, step in enumerate(report["steps"], start=1):
            self.assertEqual(index, step["step"])
            self.assertTrue(step["triggered_by"])
            for trigger in step["triggered_by"]:
                if not trigger.startswith("external:"):
                    self.assertIn(trigger, {item["id"] for item in report["steps"][: index - 1]})

    def test_incremental_verification_reports_first_differing_pod(self) -> None:
        session = scheduled_sample()
        session.edit_node_label("node-west-a", "disk", "hdd")
        session.results["api-front-a"]["selected"] = "node-east-b"
        with self.assertRaises(BusinessError) as raised:
            session.verify_full_consistency()
        self.assertEqual(E_INCR, raised.exception.code)
        self.assertEqual("api-front-a", raised.exception.details["pod"])
        self.assertEqual("selected", raised.exception.details["field"])

    def test_rejected_mutation_rolls_back_facts_versions_and_results(self) -> None:
        session = scheduled_sample()
        digest_before = session.content_digest()
        versions_before = session.versions()
        results_before = canonical_json(session.results_payload())
        fact_revisions_before = dict(session.fact_revisions)
        with self.assertRaises(BusinessError) as raised:
            session.edit_pod_resources("probe-west", 99, 512)
        self.assertEqual(E_INCR, raised.exception.code)
        self.assertEqual(digest_before, session.content_digest())
        self.assertEqual(versions_before, session.versions())
        self.assertEqual(results_before, canonical_json(session.results_payload()))
        self.assertEqual(fact_revisions_before, session.fact_revisions)

    def test_unused_node_label_change_invalidates_no_pod(self) -> None:
        session = scheduled_sample()
        report = session.edit_node_label("node-west-a", "role", "compute")
        self.assertEqual([], report["pods"])
        self.assertEqual([], report["stats"])
        self.assertEqual([], session.last_recomputed_pods)
        self.assertEqual(5, session._unaffected_check["count"])
        self.assertTrue(session.verify_full_consistency()["ok"])

    def test_taint_change_invalidates_all_taint_gates(self) -> None:
        session = scheduled_sample()
        report = session.edit_node_taints(
            "node-east-a", (Taint("maintenance", "planned", "PreferNoSchedule"),)
        )
        self.assertEqual(
            ["probe-west", "api-front-a", "api-front-b", "cache-index", "batch-report"],
            report["pods"],
        )
        self.assertTrue(any("stat:taint-gate:" in stat for stat in report["stats"]))
        self.assertTrue(session.verify_full_consistency()["ok"])

    def test_resource_change_invalidates_own_and_downstream_pods(self) -> None:
        session = scheduled_sample()
        report = session.edit_pod_resources("probe-west", 2, 512)
        self.assertEqual(
            ["probe-west", "api-front-a", "api-front-b", "cache-index", "batch-report"],
            report["pods"],
        )
        self.assertTrue(any("stat:resource-remaining:" in stat for stat in report["stats"]))
        self.assertTrue(session.verify_full_consistency()["ok"])


class SnapshotAndReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="sched-diag-snapshot-"))
        self.persistence = Persistence(self.temp)
        self.session = scheduled_sample()

    def tearDown(self) -> None:
        cleanup_tree(self.temp)

    def _incremental_snapshot(self) -> tuple[WorkbenchSession, dict[str, Any]]:
        self.session.edit_node_label("node-west-a", "disk", "hdd")
        self.persistence.save_snapshot(self.session, "after-increment")
        return self.session, self.persistence.load_snapshot("after-increment")

    def test_snapshot_revisions_are_read_only_and_preserved(self) -> None:
        first = self.persistence.save_snapshot(self.session, "capture")
        second = self.persistence.save_snapshot(self.session, "capture")
        self.assertEqual(1, first["revision"])
        self.assertEqual(2, second["revision"])
        snapshots = self.persistence.list_snapshots()
        self.assertEqual([1, 2], [item["revision"] for item in snapshots])
        for item in snapshots:
            self.assertTrue(Path(item["backupPath"]).exists())
            if os.name != "nt":
                self.assertEqual(0, stat.S_IMODE(os.stat(item["backupPath"]).st_mode) & stat.S_IWUSR)

    def test_restore_replaces_whole_session_and_lists_changes(self) -> None:
        self.session, payload = self._incremental_snapshot()
        self.session.edit_node_label("node-east-a", "disk", "nvme")
        self.session.edit_pod_resources("probe-west", 2, 512)
        divergent_digest = self.session.content_digest()
        restored, report = restore_payload(payload, current_session=self.session)
        self.assertTrue(report["replaceRequired"])
        self.assertFalse(report["versionsMatch"])
        self.assertFalse(report["digestMatch"])
        self.assertEqual(["node-east-a"], report["changedNodes"])
        self.assertEqual(["probe-west"], report["changedPods"])
        self.assertNotEqual(divergent_digest, restored.content_digest())
        self.assertEqual(payload["contentDigest"], restored.content_digest())

    def test_restore_same_state_is_replace_but_versions_and_digest_match(self) -> None:
        self.session, payload = self._incremental_snapshot()
        restored, report = restore_payload(payload, current_session=self.session)
        self.assertFalse(report["replaceRequired"])
        self.assertTrue(report["versionsMatch"])
        self.assertTrue(report["digestMatch"])
        self.assertEqual([], report["changedNodes"])
        self.assertEqual([], report["changedPods"])
        self.assertEqual(self.session.content_digest(), restored.content_digest())

    def test_replay_reproduces_invalidation_and_matches_full_recompute(self) -> None:
        _, payload = self._incremental_snapshot()
        report = replay_payload(payload)
        self.assertTrue(report["ok"])
        self.assertEqual(["cache-index", "batch-report"], report["invalidation"]["pods"])
        self.assertEqual(["cache-index", "batch-report"], report["replayedAffectedPods"])
        self.assertTrue(report["fullRecomputeMatched"])
        self.assertEqual(["probe-west", "api-front-a", "api-front-b", "cache-index", "batch-report"], report["comparedPods"])

    def test_snapshot_digest_mismatch_is_snapshot_error(self) -> None:
        _, payload = self._incremental_snapshot()
        altered = copy.deepcopy(payload)
        altered["contentDigest"] = "sha256:bad"
        with self.assertRaises(BusinessError) as raised:
            restore_payload(altered, current_session=self.session)
        self.assertEqual(E_SNAPSHOT, raised.exception.code)

    def test_missing_version_is_replay_error(self) -> None:
        _, payload = self._incremental_snapshot()
        altered = copy.deepcopy(payload)
        altered["versions"].pop("node")
        with self.assertRaises(BusinessError) as raised:
            restore_payload(altered, current_session=self.session)
        self.assertEqual(E_REPLAY, raised.exception.code)
        self.assertIn("node", raised.exception.details["missingVersions"])

    def test_plugin_weight_basis_mismatch_is_replay_error(self) -> None:
        _, payload = self._incremental_snapshot()
        altered = copy.deepcopy(payload)
        altered["state"]["results"][0]["plugin_weights"]["node_affinity"] = 999
        reseal(altered)
        with self.assertRaises(BusinessError) as raised:
            replay_payload(altered)
        self.assertEqual(E_REPLAY, raised.exception.code)
        self.assertEqual("plugin_weight_basis_mismatch", raised.exception.details["basis"])

    def test_replay_reports_first_pod_and_field_difference(self) -> None:
        _, payload = self._incremental_snapshot()
        altered = copy.deepcopy(payload)
        altered["state"]["results"][0]["selected"] = "node-east-a"
        reseal(altered)
        with self.assertRaises(BusinessError) as raised:
            replay_payload(altered)
        self.assertEqual(E_REPLAY, raised.exception.code)
        self.assertEqual("probe-west", raised.exception.details["pod"])
        self.assertEqual("selected", raised.exception.details["field"])

    def test_current_sqlite_session_can_be_reopened_and_replayed(self) -> None:
        self.token = self.persistence.save_current(self.session)
        payload = self.persistence.load_current()
        self.assertIsNotNone(payload)
        restored, report = restore_payload(payload, current_session=None)
        self.assertTrue(report["replaced"])
        replay = replay_payload(payload)
        self.assertTrue(replay["ok"])
        self.assertEqual(self.session.content_digest(), restored.content_digest())


class CliAndDemoTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = Path(tempfile.mkdtemp(prefix="sched-diag-cli-"))

    def tearDown(self) -> None:
        cleanup_tree(self.temp)

    def _invoke(self, arguments: list[str]) -> tuple[int, str, str]:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(arguments)
        return code, stdout.getvalue(), stderr.getvalue()

    def test_manifest_and_ref_injections_have_business_codes_and_structured_logs(self) -> None:
        base = [
            "--nodes",
            str(SAMPLES / "nodes.json"),
            "--pods",
            str(SAMPLES / "pods.json"),
            "--plugins",
            str(SAMPLES / "plugins.json"),
            "--workdir",
            str(self.temp / "work"),
        ]
        code, _, stderr = self._invoke(["run", *base, "--inject", "manifest"])
        self.assertEqual(2, code)
        self.assertIn("E_MANIFEST", stderr)
        code, _, stderr = self._invoke(["run", *base, "--inject", "ref"])
        self.assertEqual(2, code)
        self.assertIn("E_REF", stderr)
        log_path = self.temp / "work" / "logs" / "workbench.jsonl"
        records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
        codes = [record.get("code") for record in records if record.get("level") == "ERROR"]
        self.assertIn("E_MANIFEST", codes)
        self.assertIn("E_REF", codes)

    def test_snapshot_digest_injection_uses_altered_file_and_logs_snapshot_code(self) -> None:
        code, _, _ = self._invoke(
            [
                "snapshot",
                "--nodes",
                str(SAMPLES / "nodes.json"),
                "--pods",
                str(SAMPLES / "pods.json"),
                "--plugins",
                str(SAMPLES / "plugins.json"),
                "--workdir",
                str(self.temp / "work"),
                "--name",
                "capture",
                "--json",
            ]
        )
        self.assertEqual(0, code)
        code, _, stderr = self._invoke(
            [
                "restore",
                "--workdir",
                str(self.temp / "work"),
                "--name",
                "capture",
                "--nodes",
                str(SAMPLES / "nodes.json"),
                "--pods",
                str(SAMPLES / "pods.json"),
                "--plugins",
                str(SAMPLES / "plugins.json"),
                "--inject",
                "snapshot-digest",
            ]
        )
        self.assertEqual(2, code)
        self.assertIn("E_SNAPSHOT", stderr)

    def test_demo_runs_all_three_scenarios(self) -> None:
        lines: list[str] = []
        result = run_demo(self.temp / "demo", printer=lines.append)
        self.assertTrue(result["ok"])
        self.assertIn("DEMO_OK full_schedule=true incremental=true snapshot_restore=true replay=true", lines)
        self.assertTrue(any("INCREMENTAL_VERIFY ok=True" in line for line in lines))
        self.assertTrue(any("fullRecomputeMatched=True" in line for line in lines))
        self.assertEqual(["cache-index", "batch-report"], result["incremental"]["affectedPods"])


if __name__ == "__main__":
    unittest.main()
