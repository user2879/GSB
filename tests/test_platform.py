"""configdist 平台测试：格式校验、归一续做、调度、收敛、重放与恢复。"""

import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configdist.errors import (FormatError, PortError, PreviewError,
                               ProtoError, ReplayError)
from configdist.latency import LatencyRecorder
from configdist.manifest import decode_manifest, encode_manifest
from configdist.orchestrator import validate_receipt
from configdist.platform import Platform


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class WorkspaceMixin(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="configdist-test-")
        self.addCleanup(shutil.rmtree, self.workdir, True)

    def make_platform(self, fudge_node=None):
        platform = Platform(self.workdir)
        platform.gen_samples()
        with open(platform.registry_path, encoding="utf-8") as fh:
            reg = json.load(fh)
        for node in reg["nodes"]:
            node["port"] = free_port()
            if node["id"] == fudge_node:
                node["fudge"] = 1
        with open(platform.registry_path, "w", encoding="utf-8") as fh:
            json.dump(reg, fh)
        platform.normalize()
        platform.open()
        self.addCleanup(platform.close)
        return platform


class TestFormat(WorkspaceMixin):
    def test_unknown_version(self):
        with self.assertRaises(FormatError) as ctx:
            decode_manifest('{"format":"v9","entries":[]}')
        self.assertEqual(ctx.exception.code, "E_FORMAT")
        self.assertEqual(ctx.exception.fields["path"], "format")

    def test_missing_field(self):
        with self.assertRaises(FormatError) as ctx:
            decode_manifest('{"format":"v2","entries":[{"key":"a",'
                            '"value":1}]}')
        self.assertEqual(ctx.exception.code, "E_FORMAT")
        self.assertEqual(ctx.exception.fields["source_version"], "v2")
        self.assertEqual(ctx.exception.fields["path"], "manifest_version")

    def test_duplicate_key(self):
        text = ('{"format":"v1","entries":[{"key":"a","value":1},'
                '{"key":"a","value":2}]}')
        with self.assertRaises(FormatError) as ctx:
            decode_manifest(text)
        self.assertIn("entries[1].key", ctx.exception.fields["path"])

    def test_duplicate_json_object_key(self):
        with self.assertRaises(FormatError):
            decode_manifest('{"format":"v1","format":"v1","entries":[]}')

    def test_unsupported_value_type(self):
        text = '{"format":"v1","entries":[{"key":"a","value":1.5}]}'
        with self.assertRaises(FormatError) as ctx:
            decode_manifest(text)
        self.assertEqual(ctx.exception.fields["path"], "entries[0].value")

    def test_derived_unknown_ref(self):
        text = ('{"format":"v3","manifest_version":1,'
                '"entries":[{"key":"a","value":1}],'
                '"derived":[{"key":"d","op":"add","inputs":["a","nope"]}]}')
        with self.assertRaises(FormatError) as ctx:
            decode_manifest(text)
        self.assertIn("derived[0].inputs", ctx.exception.fields["path"])

    def test_valid_v3(self):
        text = ('{"format":"v3","manifest_version":2,'
                '"strategy":{"timeout_ms":100},'
                '"entries":[{"key":"a","value":1,"targets":["*"],'
                '"priority":5}],'
                '"derived":[{"key":"d","op":"add","inputs":["a","a"]}]}')
        manifest = decode_manifest(text)
        self.assertEqual(manifest.manifest_version, 2)
        self.assertEqual(manifest.entries[0].priority, 5)


class TestNormalize(WorkspaceMixin):
    def test_normalize_three_versions(self):
        platform = Platform(self.workdir)
        platform.gen_samples()
        reports = platform.normalize()
        by_file = {r["file"]: r for r in reports}
        self.assertEqual(by_file["app_v1.json"]["source_version"], "v1")
        self.assertEqual(by_file["app_v1.json"]["target_version"], "v3")
        self.assertEqual(by_file["app_v2.json"]["source_version"], "v2")
        self.assertEqual(by_file["app_v3.json"]["status"], "already-current")
        for name, ver in (("app_v1.json", "v1"), ("app_v2.json", "v2")):
            backup = os.path.join(platform.manifest_dir, f"{name}.{ver}.bak")
            self.assertTrue(os.path.exists(backup))
        again = platform.normalize()
        self.assertTrue(all(r["status"] == "already-current" for r in again))

    def test_resume_after_kill(self):
        mdir = os.path.join(self.workdir, "samples", "manifests")
        os.makedirs(mdir)
        entries = [{"key": f"k{i:03d}", "value": i} for i in range(25)]
        with open(os.path.join(mdir, "big.json"), "w",
                  encoding="utf-8") as fh:
            json.dump({"format": "v1", "entries": entries}, fh)
        env = dict(os.environ, CONFIGDIST_UPGRADE_DELAY_MS="150")
        proc = subprocess.Popen(
            [sys.executable, "-m", "configdist.cli", "normalize",
             "--workdir", self.workdir],
            env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        time.sleep(1.0)
        proc.kill()
        proc.wait()
        ckpt = os.path.join(mdir, "big.json.upgchk.json")
        with open(ckpt, encoding="utf-8") as fh:
            done = json.load(fh)["next_index"]
        self.assertGreater(done, 0)
        self.assertLess(done, 25)
        platform = Platform(self.workdir)
        report = platform.normalize()[0]
        self.assertEqual(report["resumed_from"], done)
        self.assertEqual(report["records"], 25)
        self.assertFalse(os.path.exists(ckpt))


class TestDispatch(WorkspaceMixin):
    def test_dispatch_converge_and_quota(self):
        platform = self.make_platform()
        platform.start_nodes()
        manifest = platform.load_manifest("app_v3.json")
        report = platform.dispatch(manifest)
        self.assertEqual(report["deviations"], [])
        self.assertEqual(report["ok"], report["slices"])
        self.assertLessEqual(report["quota"]["global_peak"], 3)
        self.assertLessEqual(max(report["quota"]["node_peak"].values()), 2)
        self.assertEqual(report["start_order"][0][0], "rho")
        result = platform.recompute_authoritative()
        self.assertTrue(result["match"])

    def test_converge_mismatch(self):
        platform = self.make_platform(fudge_node="node-a")
        platform.start_nodes()
        manifest = platform.load_manifest("app_v3.json")
        report = platform.dispatch(manifest)
        self.assertTrue(report["deviations"])
        self.assertEqual(report["error"]["code"], "E_CONVERGE")
        nodes = {d["node"] for d in report["deviations"]}
        self.assertEqual(nodes, {"node-a"})
        for d in report["deviations"]:
            self.assertIn("expected", d)
            self.assertIn("actual", d)
            self.assertIn("key", d)

    def test_node_restart_recovers_state(self):
        platform = self.make_platform()
        platform.start_nodes()
        manifest = platform.load_manifest("app_v3.json")
        platform.dispatch(manifest)
        ack = platform.restart_node("node-a")
        self.assertGreater(ack["state"]["applied"], 0)
        self.assertEqual(ack["state"]["manifest_version"],
                         manifest.manifest_version)
        report = platform.dispatch(manifest)
        self.assertEqual(report["deviations"], [])

    def test_port_conflict(self):
        platform = self.make_platform()
        spec = next(iter(platform.specs.values()))
        blocker = socket.socket()
        blocker.bind(("127.0.0.1", spec.port))
        blocker.listen(1)
        self.addCleanup(blocker.close)
        with self.assertRaises(PortError) as ctx:
            platform.start_nodes()
        self.assertEqual(ctx.exception.code, "E_PORT")

    def test_dispatch_timeout_and_retry(self):
        platform = self.make_platform()
        platform.start_nodes()
        manifest = platform.load_manifest("app_v3.json")
        platform.nodes.kill("node-a")  # 节点离线：连接被拒 -> E_TIMEOUT 重试
        report = platform.dispatch(manifest)
        node_a_devs = [d for d in report["deviations"]
                       if d["node"] == "node-a"]
        self.assertTrue(node_a_devs)
        self.assertTrue(all(d["error"]["code"] == "E_TIMEOUT"
                            for d in node_a_devs))
        rows = platform.ledger.conn.execute(
            "SELECT attempts FROM slices WHERE node='node-a'").fetchall()
        self.assertTrue(all(r[0] == 3 for r in rows))  # 1 + 2 次重试

    def test_proto_error(self):
        with self.assertRaises(ProtoError):
            validate_receipt({"op": "receipt", "node": "n", "view": "s2",
                              "manifest_version": 1, "digest": "x" * 64,
                              "elapsed_ms": 1, "key": "a"}, "n", "s2")


class TestLedgerRecovery(WorkspaceMixin):
    def test_crash_recovery_and_replay(self):
        crash_dir = os.path.join(self.workdir, "crash")
        proc = subprocess.run(
            [sys.executable, "-m", "configdist.cli", "crash-run",
             "--workdir", crash_dir],
            capture_output=True, text=True)
        self.assertNotEqual(proc.returncode, 0)
        platform = Platform(crash_dir)
        recovery = platform.open()
        self.addCleanup(platform.close)
        self.assertEqual(len(recovery["unknown_batches"]), 1)
        platform.start_nodes()
        manifest = platform.load_manifest("app_v3.json")
        report = platform.dispatch(manifest)
        self.assertEqual(report["deviations"], [])
        result = platform.recompute_authoritative()
        self.assertTrue(result["match"])

    def test_replay_tamper_detected(self):
        platform = self.make_platform()
        platform.start_nodes()
        manifest = platform.load_manifest("app_v3.json")
        platform.dispatch(manifest)
        platform.ledger.conn.execute(
            "UPDATE receipts SET digest=? WHERE key=? AND node=?",
            ("0" * 64, "alpha", "node-a"))
        with self.assertRaises(ReplayError) as ctx:
            platform.recompute_authoritative()
        exc = ctx.exception
        self.assertEqual(exc.code, "E_REPLAY")
        self.assertEqual(exc.fields["key"], "alpha")
        self.assertEqual(exc.fields["node"], "node-a")
        self.assertEqual(exc.fields["field"], "digest")


class TestPreviewAndStaleness(WorkspaceMixin):
    def test_preview_mismatch(self):
        platform = self.make_platform()
        platform.start_nodes()
        manifest = platform.load_manifest("app_v3.json")
        platform.dispatch(manifest)
        victim = next(iter(platform.preview))
        platform.preview[victim]["digest"] = "f" * 64
        with self.assertRaises(PreviewError) as ctx:
            platform.recompute_authoritative()
        self.assertEqual(ctx.exception.code, "E_PREVIEW")
        self.assertEqual(ctx.exception.fields["node"], victim[0])
        self.assertEqual(ctx.exception.fields["key"], victim[1])

    def test_edit_staleness_and_replacement(self):
        platform = self.make_platform()
        platform.start_nodes()
        manifest = platform.load_manifest("app_v3.json")
        platform.dispatch(manifest)
        old_version = manifest.manifest_version
        new_obj = json.loads(encode_manifest(manifest))
        for entry in new_obj["entries"]:
            if entry["key"] == "alpha":
                entry["value"] = 2000
        edit = platform.edit_manifest("app_v3.json", json.dumps(new_obj))
        self.assertEqual(edit["manifest_version"], old_version + 1)
        self.assertEqual(edit["changed_keys"], ["alpha"])
        self.assertTrue(all(e["stale"] for e in platform.preview.values()))
        view = platform.ui_query()
        self.assertEqual(view["receipts"], [])
        platform.dispatch(platform.current_manifest)
        replacement = platform.replacement_report(old_version,
                                                  edit["manifest_version"])
        alpha_rows = [r for r in replacement["changes"]
                      if r["key"] == "alpha"]
        self.assertTrue(alpha_rows)
        self.assertTrue(all(r["old"] != r["new"] for r in alpha_rows))
        result = platform.recompute_authoritative()
        self.assertTrue(result["match"])


class TestLatency(unittest.TestCase):
    def test_stats(self):
        recorder = LatencyRecorder()
        for ms in [1, 2, 3, 4, 100]:
            recorder.record("ui_query", ms)
        stats = recorder.stats("ui_query")
        self.assertEqual(stats["count"], 5)
        self.assertEqual(stats["mean_milli_ms"], 22000)
        self.assertEqual(stats["max_ms"], 100)
        self.assertEqual(stats["p95_ms"], 100)
        self.assertEqual(stats["quant_bound_milli_ms"], 500)

    def test_latency_event(self):
        platform_dir = tempfile.mkdtemp(prefix="configdist-lat-")
        self.addCleanup(shutil.rmtree, platform_dir, True)
        platform = Platform(platform_dir)
        platform.recorder.record("ui_query", 500)
        report = platform.latency_report(budgets={"ui_query": 50})
        self.assertEqual(report["events"][0]["code"], "E_LATENCY")


if __name__ == "__main__":
    unittest.main()
