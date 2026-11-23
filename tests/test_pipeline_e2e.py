import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from cfgpipe.errors import ConvergeError, ReplayError
from cfgpipe.pipeline import Pipeline, free_port
from cfgpipe.replay import replay_verify

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_workspace(workdir):
    os.makedirs(os.path.join(workdir, "changesets"), exist_ok=True)
    decl = {
        "session": {"scale_exp": 3, "groups": 8, "global_concurrency": 2,
                    "node_concurrency": 1, "dispatch_timeout_ms": 800,
                    "backoff_ms": [20, 40]},
        "items": [
            {"key": "alpha", "value": 1.5, "version": 1, "expire_ms": 5000},
            {"key": "beta", "value": 0, "version": 2, "expire_ms": 5000,
             "derived_from": "alpha", "derived_coeff": 2.5},
            {"key": "gamma", "value": 3.25, "version": 3, "expire_ms": 5000},
            {"key": "delta", "value": 0, "version": 4, "expire_ms": 5000,
             "derived_from": "beta", "derived_coeff": 0.5},
            {"key": "epsilon", "value": 7, "version": 5, "expire_ms": 5000},
        ],
        "nodes": [
            {"id": "n1", "host": "127.0.0.1", "port": free_port(),
             "subscribes": ["alpha", "beta", "delta"], "seed": 11,
             "sample_size": 120},
            {"id": "n2", "host": "127.0.0.1", "port": free_port(),
             "subscribes": ["beta", "delta", "gamma"], "seed": 22,
             "sample_size": 90},
            {"id": "n3", "host": "127.0.0.1", "port": free_port(),
             "subscribes": ["gamma", "epsilon"], "seed": 33,
             "sample_size": 60},
        ],
    }
    with open(os.path.join(workdir, "declaration.json"), "w") as fh:
        json.dump(decl, fh)
    cs1 = {"version": 10, "changes": [{"key": "alpha", "value": 1.75}]}
    cs2 = {"version": 11, "changes": [
        {"key": "epsilon", "value": 8.5},
        {"key": "gamma", "subscribers_add": ["n1"]},
    ]}
    for idx, cs in enumerate((cs1, cs2), start=1):
        with open(os.path.join(workdir, "changesets",
                               "changeset_%04d.json" % idx), "w") as fh:
            json.dump(cs, fh)


class PipelineCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workdir = tempfile.mkdtemp(prefix="cfgpipe_e2e_")
        write_workspace(cls.workdir)
        cls.pipe = Pipeline(cls.workdir)
        cls.pipe.load()
        cls.pipe.open_ledger()
        cls.pipe.start_nodes()
        cls.report = cls.pipe.run(progress=lambda *_: None)

    @classmethod
    def tearDownClass(cls):
        cls.pipe.stop_nodes()
        cls.pipe.ledger.close()
        shutil.rmtree(cls.workdir, ignore_errors=True)

    def test_invalidation_sets(self):
        b1, b2 = self.report["batches"][1], self.report["batches"][2]
        self.assertEqual(b1["affected_items"], ["alpha", "beta", "delta"])
        self.assertEqual(b1["affected_nodes"], ["n1", "n2"])
        self.assertEqual(b2["affected_items"], ["epsilon"])
        self.assertEqual(b2["affected_nodes"], ["n1", "n3"])

    def test_byte_stability(self):
        for entry in self.report["byte_stability"]:
            for item in entry["unchanged_items"]:
                self.assertTrue(item["identical"], item)
        keys1 = {e["key"] for e in self.report["byte_stability"][0]
                 ["unchanged_items"]}
        self.assertEqual(keys1, {"gamma", "epsilon"})

    def test_incremental_dispatch_only_touches_affected(self):
        log = {e["batch_id"]: e for e in self.pipe.dispatch_log}
        self.assertEqual(log["batch_0001"]["upsert"],
                         {"n1": ["alpha", "beta", "delta"],
                          "n2": ["beta", "delta"]})
        self.assertEqual(log["batch_0002"]["upsert"],
                         {"n1": ["gamma"], "n3": ["epsilon"]})

    def test_carried_receipts(self):
        receipts = self.report["batches"][2]["receipts"]
        self.assertEqual(receipts["n2"], "carried")
        self.assertEqual(receipts["n1"], "new")

    def test_full_rebuild_compare(self):
        result = self.pipe.full_rebuild_compare()
        self.assertEqual(result["item_diffs"], [])

    def test_replay_verify_ok(self):
        replay_verify(self.pipe.ledger, self.pipe.decl, self.pipe.changesets)

    def test_ledger_content(self):
        info = self.pipe.ledger.recovery_scan()
        self.assertEqual(info.cursor, 2)
        self.assertEqual(info.unknown_nodes, [])
        bounds = self.pipe.ledger.error_bounds_of("batch_0001")
        self.assertIn("alpha", bounds)
        tr, de, ag, total = bounds["alpha"]
        self.assertEqual((tr, de), (1, 0))
        self.assertEqual(total, tr + de + ag)


class CrashRecoveryCase(unittest.TestCase):
    def test_crash_marks_unknown_and_resume(self):
        workdir = tempfile.mkdtemp(prefix="cfgpipe_crash_")
        try:
            write_workspace(workdir)
            pipe = Pipeline(workdir)
            pipe.load()
            pipe.open_ledger()
            pipe.start_nodes()
            pipe.run(upto=1, progress=lambda *_: None)
            # 强制终止模拟：子进程写入 inflight 后 os._exit
            proc = subprocess.run(
                [sys.executable, "-m", "cfgpipe.crash_batch",
                 "--workdir", workdir, "--batch-id", "batch_crash",
                 "--nodes", "n1,n3"],
                cwd=REPO, capture_output=True)
            self.assertNotEqual(proc.returncode, 0)

            pipe2 = Pipeline(workdir)
            pipe2.load()
            pipe2.open_ledger()
            report, info = pipe2.resume(progress=lambda *_: None)
            self.assertEqual(info.unknown_nodes, ["n1", "n3"])
            committed = [b["batch_id"] for b in
                         pipe2.ledger.recovery_scan().batches]
            self.assertEqual(committed,
                             ["batch_0000", "batch_0001", "batch_0002"])
            replay_verify(pipe2.ledger, pipe2.decl, pipe2.changesets)
            pipe.stop_nodes()
            pipe2.ledger.close()
            pipe.ledger.close()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


class ConvergeFailureCase(unittest.TestCase):
    def test_digest_mismatch_raises_e_converge(self):
        workdir = tempfile.mkdtemp(prefix="cfgpipe_conv_")
        try:
            write_workspace(workdir)
            pipe = Pipeline(workdir)
            pipe.load()
            pipe.open_ledger()
            pipe.start_nodes()
            try:
                pipe.run(upto=0, progress=lambda *_: None)
                # 篡改期望：人为改动一个节点的回执摘要后触发收敛判定
                from cfgpipe.converge import check_convergence
                from cfgpipe.converge import expected_view
                from cfgpipe.compiler import compile_all
                from cfgpipe.graph import node_subscription_map
                s = pipe.decl.session
                artifacts = compile_all(pipe.state, s.scale_exp, pipe.topo)
                subs = node_subscription_map(pipe.state)
                views = {nd.node_id: expected_view(
                    pipe.state.items, subs.get(nd.node_id, []), artifacts,
                    s.scale_exp) for nd in pipe.decl.nodes}
                from cfgpipe import sample as _sample
                good_version = _sample.config_version_of(views["n1"])
                bad_receipt = {"scale_exp": s.scale_exp,
                               "config_version": good_version,
                               "digest": "0" * 64, "key_digests": {},
                               "elapsed_ms": 1}
                with self.assertRaises(ConvergeError) as ctx:
                    check_convergence(
                        list(pipe.decl.nodes), {"n1"},
                        {"n1": bad_receipt}, views,
                        pipe.ledger.last_receipts(), s.scale_exp, s.groups)
                err = ctx.exception
                self.assertEqual(err.code, "E_CONVERGE")
                self.assertEqual(err.node, "n1")
                self.assertEqual(err.field, "digest")
                # 首个偏差键稳定（字典序最小）
                self.assertEqual(err.key, "alpha")
            finally:
                pipe.stop_nodes()
                pipe.ledger.close()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


class ReplayFailureCase(unittest.TestCase):
    def test_tampered_ledger_raises_e_replay(self):
        workdir = tempfile.mkdtemp(prefix="cfgpipe_replay_")
        try:
            write_workspace(workdir)
            pipe = Pipeline(workdir)
            pipe.load()
            pipe.open_ledger()
            pipe.start_nodes()
            pipe.run(progress=lambda *_: None)
            pipe.stop_nodes()
            # 篡改账本中的一条回执摘要
            pipe.ledger.db.execute(
                "UPDATE receipts SET digest = ? WHERE batch_id = 'batch_0001'"
                " AND node_id = 'n1'", ("f" * 64,))
            pipe.ledger.db.commit()
            with self.assertRaises(ReplayError) as ctx:
                replay_verify(pipe.ledger, pipe.decl, pipe.changesets)
            err = ctx.exception
            self.assertEqual(err.code, "E_REPLAY")
            self.assertEqual(err.node, "n1")
            self.assertEqual(err.field, "digest")
            pipe.ledger.close()
        finally:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
