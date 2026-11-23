from __future__ import annotations

import json
import math
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

from gateway_diag.backend_client import BackendClient
from gateway_diag.backend_server import MockBackendServer
from gateway_diag.compare import validate_incremental_vs_full
from gateway_diag.errors import E_CYCLE, E_REF, E_REPLAY, E_SNAPSHOT, E_SPEC, E_VERIFY, GatewayError
from gateway_diag.graph import dependency_graph_digest
from gateway_diag.ledger import Ledger
from gateway_diag.loaders import load_spec
from gateway_diag.logging_utils import StructuredLogger
from gateway_diag.models import Backend, Spec
from gateway_diag.routing import Router
from gateway_diag.util import canonical_json, digest_value
from gateway_diag.verify import verify_ledger
from gateway_diag.workflow import run_session

ROOT = Path(__file__).resolve().parents[1]


def free_ports(count: int) -> list[int]:
    import socket

    sockets = []
    ports: list[int] = []
    try:
        for _ in range(count):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("127.0.0.1", 0))
            sockets.append(sock)
            ports.append(int(sock.getsockname()[1]))
    finally:
        for sock in sockets:
            sock.close()
    return ports


def make_payload(
    ports: list[int],
    *,
    reverse: bool = False,
    update_rule: bool = False,
    update_backend: bool = False,
    cycle: bool = False,
    dangling: bool = False,
) -> dict[str, Any]:
    backends = [
        {"id": "b1", "host": "127.0.0.1", "port": ports[0], "model_dim": 4, "hidden_dim": 3, "seed": 11, "quantization": "fp32"},
        {"id": "b2", "host": "127.0.0.1", "port": ports[1], "model_dim": 4, "hidden_dim": 2, "seed": 22, "quantization": "fp16"},
        {"id": "b3", "host": "127.0.0.1", "port": ports[2], "model_dim": 3, "hidden_dim": 2, "seed": 33, "quantization": "int8"},
        {"id": "b4", "host": "127.0.0.1", "port": ports[3], "model_dim": 3, "hidden_dim": 3, "seed": 44, "quantization": "fp32"},
    ]
    rules = [
        {"id": "r1", "request_prefix": "/a", "stable_backend": "b1", "canary_backend": "b2", "canary_weight_bp": 4000, "batch_size": 2, "version": 1, "depends_on": []},
        {"id": "r2", "request_prefix": "/a/deep", "stable_backend": "b1", "canary_backend": {"from_rule": "r1", "select": "conclusion"}, "canary_weight_bp": 6000, "batch_size": 2, "version": 1},
        {"id": "r3", "request_prefix": "/b", "stable_backend": "b3", "canary_backend": "b4", "canary_weight_bp": 5000, "batch_size": 3, "version": 1, "depends_on": []},
        {"id": "r4", "request_prefix": "/b/audit", "stable_backend": "b3", "canary_backend": {"from_rule": "r3", "select": "conclusion"}, "canary_weight_bp": 3000, "batch_size": 2, "version": 1, "depends_on": ["r3"]},
    ]
    if update_rule:
        rules[0]["canary_weight_bp"] = 7300
        rules[0]["version"] = 2
    if update_backend:
        backends[1]["seed"] = 909
        backends[1]["hidden_dim"] = 4
    if cycle:
        rules[0]["canary_backend"] = {"from_rule": "r3", "select": "conclusion"}
        rules[2]["canary_backend"] = {"from_rule": "r1", "select": "conclusion"}
    if dangling:
        rules[0]["canary_backend"] = "__missing__"
        rules[2]["canary_backend"] = "__missing__"
    if reverse:
        backends = list(reversed(backends))
        rules = list(reversed(rules))
    return {"schema_version": 1, "batch_limit": 4, "backends": backends, "rules": rules}


def write_requests(path: Path) -> None:
    prefixes = ["/a", "/a/deep", "/b", "/b/audit"]
    lines = []
    for index in range(1, 31):
        prefix = prefixes[(index - 1) % len(prefixes)]
        dim = 4 if prefix.startswith("/a") else 3
        vector = [
            round(math.sin(index * 0.13 + component * 0.71) + math.cos(index * 0.07), 6)
            for component in range(dim)
        ]
        lines.append(
            json.dumps(
                {
                    "request_id": f"req-{index:04d}",
                    "arrival_seq": index,
                    "client_key": f"client-{index % 7:02d}",
                    "prefix": prefix,
                    "vector": vector,
                },
                separators=(",", ":"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


class CoreValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="gwdiag-core-")
        self.dir = Path(self.temp.name)
        self.ports = free_ports(4)
        self.requests = self.dir / "requests.ndjson"
        write_requests(self.requests)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write(self, name: str, payload: Any) -> Path:
        path = self.dir / name
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return path

    def test_graph_digest_independent_of_declaration_order(self) -> None:
        first = self._write("first.json", make_payload(self.ports))
        second = self._write("second.json", make_payload(self.ports, reverse=True))
        spec1 = load_spec(config_path=first, requests_path=self.requests)
        spec2 = load_spec(config_path=second, requests_path=self.requests)
        self.assertEqual(dependency_graph_digest(spec1), dependency_graph_digest(spec2))
        self.assertEqual(spec1.declaration_digest, spec2.declaration_digest)

    def test_cycle_reports_rule_sequence(self) -> None:
        path = self._write("cycle.json", make_payload(self.ports, cycle=True))
        with self.assertRaises(GatewayError) as caught:
            load_spec(config_path=path, requests_path=self.requests)
        self.assertEqual(caught.exception.code, E_CYCLE)
        self.assertGreaterEqual(len(caught.exception.details["cycle"]), 3)
        self.assertEqual(
            caught.exception.details["cycle"][0],
            caught.exception.details["cycle"][-1],
        )

    def test_dangling_reference_lists_all_referrers(self) -> None:
        path = self._write("dangling.json", make_payload(self.ports, dangling=True))
        with self.assertRaises(GatewayError) as caught:
            load_spec(config_path=path, requests_path=self.requests)
        self.assertEqual(caught.exception.code, E_REF)
        self.assertEqual(caught.exception.details["referrers"], ["r1", "r3"])

    def test_bad_vector_length_is_e_spec(self) -> None:
        payload = make_payload(self.ports)
        path = self._write("config.json", payload)
        bad_requests = self.dir / "bad.ndjson"
        bad_requests.write_text(
            json.dumps(
                {"request_id": "x", "arrival_seq": 1, "client_key": "c", "prefix": "/a", "vector": [1.0, 2.0]},
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        with self.assertRaises(GatewayError) as caught:
            load_spec(config_path=path, requests_path=bad_requests)
        self.assertEqual(caught.exception.code, E_SPEC)
        self.assertEqual(caught.exception.field, "vector")

    def test_stable_hash_and_weight_endpoints(self) -> None:
        path = self._write("config.json", make_payload(self.ports))
        spec = load_spec(config_path=path, requests_path=self.requests)
        first = Router.bucket("r1", "same-client")
        self.assertEqual(first, Router.bucket("r1", "same-client"))
        self.assertEqual(Router.choose_destination(first, 0), "stable")
        self.assertEqual(Router.choose_destination(first, 10000), "canary")
        router = Router(spec)
        self.assertEqual(router.decide("r1", "same-client"), router.decide("r1", "same-client"))


class EndToEndTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.ports = free_ports(4)
        cls.backends = [
            Backend("b1", "127.0.0.1", cls.ports[0], 4, 3, 11, "fp32"),
            Backend("b2", "127.0.0.1", cls.ports[1], 4, 2, 22, "fp16"),
            Backend("b3", "127.0.0.1", cls.ports[2], 3, 2, 33, "int8"),
            Backend("b4", "127.0.0.1", cls.ports[3], 3, 3, 44, "fp32"),
        ]
        cls.server = MockBackendServer(cls.backends, allow_dynamic=True).start()
        cls.requests_file = Path(tempfile.mkdtemp(prefix="gwdiag-requests-")) / "requests.ndjson"
        write_requests(cls.requests_file)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.stop()

    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory(prefix="gwdiag-e2e-")
        self.dir = Path(self.temp.name)
        self.logger = StructuredLogger(self.dir / "events.jsonl")
        self.client = BackendClient(timeout=10)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _config(self, name: str, **kwargs) -> Spec:
        path = self.dir / name
        path.write_text(json.dumps(make_payload(self.ports, **kwargs), ensure_ascii=False, indent=2) + "\n")
        return load_spec(config_path=path, requests_path=self.requests_file)

    def test_incremental_matches_full_and_preserves_unaffected_bytes(self) -> None:
        before = self._config("before.json")
        after = self._config("after.json", update_rule=True)
        base_dir = self.dir / "base"
        incr_dir = self.dir / "incremental"
        full_dir = self.dir / "full"
        run_session(before, workdir=base_dir, client=self.client, logger=self.logger)
        with Ledger(base_dir) as ledger:
            baseline_r3 = [row["receipt_json"] for row in ledger.get_rule_receipt_records("r3")]
            baseline_r4 = [row["receipt_json"] for row in ledger.get_rule_receipt_records("r4")]
        shutil.copytree(base_dir, incr_dir)
        update_report = run_session(after, workdir=incr_dir, client=self.client, logger=self.logger)
        self.assertEqual(
            [step["rule_id"] for step in update_report["invalidation"]["steps"]],
            ["r1", "r2"],
        )
        self.assertEqual(update_report["invalidation"]["unaffected_rules"], ["r3", "r4"])
        self.assertEqual(update_report["replacement"]["rules"], ["r1", "r2"])
        with Ledger(incr_dir) as ledger:
            self.assertEqual(
                baseline_r3,
                [row["receipt_json"] for row in ledger.get_rule_receipt_records("r3")],
            )
            self.assertEqual(
                baseline_r4,
                [row["receipt_json"] for row in ledger.get_rule_receipt_records("r4")],
            )
            for rule_id in ("r3", "r4"):
                for receipt_json in [row["receipt_json"] for row in ledger.get_rule_receipt_records(rule_id)]:
                    receipt = json.loads(receipt_json)
                    self.assertNotIn("code", receipt)
                    self.assertNotIn("stale", receipt)
        run_session(after, workdir=full_dir, client=self.client, logger=self.logger)
        with Ledger(incr_dir, read_only=True) as incr, Ledger(full_dir, read_only=True) as full:
            incr_receipts = incr.all_rule_receipts()
            full_receipts = full.all_rule_receipts()
            for rule_id in ("r1", "r2", "r3", "r4"):
                self.assertEqual(
                    [item["output_digest"] for item in incr_receipts[rule_id]],
                    [item["output_digest"] for item in full_receipts[rule_id]],
                )
                self.assertEqual(
                    [item["request_ids"] for item in incr_receipts[rule_id]],
                    [item["request_ids"] for item in full_receipts[rule_id]],
                )
        expected_batches = sum(len(value) for value in incr_receipts.values())
        report = verify_ledger(after, workdir=incr_dir, client=self.client, logger=self.logger)
        self.assertEqual(report["result"], "verified")
        self.assertEqual(report["batches_checked"], expected_batches)

    def test_backend_update_propagates_and_lists_allowed_difference(self) -> None:
        before = self._config("before.json")
        after = self._config("after.json", update_backend=True)
        result = validate_incremental_vs_full(
            before=before,
            after=after,
            workdir=self.dir / "validation",
            client=self.client,
            logger=self.logger,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(
            [step["rule_id"] for step in result["invalidation"]["steps"]],
            ["r1", "r2"],
        )
        self.assertEqual(result["invalidation"]["unaffected_rules"], ["r3", "r4"])
        self.assertTrue(result["unaffected_receipts"]["identical"])
        self.assertEqual(result["comparison"]["result"], "identical")
        self.assertTrue(any(item["backend_id"] == "b2" for item in result["allowed_differences"]))

    def test_prefix_addition_invalidates_rules_that_lose_requests(self) -> None:
        special_requests = self.dir / "special.ndjson"
        lines = []
        for index in range(1, 13):
            prefix = "/a/special" if index in {3, 6, 9} else ("/a/deep" if index % 2 else "/a")
            lines.append(
                json.dumps(
                    {
                        "request_id": f"prefix-{index:04d}",
                        "arrival_seq": index,
                        "client_key": f"client-{index % 5}",
                        "prefix": prefix,
                        "vector": [round(math.sin(index + j), 6) for j in range(4)],
                    },
                    separators=(",", ":"),
                )
            )
        special_requests.write_text("\n".join(lines) + "\n", encoding="utf-8")
        before_payload = make_payload(self.ports)
        after_payload = make_payload(self.ports)
        after_payload["rules"].append(
            {
                "id": "r5",
                "request_prefix": "/a/special",
                "stable_backend": "b1",
                "canary_backend": "b2",
                "canary_weight_bp": 5000,
                "batch_size": 2,
                "version": 1,
            }
        )
        before_path = self.dir / "prefix-before.json"
        after_path = self.dir / "prefix-after.json"
        before_path.write_text(json.dumps(before_payload), encoding="utf-8")
        after_path.write_text(json.dumps(after_payload), encoding="utf-8")
        before = load_spec(config_path=before_path, requests_path=special_requests)
        after = load_spec(config_path=after_path, requests_path=special_requests)
        base_dir = self.dir / "prefix-base"
        incr_dir = self.dir / "prefix-incr"
        full_dir = self.dir / "prefix-full"
        run_session(before, workdir=base_dir, client=self.client, logger=self.logger)
        shutil.copytree(base_dir, incr_dir)
        report = run_session(after, workdir=incr_dir, client=self.client, logger=self.logger)
        self.assertIn("r1", report["invalidation"]["affected_rules"])
        r1_step = next(step for step in report["invalidation"]["steps"] if step["rule_id"] == "r1")
        self.assertEqual(r1_step["basis"]["type"], "request_assignment_changed")
        run_session(after, workdir=full_dir, client=self.client, logger=self.logger)
        with Ledger(incr_dir, read_only=True) as incr, Ledger(full_dir, read_only=True) as full:
            self.assertEqual(incr.all_rule_receipts(), full.all_rule_receipts())

    def test_completed_run_regenerates_missing_or_stale_snapshot(self) -> None:
        before = self._config("before.json")
        after = self._config("after.json", update_rule=True)
        workdir = self.dir / "snapshot-repair"
        run_session(before, workdir=workdir, client=self.client, logger=self.logger)
        old_snapshot = (workdir / "snapshot.json").read_bytes()
        run_session(after, workdir=workdir, client=self.client, logger=self.logger)
        completed_snapshot = (workdir / "snapshot.json").read_bytes()

        (workdir / "snapshot.json").unlink()
        missing = run_session(after, workdir=workdir, client=self.client, logger=self.logger)
        self.assertTrue(missing["snapshot_repaired"])
        self.assertTrue((workdir / "snapshot.json").exists())

        (workdir / "snapshot.json").write_bytes(old_snapshot)
        stale = run_session(after, workdir=workdir, client=self.client, logger=self.logger)
        self.assertTrue(stale["snapshot_repaired"])
        self.assertNotEqual(old_snapshot, (workdir / "snapshot.json").read_bytes())
        self.assertIn(b'"run_status":"complete"', completed_snapshot)

    def test_crash_recovery_and_replay_mismatch(self) -> None:
        config_path = self.dir / "config.json"
        config_path.write_text(json.dumps(make_payload(self.ports), ensure_ascii=False, indent=2) + "\n")
        workdir = self.dir / "crash"
        command = [
            sys.executable,
            "-m",
            "gateway_diag",
            "run",
            "--config",
            str(config_path),
            "--requests",
            str(self.requests_file),
            "--workdir",
            str(workdir),
            "--crash-after-batches",
            "2",
        ]
        crashed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(crashed.returncode, 137)
        self.assertEqual(crashed.stdout, "")
        self.assertFalse((workdir / "snapshot.json").exists())
        with Ledger(workdir) as ledger:
            states = ledger.get_rule_states()
            self.assertTrue(any(state["status"] == "in_progress" for state in states.values()))
            self.assertFalse(any(state["status"] == "complete" for state in states.values()))

        resumed = subprocess.run(command[:-2], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
        payload = json.loads(resumed.stdout)
        self.assertTrue(payload["recovered"])
        self.assertTrue(payload["recovery"]["marked_unknown_rules"])
        spec = load_spec(config_path=config_path, requests_path=self.requests_file)
        verification = verify_ledger(spec, workdir=workdir, client=self.client, logger=self.logger)
        self.assertEqual(verification["result"], "verified")

        replay_dir = self.dir / "replay"
        replay_command = command[:-1] + ["2"]
        replay_command = [part.replace(str(workdir), str(replay_dir)) for part in replay_command]
        crashed2 = subprocess.run(replay_command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(crashed2.returncode, 137)
        self._tamper_first_partial_receipt(replay_dir)
        replay = subprocess.run(
            [part for part in replay_command if part not in {"--crash-after-batches", "2"}],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(replay.returncode, 2, replay.stdout + replay.stderr)
        error = json.loads(replay.stdout)
        self.assertEqual(error["code"], E_REPLAY)
        self.assertIn(error["field"], {"output_digest", "receipt_digest"})

    def _tamper_first_partial_receipt(self, workdir: Path) -> None:
        path = workdir / "ledger.sqlite3"
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT rule_id,batch_index,receipt_json FROM receipts ORDER BY rule_id,batch_index LIMIT 1"
        ).fetchone()
        self.assertIsNotNone(row)
        receipt = json.loads(row["receipt_json"])
        receipt.pop("receipt_digest")
        receipt["output_digest"] = "sha256:" + "f" * 64
        receipt["receipt_digest"] = digest_value(receipt)
        with conn:
            conn.execute(
                "UPDATE receipts SET receipt_json=?,receipt_digest=?,output_digest=? WHERE rule_id=? AND batch_index=?",
                (
                    canonical_json(receipt),
                    receipt["receipt_digest"],
                    receipt["output_digest"],
                    row["rule_id"],
                    row["batch_index"],
                ),
            )
            conn.execute(
                "UPDATE batch_commits SET receipt_digest=? WHERE rule_id=? AND batch_index=?",
                (receipt["receipt_digest"], row["rule_id"], row["batch_index"]),
            )
        holder = Ledger.__new__(Ledger)
        holder.conn = conn
        holder.read_only = False
        with conn:
            conn.execute(
                "UPDATE meta SET value=? WHERE key='content_digest'",
                (holder.compute_content_digest(),),
            )
        conn.close()

    def test_injection_codes_and_verifier_detection(self) -> None:
        config_path = self.dir / "config.json"
        config_path.write_text(json.dumps(make_payload(self.ports), ensure_ascii=False, indent=2) + "\n")
        broken = self.dir / "broken.json"
        broken.write_text('{"batch_limit": 4, "backends": [', encoding="utf-8")
        with self.assertRaises(GatewayError) as spec_error:
            load_spec(config_path=broken, requests_path=self.requests_file)
        self.assertEqual(spec_error.exception.code, E_SPEC)

        dangling = self.dir / "dangling.json"
        dangling.write_text(json.dumps(make_payload(self.ports, dangling=True), ensure_ascii=False) + "\n")
        with self.assertRaises(GatewayError) as ref_error:
            load_spec(config_path=dangling, requests_path=self.requests_file)
        self.assertEqual(ref_error.exception.code, E_REF)

        workdir = self.dir / "verified"
        spec = load_spec(config_path=config_path, requests_path=self.requests_file)
        run_session(spec, workdir=workdir, client=self.client, logger=self.logger)
        conn = sqlite3.connect(workdir / "ledger.sqlite3")
        with conn:
            conn.execute("UPDATE meta SET value=? WHERE key='content_digest'", ("sha256:" + "0" * 64,))
        conn.close()
        with self.assertRaises(GatewayError) as snapshot_error:
            Ledger(workdir)
        self.assertEqual(snapshot_error.exception.code, E_SNAPSHOT)

        tampered = self.dir / "tampered"
        shutil.copytree(workdir, tampered)
        self._coherently_tamper_completed_receipt(tampered)
        with self.assertRaises(GatewayError) as verify_error:
            verify_ledger(spec, workdir=tampered, client=self.client, logger=self.logger)
        self.assertEqual(verify_error.exception.code, E_VERIFY)

    def _coherently_tamper_completed_receipt(self, workdir: Path) -> None:
        conn = sqlite3.connect(workdir / "ledger.sqlite3")
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT rule_id,batch_index,receipt_json FROM receipts ORDER BY rule_id,batch_index LIMIT 1"
        ).fetchone()
        receipt = json.loads(row["receipt_json"])
        receipt.pop("receipt_digest")
        receipt["output_digest"] = "sha256:" + "e" * 64
        receipt["receipt_digest"] = digest_value(receipt)
        with conn:
            conn.execute(
                "UPDATE receipts SET receipt_json=?,receipt_digest=?,output_digest=? WHERE rule_id=? AND batch_index=?",
                (
                    canonical_json(receipt),
                    receipt["receipt_digest"],
                    receipt["output_digest"],
                    row["rule_id"],
                    row["batch_index"],
                ),
            )
            conn.execute(
                "UPDATE batch_commits SET receipt_digest=? WHERE rule_id=? AND batch_index=?",
                (receipt["receipt_digest"], row["rule_id"], row["batch_index"]),
            )
        holder = Ledger.__new__(Ledger)
        holder.conn = conn
        holder.read_only = False
        with conn:
            new_digest = holder.compute_content_digest()
            conn.execute("UPDATE meta SET value=? WHERE key='content_digest'", (new_digest,))
        conn.close()
        snapshot_path = workdir / "snapshot.json"
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        snapshot["ledger_content_digest"] = new_digest
        snapshot_path.write_text(canonical_json(snapshot) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
