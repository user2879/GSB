from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
import random
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from batch_ledger.codes import (  # noqa: E402
    E_ARRIVAL_ORDER,
    E_BATCH_SIZE_INVALID,
    E_RECEIPT_VERSION,
    E_VIEW_DIVERGENCE,
    E_WORKER_INVALID,
)
from batch_ledger.conversion import ConversionBook, SidecarStore, upgrade_document  # noqa: E402
from batch_ledger.engine import LedgerEngine, build_report  # noqa: E402
from batch_ledger.fixedpoint import deterministic_forward  # noqa: E402
from batch_ledger.lifecycle import GracefulStop  # noqa: E402
from batch_ledger.model import MISSING  # noqa: E402
from batch_ledger.receipts import parse_receipt, parse_r1, parse_r3  # noqa: E402
from batch_ledger.samples import FIXED_SEED, make_sample_state, write_sample_workspace  # noqa: E402
from batch_ledger.snapshot import SnapshotStore  # noqa: E402
from batch_ledger.state import apply_command, clone_state  # noqa: E402
from batch_ledger.view import interactive_view  # noqa: E402
from batch_ledger.worker import infer_over_loopback, launch_worker_subprocess  # noqa: E402
from batch_ledger.workspace import load_workspace_dir  # noqa: E402


def final_sample_state() -> dict:
    state = make_sample_state()
    for command in state["command_stream"]:
        apply_command(state, command)
    return state


class ValidationTests(unittest.TestCase):
    def test_all_business_validation_codes_continue_loading(self) -> None:
        state = make_sample_state()
        report = build_report(state)
        codes = report["summary"]["failure_code_counts"]
        self.assertIn(E_ARRIVAL_ORDER, codes)
        self.assertIn(E_RECEIPT_VERSION, codes)
        self.assertIn(E_WORKER_INVALID, codes)
        self.assertGreaterEqual(codes[E_WORKER_INVALID], 2)
        self.assertEqual(report["summary"]["receipt_count"], 32)
        b1_aggregate = next(
            item for item in report["worker_aggregates"] if item["batch_id"] == "B-0001"
        )
        self.assertEqual(b1_aggregate["worker_id"], MISSING)
        self.assertEqual(b1_aggregate["effective_worker_id"], "W1")

        bad_batch = clone_state(state)
        result = apply_command(bad_batch, {"type": "set_batch_size", "batch_size": 0})
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], E_BATCH_SIZE_INVALID)
        bad_batch["config"]["batch_size"] = 0
        bad_report = build_report(bad_batch)
        self.assertIn(E_BATCH_SIZE_INVALID, bad_report["summary"]["failure_code_counts"])

    def test_workspace_files_load_and_generate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = write_sample_workspace(Path(tmp) / "sample")
            loaded = load_workspace_dir(root)
            report = build_report(loaded)
            self.assertEqual(report["summary"]["receipt_count"], 32)
            self.assertTrue((root / "receipts" / "r1.jsonl").exists())
            self.assertTrue((root / "receipts" / "r2.jsonl").exists())
            self.assertTrue((root / "receipts" / "r3.jsonl").exists())


class ParserConversionTests(unittest.TestCase):
    def test_versioned_and_generic_entrypoints_match(self) -> None:
        r3 = {
            "v": "r3",
            "batch_id": "B1",
            "request_id": "Q1",
            "output_digest": "d1",
            "param_version": 3,
            "elapsed_ms": 9,
            "worker_id": "W1",
            "rank_in_batch": 2,
            "supersedes_ref": "rr-old",
            "param_digest": "p1",
        }
        self.assertEqual(parse_receipt(r3).to_dict(), parse_r3(r3).to_dict())
        r1 = {"v": "r1", "batch_id": "B1", "request_id": "Q1", "output_digest": "d1"}
        parsed = parse_r1(r1)
        self.assertEqual(parsed.param_version, MISSING)
        self.assertEqual(parsed.rank_in_batch, MISSING)
        self.assertIn("param_version", parsed.missing_fields)

    def test_r3_downgrade_sidecar_and_restore_roundtrip(self) -> None:
        state = make_sample_state()
        report = build_report(state)
        r3_dicts = [
            item
            for item in report["receipts"]
            if item["source_version"] == "r3" and item["supersedes_ref"] != MISSING
        ]
        self.assertTrue(r3_dicts)

        sidecar = SidecarStore()
        book = ConversionBook()
        restored = []
        for data in r3_dicts:
            direct = parse_receipt(
                {
                    "v": data["source_version"],
                    **{
                        k: v
                        for k, v in data.items()
                        if k
                        not in (
                            "source_version",
                            "missing_fields",
                            "completeness_label",
                            "completeness_score",
                        )
                    },
                }
            )
            key = sidecar.put_receipt(direct)
            book.record_downgrade(direct, "r1", key)
            r1_doc = {"v": "r1", "batch_id": direct.batch_id, "request_id": direct.request_id, "output_digest": direct.output_digest}
            again = upgrade_document(r1_doc, "r3", sidecar)
            book.record_conversion(again, "r3", True)
            restored.append(again)
            self.assertEqual(again.to_dict(), direct.to_dict())

        stats = {
            "entry_count": len(restored),
            "downgrade_count": book.to_dict()["downgrade_entry_count"],
        }
        self.assertEqual(stats["entry_count"], len(r3_dicts))
        self.assertEqual(stats["downgrade_count"], len(r3_dicts))


class AdjudicationTests(unittest.TestCase):
    def _receipt(self, version: str, **overrides):
        doc = {
            "v": version,
            "batch_id": "B1",
            "request_id": "Q1",
            "output_digest": "same-digest",
        }
        if version in ("r2", "r3"):
            doc.update(param_version=1, elapsed_ms=10, worker_id="W1")
        if version == "r3":
            doc.update(rank_in_batch=0, supersedes_ref=MISSING, param_digest="p")
        doc.update(overrides)
        return parse_receipt(doc)

    def test_coverage_reference_selects_authoritative_receipt(self) -> None:
        loser = self._receipt("r2", batch_id="B-old", param_version=10, elapsed_ms=99)
        winner = self._receipt(
            "r3",
            batch_id="B-new",
            worker_id="W2",
            param_version=2,
            elapsed_ms=5,
            supersedes_ref=loser.receipt_ref,
        )
        from batch_ledger.adjudication import adjudicate_requests

        decision = adjudicate_requests([loser, winner])[0]
        self.assertEqual(decision["authoritative_ref"], winner.receipt_ref)
        self.assertIn("coverage_ref", decision["basis"])
        self.assertEqual(decision["rejected_refs"], [loser.receipt_ref])

    def test_no_coverage_uses_version_then_batch(self) -> None:
        from batch_ledger.adjudication import adjudicate_requests

        older_high = self._receipt("r2", batch_id="B-zzz", param_version=9)
        newer_low = self._receipt("r2", batch_id="B-aaa", param_version=2)
        tied_a = self._receipt("r2", batch_id="B-a", param_version=5)
        tied_b = self._receipt("r2", batch_id="B-b", param_version=5)
        decision = adjudicate_requests([newer_low, older_high])[0]
        self.assertEqual(decision["authoritative_ref"], older_high.receipt_ref)
        tied_decision = adjudicate_requests([tied_b, tied_a])[0]
        self.assertEqual(tied_decision["authoritative_batch_id"], "B-a")

    def test_elapsed_orders_multiple_covering_receipts(self) -> None:
        from batch_ledger.adjudication import adjudicate_requests

        loser1 = self._receipt("r2", batch_id="B-l1")
        loser2 = self._receipt("r2", batch_id="B-l2")
        slow_cover = self._receipt("r3", batch_id="B-slow", elapsed_ms=30, supersedes_ref=loser1.receipt_ref)
        fast_cover = self._receipt("r3", batch_id="B-fast", elapsed_ms=4, supersedes_ref=loser2.receipt_ref)
        decision = adjudicate_requests([loser1, loser2, slow_cover, fast_cover])[0]
        self.assertEqual(decision["authoritative_ref"], fast_cover.receipt_ref)


class ConsistencyTests(unittest.TestCase):
    def test_incremental_each_command_matches_blank_full(self) -> None:
        initial = make_sample_state()
        engine = LedgerEngine(clone_state(initial))
        engine.run(parallel=False)
        sequences = [
            initial["command_stream"],
            list(reversed(initial["command_stream"])),
            [
                {"type": "set_batch_size", "batch_size": 1},
                {"type": "set_batch_size", "batch_size": 7},
                {"type": "set_merge_rule", "merge_rule": "version_then_batch_v1"},
            ],
        ]
        for commands in sequences:
            state = make_sample_state()
            engine = LedgerEngine(state)
            engine.run(parallel=False)
            for command in commands:
                engine.apply(command, parallel=False)
                full = build_report(clone_state(engine.state), parallel=False)
                self.assertEqual(engine.current_report, full)

    def test_parallel_and_serial_reports_are_field_equal(self) -> None:
        state = final_sample_state()
        serial = build_report(clone_state(state), parallel=False)
        parallel = build_report(clone_state(state), parallel=True)
        self.assertEqual(serial, parallel)
        again = build_report(clone_state(state), parallel=True)
        self.assertEqual(parallel, again)

    def test_param_version_invalidation_boundary_and_hits(self) -> None:
        state = make_sample_state()
        engine = LedgerEngine(state)
        engine.run(parallel=False)
        switch = next(command for command in state["command_stream"] if command["type"] == "switch_param_version")
        engine.apply(switch, parallel=False)
        stats = engine.last_cache_stats
        self.assertEqual(stats["hits"], 6)
        self.assertEqual(stats["misses"], 2)
        hit_workers = {key.rsplit(":", 1)[-1] for key in stats["hit_resource_keys"]}
        self.assertTrue({"W1", "W3"}.issubset(hit_workers))
        self.assertIn("resource:B-0002:W2", stats["hit_resource_keys"])
        invalidation = engine.current_report["invalidation"]
        self.assertEqual(invalidation["invalid_batch_ids"], ["B-0005", "B-0008"])
        self.assertIn("Q0017", invalidation["invalid_request_ids"])
        self.assertNotIn("B-0001", invalidation["invalid_batch_ids"])


class SnapshotViewLifecycleTests(unittest.TestCase):
    def test_snapshot_commit_resume_and_fingerprint_rebuild(self) -> None:
        state = final_sample_state()
        report = build_report(state)
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(Path(tmp) / "snap")
            manifest = store.commit(report, state)
            loaded = store.load_current_report()
            self.assertEqual(loaded["report_hash"], report["report_hash"])

            decision = store.recovery_decision(
                expected_global_version=manifest["global_version"],
                expected_config_hash=manifest["config_hash"],
                expected_receipt_fingerprints=manifest["receipt_fingerprints"],
            )
            self.assertTrue(decision["resume"])

            changed = store.recovery_decision(
                expected_global_version=manifest["global_version"],
                expected_config_hash="sha256:changed",
                expected_receipt_fingerprints=manifest["receipt_fingerprints"],
            )
            self.assertFalse(changed["resume"])
            self.assertEqual(changed["reason"], "config_hash_changed")

            orphan = store.snapshots_dir / ".staging-orphan"
            orphan.mkdir()
            (orphan / "partial.json").write_text("partial", encoding="utf-8")
            reload_decision = store.recovery_decision(
                expected_global_version=manifest["global_version"],
                expected_config_hash=manifest["config_hash"],
                expected_receipt_fingerprints=manifest["receipt_fingerprints"],
            )
            self.assertTrue(reload_decision["resume"])

    def test_view_consistency_and_divergence_code(self) -> None:
        report = build_report(final_sample_state())
        ids = [row["request_id"] for row in report["decisions"][:8]]
        ok = interactive_view(report, completed_request_ids=ids, elapsed_ms=1)
        self.assertEqual(ok["status"], "consistent")
        self.assertEqual(ok["summary"]["completed_entries"], 8)

        divergence = interactive_view(
            report,
            completed_request_ids=ids[:1],
            previous_summary=ok["summary"],
            mutate_fast_row=lambda row: {**row, "basis": "WRONG"},
        )
        self.assertEqual(divergence["status"], E_VIEW_DIVERGENCE)
        self.assertEqual(divergence["divergences"][0]["code"], E_VIEW_DIVERGENCE)
        self.assertTrue(divergence["preferred_last_completed_summary"])

        budget = interactive_view(report, completed_request_ids=ids[:2], budget_ms=1, elapsed_ms=5)
        self.assertEqual(budget["status"], "budget_exceeded_cached_summary")

    def test_graceful_stop_finishes_current_batch(self) -> None:
        guard = GracefulStop()
        completed = []

        def commit(values):
            completed.extend(values)

        result = guard.run_batches(
            ["B1", "B2", "B3"],
            lambda batch_id: guard.request_stop() or batch_id,
            commit,
        )
        self.assertEqual(result["completed_ids"], ["B1"])
        self.assertEqual(completed, ["B1"])
        self.assertTrue(result["graceful_stop"])


class ArbitraryCommandSequenceTests(unittest.TestCase):
    def test_deterministic_random_sequences_match_blank_full(self) -> None:
        for sequence_index in range(6):
            rng = random.Random(9100 + sequence_index)
            state = make_sample_state()
            engine = LedgerEngine(state)
            engine.run(parallel=False)
            add_counter = 0

            for step in range(10):
                kind = rng.choice(
                    [
                        "batch_size",
                        "merge_rule",
                        "switch",
                        "replace",
                        "add_batch",
                        "invalid_batch_size",
                    ]
                )
                if kind == "batch_size":
                    command = {"type": "set_batch_size", "batch_size": rng.choice([1, 2, 3, 4, 6])}
                elif kind == "merge_rule":
                    command = {
                        "type": "set_merge_rule",
                        "merge_rule": rng.choice(
                            ["coverage_then_version_v1", "version_then_batch_v1"]
                        ),
                    }
                elif kind == "switch":
                    command = {
                        "type": "switch_param_version",
                        "worker_id": rng.choice(["W1", "W2", "W3"]),
                        "param_version": rng.randint(40, 99),
                    }
                elif kind == "replace":
                    old = next(
                        item
                        for item in state["receipts"]
                        if item.get("batch_id") == "B-0007"
                        and item.get("request_id") == "Q0025"
                    )
                    replacement = {
                        key: value
                        for key, value in old.items()
                        if not key.startswith("_")
                    }
                    replacement["elapsed_ms"] = rng.randint(1, 120)
                    command = {
                        "type": "replace_receipt",
                        "match": {"batch_id": "B-0007", "request_id": "Q0025"},
                        "receipt": replacement,
                    }
                elif kind == "invalid_batch_size":
                    command = {"type": "set_batch_size", "batch_size": 0}
                else:
                    add_counter += 1
                    batch_id = f"B-RAND-{sequence_index}-{add_counter}"
                    base_seq = 500 + sequence_index * 100 + add_counter * 10
                    requests = [
                        {
                            "seq": base_seq + offset,
                            "request_id": f"QR{sequence_index}{add_counter}{offset}",
                            "arrival_ms": 5000 + base_seq + offset * 3,
                        }
                        for offset in range(2)
                    ]
                    receipts = [
                        {
                            "v": "r3",
                            "batch_id": batch_id,
                            "request_id": item["request_id"],
                            "output_digest": f"sha256:random-{sequence_index}-{add_counter}-{offset}",
                            "param_version": 77,
                            "elapsed_ms": 8 + offset,
                            "worker_id": "W1",
                            "rank_in_batch": offset,
                            "supersedes_ref": MISSING,
                            "param_digest": "param-random",
                        }
                        for offset, item in enumerate(requests)
                    ]
                    command = {
                        "type": "add_batch",
                        "batch": {
                            "batch_id": batch_id,
                            "requests": requests,
                            "receipts": receipts,
                        },
                    }

                engine.apply(command, parallel=True)
                full = build_report(clone_state(engine.state), parallel=False)
                self.assertEqual(engine.current_report, full)

            parallel = build_report(clone_state(engine.state), parallel=True)
            serial = build_report(clone_state(engine.state), parallel=False)
            self.assertEqual(parallel, serial)


class WorkerSubprocessTests(unittest.TestCase):
    def test_loopback_workers_return_deterministic_digests(self) -> None:
        processes = []
        try:
            w1, port1 = launch_worker_subprocess("test-W1", preferred=29301, project_root=PROJECT_ROOT)
            processes.append(w1)
            w2, port2 = launch_worker_subprocess("test-W2", preferred=29302, project_root=PROJECT_ROOT)
            processes.append(w2)
            self.assertNotEqual(port1, port2)
            for port in (port1, port2):
                self.assertTrue(20000 <= port <= 39999)
            response = infer_over_loopback(
                port1,
                {"requests": [{"request_id": "Q0001", "request_number": 1}]},
            )
            self.assertTrue(response["ok"])
            expected = deterministic_forward(
                seed=FIXED_SEED, arrival_seq=1, model_dim=16, hidden_dim=32
            ).digest
            self.assertEqual(response["output_digests"]["Q0001"], expected)
            response2 = infer_over_loopback(
                port2,
                {"requests": [{"request_id": "Q0005", "request_number": 5}]},
            )
            self.assertTrue(response2["ok"])
        finally:
            for process in processes:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except Exception:
                    process.kill()
                finally:
                    if process.stdout is not None:
                        process.stdout.close()
                    if process.stderr is not None:
                        process.stderr.close()


class TkUiWiringTests(unittest.TestCase):
    def test_launch_ui_renders_authoritative_rows_with_stub(self) -> None:
        from unittest.mock import MagicMock

        saved_tk = sys.modules.get("tkinter")
        saved_ttk = sys.modules.get("tkinter.ttk")
        fake_tk = MagicMock(name="tkinter")
        fake_ttk = MagicMock(name="tkinter.ttk")
        fake_tk.ttk = fake_ttk
        tree = fake_ttk.Treeview.return_value
        tree.get_children.return_value = []
        sys.modules["tkinter"] = fake_tk
        sys.modules["tkinter.ttk"] = fake_ttk
        try:
            with tempfile.TemporaryDirectory() as tmp:
                report_path = Path(tmp) / "report.json"
                report_path.write_text(
                    json.dumps(build_report(final_sample_state())), encoding="utf-8"
                )
                from batch_ledger.ui import launch_ui

                result = launch_ui(str(report_path))
            self.assertEqual(result, 0)
            self.assertGreater(tree.insert.call_count, 0)
            fake_tk.Tk.return_value.mainloop.assert_called_once()
        finally:
            if saved_tk is None:
                sys.modules.pop("tkinter", None)
            else:
                sys.modules["tkinter"] = saved_tk
            if saved_ttk is None:
                sys.modules.pop("tkinter.ttk", None)
            else:
                sys.modules["tkinter.ttk"] = saved_ttk


if __name__ == "__main__":
    unittest.main()
