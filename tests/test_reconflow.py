from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reconflow.canonical import canonical_bytes  # noqa: E402
from reconflow.demo import run_demo, rpc_failure_self_checks, spec_self_check  # noqa: E402
from reconflow.errors import ReplayError, SpecError  # noqa: E402
from reconflow.incremental import (  # noqa: E402
    edit_shard_size,
    edit_threshold,
    edit_weight,
    initialize_fast_view,
    verify_fast_matches_authoritative,
)
from reconflow.ledger import Ledger  # noqa: E402
from reconflow.pipeline import compute_conclusion, conclusion_bytes, first_divergence  # noqa: E402
from reconflow.repository import open_initialized_ledger, persist_current_view, replay_ledger  # noqa: E402
from reconflow.scheduler import recompute_shards_tcp  # noqa: E402
from reconflow.specs import load_bundle, load_domain_records  # noqa: E402
from reconflow.synthetic import DEMO_DOMAINS, generate_batches, generate_workspace  # noqa: E402
from reconflow.workspace import Workspace  # noqa: E402


class SpecValidationTests(unittest.TestCase):
    def test_generated_bundle_is_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ws"
            generate_workspace(root)
            bundle = load_bundle(root)
            self.assertEqual(len(bundle.domain_ids), 4)
            self.assertEqual(sum(len(b["docs"]) for b in bundle.batches), 96)

    def test_negative_seed_reports_ndjson_line_and_field(self) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".ndjson", delete=False, encoding="utf-8") as handle:
            handle.write(json.dumps(DEMO_DOMAINS[0]) + "\n")
            bad = dict(DEMO_DOMAINS[1])
            bad["seed"] = -1
            handle.write(json.dumps(bad) + "\n")
            path = handle.name
        try:
            with self.assertRaises(SpecError) as ctx:
                load_domain_records(path)
            self.assertEqual(ctx.exception.code, "E_SPEC")
            self.assertEqual(ctx.exception.line, 2)
            self.assertEqual(ctx.exception.field, "seed")
        finally:
            Path(path).unlink(missing_ok=True)

    def test_bitmap_length_mismatch_reports_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_workspace(root)
            batches_path = root / "data" / "batches.ndjson"
            records = [json.loads(line) for line in batches_path.read_text().splitlines()]
            records[0]["docs"][0]["missing_bitmap"].pop()
            batches_path.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
            with self.assertRaises(SpecError) as ctx:
                load_bundle(root, verify_digests=False)
            self.assertEqual(ctx.exception.code, "E_SPEC")
            self.assertEqual(ctx.exception.field, "docs.missing_bitmap")

    def test_spec_self_check(self) -> None:
        result = spec_self_check()["E_SPEC"]
        self.assertTrue(result["raised"])
        self.assertEqual(result["line"], 2)


class DeterminismTests(unittest.TestCase):
    def test_domain_order_and_transport_rechunk_do_not_change_digests(self) -> None:
        forward = generate_batches(DEMO_DOMAINS)
        reverse = generate_batches(list(reversed(DEMO_DOMAINS)))
        self.assertEqual(
            canonical_bytes([b["batch_digest"] for b in forward]),
            canonical_bytes([b["batch_digest"] for b in reverse]),
        )
        flat_docs = [doc for batch in forward for doc in batch["docs"]]
        rechunked = [flat_docs[i : i + 3] for i in range(0, len(flat_docs), 3)]
        rechunked_digests = [doc["content_digest"] for chunk in rechunked for doc in chunk]
        original_digests = [doc["content_digest"] for doc in flat_docs]
        self.assertEqual(rechunked_digests, original_digests)

    def test_conclusion_bytes_are_stable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_workspace(root)
            bundle = load_bundle(root)
            first = canonical_bytes(compute_conclusion(bundle))
            second = canonical_bytes(compute_conclusion(load_bundle(root)))
            self.assertEqual(first, second)

    def test_reversed_domain_and_batch_order_remains_identical(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_workspace(root)
            expected = conclusion_bytes(compute_conclusion(load_bundle(root)))

            domains_path = root / "specs" / "domains.json"
            domains_payload = json.loads(domains_path.read_text(encoding="utf-8"))
            domains_payload["domains"].reverse()
            domains_path.write_text(json.dumps(domains_payload), encoding="utf-8")

            batches_path = root / "data" / "batches.ndjson"
            records = [json.loads(line) for line in batches_path.read_text().splitlines()]
            records.reverse()
            batches_path.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

            actual = conclusion_bytes(compute_conclusion(load_bundle(root)))
            self.assertEqual(expected, actual)

    def test_tcp_shard_results_match_local(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_workspace(root)
            bundle = load_bundle(root)
            conclusion = compute_conclusion(bundle)
            remote = recompute_shards_tcp(bundle)
            remote_digests = {s["shard_id"]: s["shard_digest"] for s in remote["shard_results"]}
            for shard in conclusion["shards"]:
                self.assertEqual(shard["shard_digest"], remote_digests[shard["shard_id"]])
            scheduling = remote["scheduling"]
            self.assertEqual(scheduling["invariant_violations"], [])
            self.assertLessEqual(scheduling["peak_global_points"], bundle.config["global_sampling_point_budget"])
            for domain, peak in scheduling["peak_domain_points"].items():
                self.assertLessEqual(peak, bundle.config["domain_sampling_point_quota"])


class InvalidationTests(unittest.TestCase):
    def test_weight_and_threshold_subsets_are_exact_and_byte_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_workspace(root)
            ws = Workspace(root)
            initialize_fast_view(ws)

            before = {s["shard_id"]: s["shard_digest"] for s in ws.load_fast_view()["conclusion"]["shards"]}
            propagation = edit_weight(ws, "alpha", 4000)
            after = {s["shard_id"]: s["shard_digest"] for s in ws.load_fast_view()["conclusion"]["shards"]}
            changed = sorted(sid for sid in set(before) | set(after) if before.get(sid) != after.get(sid))
            self.assertEqual(changed, propagation["affected_shards"])
            self.assertTrue(all(c["byte_identical"] for c in propagation["preservation_checks"] if c["ref"].startswith("dedup:") and c["ref"].split(":", 1)[1] in propagation["preserved_shards"]))
            self.assertTrue(verify_fast_matches_authoritative(ws)["identical"])

            before = after
            propagation = edit_threshold(ws, 8800)
            after = {s["shard_id"]: s["shard_digest"] for s in ws.load_fast_view()["conclusion"]["shards"]}
            changed = sorted(sid for sid in set(before) | set(after) if before.get(sid) != after.get(sid))
            self.assertEqual(changed, propagation["affected_shards"])
            self.assertTrue(verify_fast_matches_authoritative(ws)["identical"])

            before = after
            propagation = edit_shard_size(ws, 8)
            after = {s["shard_id"]: s["shard_digest"] for s in ws.load_fast_view()["conclusion"]["shards"]}
            changed = sorted(sid for sid in set(before) | set(after) if before.get(sid) != after.get(sid))
            expected_invalidated = sorted(set(propagation["affected_shards"]) | set(propagation["removed_shards"]))
            self.assertEqual(changed, expected_invalidated)
            self.assertTrue(verify_fast_matches_authoritative(ws)["identical"])

    def test_first_divergence_locates_ratio_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_workspace(root)
            bundle = load_bundle(root)
            expected = compute_conclusion(bundle)
            actual = json.loads(canonical_bytes(expected))
            actual["ratio_normalization"]["domains"][0]["deviation_bp"] += 7
            divergence = first_divergence(expected, actual)
            self.assertEqual(divergence["stage"], "ratio_normalization")
            self.assertEqual(divergence["field"], "deviation_bp")
            self.assertTrue(divergence["domain_id"])


class LedgerAndDemoTests(unittest.TestCase):
    def test_full_demo_kill_recovery_replay_and_rpc_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = run_demo(Path(tmp) / "demo")
            self.assertTrue(report["success"], json.dumps(report.get("error"), indent=2))
            self.assertEqual(report["forced_termination"]["returncode"], -9)
            self.assertGreater(report["recovery"]["unknown_count"], 0)
            self.assertTrue(report["replay"]["identical"])
            self.assertTrue(all(item["subset_exact"] for item in report["edits"]))
            self.assertTrue(report["rpc_error_codes_verified"])

    def test_replay_detects_tampered_ratio_field_with_batch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_workspace(root)
            ws = Workspace(root)
            initialize_fast_view(ws)
            ledger = open_initialized_ledger(ws)
            from reconflow.repository import process_pending_batches

            process_pending_batches(ws, ledger)
            persist_current_view(ws, ledger)
            with ledger.connect() as conn:
                row = conn.execute("SELECT canonical_json FROM stage_items WHERE item_key=?", ("ratio:alpha",)).fetchone()
                payload = json.loads(row[0])
                payload["deviation_bp"] = payload["deviation_bp"] + 123
                conn.execute(
                    "UPDATE stage_items SET canonical_json=? WHERE item_key=?",
                    (json.dumps(payload, sort_keys=True), "ratio:alpha"),
                )
            with self.assertRaises(ReplayError) as ctx:
                replay_ledger(ws, ledger)
            self.assertEqual(ctx.exception.code, "E_REPLAY")
            self.assertTrue(ctx.exception.details["batch_id"])
            self.assertEqual(ctx.exception.details["field"], "deviation_bp")


class RpcSelfCheckTests(unittest.TestCase):
    def test_rpc_failure_codes(self) -> None:
        result = rpc_failure_self_checks()
        self.assertEqual({key: value["code"] for key, value in result.items()}, {"E_PORT": "E_PORT", "E_TIMEOUT": "E_TIMEOUT", "E_FIELD": "E_FIELD"})
        self.assertTrue(all(value["raised"] for value in result.values()))


class TkSmokeTests(unittest.TestCase):
    def test_tk_diagnostic_app_smoke(self) -> None:
        try:
            import tkinter  # noqa: F401
        except Exception:
            self.skipTest("Tkinter libraries are unavailable")
        if not os.environ.get("DISPLAY"):
            self.skipTest("DISPLAY is unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            generate_workspace(root)
            ws = Workspace(root)
            initialize_fast_view(ws)
            from reconflow.ui import run_tk_app

            rc = run_tk_app(ws, smoke_ms=100)
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
