from __future__ import annotations

import json
import random
import tempfile
import time
import unittest
from pathlib import Path

from dense_index.canonical import canonical_dumps
from dense_index.commands import CommandState, diff_objects, load_all_receipts
from dense_index.concurrency import ConcurrencyMonitor, run_ordered
from dense_index.config import validate_config
from dense_index.diagnostics import run_fault_injections
from dense_index.engine import IndexEngine
from dense_index.quant import bucket_key_from_vector, diagnose_pair, quantize_vector, routing_tie_flip
from dense_index.receipts import build_unified_ledger, convert_with_sidecar, load_receipt_file, restore_from_sidecar
from dense_index.replay import compare_command_sequence, run_stream_equivalence
from dense_index.sample import base_config, generate_sample
from dense_index.snapshot import SnapshotStore


class PipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp = tempfile.TemporaryDirectory(prefix="dense-index-test-")
        cls.root = Path(cls.temp.name) / "sample"
        generate_sample(cls.root)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temp.cleanup()

    def test_01_valid_sample_loads(self) -> None:
        state = CommandState.from_sample_dir(self.root)
        self.assertGreaterEqual(len(state.records), 24)
        self.assertGreater(len(state.blocks), 40)
        receipts = load_all_receipts(self.root)
        self.assertEqual(len(receipts), 12)

    def test_02_invalid_input_codes(self) -> None:
        from dense_index.corpus import load_corpus_file, load_vocabulary
        from dense_index.errors import DiagnosticError

        vocab = load_vocabulary(self.root / "config" / "vocabulary.json")
        with self.assertRaises(DiagnosticError) as ctx:
            load_corpus_file(self.root / "fixtures" / "corpus_invalid.jsonl", vocab)
        self.assertEqual(ctx.exception.code, "E_CORPUS_INVALID")
        self.assertGreaterEqual(len(ctx.exception.failures), 4)

        with self.assertRaises(DiagnosticError) as ctx:
            validate_config({"chunk_window": 4, "chunk_overlap": 4})
        self.assertEqual(ctx.exception.code, "E_CONFIG_INVALID")

        with self.assertRaises(DiagnosticError) as ctx:
            load_receipt_file(self.root / "fixtures" / "receipt_invalid_v1.json", forced_version="v1")
        self.assertEqual(ctx.exception.code, "E_RECEIPT_INVALID")

    def test_03_chunk_projection_deterministic(self) -> None:
        state = CommandState.from_sample_dir(self.root)
        engine_a = IndexEngine(state.config)
        engine_b = IndexEngine(state.config)
        view_a = engine_a.run_batch(state.blocks, budget=10**9)["view"]
        view_b = engine_b.run_batch(state.blocks, budget=10**9)["view"]
        self.assertEqual(canonical_dumps(view_a), canonical_dumps(view_b))
        block = view_a["blocks"][0]
        self.assertEqual(len(block["raw_vector"]), state.config.data["projection_dim"])
        self.assertEqual(len(block["fixed_vector"]), state.config.data["projection_dim"])

    def test_04_quant_level_changes_vectors_and_buckets(self) -> None:
        state_q16 = CommandState.from_sample_dir(self.root)
        report_q16 = IndexEngine(state_q16.config).run_batch(state_q16.blocks, budget=10**9)

        q8_config = validate_config({**state_q16.config.data, "quant_level": "Q8"})
        state_q8 = CommandState(config=q8_config, records=state_q16.records, vocabulary=state_q16.vocabulary)
        report_q8 = IndexEngine(q8_config).run_batch(state_q8.blocks, budget=10**9)

        q16_by_id = {item["block_id"]: item for item in report_q16["records"]}
        fixed_changes = 0
        digest_changes = 0
        bucket_changes = 0
        for q8_block in report_q8["records"]:
            q16_block = q16_by_id[q8_block["block_id"]]
            fixed_changes += q16_block["fixed_vector"] != q8_block["fixed_vector"]
            digest_changes += q16_block["vector_digest"] != q8_block["vector_digest"]
            bucket_changes += q16_block["bucket_id"] != q8_block["bucket_id"]
        self.assertGreater(fixed_changes, 50)
        self.assertGreater(digest_changes, 50)
        self.assertGreaterEqual(bucket_changes, 1)

    def test_05_receipt_roundtrip_and_authority(self) -> None:
        entries = load_receipt_file(self.root / "receipts" / "receipts_v3.json", forced_version="v3")
        original = entries[0]
        strict_v1, sidecar = convert_with_sidecar(original, "v1")
        self.assertEqual(set(strict_v1), {"bucket_id", "hit_rate_permille"})
        restored = restore_from_sidecar(strict_v1, sidecar)
        self.assertEqual(canonical_dumps(restored), canonical_dumps(original))
        self.assertTrue(sidecar["degradations"][0]["dropped_fields"])

        state = CommandState.from_sample_dir(self.root)
        report = IndexEngine(state.config, receipts=load_all_receipts(self.root)).run_batch(state.blocks, budget=10**9)
        authoritative = [item for item in report["ledger"] if item["authoritative"]]
        self.assertTrue(any(item["authority_reason"] == "v3_content_digest_match" for item in authoritative))

    def test_06_full_command_stream_equivalence(self) -> None:
        report = run_stream_equivalence(self.root, self.root / "outputs-test")
        self.assertTrue(report["all_fields_match"], json.dumps([s for s in report["steps"] if not s["match"]], indent=2))
        self.assertTrue(report["budget_pending_seen"])
        by_id = {step["command_id"]: step for step in report["steps"]}
        self.assertEqual(by_id["cmd-007-projection-dim"]["invalidation_summary"]["reprojected_blocks"], 0)
        self.assertGreaterEqual(by_id["cmd-009-quant-level"]["invalidation_summary"]["bucket_key_changed_blocks"], 1)
        self.assertEqual(by_id["cmd-003-replace-block"]["invalidation_summary"]["reprojected_blocks"], 1)

    def test_07_budget_pending_and_snapshot_resume(self) -> None:
        state = CommandState.from_sample_dir(self.root)
        store = SnapshotStore(self.root / "outputs-test" / "budget-snapshots")
        engine = IndexEngine(state.config, snapshot_store=store)
        first = engine.run_batch(state.blocks, budget=2)
        self.assertEqual(first["status"], "E_POINT_BUDGET_PENDING")
        canonical = sorted(state.blocks, key=lambda item: (item["document_id"], item["block_seq"]))
        self.assertEqual(first["scheduled_ids"], [item["block_id"] for item in canonical[:2]])
        self.assertEqual(first["pending_ids"], [item["block_id"] for item in canonical[2:]])

        recovered = IndexEngine.recover(state.config, store)
        self.assertTrue(recovered.recovery["recovered"])
        final = recovered.run_batch(state.blocks, budget=10**9)
        fresh = IndexEngine(state.config).run_batch(state.blocks, budget=10**9)
        self.assertEqual(diff_objects(fresh["view"], final["view"]), [])

    def test_08_concurrency_order_and_cap(self) -> None:
        monitor = ConcurrencyMonitor()

        def slow(value: int) -> int:
            time.sleep(0.01)
            return value * value

        results, errors = run_ordered(list(range(12)), slow, max_workers=3, monitor=monitor)
        self.assertIsNone(next((error for error in errors if error), None))
        self.assertEqual(results, [value * value for value in range(12)])
        self.assertLessEqual(monitor.observed_max, 3)
        self.assertGreaterEqual(monitor.observed_max, 2)

        state = CommandState.from_sample_dir(self.root)
        engine = IndexEngine(validate_config({**state.config.data, "max_concurrency": 2}))
        report = engine.run_batch(state.blocks[:8], budget=10**9)
        self.assertLessEqual(engine.monitor.observed_max, 2)
        self.assertEqual(report["bucket_write_fragments"], 8)

    def test_09_fault_injections_all_hit(self) -> None:
        report = run_fault_injections(self.root)
        self.assertTrue(report["all_hit"], json.dumps(report["items"], indent=2, default=str))

    def test_10_boundary_diagnostics_include_reference(self) -> None:
        config = validate_config({**base_config(), "quant_level": "Q8", "fractional_bits": 4, "bucket_bits": 2})
        width = int(config.data["projection_dim"])
        query_raw = [1.0] + [0.0] * (width - 1)
        block_raw = [1.0, 0.49] + [0.0] * (width - 2)
        query_quant = quantize_vector(query_raw, config)
        block_quant = quantize_vector(block_raw, config)
        diagnosis = diagnose_pair(query_raw, query_quant["fixed_vector"], block_raw, block_quant["fixed_vector"], "B1", config, 0.9)
        self.assertIn("reference_score", diagnosis)
        self.assertIn("reference_difference", diagnosis)
        self.assertEqual(diagnosis["code"], "E_BOUNDARY_FLIP")
        self.assertEqual(diagnosis["cause"], "quantization_truncation")

        tie_config = validate_config({**base_config(), "bucket_bits": 2})
        zero = [0.0] * width
        positive = [1.0, 1.0] + [0.0] * (width - 2)
        qq = quantize_vector(zero, tie_config)
        bq = quantize_vector(positive, tie_config)
        current_q = bucket_key_from_vector(zero, qq["fixed_vector"], tie_config)
        current_b = bucket_key_from_vector(positive, bq["fixed_vector"], tie_config)
        alt_q = bucket_key_from_vector(zero, qq["fixed_vector"], tie_config, alternative_tie_rule="negative")
        alt_b = bucket_key_from_vector(positive, bq["fixed_vector"], tie_config, alternative_tie_rule="negative")
        routing = routing_tie_flip(current_q, current_b, alt_q, alt_b)
        self.assertEqual(routing["code"], "E_BOUNDARY_FLIP")
        self.assertEqual(routing["cause"], "tie_rule")

    def test_11_random_valid_sequences_match_full_build(self) -> None:
        rng = random.Random(424242)
        vocabulary = [term for term in CommandState.from_sample_dir(self.root).vocabulary if term != "<PAD>"]

        def terms(size=None):
            return [rng.choice(vocabulary) for _ in range(size or rng.randint(5, 13))]

        for sequence_index in range(2):
            probe_state = CommandState.from_sample_dir(self.root)
            commands = []
            for step in range(5):
                op = rng.choice(["append", "paragraph", "block", "window", "dim", "quant", "threshold", "receipt"])
                command = {"id": f"random-{sequence_index}-{step}", "op": op}
                if op == "append":
                    command.update(
                        op="append_batch",
                        paragraphs=[{
                            "paragraph_id": f"R{sequence_index}-{step}-{rng.randrange(10**6)}",
                            "document_id": f"RND{sequence_index}",
                            "sequence_no": step,
                            "terms": terms(),
                        }],
                    )
                elif op == "paragraph":
                    record = rng.choice(probe_state.records)
                    command.update(op="replace_paragraph", target_id=record["paragraph_id"], paragraph={**record, "terms": terms()})
                elif op == "block":
                    block = rng.choice(probe_state.blocks)
                    command.update(op="replace_block", target_id=block["block_id"], terms=terms(probe_state.config.data["chunk_window"]))
                elif op == "window":
                    window = rng.randint(6, 14)
                    command.update(op="set_config", values={"chunk_window": window, "chunk_overlap": rng.randint(1, window - 1)})
                elif op == "dim":
                    command.update(op="set_config", values={"projection_dim": rng.choice([8, 12, 16, 24])})
                elif op == "quant":
                    command.update(op="set_config", values={"quant_level": rng.choice(["Q8", "Q16", "Q24"])})
                elif op == "threshold":
                    command.update(op="set_threshold", threshold=round(rng.uniform(0.05, 0.95), 3))
                else:
                    command.update(op="switch_receipt_version", target_version=rng.choice(["v1", "v2", "v3"]))
                probe_state.apply(command)
                commands.append(command)
            result = compare_command_sequence(self.root, commands)
            self.assertTrue(result["all_fields_match"], json.dumps(result["differences"], default=str)[:4000])


if __name__ == "__main__":
    unittest.main()
