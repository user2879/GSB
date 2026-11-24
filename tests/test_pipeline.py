from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path

from mmledger.pipeline import run_pipeline
from mmledger.supervisor import infer_batch, start_workers, stop_workers

from tests.helpers import generated_workspace, read_json, write_json


def ledger_payload(result) -> str:
    return json.dumps(result.ledger, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def mutate_receipt_duration(root: Path) -> None:
    path = root / "samples" / "receipts" / "r2_batch_002.json"
    payload = read_json(path)
    for record in payload["receipts"]:
        record["duration_ms"] = 17
    write_json(path, payload)


def mutate_add_batch(root: Path) -> None:
    samples = root / "samples"
    requests_path = samples / "requests.json"
    requests = read_json(requests_path)
    requests["requests"].append(
        {
            "sequence": 14,
            "request_id": "req-013",
            "kind": "image_text",
            "arrival_ms": 98,
            "batch_id": "batch-006",
            "worker_id": "w-alpha",
            "input_seed": 1013,
        }
    )
    write_json(requests_path, requests)
    write_json(
        samples / "receipts" / "r3_batch_006.json",
        {
            "version": "r3",
            "receipts": [
                {
                    "batch_id": "batch-006",
                    "request_id": "req-013",
                    "output_digest": "added-batch-digest",
                    "parameter_version": 3,
                    "duration_ms": 7,
                    "worker_id": "w-alpha",
                    "stable_rank": 1,
                    "overwritten_ref": None,
                    "parameter_digest": "sha256:pv3-alpha",
                }
            ],
        },
    )


def mutate_parameter_version(root: Path) -> None:
    samples = root / "samples"
    config_path = samples / "config.json"
    config = read_json(config_path)
    config["workers"][0]["parameter_version"] = 5
    config["workers"][0]["parameter_digest"] = "sha256:pv5-alpha"
    config["parameters"].append({"parameter_version": 5, "digest": "sha256:pv5-alpha"})
    write_json(config_path, config)
    rules_path = samples / "merge_rules.json"
    rules = read_json(rules_path)
    rules["parameter_version_sets"]["production"] = [3, 4, 5]
    write_json(rules_path, rules)


def mutate_batch_size(root: Path) -> None:
    path = root / "samples" / "config.json"
    config = read_json(path)
    config["limits"]["batch_size"] = 4
    write_json(path, config)


def mutate_merge_rules(root: Path) -> None:
    path = root / "samples" / "merge_rules.json"
    rules = read_json(path)
    rules["worker_priority"] = ["w-beta", "w-alpha"]
    rules["worker_sets"]["audit"] = ["w-beta"]
    write_json(path, rules)


class PipelineTests(unittest.TestCase):
    def test_concurrent_serial_and_incremental_ledgers_are_identical(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = generated_workspace(Path(temp))
            config = root / "samples" / "config.json"
            serial = run_pipeline(config, jobs=1, full_rebuild=True)
            serial_payload = ledger_payload(serial)
            concurrent = run_pipeline(config, jobs=8, full_rebuild=True)
            self.assertEqual(serial_payload, ledger_payload(concurrent))
            incremental = run_pipeline(config, jobs=4)
            self.assertEqual(serial_payload, ledger_payload(incremental))
            self.assertEqual(incremental.ledger["ledger_digest"], serial.ledger["ledger_digest"])
            self.assertTrue(
                all(
                    not result
                    for result in [
                        incremental.report["actual_invalidation"]["normalization_computed_batches"],
                        incremental.report["actual_invalidation"]["resource_computed_batches"],
                    ]
                )
            )

    def test_fault_drill_isolates_all_four_injections(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = generated_workspace(Path(temp))
            result = run_pipeline(root / "samples" / "config.json", jobs=4, fault_drill=True, commit_snapshot=False)
            hits = {item["injection_id"]: item for item in result.report["injection_hits"]}
            self.assertEqual(len(hits), 4)
            self.assertTrue(all(item["hit"] for item in hits.values()))
            self.assertEqual(hits["INJ-TRUNCATED"]["code"], "E_TRUNCATED_RECEIPT")
            self.assertEqual(hits["INJ-UNKNOWN-PARAM"]["code"], "E_UNKNOWN_PARAM_VERSION")
            self.assertEqual(hits["INJ-NONFINITE-DURATION"]["code"], "E_DURATION_NONFINITE")
            self.assertEqual(hits["INJ-DUPLICATE-BATCH"]["code"], "E_DUPLICATE_BATCH_ID")
            # Valid receipts continue to be normalized despite four bad inputs.
            self.assertGreater(result.ledger["counts"]["authoritative_requests"], 0)
            self.assertEqual(len(result.ledger["failures"]), 4)

    def test_mutation_sequence_incremental_equals_final_full_rebuild(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = generated_workspace(Path(temp))
            config = root / "samples" / "config.json"
            run_pipeline(config, jobs=2, full_rebuild=True)
            mutations = [
                mutate_receipt_duration,
                mutate_add_batch,
                mutate_parameter_version,
                mutate_batch_size,
                mutate_merge_rules,
            ]
            for mutate in mutations:
                mutate(root)
                run_pipeline(config, jobs=4)
            incremental = run_pipeline(config, jobs=4)

            full_root = Path(temp) / "full-final-input"
            (full_root / "snapshots").mkdir(parents=True)
            (full_root / "reports").mkdir(parents=True)
            shutil.copytree(root / "samples", full_root / "samples")
            full = run_pipeline(full_root / "samples" / "config.json", jobs=1, full_rebuild=True)
            self.assertEqual(ledger_payload(full), ledger_payload(incremental))
            self.assertEqual(full.ledger["ledger_digest"], incremental.ledger["ledger_digest"])

    def test_snapshot_uses_generation_directory_and_atomic_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = generated_workspace(Path(temp))
            result = run_pipeline(root / "samples" / "config.json", jobs=2)
            self.assertEqual(result.snapshot_version, 1)
            pointer = read_json(root / "snapshots" / "CURRENT.json")
            self.assertEqual(pointer["generation"], "g00000001")
            generation = root / "snapshots" / "generations" / "g00000001"
            for stage in ("parse", "ownership", "resources", "ledger"):
                self.assertTrue((generation / stage / "payload.json").exists())
            manifest = read_json(generation / "manifest.json")
            self.assertEqual(manifest["completed_stages"], ["parse", "ownership", "resources", "ledger"])
            self.assertEqual(manifest["consumed_budget"], 13)
            self.assertIn("w-alpha", manifest["completed_workers"])

    def test_real_loopback_worker_subprocesses_return_deterministic_receipts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = generated_workspace(Path(temp))
            config_path = root / "samples" / "config.json"
            bundle, processes = start_workers(config_path)
            try:
                requests = [request for request in bundle.requests if request.batch_id == "batch-001"]
                first = infer_batch(bundle.workers_by_id["w-alpha"], "batch-001", requests)
                second = infer_batch(bundle.workers_by_id["w-alpha"], "batch-001", requests)
            finally:
                stop_workers(processes)
            self.assertEqual(first["worker_id"], "w-alpha")
            self.assertEqual(first["requests"], second["requests"])
            self.assertEqual(len(first["requests"]), 3)
            self.assertGreater(first["duration_ms"], 0)
            for item in first["requests"]:
                self.assertEqual(len(item["output_digest"]), 64)
                self.assertIsInstance(item["fixed_score"], int)
                self.assertIsInstance(item["reference_score_ppm"], int)


if __name__ == "__main__":
    unittest.main()
