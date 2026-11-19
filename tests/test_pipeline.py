from __future__ import annotations

import base64
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from panorama_pipeline.codec import (
    V1_HEADER,
    V1_MAGIC,
    V1_RECORD,
    V1_RUN,
    V1_VERSION,
    load_metadata_document,
    parse_v1_package,
    parse_v2_package,
)
from panorama_pipeline.config import PipelineConfig
from panorama_pipeline.conversion import v1_to_v2, v2_to_v1
from panorama_pipeline.errors import (
    E_BOUNDARY_SENSITIVE,
    E_INTERACT_BUDGET_PENDING,
    E_MASK_FORMAT_INVALID,
    E_SNAPSHOT_INVALID,
    SnapshotRecoveryError,
)
from panorama_pipeline.interactive import InteractiveEngine
from panorama_pipeline.io_utils import read_json
from panorama_pipeline.pipeline import SampleLayout, build_session
from panorama_pipeline.snapshot import SnapshotStore
from tools.generate_samples import generate


class PipelineFixture(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary_root = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary_root.name)
        cls.sample = cls.root / "samples"
        generate(cls.sample)
        cls.layout = SampleLayout.discover(cls.sample)
        cls.config = PipelineConfig.load(cls.layout.config_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary_root.cleanup()

    def clone_sample(self, name: str) -> Path:
        destination = self.root / name
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(self.sample, destination)
        return destination


class ParsingTests(PipelineFixture):
    def test_fixed_seed_sample_is_reproducible(self) -> None:
        second = self.root / "samples_reproduced"
        if second.exists():
            shutil.rmtree(second)
        generate(second)
        first_manifest = read_json(self.sample / "sample_manifest.json")
        second_manifest = read_json(second / "sample_manifest.json")
        self.assertEqual(first_manifest, second_manifest)

    def test_parse_both_formats_and_boundary_diagnostic(self) -> None:
        session = build_session(self.sample)
        self.assertEqual(len(session.instances), 16)
        self.assertFalse(session.failures)
        self.assertEqual((session.width, session.height), (64, 48))
        self.assertEqual(len(session.fusion_result.boundary_sensitive), 1)
        self.assertEqual(
            session.fusion_result.boundary_sensitive[0]["pair_key"],
            "v1:2|v2:2",
        )
        self.assertIn(E_BOUNDARY_SENSITIVE, session.fusion_result.failure_codes)

    def test_v1_overlapping_runs_are_rejected(self) -> None:
        destination = self.clone_sample("bad_v1")
        payload = b"".join(
            [
                V1_HEADER.pack(V1_MAGIC, V1_VERSION, 0, 8, 4, 1),
                V1_RECORD.pack(1, 1, 2),
                V1_RUN.pack(0, 5),
                V1_RUN.pack(3, 2),
            ]
        )
        (destination / "masks_v1.bin").write_bytes(payload)
        bundle = parse_v1_package(
            destination / "masks_v1.bin",
            config=self.config,
            metadata=load_metadata_document(destination / "instances.json"),
        )
        self.assertEqual(len(bundle.instances), 0)
        self.assertEqual(bundle.failures[0].code, E_MASK_FORMAT_INVALID)
        self.assertIn("overlaps", bundle.failures[0].message)

    def test_v1_out_of_bounds_run_is_rejected(self) -> None:
        destination = self.clone_sample("bad_v1_bounds")
        payload = b"".join(
            [
                V1_HEADER.pack(V1_MAGIC, V1_VERSION, 0, 8, 4, 1),
                V1_RECORD.pack(1, 1, 1),
                V1_RUN.pack(30, 5),
            ]
        )
        (destination / "masks_v1.bin").write_bytes(payload)
        bundle = parse_v1_package(
            destination / "masks_v1.bin",
            config=self.config,
            metadata=load_metadata_document(destination / "instances.json"),
        )
        self.assertEqual(bundle.failures[0].code, E_MASK_FORMAT_INVALID)
        self.assertIn("bounds", bundle.failures[0].message)

    def test_v2_bad_base64_length_and_missing_field(self) -> None:
        destination = self.clone_sample("bad_v2")
        path = destination / "masks_v2.json"
        payload = read_json(path)
        payload["instances"][0]["mask_base64"] = base64.b64encode(b"\x01" * 3).decode()
        del payload["instances"][1]["confidence"]
        path.write_text(json.dumps(payload), encoding="utf-8")
        bundle = parse_v2_package(path, config=self.config)
        failure_keys = [failure.instance_key for failure in bundle.failures]
        self.assertEqual(failure_keys, ["v2:1", "v2:2"])
        self.assertTrue(
            all(failure.code == E_MASK_FORMAT_INVALID for failure in bundle.failures)
        )
        self.assertIn("expected", bundle.failures[0].message)
        self.assertIn("missing fields", bundle.failures[1].message)

    def test_parallel_failure_order_matches_serial(self) -> None:
        destination = self.clone_sample("parallel_bad_v2")
        path = destination / "masks_v2.json"
        payload = read_json(path)
        by_id = {entry["instance_id"]: entry for entry in payload["instances"]}
        by_id[1]["mask_base64"] = base64.b64encode(b"\x00" * 4).decode()
        by_id[2]["crowd"] = "false"
        path.write_text(json.dumps(payload), encoding="utf-8")
        parallel = parse_v2_package(path, config=self.config, max_workers=4)
        serial = parse_v2_package(path, config=self.config, max_workers=1)
        self.assertEqual(
            [item.key for item in parallel.instances],
            [item.key for item in serial.instances],
        )
        self.assertEqual(
            [failure.instance_key for failure in parallel.failures],
            [failure.instance_key for failure in serial.failures],
        )
        self.assertEqual(
            [failure.instance_key for failure in parallel.failures],
            ["v2:1", "v2:2"],
        )


class ConversionTests(PipelineFixture):
    def test_bidirectional_conversion_is_pixel_exact(self) -> None:
        session = build_session(self.sample)
        v1_bundle = parse_v1_package(
            self.layout.v1_path,
            config=self.config,
            metadata=session.metadata_document,
        )
        v2_bundle = parse_v2_package(self.layout.v2_path, config=self.config)
        output_dir = self.root / "conversion"
        output_dir.mkdir(exist_ok=True)
        up = v1_to_v2(v1_bundle, output_dir / "up.json", config=self.config)
        down = v2_to_v1(
            v2_bundle,
            output_dir / "down.bin",
            output_dir / "down_sidecar.json",
            config=self.config,
        )
        self.assertTrue(up["comparison"]["pixel_equal"])
        self.assertTrue(down["comparison"]["pixel_equal"])
        defaulted_keys = {
            entry["instance_key"] for entry in up["defaulted_instances"]
        }
        self.assertIn("v1:7", defaulted_keys)
        self.assertIn("v1:8", defaulted_keys)
        self.assertEqual(len(down["degradation_losses"]), len(v2_bundle.instances))
        self.assertTrue(all(loss["lost_fields"] for loss in down["degradation_losses"]))


class IncrementalTests(PipelineFixture):
    def test_three_trigger_classes_and_decode_reuse(self) -> None:
        session = build_session(self.sample)
        initial_counts = dict(session.decode_counts)
        metadata_preview = session.execute(
            {
                "type": "replace_metadata",
                "instance_key": "v1:3",
                "confidence": 0.99,
                "crowd": False,
            },
            command_seq=1,
        )
        event = session.events[-1]
        self.assertEqual(event["trigger"], "metadata_replacement")
        self.assertEqual(event["invalidated_instances"], ["v1:3"])
        self.assertNotIn("v2:3", event["invalidated_instances"])
        self.assertEqual(session.decode_counts["v1:3"], 2)
        self.assertEqual(session.decode_counts["v1:4"], initial_counts["v1:4"])
        self.assertTrue(metadata_preview["is_latest"])

        counts_before_retention = dict(session.decode_counts)
        session.execute(
            {
                "type": "set_params",
                "retention_mode": "confidence",
                "min_coverage": 0.4,
            },
            command_seq=2,
        )
        self.assertEqual(session.events[-1]["trigger"], "retention_policy_change")
        self.assertEqual(session.decode_counts, counts_before_retention)

        session.execute({"type": "set_formats", "formats": ["v1"]}, command_seq=3)
        removed_event = session.events[-1]
        self.assertEqual(removed_event["trigger"], "format_version_switch")
        self.assertTrue(
            all(key.startswith("v2:") for key in removed_event["invalidated_instances"])
        )
        self.assertEqual(session.decode_counts["v1:3"], 2)
        session.execute(
            {"type": "set_formats", "formats": ["v1", "v2"]},
            command_seq=4,
        )
        self.assertEqual(session.events[-1]["trigger"], "format_version_switch")
        self.assertEqual(session.decode_counts["v1:3"], 2)
        self.assertEqual(session.decode_counts["v2:3"], 2)

    def test_final_incremental_matches_full_rebuild(self) -> None:
        session = build_session(self.sample)
        session.execute(
            {
                "type": "set_params",
                "iou_match_threshold": 0.4,
                "retention_mode": "prefer_non_crowd",
                "min_coverage": 0.25,
            },
            command_seq=1,
        )
        session.execute(
            {
                "type": "set_params",
                "iou_match_threshold": 0.333333333,
                "retention_mode": "prefer_non_crowd",
                "min_coverage": 0.333333333,
            },
            command_seq=2,
        )
        session.execute(
            {
                "type": "replace_metadata",
                "instance_key": "v1:2",
                "confidence": 0.97,
                "crowd": False,
            },
            command_seq=3,
        )
        oracle = build_session(self.sample, config=session.config)
        oracle.execute(
            {
                "type": "replace_metadata",
                "instance_key": "v1:2",
                "confidence": 0.97,
                "crowd": False,
            },
            command_seq=1,
        )
        self.assertTrue(
            np.array_equal(
                session.fusion_result.category_grid,
                oracle.fusion_result.category_grid,
            )
        )
        self.assertEqual(
            {key: item.cache_key for key, item in session.instances.items()},
            {key: item.cache_key for key, item in oracle.instances.items()},
        )


class InteractionBudgetTests(PipelineFixture):
    def test_pending_response_never_claims_latest(self) -> None:
        session = build_session(self.sample)

        def slow_worker(work):
            time.sleep(0.08)
            return work()

        engine = InteractiveEngine(
            session,
            budget_ms=5,
            worker_wrapper=slow_worker,
        )
        try:
            started = time.perf_counter()
            response = engine.submit(
                {"type": "set_params", "iou_match_threshold": 0.4}
            )
            elapsed = time.perf_counter() - started
            self.assertLess(elapsed, 0.06)
            self.assertEqual(response["status"], "pending")
            self.assertFalse(response["is_latest"])
            self.assertEqual(response["business_code"], E_INTERACT_BUDGET_PENDING)
            self.assertEqual(response["preview_source_command_seq"], 0)
            self.assertFalse(response["preview"]["is_latest"])
            completed = engine.wait_command(response["command_seq"], timeout=2)
            self.assertEqual(completed["status"], "complete")
            self.assertTrue(completed["is_latest"])
            self.assertEqual(completed["business_code"], "OK")
        finally:
            engine.close()


class SnapshotTests(PipelineFixture):
    def test_commit_and_recover(self) -> None:
        session = build_session(self.sample)
        store = SnapshotStore(self.root / "snapshot_ok")
        first = store.commit(session, report_payload={"phase": "first"})
        recovered = store.recover()
        self.assertEqual(recovered.generation, first["generation"])
        self.assertTrue(
            np.array_equal(
                recovered.category_grid,
                session.fusion_result.category_grid,
            )
        )
        self.assertEqual(
            recovered.manifest["instance_result_versions"],
            {item.key: item.cache_key for item in session.sorted_instances},
        )
        session.execute(
            {"type": "set_params", "iou_match_threshold": 0.45},
            command_seq=1,
        )
        second = store.commit(session, report_payload={"phase": "second"})
        self.assertEqual(
            second["version_changes"]["previous_global_version"],
            first["global_version"],
        )
        self.assertEqual(second["global_version"], first["global_version"] + 1)

    def test_corrupted_published_instance_blocks_recovery(self) -> None:
        session = build_session(self.sample)
        store = SnapshotStore(self.root / "snapshot_corrupt")
        store.commit(session)
        generation = store.current_link.resolve()
        instance_file = generation / "instances/v1_1.npz"
        with instance_file.open("ab") as handle:
            handle.write(b"corruption")
        with self.assertRaises(SnapshotRecoveryError) as context:
            store.recover()
        self.assertEqual(context.exception.code, E_SNAPSHOT_INVALID)
        self.assertIn("hash mismatch", context.exception.message)

    def test_incomplete_temp_generation_is_ignored(self) -> None:
        session = build_session(self.sample)
        store = SnapshotStore(self.root / "snapshot_temp")
        manifest = store.commit(session)
        incomplete = store.root / ".tmp-generation-killed"
        (incomplete / "instances").mkdir(parents=True)
        (incomplete / "manifest.json").write_text(
            json.dumps({"committed": False, "schema_version": 1}),
            encoding="utf-8",
        )
        recovered = store.recover()
        self.assertEqual(recovered.generation, manifest["generation"])


if __name__ == "__main__":
    unittest.main()
