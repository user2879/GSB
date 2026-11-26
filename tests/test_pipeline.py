from __future__ import annotations

import copy
import json
import random
import tempfile
import unittest
from pathlib import Path

from audio_fingerprint.acceptance import make_tiny_inputs
from audio_fingerprint.buckets import build_buckets
from audio_fingerprint.config import TIME_GEAR_STEPS
from audio_fingerprint.demo import create_demo_workspace
from audio_fingerprint.engine import FingerprintEngine, deep_diff
from audio_fingerprint.fingerprints import fingerprint_key_for, quantize_time_delta
from audio_fingerprint.manifest import validate_manifest
from audio_fingerprint.retrieval import arbitrate_versions, rank_candidates, vote_candidate
from audio_fingerprint.store import PersistentStore
from audio_fingerprint.workspace import apply_command_to_inputs


def pair_stub(index: int, key: str, track: str, strength: int, version: int = 1) -> dict:
    return {
        "product_id": f"pair-{index}",
        "cache_key": f"pair-cache-{index}",
        "fingerprint_key": key,
        "clip_id": f"clip-{track}",
        "track_id": track,
        "version": version,
        "anchor_frame": index,
        "target_frame": index + 1,
        "strength_q": strength,
    }


class RuleTests(unittest.TestCase):
    def test_time_quant_gear_changes_key_and_records_direction(self) -> None:
        fine = quantize_time_delta(3, "tq_fine", TIME_GEAR_STEPS["tq_fine"])
        standard = quantize_time_delta(3, "tq_standard", TIME_GEAR_STEPS["tq_standard"])
        coarse = quantize_time_delta(3, "tq_coarse", TIME_GEAR_STEPS["tq_coarse"])
        self.assertEqual((fine["quantized_delta_frames"], fine["rounding_direction"]), (3, "exact"))
        self.assertEqual((standard["quantized_delta_frames"], standard["rounding_direction"]), (4, "up"))
        self.assertEqual((coarse["quantized_delta_frames"], coarse["rounding_direction"]), (4, "up"))
        key_fine = fingerprint_key_for(1, 4, fine["quantized_delta_frames"], "tq_fine")
        key_standard = fingerprint_key_for(1, 4, standard["quantized_delta_frames"], "tq_standard")
        self.assertNotEqual(key_fine, key_standard)

    def test_bucket_capacity_eviction_order(self) -> None:
        pairs = [
            pair_stub(0, "same", "trk-low", 10),
            pair_stub(1, "same", "trk-a", 90),
            pair_stub(2, "same", "trk-b", 90),
            pair_stub(3, "same", "trk-high", 120),
        ]
        bucket = build_buckets(pairs, {"bucket_capacity": 2})[0]
        retained_tracks = [entry["track_id"] for entry in bucket["entries"]]
        self.assertEqual(retained_tracks, ["trk-a", "trk-high"])
        self.assertEqual(len(bucket["evicted_entries"]), 2)
        self.assertTrue(all(item["eviction_reason"] == "bucket_capacity_exceeded" for item in bucket["evicted_entries"]))

    def test_candidate_tie_track_order_and_boundary_delta(self) -> None:
        hits = []
        for track in ("trk-z", "trk-a", "trk-mid"):
            hits.extend(
                [
                    {"track_id": track, "version": 1, "clip_id": track, "offset_frames": 1, "strength_q": 1}
                    for _ in range(3)
                ]
            )
        ranked = rank_candidates(hits, {"candidate_limit": 2})
        self.assertEqual([item["track_id"] for item in ranked], ["trk-a", "trk-mid", "trk-z"])
        self.assertTrue(ranked[1]["boundary_candidate"])
        self.assertEqual(ranked[1]["boundary_detail"]["hit_count_delta"], 0)
        self.assertTrue(ranked[1]["boundary_detail"]["at_cutoff_tie"])

    def test_vote_offset_tie_prefers_smaller_absolute_offset(self) -> None:
        candidate = {
            "hits": [
                {"offset_frames": -2, "strength_q": 10},
                {"offset_frames": 2, "strength_q": 10},
            ]
        }
        vote = vote_candidate(candidate, {"vote_fraction_bits": 8})
        self.assertEqual(vote["main_offset_frames"], -2)

    def test_version_conflict_records_covered_version(self) -> None:
        conclusions = [
            {
                "track_id": "trk-x",
                "version": 1,
                "clip_id": "x-v1",
                "main_votes_q": 100,
                "hit_count": 4,
                "main_hit_count": 2,
                "total_strength_q": 1000,
                "main_offset_frames": 2,
            },
            {
                "track_id": "trk-x",
                "version": 2,
                "clip_id": "x-v2",
                "main_votes_q": 120,
                "hit_count": 5,
                "main_hit_count": 5,
                "total_strength_q": 2000,
                "main_offset_frames": 9,
            },
        ]
        arbitration = arbitrate_versions(conclusions)[0]
        self.assertTrue(arbitration["offset_conflict"])
        self.assertEqual(arbitration["authoritative_version"], 2)
        self.assertEqual(arbitration["authoritative_offset_frames"], 9)
        self.assertEqual(arbitration["decision_basis"], "higher_aligned_hit_count")
        self.assertEqual(arbitration["covered_versions"][0]["version"], 1)


class EngineTests(unittest.TestCase):
    def test_demo_validation_codes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "ws"
            create_demo_workspace(root)
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            clips, issues = validate_manifest(manifest, root / "data")
            self.assertEqual(
                [issue["code"] for issue in issues],
                ["E1001", "E1002", "E1003", "E1004", "E1005", "E2002"],
            )
            self.assertEqual(len(clips), 9)

    def test_incremental_sequence_matches_blank_full(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, config, data_dir = make_tiny_inputs(root / "audio")
            engine = FingerprintEngine(copy.deepcopy(manifest), copy.deepcopy(config), data_dir, root / "incremental")
            engine.start_run(run_id="base")
            self.assertTrue(engine.run_until_complete()["completed"])

            new_clip = {
                "clip_id": "tiny-new",
                "track_id": "trk-tiny-new",
                "version": 1,
                "seed": 3404,
                "path": "clips/tiny-new.wav",
                "internal_sample_rate": 11025,
                "internal_sample_count": 4096,
                "source_sample_rate": 22050,
                "bit_depth": 16,
                "declared_sample_rates": [11025, 22050, 44100],
                "declared_bit_depths": [8, 16, 24, 32],
            }
            commands = [
                {"op": "set_peak_threshold", "value": 1400, "command_id": "c0"},
                {"op": "set_frame_threshold", "clip_id": "tiny-alpha", "frame_index": 7, "value": 18000, "command_id": "c1"},
                {"op": "set_bucket_capacity", "value": 3, "command_id": "c2"},
                {"op": "set_time_quant_gear", "value": "tq_coarse", "command_id": "c3"},
                {"op": "replace_clip", "clip_id": "tiny-beta", "seed": 9917, "command_id": "c4"},
                {"op": "set_clip_format", "clip_id": "tiny-gamma", "sample_rate": 44100, "bit_depth": 8, "command_id": "c5"},
                {"op": "add_clip", "clip": new_clip, "command_id": "c6"},
                {"op": "delete_clip", "clip_id": "tiny-gamma", "command_id": "c7"},
            ]
            for command in commands:
                issues = apply_command_to_inputs(engine.manifest, engine.config, command, data_dir)
                self.assertEqual(issues, [])
                engine._context = None
                engine.start_run(command)
                self.assertTrue(engine.run_until_complete()["completed"])

            incremental = engine.export_canonical()
            full = FingerprintEngine(
                copy.deepcopy(engine.manifest),
                copy.deepcopy(engine.config),
                data_dir,
                root / "full",
            )
            full.start_run(run_id="full")
            self.assertTrue(full.run_until_complete()["completed"])
            self.assertEqual(deep_diff(incremental, full.export_canonical()), [])

    def test_budget_exhaustion_and_concurrency_equivalence(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest, base_config, data_dir = make_tiny_inputs(root / "audio")
            unlimited_config = copy.deepcopy(base_config)
            unlimited = FingerprintEngine(copy.deepcopy(manifest), unlimited_config, data_dir, root / "unlimited")
            unlimited.start_run()
            unlimited.run_until_complete()
            expected = unlimited.canonical_fingerprint()

            for index, budget in enumerate((256, 768, 2048)):
                config = copy.deepcopy(base_config)
                config["batch_sample_budget"] = budget
                config["clip_sample_quota"] = 100000
                engine = FingerprintEngine(
                    copy.deepcopy(manifest),
                    config,
                    data_dir,
                    root / f"budget-{budget}",
                    serial=True,
                )
                engine.start_run()
                self.assertTrue(engine.run_until_complete()["completed"])
                self.assertEqual(engine.canonical_fingerprint(), expected, msg=f"budget {budget}")

            quota_config = copy.deepcopy(base_config)
            quota_config.update({"batch_sample_budget": 768, "clip_sample_quota": 512, "max_concurrency": 1})
            quota_engine = FingerprintEngine(copy.deepcopy(manifest), quota_config, data_dir, root / "quota", serial=True)
            quota_engine.start_run()
            first = quota_engine.run_next_batch()
            self.assertFalse(first["completed"])
            self.assertIn("B4001", {event["code"] for event in first["batch"]["events"]})
            self.assertIn("B4002", {event["code"] for event in first["batch"]["events"]})
            self.assertTrue(quota_engine.run_until_complete()["completed"])
            self.assertEqual(quota_engine.canonical_fingerprint(), expected)

            fingerprints = set()
            for mode, workers, serial in (("serial", 1, True), ("two", 2, False), ("four", 4, False)):
                config = copy.deepcopy(base_config)
                config["max_concurrency"] = workers
                config["batch_sample_budget"] = 768
                engine = FingerprintEngine(
                    copy.deepcopy(manifest),
                    config,
                    data_dir,
                    root / f"concurrency-{mode}",
                    serial=serial,
                )
                engine.start_run()
                engine.run_until_complete()
                fingerprints.add(engine.canonical_fingerprint())
            self.assertEqual(fingerprints, {expected})

    def test_deterministic_fuzz_command_sequences(self) -> None:
        rates = [11025, 22050, 44100]
        depths = [8, 16, 24, 32]
        gears = list(TIME_GEAR_STEPS)
        for case in range(3):
            rng = random.Random(91000 + case)
            with tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                manifest, config, data_dir = make_tiny_inputs(root / "audio")
                config["batch_sample_budget"] = (512, 1024, 200000)[case]
                engine = FingerprintEngine(copy.deepcopy(manifest), copy.deepcopy(config), data_dir, root / "inc")
                engine.start_run()
                engine.run_until_complete()
                library_ids = [clip["clip_id"] for clip in manifest["clips"] if clip["role"] == "library"]
                commands = []
                for index in range(8):
                    op = rng.choice(
                        [
                            "set_peak_threshold",
                            "set_bucket_capacity",
                            "set_time_quant_gear",
                            "replace_clip",
                            "set_clip_format",
                            "set_frame_threshold",
                        ]
                    )
                    command: dict = {"op": op, "command_id": f"fuzz-{case}-{index}"}
                    clip_id = rng.choice(library_ids)
                    if op == "set_peak_threshold":
                        command["value"] = rng.choice([300, 900, 1600, 4000])
                    elif op == "set_bucket_capacity":
                        command["value"] = rng.choice([1, 2, 5, 12])
                    elif op == "set_time_quant_gear":
                        command["value"] = rng.choice(gears)
                    elif op == "replace_clip":
                        command.update({"clip_id": clip_id, "seed": rng.randrange(10000, 90000)})
                    elif op == "set_clip_format":
                        command.update({"clip_id": clip_id, "sample_rate": rng.choice(rates), "bit_depth": rng.choice(depths)})
                    else:
                        command.update(
                            {
                                "clip_id": clip_id,
                                "frame_index": rng.randrange(0, 20),
                                "value": rng.choice([None, 200, 1200, 20000]),
                            }
                        )
                    commands.append(command)
                    issues = apply_command_to_inputs(engine.manifest, engine.config, command, data_dir)
                    self.assertEqual(issues, [])
                    engine._context = None
                    engine.start_run(command)
                    engine.run_until_complete()

                full = FingerprintEngine(engine.manifest, copy.deepcopy(engine.config), data_dir, root / "full")
                full.start_run()
                full.run_until_complete()
                self.assertEqual(deep_diff(engine.export_canonical(), full.export_canonical()), [])

    def test_atomic_snapshot_recovery(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            store = PersistentStore(Path(temp) / "state")
            store.commit_snapshot({"global_version": 7, "ok": True})
            (store.root / "snapshot.json.tmp").write_text("{broken", encoding="utf-8")
            reloaded = PersistentStore(store.root).load_snapshot()
            self.assertIsNotNone(reloaded)
            self.assertEqual(reloaded["global_version"], 7)
            self.assertTrue(reloaded["ok"])


if __name__ == "__main__":
    unittest.main()
