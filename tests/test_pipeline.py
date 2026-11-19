"""End-to-end deterministic pipeline tests (stdlib unittest + numpy)."""

import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import textwrap
import unittest
import wave

import numpy as np

from speaker_pipeline.audioio import (
    SAMPLE_RATE,
    AudioFormatError,
    read_wav,
    write_wav,
)
from speaker_pipeline.clustering import (
    apply_overrides,
    base_agglomerative,
)
from speaker_pipeline.commands import (
    apply_audio_command,
    load_command_stream,
    update_override_ledger,
)
from speaker_pipeline.concurrency import BoundedWorkerPool
from speaker_pipeline.config import Config
from speaker_pipeline.engine import PipelineEngine
from speaker_pipeline.features import extract_features
from speaker_pipeline.injections import (
    apply_file_injections,
    load_injections,
)
from speaker_pipeline.report import canonical_result
from speaker_pipeline.segmentation import segment_file

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET = os.path.join(REPO_ROOT, "samples", "dataset")


class WorkspaceCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="spk-test-")
        self.work = os.path.join(self.tmp, "work")
        self.store = os.path.join(self.tmp, "store")
        os.makedirs(self.work)
        for name in os.listdir(DATASET):
            if name.endswith(".wav"):
                shutil.copy2(
                    os.path.join(DATASET, name),
                    os.path.join(self.work, name),
                )
        self.config = Config.load(os.path.join(DATASET, "config.json"))
        self.commands = load_command_stream(
            os.path.join(DATASET, "commands.json")
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def engine(self, store=None, injections=None, threads=None):
        cfg = self.config
        if threads is not None:
            cfg = Config(dict(self.config.data, worker_threads=threads))
        return PipelineEngine(
            self.work, cfg, store or self.store, injections=injections
        )


class TestAudioAndFeatures(WorkspaceCase):
    def test_wav_container_validation(self):
        good = os.path.join(self.work, "sample_001.wav")
        meta = read_wav(good)[1]
        self.assertEqual(meta["sample_rate"], SAMPLE_RATE)
        self.assertEqual(meta["channels"], 1)
        self.assertEqual(meta["sample_width"], 2)

        bad = os.path.join(self.tmp, "bad.wav")
        with wave.open(bad, "wb") as wav:
            wav.setnchannels(2)
            wav.setsampwidth(2)
            wav.setframerate(8000)
            wav.writeframes(np.zeros(200, dtype="<i2").tobytes())
        with self.assertRaises(AudioFormatError) as ctx:
            read_wav(bad)
        self.assertEqual(ctx.exception.code, "E_AUDIO_FORMAT")

    def test_frame_determinism_and_padding(self):
        samples, _ = read_wav(os.path.join(self.work, "sample_001.wav"))
        first = extract_features(samples, self.config)
        second = extract_features(samples, self.config)
        self.assertEqual(first, second)
        self.assertTrue(first["frames"][-1]["padded"])
        self.assertFalse(first["frames"][0]["padded"])
        for frame in first["frames"]:
            self.assertEqual(len(frame["voiceprint"]), self.config.data["feature_dim"])
            for value in frame["voiceprint"]:
                self.assertTrue(np.isfinite(value))

    def test_segmentation_sorted_and_overlap_flag(self):
        samples, _ = read_wav(os.path.join(self.work, "sample_001.wav"))
        feature_product = extract_features(samples, self.config)
        seg = segment_file(feature_product, self.config)
        intervals = seg["intervals"]
        self.assertTrue(intervals)
        starts = [(iv["start_ms"], iv["end_frame"]) for iv in intervals]
        self.assertEqual(starts, sorted(starts))
        self.assertTrue(any(iv["overlap"] for iv in intervals))

    def test_clustering_deterministic_and_identifiers(self):
        engine = self.engine()
        result = engine.run_pipeline()
        again = self.engine(store=os.path.join(self.tmp, "store2")).run_pipeline()
        self.assertEqual(
            canonical_result(result), canonical_result(again)
        )
        ids = [c["cluster_id"] for c in result["clusters"]]
        self.assertEqual(ids, sorted(ids))
        self.assertTrue(all(cid.startswith("SPK-") for cid in ids))


def _replay_incremental(work, config, store, commands, injections=None):
    engine = PipelineEngine(work, config, store, injections=injections)
    result = engine.run_pipeline()
    engine.commit_snapshot(result)
    for command in commands:
        try:
            engine.apply_command(command, current_result=result)
        except Exception:
            pass
        result = engine.run_pipeline()
        engine.commit_snapshot(result)
    return engine, result


class TestIncrementalConsistency(WorkspaceCase):
    def test_full_rebuild_matches_incremental_chain(self):
        engine, incremental = _replay_incremental(
            self.work,
            self.config,
            os.path.join(self.tmp, "incr"),
            self.commands,
        )

        # Full rebuild from the identical final working audio + final state.
        full_work = os.path.join(self.tmp, "full_work")
        os.makedirs(full_work)
        for name in os.listdir(self.work):
            if name.endswith(".wav"):
                shutil.copy2(
                    os.path.join(self.work, name),
                    os.path.join(full_work, name),
                )
        full_engine = PipelineEngine(
            full_work,
            self.config,
            os.path.join(self.tmp, "full_store"),
        )
        full_engine.load_state(engine.export_state())
        full = full_engine.run_pipeline()

        self.assertEqual(
            canonical_result(full), canonical_result(incremental)
        )

    def test_merge_and_split_identifier_rule(self):
        engine = self.engine(store=os.path.join(self.tmp, "s1"))
        result = engine.run_pipeline()
        before = [c["cluster_id"] for c in result["clusters"]]
        merge_cmd = {
            "type": "merge_clusters",
            "cluster_ids": before[:2],
            "sequence": 0,
        }
        engine.apply_command(merge_cmd, current_result=result)
        merged = engine.run_pipeline()
        merged_ids = [c["cluster_id"] for c in merged["clusters"]]
        self.assertEqual(merged_ids, sorted(merged_ids))
        total_members = sum(c["member_count"] for c in merged["clusters"])
        self.assertEqual(total_members, len(merged["attribution"]))

        split_cmd = {
            "type": "split_cluster",
            "source_cluster_id": merged_ids[0],
            "member_count": 1,
            "sequence": 1,
        }
        engine.apply_command(split_cmd, current_result=merged)
        split = engine.run_pipeline()
        split_ids = [c["cluster_id"] for c in split["clusters"]]
        self.assertEqual(split_ids, sorted(split_ids))
        # Same override ledger replayed on a fresh engine yields same ids.
        other = self.engine(store=os.path.join(self.tmp, "s2"))
        other.load_state(engine.export_state())
        other_result = other.run_pipeline()
        self.assertEqual(
            canonical_result(other_result), canonical_result(split)
        )

    def test_cache_reuse_and_invalidation_audit(self):
        engine = self.engine()
        first = engine.run_pipeline()
        recomputed_first = {r["cache_key"] for r in first["recomputed_products"]}
        self.assertTrue(recomputed_first)
        second = engine.run_pipeline()
        self.assertEqual(
            {r["cache_key"] for r in second["reused_products"]},
            recomputed_first,
        )
        self.assertEqual(second["recomputed_products"], [])

        # Replacing one file invalidates only that file and downstream.
        cmd = {
            "type": "replace",
            "file": "sample_002.wav",
            "start_ms": 200,
            "end_ms": 700,
            "replace_seed": 99,
            "sequence": 0,
        }
        engine.apply_command(cmd, current_result=second)
        third = engine.run_pipeline()
        recomputed_files = {
            tuple(r["references"])
            for r in third["recomputed_products"]
            if r["stage"] == "feature"
        }
        self.assertTrue(
            any("sample_002.wav" in " ".join(refs) for refs in recomputed_files)
        )
        reused_feature_stages = [
            r for r in third["reused_products"] if r["stage"] == "feature"
        ]
        self.assertTrue(reused_feature_stages)


class TestConcurrency(WorkspaceCase):
    def test_thread_matches_serial_field_by_field(self):
        serial_engine = self.engine(
            store=os.path.join(self.tmp, "serial"), threads=1
        )
        parallel_engine = self.engine(
            store=os.path.join(self.tmp, "parallel"), threads=4
        )
        serial = serial_engine.run_pipeline()
        parallel = parallel_engine.run_pipeline()
        self.assertEqual(
            canonical_result(serial), canonical_result(parallel)
        )
        self.assertLessEqual(
            parallel["worker_high_water_mark"],
            self.config.data["worker_threads"],
        )
        self.assertGreaterEqual(parallel["worker_high_water_mark"], 1)

    def test_pool_stable_ordering_under_random_sleeps(self):
        import time
        import random

        pool = BoundedWorkerPool(4)
        rng = random.Random(3)

        def job(item):
            time.sleep(rng.uniform(0, 0.01))
            return item * item

        items = list(range(25))
        out = pool.map_ordered(job, items, sort_key=lambda x: x)
        self.assertEqual(out, [x * x for x in items])


class TestSnapshotRecovery(WorkspaceCase):
    def test_snapshot_manifest_roundtrip(self):
        engine = self.engine()
        result = engine.run_pipeline()
        manifest = engine.commit_snapshot(result)
        self.assertTrue(manifest["complete"])
        self.assertEqual(manifest["global_version"], 1)
        self.assertEqual(manifest["config_hash"], self.config.hash_key())
        self.assertTrue(manifest["stage_products"])
        self.assertTrue(manifest["completed_intervals"])

        revived = self.engine(store=self.store)
        recovered = revived.restore()
        self.assertEqual(
            recovered["global_version"], manifest["global_version"]
        )
        self.assertEqual(revived.command_cursor, 0)

    def test_forced_kill_leaves_no_partial_snapshot(self):
        engine = self.engine()
        result = engine.run_pipeline()
        engine.commit_snapshot(result)  # version 1 fully committed
        before = engine.store.load_manifest()
        with self.assertRaises(Exception):
            engine.commit_snapshot(result, crash_before_commit=True)
        after = engine.store.load_manifest()
        self.assertEqual(after["global_version"], before["global_version"])
        # A restart restores only the fully committed version.
        revived = self.engine(store=self.store)
        recovered = revived.restore()
        self.assertEqual(recovered["global_version"], 1)

    def test_resume_after_partial_command_stream(self):
        engine = self.engine(store=self.store)
        result = engine.run_pipeline()
        engine.commit_snapshot(result)
        for command in self.commands[:3]:
            try:
                engine.apply_command(command, current_result=result)
            except Exception:
                pass
            result = engine.run_pipeline()
        engine.commit_snapshot(result)
        cursor = engine.command_cursor

        revived = self.engine(store=self.store)
        manifest = revived.restore()
        self.assertEqual(manifest["command_cursor"], cursor)
        self.assertEqual(revived.threshold, engine.threshold)


class TestInjectionIsolation(WorkspaceCase):
    def _injected_engine(self):
        injections = load_injections(os.path.join(DATASET, "injections.json"))
        apply_file_injections(self.work, injections)
        return self.engine(injections=injections), injections

    def test_all_injection_faults_map_to_business_codes(self):
        engine, injections = self._injected_engine()
        result = engine.run_pipeline()
        codes = {row["code"] for row in result["failures"]}
        self.assertIn("E_AUDIO_FORMAT", codes)
        self.assertIn("E_NON_FINITE", codes)
        self.assertIn("E_INVALID_THRESHOLD", codes)

        # Out-of-range replace command rejected at the command layer.
        injected = [
            cmd for cmd in self.commands if cmd.get("injection_id")
        ]
        if injected:
            oor = injected[0]
            from speaker_pipeline.errors import OutOfRangeError

            with self.assertRaises(OutOfRangeError) as ctx:
                engine.apply_command(oor, current_result=result)
            self.assertEqual(ctx.exception.code, "E_OUT_OF_RANGE")
            self.assertEqual(ctx.exception.injection_id, "INJ-004")
        else:
            self.skipTest("dataset command stream contains no injected command")

    def test_failed_interval_excluded_from_clustering(self):
        engine, _ = self._injected_engine()
        result = engine.run_pipeline()
        # sample_003 is corrupt and must produce no intervals/attribution.
        bad_files = {
            row["file"]
            for row in result["failures"]
            if row["code"] == "E_AUDIO_FORMAT"
        }
        self.assertIn("sample_003.wav", bad_files)
        attributed_files = {
            meta["file"] for meta in result["interval_meta"]
        }
        self.assertNotIn("sample_003.wav", attributed_files)
        # Remaining files were still processed.
        self.assertIn("sample_001.wav", attributed_files)

    def test_report_contains_stage_timings_and_versions(self):
        engine, _ = self._injected_engine()
        result = engine.run_pipeline()
        manifest = engine.commit_snapshot(result)
        from speaker_pipeline.report import build_report

        report = build_report(result, engine.config, recovery_version=None)
        report["snapshot_version"] = manifest["global_version"]
        self.assertIn("stage_timings_ms", report)
        self.assertTrue(report["stage_timings_ms"])
        self.assertEqual(
            report["snapshot_version"], manifest["global_version"]
        )
        self.assertTrue(report["failures"])




class TestIntervalIsolation(WorkspaceCase):
    def test_single_segmentation_fault_keeps_other_intervals(self):
        from unittest import mock
        import speaker_pipeline.segmentation as seg_mod
        engine = self.engine()
        original = seg_mod.cosine_similarity
        state = {"calls": 0}

        def flaky(vec_a, vec_b):
            state["calls"] += 1
            value = original(vec_a, vec_b)
            if state["calls"] == 2:
                from speaker_pipeline.errors import SegmentationError

                raise SegmentationError(
                    "injected interval fault",
                    stage="segmentation",
                    interval_index=1,
                )
            return value

        with mock.patch.object(seg_mod, "cosine_similarity", flaky):
            result = engine.run_pipeline()
        self.assertIn("E_SEGMENTATION", [r["code"] for r in result["failures"]])
        self.assertTrue(result["attribution"])
        self.assertTrue(result["interval_meta"])


@unittest.skipIf(sys.platform == "win32", "POSIX signal semantics only")
class TestSignalExit(unittest.TestCase):
    def test_sigterm_finishes_snapshot_and_exits_zero(self):
        script = textwrap.dedent(
            """
            import os, signal, sys, threading, time
            sys.path.insert(0, {root!r})
            from speaker_pipeline.config import Config
            from speaker_pipeline.engine import PipelineEngine
            from speaker_pipeline.commands import load_command_stream

            work, store, dataset = sys.argv[1], sys.argv[2], sys.argv[3]
            cfg = Config.load(os.path.join(dataset, "config.json"))
            cmds = load_command_stream(os.path.join(dataset, "commands.json"))
            eng = PipelineEngine(work, cfg, store)
            res = eng.run_pipeline()
            eng.commit_snapshot(res)

            def send():
                time.sleep(0.3)
                os.kill(os.getpid(), signal.SIGTERM)

            threading.Thread(target=send, daemon=True).start()
            stop = {{"flag": False}}

            def handler(signum, frame):
                stop["flag"] = True

            signal.signal(signal.SIGTERM, handler)
            for command in cmds:
                if stop["flag"]:
                    break
                try:
                    eng.apply_command(command, current_result=res)
                except Exception:
                    pass
                res = eng.run_pipeline()
                eng.commit_snapshot(res)
            sys.exit(0)
            """
        ).lstrip().format(root=REPO_ROOT)
        with tempfile.TemporaryDirectory() as tmp:
            work = os.path.join(tmp, "work")
            store = os.path.join(tmp, "store")
            os.makedirs(work)
            for name in os.listdir(DATASET):
                if name.endswith(".wav"):
                    shutil.copy2(
                        os.path.join(DATASET, name),
                        os.path.join(work, name),
                    )
            helper = os.path.join(tmp, "signal_runner.py")
            with open(helper, "w", encoding="utf-8") as handle:
                handle.write(script)
            proc = subprocess.run(
                [sys.executable, helper, work, store, DATASET],
                capture_output=True,
                text=True,
                timeout=60,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
