from __future__ import annotations

import json
import os
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

from vad.audio import read_wav
from vad.config import load_config
from vad.demo import generate_demo
from vad.features import extract_features, frame_count
from vad.pipeline import run_vad
from vad.state_machine import run_state_machine


PROJECT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT / "demo" / "vad_config.json"


def sample_projection(report: dict) -> list[dict]:
    return [
        {key: value for key, value in sample.items() if key != "cache_events"}
        for sample in report["samples"]
    ]


class VADTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory(prefix="vad-tests-")
        cls.root = Path(cls.temporary.name)
        cls.data_dir = cls.root / "input"
        cls.config_path = cls.root / "config.json"
        cls.manifest_path = cls.root / "manifest.json"
        shutil.copyfile(CONFIG_PATH, cls.config_path)
        generate_demo(
            cls.data_dir,
            cls.config_path,
            cls.manifest_path,
            seed=20261120,
        )
        cls.baseline = run_vad(
            cls.data_dir,
            cls.config_path,
            cls.manifest_path,
            cls.root / "state-baseline",
            cls.root / "baseline.json",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def _write_config(self, name: str, **changes: object) -> Path:
        raw = json.loads(self.config_path.read_text(encoding="utf-8"))
        raw.update(changes)
        path = self.root / name
        path.write_text(
            json.dumps(raw, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        return path

    def test_endpoint_report_and_noise_tolerance(self) -> None:
        report = self.baseline
        self.assertEqual(report["status"], "complete")
        self.assertEqual(len(report["samples"]), 6)
        self.assertEqual(report["failures"], [])
        self.assertEqual(report["summary"]["endpoint_segments"], 28)
        self.assertEqual(report["summary"]["segment_count_mismatches"], 0)
        self.assertEqual(report["summary"]["noise_tolerance_failures"], 0)
        self.assertEqual(
            {sample["scale"]["selected_profile"] for sample in report["samples"]},
            {"small", "medium", "large"},
        )
        previous_file = ""
        for sample in report["samples"]:
            self.assertGreater(sample["stable_sort_key"], previous_file)
            previous_file = sample["stable_sort_key"]
            self.assertTrue(sample["segment_count_ok"])
            self.assertTrue(sample["noise_tolerance"]["ok"])
            starts = [segment["start_ms"] for segment in sample["segments"]]
            self.assertEqual(starts, sorted(starts))
            for left, right in zip(sample["segments"], sample["segments"][1:]):
                self.assertLess(
                    (left["start_ms"], left["end_ms"]),
                    (right["start_ms"], right["end_ms"]),
                )
            for expected, frame in enumerate(sample["frames"]):
                self.assertEqual(frame["frame_index"], expected)
                self.assertAlmostEqual(frame["start_ms"], expected * 10.0, places=9)
                self.assertIn(frame["state"], {"SILENCE", "SUSPECT", "SPEECH", "TRAIL"})
                self.assertTrue(frame["transition_criterion"])
        padded = report["samples"][4]
        self.assertEqual(padded["file"], "padded_single.wav")
        self.assertEqual(padded["frame_count"], 201)
        self.assertTrue(padded["frames"][-1]["is_last_frame"])
        self.assertEqual(padded["frames"][-1]["valid_samples"], 80)
        self.assertEqual(padded["frames"][-1]["start_ms"], 2000.0)

    def test_audio_validation_and_frame_geometry(self) -> None:
        config = load_config(self.config_path)
        audio = read_wav(
            self.data_dir / "padded_single.wav",
            config.audio,
            "padded_single.wav",
        )
        _, hop_samples = 320, 160
        self.assertEqual(audio.samples.size, 32080)
        self.assertEqual(frame_count(audio.samples.size, hop_samples), 201)
        features = extract_features(audio.samples, 16000, config.raw, 8)
        self.assertEqual(features["frame_count"], 201)
        self.assertEqual(features["frames"][0]["start_ms"], 0.0)
        self.assertEqual(features["frames"][1]["start_ms"], 10.0)
        self.assertTrue(features["frames"][-1]["is_last_frame"])
        self.assertEqual(features["frames"][-1]["valid_samples"], 80)

    def test_invalid_audio_is_isolated_with_business_code(self) -> None:
        input_dir = self.root / "invalid-input"
        input_dir.mkdir()
        shutil.copyfile(
            self.data_dir / "short_two.wav", input_dir / "short_two.wav"
        )
        (input_dir / "broken.wav").write_bytes(b"RIFF\x10\x00\x00\x00WAVE")
        manifest = {
            "schema_version": "1.0",
            "noise_reference_samples": [],
            "samples": [
                {
                    "file": "short_two.wav",
                    "shard_count": 1,
                    "expected_segments": [2, 2],
                    "scale_profile": "small",
                },
                {
                    "file": "broken.wav",
                    "shard_count": 1,
                    "expected_segments": [0, 0],
                    "scale_profile": "small",
                },
            ],
        }
        manifest_path = self.root / "invalid-manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        report = run_vad(
            input_dir,
            self.config_path,
            manifest_path,
            self.root / "state-invalid",
        )
        self.assertEqual(report["summary"]["successful_samples"], 1)
        self.assertEqual(report["summary"]["failed_samples"], 1)
        self.assertEqual(report["failures"][0]["code"], "E_AUDIO_FORMAT")
        self.assertEqual(report["failures"][0]["file"], "broken.wav")
        self.assertEqual(report["samples"][0]["file"], "short_two.wav")

    def test_state_machine_requires_consecutive_evidence_and_silence_first(self) -> None:
        config = load_config(self.config_path).raw
        frames = []
        for index in range(8):
            spike = index == 3
            frames.append(
                {
                    "frame_index": index,
                    "start_ms": index * 10.0,
                    "is_last_frame": False,
                    "rms_log": 8.0 if spike else 4.0,
                    "zcr": 0.02 if spike else 0.5,
                    "flatness": 0.05 if spike else 0.85,
                }
            )
        features = {"frames": frames}
        noise = {
            "low_threshold_ln": 4.5,
            "high_threshold_ln": 6.0,
        }
        result = run_state_machine(features, noise, config)
        states = [frame["state"] for frame in result["frame_states"]]
        self.assertNotIn("SPEECH", states)
        self.assertEqual(states[3], "SUSPECT")
        self.assertEqual(states[4], "SILENCE")

        boundary_frames = [
            {
                "frame_index": 0,
                "start_ms": 0.0,
                "is_last_frame": False,
                "rms_log": 6.0,
                "zcr": 0.1,
                "flatness": 0.1,
            }
        ]
        boundary = run_state_machine({"frames": boundary_frames}, noise, config)
        self.assertEqual(boundary["frame_states"][0]["state"], "SILENCE")
        self.assertEqual(boundary["frame_states"][0]["state_before"], "SILENCE")

        release_config = dict(config)
        release_config["release_frames"] = 1
        speech_then_low = {
            "frames": [
                {
                    "frame_index": index,
                    "start_ms": index * 10.0,
                    "is_last_frame": False,
                    "rms_log": 8.0 if index < 2 else 4.0,
                    "zcr": 0.02 if index < 2 else 0.5,
                    "flatness": 0.05 if index < 2 else 0.85,
                }
                for index in range(3)
            ]
        }
        release = run_state_machine(speech_then_low, noise, release_config)
        self.assertEqual(
            [frame["state"] for frame in release["frame_states"]],
            ["SUSPECT", "SPEECH", "SILENCE"],
        )

    def test_cache_reuse_and_threshold_stage_invalidation(self) -> None:
        state_dir = self.root / "state-incremental"
        first = run_vad(
            self.data_dir,
            self.config_path,
            self.manifest_path,
            state_dir,
        )
        second = run_vad(
            self.data_dir,
            self.config_path,
            self.manifest_path,
            state_dir,
        )
        self.assertEqual(second["cache_audit"]["recompute_count"], 0)
        self.assertEqual(second["cache_audit"]["reuse_count"], 24)
        self.assertEqual(first["result_digest"], second["result_digest"])

        changed = self._write_config("config-threshold.json", high_threshold=5.1)
        incremental = run_vad(
            self.data_dir,
            changed,
            self.manifest_path,
            state_dir,
        )
        feature_events = [
            event
            for event in incremental["cache_audit"]["reuse"]
            + incremental["cache_audit"]["recompute"]
            if event["stage"] == "features"
        ]
        state_events = [
            event
            for event in incremental["cache_audit"]["reuse"]
            + incremental["cache_audit"]["recompute"]
            if event["stage"] == "state_machine"
        ]
        endpoint_events = [
            event
            for event in incremental["cache_audit"]["reuse"]
            + incremental["cache_audit"]["recompute"]
            if event["stage"] == "endpoints"
        ]
        self.assertTrue(all(event["action"] == "reuse" for event in feature_events))
        self.assertTrue(all(event["action"] == "recompute" for event in state_events))
        self.assertTrue(
            all(event["action"] == "recompute" for event in endpoint_events)
        )
        clean = run_vad(
            self.data_dir,
            changed,
            self.manifest_path,
            self.root / "state-threshold-clean",
        )
        self.assertEqual(incremental["result_digest"], clean["result_digest"])
        self.assertEqual(sample_projection(incremental), sample_projection(clean))

    def test_scale_profile_switch_incremental_matches_full(self) -> None:
        config = self._write_config("config-large.json", scale_profile_override="large")
        state_dir = self.root / "state-switch"
        baseline = run_vad(
            self.data_dir,
            self.config_path,
            self.manifest_path,
            state_dir,
        )
        switched = run_vad(
            self.data_dir,
            config,
            self.manifest_path,
            state_dir,
        )
        clean = run_vad(
            self.data_dir,
            config,
            self.manifest_path,
            self.root / "state-switch-clean",
        )
        self.assertEqual(baseline["status"], "complete")
        self.assertTrue(
            all(sample["scale"]["selected_profile"] == "large" for sample in switched["samples"])
        )
        self.assertEqual(switched["result_digest"], clean["result_digest"])
        self.assertEqual(sample_projection(switched), sample_projection(clean))
        policy_events = [
            event
            for event in switched["cache_audit"]["reuse"]
            + switched["cache_audit"]["recompute"]
            if event["stage"] == "policy"
        ]
        self.assertTrue(all(event["action"] == "recompute" for event in policy_events))

    def test_replacing_one_audio_invalidates_only_that_sample(self) -> None:
        state_dir = self.root / "state-audio-replace"
        baseline = run_vad(
            self.data_dir,
            self.config_path,
            self.manifest_path,
            state_dir,
        )
        replacement_dir = self.root / "replacement-input"
        shutil.copytree(self.data_dir, replacement_dir)
        target = replacement_dir / "short_two.wav"
        blob = bytearray(target.read_bytes())
        data_offset = blob.find(b"data")
        payload_offset = data_offset + 8
        first_sample = struct.unpack_from("<h", blob, payload_offset)[0]
        struct.pack_into("<h", blob, payload_offset, min(32767, first_sample + 1))
        target.write_bytes(bytes(blob))
        incremental = run_vad(
            replacement_dir,
            self.config_path,
            self.manifest_path,
            state_dir,
        )
        changed_events = [
            event
            for event in incremental["cache_audit"]["reuse"]
            + incremental["cache_audit"]["recompute"]
            if event["file"] == "short_two.wav"
        ]
        other_reuse = [
            event
            for event in incremental["cache_audit"]["reuse"]
            if event["file"] != "short_two.wav"
        ]
        self.assertEqual(len(changed_events), 4)
        self.assertTrue(all(event["action"] == "recompute" for event in changed_events))
        self.assertTrue(any(event["reason"] == "audio_content_changed" for event in changed_events))
        self.assertEqual(len(other_reuse), 20)
        clean = run_vad(
            replacement_dir,
            self.config_path,
            self.manifest_path,
            self.root / "state-audio-replace-clean",
        )
        self.assertNotEqual(baseline["result_digest"], incremental["result_digest"])
        self.assertEqual(incremental["result_digest"], clean["result_digest"])
        self.assertEqual(sample_projection(incremental), sample_projection(clean))

    def test_serial_and_thread_pool_are_field_identical(self) -> None:
        serial = run_vad(
            self.data_dir,
            self.config_path,
            self.manifest_path,
            self.root / "state-serial",
            concurrency_override=1,
        )
        parallel = run_vad(
            self.data_dir,
            self.config_path,
            self.manifest_path,
            self.root / "state-parallel",
            concurrency_override=4,
        )
        self.assertEqual(serial["execution"]["mode"], "serial")
        self.assertEqual(parallel["execution"]["mode"], "thread_pool")
        self.assertLessEqual(parallel["execution"]["observed_max_concurrency"], 4)
        self.assertEqual(serial["deterministic_digest"], parallel["deterministic_digest"])
        self.assertEqual(serial["samples"], parallel["samples"])
        self.assertEqual(serial["cache_audit"], parallel["cache_audit"])
        self.assertEqual(serial["failures"], parallel["failures"])

    def test_snapshot_resume_and_incomplete_snapshot_rejection(self) -> None:
        state_dir = self.root / "state-interrupt"
        stop_event = threading.Event()
        watcher_done = threading.Event()
        manifest_path = state_dir / "snapshot" / "manifest.json"

        def watcher() -> None:
            deadline = time.monotonic() + 20.0
            while time.monotonic() < deadline and not watcher_done.is_set():
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    time.sleep(0.002)
                    continue
                if manifest.get("cursor", 0) >= 2:
                    stop_event.set()
                    return
                time.sleep(0.002)

        thread = threading.Thread(target=watcher, daemon=True)
        thread.start()
        interrupted = run_vad(
            self.data_dir,
            self.config_path,
            self.manifest_path,
            state_dir,
            concurrency_override=1,
            stop_event=stop_event,
        )
        watcher_done.set()
        thread.join(timeout=2.0)
        self.assertEqual(interrupted["status"], "interrupted")
        self.assertGreaterEqual(interrupted["recovery"]["resume_cursor"], 2)
        self.assertLess(interrupted["recovery"]["resume_cursor"], 6)

        resumed = run_vad(
            self.data_dir,
            self.config_path,
            self.manifest_path,
            state_dir,
            concurrency_override=1,
        )
        self.assertEqual(resumed["status"], "complete")
        self.assertEqual(resumed["recovery"]["mode"], "snapshot_resume")
        self.assertGreaterEqual(resumed["recovery"]["recovered_sample_count"], 2)
        self.assertEqual(resumed["result_digest"], self.baseline["result_digest"])

        corrupt_state = self.root / "state-corrupt"
        shutil.copytree(state_dir, corrupt_state)
        corrupt_manifest = json.loads(
            (corrupt_state / "snapshot" / "manifest.json").read_text(encoding="utf-8")
        )
        corrupt_manifest["status"] = "interrupted"
        corrupt_manifest["cursor"] = 3
        corrupt_manifest["sample_entries"] = corrupt_manifest["sample_entries"][:3]
        manifest_target = corrupt_state / "snapshot" / "manifest.json"
        manifest_target.write_text(json.dumps(corrupt_manifest), encoding="utf-8")
        snapshot_name = corrupt_manifest["sample_entries"][1]["snapshot"]
        (corrupt_state / "snapshot" / "samples" / snapshot_name).write_text(
            "{not valid json", encoding="utf-8"
        )
        recovered = run_vad(
            self.data_dir,
            self.config_path,
            self.manifest_path,
            corrupt_state,
            concurrency_override=1,
        )
        self.assertEqual(recovered["status"], "complete")
        self.assertEqual(recovered["recovery"]["validated_cursor"], 1)
        self.assertTrue(recovered["recovery"]["recovery_problems"])
        self.assertEqual(recovered["result_digest"], self.baseline["result_digest"])

    def test_cli_signal_exit_code_is_zero(self) -> None:
        state_dir = self.root / "state-cli-signal"
        output = self.root / "cli-signal.json"
        command = [
            sys.executable,
            "-m",
            "vad.cli",
            "run",
            "--input",
            str(self.data_dir),
            "--config",
            str(self.config_path),
            "--manifest",
            str(self.manifest_path),
            "--state-dir",
            str(state_dir),
            "--output",
            str(output),
            "--serial",
        ]
        popen_options: dict[str, object] = {}
        if os.name == "nt":
            popen_options["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        process = subprocess.Popen(
            command,
            cwd=PROJECT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            **popen_options,
        )
        deadline = time.monotonic() + 10.0
        manifest_path = state_dir / "snapshot" / "manifest.json"
        while time.monotonic() < deadline:
            if manifest_path.exists():
                break
            time.sleep(0.01)
        if os.name == "nt":
            process.send_signal(getattr(signal, "CTRL_BREAK_EVENT", signal.SIGTERM))
        else:
            process.send_signal(signal.SIGINT)
        stdout, stderr = process.communicate(timeout=30.0)
        self.assertEqual(process.returncode, 0, msg=f"{stdout}\n{stderr}")
        report = json.loads(output.read_text(encoding="utf-8"))
        self.assertIn(report["status"], {"interrupted", "complete"})
        if report["status"] == "interrupted":
            self.assertLess(report["recovery"]["resume_cursor"], 6)


if __name__ == "__main__":
    unittest.main()
