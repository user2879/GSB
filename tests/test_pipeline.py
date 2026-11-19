import copy
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest

import numpy as np

from mocbench.canonical import version_digest
from mocbench.cli import main
from mocbench.engine import SolveEngine
from mocbench.errors import BenchError
from mocbench.fill import fill_missing
from mocbench.parser import load_bundle
from mocbench.persistence import read_chunk, read_manifest, verify_state
from mocbench.quat import slerp
from mocbench.report import build_poses, build_report, export_atomic
from mocbench.samplegen import build_sample
from mocbench.solver import solve_frame

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = os.path.join(ROOT, "samples", "sample.json")


class WorkbenchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.makedirs(os.path.join(ROOT, "samples"), exist_ok=True)
        if not os.path.exists(SAMPLE):
            with open(SAMPLE, "w", encoding="utf-8") as handle:
                json.dump(build_sample(), handle, indent=2, sort_keys=True)
        cls.bundle = load_bundle(SAMPLE)
        cls.tmp = tempfile.mkdtemp(prefix="mocbench_test_")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def state_dir(self, name):
        path = os.path.join(self.tmp, name)
        shutil.rmtree(path, ignore_errors=True)
        return path

    def test_01_quaternion_and_fill_determinism(self):
        q0 = np.array([1.0, 0.0, 0.0, 0.0])
        q1 = np.array([0.0, 1.0, 0.0, 0.0])
        mid = slerp(q0, q1, 0.5)
        self.assertAlmostEqual(mid[0], mid[1], places=12)
        result_a = fill_missing(self.bundle)
        result_b = fill_missing(self.bundle)
        for name in self.bundle.skeleton.order:
            for frame in range(self.bundle.curves.frame_count):
                np.testing.assert_array_equal(result_a.rotations[name][frame], result_b.rotations[name][frame])
        self.assertEqual(result_a.filled_frame_count, 18)
        self.assertEqual(len(result_a.intervals), 6)
        methods = {(item["bone"], item["method"]) for item in result_a.intervals}
        self.assertIn(("root", "rest_pose"), methods)
        self.assertIn(("spine", "slerp"), methods)

    def test_02_solve_convergence_profile(self):
        filled = fill_missing(self.bundle)
        unconverged = []
        for frame in range(self.bundle.curves.frame_count):
            local = {name: filled.rotations[name][frame] for name in self.bundle.skeleton.order}
            result = solve_frame(frame, local, self.bundle.curves.root_positions[frame], self.bundle)
            if not result["converged"]:
                unconverged.append(result)
        self.assertEqual([item["frame"] for item in unconverged], [35])
        failed = [c for c in unconverged[0]["residuals"] if not c["passed"]]
        self.assertEqual([c["constraint"] for c in failed], ["ik"])
        self.assertEqual(unconverged[0]["iterations_used"], self.bundle.params.max_iterations)
        for residual in unconverged[0]["residuals"]:
            self.assertEqual(len(residual["last_iteration_trace"]), self.bundle.params.max_iterations)

    def test_03_engine_report_and_serial_equivalence(self):
        state_dir = self.state_dir("engine")
        engine = SolveEngine(self.bundle, state_dir)
        engine.run()
        frames = engine.load_all_frames()
        report = build_report(frames, engine.fill_result, self.bundle, engine.version)
        poses = build_poses(frames, self.bundle, engine.version)
        self.assertEqual(report["frame_count"], 60)
        self.assertEqual(report["passed_frame_count"], 59)
        self.assertEqual(report["unconverged_frame_count"], 1)
        self.assertEqual(report["solve_order"], ["angle_limit", "ik", "bone_length", "contact"])
        self.assertEqual(report["per_constraint_pass_rate"]["bone_length"]["pass_rate"], 1.0)
        self.assertEqual(report["per_constraint_pass_rate"]["contact"]["evaluated_frames"], 4)

        filled = engine.fill_result
        worst = 0.0
        for frame in frames:
            local = {name: filled.rotations[name][frame["frame"]] for name in self.bundle.skeleton.order}
            direct = solve_frame(frame["frame"], local, self.bundle.curves.root_positions[frame["frame"]], self.bundle)
            for name in self.bundle.skeleton.order:
                worst = max(worst, np.max(np.abs(direct["local_rotations"][name] - np.asarray(frame["local_rotations"][name]))))
            worst = max(worst, np.max(np.abs(direct["root_position"] - np.asarray(frame["root_position"]))))
        self.assertEqual(worst, 0.0)

        output_dir = os.path.join(self.tmp, "export")
        info_first = export_atomic(output_dir, report, poses)
        info_second = export_atomic(output_dir, report, poses)
        self.assertTrue(info_first["replaced"])
        self.assertFalse(info_second["replaced"])

    def test_04_interrupt_and_resume(self):
        state_dir = os.path.join(self.tmp, "interrupt_state")
        output_dir = os.path.join(self.tmp, "interrupt_out")
        shutil.rmtree(state_dir, ignore_errors=True)
        proc = subprocess.Popen(
            [sys.executable, os.path.join(ROOT, "main.py"), "solve",
             "--input", SAMPLE, "--state-dir", state_dir, "--output", output_dir],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        deadline = 0
        while not os.path.exists(os.path.join(state_dir, "chunk_000000.json")) and deadline < 200:
            time.sleep(0.02)
            deadline += 1
        self.assertTrue(os.path.exists(os.path.join(state_dir, "chunk_000000.json")))
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=20)
        self.assertIn(proc.returncode, (143, -signal.SIGTERM))
        if proc.returncode == 143:
            self.assertTrue(os.path.exists(os.path.join(state_dir, "interrupt.flag")))
        manifest = read_manifest(state_dir)
        on_disk = [name for name in os.listdir(state_dir) if name.startswith("chunk_")]
        self.assertEqual(len(on_disk), len(manifest["completed_chunks"]))
        self.assertFalse(os.path.exists(os.path.join(output_dir, "poses.json")))

        rc = main(["resume", "--input", SAMPLE, "--state-dir", state_dir, "--output", output_dir])
        self.assertEqual(rc, 0)
        resumed = json.load(open(os.path.join(output_dir, "poses.json")))
        clean = SolveEngine(self.bundle, self.state_dir("clean"))
        clean.run()
        clean_frames = clean.load_all_frames()
        self.assertEqual(resumed["frame_count"], len(clean_frames))
        for left, right in zip(resumed["frames"], clean_frames):
            self.assertEqual(left["local_rotations"], right["local_rotations"])
            self.assertEqual(left["root_position"], right["root_position"])

    def test_05_digest_tamper_recovery_and_verify(self):
        state_dir = self.state_dir("tamper")
        engine = SolveEngine(self.bundle, state_dir)
        engine.run()
        chunk_path = os.path.join(state_dir, "chunk_000002.json")
        payload = json.load(open(chunk_path))
        payload["frames"][0]["root_position"][1] += 3.0
        json.dump(payload, open(chunk_path, "w"))
        rc = main(["resume", "--input", SAMPLE, "--state-dir", state_dir,
                   "--output", os.path.join(self.tmp, "tamper_out")])
        self.assertEqual(rc, 0)
        self.assertTrue(any(name.endswith(".invalid") for name in os.listdir(state_dir)))
        verify_state(state_dir, engine.version, engine.frame_count, engine.chunk_size)

        verify_dir = self.state_dir("verify_bad")
        other = SolveEngine(self.bundle, verify_dir)
        other.run()
        bad_path = os.path.join(verify_dir, "chunk_000001.json")
        bad = json.load(open(bad_path))
        bad["digest"] = "0" * 64
        json.dump(bad, open(bad_path, "w"))
        with self.assertRaises(BenchError) as context:
            verify_state(verify_dir, other.version, other.frame_count, other.chunk_size)
        self.assertEqual(context.exception.code, "E_DIGEST")
        verify_rc = main(["verify", "--input", SAMPLE, "--state-dir", verify_dir])
        self.assertEqual(verify_rc, 12)

    def test_06_validation_error_codes(self):
        cases = [
            ("E_BONE", lambda d: d["skeleton"]["bones"].__setitem__(3, {**d["skeleton"]["bones"][3], "name": "root"})),
            ("E_HIERARCHY", lambda d: d["skeleton"]["bones"][1].__setitem__("parent", "head")),
            ("E_LENGTH", lambda d: d["skeleton"]["bones"][2].__setitem__("length", -1.0)),
            ("E_CURVE", lambda d: d["curves"]["rotations"].__setitem__("root", d["curves"]["rotations"]["root"][:-1])),
            ("E_QUATERNION", lambda d: d["curves"]["rotations"]["spine"][5].__setitem__(2, float("inf"))),
            ("E_LIMIT", lambda d: d["constraints"]["angle_limits"][0].__setitem__("lower", 9.0)),
            ("E_TARGET", lambda d: d["constraints"]["ik_targets"][0].__setitem__("frames", [999])),
            ("E_CONTACT", lambda d: d["constraints"].__setitem__("contact_frames", [77])),
            ("E_PARAM", lambda d: d["params"].__setitem__("damping", 2.0)),
        ]
        for expected, mutate in cases:
            data = copy.deepcopy(self.bundle.raw)
            mutate(data)
            path = os.path.join(self.tmp, f"bad_{expected}.json")
            json.dump(data, open(path, "w"))
            with self.assertRaises(BenchError) as context:
                load_bundle(path)
            self.assertEqual(context.exception.code, expected)

    def test_07_version_change_replaces_output(self):
        output_dir = os.path.join(self.tmp, "versioned")
        state_a = self.state_dir("va")
        state_b = self.state_dir("vb")
        main(["solve", "--input", SAMPLE, "--state-dir", state_a, "--output", output_dir])
        first = json.load(open(os.path.join(output_dir, "report.json")))
        v2_path = os.path.join(self.tmp, "v2.json")
        json.dump(build_sample(damping=0.45), open(v2_path, "w"), indent=2, sort_keys=True)
        main(["solve", "--input", v2_path, "--state-dir", state_b, "--output", output_dir])
        second_report = json.load(open(os.path.join(output_dir, "report.json")))
        second_poses = json.load(open(os.path.join(output_dir, "poses.json")))
        self.assertNotEqual(first["version_digest"], second_report["version_digest"])
        self.assertEqual(second_report["version_digest"], second_poses["version_digest"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
