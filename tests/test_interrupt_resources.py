"""Business rules 7-8: interrupt/resume, hard kill, memory cap, fault chain."""

import json
import os
import signal
import subprocess
import sys
import time
import unittest

import numpy as np

from pointcloud_engine.engine import render_scene
from pointcloud_engine.scheduler import INTERRUPT_EXIT_CODE, InterruptFlag
from tests._harness import Workspace

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _slow_scene(workdir: str, tiles: int = 16, delay: float = 0.3) -> str:
    path = os.path.join(workdir, "scene.json")
    from tests._harness import base_scene, make_cloud
    from pointcloud_engine.pointcloud import save_point_cloud

    cloud = make_cloud(n=120)
    cloud.source_path = os.path.join(workdir, "cloud.pts3")
    save_point_cloud(cloud.source_path, cloud)
    scene = base_scene(workdir)
    scene["width"] = 48
    scene["height"] = 32
    scene["supersample"] = 2
    scene["tile_width"] = 24
    scene["tile_height"] = 16
    scene["concurrency"] = 2
    scene["tile_delay_seconds"] = delay
    scene["memory_budget_bytes"] = 24 * 16 * 8 * 2
    scene["point_radius"] = 0.1
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(scene, handle, indent=2, sort_keys=True)
    return path


class InterruptTests(unittest.TestCase):
    def test_in_process_interrupt_flag_at_tile_boundary(self):
        import threading

        ws = Workspace()
        path = _slow_scene(ws.dir, delay=0.3)
        flag = InterruptFlag(os.path.join(ws.dir, "state"))
        flag.install()
        from pointcloud_engine.scene import load_scene
        from pointcloud_engine.render import prepare_points
        from pointcloud_engine.state import TileStore
        from pointcloud_engine.scheduler import run_scheduler

        scene = load_scene(path)
        store = TileStore(scene, scene.version_id(scene.cloud))
        prepared = prepare_points(scene)

        timer = threading.Timer(0.7, flag.event.set)
        timer.start()
        outcome = run_scheduler(scene, prepared, store, flag)
        timer.join()
        flag.restore()

        self.assertTrue(outcome.interrupted)
        self.assertTrue(os.path.isfile(os.path.join(ws.dir, "state", "interrupted.json")))
        remaining = [n for n in os.listdir(os.path.join(ws.dir, "state")) if n.endswith(".npz")]
        self.assertGreater(len(remaining), 0)
        self.assertLess(len(remaining), scene.tile_count)
        ws.cleanup()

    def test_sigterm_subprocess_resumes_and_completes(self):
        ws = Workspace()
        path = _slow_scene(ws.dir, delay=0.4)
        env = dict(os.environ, PYTHONPATH=PROJECT_ROOT)
        cmd = [sys.executable, "-m", "pointcloud_engine.cli", "render", path]

        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        try:
            time.sleep(0.9)
            proc.send_signal(signal.SIGTERM)
            stdout, stderr = proc.communicate(timeout=20)
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.stderr is not None:
                proc.stderr.close()
        self.assertEqual(proc.returncode, INTERRUPT_EXIT_CODE, msg=stderr)
        self.assertIn("interrupted", stdout)

        state_files = [n for n in os.listdir(os.path.join(ws.dir, "state"))
                       if n.endswith(".npz")]
        self.assertGreater(len(state_files), 0)
        self.assertLess(len(state_files), 16)

        # Resume: reuses persisted tiles and completes without re-rendering all.
        resume = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(resume.returncode, 0, msg=resume.stderr)
        report = json.loads(resume.stdout)
        self.assertEqual(report["status"], "completed")
        self.assertGreater(len(report["report"]["tiles_reused"]), 0)
        self.assertTrue(os.path.isfile(os.path.join(ws.dir, "output", "gray.pgm")))
        self.assertTrue(os.path.isfile(os.path.join(ws.dir, "output", "depth.pfm")))
        ws.cleanup()

    def test_hard_kill_recovers_from_persisted_tiles_only(self):
        ws = Workspace()
        path = _slow_scene(ws.dir, delay=0.4)
        env = dict(os.environ, PYTHONPATH=PROJECT_ROOT)
        cmd = [sys.executable, "-m", "pointcloud_engine.cli", "render", path]
        proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, text=True)
        try:
            time.sleep(0.9)
            proc.kill()  # SIGKILL: no marker, no graceful report promised
            proc.wait()
        finally:
            if proc.stdout is not None:
                proc.stdout.close()
            if proc.stderr is not None:
                proc.stderr.close()
        # No interrupted.json is guaranteed after a hard kill.
        marker = os.path.join(ws.dir, "state", "interrupted.json")
        self.assertFalse(os.path.exists(marker))

        resume = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
        self.assertEqual(resume.returncode, 0, msg=resume.stderr)
        self.assertEqual(json.loads(resume.stdout)["status"], "completed")
        ws.cleanup()


class ResourceAndFaultTests(unittest.TestCase):
    def test_memory_cap_limits_inflight_tiles_without_reducing_supersample(self):
        ws = Workspace()
        path = ws.write()
        with open(path, encoding="utf-8") as handle:
            scene = json.load(handle)
        per_tile = scene["tile_width"] * scene["tile_height"] * 8
        scene["concurrency"] = 8
        scene["memory_budget_bytes"] = per_tile * 2  # only two tiles in flight
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(scene, handle)
        result = render_scene(path, install_signals=False)
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["report"]["max_inflight_tiles"], 2)
        # Supersampling unchanged: grid still fully covered.
        self.assertEqual(result["report"]["grid"],
                         [scene["width"] * scene["supersample"],
                          scene["height"] * scene["supersample"]])
        ws.cleanup()

    def test_tile_runtime_injection_cancels_others_and_reports_first_error(self):
        ws = Workspace()
        path = _slow_scene(ws.dir, delay=0.05)
        with open(path, encoding="utf-8") as handle:
            scene = json.load(handle)
        scene["fault_injections"] = [{"type": "tile_runtime_error", "tile_index": 3}]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(scene, handle)
        result = render_scene(path, install_signals=False)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error"]["error"]["code"], "E_TILE_RUNTIME")
        self.assertEqual(result["report"]["failed_tile"], 3)
        self.assertNotIn("output", result["report"])
        ws.cleanup()

    def test_parameter_injections_return_business_codes(self):
        cases = [
            ([{"type": "zero_radius"}], "E_RADIUS"),
            ([{"type": "non_finite_normal", "point_index": 0}], "E_FINITE"),
            ([{"type": "out_of_bounds_tile"}], "E_TILE"),
            ([{"type": "unknown_thing"}], "E_INJECT"),
        ]
        for injections, code in cases:
            ws = Workspace()
            path = ws.write()
            with open(path, encoding="utf-8") as handle:
                scene = json.load(handle)
            scene["fault_injections"] = injections
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(scene, handle)
            from pointcloud_engine.errors import EngineError

            try:
                render_scene(path, install_signals=False)
            except EngineError as exc:
                self.assertEqual(exc.code, code, msg=str(injections))
            else:
                self.fail(f"expected {code} for {injections}")
            ws.cleanup()


if __name__ == "__main__":
    unittest.main()
