"""End-to-end acceptance driver for the point-cloud render engine.

Runs deterministic local scenarios that exercise every business requirement:
baseline render + serial equivalence, resume/reuse, version-driven archival,
local point-edit block invalidation, SIGTERM interrupt/resume, hard-kill
recovery, memory cap and data-driven fault injection.

Usage:  python3 acceptance.py [output_dir]
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from pointcloud_engine.engine import render_scene, render_serial_reference  # noqa: E402
from pointcloud_engine.errors import EngineError  # noqa: E402
from pointcloud_engine.pointcloud import (  # noqa: E402
    PointCloud,
    generate_sample_cloud,
    load_point_cloud,
    save_point_cloud,
)


def heading(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def write_scene(workdir: str, **overrides) -> str:
    width = overrides.pop("width", 60)
    height = overrides.pop("height", 40)
    supersample = overrides.pop("supersample", 3)
    tile_width = overrides.pop("tile_width", 30)
    tile_height = overrides.pop("tile_height", 20)
    concurrency = overrides.pop("concurrency", 4)
    scene = {
        "point_cloud": "cloud.pts3",
        "world_scale": 1.0,
        "origin_offset": [0.0, 0.0, 0.0],
        "camera": {
            "position": [0.0, 0.0, 4.0],
            "target": [0.0, 0.0, 0.0],
            "up": [0.0, 1.0, 0.0],
            "vfov_deg": 60.0,
            "aspect": width / height,
        },
        "width": width,
        "height": height,
        "supersample": supersample,
        "point_radius": 0.1,
        "depth_decay": {"kind": "exp", "coefficient": 0.05},
        "tile_width": tile_width,
        "tile_height": tile_height,
        "concurrency": concurrency,
        "memory_budget_bytes": tile_width * tile_height * 8 * concurrency,
        "state_dir": "state",
        "output_dir": "output",
        "near": 1e-6,
        "tile_delay_seconds": 0.0,
        "fault_injections": [],
    }
    scene.update(overrides)
    path = os.path.join(workdir, "scene.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(scene, handle, ensure_ascii=False, indent=2, sort_keys=True)
    return path


def main() -> int:
    base = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "workspace", "acceptance")
    if os.path.exists(base):
        shutil.rmtree(base)
    os.makedirs(base)

    # 1) deterministic sample + baseline render and serial equivalence --------
    heading("1. Deterministic sample, baseline render, serial equivalence")
    workdir = os.path.join(base, "baseline")
    os.makedirs(workdir)
    cloud = generate_sample_cloud(os.path.join(workdir, "cloud.pts3"))
    scene_path = write_scene(workdir)
    print(f"generated {cloud.count} deterministic points -> {workdir}")

    result = render_scene(scene_path, install_signals=False)
    report = result["report"]
    print(f"status={result['status']} tiles={report['tile_count']} "
          f"rendered={len(report['tiles_rendered'])} reused={len(report['tiles_reused'])}")
    print(f"points: total={report['point_count']} rendered={report['points_rendered']} "
          f"behind_near={report['points_behind_near']} culled_frustum={report['points_culled_frustum']}")

    _, _, _, _, serial_gray, serial_depth = render_serial_reference(scene_path)
    par_gray = np.load(os.path.join(workdir, "output", "gray.npy"))
    par_depth = np.load(os.path.join(workdir, "output", "depth.npy"))
    eq = np.array_equal(par_gray, serial_gray) and np.array_equal(par_depth, serial_depth)
    print(f"parallel == serial pixel-for-pixel: {eq}")
    assert eq

    # 2) resume reuse -----------------------------------------------------------
    heading("2. Re-run: version match, all tiles reused, outputs not replaced")
    rerun = render_scene(scene_path, install_signals=False)
    print(f"reused={len(rerun['report']['tiles_reused'])} "
          f"rendered={rerun['report']['tiles_rendered']} "
          f"output_replaced={rerun['report']['output']['write']['replaced']}")
    assert len(rerun["report"]["tiles_reused"]) == report["tile_count"]

    # 3) local point edit -> only affected tiles invalidated -------------------
    heading("3. Local point edit: only footprint-covered tiles are invalidated")
    edit_dir = os.path.join(base, "edit")
    shutil.copytree(workdir, edit_dir)
    edit_cloud = load_point_cloud(os.path.join(edit_dir, "cloud.pts3"))
    edit_cloud.grays[10] = 0.99
    save_point_cloud(os.path.join(edit_dir, "cloud.pts3"), edit_cloud)
    edit_path = os.path.join(edit_dir, "scene.json")
    edit_result = render_scene(edit_path, install_signals=False)
    er = edit_result["report"]
    print(f"invalidated={er['tiles_invalidated']} rerendered={er['tiles_rerendered']} "
          f"reused={len(er['tiles_reused'])}/{er['tile_count']}")
    assert set(er["tiles_invalidated"]) == set(er["tiles_rerendered"])
    assert len(er["tiles_reused"]) == er["tile_count"] - len(er["tiles_invalidated"])

    # 4) version change -> archival full re-render -----------------------------
    heading("4. Camera parameter change: new version archives old tiles")
    with open(scene_path, encoding="utf-8") as handle:
        scene_json = json.load(handle)
    scene_json["camera"]["vfov_deg"] = 75.0
    with open(scene_path, "w", encoding="utf-8") as handle:
        json.dump(scene_json, handle)
    ver_result = render_scene(scene_path, install_signals=False)
    vr = ver_result["report"]
    archived_npz = [p for p in vr["archived_files"] if p.endswith(".npz")]
    print(f"rendered={len(vr['tiles_rendered'])} archived_tiles={len(archived_npz)}")
    assert len(vr["tiles_rendered"]) == vr["tile_count"]
    assert archived_npz

    # 5) SIGTERM interrupt then resume -----------------------------------------
    heading("5. SIGTERM interrupt at tile boundary, then resume to completion")
    sig_dir = os.path.join(base, "interrupt")
    os.makedirs(sig_dir)
    generate_sample_cloud(os.path.join(sig_dir, "cloud.pts3"))
    slow_path = write_scene(sig_dir, concurrency=2, tile_delay_seconds=0.4)
    env = dict(os.environ, PYTHONPATH=ROOT)
    cmd = [sys.executable, "-m", "pointcloud_engine.cli", "render", slow_path]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(0.9)
    proc.send_signal(signal.SIGTERM)
    out, err = proc.communicate(timeout=20)
    print(f"interrupt exit code={proc.returncode} (expected 130)")
    assert proc.returncode == 130, err
    partial = len([n for n in os.listdir(os.path.join(sig_dir, "state")) if n.endswith(".npz")])
    print(f"persisted tiles at interrupt: {partial}")
    resume = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=60)
    resume_json = json.loads(resume.stdout)
    print(f"resume status={resume_json['status']} reused={len(resume_json['report']['tiles_reused'])} "
          f"rendered={len(resume_json['report']['tiles_rendered'])}")
    assert resume.returncode == 0 and resume_json["status"] == "completed"

    # 6) hard kill recovery -----------------------------------------------------
    heading("6. Hard kill (SIGKILL): recover only from valid persisted tiles")
    kill_dir = os.path.join(base, "hardkill")
    os.makedirs(kill_dir)
    generate_sample_cloud(os.path.join(kill_dir, "cloud.pts3"))
    kill_path = write_scene(kill_dir, concurrency=2, tile_delay_seconds=0.4)
    kproc = subprocess.Popen(cmd[:-1] + [kill_path], env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(0.9)
    kproc.kill()
    kproc.wait()
    print("hard-killed process; no interrupted marker guaranteed:",
          not os.path.exists(os.path.join(kill_dir, "state", "interrupted.json")))
    krecovery = subprocess.run(cmd[:-1] + [kill_path], env=env,
                               capture_output=True, text=True, timeout=60)
    print(f"recovery status={json.loads(krecovery.stdout)['status']} rc={krecovery.returncode}")
    assert krecovery.returncode == 0

    # 7) memory cap -------------------------------------------------------------
    heading("7. Memory budget caps in-flight tiles without reducing supersampling")
    mem_dir = os.path.join(base, "memory")
    os.makedirs(mem_dir)
    generate_sample_cloud(os.path.join(mem_dir, "cloud.pts3"))
    mem_path = write_scene(mem_dir, concurrency=8, tile_width=30, tile_height=20)
    with open(mem_path, encoding="utf-8") as handle:
        mem_scene = json.load(handle)
    per_tile = mem_scene["tile_width"] * mem_scene["tile_height"] * 8
    mem_scene["memory_budget_bytes"] = per_tile * 2
    with open(mem_path, "w", encoding="utf-8") as handle:
        json.dump(mem_scene, handle)
    mem_result = render_scene(mem_path, install_signals=False)
    print(f"max_inflight_tiles={mem_result['report']['max_inflight_tiles']} "
          f"grid={mem_result['report']['grid']} supersample unchanged")
    assert mem_result["report"]["max_inflight_tiles"] == 2

    # 8) fault injection --------------------------------------------------------
    heading("8. Data-driven fault injection returns structured E_xxx codes")
    for injection, code in [
        ([{"type": "zero_radius"}], "E_RADIUS"),
        ([{"type": "non_finite_normal", "point_index": 0}], "E_FINITE"),
        ([{"type": "out_of_bounds_tile"}], "E_TILE"),
        ([{"type": "unknown_thing"}], "E_INJECT"),
    ]:
        inj_dir = os.path.join(base, "inject_" + code.lower())
        os.makedirs(inj_dir)
        generate_sample_cloud(os.path.join(inj_dir, "cloud.pts3"))
        inj_path = write_scene(inj_dir, fault_injections=injection)
        try:
            render_scene(inj_path, install_signals=False)
        except EngineError as exc:
            print(f"injection={injection[0]['type']:<20} -> {exc.code} (expected {code})")
            assert exc.code == code
        else:
            raise AssertionError(f"expected {code}")

    heading("ALL ACCEPTANCE SCENARIOS PASSED")
    print(f"artifacts under: {base}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
