from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from vgcover import load_scene, scene_hash, verify_export_manifest, verify_ledger


ROOT = Path(__file__).resolve().parents[1]


def test_example_entry_writes_json_buffers_ledger_and_incremental_receipts(tmp_path):
    out_dir = tmp_path / "demo-output"
    completed = subprocess.run(
        [sys.executable, "examples/run_demo.py", "--out", str(out_dir), "--seed", "987654"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is True
    assert payload["ledger_complete"] is True
    assert payload["incremental_steps"] == 4

    required_files = [
        "scene.json",
        "scene.sha256",
        "ledger.json",
        "ledger.sha256",
        "composite.bgra",
        "manifest.json",
        "summary.json",
        "incremental_demo.json",
        "regions/region_0010/gray.u8",
        "regions/region_0010/coverage_difference.f32",
        "regions/region_0040/layer.bgra",
        "incremental/ledger.json",
        "incremental/composite.bgra",
    ]
    for relative in required_files:
        assert (out_dir / relative).is_file(), relative

    scene = load_scene(out_dir / "scene.json")
    ledger = json.loads((out_dir / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["scene_hash"] == scene_hash(scene)
    assert verify_ledger(scene, ledger)["ok"] is True
    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    assert verify_export_manifest(out_dir, manifest)["ok"] is True

    incremental = json.loads((out_dir / "incremental_demo.json").read_text(encoding="utf-8"))
    for step in incremental["steps"]:
        receipt = step["receipt"]
        assert receipt["actual_matches_closure"] is True
        assert receipt["outside_layer_hashes_preserved"] is True
        assert receipt["outside_ledger_entries_preserved"] is True

    verify_run = subprocess.run(
        [
            sys.executable,
            "-m",
            "vgcover.cli",
            "verify",
            "--scene",
            str(out_dir / "scene.json"),
            "--ledger",
            str(out_dir / "ledger.json"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert verify_run.returncode == 0, verify_run.stderr
    assert json.loads(verify_run.stdout)["ok"] is True

