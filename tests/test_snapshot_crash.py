from __future__ import annotations

import json
import os
import subprocess
import sys
import hashlib
from pathlib import Path

from mmfm.renderer import Renderer
from mmfm.snapshot import SnapshotStore


def test_hard_kill_after_commit_then_resume_matches_full(repo_root, workspace, tmp_path):
    crashed_snapshot = tmp_path / "crashed"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src")
    env["MMFM_FAULT_AFTER_COMMIT"] = "3"
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "mmfm.cli",
            "render",
            "--config",
            str(repo_root / "configs" / "render.json"),
            "--conditions",
            str(repo_root / "samples" / "conditions.json"),
            "--snapshot-dir",
            str(crashed_snapshot),
            "--budget",
            "100",
        ],
        cwd=repo_root,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 137
    store = SnapshotStore(crashed_snapshot)
    committed = store.load_committed()
    assert committed is not None
    assert committed.state["pending"]
    assert committed.state["workspace"]
    # An uncommitted temporary generation is ignored and reported with a code.
    orphan = crashed_snapshot / ".tmp-gen-999999-deadbeef"
    orphan.mkdir()
    report = store.uncommitted_report()
    assert any(item["code"] == "E_SNAPSHOT_UNCOMMITTED" for item in report["uncommitted"])

    full = Renderer(workspace, tmp_path / "full", budget_override=100).render()
    resumed = Renderer(
        type(workspace).from_serialised(committed.state["workspace"]),
        crashed_snapshot,
        budget_override=100,
    ).render()
    assert resumed["code"] == "OK"
    assert [frame["pixel_digest"] for frame in resumed["frames"]] == [
        frame["pixel_digest"] for frame in full["frames"]
    ]
    assert [frame["png_sha256"] for frame in resumed["frames"]] == [
        frame["png_sha256"] for frame in full["frames"]
    ]
    assert store.load_committed().state["pending"] == []


def test_snapshot_hashes_are_generation_bound(workspace, tmp_path):
    snapshot = tmp_path / "snap"
    result = Renderer(workspace, snapshot, budget_override=100).render()
    store = SnapshotStore(snapshot)
    committed = store.load_committed()
    assert committed is not None
    pointer = json.loads((snapshot / "COMMITTED").read_text())
    manifest = json.loads((snapshot / committed.generation / "manifest.json").read_text())
    manifest_bytes = (snapshot / committed.generation / "manifest.json").read_bytes()
    assert pointer["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert pointer["generation"] == result["generation"]
    assert manifest["config_hash"] == result["config_hash"]
    assert manifest["input_hash"] == result["input_hash"]
    assert manifest["state_sha256"] == committed.manifest["state_sha256"]
