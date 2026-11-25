from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from fixedtex.renderer import full_render_serial
from fixedtex.session import Session
from fixedtex.snapshot import SnapshotStore

from .helpers import (
    ROOT,
    SAMPLE_CONFIG_LOW,
    SAMPLE_MATERIAL,
    assert_session_matches_full,
    open_session,
    write_commands,
    write_config,
    write_json,
)


def test_preview_returns_stale_summary_within_budget(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        preview_budget_ms=5,
        commit_interval_commands=99,
        name="preview",
    )
    injection = {
        "layer": 0,
        "stage": "noise_synthesis",
        "delay_ms": 500,
        "code": "E_INJECTED_PREVIEW",
    }
    session = open_session(tmp_path, config_path=config_path, injections=[injection])
    started = time.monotonic()
    result = session.apply_command({"command": "set_layer_param", "layer": 0, "frequency": 2.0})
    elapsed = time.monotonic() - started
    assert elapsed < 0.30
    assert result["preview"]["stale"] is True
    assert result["preview"]["stale_reason"] == "E_PREVIEW_STALE"
    assert "E_PREVIEW_STALE" in result["preview"]["codes"]
    assert result["preview"]["background_render"] is True
    session._join_job()
    assert any(item.get("code") == "E_INJECTED_PREVIEW" for item in session.state.failures)


def test_due_commit_does_not_block_stale_preview(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        preview_budget_ms=5,
        commit_interval_commands=1,
        name="preview_commit",
    )
    injection = {
        "layer": 0,
        "stage": "noise_synthesis",
        "delay_ms": 350,
        "code": "E_INJECTED_COMMIT_PREVIEW",
    }
    state_dir = tmp_path / "state_preview_commit"
    session = Session.open(config_path, SAMPLE_MATERIAL, state_dir, resume=False, injections=[injection])
    started = time.monotonic()
    result = session.apply_command({"command": "set_layer_param", "layer": 0, "frequency": 2.5})
    assert time.monotonic() - started < 0.30
    assert result["preview"]["stale"] is True
    assert result.get("commit_pending") is True
    assert not (state_dir / "CURRENT").exists()
    session._join_job()
    session.commit()
    assert (state_dir / "CURRENT").exists()


def test_partial_budget_snapshot_resume_matches_full(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, source=SAMPLE_CONFIG_LOW, name="partial")
    state_dir = tmp_path / "state_partial"
    writer = Session.open(config_path, SAMPLE_MATERIAL, state_dir, resume=False)
    writer.advance(wait=True)
    assert writer.state.pending
    assert writer.state.partials
    writer.commit()
    snapshot_pending = list(writer.state.pending)

    resumed = Session.open(config_path, SAMPLE_MATERIAL, state_dir, resume=True)
    assert resumed.resume_source.startswith("snapshot:")
    assert len(resumed.state.pending) == len(snapshot_pending)
    resumed.complete()
    full_layers, full_channels, full_composite = full_render_serial(
        resumed.config, resumed.material, resumed.texture
    )
    assert_session_matches_full(resumed, full_layers, full_channels, full_composite)


def test_config_hash_mismatch_forces_rebuild(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, name="before")
    state_dir = tmp_path / "state_hash"
    writer = Session.open(config_path, SAMPLE_MATERIAL, state_dir, resume=False)
    writer.advance(wait=True)
    writer.commit()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    raw["name"] = "after"
    config_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    resumed = Session.open(config_path, SAMPLE_MATERIAL, state_dir, resume=True)
    codes = {item["code"] for item in resumed.issues}
    assert resumed.rebuilt is True
    assert "E_CONFIG_CHANGED_REBUILD" in codes
    assert resumed.config.name == "after"
    assert resumed.state.composite is None


def test_external_texture_replacement_invalidates_resumed_products(tmp_path: Path) -> None:
    texture_path = tmp_path / "texture.png"
    shutil.copy2(ROOT / "samples/base_texture.png", texture_path)
    raw = json.loads((ROOT / "samples/config.json").read_text(encoding="utf-8"))
    raw["texture"] = str(texture_path)
    raw["name"] = "external-texture"
    config_path = tmp_path / "external_config.json"
    config_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    state_dir = tmp_path / "state_external"
    writer = Session.open(config_path, SAMPLE_MATERIAL, state_dir, resume=False)
    writer.advance(wait=True)
    writer.commit()
    shutil.copy2(ROOT / "samples/base_texture_v2.png", texture_path)
    resumed = Session.open(config_path, SAMPLE_MATERIAL, state_dir, resume=True)
    assert any(item["code"] == "E_TEXTURE_EXTERNAL_CHANGE" for item in resumed.issues)
    assert resumed.state.composite is None
    resumed.complete()
    full_layers, full_channels, full_composite = full_render_serial(
        resumed.config, resumed.material, resumed.texture
    )
    assert_session_matches_full(resumed, full_layers, full_channels, full_composite)


def test_corrupt_snapshot_artifact_is_detected(tmp_path: Path) -> None:
    config_path = write_config(tmp_path, name="corrupt")
    state_dir = tmp_path / "state_corrupt"
    writer = Session.open(config_path, SAMPLE_MATERIAL, state_dir, resume=False)
    writer.advance(wait=True)
    writer.commit()
    store = SnapshotStore(state_dir)
    current = json.loads((state_dir / "CURRENT").read_text(encoding="utf-8"))
    manifest_path = state_dir / "snapshots" / current["snapshot"] / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    victim = state_dir / "snapshots" / current["snapshot"] / next(iter(manifest["artifacts"]))
    victim.write_bytes(victim.read_bytes() + b"corruption")
    loaded, issues = store.load()
    assert loaded is None
    assert issues and issues[0]["code"] == "E_SNAPSHOT_CORRUPT"


def test_sigkill_before_commit_keeps_previous_snapshot(tmp_path: Path) -> None:
    config_path = write_config(
        tmp_path,
        preview_budget_ms=50,
        commit_interval_commands=99,
        name="kill",
    )
    state_dir = tmp_path / "state_kill"
    writer = Session.open(config_path, SAMPLE_MATERIAL, state_dir, resume=False)
    writer.advance(wait=True)
    writer.commit()
    version_before = (state_dir / "VERSION").read_text(encoding="utf-8")

    commands_path = write_commands(
        tmp_path / "kill_commands.jsonl",
        [{"command": "set_layer_param", "layer": 0, "frequency": 3.75}],
    )
    injection_path = write_json(
        tmp_path / "kill_injection.json",
        {
            "injections": [
                {
                    "layer": 0,
                    "stage": "noise_synthesis",
                    "delay_ms": 4000,
                    "code": "E_INJECTED_SLOW_KILL",
                }
            ]
        },
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT)
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "fixedtex",
            "run",
            "--config",
            str(config_path),
            "--material",
            str(SAMPLE_MATERIAL),
            "--commands",
            str(commands_path),
            "--injections",
            str(injection_path),
            "--state-dir",
            str(state_dir),
            "--no-drain",
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        time.sleep(0.9)
        assert process.poll() is None
        process.kill()
        process.wait(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
    assert (state_dir / "VERSION").read_text(encoding="utf-8") == version_before

    recovered = Session.open(config_path, SAMPLE_MATERIAL, state_dir, resume=True)
    assert recovered.resume_source.startswith("snapshot:")
    recovered.complete()
    full_layers, full_channels, full_composite = full_render_serial(
        recovered.config, recovered.material, recovered.texture
    )
    assert_session_matches_full(recovered, full_layers, full_channels, full_composite)
    assert np.array_equal(recovered.state.composite.array, full_composite.array)
