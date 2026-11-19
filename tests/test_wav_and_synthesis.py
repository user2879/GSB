from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np

from acoustic_workbench.engine import IncrementalEngine
from acoustic_workbench.rir import check_wav_roundtrip, read_pcm16_wav_bytes, synthesize_channel
from acoustic_workbench.scene import default_scene


def test_wav_roundtrip_is_48k_16bit_4ch(tmp_path: Path) -> None:
    result = check_wav_roundtrip(tmp_path / "probe.wav")
    assert result["sample_rate"] == 48000
    assert result["bit_depth"] == 16
    assert result["channels"] == 4
    assert result["exact"] is True


def test_channel_synthesis_and_four_channel_export(workspace: Path) -> None:
    scene = default_scene()
    first = synthesize_channel(scene, 0)
    second = synthesize_channel(scene, 0)
    assert np.array_equal(first.samples, second.samples)
    assert first.wav_bytes == second.wav_bytes
    decoded, rate, channels = read_pcm16_wav_bytes(first.wav_bytes)
    assert rate == 48000
    assert channels == 1
    assert decoded.shape[0] == 21600
    assert first.metadata["walls_with_paths"]
    assert first.metadata["first_arrival_sample"] is not None
    assert first.metadata["peak_sample"] is not None

    export_path = workspace / "outputs" / "room_rir_48k_16bit_4ch.wav"
    data, rate, channels = read_pcm16_wav_bytes(export_path.read_bytes())
    assert rate == 48000
    assert channels == 4
    assert data.shape == (21600, 4)


def test_full_build_contains_all_graph_layers(workspace: Path) -> None:
    engine = IncrementalEngine.open(workspace)
    node_ids = set(engine.state["nodes"])
    assert {f"rir:c{i}" for i in range(4)} <= node_ids
    assert {f"sh:c{i}" for i in range(4)} <= node_ids
    assert len([node for node in node_ids if node.startswith("energy:")]) == 32
    assert len([node for node in node_ids if node.startswith("tile:")]) == 32
    assert len(node_ids) == 72
    for edge in engine.state["edges"]:
        assert edge["trigger_keys"], edge


def test_restart_is_stable_and_does_not_recompute(baseline_workspace: Path, tmp_path: Path) -> None:
    target = tmp_path / "restart"
    shutil.copytree(baseline_workspace, target)
    first = IncrementalEngine.open(target)
    manifest = first.manifest_bytes()
    state_bytes = (target / "state.json").read_bytes()
    second = IncrementalEngine.open(target)
    assert second.manifest_bytes() == manifest
    assert (target / "state.json").read_bytes() == state_bytes
    assert second.last_snapshot_hash == first.last_snapshot_hash
