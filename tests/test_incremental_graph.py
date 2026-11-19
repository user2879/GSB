from __future__ import annotations

from pathlib import Path

import numpy as np

from acoustic_workbench.engine import IncrementalEngine
from acoustic_workbench.rir import read_pcm16_wav_bytes
from acoustic_workbench.store import WorkspaceStore


def _full_engine_from(engine: IncrementalEngine, root: Path) -> IncrementalEngine:
    store = WorkspaceStore(root)
    store.ensure()
    store.save_scene_document(engine.scene, engine.versions)
    return IncrementalEngine.open(root)


def test_wall_absorption_recomputes_only_path_channels(workspace: Path, tmp_path: Path) -> None:
    engine = IncrementalEngine.open(workspace)
    channels_with_xmax = {
        node["payload"]["channel_id"]
        for node in engine.state["nodes"].values()
        if node["type"] == "rir" and "xmax" in node["payload"]["walls_with_paths"]
    }
    assert channels_with_xmax
    old_hashes = {node_id: node["content_hash"] for node_id, node in engine.state["nodes"].items()}
    old_samples = {
        channel_id: read_pcm16_wav_bytes(
            engine.store.read_blob(
                engine.state["nodes"][f"rir:c{channel_id}"]["artifact_hash"],
                node_type="rir",
            )
        )[0]
        for channel_id in range(4)
    }
    report = engine.submit({"wall.xmax.alpha": 0.36}, expected_revision=1)
    recomputed_rir = {node for node in report["recomputed_nodes"] if node.startswith("rir:")}
    assert recomputed_rir == {f"rir:c{channel_id}" for channel_id in channels_with_xmax}
    for channel_id in range(4):
        node_id = f"rir:c{channel_id}"
        if channel_id in channels_with_xmax:
            assert node_id in report["recomputed_nodes"]
        else:
            assert node_id in report["reused_nodes"]
            assert engine.state["nodes"][node_id]["content_hash"] == old_hashes[node_id]
            new_samples = read_pcm16_wav_bytes(
                engine.store.read_blob(engine.state["nodes"][node_id]["artifact_hash"], node_type="rir")
            )[0]
            assert np.array_equal(new_samples, old_samples[channel_id])
    reused_tiles = [node for node in report["reused_nodes"] if node.startswith("tile:")]
    recomputed_tiles = [node for node in report["recomputed_nodes"] if node.startswith("tile:")]
    assert reused_tiles and recomputed_tiles
    for tile_id in reused_tiles:
        assert engine.state["nodes"][tile_id]["content_hash"] == old_hashes[tile_id]
    full = _full_engine_from(engine, tmp_path / "full-wall")
    assert engine.manifest_bytes() == full.manifest_bytes()
    for tile_id in reused_tiles:
        assert engine.state["nodes"][tile_id]["content_hash"] == full.state["nodes"][tile_id]["content_hash"]


def test_source_position_recomputes_source_channels_and_partial_tiles(workspace: Path, tmp_path: Path) -> None:
    engine = IncrementalEngine.open(workspace)
    source_channels = {
        node["payload"]["channel_id"]
        for node in engine.state["nodes"].values()
        if node["type"] == "rir" and node["payload"]["source_id"] == "s0"
    }
    assert source_channels == {0, 1}
    old_hashes = {node_id: node["content_hash"] for node_id, node in engine.state["nodes"].items()}
    report = engine.submit({"source.s0.position": [1.12, 0.92, 1.27]}, expected_revision=1)
    recomputed_rir = {node for node in report["recomputed_nodes"] if node.startswith("rir:")}
    assert recomputed_rir == {f"rir:c{channel_id}" for channel_id in source_channels}
    for channel_id in (2, 3):
        assert engine.state["nodes"][f"rir:c{channel_id}"]["content_hash"] == old_hashes[f"rir:c{channel_id}"]
    assert any(node.startswith("tile:") for node in report["reused_nodes"])
    assert any(node.startswith("tile:") for node in report["recomputed_nodes"])
    full = _full_engine_from(engine, tmp_path / "full-source")
    assert engine.manifest_bytes() == full.manifest_bytes()


def test_order_and_density_invalidate_sh_downstream_but_not_rir(workspace: Path, tmp_path: Path) -> None:
    engine = IncrementalEngine.open(workspace)
    rir_hashes = {f"rir:c{i}": engine.state["nodes"][f"rir:c{i}"]["content_hash"] for i in range(4)}
    report = engine.submit({"receiver.sample_density": 192, "reconstruction.order": 4}, expected_revision=1)
    assert not [node for node in report["recomputed_nodes"] if node.startswith("rir:")]
    assert all(f"rir:c{i}" in report["reused_nodes"] for i in range(4))
    assert all(f"sh:c{i}" in report["recomputed_nodes"] for i in range(4))
    assert all(f"rir:c{i}" in report["reused_nodes"] for i in range(4))
    for channel_id in range(4):
        node_id = f"rir:c{channel_id}"
        assert engine.state["nodes"][node_id]["content_hash"] == rir_hashes[node_id]
        payload = engine.state["nodes"][f"sh:c{channel_id}"]["payload"]
        assert payload["sample_density"] == 192
        assert payload["truncation_order"] == 4
        assert payload["order_bound"] >= 4
        assert payload["coverage"].startswith("full sphere")
    full = _full_engine_from(engine, tmp_path / "full-order")
    assert engine.manifest_bytes() == full.manifest_bytes()
    assert all(
        engine.state["nodes"][node_id]["content_hash"] == full.state["nodes"][node_id]["content_hash"]
        for node_id in engine.state["nodes"]
        if node_id.startswith("tile:")
    )


def test_restart_uses_version_stamps_to_reconcile_external_scene_update(workspace: Path) -> None:
    store = WorkspaceStore(workspace)
    engine = IncrementalEngine.open(workspace)
    document = store.load_scene_document()
    from acoustic_workbench.scene import apply_patch

    new_scene, new_versions, changed = apply_patch(
        document["scene"],
        document["versions"],
        {"wall.xmax.alpha": 0.41},
        revision=2,
    )
    store.save_scene_document(new_scene, new_versions)
    reopened = IncrementalEngine.open(workspace)
    assert reopened.scene["revision"] == 2
    assert changed == ["wall.xmax.alpha"]
    report = reopened.state["build"]
    assert report["closure_size"] > 0
    assert any(node.startswith("rir:") for node in report["recomputed_nodes"])
    assert report["reused_count"] > 0
