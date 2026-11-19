"""Immutable diagnostic snapshot construction and local WAV/projection exports."""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

from .config import (
    CHANNEL_SPECS,
    MAX_ORDER_BOUNDS_DECLARATION,
    SPHERICAL_COVERAGE_DECLARATION,
    WALL_IDS,
)
from .errors import E_DEPENDENCY_STALE, fail
from .hashing import canonical_json_bytes, sha256_bytes
from .projection import assemble_preview_png, rgb_from_tile_bytes, tile_layout
from .rir import pcm16_wav_bytes, read_pcm16_wav_bytes
from .store import WorkspaceStore


def export_rir_bundle(store: WorkspaceStore, scene: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    nodes = state["nodes"]
    channel_records = []
    channel_arrays = []
    for channel_id in range(4):
        node = nodes.get(f"rir:c{channel_id}")
        if node is None:
            fail(E_DEPENDENCY_STALE, "RIR channel node is missing", channel_id=channel_id)
        data = store.read_blob(node["artifact_hash"], node_type="rir")
        samples, sample_rate, channels = read_pcm16_wav_bytes(data)
        if sample_rate != 48000 or channels != 1:
            fail(E_DEPENDENCY_STALE, "cached RIR channel is not a 48 kHz mono PCM artifact", channel_id=channel_id)
        channel_arrays.append(samples)
        channel_records.append(
            {
                "channel_id": channel_id,
                "node_id": node["id"],
                "content_hash": node["content_hash"],
                "artifact_hash": node["artifact_hash"],
                "peak_sample": node["payload"].get("peak_sample"),
                "first_arrival_sample": node["payload"].get("first_arrival_sample"),
                "walls_with_paths": node["payload"].get("walls_with_paths", []),
            }
        )
    if any(array.shape[0] != channel_arrays[0].shape[0] for array in channel_arrays):
        fail(E_DEPENDENCY_STALE, "RIR channels have inconsistent frame counts")
    interleaved = np.hstack(channel_arrays)
    wav_bytes = pcm16_wav_bytes(interleaved)
    decoded, sample_rate, channels = read_pcm16_wav_bytes(wav_bytes)
    if sample_rate != 48000 or channels != 4 or not np.array_equal(decoded, interleaved):
        fail(E_DEPENDENCY_STALE, "four-channel RIR export failed readback verification")
    wav_path = store.export_output("room_rir_48k_16bit_4ch.wav", wav_bytes)
    bundle = {
        "path": str(wav_path),
        "sample_rate": sample_rate,
        "bit_depth": 16,
        "channels": channels,
        "frames": int(interleaved.shape[0]),
        "sha256": sha256_bytes(wav_bytes),
        "channels_meta": channel_records,
        "scene_revision": int(scene["revision"]),
    }
    store.export_output("rir_metadata.json", canonical_json_bytes(bundle))
    return bundle


def publish_snapshot(
    store: WorkspaceStore,
    scene: dict[str, Any],
    versions: dict[str, int],
    state: dict[str, Any],
) -> tuple[dict[str, Any], str, str]:
    nodes = state["nodes"]
    tile_rgb: dict[str, np.ndarray] = {}
    artifacts: dict[str, bytes] = {}
    for node_id in sorted(nodes):
        node = nodes[node_id]
        artifact = store.read_blob(node["artifact_hash"], node_type=node["type"])
        artifacts[node_id] = artifact
        if node["type"] == "tile":
            tile_spec = _tile_spec_by_id(scene, node_id)
            tile_rgb[tile_spec.tile_id] = rgb_from_tile_bytes(artifact, tile_spec)
    preview_png = assemble_preview_png(scene, tile_rgb)
    preview_path = store.export_output("equirectangular_projection.png", preview_png)
    rir_bundle = export_rir_bundle(store, scene, state)
    build = state.get("build", {})
    statuses = build.get("statuses", {})
    reasons = build.get("reasons", {})
    snapshot_nodes = []
    for node_id in sorted(nodes):
        node = nodes[node_id]
        entry = {
            "id": node_id,
            "type": node["type"],
            "content_hash": node["content_hash"],
            "artifact_hash": node["artifact_hash"],
            "dependencies": node.get("dependencies", []),
            "input_hashes": node.get("input_hashes", {}),
            "param_stamps": node.get("param_stamps", {}),
            "trigger_keys": node.get("trigger_keys", []),
            "computed_revision": node.get("computed_revision"),
            "last_reuse_source_revision": node.get("last_reuse_source_revision"),
            "status": statuses.get(node_id, node.get("last_build_status", "reused")),
            "reason": reasons.get(node_id, {}),
            "payload": node.get("payload", {}),
        }
        if node["type"] == "tile":
            entry["tile_hash"] = node["content_hash"]
        snapshot_nodes.append(entry)
    tiles = []
    for tile in tile_layout(scene):
        tile_node = nodes[f"tile:r{tile.row}c{tile.col}"]
        tiles.append(
            {
                **tile.as_dict(),
                "tile_hash": tile_node["content_hash"],
                "status": statuses.get(tile_node["id"], "reused"),
            }
        )
    snapshot = {
        "schema": 1,
        "workbench": "room-acoustic-incremental-diagnostics",
        "revision": int(scene["revision"]),
        "scene_hash": state["scene_hash"],
        "scene": scene,
        "versions": versions,
        "build": build,
        "width": int(scene["grid"]["width"]),
        "height": int(scene["grid"]["height"]),
        "tile_rows": int(scene["grid"]["tile_rows"]),
        "tile_cols": int(scene["grid"]["tile_cols"]),
        "projection_bounds": [
            scene["grid"]["lon_min"],
            scene["grid"]["lon_max"],
            scene["grid"]["lat_min"],
            scene["grid"]["lat_max"],
        ],
        "tiles": tiles,
        "nodes": snapshot_nodes,
        "edges": state.get("edges", []),
        "parameter_keys": sorted(versions),
        "declarations": {
            "spherical_coverage": SPHERICAL_COVERAGE_DECLARATION,
            "truncation_order_bound": MAX_ORDER_BOUNDS_DECLARATION,
            "tile_grid": "fixed pixel-aligned equirectangular blocks",
            "walls": list(WALL_IDS),
        },
        "rir_bundle": rir_bundle,
        "projection": {
            "preview_png_base64": base64.b64encode(preview_png).decode("ascii"),
            "preview_sha256": sha256_bytes(preview_png),
            "preview_path": str(preview_path),
            "tone_mapping": "10*log10(E+1e-12), fixed [-100,-10] dB palette",
        },
        "channel_specs": [
            {
                "channel_id": spec.channel_id,
                "source_id": spec.source_id,
                "azimuth_deg": spec.azimuth_deg,
                "elevation_deg": spec.elevation_deg,
                "aperture_deg": spec.aperture_deg,
                "support_deg": spec.support_deg,
            }
            for spec in CHANNEL_SPECS
        ],
    }
    snapshot_name, snapshot_hash = store.save_snapshot(snapshot)
    return snapshot, snapshot_name, snapshot_hash


def manifest_from_state(state: dict[str, Any]) -> bytes:
    nodes = state["nodes"]
    payload = {
        "schema": 1,
        "revision": state["revision"],
        "scene_hash": state["scene_hash"],
        "nodes": [
            {
                "id": node_id,
                "type": nodes[node_id]["type"],
                "content_hash": nodes[node_id]["content_hash"],
                "artifact_hash": nodes[node_id]["artifact_hash"],
            }
            for node_id in sorted(nodes)
        ],
        "edges": sorted(
            (
                {
                    "source": edge["source"],
                    "target": edge["target"],
                    "kind": edge["kind"],
                    "trigger_keys": edge["trigger_keys"],
                }
                for edge in state.get("edges", [])
            ),
            key=lambda edge: (edge["source"], edge["target"], edge["kind"]),
        ),
    }
    return canonical_json_bytes(payload)


def _tile_spec_by_id(scene: dict[str, Any], node_id: str):
    if not node_id.startswith("tile:r"):
        raise KeyError(node_id)
    for tile in tile_layout(scene):
        if tile.tile_id == node_id:
            return tile
    raise KeyError(node_id)
