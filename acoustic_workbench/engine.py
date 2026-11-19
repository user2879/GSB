"""Incremental dependency graph engine for RIR, spherical, energy, and tile nodes."""

from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .config import CHANNEL_SPECS, sph_order_bound
from .diagnostics import manifest_from_state, publish_snapshot
from .errors import E_SNAPSHOT_CONFLICT, fail
from .hashing import locked
from .projection import (
    TileSpec,
    compose_energy_block,
    energy_to_tile_bytes,
    grid_geometry,
    tile_layout,
    tile_support_channels,
)
from .rir import read_pcm16_wav_bytes, rir_param_keys, synthesize_channel
from .scene import (
    apply_patch,
    default_scene,
    flatten_params,
    initial_versions,
    scene_hash,
    validate_scene,
)
from .spherical import coefficients_from_bytes, coefficients_to_bytes, fit_channel
from .store import WorkspaceStore, make_node_record


ARTIFACT_EDGE = "artifact"
PARAMETER_EDGE = "parameter"
GRID_PARAM_KEYS = tuple(
    f"grid.{key}" for key in ("width", "height", "tile_rows", "tile_cols", "lon_min", "lon_max", "lat_min", "lat_max")
)


def rir_node_id(channel_id: int) -> str:
    return f"rir:c{channel_id}"


def sh_node_id(channel_id: int) -> str:
    return f"sh:c{channel_id}"


def energy_node_id(tile: TileSpec) -> str:
    return f"energy:r{tile.row}c{tile.col}"


def tile_node_id(tile: TileSpec) -> str:
    return f"tile:r{tile.row}c{tile.col}"


class IncrementalEngine:
    def __init__(self, workspace: Path | str):
        self.store = WorkspaceStore(workspace)
        self.scene: dict[str, Any] = {}
        self.versions: dict[str, int] = {}
        self.state: dict[str, Any] = {}
        self.last_snapshot_name = ""
        self.last_snapshot_hash = ""

    @classmethod
    def open(cls, workspace: Path | str) -> "IncrementalEngine":
        engine = cls(workspace)
        engine._initialize()
        return engine

    def _initialize(self) -> None:
        self.store.ensure()
        document = self.store.load_scene_document()
        if document is None:
            scene = default_scene(revision=1)
            versions = initial_versions(scene, revision=1)
            self.store.save_scene_document(scene, versions)
        else:
            scene = document["scene"]
            versions = _normalize_versions(scene, document["versions"])
        validate_scene(scene)
        state = self.store.load_state()
        if state is not None:
            self.store.verify_state_artifacts(state)
        needs_build = (
            state is None
            or state.get("scene_hash") != scene_hash(scene)
            or int(state.get("revision", -1)) != int(scene["revision"])
            or state.get("versions") != versions
        )
        if needs_build:
            self._build(scene, versions, base_state=state, force_full=state is None)
        else:
            self.scene = scene
            self.versions = versions
            self.state = state
            self._ensure_snapshot()

    def _ensure_snapshot(self) -> None:
        try:
            snapshot, snapshot_hash = self.store.load_latest_snapshot()
        except Exception:
            snapshot = None
            snapshot_hash = ""
        if (
            snapshot is not None
            and int(snapshot.get("revision", -1)) == int(self.state.get("revision", -2))
            and snapshot.get("scene_hash") == self.state.get("scene_hash")
        ):
            self.last_snapshot_hash = snapshot_hash
            return
        _snapshot, name, snapshot_hash = publish_snapshot(self.store, self.scene, self.versions, self.state)
        self.last_snapshot_name = name
        self.last_snapshot_hash = snapshot_hash

    def submit(
        self,
        patch: dict[str, Any],
        *,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        with locked(self.store.lock_path):
            document = self.store.load_scene_document()
            if document is None:
                scene = default_scene(revision=1)
                versions = initial_versions(scene, revision=1)
                self.store.save_scene_document(scene, versions)
            else:
                scene = document["scene"]
                versions = _normalize_versions(scene, document["versions"])
            if expected_revision is not None and int(expected_revision) != int(scene["revision"]):
                fail(
                    E_SNAPSHOT_CONFLICT,
                    "parameter submission used a stale revision token",
                    expected_revision=int(expected_revision),
                    current_revision=int(scene["revision"]),
                )
            next_revision = int(scene["revision"]) + 1
            old_scene = copy.deepcopy(scene)
            old_versions = dict(versions)
            new_scene, new_versions, changed = apply_patch(
                scene,
                versions,
                patch,
                revision=next_revision,
            )
            base_state = self.store.load_state()
            if base_state is not None:
                self.store.verify_state_artifacts(base_state)
            self.store.save_scene_document(new_scene, new_versions)
            try:
                self._build(
                    new_scene,
                    new_versions,
                    base_state=base_state,
                    force_full=False,
                    changed_keys=changed,
                )
            except BaseException:
                self.store.save_scene_document(old_scene, old_versions)
                raise
            return copy.deepcopy(self.state["build"])

    def rebuild_full(self) -> dict[str, Any]:
        self._build(self.scene, self.versions, base_state=self.state, force_full=True)
        return copy.deepcopy(self.state["build"])

    def manifest_bytes(self) -> bytes:
        return manifest_from_state(self.state)

    def _build(
        self,
        scene: dict[str, Any],
        versions: dict[str, int],
        *,
        base_state: dict[str, Any] | None,
        force_full: bool,
        changed_keys: list[str] | None = None,
    ) -> None:
        started = time.perf_counter()
        validate_scene(scene)
        versions = _normalize_versions(scene, versions)
        tile_specs = tile_layout(scene)
        expected_ids = _expected_node_ids(tile_specs)
        if base_state is None:
            artifacts: dict[str, bytes] = {}
            old_nodes: dict[str, Any] = {}
            old_edges: list[dict[str, Any]] = []
        else:
            artifacts = self.store.verify_state_artifacts(base_state)
            old_nodes = base_state.get("nodes", {})
            old_edges = base_state.get("edges", [])
        dirty, seed_reasons, closure_reasons = _compute_invalidation(
            scene=scene,
            versions=versions,
            old_nodes=old_nodes,
            old_edges=old_edges,
            old_scene=(base_state or {}).get("scene"),
            expected_ids=expected_ids,
            force_full=force_full or base_state is None,
        )
        nodes: dict[str, Any] = {}
        statuses: dict[str, str] = {}
        reasons: dict[str, Any] = {}
        stage_ms = {"rir": 0.0, "sph": 0.0, "energy": 0.0, "tile": 0.0}

        for channel_id in range(len(CHANNEL_SPECS)):
            node_id = rir_node_id(channel_id)
            if node_id in dirty:
                stage_started = time.perf_counter()
                rir = synthesize_channel(scene, channel_id)
                payload = copy.deepcopy(rir.metadata)
                artifact = rir.wav_bytes
                artifact_hash = self.store.write_blob(artifact)
                stamps = {key: versions[key] for key in rir_param_keys(payload)}
                node = make_node_record(
                    node_id=node_id,
                    node_type="rir",
                    payload=payload,
                    artifact=artifact,
                    artifact_hash=artifact_hash,
                    dependencies=[],
                    input_hashes={},
                    param_stamps=stamps,
                    trigger_keys=list(stamps),
                    revision=int(scene["revision"]),
                )
                nodes[node_id] = node
                artifacts[node_id] = artifact
                statuses[node_id] = "recomputed"
                reasons[node_id] = closure_reasons.get(node_id, {"kind": "recomputed"})
                stage_ms["rir"] += (time.perf_counter() - stage_started) * 1000.0
            else:
                node = _reuse_node(old_nodes, artifacts, node_id)
                nodes[node_id] = node
                statuses[node_id] = "reused"
                reasons[node_id] = {
                    "kind": "cache_hit",
                    "reused_from_revision": int(node.get("computed_revision", 0)),
                }

        sample_density = int(scene["receiver"]["sample_density"])
        order = int(scene["reconstruction"]["order"])
        bound = sph_order_bound(sample_density)
        if order > bound:
            from .errors import E_SPH_ORDER

            fail(E_SPH_ORDER, "spherical-harmonic order exceeds sampling-density bound", order=order, bound=bound)

        for channel_id in range(len(CHANNEL_SPECS)):
            node_id = sh_node_id(channel_id)
            rir_id = rir_node_id(channel_id)
            rir_node = nodes[rir_id]
            if node_id in dirty:
                stage_started = time.perf_counter()
                coefficients, fit_metadata = fit_channel(rir_node["payload"], sample_density, order)
                artifact = coefficients_to_bytes(coefficients)
                artifact_hash = self.store.write_blob(artifact)
                payload = {
                    **fit_metadata,
                    "coefficient_order": order,
                    "rir_content_hash": rir_node["content_hash"],
                }
                local_keys = ("receiver.sample_density", "reconstruction.order")
                stamps = {key: versions[key] for key in local_keys}
                trigger_keys = sorted(set(stamps) | set(rir_node["trigger_keys"]))
                node = make_node_record(
                    node_id=node_id,
                    node_type="sph",
                    payload=payload,
                    artifact=artifact,
                    artifact_hash=artifact_hash,
                    dependencies=[rir_id],
                    input_hashes={rir_id: rir_node["content_hash"]},
                    param_stamps=stamps,
                    trigger_keys=trigger_keys,
                    revision=int(scene["revision"]),
                )
                nodes[node_id] = node
                artifacts[node_id] = artifact
                statuses[node_id] = "recomputed"
                reasons[node_id] = closure_reasons.get(node_id, {"kind": "recomputed"})
                stage_ms["sph"] += (time.perf_counter() - stage_started) * 1000.0
            else:
                node = _reuse_node(old_nodes, artifacts, node_id)
                nodes[node_id] = node
                statuses[node_id] = "reused"
                reasons[node_id] = {
                    "kind": "cache_hit",
                    "reused_from_revision": int(node.get("computed_revision", 0)),
                }

        directions, _lon, _lat = grid_geometry(scene)
        support_map = tile_support_channels(scene, len(CHANNEL_SPECS), directions)
        grid_stamps = {key: versions[key] for key in GRID_PARAM_KEYS}

        for tile in tile_specs:
            node_id = energy_node_id(tile)
            if node_id in dirty:
                stage_started = time.perf_counter()
                support_channels = support_map[tile.tile_id]
                channel_fields = []
                input_hashes: dict[str, str] = {}
                sh_trigger_keys: set[str] = set()
                for channel_id in support_channels:
                    sh_id = sh_node_id(channel_id)
                    sh_node = nodes[sh_id]
                    coefficients = coefficients_from_bytes(artifacts[sh_id])
                    channel_fields.append((channel_id, coefficients, int(sh_node["payload"]["truncation_order"])))
                    input_hashes[sh_id] = sh_node["content_hash"]
                    sh_trigger_keys.update(sh_node["trigger_keys"])
                energy = compose_energy_block(scene, tile, channel_fields, directions=directions)
                artifact = np.asarray(energy, dtype="<f8").tobytes(order="C")
                artifact_hash = self.store.write_blob(artifact)
                payload = {
                    "tile": tile.as_dict(),
                    "shape": [int(energy.shape[0]), int(energy.shape[1])],
                    "support_channels": support_channels,
                    "sh_content_hashes": input_hashes,
                }
                trigger_keys = sorted(sh_trigger_keys | set(GRID_PARAM_KEYS))
                node = make_node_record(
                    node_id=node_id,
                    node_type="energy",
                    payload=payload,
                    artifact=artifact,
                    artifact_hash=artifact_hash,
                    dependencies=sorted(input_hashes),
                    input_hashes=input_hashes,
                    param_stamps=grid_stamps,
                    trigger_keys=trigger_keys,
                    revision=int(scene["revision"]),
                )
                nodes[node_id] = node
                artifacts[node_id] = artifact
                statuses[node_id] = "recomputed"
                reasons[node_id] = closure_reasons.get(node_id, {"kind": "recomputed"})
                stage_ms["energy"] += (time.perf_counter() - stage_started) * 1000.0
            else:
                node = _reuse_node(old_nodes, artifacts, node_id)
                nodes[node_id] = node
                statuses[node_id] = "reused"
                reasons[node_id] = {
                    "kind": "cache_hit",
                    "reused_from_revision": int(node.get("computed_revision", 0)),
                }

        for tile in tile_specs:
            node_id = tile_node_id(tile)
            energy_id = energy_node_id(tile)
            energy_node = nodes[energy_id]
            if node_id in dirty:
                stage_started = time.perf_counter()
                rows, cols = energy_node["payload"]["shape"]
                energy = np.frombuffer(artifacts[energy_id], dtype="<f8").reshape(int(rows), int(cols))
                rgb = energy_to_tile_bytes(energy)
                artifact_hash = self.store.write_blob(rgb)
                payload = {
                    "tile": tile.as_dict(),
                    "energy_node": energy_id,
                    "energy_hash": energy_node["content_hash"],
                }
                node = make_node_record(
                    node_id=node_id,
                    node_type="tile",
                    payload=payload,
                    artifact=rgb,
                    artifact_hash=artifact_hash,
                    dependencies=[energy_id],
                    input_hashes={energy_id: energy_node["content_hash"]},
                    param_stamps=grid_stamps,
                    trigger_keys=list(energy_node["trigger_keys"]),
                    revision=int(scene["revision"]),
                )
                node["tile_hash"] = node["content_hash"]
                nodes[node_id] = node
                artifacts[node_id] = rgb
                statuses[node_id] = "recomputed"
                reasons[node_id] = closure_reasons.get(node_id, {"kind": "recomputed"})
                stage_ms["tile"] += (time.perf_counter() - stage_started) * 1000.0
            else:
                node = _reuse_node(old_nodes, artifacts, node_id)
                node["tile_hash"] = node["content_hash"]
                nodes[node_id] = node
                statuses[node_id] = "reused"
                reasons[node_id] = {
                    "kind": "cache_hit",
                    "reused_from_revision": int(node.get("computed_revision", 0)),
                }

        for node_id, node in nodes.items():
            node["last_build_status"] = statuses[node_id]
        edges = _build_edges(nodes, support_map, tile_specs)
        total_ms = (time.perf_counter() - started) * 1000.0
        report = {
            "revision": int(scene["revision"]),
            "scene_hash": scene_hash(scene),
            "force_full": bool(force_full or base_state is None),
            "changed_keys": sorted(changed_keys or []),
            "closure": sorted(dirty),
            "closure_size": len(dirty),
            "seeds": sorted(seed_reasons),
            "seed_reasons": seed_reasons,
            "reused_nodes": sorted(set(expected_ids) - dirty),
            "recomputed_nodes": sorted(dirty),
            "reused_count": len(set(expected_ids) - dirty),
            "recomputed_count": len(dirty),
            "statuses": statuses,
            "reasons": reasons,
            "timings_ms": {**{key: round(value, 3) for key, value in stage_ms.items()}, "total": round(total_ms, 3)},
        }
        state = {
            "schema": 1,
            "revision": int(scene["revision"]),
            "scene_hash": scene_hash(scene),
            "versions": versions,
            "scene": copy.deepcopy(scene),
            "nodes": nodes,
            "edges": edges,
            "build": report,
        }
        self.store.save_state(state)
        self.scene = copy.deepcopy(scene)
        self.versions = dict(versions)
        self.state = state
        _snapshot, snapshot_name, snapshot_hash = publish_snapshot(self.store, scene, versions, state)
        self.last_snapshot_name = snapshot_name
        self.last_snapshot_hash = snapshot_hash


def _normalize_versions(scene: dict[str, Any], versions: dict[str, int]) -> dict[str, int]:
    flat = flatten_params(scene)
    normalized = {}
    for key in flat:
        value = versions.get(key, 1)
        normalized[key] = int(value) if isinstance(value, int) and value > 0 else 1
    return normalized


def _expected_node_ids(tile_specs: list[TileSpec]) -> set[str]:
    ids = {rir_node_id(channel_id) for channel_id in range(len(CHANNEL_SPECS))}
    ids.update(sh_node_id(channel_id) for channel_id in range(len(CHANNEL_SPECS)))
    ids.update(energy_node_id(tile) for tile in tile_specs)
    ids.update(tile_node_id(tile) for tile in tile_specs)
    return ids


def _reuse_node(old_nodes: dict[str, Any], artifacts: dict[str, bytes], node_id: str) -> dict[str, Any]:
    node = copy.deepcopy(old_nodes[node_id])
    if node_id not in artifacts:
        raise KeyError(node_id)
    node["last_reuse_source_revision"] = int(node.get("computed_revision", 0))
    node["last_build_status"] = "reused"
    return node


def _compute_invalidation(
    *,
    scene: dict[str, Any],
    versions: dict[str, int],
    old_nodes: dict[str, Any],
    old_edges: list[dict[str, Any]],
    old_scene: dict[str, Any] | None,
    expected_ids: set[str],
    force_full: bool,
) -> tuple[set[str], dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    if force_full or not old_nodes:
        dirty = set(expected_ids)
        reason = {"kind": "initial_build" if not old_nodes else "forced_full_rebuild"}
        return dirty, {node_id: dict(reason) for node_id in dirty}, {node_id: dict(reason) for node_id in dirty}

    value_changed: set[str] = set()
    if isinstance(old_scene, dict):
        try:
            old_flat = flatten_params(old_scene)
            new_flat = flatten_params(scene)
        except Exception:
            old_flat = {}
            new_flat = {}
        value_changed = {key for key, value in new_flat.items() if old_flat.get(key) != value}

    dirty: set[str] = set()
    reasons: dict[str, dict[str, Any]] = {}
    for node_id in sorted(expected_ids):
        node = old_nodes.get(node_id)
        if node is None:
            dirty.add(node_id)
            reasons[node_id] = {"kind": "missing_cached_node"}
            continue
        stamps = node.get("param_stamps", {})
        version_changes = sorted(key for key, stamp in stamps.items() if versions.get(key) != stamp)
        value_changes = sorted(key for key in stamps if key in value_changed)
        if version_changes or value_changes:
            dirty.add(node_id)
            reasons[node_id] = {
                "kind": "parameter_change",
                "changed_version_keys": version_changes,
                "changed_value_keys": value_changes,
            }

    adjacency: dict[str, set[str]] = {}
    for edge in old_edges:
        if edge.get("kind") != ARTIFACT_EDGE:
            continue
        source = str(edge.get("source"))
        target = str(edge.get("target"))
        if source in expected_ids and target in expected_ids:
            adjacency.setdefault(source, set()).add(target)

    queue = sorted(dirty)
    propagated: dict[str, set[str]] = {}
    index = 0
    while index < len(queue):
        source = queue[index]
        index += 1
        for target in sorted(adjacency.get(source, ())):
            propagated.setdefault(target, set()).add(source)
            if target not in dirty:
                dirty.add(target)
                queue.append(target)

    for node_id in sorted(propagated):
        entry = reasons.get(node_id, {"kind": "upstream_invalidation"})
        entry["upstream"] = sorted(propagated[node_id])
        if entry.get("kind") == "upstream_invalidation":
            entry["kind"] = "upstream_invalidation"
        reasons[node_id] = entry
    for node_id in sorted(dirty):
        reasons.setdefault(node_id, {"kind": "recomputed"})
    return dirty, {node_id: reasons[node_id] for node_id in sorted(dirty)}, reasons


def _build_edges(
    nodes: dict[str, Any],
    support_map: dict[str, list[int]],
    tile_specs: list[TileSpec],
) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(source: str, target: str, trigger_keys: Iterable[str], kind: str) -> None:
        key = (source, target, kind)
        if key in seen:
            return
        seen.add(key)
        edges.append(
            {
                "source": source,
                "target": target,
                "kind": kind,
                "trigger_keys": sorted(set(str(item) for item in trigger_keys)),
            }
        )

    for channel_id in range(len(CHANNEL_SPECS)):
        rir = nodes[rir_node_id(channel_id)]
        add(rir["id"], sh_node_id(channel_id), rir["trigger_keys"], ARTIFACT_EDGE)
    for tile in tile_specs:
        energy_id = energy_node_id(tile)
        for channel_id in support_map[tile.tile_id]:
            sh = nodes[sh_node_id(channel_id)]
            add(sh["id"], energy_id, sh["trigger_keys"], ARTIFACT_EDGE)
        add(energy_id, tile_node_id(tile), nodes[energy_id]["trigger_keys"], ARTIFACT_EDGE)
    for node_id in sorted(nodes):
        for param_key in sorted(nodes[node_id].get("param_stamps", {})):
            add(f"param:{param_key}", node_id, [param_key], PARAMETER_EDGE)
    return sorted(edges, key=lambda edge: (edge["source"], edge["target"], edge["kind"]))
