"""Incremental, versioned differentiable rendering pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .errors import DiffRenderError, E_CLOSURE_MISMATCH, E_DEPENDENCY_STALE, E_SCENE_INVALID
from .graph import DependencyGraph, Node
from .hashing import (
    b64decode,
    b64encode,
    canonical_json_bytes,
    digest_sequence,
    pack_floats,
    pack_ints,
    sha256_bytes,
    unpack_floats,
    unpack_ints,
)
from .renderer import (
    GradientResult,
    Projection,
    RenderResult,
    TileResult,
    _projection_bundles,
    compose_image,
    compare_gradient_results,
    compare_render_results,
    grad_aggregate_node_key,
    grad_chunk_node_key,
    gradient_full,
    gradient_hash,
    gradient_tile,
    image_hash,
    mean_squared_loss,
    param_node_key,
    project_sphere,
    raster_manifest_hash,
    raster_node_key,
    render_full,
    render_tile,
    shade_manifest_hash,
    shade_node_key,
    shade_tile,
    tile_key,
    tile_layout,
)
from .scene import Scene


@dataclass(frozen=True)
class ToleranceConfig:
    sphere_geometry: float = 1.0e-9
    sphere_material: float = 1.0e-9
    light: float = 1.0e-9
    camera: float = 1.0e-9

    def for_key(self, key: str) -> float:
        if key.startswith("sphere.") and key.rsplit(".", 1)[-1] in ("cx", "cy", "cz", "radius"):
            return self.sphere_geometry
        if key.startswith("sphere."):
            return self.sphere_material
        if key.startswith("light."):
            return self.light
        if key.startswith("camera."):
            return self.camera
        raise DiffRenderError(E_SCENE_INVALID, f"no tolerance defined for {key!r}")

    def to_dict(self) -> dict[str, float]:
        return {
            "sphere_geometry": self.sphere_geometry,
            "sphere_material": self.sphere_material,
            "light": self.light,
            "camera": self.camera,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ToleranceConfig":
        return cls(
            sphere_geometry=float(data.get("sphere_geometry", 1.0e-9)),
            sphere_material=float(data.get("sphere_material", 1.0e-9)),
            light=float(data.get("light", 1.0e-9)),
            camera=float(data.get("camera", 1.0e-9)),
        )


STATE_SCHEMA = "diffrender-state-v1"


class IncrementalPipeline:
    """A cache-aware renderer with an explicit parameter dependency graph."""

    def __init__(
        self,
        scene: Scene,
        target: Sequence[float],
        tolerances: ToleranceConfig | None = None,
    ) -> None:
        if len(target) != scene.width * scene.height:
            raise DiffRenderError(E_SCENE_INVALID, "target image dimension mismatch")
        self.scene = scene
        self.target = tuple(float(v) for v in target)
        self.tolerances = tolerances or ToleranceConfig()
        self._param_versions = {key: 0 for key in scene.param_keys()}
        self._node_versions: dict[str, int] = {}
        self._ledger: list[dict[str, Any]] = []
        self.revision = 0
        self.snapshot_version = 0
        self._tiles: dict[tuple[int, int], TileResult] = {}
        self._grad_chunks: dict[str, dict[tuple[int, int], tuple[float, ...]]] = {
            key: {} for key in scene.param_keys()
        }
        self._build_initial_caches()
        self._build_graph()

    def _build_initial_caches(self) -> None:
        projections = tuple(
            project_sphere(self.scene, i) for i in range(len(self.scene.spheres))
        )
        bundles = _projection_bundles(self.scene)
        layout = tile_layout(self.scene)
        for ty, tx, *_ in layout:
            self._tiles[(ty, tx)] = render_tile(self.scene, ty, tx, projections)
        for ty, tx, *_ in layout:
            chunks = gradient_tile(
                self.scene, self.target, self._tiles[(ty, tx)], bundles
            )
            for key in self.scene.param_keys():
                self._grad_chunks[key][(ty, tx)] = chunks[key]
        self._image = compose_image(self.scene, self._tiles)
        self._loss = mean_squared_loss(self._image, self.target)

    # ------------------------------------------------------------------
    # State inspection

    @property
    def image(self) -> tuple[float, ...]:
        return self._image

    @property
    def loss(self) -> float:
        return self._loss

    @property
    def graph(self) -> DependencyGraph:
        return self._graph

    @property
    def ledger(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._ledger)

    @property
    def param_versions(self) -> dict[str, int]:
        return dict(self._param_versions)

    def tiles(self) -> dict[tuple[int, int], TileResult]:
        return dict(self._tiles)

    def render_result(self) -> RenderResult:
        raster_hash = raster_manifest_hash(self.scene, self._tiles)
        shade_hash = shade_manifest_hash(self.scene, self._tiles)
        img_hash = image_hash(self._image)
        return RenderResult(
            scene=self.scene,
            tiles=dict(self._tiles),
            image=self._image,
            raster_hash=raster_hash,
            shade_hash=shade_hash,
            image_hash=img_hash,
            render_hash=digest_sequence(
                (
                    raster_hash.encode("ascii"),
                    shade_hash.encode("ascii"),
                    img_hash.encode("ascii"),
                )
            ),
            parameter_hash=self.scene.parameter_hash(),
        )

    def _compose_gradient(self, key: str) -> tuple[float, ...]:
        pixels = [0.0] * (self.scene.width * self.scene.height)
        for ty, tx, *_ in tile_layout(self.scene):
            values = self._grad_chunks[key][(ty, tx)]
            for local_y in range(self.scene.tile_size):
                row = (ty * self.scene.tile_size + local_y) * self.scene.width + tx * self.scene.tile_size
                src = local_y * self.scene.tile_size
                pixels[row : row + self.scene.tile_size] = values[src : src + self.scene.tile_size]
        return tuple(pixels)

    def gradient_result(self) -> GradientResult:
        keys = self.scene.param_keys()
        pixel_gradients = {key: self._compose_gradient(key) for key in keys}
        scalar_gradients = {key: sum(values) for key, values in pixel_gradients.items()}
        return GradientResult(
            loss=self._loss,
            image_hash=image_hash(self._image),
            pixel_gradients=pixel_gradients,
            scalar_gradients=scalar_gradients,
            chunks={key: dict(chunks) for key, chunks in self._grad_chunks.items()},
            gradient_hash=gradient_hash(pixel_gradients, keys),
            parameter_hash=self.scene.parameter_hash(),
        )

    # ------------------------------------------------------------------
    # Tile impact and dependency graph

    def _all_tiles(self) -> tuple[tuple[int, int], ...]:
        return tuple((ty, tx) for ty, tx, *_ in tile_layout(self.scene))

    def _coverage_tiles(self, scene: Scene, index: int) -> set[tuple[int, int]]:
        projection = project_sphere(scene, index)
        result: set[tuple[int, int]] = set()
        for py in range(projection.y0, projection.y1):
            for px in range(projection.x0, projection.x1):
                cov, _dx, _dy, _dist, _t, _normal, _unused = self._sphere_pixel_for_impact(
                    scene, projection, px + 0.5, py + 0.5
                )
                if cov > 0.0:
                    result.add((py // scene.tile_size, px // scene.tile_size))
        return result

    @staticmethod
    def _sphere_pixel_for_impact(
        scene: Scene, projection: Projection, x: float, y: float
    ) -> tuple[float, float, float, float, float, tuple[float, float, float], int]:
        # Local import keeps the public renderer surface small while reusing the
        # exact same soft-coverage definition as rasterization.
        from .renderer import _sphere_pixel

        return _sphere_pixel(scene, projection, x, y)

    def _visible_tiles(self, index: int) -> set[tuple[int, int]]:
        return {
            (ty, tx)
            for (ty, tx), tile in self._tiles.items()
            if index in tile.ids
        }

    def _tiles_with_visible_geometry(self) -> set[tuple[int, int]]:
        return {(ty, tx) for (ty, tx), tile in self._tiles.items() if any(v >= 0 for v in tile.ids)}

    def affected_tiles(
        self, key: str, old_scene: Scene, new_scene: Scene
    ) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        """Return (raster dirty, shade dirty) tiles for one parameter edit."""

        all_tiles = set(self._all_tiles())
        parts = key.split(".")
        if parts[0] == "camera":
            return all_tiles, all_tiles
        if parts[0] == "light":
            visible = self._tiles_with_visible_geometry()
            return set(), visible
        if parts[0] == "sphere":
            index = int(parts[1])
            field = parts[2]
            if field in ("cx", "cy", "cz", "radius"):
                dirty_raster = self._coverage_tiles(old_scene, index) | self._coverage_tiles(new_scene, index)
                return dirty_raster, set(dirty_raster)
            dirty_shade = self._visible_tiles(index)
            return set(), dirty_shade
        raise DiffRenderError(E_SCENE_INVALID, f"unknown parameter key {key!r}")

    def _expected_closure(
        self,
        changed_key: str,
        dirty_raster: set[tuple[int, int]],
        dirty_shade: set[tuple[int, int]],
    ) -> tuple[str, ...]:
        tile_set = set(dirty_raster) | set(dirty_shade)
        nodes = {param_node_key(changed_key)}
        for tile in dirty_raster:
            nodes.add(raster_node_key(tile[0], tile[1]))
        for tile in dirty_shade:
            nodes.add(shade_node_key(tile[0], tile[1]))
        for tile in tile_set:
            for param_key in self.scene.param_keys():
                nodes.add(grad_chunk_node_key(tile[0], tile[1], param_key))
        if tile_set:
            for param_key in self.scene.param_keys():
                nodes.add(grad_aggregate_node_key(param_key))
            nodes.add("image:final")
            nodes.add("loss:l2")
        return tuple(sorted(nodes))

    def _scope_version_stamp(self, node_key: str, scope: Sequence[str]) -> str:
        chunks = [node_key.encode("utf-8")]
        for key in sorted(set(scope)):
            chunks.append(f"{key}:{self._param_versions[key]}".encode("utf-8"))
        return digest_sequence(chunks)

    def _tile_scopes(self) -> dict[tuple[int, int], tuple[set[str], set[str]]]:
        camera_keys = self.scene.camera_param_keys()
        light_keys = self.scene.light_param_keys()
        scopes: dict[tuple[int, int], tuple[set[str], set[str]]] = {}
        for tile_key_value, tile in self._tiles.items():
            raster_scope: set[str] = set(camera_keys)
            for index in tile.spheres:
                raster_scope.update(self.scene.sphere_geometry_keys(index))
            visible = {index for index in tile.ids if index >= 0}
            shade_scope = set(raster_scope)
            if visible:
                shade_scope.update(light_keys)
            for index in visible:
                shade_scope.update(self.scene.sphere_material_keys(index))
            scopes[tile_key_value] = (raster_scope, shade_scope)
        return scopes

    def _content_hashes(self) -> dict[str, str]:
        hashes: dict[str, str] = {}
        for key in self.scene.param_keys():
            hashes[param_node_key(key)] = self.scene.param_content_hash(key)
        for (ty, tx), tile in self._tiles.items():
            hashes[raster_node_key(ty, tx)] = tile.raster_hash()
            hashes[shade_node_key(ty, tx)] = tile.shade_hash()
            for key in self.scene.param_keys():
                hashes[grad_chunk_node_key(ty, tx, key)] = sha256_bytes(
                    pack_floats(self._grad_chunks[key][(ty, tx)])
                )
        for key in self.scene.param_keys():
            chunks = [
                hashes[grad_chunk_node_key(ty, tx, key)]
                for ty, tx in sorted(self._tiles)
            ]
            hashes[grad_aggregate_node_key(key)] = digest_sequence(
                [chunk.encode("ascii") for chunk in chunks]
            )
        hashes["image:final"] = image_hash(self._image)
        hashes["loss:l2"] = sha256_bytes(pack_floats((self._loss,)))
        return hashes

    def _add_node(self, graph: DependencyGraph, key: str, kind: str, scope: Sequence[str], hashes: Mapping[str, str]) -> None:
        version = self._node_versions.get(key, 0)
        graph.add_node(
            Node(
                key=key,
                kind=kind,
                content_hash=hashes[key],
                version=version,
                version_stamp=self._scope_version_stamp(key, scope),
                scope=tuple(sorted(set(scope))),
            )
        )

    def _build_graph(self) -> None:
        hashes = self._content_hashes()
        graph = DependencyGraph()
        param_keys = self.scene.param_keys()
        for key in param_keys:
            node_key = param_node_key(key)
            self._node_versions[node_key] = self._param_versions[key]
            graph.add_node(
                Node(
                    key=node_key,
                    kind="parameter",
                    content_hash=hashes[node_key],
                    version=self._param_versions[key],
                    version_stamp=self._scope_version_stamp(node_key, (key,)),
                    scope=(key,),
                )
            )

        scopes = self._tile_scopes()
        for ty, tx in sorted(self._tiles):
            raster_scope, shade_scope = scopes[(ty, tx)]
            rkey = raster_node_key(ty, tx)
            skey = shade_node_key(ty, tx)
            self._add_node(graph, rkey, "raster_tile", sorted(raster_scope), hashes)
            self._add_node(graph, skey, "shade_tile", sorted(shade_scope), hashes)
            for key in sorted(raster_scope):
                graph.add_edge(param_node_key(key), rkey, (key,))
            for key in sorted(shade_scope):
                graph.add_edge(param_node_key(key), skey, (key,))
            graph.add_edge(rkey, skey)
            for key in param_keys:
                gkey = grad_chunk_node_key(ty, tx, key)
                scope = set(shade_scope) | {key}
                self._add_node(graph, gkey, "gradient_chunk", sorted(scope), hashes)
                graph.add_edge(rkey, gkey)
                graph.add_edge(skey, gkey)
                if key in shade_scope:
                    graph.add_edge(param_node_key(key), gkey, (key,))

        all_scopes: set[str] = set()
        for scope_pair in scopes.values():
            all_scopes.update(scope_pair[1])
        for key in param_keys:
            gkey = grad_aggregate_node_key(key)
            self._add_node(graph, gkey, "gradient", sorted(all_scopes | {key}), hashes)
            for ty, tx in sorted(self._tiles):
                graph.add_edge(grad_chunk_node_key(ty, tx, key), gkey)
        self._add_node(graph, "image:final", "image", sorted(all_scopes), hashes)
        for ty, tx in sorted(self._tiles):
            graph.add_edge(shade_node_key(ty, tx), "image:final")
        self._add_node(graph, "loss:l2", "loss", sorted(all_scopes), hashes)
        graph.add_edge("image:final", "loss:l2")
        for key in graph.nodes:
            self._node_versions.setdefault(key, graph.nodes[key].version)
        self._graph = graph

    def _bump_closure(self, closure: Sequence[str], changed_key: str) -> None:
        self._param_versions[changed_key] += 1
        for key in closure:
            if key == param_node_key(changed_key):
                self._node_versions[key] = self._param_versions[changed_key]
            else:
                self._node_versions[key] = self._node_versions.get(key, 0) + 1

    def _node_action_records(self, closure: Sequence[str]) -> tuple[list[str], list[str]]:
        closure_set = set(closure)
        recomputed: list[str] = []
        reused: list[str] = []
        for key in sorted(self._node_versions):
            node = self._graph.nodes.get(key)
            if node is None:
                continue
            token = f"{key}@v{node.version}#{node.version_stamp}"
            if key in closure_set:
                recomputed.append(token)
            else:
                reused.append(token)
        return recomputed, reused

    # ------------------------------------------------------------------
    # Edits

    def _record_noop(
        self,
        key: str,
        old_value: float,
        new_value: float,
        old_hash: str,
        new_hash: str,
        tolerance: float,
        reason: str,
    ) -> dict[str, Any]:
        self.revision += 1
        reused: list[str] = []
        for node_key in sorted(self._node_versions):
            node = self._graph.nodes.get(node_key)
            if node is not None:
                reused.append(f"{node_key}@v{node.version}#{node.version_stamp}")
        entry = {
            "sequence": len(self._ledger) + 1,
            "revision": self.revision,
            "snapshot_version": self.snapshot_version,
            "key": key,
            "old_value_hex": float(old_value).hex(),
            "new_value_hex": float(new_value).hex(),
            "old_content_hash": old_hash,
            "new_content_hash": new_hash,
            "delta": 0.0,
            "tolerance": tolerance,
            "meaningful": False,
            "reason": reason,
            "dirty_raster_tiles": [],
            "dirty_shade_tiles": [],
            "closure": [],
            "closure_size": 0,
            "recomputed": [],
            "reused": reused,
            "reused_count": len(reused),
            "reused_manifest_hash": digest_sequence(token.encode("utf-8") for token in reused),
            "reuse_provenance": "incremental_cache",
            "recomputation_provenance": "none",
        }
        self._ledger.append(entry)
        return entry

    def set_parameter(self, key: str, value: float) -> dict[str, Any]:
        if key not in self._param_versions:
            raise DiffRenderError(E_SCENE_INVALID, f"unknown parameter key {key!r}")
        old_value = self.scene.value(key)
        new_value = float(value)
        tolerance = self.tolerances.for_key(key)
        old_hash = self.scene.param_content_hash(key)
        candidate = self.scene.with_value(key, new_value)
        new_hash = candidate.param_content_hash(key)
        delta = new_value - old_value
        if new_hash == old_hash:
            return self._record_noop(key, old_value, new_value, old_hash, new_hash, tolerance, "identical_content")
        if abs(delta) <= tolerance:
            return self._record_noop(
                key, old_value, new_value, old_hash, new_hash, tolerance, "within_tolerance"
            )

        dirty_raster, dirty_shade = self.affected_tiles(key, self.scene, candidate)
        if not dirty_shade.issubset(dirty_shade | dirty_raster):
            raise DiffRenderError(E_CLOSURE_MISMATCH, "dirty shade set lost raster dependencies")
        dirty_shade = set(dirty_shade) | set(dirty_raster)
        extra_roots = [raster_node_key(ty, tx) for ty, tx in dirty_raster]
        extra_roots += [shade_node_key(ty, tx) for ty, tx in dirty_shade]
        closure = self._graph.invalidate((key,), extra_roots=extra_roots)
        expected = self._expected_closure(key, dirty_raster, dirty_shade)
        self._graph.assert_closure(
            closure,
            expected,
            {"key": key, "dirty_raster": sorted(dirty_raster), "dirty_shade": sorted(dirty_shade)},
        )

        self.scene = candidate
        bundles = _projection_bundles(self.scene)
        projections = tuple(bundle.projection for bundle in bundles)
        for ty, tx in sorted(dirty_raster):
            self._tiles[(ty, tx)] = render_tile(self.scene, ty, tx, projections)
        for ty, tx in sorted(dirty_shade):
            tile = self._tiles[(ty, tx)]
            self._tiles[(ty, tx)] = shade_tile(self.scene, tile)
        for ty, tx in sorted(set(dirty_raster) | set(dirty_shade)):
            chunk_map = gradient_tile(
                self.scene, self.target, self._tiles[(ty, tx)], bundles
            )
            for param_key in self.scene.param_keys():
                self._grad_chunks[param_key][(ty, tx)] = chunk_map[param_key]
        self._image = compose_image(self.scene, self._tiles)
        self._loss = mean_squared_loss(self._image, self.target)

        self._bump_closure(closure, key)
        self._build_graph()
        recomputed, reused = self._node_action_records(closure)
        self.revision += 1
        entry = {
            "sequence": len(self._ledger) + 1,
            "revision": self.revision,
            "snapshot_version": self.snapshot_version,
            "key": key,
            "old_value_hex": float(old_value).hex(),
            "new_value_hex": float(new_value).hex(),
            "old_content_hash": old_hash,
            "new_content_hash": new_hash,
            "delta": delta,
            "tolerance": tolerance,
            "meaningful": True,
            "reason": "numeric_change_above_tolerance",
            "dirty_raster_tiles": [tile_key(ty, tx) for ty, tx in sorted(dirty_raster)],
            "dirty_shade_tiles": [tile_key(ty, tx) for ty, tx in sorted(dirty_shade)],
            "closure": list(closure),
            "closure_size": len(closure),
            "recomputed": recomputed,
            "reused": reused,
            "reused_count": len(reused),
            "reused_manifest_hash": digest_sequence(token.encode("utf-8") for token in reused),
            "reuse_provenance": "incremental_cache",
            "recomputation_provenance": "dirty_tile_and_parameter_chunks",
            "image_hash": image_hash(self._image),
            "loss": self._loss,
            "parameter_version": self._param_versions[key],
        }
        self._ledger.append(entry)
        return entry

    # ------------------------------------------------------------------
    # Consistency checks and persistence

    def compare_with_full(self, tolerance: float = 1.0e-11) -> dict[str, object]:
        incremental_render = self.render_result()
        full_render = render_full(self.scene)
        image_report = compare_render_results(incremental_render, full_render, tolerance)
        incremental_gradient = self.gradient_result()
        full_gradient = gradient_full(self.scene, self.target)
        gradient_report = compare_gradient_results(incremental_gradient, full_gradient, tolerance)
        return {
            "image": image_report,
            "gradient": gradient_report,
            "incremental_image_hash": incremental_render.image_hash,
            "full_image_hash": full_render.image_hash,
            "incremental_gradient_hash": incremental_gradient.gradient_hash,
            "full_gradient_hash": full_gradient.gradient_hash,
            "loss": self._loss,
            "closure_size": self._ledger[-1]["closure_size"] if self._ledger else 0,
        }

    def node_hash_manifest(self) -> dict[str, str]:
        return self._graph.hash_manifest()

    def node_stamp_manifest(self) -> dict[str, str]:
        return self._graph.stamp_manifest()

    def _encode_tile(self, tile: TileResult) -> dict[str, Any]:
        return {
            "ty": tile.ty,
            "tx": tile.tx,
            "ids": b64encode(pack_ints(tile.ids)),
            "bary": b64encode(pack_floats(tile.bary)),
            "depth": b64encode(pack_floats(tile.depth)),
            "coverage": b64encode(pack_floats(tile.coverage)),
            "normals": b64encode(pack_floats(tile.normals)),
            "ambient": b64encode(pack_floats(tile.ambient_channel)),
            "diffuse": b64encode(pack_floats(tile.diffuse_channel)),
            "intensity": b64encode(pack_floats(tile.intensity)),
            "spheres": list(tile.spheres),
        }

    def _decode_tile(self, raw: Mapping[str, Any]) -> TileResult:
        ty, tx = int(raw["ty"]), int(raw["tx"])
        y0, y1 = ty * self.scene.tile_size, (ty + 1) * self.scene.tile_size
        x0, x1 = tx * self.scene.tile_size, (tx + 1) * self.scene.tile_size
        return TileResult(
            ty=ty,
            tx=tx,
            y0=y0,
            y1=y1,
            x0=x0,
            x1=x1,
            ids=unpack_ints(b64decode(str(raw["ids"]))),
            bary=unpack_floats(b64decode(str(raw["bary"]))),
            depth=unpack_floats(b64decode(str(raw["depth"]))),
            coverage=unpack_floats(b64decode(str(raw["coverage"]))),
            normals=unpack_floats(b64decode(str(raw["normals"]))),
            ambient_channel=unpack_floats(b64decode(str(raw["ambient"]))),
            diffuse_channel=unpack_floats(b64decode(str(raw["diffuse"]))),
            intensity=unpack_floats(b64decode(str(raw["intensity"]))),
            spheres=tuple(int(v) for v in raw["spheres"]),
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": STATE_SCHEMA,
            "snapshot_version": self.snapshot_version,
            "revision": self.revision,
            "scene": self.scene.to_dict(),
            "target": b64encode(pack_floats(self.target)),
            "tolerances": self.tolerances.to_dict(),
            "param_versions": {key: self._param_versions[key] for key in sorted(self._param_versions)},
            "param_hashes": {key: self.scene.param_content_hash(key) for key in sorted(self._param_versions)},
            "ledger": self._ledger,
            "tiles": [self._encode_tile(self._tiles[(ty, tx)]) for ty, tx in sorted(self._tiles)],
            "gradient_chunks": [
                {
                    "ty": ty,
                    "tx": tx,
                    "param": key,
                    "data": b64encode(pack_floats(self._grad_chunks[key][(ty, tx)])),
                }
                for ty, tx in sorted(self._tiles)
                for key in self.scene.param_keys()
            ],
            "graph": self._graph.to_dict(),
            "image_hash": image_hash(self._image),
            "gradient_hash": self.gradient_result().gradient_hash,
            "loss": self._loss,
        }

    def commit(self, store: Any, closed: bool = False) -> int:
        self.snapshot_version = int(store.save(self.state_dict(), closed=closed))
        return self.snapshot_version

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any], expected_graph: DependencyGraph | None = None) -> "IncrementalPipeline":
        if state.get("schema") != STATE_SCHEMA:
            raise DiffRenderError(E_DEPENDENCY_STALE, "unsupported persisted state schema")
        try:
            scene = Scene.from_dict(state["scene"])  # type: ignore[arg-type]
            target = unpack_floats(b64decode(str(state["target"])))
            tolerances = ToleranceConfig.from_dict(state.get("tolerances", {}))
            param_versions = {str(k): int(v) for k, v in state["param_versions"].items()}  # type: ignore[union-attr]
            graph_data = state["graph"]
            loaded_graph = DependencyGraph.from_dict(graph_data)  # type: ignore[arg-type]
            obj = cls.__new__(cls)
            obj.scene = scene
            obj.target = tuple(target)
            obj.tolerances = tolerances
            obj._param_versions = param_versions
            obj._node_versions = {key: node.version for key, node in loaded_graph.nodes.items()}
            obj._ledger = list(state.get("ledger", []))
            obj.revision = int(state.get("revision", 0))
            obj.snapshot_version = int(state.get("snapshot_version", 0))
            obj._tiles = {}
            for raw in state.get("tiles", []):
                tile = obj._decode_tile(raw)
                obj._tiles[(tile.ty, tile.tx)] = tile
            obj._grad_chunks = {key: {} for key in scene.param_keys()}
            for raw in state.get("gradient_chunks", []):
                ty, tx = int(raw["ty"]), int(raw["tx"])
                param = str(raw["param"])
                obj._grad_chunks[param][(ty, tx)] = unpack_floats(b64decode(str(raw["data"])))
            for key in scene.param_keys():
                for tile in obj._all_tiles():
                    if tile not in obj._grad_chunks[key]:
                        raise DiffRenderError(
                            E_DEPENDENCY_STALE,
                            "persisted gradient chunk is missing",
                            {"param": key, "tile": list(tile)},
                        )
            obj._image = compose_image(scene, obj._tiles)
            obj._loss = mean_squared_loss(obj._image, obj.target)
            obj._build_graph()
            actual_graph = obj._graph
            expected_dump = canonical_json_bytes(loaded_graph.to_dict())
            actual_dump = canonical_json_bytes(actual_graph.to_dict())
            if expected_dump != actual_dump:
                # Find a precise, useful first mismatch for the business error.
                for key in sorted(set(loaded_graph.nodes) | set(actual_graph.nodes)):
                    left = loaded_graph.nodes.get(key)
                    right = actual_graph.nodes.get(key)
                    if left is None or right is None or left.to_dict() != right.to_dict():
                        raise DiffRenderError(
                            E_DEPENDENCY_STALE,
                            "rebuilt dependency node differs from committed snapshot",
                            {
                                "node": key,
                                "committed": None if left is None else left.to_dict(),
                                "rebuilt": None if right is None else right.to_dict(),
                            },
                        )
                raise DiffRenderError(
                    E_DEPENDENCY_STALE,
                    "rebuilt dependency edges differ from committed snapshot",
                )
            if expected_graph is not None:
                expected_graph.verify_nodes(actual_graph.hash_manifest())
            return obj
        except DiffRenderError:
            raise
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise DiffRenderError(E_DEPENDENCY_STALE, f"invalid persisted pipeline state: {exc}") from exc
