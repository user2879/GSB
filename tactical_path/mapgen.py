"""Deterministic tactical map, unit and query generation."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .canonical import atomic_write_bytes, atomic_write_json, sha256_json
from .errors import ErrorCode, TacticalError
from .models import (
    MapBundle,
    MapData,
    MobilityRule,
    Query,
    TierSpec,
    Unit,
    compute_map_hash,
    flat,
    xy,
)


NEIGHBORS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _seed_for(seed: int, spec: TierSpec) -> np.random.Generator:
    sequence = np.random.SeedSequence([int(seed), int(spec.size), int(spec.region_size)])
    return np.random.default_rng(sequence)


def _make_regions(size: int, region_size: int) -> np.ndarray:
    if size % region_size != 0:
        raise TacticalError(
            ErrorCode.MAP_SHAPE,
            "map size must be divisible by region size",
            context={"size": size, "region_size": region_size},
        )
    grid = np.arange((size // region_size) ** 2, dtype=np.uint16).reshape(
        size // region_size, size // region_size
    )
    return np.repeat(np.repeat(grid, region_size, axis=0), region_size, axis=1)


def _make_sectors(region_shape: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    rows, cols = region_shape
    sector_shape = ((rows + 1) // 2, (cols + 1) // 2)
    sectors = np.zeros((rows, cols), dtype=np.uint16)
    for row in range(rows):
        for col in range(cols):
            sectors[row, col] = (row // 2) * sector_shape[1] + (col // 2)
    return sectors, sector_shape


def _apply_boundary_walls(
    passable: np.ndarray,
    regions: np.ndarray,
    region_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    height, width = passable.shape
    rows = height // region_size
    cols = width // region_size
    door_mask = np.zeros_like(passable, dtype=bool)
    offsets = [max(2, region_size // 2)]

    for boundary in range(1, cols):
        left = boundary * region_size - 1
        right = boundary * region_size
        passable[:, left] = False
        passable[:, right] = False
        for row in range(rows):
            for offset in offsets:
                y = row * region_size + offset
                if 0 <= y < height:
                    passable[y, left] = True
                    passable[y, right] = True
                    door_mask[y, left] = True
                    door_mask[y, right] = True

    for boundary in range(1, rows):
        upper = boundary * region_size - 1
        lower = boundary * region_size
        passable[upper, :] = False
        passable[lower, :] = False
        for col in range(cols):
            for offset in offsets:
                x = col * region_size + offset
                if 0 <= x < width:
                    passable[upper, x] = True
                    passable[lower, x] = True
                    door_mask[upper, x] = True
                    door_mask[lower, x] = True
    return door_mask


def _apply_interior_walls(
    passable: np.ndarray,
    region_size: int,
    regions: np.ndarray,
    rng: np.random.Generator,
) -> None:
    rows = passable.shape[0] // region_size
    cols = passable.shape[1] // region_size
    for row in range(rows):
        for col in range(cols):
            x0 = col * region_size
            y0 = row * region_size
            wall_x = x0 + region_size // 2
            gap_y = y0 + region_size // 2
            for y in range(y0 + 2, y0 + region_size - 2):
                if y != gap_y:
                    passable[y, wall_x] = False

            if rng.random() < 0.42:
                cx = x0 + int(rng.integers(3, max(4, region_size - 5)))
                cy = y0 + int(rng.integers(3, max(4, region_size - 5)))
                if cx != wall_x:
                    passable[cy : min(cy + 2, y0 + region_size - 1), cx : min(cx + 2, x0 + region_size - 1)] = False


def largest_component(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    seen = np.zeros(mask.size, dtype=bool)
    flat_mask = mask.ravel()
    best: list[int] = []
    best_mark = np.zeros(mask.size, dtype=bool)
    for start in np.flatnonzero(flat_mask):
        start_i = int(start)
        if seen[start_i]:
            continue
        seen[start_i] = True
        queue: deque[int] = deque([start_i])
        component: list[int] = []
        while queue:
            cell = queue.popleft()
            component.append(cell)
            x = cell % width
            y = cell // width
            if x > 0:
                nb = cell - 1
                if flat_mask[nb] and not seen[nb]:
                    seen[nb] = True
                    queue.append(nb)
            if x + 1 < width:
                nb = cell + 1
                if flat_mask[nb] and not seen[nb]:
                    seen[nb] = True
                    queue.append(nb)
            if y > 0:
                nb = cell - width
                if flat_mask[nb] and not seen[nb]:
                    seen[nb] = True
                    queue.append(nb)
            if y + 1 < height:
                nb = cell + width
                if flat_mask[nb] and not seen[nb]:
                    seen[nb] = True
                    queue.append(nb)
        if len(component) > len(best):
            best = component
    if best:
        best_mark[np.asarray(best, dtype=np.int64)] = True
    return best_mark.reshape(height, width)


def generate_bundle(
    spec: TierSpec,
    *,
    seed: int,
    mobility_rules: dict[str, MobilityRule],
) -> MapBundle:
    rng = _seed_for(seed, spec)
    size = spec.size
    region_size = spec.region_size
    regions = _make_regions(size, region_size)
    region_shape = (size // region_size, size // region_size)
    sector_grid, sector_shape = _make_sectors(region_shape)

    region_base = rng.integers(1, 5, size=region_shape, dtype=np.uint8)
    noise = rng.integers(0, 3, size=(size, size), dtype=np.uint8)
    terrain = np.clip(region_base.reshape(-1)[regions] + noise, 1, 9).astype(np.uint8)
    passable_bool = np.ones((size, size), dtype=bool)

    door_mask = _apply_boundary_walls(passable_bool, regions, region_size, rng)
    if spec.interior_walls:
        _apply_interior_walls(passable_bool, region_size, regions, rng)
    passable = passable_bool.astype(np.uint8)
    terrain[door_mask] = 5

    sectors = np.repeat(
        np.repeat(sector_grid, region_size, axis=0), region_size, axis=1
    ).astype(np.uint16)

    content_hash = compute_map_hash(
        tier=spec.name,
        seed=seed,
        width=size,
        height=size,
        region_size=region_size,
        terrain=terrain,
        passable=passable,
        regions=regions,
        sectors=sectors,
        mobility=mobility_rules,
    )
    map_data = MapData(
        tier=spec.name,
        seed=seed,
        width=size,
        height=size,
        region_size=region_size,
        region_shape=region_shape,
        sector_shape=sector_shape,
        terrain=terrain,
        passable=passable,
        regions=regions,
        sectors=sectors,
        mobility_rules=dict(mobility_rules),
        content_hash=content_hash,
        map_version=f"{spec.name}:{content_hash[:16]}",
    )
    map_data.validate()

    mobility_names = list(spec.mobility_classes) if spec.mobility_classes else sorted(mobility_rules)
    unknown = sorted(set(mobility_names) - set(mobility_rules))
    if unknown:
        raise TacticalError(
            ErrorCode.UNIT_INVALID,
            "tier references unknown mobility classes",
            context={"tier": spec.name, "mobility": unknown},
        )
    mobility_names = sorted(mobility_names)
    allowed = {name: map_data.allowed_mask(name) for name in mobility_names}
    components = {
        name: largest_component(mask) for name, mask in sorted(allowed.items())
    }
    intersection = np.ones_like(passable, dtype=bool)
    for mask in components.values():
        intersection &= mask

    target_regions: list[int] = []
    if spec.query_mode == "intra_region":
        candidates = []
        for region_id in range(map_data.region_count):
            region_mask = (regions == region_id) & intersection
            if int(region_mask.sum()) >= 8:
                candidates.append(region_id)
        if len(candidates) < min(2, spec.target_groups):
            raise TacticalError(
                ErrorCode.UNIT_INVALID,
                "not enough connected tactical regions for intra-region queries",
                context={"tier": spec.name, "candidates": candidates},
            )
        picks = rng.choice(
            np.asarray(candidates, dtype=np.int64),
            size=min(2, spec.target_groups),
            replace=False,
        )
        target_regions = sorted(int(value) for value in picks)

    units: list[Unit] = []
    used_cells: set[int] = set()
    for index in range(spec.unit_count):
        mobility = mobility_names[index % len(mobility_names)]
        rule = mobility_rules[mobility]
        if spec.query_mode == "intra_region" and target_regions:
            region_id = target_regions[index % len(target_regions)]
            candidate_mask = components[mobility] & (regions == region_id)
        else:
            region_id = -1
            candidate_mask = components[mobility]
        cells = np.flatnonzero(candidate_mask.ravel())
        if len(cells) == 0:
            raise TacticalError(
                ErrorCode.UNIT_INVALID,
                "movement class has no connected placement cell",
                context={"tier": spec.name, "mobility": mobility},
            )
        cell = -1
        for _ in range(4096):
            candidate = int(rng.choice(cells))
            if candidate not in used_cells:
                cell = candidate
                break
        if cell < 0:
            cell = int(cells[index % len(cells)])
        used_cells.add(cell)
        units.append(
            Unit(
                id=f"u-{spec.name}-{index + 1:04d}",
                tier=spec.name,
                cell=cell,
                faction="blue" if index % 2 == 0 else "red",
                mobility=mobility,
                cost_multiplier=rule.cost_multiplier,
                max_terrain=rule.max_terrain,
            )
        )

    for unit in units:
        unit.validate(map_data)

    queries: list[Query] = []
    target_pools: dict[str, np.ndarray] = {}
    if spec.query_mode == "cross_region":
        for mobility in mobility_names:
            cells = np.flatnonzero(components[mobility].ravel())
            count = min(max(spec.target_groups, 1), len(cells))
            target_pools[mobility] = np.asarray(
                sorted(int(value) for value in rng.choice(cells, size=count, replace=False)),
                dtype=np.int64,
            )

    objectives: list[int] = []
    if spec.query_mode == "intra_region":
        for region_id in target_regions:
            cells = np.flatnonzero((components[mobility_names[0]] & intersection & (regions == region_id)).ravel())
            if len(cells) == 0:
                cells = np.flatnonzero((intersection & (regions == region_id)).ravel())
            objectives.append(int(rng.choice(cells)))

    for index in range(spec.query_count):
        if spec.query_mode == "intra_region":
            group = index % len(target_regions)
            group_units = [
                unit
                for pos, unit in enumerate(units)
                if pos % len(target_regions) == group
            ]
            if not group_units:
                group_units = units
            unit = group_units[index % len(group_units)]
            target = objectives[group]
            if target == unit.cell:
                region_cells = np.flatnonzero(
                    (components[unit.mobility] & (regions == map_data.region_of_cell(target))).ravel()
                )
                alternatives = [int(cell) for cell in region_cells if int(cell) != unit.cell]
                target = alternatives[index % len(alternatives)]
        else:
            unit = units[index % len(units)]
            pool = target_pools[unit.mobility]
            offset = 0
            while True:
                target = int(pool[(index + offset) % len(pool)])
                same_region = map_data.region_of_cell(target) == map_data.region_of_cell(unit.cell)
                if target != unit.cell and not same_region:
                    break
                offset += 1
                if offset >= len(pool) * 2:
                    if target != unit.cell:
                        break
                    raise TacticalError(
                        ErrorCode.UNIT_INVALID,
                        "cannot construct a cross-region query target",
                        context={"tier": spec.name, "unit_id": unit.id},
                    )
        query = Query(
            id=f"q-{spec.name}-{index + 1:04d}",
            tier=spec.name,
            unit_id=unit.id,
            start=unit.cell,
            target=target,
            mobility=unit.mobility,
        )
        query.validate(map_data)
        queries.append(query)

    units_manifest = [unit.to_manifest(size) for unit in units]
    queries_manifest = [query.to_manifest(size) for query in queries]
    metadata = {
        "map": map_data.map_manifest(),
        "map_content_hash": content_hash,
        "region_count": map_data.region_count,
        "sector_count": map_data.sector_count,
        "unit_count": len(units),
        "query_count": len(queries),
        "unit_manifest_hash": sha256_json(units_manifest),
        "query_manifest_hash": sha256_json(queries_manifest),
        "query_mode": spec.query_mode,
        "target_regions": target_regions,
    }
    return MapBundle(
        map_data=map_data,
        units=units,
        queries=queries,
        metadata=metadata,
    )


def save_bundle(bundle: MapBundle, directory: str | Path) -> dict[str, Any]:
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    arrays = {
        "terrain.u8": np.asarray(bundle.map_data.terrain, dtype=np.uint8).tobytes(order="C"),
        "passable.u8": np.asarray(bundle.map_data.passable, dtype=np.uint8).tobytes(order="C"),
        "regions.u16": np.asarray(bundle.map_data.regions, dtype="<u2").tobytes(order="C"),
        "sectors.u16": np.asarray(bundle.map_data.sectors, dtype="<u2").tobytes(order="C"),
    }
    for name, data in arrays.items():
        atomic_write_bytes(path / name, data)
    units_manifest = [unit.to_manifest(bundle.map_data.width) for unit in bundle.units]
    queries_manifest = [
        query.to_manifest(bundle.map_data.width) for query in bundle.queries
    ]
    atomic_write_json(path / "units.json", units_manifest)
    atomic_write_json(path / "queries.json", queries_manifest)
    atomic_write_json(path / "metadata.json", bundle.metadata)
    return bundle.metadata


def load_bundle(directory: str | Path, mobility_rules: dict[str, MobilityRule]) -> MapBundle:
    path = Path(directory)
    try:
        metadata = __import__("json").loads((path / "metadata.json").read_text(encoding="utf-8"))
        manifest = metadata["map"]
        width = int(manifest["width"])
        height = int(manifest["height"])
        terrain = np.frombuffer((path / "terrain.u8").read_bytes(), dtype=np.uint8).reshape(height, width).copy()
        passable = np.frombuffer((path / "passable.u8").read_bytes(), dtype=np.uint8).reshape(height, width).copy()
        regions = np.frombuffer((path / "regions.u16").read_bytes(), dtype="<u2").reshape(height, width).copy()
        sectors = np.frombuffer((path / "sectors.u16").read_bytes(), dtype="<u2").reshape(height, width).copy()
        units_raw = __import__("json").loads((path / "units.json").read_text(encoding="utf-8"))
        queries_raw = __import__("json").loads((path / "queries.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise TacticalError(
            ErrorCode.SNAPSHOT_VERSION,
            "unable to load map bundle",
            context={"path": str(path)},
        ) from exc
    map_data = MapData(
        tier=str(manifest["tier"]),
        seed=int(manifest["seed"]),
        width=width,
        height=height,
        region_size=int(manifest["region_size"]),
        region_shape=tuple(int(value) for value in manifest["region_shape"]),
        sector_shape=tuple(int(value) for value in manifest["sector_shape"]),
        terrain=terrain,
        passable=passable,
        regions=regions,
        sectors=sectors,
        mobility_rules=dict(mobility_rules),
        content_hash=str(manifest["content_hash"]),
        map_version=str(manifest["map_version"]),
    )
    map_data.validate()
    units = [
        Unit(
            id=str(item["id"]),
            tier=str(item["tier"]),
            cell=int(item["cell"]),
            faction=str(item["faction"]),
            mobility=str(item["mobility"]),
            cost_multiplier=int(item["cost_multiplier"]),
            max_terrain=int(item["max_terrain"]),
        )
        for item in units_raw
    ]
    queries = [
        Query(
            id=str(item["id"]),
            tier=str(item["tier"]),
            unit_id=str(item["unit_id"]),
            start=int(item["start"]),
            target=int(item["target"]),
            mobility=str(item["mobility"]),
        )
        for item in queries_raw
    ]
    for unit in units:
        unit.validate(map_data)
    for query in queries:
        query.validate(map_data)
    return MapBundle(map_data=map_data, units=units, queries=queries, metadata=metadata)


def generate_all(
    *,
    seed: int,
    tiers: dict[str, TierSpec],
    mobility_rules: dict[str, MobilityRule],
    output_root: str | Path,
) -> dict[str, Any]:
    root = Path(output_root)
    report: dict[str, Any] = {"seed": seed, "tiers": []}
    for name, spec in tiers.items():
        bundle = generate_bundle(spec, seed=seed, mobility_rules=mobility_rules)
        save_bundle(bundle, root / name)
        report["tiers"].append(bundle.metadata)
    atomic_write_json(root / "generation_manifest.json", report)
    return report
