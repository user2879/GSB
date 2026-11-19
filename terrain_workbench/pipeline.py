"""Frame pipeline: LOD selection -> budget -> residency -> crack check -> report.

Everything is deterministic: identical (terrain, track, budget config) inputs
produce byte-identical reports across runs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .budget import enforce_triangle_budget
from .camera import CameraTrack
from .diagnostics import RunReport, canonical_json
from .errors import E_GL_CONTEXT
from .gl_backend import create_backend
from .heightfield import generate_heightfield, heightfield_hash
from .mesh import build_neighbor_map, check_cracks
from .pyramid import build_pyramid
from .quadtree import Quadtree
from .residency import ResidencyManager
from .selection import HEIGHT_SCALE, select_lod
from .shading import DEFAULT_SHADING, ShadingParams


def tile_bytes_and_hash(qt, heights, key) -> tuple[int, str]:
    """Deterministic resident-tile payload: elevation verts + normal bytes."""
    from .mesh import block_stride, block_region_cells

    s = block_stride(qt, key.depth, key.density)
    x0, y0, span = block_region_cells(qt, key.depth, key.bx, key.by)
    idx = np.arange(0, span + 1, s, dtype=np.int64)
    ii = x0 + idx
    jj = y0 + idx
    tile = heights[np.ix_(jj, ii)]
    elev_bytes = tile.tobytes()
    nbytes = tile.size * 2 + tile.size * 3  # uint16 elevation + rgb8 normal
    sha = hashlib.sha256(elev_bytes).hexdigest()
    return nbytes, sha


def run_pipeline(
    size: int,
    seed: int,
    track: CameraTrack,
    tri_budget: int,
    mem_budget: int,
    tau: float,
    shading: ShadingParams = DEFAULT_SHADING,
    prefer_gl: bool = True,
    heights: np.ndarray | None = None,
    cum_dev=None,
):
    if heights is None:
        heights = generate_heightfield(size, seed)
    if cum_dev is None:
        _, meta = build_pyramid(heights)
        cum_dev = meta["cumulative_dev"]
    qt = Quadtree(heights)
    world_xy = (size - 1) * 1.0
    z_max = 65535 * HEIGHT_SCALE * 2.0
    track.validate(world_xy, z_max)

    backend, gl_status = create_backend(prefer_gl=prefer_gl)
    residency = ResidencyManager(mem_budget)
    run = RunReport()
    manifest_frames = []

    for frame in range(track.frames):
        cam = track.sample(frame)
        blocks = select_lod(qt, cum_dev, cam, tau)
        blocks, degradations = enforce_triangle_budget(
            blocks, qt, cum_dev, cam, tri_budget
        )
        neighbors, shared, regions = build_neighbor_map(qt, blocks)
        crack = check_cracks(qt, heights, blocks, neighbors, shared, regions)

        evictions_before = sum(1 for e in residency.ledger if e.op == "evict")
        for b in blocks:
            nbytes, sha = tile_bytes_and_hash(qt, heights, b.key)
            key = f"{b.key.depth}/{b.key.bx}/{b.key.by}/{b.key.density}"
            residency.require(key, nbytes, sha, frame)
        evictions_after = sum(1 for e in residency.ledger if e.op == "evict")

        lod_hist = {}
        for b in blocks:
            tag = f"d{b.key.depth}g{b.key.density}"
            lod_hist[tag] = lod_hist.get(tag, 0) + 1
        triangles = sum(b.triangles for b in blocks)
        max_err = max((b.screen_error for b in blocks), default=0.0)

        manifest = {
            "frame": frame,
            "camera": {
                "pos": [round(v, 6) for v in cam.pos],
                "target": [round(v, 6) for v in cam.target],
                "fov_y_deg": cam.fov_y_deg,
                "viewport": list(cam.viewport),
            },
            "shading": shading.to_dict(),
            "blocks": [b.manifest_entry() for b in blocks],
            "degradations": [d.to_dict() for d in degradations],
        }
        manifest_sha = hashlib.sha256(canonical_json(manifest).encode()).hexdigest()
        manifest["sha"] = manifest_sha
        manifest_frames.append(manifest)

        report = {
            "frame": frame,
            "camera_pos": [round(v, 6) for v in cam.pos],
            "visible_blocks": len(blocks),
            "lod_histogram": lod_hist,
            "max_screen_error": round(max_err, 6),
            "triangles": triangles,
            "tri_budget": tri_budget,
            "tri_usage": round(triangles / tri_budget, 6) if tri_budget else None,
            "mem_resident_bytes": residency.resident_bytes,
            "mem_budget": mem_budget,
            "mem_usage": round(residency.resident_bytes / mem_budget, 6),
            "degraded_blocks": len(degradations),
            "evictions_total": evictions_after,
            "evictions_this_frame": evictions_after - evictions_before,
            "crack_edges_checked": crack["edges_checked"],
            "crack_max_gap": crack["max_gap"],
            "gl_status": gl_status,
            "manifest_sha": manifest_sha,
        }
        run.add(report)

    audit = residency.audit()
    summary = {
        "run_hash": run.run_hash,
        "frames": track.frames,
        "size": size,
        "seed": seed,
        "heightfield_sha": heightfield_hash(heights),
        "gl_status": gl_status,
        "residency_audit": audit,
        "manifest_sha": hashlib.sha256(
            canonical_json([m["sha"] for m in manifest_frames]).encode()
        ).hexdigest(),
    }
    return {
        "summary": summary,
        "frames": run.frames,
        "manifests": manifest_frames,
        "ledger": [e.to_dict() for e in residency.ledger],
        "report_jsonl": run.to_jsonl(),
    }


def write_outputs(result: dict, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.jsonl").write_text(result["report_jsonl"])
    (out / "summary.json").write_text(
        json.dumps(result["summary"], indent=2, sort_keys=True) + "\n"
    )
    (out / "manifest.json").write_text(
        json.dumps(result["manifests"], indent=1, sort_keys=True) + "\n"
    )
    (out / "ledger.json").write_text(
        json.dumps(result["ledger"], indent=1, sort_keys=True) + "\n"
    )
