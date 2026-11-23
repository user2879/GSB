"""Edit command stream application (band patches, tile add/remove, adjacency,
weight and fusion strategy changes). Edits mutate the *sample inputs* on disk,
never the workbench source."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image

from . import samplegen


def _load_manifest(samples_dir):
    path = Path(samples_dir) / "manifest.json"
    return path, json.loads(path.read_text(encoding="utf-8"))


def _save_manifest(path, manifest):
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _tile_entry(manifest, tile_id):
    for entry in manifest["tiles"]:
        if entry["id"] == tile_id:
            return entry
    return None


def apply_edit(samples_dir, config_path, edit: dict):
    op = edit["op"]
    samples_dir = Path(samples_dir)
    mpath, manifest = _load_manifest(samples_dir)
    tile_size = int(manifest["tile_size"])
    overlap = int(manifest["overlap_px"])
    if op == "replace_band":
        entry = _tile_entry(manifest, edit["tile"])
        seed = samplegen.GLOBAL_SEED + int(edit.get("seed_delta", 0))
        gt = samplegen.tile_ground_truth(entry["row"], entry["col"],
                                         tile_size=tile_size, overlap=overlap)
        bands = samplegen.synthesize_bands(gt, entry["row"], entry["col"],
                                           tile_size=tile_size, overlap=overlap,
                                           seed=seed)
        band = edit["band"]
        Image.fromarray(bands[band], mode="L").save(
            samples_dir / "tiles" / edit["tile"] / f"{band}.png")
    elif op == "add_tile":
        row, col = int(edit["row"]), int(edit["col"])
        tid = samplegen.generate_tile(samples_dir, row, col,
                                      band_set_version=edit.get("band_set_version", "bs-v1"),
                                      tile_size=tile_size, overlap=overlap)
        neighbors = sorted(edit.get("neighbors", []))
        manifest["tiles"].append({"id": tid, "row": row, "col": col,
                                  "neighbors": neighbors,
                                  "band_set_version": edit.get("band_set_version", "bs-v1")})
        for nb in neighbors:
            nb_entry = _tile_entry(manifest, nb)
            if nb_entry is not None and tid not in nb_entry["neighbors"]:
                nb_entry["neighbors"] = sorted(nb_entry["neighbors"] + [tid])
        _save_manifest(mpath, manifest)
    elif op == "remove_tile":
        tid = edit["tile"]
        manifest["tiles"] = [e for e in manifest["tiles"] if e["id"] != tid]
        for entry in manifest["tiles"]:
            if tid in entry["neighbors"]:
                entry["neighbors"] = sorted(n for n in entry["neighbors"] if n != tid)
        tdir = samples_dir / "tiles" / tid
        if tdir.exists():
            shutil.rmtree(tdir)
        _save_manifest(mpath, manifest)
    elif op == "set_adjacency":
        entry = _tile_entry(manifest, edit["tile"])
        old = set(entry["neighbors"])
        new = set(edit["neighbors"])
        entry["neighbors"] = sorted(new)
        for nb in new - old:
            nb_entry = _tile_entry(manifest, nb)
            if nb_entry is not None and entry["id"] not in nb_entry["neighbors"]:
                nb_entry["neighbors"] = sorted(nb_entry["neighbors"] + [entry["id"]])
        for nb in old - new:
            nb_entry = _tile_entry(manifest, nb)
            if nb_entry is not None:
                nb_entry["neighbors"] = sorted(n for n in nb_entry["neighbors"]
                                               if n != entry["id"])
        _save_manifest(mpath, manifest)
    elif op == "set_index_weights":
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        cfg["index_weights"] = edit["weights"]
        Path(config_path).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    elif op == "set_fusion_strategy":
        cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
        cfg["fusion_strategy_version"] = edit["version"]
        Path(config_path).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    else:
        raise ValueError(f"unknown edit op {op}")
    if op in ("replace_band",):
        _save_manifest(mpath, manifest)
