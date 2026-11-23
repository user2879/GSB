import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

from workbench import errors
from workbench.pipeline import Pipeline


def _codes(report):
    return {f["code"] for f in report["failed"]}


def test_clean_run_has_no_failures(small_samples, tmp_path):
    report = Pipeline(small_samples, tmp_path / "w").run()
    assert report["failed"] == []
    assert len(report["per_tile_class_counts"]) == 4


def test_band_missing(small_samples, tmp_path):
    (small_samples / "tiles" / "tile_r0_c0" / "G.png").unlink()
    report = Pipeline(small_samples, tmp_path / "w").run()
    assert errors.E_BAND_MISSING in _codes(report)
    # other tiles still classified: failure does not abort the run
    assert "tile_r0_c1" in report["per_tile_class_counts"]
    assert "tile_r0_c0" not in report["per_tile_class_counts"]


def test_tile_shape(small_samples, tmp_path):
    Image.fromarray(np.zeros((16, 16), dtype=np.uint8), mode="L").save(
        small_samples / "tiles" / "tile_r0_c1" / "R.png")
    report = Pipeline(small_samples, tmp_path / "w").run()
    assert errors.E_TILE_SHAPE in _codes(report)
    assert "tile_r0_c0" in report["per_tile_class_counts"]


def test_adj_asymmetric(small_samples, tmp_path):
    mpath = small_samples / "manifest.json"
    manifest = json.loads(mpath.read_text())
    for entry in manifest["tiles"]:
        if entry["id"] == "tile_r0_c1":
            entry["neighbors"] = [n for n in entry["neighbors"] if n != "tile_r0_c0"]
    mpath.write_text(json.dumps(manifest, indent=2))
    report = Pipeline(small_samples, tmp_path / "w").run()
    assert errors.E_ADJ_ASYMMETRIC in _codes(report)


def test_band_version_unknown(small_samples, tmp_path):
    mpath = small_samples / "manifest.json"
    manifest = json.loads(mpath.read_text())
    manifest["tiles"][0]["band_set_version"] = "bs-v9"
    mpath.write_text(json.dumps(manifest, indent=2))
    report = Pipeline(small_samples, tmp_path / "w").run()
    assert errors.E_BAND_VERSION_UNKNOWN in _codes(report)
    assert len(report["per_tile_class_counts"]) == 3


def test_adj_unknown_ref(small_samples, tmp_path):
    mpath = small_samples / "manifest.json"
    manifest = json.loads(mpath.read_text())
    manifest["tiles"][0]["neighbors"].append("tile_ghost")
    mpath.write_text(json.dumps(manifest, indent=2))
    report = Pipeline(small_samples, tmp_path / "w").run()
    assert errors.E_ADJ_UNKNOWN_REF in _codes(report)
