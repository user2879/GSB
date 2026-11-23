import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests._util import ROOT, make_workspace
from mstc.config import load_config
from mstc.errors import Code, PipelineFatal
from mstc.loader import load_manifest, validate_and_load
from mstc.paths import to_path


class ConfigValidationTests(unittest.TestCase):
    def test_valid_config_loads(self):
        cfg = load_config(ROOT / "configs" / "config.yaml")
        self.assertEqual(cfg.window.size, 8)
        self.assertEqual(cfg.fixed.gear.value, "NORMAL")

    def test_bad_weight_enum(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        text = (ws / "config.yaml").read_text().replace("vegetation: 1.0", "vegetation: .nan")
        (ws / "bad.yaml").write_text(text)
        with self.assertRaises(PipelineFatal) as ctx:
            load_config(ws / "bad.yaml")
        self.assertEqual(ctx.exception.code, Code.E_WEIGHT_INVALID)

    def test_all_zero_weights_rejected(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        text = (ws / "config.yaml").read_text()
        for name in ("vegetation", "water", "texture", "slope"):
            text = text.replace(f"{name}: ", f"{name}: 0.0 #", 1) if False else text
        import re
        text = re.sub(r"(vegetation|water|texture|slope): [0-9.]+", r"\1: 0.0", text)
        (ws / "zero.yaml").write_text(text)
        with self.assertRaises(PipelineFatal) as ctx:
            load_config(ws / "zero.yaml")
        self.assertEqual(ctx.exception.code, Code.E_WEIGHT_INVALID)

    def test_bad_fixed_enum(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        text = (ws / "config.yaml").read_text().replace("NORMAL", "WEIRD")
        (ws / "enum.yaml").write_text(text)
        with self.assertRaises(PipelineFatal) as ctx:
            load_config(ws / "enum.yaml")
        self.assertEqual(ctx.exception.code, Code.E_FIXED_ENUM)

    def test_bad_budget(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        text = (ws / "config.yaml").read_text().replace(
            "global_pixel_budget: 100000", "global_pixel_budget: 0")
        (ws / "budget.yaml").write_text(text)
        with self.assertRaises(PipelineFatal) as ctx:
            load_config(ws / "budget.yaml")
        self.assertEqual(ctx.exception.code, Code.E_BUDGET_INVALID)


class LoaderValidationTests(unittest.TestCase):
    def _load(self, ws):
        cfg = load_config(ws / "config.yaml")
        manifest = load_manifest(to_path(cfg.manifest_path, ws), ws)
        return cfg, validate_and_load(manifest, cfg, ws)

    def test_clean_manifest_loads_all_tiles(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        cfg, (tiles, failures) = self._load(ws)
        self.assertEqual(len(tiles), 9)
        self.assertFalse(failures)

    def test_missing_band_is_isolated(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        manifest_p = ws / "samples" / "manifest.json"
        doc = json.loads(manifest_p.read_text())
        target = next(t for t in doc["tiles"] if t["tile_id"] == "t01")
        band_path = to_path(target["bands"]["NIR"], ws)
        band_path.unlink()
        target["bands"].pop("NIR")
        manifest_p.write_text(json.dumps(doc))
        cfg, (tiles, failures) = self._load(ws)
        self.assertNotIn("t01", tiles)
        self.assertIn(Code.E_BAND_COUNT_MISMATCH, failures.codes())
        self.assertEqual(len(tiles), 8)

    def test_truncated_band_code(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        manifest_p = ws / "samples" / "manifest.json"
        doc = json.loads(manifest_p.read_text())
        target = next(t for t in doc["tiles"] if t["tile_id"] == "t02")
        bp = to_path(target["bands"]["R"], ws)
        raw = bp.read_bytes()
        bp.write_bytes(raw[: len(raw) // 3])
        manifest_p.write_text(json.dumps(doc))
        cfg, (tiles, failures) = self._load(ws)
        self.assertIn(Code.E_BAND_TRUNCATED, failures.codes())
        self.assertNotIn("t02", tiles)

    def test_resolution_mismatch(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        manifest_p = ws / "samples" / "manifest.json"
        doc = json.loads(manifest_p.read_text())
        target = next(t for t in doc["tiles"] if t["tile_id"] == "t10")
        bp = to_path(target["bands"]["B"], ws)
        arr = np.load(bp)
        np.save(bp, arr[:-1, :])
        manifest_p.write_text(json.dumps(doc))
        cfg, (tiles, failures) = self._load(ws)
        self.assertIn(Code.E_RESOLUTION_MISMATCH, failures.codes())

    def test_dangling_and_asymmetric_adjacency(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        manifest_p = ws / "samples" / "manifest.json"
        doc = json.loads(manifest_p.read_text())
        next(t for t in doc["tiles"] if t["tile_id"] == "t22")["neighbors"].append("t99")
        manifest_p.write_text(json.dumps(doc))
        cfg, (tiles, failures) = self._load(ws)
        self.assertIn(Code.E_ADJ_DANGLING, failures.codes())

    def test_asymmetric_adjacency_code(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        manifest_p = ws / "samples" / "manifest.json"
        doc = json.loads(manifest_p.read_text())
        # t00 lists t01 as neighbour, but remove the reciprocal edge on t01
        t00 = next(t for t in doc["tiles"] if t["tile_id"] == "t00")
        t01 = next(t for t in doc["tiles"] if t["tile_id"] == "t01")
        if "t01" not in t00["neighbors"]:
            t00["neighbors"].append("t01")
        t01["neighbors"] = [n for n in t01["neighbors"] if n != "t00"]
        manifest_p.write_text(json.dumps(doc))
        cfg = load_config(ws / "config.yaml")
        manifest = load_manifest(manifest_p, ws)
        _, failures = validate_and_load(manifest, cfg, ws)
        self.assertIn(Code.E_ADJ_NOT_SYMMETRIC, failures.codes())

    def test_duplicate_tile_id(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        manifest_p = ws / "samples" / "manifest.json"
        doc = json.loads(manifest_p.read_text())
        dup = dict(doc["tiles"][0])
        doc["tiles"].append(dup)
        manifest_p.write_text(json.dumps(doc))
        cfg = load_config(ws / "config.yaml")
        manifest = load_manifest(manifest_p, ws)
        _, failures = validate_and_load(manifest, cfg, ws)
        self.assertIn(Code.E_TILE_ID_DUPLICATE, failures.codes())

    def test_window_not_divisible(self):
        ws = make_workspace(Path(tempfile.mkdtemp()))
        text = (ws / "config.yaml").read_text().replace("size: 8", "size: 7")
        (ws / "win.yaml").write_text(text)
        cfg = load_config(ws / "win.yaml")
        manifest = load_manifest(to_path(cfg.manifest_path, ws), ws)
        _, failures = validate_and_load(manifest, cfg, ws)
        self.assertIn(Code.E_WINDOW_NOT_DIVISIBLE, failures.codes())


if __name__ == "__main__":
    unittest.main()
