"""Fault injection: faults are applied to in-memory copies of the inputs (or
to byte streams at read time); the workbench source and the on-disk samples
are never modified."""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from . import errors


class InjectionSet:
    def __init__(self, injections):
        self.injections = list(injections or [])
        self.hits = []

    @classmethod
    def load(cls, path):
        if path is None:
            return cls([])
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def _matching(self, tile_id, stage, fault):
        out = []
        for inj in self.injections:
            if inj["stage"] != stage or inj["fault"] != fault:
                continue
            if inj["tile"] not in ("*", tile_id):
                continue
            out.append(inj)
        return out

    # ---- config stage ------------------------------------------------------
    def patch_config_raw(self, raw: dict):
        raw = dict(raw)
        for inj in self.injections:
            if inj["stage"] == "config" and inj["fault"] == "negative_min_connected":
                raw["min_connected_pixels"] = int(inj["value"])
                self.hits.append(inj["inj_id"])
        return raw

    # ---- load stage --------------------------------------------------------
    def patch_manifest(self, manifest):
        for inj in self.injections:
            if inj["stage"] == "load" and inj["fault"] == "adjacency_unknown_ref":
                tid = inj["tile"]
                if tid in manifest.tiles:
                    manifest.tiles[tid].neighbors.append(inj["ghost"])
                    self.hits.append(inj["inj_id"])
        return manifest

    def band_reader(self, tile_id):
        """Return a band reader applying truncate_band_bytes faults."""
        def reader(path, tid, band):
            data = Path(path).read_bytes()
            for inj in self._matching(tile_id, "load", "truncate_band_bytes"):
                if inj["band"] == band:
                    data = data[: int(inj["keep_bytes"])]
                    self.hits.append(inj["inj_id"])
            try:
                with Image.open(io.BytesIO(data)) as img:
                    return np.asarray(img)
            except Exception as exc:
                from .loader import BandLoadError
                raise BandLoadError(errors.E_BAND_CORRUPT,
                                    f"band {band} undecodable: {exc}")
        return reader

    # ---- classify stage ----------------------------------------------------
    def patch_bands(self, tile_id, bands: dict):
        for inj in self._matching(tile_id, "classify", "nonfinite_band_value"):
            band = inj["band"]
            if band in bands:
                arr = bands[band].astype(np.float64)
                r, c = inj["position"]
                arr[r, c] = np.nan
                bands[band] = arr
                self.hits.append(inj["inj_id"])
        return bands

    def inj_id_for(self, tile_id, stage):
        for inj in self.injections:
            if inj["stage"] == stage and inj["tile"] in ("*", tile_id):
                return inj["inj_id"]
        return None
