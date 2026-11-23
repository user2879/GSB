"""Fault injection harness and diagnostic report.

An independent injection config names concrete input faults.  Each injection is
applied to a *throwaway copy* of the sample workspace so the clean dataset is
never mutated, and every injection id is echoed into the report to prove it was
hit and map it to its business code.  A single failing tile must not affect the
other tiles.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .config import load_config
from .errors import Code
from .paths import to_path, to_win
from .pipeline import Pipeline


@dataclass
class Injection:
    injection_id: str
    kind: str
    tile_id: str
    detail: Dict[str, Any]


def load_injections(path: str | Path) -> List[Injection]:
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        Injection(item["id"], item["kind"], item.get("tile_id", ""), item.get("detail", {}))
        for item in doc["injections"]
    ]


def apply_injection(root: Path, manifest_doc: dict, inj: Injection) -> None:
    tiles = {t["tile_id"]: t for t in manifest_doc["tiles"]}

    if inj.kind == "truncated_band":
        spec = tiles[inj.tile_id]
        band = inj.detail.get("band", "R")
        native = to_path(spec["bands"][band], root)
        raw = native.read_bytes()
        native.write_bytes(raw[: max(8, len(raw) // 3)])

    elif inj.kind == "missing_band":
        spec = tiles[inj.tile_id]
        band = inj.detail.get("band", "NIR")
        native = to_path(spec["bands"][band], root)
        if native.exists():
            native.unlink()
        # also drop the declaration to create an internal mismatch path
        spec["bands"].pop(band, None)

    elif inj.kind == "dangling_neighbor":
        spec = tiles[inj.tile_id]
        ghost = inj.detail.get("ghost_id", "t99")
        if ghost not in spec["neighbors"]:
            spec["neighbors"].append(ghost)

    elif inj.kind == "non_finite_weight":
        cfg_path = root / "configs" / "config.injected.yaml"
        text = cfg_path.read_text(encoding="utf-8")
        text = text.replace(
            "vegetation: 1.0",
            "vegetation: .nan",
        )
        cfg_path.write_text(text, encoding="utf-8")

    elif inj.kind == "window_not_divisible":
        cfg_path = root / "configs" / "config.injected.yaml"
        text = cfg_path.read_text(encoding="utf-8")
        text = text.replace("size: 8", "size: 7")
        cfg_path.write_text(text, encoding="utf-8")

    elif inj.kind == "resolution_mismatch":
        spec = tiles[inj.tile_id]
        band = inj.detail.get("band", "B")
        native = to_path(spec["bands"][band], root)
        arr = np.load(native)
        np.save(native, arr[:-1, :])

    else:
        raise ValueError(f"unknown injection kind {inj.kind}")


# Expected business code for every supported injection kind.
INJECTION_CODE = {
    "truncated_band": Code.E_BAND_TRUNCATED,
    "missing_band": Code.E_BAND_COUNT_MISMATCH,
    "dangling_neighbor": Code.E_ADJ_DANGLING,
    "non_finite_weight": Code.E_WEIGHT_INVALID,
    "window_not_divisible": Code.E_WINDOW_NOT_DIVISIBLE,
    "resolution_mismatch": Code.E_RESOLUTION_MISMATCH,
}


def run_injection_suite(base_root: Path, config_rel: str, manifest_rel: str,
                        injections: List[Injection], grid: tuple, edge: int, overlap: int,
                        report_path: Path) -> dict:
    entries: List[dict] = []
    for inj in injections:
        sandbox = base_root.parent / f"inj_sandbox_{inj.injection_id}"
        if sandbox.exists():
            shutil.rmtree(sandbox)
        shutil.copytree(base_root, sandbox)

        manifest_native = to_path(manifest_rel, sandbox)
        manifest_doc = json.loads(manifest_native.read_text(encoding="utf-8"))

        # ensure an injectable config copy exists
        src_cfg = to_path(config_rel, sandbox)
        inj_cfg = sandbox / "configs" / "config.injected.yaml"
        inj_cfg.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_cfg, inj_cfg)

        apply_injection(sandbox, manifest_doc, inj)
        manifest_native.write_text(json.dumps(manifest_doc, indent=2), encoding="utf-8")

        config_target = "configs\\config.injected.yaml" if inj.kind in (
            "non_finite_weight", "window_not_divisible") else config_rel

        entry: Dict[str, Any] = {
            "injection_id": inj.injection_id,
            "kind": inj.kind,
            "tile_id": inj.tile_id,
            "expected_code": INJECTION_CODE[inj.kind],
            "hit": False,
            "observed_codes": [],
            "other_tiles_still_classified": [],
            "status": "not_run",
        }
        try:
            config = load_config(to_path(config_target, sandbox))
            pipeline = Pipeline(config, sandbox, global_version=1)
            report = pipeline.run(grid, edge, overlap)
            observed = report.failures.codes()
            entry["observed_codes"] = observed
            entry["hit"] = INJECTION_CODE[inj.kind] in observed
            entry["other_tiles_still_classified"] = sorted(report.tile_results.keys())
            entry["status"] = "passed" if entry["hit"] else "missing_expected_code"
        except Exception as exc:  # config-level fatal faults land here
            code = getattr(exc, "code", "FATAL")
            entry["observed_codes"] = [code]
            entry["hit"] = code == INJECTION_CODE[inj.kind]
            entry["status"] = "passed" if entry["hit"] else f"fatal:{code}"
            entry["error"] = str(exc)
        entries.append(entry)

    summary = {
        "total": len(entries),
        "hit": sum(1 for e in entries if e["hit"]),
        "passed": sum(1 for e in entries if e["status"] == "passed"),
    }
    doc = {"summary": summary, "injections": entries}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    return doc
