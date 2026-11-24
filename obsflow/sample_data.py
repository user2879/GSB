"""Deterministic local sample generation (fixed integer seed).

All samples are derived from a single integer seed via NumPy's PCG64
generator, so repeated runs produce byte-identical files with no network or
external data dependencies.
"""
from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from .codec import write_stream
from .models import NormalizedRecord, SUPPORTED_SCALES

SEED = 20240517


def _rng(seed: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(seed))


def generate_samples(output_dir: str, *, seed: int = SEED,
                     records_per_source: int = 24) -> dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    rng = _rng(seed)

    specs = [
        # source_id, format version, default metric, default scale, mapping
        ("src-v1", 1, "temp", 100, {}),
        ("src-v2", 2, "pressure", 1_000, {}),
        ("src-v3", 3, "humidity", 10_000, {"raw_hum": "humidity"}),
    ]
    declarations: list[dict[str, Any]] = []
    manifest: dict[str, Any] = {"seed": seed, "sources": []}

    for source_index, (source_id, version, metric, scale,
                       mapping) in enumerate(specs):
        records: list[NormalizedRecord] = []
        seq = 1000 + source_index * 100
        ts = source_index * 50
        for index in range(records_per_source):
            seq += 1
            ts += int(rng.integers(5, 25))
            value = int(rng.integers(-500, 5_000))
            record_scale = scale
            metric_key = metric
            if version == 3 and index % 6 == 5:
                metric_key = "raw_hum"  # exercises the declaration mapping
            records.append(NormalizedRecord(
                source_id=source_id,
                seq=seq,
                ts_ms=ts,
                metric_key=metric_key,
                raw_value=value,
                record_scale=record_scale,
                source_format=version,
                filled_fields=(),
                fill_basis={},
                line_no=index + 2,
            ))

        file_name = f"{source_id}.jsonl"
        file_path = os.path.join(output_dir, file_name)
        write_stream(file_path, version, records)
        declaration = {
            "source_id": source_id,
            "format_version": version,
            "default_metric_key": metric,
            "default_scale": scale,
            "metric_mapping": mapping,
            "file_path": file_name,
        }
        declarations.append(declaration)
        manifest["sources"].append(
            {**declaration, "record_count": len(records)})

    source_set = {
        "source_set_version": 1,
        "sources": declarations,
    }
    declaration_path = os.path.join(output_dir, "sources.json")
    with open(declaration_path, "w", encoding="utf-8") as handle:
        json.dump(source_set, handle, indent=2, sort_keys=True)

    window_params = {
        "window_size": 5,
        "slide_step": 1,
        "scale": 100,
        "global_concurrency": 2,
        "per_source_concurrency": 1,
        "lease_ttl_ms": 60_000,
        "scale_convention_version": 1,
    }
    window_path = os.path.join(output_dir, "window_params.json")
    with open(window_path, "w", encoding="utf-8") as handle:
        json.dump(window_params, handle, indent=2, sort_keys=True)

    manifest["declaration_path"] = declaration_path
    manifest["window_path"] = window_path
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    return {
        "output_dir": output_dir,
        "declaration_path": declaration_path,
        "window_path": window_path,
        "manifest_path": manifest_path,
        "manifest": manifest,
        "supported_scales": list(SUPPORTED_SCALES),
    }
