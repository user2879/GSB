"""Fault injection.

Injections mutate only input files or spec parameters - never source code or
engine logic.  Each variant is materialized into an isolated workspace copy so
the original fixtures stay byte-stable for determinism checks.
"""

from __future__ import annotations

import json
import os
import shutil
from typing import Any

from .errors import WorkbenchError

INJECTIONS = ("data", "param", "outlier")


def prepare_injection(spec_path: str, target_dir: str,
                      kind: str) -> str:
    if kind not in INJECTIONS:
        raise WorkbenchError("E_PARAM",
                             f"unknown injection kind {kind!r}",
                             field_path="$.inject")
    os.makedirs(target_dir, exist_ok=True)
    with open(spec_path, "r", encoding="utf-8") as handle:
        spec = json.load(handle)
    sample_dir = os.path.join(target_dir, "samples")
    os.makedirs(sample_dir, exist_ok=True)
    base_dir = os.path.dirname(os.path.abspath(spec_path))
    for source in spec["sources"]:
        rel = source["path"]
        source_file = rel if os.path.isabs(rel) else os.path.join(
            base_dir, rel)
        dest = os.path.join(sample_dir, os.path.basename(rel))
        shutil.copy2(source_file, dest)
        source["path"] = os.path.join("samples", os.path.basename(rel))
    if kind == "data":
        _inject_data(spec, sample_dir)
    elif kind == "param":
        _inject_param(spec)
    else:
        _inject_outlier(spec, sample_dir)
    dest_spec = os.path.join(target_dir, "pipeline.json")
    with open(dest_spec, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(spec, handle, indent=2, sort_keys=False)
    return dest_spec


def _inject_data(spec: dict[str, Any], sample_dir: str) -> None:
    # Truncate the first sample file: corrupt input bytes only.
    first = spec["sources"][0]
    path = os.path.join(sample_dir, os.path.basename(first["path"]))
    with open(path, "r", encoding="utf-8") as handle:
        content = handle.read()
    cut = max(1, len(content) // 2)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(content[:cut])


def _inject_param(spec: dict[str, Any]) -> None:
    for stage in spec["stages"]:
        if stage["type"] == "normalize":
            # Bounds for scale are [0, 9]; 42 is an explicit out-of-range
            # integer and points at the named parameter.
            stage["params"]["scale"] = 42
            return
    raise WorkbenchError("E_PARAM", "pipeline has no normalize stage",
                         field_path="$.stages[*].params.scale")


def _inject_outlier(spec: dict[str, Any], sample_dir: str) -> None:
    # Amplify a float column to extreme values while keeping the file valid;
    # the clip stage's basis-point threshold then trips E_OUTLIER.
    first = spec["sources"][0]
    path = os.path.join(sample_dir, os.path.basename(first["path"]))
    with open(path, "r", encoding="utf-8") as handle:
        document = json.load(handle)
    float_columns = [name for name, meta in first["columns"].items()
                     if meta["type"] == "float"]
    target = float_columns[0]
    for row in document["rows"]:
        if row.get(target) is not None:
            row[target] = 999.9
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(document, handle, indent=2)
