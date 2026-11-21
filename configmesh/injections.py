"""Fault injection utilities.

Injection only mutates input files or session parameters; platform source code
is never modified.  Each injector returns a descriptor describing exactly what
was changed so the structured log chain can record it.
"""
from __future__ import annotations

import json
import os
import shutil
from typing import Any

from .errors import BusinessError, ErrorCode


def inject_corrupt_manifest(manifest_path: str, backup_suffix: str = ".pristine") -> dict[str, Any]:
    """Truncate/garble the manifest JSON -> E_MANIFEST on next load."""
    backup = manifest_path + backup_suffix
    if not os.path.exists(backup):
        shutil.copy2(manifest_path, backup)
    with open(manifest_path, "w", encoding="utf-8") as handle:
        handle.write('{"versions": [ { "version": 1, "items": [ BROKEN_JSON,, ] }')
    descriptor = {
        "injection": "corrupt_manifest",
        "expected_code": ErrorCode.E_MANIFEST.value,
        "path": manifest_path,
        "backup": backup,
    }
    return descriptor


def inject_derived_cycle(manifest_path: str, backup_suffix: str = ".pristine") -> dict[str, Any]:
    """Rewrite a valid manifest so two derived items reference each other."""
    backup = manifest_path + backup_suffix
    if not os.path.exists(backup):
        shutil.copy2(manifest_path, backup)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    first_version = data["versions"][0]
    items = first_version["items"]
    derived_items = [item for item in items if item.get("derived_from")]
    base_items = [item for item in items if not item.get("derived_from")]
    if len(derived_items) >= 2:
        key_a = derived_items[0]["key"]
        key_b = derived_items[1]["key"]
        derived_items[0]["derived_from"] = {key_b: 1.0}
        derived_items[1]["derived_from"] = {key_a: 1.0}
    elif base_items and derived_items:
        base_key = base_items[0]["key"]
        derived_key = derived_items[0]["key"]
        base_items[0]["derived_from"] = {derived_key: 1.0}
        if "value" in base_items[0]:
            del base_items[0]["value"]
        derived_items[0]["derived_from"] = {base_key: 1.0}
    else:
        raise BusinessError(ErrorCode.E_MANIFEST,
                            "sample manifest needs at least two items for cycle injection",
                            [{"path": manifest_path}])
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2)
    return {
        "injection": "derived_cycle",
        "expected_code": ErrorCode.E_CYCLE.value,
        "path": manifest_path,
        "backup": backup,
    }


def illegal_grid_parameter() -> Any:
    """Return an illegal precision parameter -> E_GRID when validated."""
    return -3


def restore(manifest_path: str, backup_suffix: str = ".pristine") -> str:
    backup = manifest_path + backup_suffix
    if not os.path.exists(backup):
        raise FileNotFoundError(backup)
    shutil.copy2(backup, manifest_path)
    return manifest_path
