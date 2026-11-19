"""Fault injection.

Injections only mutate *input files or parameters* (never platform source and
never the recorded ledger).  The injected files are written to a separate
working copy so the pristine samples stay untouched.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from .errors import E_GRID, ErrorDetail, PlatformError


def inject_corrupt_manifest(src: str | Path, dst: str | Path) -> Path:
    """Truncate JSON so parsing fails -> E_MANIFEST."""
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    text = Path(src).read_text(encoding="utf-8")
    dst_path.write_text(text[: max(1, len(text) // 2)], encoding="utf-8")
    return dst_path


def inject_cycle(src: str | Path, dst: str | Path) -> Path:
    """Rewrite the first item so it derives from a later item, closing a ring."""
    dst_path = Path(dst)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(Path(src).read_text(encoding="utf-8"))
    first_version = data["versions"][0]
    items = first_version["items"]
    first_key = items[0]["key"]
    target_key = items[1]["key"]
    # Make item[0] derived from item[1]; item[1] already (or will) point back.
    items[0].pop("value", None)
    items[0]["derivation"] = {
        "offset": 0,
        "coefficients": {target_key: 1},
    }
    if items[1].get("derivation") is None:
        items[1].pop("value", None)
        items[1]["derivation"] = {
            "offset": 0,
            "coefficients": {first_key: 1},
        }
    else:
        items[1]["derivation"]["coefficients"][first_key] = 1
    dst_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return dst_path


def validate_grid_q(raw: object) -> int:
    """Validate the precision grid exponent -> E_GRID when illegal."""
    if isinstance(raw, bool) or not isinstance(raw, int) or not 0 <= raw <= 12:
        raise PlatformError(
            E_GRID,
            "illegal precision grid parameter",
            [
                ErrorDetail(
                    reason=f"grid exponent must be int in [0,12], got {raw!r}",
                    source="grid",
                )
            ],
        )
    return raw


def prepare_manifest_injection(
    kind: str, samples_dir: Path, work_dir: Path
) -> Path:
    src = samples_dir / "manifest.json"
    if kind == "corrupt":
        return inject_corrupt_manifest(src, work_dir / "manifest-corrupt.json")
    if kind == "cycle":
        return inject_cycle(src, work_dir / "manifest-cycle.json")
    raise ValueError(f"unknown manifest injection: {kind}")


def copy_inputs(samples_dir: Path, work_dir: Path) -> tuple[Path, Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest = shutil.copy2(samples_dir / "manifest.json", work_dir / "manifest.json")
    registry = shutil.copy2(samples_dir / "registry.json", work_dir / "registry.json")
    return Path(manifest), Path(registry)
