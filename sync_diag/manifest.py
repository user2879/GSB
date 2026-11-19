from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import config_content_hash
from .errors import SyncError, require
from .hashing import sha256_file, sha256_json


def write_manifest(path: Path, manifest: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    path.write_bytes(data)
    digest = sha256_file(path)
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def load_manifest(path: Path, config: dict[str, Any] | None = None) -> tuple[dict[str, Any], str]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        expected = sidecar.read_text(encoding="ascii").strip()
    except (OSError, json.JSONDecodeError) as exc:
        raise SyncError("E_MANIFEST_HASH", "cannot read manifest or hash sidecar", {"path": str(path), "reason": str(exc)}) from exc
    actual = sha256_file(path)
    require(actual == expected, "E_MANIFEST_HASH", "manifest sidecar hash mismatch", path=str(path), expected=expected, actual=actual)
    if config is not None:
        require(
            manifest.get("config_sha256") == config_content_hash(config),
            "E_MANIFEST_HASH",
            "manifest/config hash mismatch",
            manifest_config_sha256=manifest.get("config_sha256"),
            config_sha256=config_content_hash(config),
        )
    require(manifest.get("channel_count") == 8, "E_CH_FORMAT", "manifest channel count mismatch", actual=manifest.get("channel_count"))
    require(manifest.get("bit_depth") == 24, "E_CH_FORMAT", "manifest bit depth mismatch", actual=manifest.get("bit_depth"))
    return manifest, actual


def manifest_content_hash(manifest: dict[str, Any]) -> str:
    return sha256_json(manifest)
