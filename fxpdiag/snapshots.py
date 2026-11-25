"""阶段化快照：先写临时目录再整体替换，启动时按版本号+配置哈希决定续跑。"""

from __future__ import annotations

import json
import os
import shutil
import tempfile

from .fingerprints import config_hash

MANIFEST_NAME = "manifest.json"


def atomic_write_json(path: str, payload: dict) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def commit_snapshot(snapshot_dir: str, artifacts: dict[str, dict], manifest: dict) -> None:
    """把本阶段全部产物写入临时目录，再整体替换提交。"""

    parent = os.path.dirname(os.path.abspath(snapshot_dir)) or "."
    os.makedirs(parent, exist_ok=True)
    tmp_dir = tempfile.mkdtemp(prefix=".snapshot-", dir=parent)
    try:
        for name, payload in artifacts.items():
            atomic_write_json(os.path.join(tmp_dir, f"{name}.json"), payload)
        manifest = dict(manifest)
        manifest["config_hash"] = config_hash(manifest.get("config", {}))
        atomic_write_json(os.path.join(tmp_dir, MANIFEST_NAME), manifest)
        if os.path.exists(snapshot_dir):
            shutil.rmtree(snapshot_dir)
        os.replace(tmp_dir, snapshot_dir)
    except BaseException:
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        raise


def can_resume(snapshot_dir: str, expected_version: int, config_dict: dict) -> bool:
    manifest_path = os.path.join(snapshot_dir, MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        return False
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return False
    if manifest.get("config_version") != expected_version:
        return False
    return manifest.get("config_hash") == config_hash(config_dict)


def load_manifest(snapshot_dir: str) -> dict | None:
    manifest_path = os.path.join(snapshot_dir, MANIFEST_NAME)
    if not os.path.exists(manifest_path):
        return None
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)
