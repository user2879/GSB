"""阶段化快照：临时目录写入、整体替换提交与恢复判定。

四个阶段（parse / attribution / resources / summary）各自产出不可变阶段
产物。写入先落临时目录，再以 ``os.replace`` 整体替换；清单记录全局版本号、
配置哈希、回执文件指纹、命令游标与各阶段产物版本，恢复时以哈希识别变化，
命中即复用，绝不逐文件覆盖。
"""

from __future__ import annotations

import json
import os

from .util import atomic_write_text, canonical_json, sha256_hex

STAGES = ("parse", "attribution", "resources", "summary")
MANIFEST_NAME = "manifest.json"


class SnapshotStore:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.stage_dir = os.path.join(output_dir, "stages")
        self.tmp_dir = os.path.join(output_dir, "tmp")
        os.makedirs(self.stage_dir, exist_ok=True)
        os.makedirs(self.tmp_dir, exist_ok=True)
        self.manifest = self._load_manifest()

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.output_dir, MANIFEST_NAME)

    def _load_manifest(self) -> dict:
        path = self.manifest_path
        if not os.path.exists(path):
            return {
                "global_version": 0,
                "config_hash": None,
                "receipt_fingerprints": [],
                "command_cursor": 0,
                "stages": {},
            }
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def can_resume(self, config_hash: str, fingerprints: list[dict]) -> bool:
        if self.manifest.get("config_hash") != config_hash:
            return False
        if canonical_json(self.manifest.get("receipt_fingerprints", [])) != canonical_json(
            fingerprints
        ):
            return False
        return os.path.isdir(self.stage_dir)

    def stage_path(self, stage: str, version: int) -> str:
        return os.path.join(self.stage_dir, f"{stage}.v{version}.json")

    def reusable_stage(self, stage: str, input_hash: str) -> dict | None:
        record = self.manifest["stages"].get(stage)
        if record is None or record.get("input_hash") != input_hash:
            return None
        path = self.stage_path(stage, record["version"])
        if not os.path.exists(path):
            return None
        with open(path, encoding="utf-8") as handle:
            return json.load(handle)

    def commit_stage(
        self,
        stage: str,
        input_hash: str,
        payload: dict,
        reused: bool,
        config_hash: str,
        fingerprints: list[dict],
        command_cursor: int,
    ) -> dict:
        previous = self.manifest["stages"].get(stage)
        version = previous["version"] + 1 if previous else 1
        target_path = self.stage_path(stage, version)

        tmp_path = os.path.join(self.tmp_dir, f"{stage}.v{version}.json.tmp")
        atomic_write_text(tmp_path, canonical_json(payload))
        os.replace(tmp_path, target_path)

        if previous:
            old_path = self.stage_path(stage, previous["version"])
            if os.path.exists(old_path):
                os.remove(old_path)

        self.manifest["global_version"] += 1
        self.manifest["config_hash"] = config_hash
        self.manifest["receipt_fingerprints"] = fingerprints
        self.manifest["command_cursor"] = command_cursor
        self.manifest["stages"][stage] = {
            "version": version,
            "input_hash": input_hash,
            "payload_hash": sha256_hex(payload),
            "reused": reused,
        }
        atomic_write_text(self.manifest_path, canonical_json(self.manifest))
        return self.manifest["stages"][stage]

    def save_report(self, name: str, payload: dict) -> str:
        path = os.path.join(self.output_dir, name)
        atomic_write_text(path, canonical_json(payload))
        return path

    def reset(self) -> None:
        for name in os.listdir(self.stage_dir):
            os.remove(os.path.join(self.stage_dir, name))
        self.manifest = {
            "global_version": 0,
            "config_hash": None,
            "receipt_fingerprints": [],
            "command_cursor": 0,
            "stages": {},
        }
        if os.path.exists(self.manifest_path):
            os.remove(self.manifest_path)
