"""阶段快照：临时目录写入 + os.replace 整体替换 + 版本清单恢复。"""

import json
import os
import tempfile

from .errors import PipelineError, E_SNAPSHOT_VERSION
from .utils import canonical_json

STAGES = ("segment", "feature", "quantize", "attach", "aggregate")
MANIFEST_NAME = "manifest.json"


class SnapshotStore:
    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def _stage_path(self, stage):
        return os.path.join(self.root, "{}_stage.json".format(stage))

    def _tmp_path(self):
        handle = tempfile.NamedTemporaryFile(
            mode="w", dir=self.root, delete=False, suffix=".tmp", encoding="utf-8")
        path = handle.name
        handle.close()
        return path

    def commit_stage(self, stage, payload, manifest):
        """产物先写同目录临时文件，再 os.replace 原子替换。"""
        tmp_payload = self._tmp_path()
        with open(tmp_payload, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_payload, self._stage_path(stage))
        tmp_manifest = self._tmp_path()
        with open(tmp_manifest, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, sort_keys=True, indent=2)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_manifest, os.path.join(self.root, MANIFEST_NAME))

    def load_manifest(self):
        path = os.path.join(self.root, MANIFEST_NAME)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_stage(self, stage):
        path = self._stage_path(stage)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def recovery_decision(self, config_hash, expected_version=None):
        """启动恢复判定：配置哈希不一致或版本回退则冷启动。"""
        manifest = self.load_manifest()
        if manifest is None:
            return {"action": "cold_start", "reason": "no_manifest", "manifest": None}
        if manifest.get("config_hash") != config_hash:
            return {"action": "cold_start", "reason": "config_hash_changed", "manifest": manifest}
        if expected_version is not None and manifest.get("global_version", 0) > expected_version:
            raise PipelineError(
                E_SNAPSHOT_VERSION,
                "快照版本 {} 新于请求版本 {}".format(
                    manifest.get("global_version"), expected_version),
                "snapshot", None,
            )
        missing = [stage for stage in STAGES if not os.path.exists(self._stage_path(stage))]
        if missing:
            return {"action": "resume_rebuild", "reason": "missing_stages:{}".format(",".join(missing)),
                    "manifest": manifest}
        return {"action": "resume", "reason": "manifest_match", "manifest": manifest}

    def write_marker(self, name, payload):
        path = os.path.join(self.root, name)
        tmp = self._tmp_path()
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True)
        os.replace(tmp, path)
        return path


def build_manifest(global_version, config, command_cursor, stage_versions, levels_by_audio):
    return {
        "global_version": int(global_version),
        "config_hash": config.hash,
        "quantization_levels": levels_by_audio,
        "command_cursor": int(command_cursor),
        "stage_versions": {stage: int(version) for stage, version in stage_versions.items()},
    }
