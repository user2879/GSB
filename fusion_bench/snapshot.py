"""阶段化快照：产物先写临时目录再整体替换提交；清单记录全局版本号、
配置哈希、输入指纹、已完成样本、已消耗预算与各阶段产物版本。"""

import hashlib
import json
import os

SNAPSHOT_STAGES = ("load_validate", "feature_extract", "fusion_score",
                   "summarize")


def _atomic_write_json(path, payload):
    tmp_dir = os.path.join(os.path.dirname(path), "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    tmp_path = os.path.join(tmp_dir, os.path.basename(path) + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, sort_keys=True, indent=1)
    os.replace(tmp_path, path)


def _hash_payload(payload):
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    return hashlib.sha256(blob).hexdigest()


class SnapshotManager:
    def __init__(self, snap_dir):
        self.snap_dir = snap_dir
        os.makedirs(os.path.join(snap_dir, "tmp"), exist_ok=True)
        self.manifest_path = os.path.join(snap_dir, "manifest.json")
        self.completed_path = os.path.join(snap_dir, "completed.json")

    def load_manifest(self):
        if not os.path.exists(self.manifest_path):
            return None
        with open(self.manifest_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def load_completed(self):
        if not os.path.exists(self.completed_path):
            return {"results": [], "failures": [], "ledger": [],
                    "cache_evictions": [], "cache_recomputes": []}
        with open(self.completed_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def commit_stage(self, stage, artifact):
        """单阶段产物：临时目录写入后整体替换，并更新清单中的阶段版本。"""
        if stage not in SNAPSHOT_STAGES:
            raise ValueError(f"unknown stage {stage}")
        path = os.path.join(self.snap_dir, f"stage_{stage}.json")
        _atomic_write_json(path, artifact)
        manifest = self.load_manifest()
        if manifest is not None:
            manifest.setdefault("stage_versions", {})[stage] = \
                _hash_payload(artifact)
            _atomic_write_json(self.manifest_path, manifest)

    def write_manifest(self, manifest):
        _atomic_write_json(self.manifest_path, manifest)

    def save_completed(self, completed):
        _atomic_write_json(self.completed_path, completed)
