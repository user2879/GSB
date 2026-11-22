"""内容寻址缓存与阶段化快照。

- 每个阶段的缓存键由上游内容指纹、阶段配置指纹与量化档位派生；
- 产物先写临时目录，再以 os.replace 原子整体替换；
- 清单记录全局版本、配置/注入哈希、音频游标与各阶段产物版本。
"""

import json
import os
import shutil
import tempfile
import threading

from .codes import E_SNAPSHOT_STALE, PipelineError
from .util import short_hash

STAGES = ("features", "segments", "clusters", "attribution", "verification")


class ContentCache:
    def __init__(self, root):
        self.root = root
        self.objects = os.path.join(root, "objects")
        self.tmp = os.path.join(root, "tmp")
        os.makedirs(self.objects, exist_ok=True)
        os.makedirs(self.tmp, exist_ok=True)
        self.ledger = []
        self._lock = threading.Lock()

    def _path(self, key):
        return os.path.join(self.objects, f"{key}.json")

    def has(self, key):
        return os.path.exists(self._path(key))

    def get(self, key):
        with open(self._path(key), "r", encoding="utf-8") as fh:
            return json.load(fh)

    def put(self, key, product):
        target = self._path(key)
        if os.path.exists(target):
            return
        fd, tmp_path = tempfile.mkstemp(dir=self.tmp, suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(product, fh, ensure_ascii=False, sort_keys=True)
            os.replace(tmp_path, target)  # 原子整体替换
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def get_or_compute(self, key, producer, audio, stage, extra=None):
        """线程安全的缓存读取/计算，返回 (product, was_reuse, ledger_entry)。"""
        with self._lock:
            if self.has(key):
                product = self.get(key)
                entry = {
                    "audio": audio, "stage": stage, "cache_key": key,
                    "action": "reuse", "extra": extra,
                }
                self.ledger.append(entry)
                return product, True, entry
        product = producer()
        product["cache_key"] = key
        product["cache_stage"] = stage
        with self._lock:
            self.put(key, product)
            entry = {
                "audio": audio, "stage": stage, "cache_key": key,
                "action": "compute", "extra": extra,
            }
            self.ledger.append(entry)
        return product, False, entry


def make_cache_key(stage, upstream_key, config, extra=None):
    payload = {
        "stage": stage,
        "upstream": upstream_key,
        "stage_config": config.stage_fingerprint(stage),
        "frac_bits": config.frac_bits,
        "extra": extra or {},
    }
    return f"{stage}_{short_hash(payload)}"


class Snapshot:
    MANIFEST = "manifest.json"

    def __init__(self, root):
        self.root = root
        os.makedirs(root, exist_ok=True)
        self.manifest_path = os.path.join(root, self.MANIFEST)

    def exists(self):
        return os.path.exists(self.manifest_path)

    def load(self):
        with open(self.manifest_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def save(self, manifest):
        fd, tmp_path = tempfile.mkstemp(dir=self.root, suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(manifest, fh, ensure_ascii=False, indent=2, sort_keys=True)
            os.replace(tmp_path, self.manifest_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    def initial_manifest(self, config_hash, injection_hash, audio_catalog):
        return {
            "global_version": 1,
            "config_hash": config_hash,
            "injection_hash": injection_hash,
            "audio_catalog": audio_catalog,
            "completed_audio_cursor": [],
            "stage_versions": {stage: 1 for stage in STAGES},
            "result_index": {},
            "failure_index": [],
            "ledger": [],
        }

    def assert_resumable(self, config_hash, injection_hash, audio_catalog, allow_edit=False):
        """启动恢复判定：版本/配置哈希不一致即 E_SNAPSHOT_STALE。"""
        if not self.exists():
            return None
        manifest = self.load()
        mismatch = (manifest["config_hash"] != config_hash
                    or manifest["injection_hash"] != injection_hash
                    or manifest.get("audio_catalog") != audio_catalog)
        if mismatch and not allow_edit:
            raise PipelineError(
                E_SNAPSHOT_STALE, "snapshot",
                "快照配置哈希、注入哈希或音频目录与当前输入不一致，禁止静默续跑",
            )
        return manifest

    def reset(self):
        if os.path.exists(self.root):
            shutil.rmtree(self.root)
        os.makedirs(self.root, exist_ok=True)
