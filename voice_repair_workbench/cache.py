"""阶段级缓存、参数指纹与失效闭包。

缓存键 = 素材内容哈希 + 工程参数指纹 + 引擎版本戳 + 阶段名 + 上游阶段键。
每个阶段条目记录输入哈希、输出哈希、版本戳与复用来源，保证：
  * 参数/版本变化后精确计算失效闭包，只重算闭包内阶段；
  * 闭包外阶段复用，并记录复用来源与版本戳；
  * 缓存文件被篡改导致哈希不匹配时抛 E_CACHE_HASH。
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import config
from .errors import E_CACHE_HASH, RepairError
from .hashing import sha256_array, sha256_json
from .pipeline import STAGE_ORDER


def stage_param_subset(project_v2: dict, stage: str) -> dict:
    if stage == "denoise":
        return {"denoise": project_v2["denoise"], "stft": project_v2["stft"]}
    if stage == "gate":
        return {"gate": project_v2["gate"]}
    if stage == "normalize":
        return {
            "normalize": project_v2["normalize"],
            "channel_order": project_v2["channel_order"],
        }
    if stage == "limit":
        return {"limit": project_v2["limit"]}
    raise KeyError(stage)


def compute_invalidation_closure(
    changed_stages: list[str],
) -> list[str]:
    """阶段为线性串联，闭包 = 最早变化阶段及其全部下游。"""
    if not changed_stages:
        return []
    earliest = min(STAGE_ORDER.index(s) for s in changed_stages)
    return list(STAGE_ORDER[earliest:])


class StageCache:
    """文件系统阶段缓存，单工作区目录内原子读写。"""

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _data_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.npy")

    def _meta_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.meta.json")

    def stage_key(
        self,
        stage: str,
        content_hash: str,
        project_v2: dict,
        upstream_key: str | None,
    ) -> str:
        payload = {
            "engine_version": config.ENGINE_VERSION,
            "stage": stage,
            "content_hash": content_hash,
            "params": stage_param_subset(project_v2, stage),
            "upstream_key": upstream_key,
        }
        return f"{stage}_{sha256_json(payload)[:16]}"

    def has(self, key: str) -> bool:
        return os.path.exists(self._data_path(key)) and os.path.exists(self._meta_path(key))

    def put(
        self,
        key: str,
        stage: str,
        data: np.ndarray,
        input_hash: str,
        upstream_key: str | None,
        project_v2: dict,
    ) -> dict:
        out_hash = sha256_array(data)
        final_data = self._data_path(key)
        tmp_data = final_data + ".tmp"
        with open(tmp_data, "wb") as fh:
            np.save(fh, data)
        os.replace(tmp_data, final_data)
        meta = {
            "key": key,
            "stage": stage,
            "engine_version": config.ENGINE_VERSION,
            "input_hash": input_hash,
            "upstream_key": upstream_key,
            "output_hash": out_hash,
            "params_fingerprint": sha256_json(stage_param_subset(project_v2, stage)),
        }
        tmp_meta = self._meta_path(key) + ".tmp"
        with open(tmp_meta, "w", encoding="utf-8") as fh:
            fh.write(json.dumps(meta, sort_keys=True, indent=2))
        os.replace(tmp_meta, self._meta_path(key))
        return meta

    def get(self, key: str) -> tuple[np.ndarray, dict]:
        with open(self._meta_path(key), "r", encoding="utf-8") as fh:
            meta = json.load(fh)
        data = np.load(self._data_path(key), allow_pickle=False)
        actual_hash = sha256_array(data)
        if actual_hash != meta["output_hash"]:
            raise RepairError(
                E_CACHE_HASH,
                f"cache content hash mismatch for stage {meta['stage']}",
                field=meta["stage"],
                context={
                    "expected": meta["output_hash"],
                    "actual": actual_hash,
                    "key": key,
                },
            )
        if config.ENGINE_VERSION != meta["engine_version"]:
            raise RepairError(
                E_CACHE_HASH,
                "cache built by a different engine version; refuse to reuse",
                field="engine_version",
                context={
                    "cached": meta["engine_version"],
                    "current": config.ENGINE_VERSION,
                },
            )
        return data, meta

    def list_entries(self) -> list[dict]:
        entries = []
        for name in sorted(os.listdir(self.cache_dir)):
            if name.endswith(".meta.json"):
                with open(os.path.join(self.cache_dir, name), "r", encoding="utf-8") as fh:
                    entries.append(json.load(fh))
        return entries

    def delete_orphans(self, live_keys: set[str]) -> list[str]:
        """删除不属于 live_keys 的缓存文件，返回被回收的文件名。"""
        orphans = []
        for name in sorted(os.listdir(self.cache_dir)):
            key = name.split(".")[0]
            if key not in live_keys:
                orphans.append(name)
                os.remove(os.path.join(self.cache_dir, name))
        return orphans

    def wipe_uncommitted(self) -> list[str]:
        """崩溃恢复时回收所有 .tmp（未提交）文件。"""
        leftovers = []
        for name in sorted(os.listdir(self.cache_dir)):
            if name.endswith(".tmp"):
                leftovers.append(name)
                os.remove(os.path.join(self.cache_dir, name))
        return leftovers
