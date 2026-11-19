"""分块级持久化、版本识别与摘要校验（业务要求 5、6）。

状态目录布局::

    state/<version_id>/
        manifest.json          # 版本标识、步游标、块索引与摘要
        chunks/chunk-XXXX.json # 块状态 + 依赖 + 内容摘要（原子替换）
        interrupted.json       # 可处理终止标记
        orphans/               # 强制终止残留的无主块
"""
from __future__ import annotations

import os

from .atomicio import atomic_write_json, canonical_json, cleanup_temp_files, digest_of, read_json
from .errors import E_DIGEST, E_VERSION, WorkbenchError

CHUNK_PREFIX = "chunk-"
MANIFEST_NAME = "manifest.json"
INTERRUPT_NAME = "interrupted.json"


def version_id(score_dict: dict, params_dict: dict, edit_script_dict: dict) -> str:
    """由乐谱、表达参数与编辑脚本的规范化摘要构成版本标识。"""
    payload = {
        "score": score_dict,
        "params": params_dict,
        "edit_script": edit_script_dict,
        "schema": "mwb-state/v1",
    }
    return digest_of(payload)[:16]


def chunk_digest(chunk: dict) -> str:
    """块内容摘要：不含 digest 字段本身。"""
    body = {k: v for k, v in chunk.items() if k != "digest"}
    return digest_of(body)


class ChunkStore:
    """管理单个版本目录下的状态块原子读写。"""

    def __init__(self, state_root: str, version: str):
        self.state_root = state_root
        self.version = version
        self.version_dir = os.path.join(state_root, version)
        self.chunk_dir = os.path.join(self.version_dir, "chunks")
        self.orphan_dir = os.path.join(self.version_dir, "orphans")
        self.manifest_path = os.path.join(self.version_dir, MANIFEST_NAME)
        self.interrupt_path = os.path.join(self.version_dir, INTERRUPT_NAME)
        os.makedirs(self.chunk_dir, exist_ok=True)

    @classmethod
    def open(cls, state_root: str, version: str) -> "ChunkStore":
        os.makedirs(state_root, exist_ok=True)
        cleanup_temp_files(state_root)
        for root, _dirs, files in os.walk(state_root):
            cleanup_temp_files(root)
        existing = [d for d in os.listdir(state_root) if os.path.isdir(os.path.join(state_root, d))]
        if existing and version not in existing:
            # 旧版本目录原样保留，绝不覆盖；新版本另起目录
            pass
        return cls(state_root, version)

    def chunk_path(self, index: int) -> str:
        return os.path.join(self.chunk_dir, f"{CHUNK_PREFIX}{index:04d}.json")

    def write_manifest(self, manifest: dict) -> None:
        atomic_write_json(self.manifest_path, manifest)

    def read_manifest(self) -> dict | None:
        if not os.path.exists(self.manifest_path):
            return None
        return read_json(self.manifest_path)

    def write_chunk(self, chunk: dict) -> None:
        index = chunk["chunk_index"]
        if chunk["declared_ticks"] != chunk["tick_hi"] - chunk["tick_lo"]:
            raise WorkbenchError(E_DIGEST, "块声明 tick 数与区间不符", {"chunk_index": index})
        if len(chunk["tempo"]) != chunk["declared_ticks"] or len(chunk["pedal"]) != chunk["declared_ticks"]:
            raise WorkbenchError(E_DIGEST, "块实际 tick 数与声明不符", {"chunk_index": index})
        payload = dict(chunk)
        payload["digest"] = chunk_digest(chunk)
        atomic_write_json(self.chunk_path(index), payload)

    def read_chunk(self, index: int) -> dict | None:
        path = self.chunk_path(index)
        if not os.path.exists(path):
            return None
        return read_json(path)

    def verify_chunk(self, index: int) -> dict:
        """读取并校验块摘要，失败返回 E_DIGEST，由调用方回退补算。"""
        chunk = self.read_chunk(index)
        if chunk is None:
            raise WorkbenchError(E_DIGEST, "状态块缺失", {"chunk_index": index})
        stored = chunk.get("digest")
        actual = chunk_digest(chunk)
        if stored != actual:
            raise WorkbenchError(E_DIGEST, "块内容摘要不匹配", {"chunk_index": index, "stored": stored, "actual": actual})
        if chunk["declared_ticks"] != chunk["tick_hi"] - chunk["tick_lo"]:
            raise WorkbenchError(E_DIGEST, "块声明 tick 数与区间不符", {"chunk_index": index})
        return chunk

    def list_chunk_indices(self) -> list[int]:
        if not os.path.isdir(self.chunk_dir):
            return []
        indices = []
        for name in os.listdir(self.chunk_dir):
            if name.startswith(CHUNK_PREFIX) and name.endswith(".json"):
                indices.append(int(name[len(CHUNK_PREFIX): -len(".json")]))
        return sorted(indices)

    def quarantine_orphans(self, valid_indices: set[int]) -> list[int]:
        """把无主块（强制终止时已写块但未登记 manifest）移入 orphans。"""
        orphans = []
        for index in self.list_chunk_indices():
            if index not in valid_indices:
                os.makedirs(self.orphan_dir, exist_ok=True)
                src = self.chunk_path(index)
                dst = os.path.join(self.orphan_dir, os.path.basename(src))
                try:
                    os.replace(src, dst)
                    orphans.append(index)
                except OSError:
                    pass
        return orphans

    def write_interrupt_marker(self, cursor: dict) -> None:
        atomic_write_json(self.interrupt_path, {"status": "interrupted", "cursor": cursor})

    def interrupt_marker(self) -> dict | None:
        if os.path.exists(self.interrupt_path):
            return read_json(self.interrupt_path)
        return None

    def clear_interrupt_marker(self) -> None:
        if os.path.exists(self.interrupt_path):
            os.unlink(self.interrupt_path)
