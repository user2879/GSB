"""阶段化快照持久化与强制终止后的恢复判定。

提交协议
--------
1. 每次运行打开一个新的暂存目录 ``snapshots/tmp-<pid>/``，
   先把上一版已完整提交的产物复制进来（命中即复用，不重算）。
2. 阶段产物以 ``<stage>.json``（+ 可选 ``<stage>.npy``）写入暂存目录，
   单文件使用临时文件 + ``os.replace`` 原子落盘。
3. 每完成一个阶段立即刷新暂存清单（含全局版本号、配置哈希、已完成阶段、
   步序/帧序游标）。
4. 运行结束后把暂存目录整体原子替换为新的已提交快照，并切换 ``current``
   符号链接；强制终止只会留下不会被恢复逻辑读取的 ``tmp-*`` 目录。
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import uuid
from typing import Any, Mapping

import numpy as np

from .hashing import fingerprint, fingerprint_arrays
from .stages import StageSpec

MANIFEST_NAME = "manifest.json"
ARTIFACT_DIRNAME = "artifacts"
CURRENT_LINK = "current"


class SnapshotError(RuntimeError):
    pass


class SnapshotStore:
    def __init__(self, workdir: str) -> None:
        self.workdir = os.path.abspath(workdir)
        self.snapshot_root = os.path.join(self.workdir, "snapshots")
        os.makedirs(self.snapshot_root, exist_ok=True)
        self.current_path = os.path.join(self.snapshot_root, CURRENT_LINK)
        self.staging_path: str | None = None
        self.staging_artifacts: str | None = None
        self._staging_manifest: dict[str, Any] | None = None

    # ---- 快照定位与恢复 ----------------------------------------------------
    def current_snapshot_dir(self) -> str | None:
        if os.path.islink(self.current_path) or os.path.isdir(self.current_path):
            target = os.path.realpath(self.current_path)
            if os.path.isdir(target):
                return target
        # 兼容无符号链接环境：查找已提交的 v* 目录。
        committed = sorted(
            name
            for name in os.listdir(self.snapshot_root)
            if name.startswith("v")
            and os.path.isdir(os.path.join(self.snapshot_root, name))
        )
        return os.path.join(self.snapshot_root, committed[-1]) if committed else None

    def load_committed_manifest(self) -> dict[str, Any] | None:
        snapshot_dir = self.current_snapshot_dir()
        if snapshot_dir is None:
            return None
        manifest_path = os.path.join(snapshot_dir, MANIFEST_NAME)
        if not os.path.isfile(manifest_path):
            return None
        with open(manifest_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def load_artifact(self, stage_name: str) -> dict[str, Any] | None:
        snapshot_dir = self.current_snapshot_dir()
        if snapshot_dir is None:
            return None
        meta_path = os.path.join(
            snapshot_dir, ARTIFACT_DIRNAME, f"{stage_name}.json"
        )
        if not os.path.isfile(meta_path):
            return None
        with open(meta_path, "r", encoding="utf-8") as handle:
            envelope = json.load(handle)
        array_rel = envelope.get("array_file")
        if array_rel:
            array_path = os.path.join(snapshot_dir, ARTIFACT_DIRNAME, array_rel)
            envelope["array"] = np.load(array_path)
        return envelope

    def recovery_point(self) -> dict[str, Any]:
        """仅依据已完整替换的产物与进度记录判定恢复点。"""
        manifest = self.load_committed_manifest()
        if manifest is None:
            return {
                "recoverable": False,
                "version": 0,
                "completed_stages": [],
                "step_cursor": -1,
                "frame_cursor": -1,
                "config_hash": None,
                "message": "无已提交快照，从空白全量渲染",
            }
        completed = sorted(
            manifest.get("completed_stages", []),
            key=lambda name: manifest["stage_sort_keys"].get(name, name),
        )
        # 校验清单内产物确实完整存在且指纹自洽，损坏产物不参与恢复。
        verified: list[str] = []
        snapshot_dir = self.current_snapshot_dir()
        assert snapshot_dir is not None
        for stage_name in completed:
            envelope = self.load_artifact(stage_name)
            if envelope is None or not self._envelope_intact(snapshot_dir, envelope):
                continue
            verified.append(stage_name)
        return {
            "recoverable": bool(verified),
            "version": manifest.get("global_version", 0),
            "config_hash": manifest.get("config_hash"),
            "completed_stages": verified,
            "step_cursor": manifest.get("step_cursor", -1),
            "frame_cursor": manifest.get("frame_cursor", -1),
            "message": f"从已提交快照 v{manifest.get('global_version', 0)} 恢复",
        }

    def _envelope_intact(self, snapshot_dir: str, envelope: Mapping[str, Any]) -> bool:
        artifact_dir = os.path.join(snapshot_dir, ARTIFACT_DIRNAME)
        if not os.path.isfile(os.path.join(artifact_dir, f"{envelope['stage']}.json")):
            return False
        array_rel = envelope.get("array_file")
        if array_rel and not os.path.isfile(os.path.join(artifact_dir, array_rel)):
            return False
        return True

    # ---- 暂存区生命周期 ----------------------------------------------------
    def open_staging(
        self,
        *,
        global_version: int,
        config_hash: str,
        stage_specs: list[StageSpec],
    ) -> dict[str, Any]:
        staging = os.path.join(self.snapshot_root, f"tmp-{os.getpid()}")
        if os.path.exists(staging):
            shutil.rmtree(staging)
        os.makedirs(os.path.join(staging, ARTIFACT_DIRNAME))
        previous = self.current_snapshot_dir()
        if previous is not None:
            previous_artifacts = os.path.join(previous, ARTIFACT_DIRNAME)
            for name in os.listdir(previous_artifacts):
                os.link(
                    os.path.join(previous_artifacts, name),
                    os.path.join(staging, ARTIFACT_DIRNAME, name),
                )
        self.staging_path = staging
        self.staging_artifacts = os.path.join(staging, ARTIFACT_DIRNAME)
        self._staging_manifest = {
            "global_version": global_version,
            "config_hash": config_hash,
            "completed_stages": [],
            "stage_sort_keys": {
                spec.name: list(spec.sort_key) for spec in stage_specs
            },
            "step_cursor": -1,
            "frame_cursor": -1,
            "committed": False,
        }
        # 继承上一版已完成阶段（暂存目录中已有对应硬链接产物）。
        previous_manifest = self.load_committed_manifest()
        if previous_manifest is not None:
            for stage_name in previous_manifest.get("completed_stages", []):
                if os.path.isfile(
                    os.path.join(self.staging_artifacts, f"{stage_name}.json")
                ):
                    self._staging_manifest["completed_stages"].append(stage_name)
            self._staging_manifest["step_cursor"] = previous_manifest.get(
                "step_cursor", -1
            )
            self._staging_manifest["frame_cursor"] = previous_manifest.get(
                "frame_cursor", -1
            )
        self._flush_manifest()
        return self._staging_manifest

    def stage_artifact(
        self,
        spec: StageSpec,
        envelope: Mapping[str, Any],
        array: np.ndarray | None = None,
    ) -> dict[str, Any]:
        if self.staging_path is None:
            raise SnapshotError("暂存区尚未打开")
        record = dict(envelope)
        record["stage"] = spec.name
        record.setdefault("kind", spec.kind)
        record["step"] = spec.step
        record["frame"] = spec.frame
        record["sort_key"] = list(spec.sort_key)

        if array is not None:
            array_name = f"{spec.name}.npy"
            self._atomic_write_npy(os.path.join(self.staging_artifacts, array_name), array)
            record["array_file"] = array_name
            record["array_fingerprint"] = fingerprint_arrays([array])
            record["summary"] = _array_summary(array)
        # 内容指纹只由阶段实际产物决定；缓存键/依赖清单属于寻址信息，
        # 纳入指纹会让相同数据在增量链与全量链中产生不同指纹并污染下游。
        record["content_fingerprint"] = fingerprint(
            {
                key: value
                for key, value in record.items()
                if key
                not in (
                    "content_fingerprint",
                    "cache_key",
                    "dependency_fingerprints",
                    "param_fingerprints",
                )
            }
        )
        meta_name = f"{spec.name}.json"
        self._atomic_write_json(
            os.path.join(self.staging_artifacts, meta_name), record
        )

        manifest = self._staging_manifest
        assert manifest is not None
        if spec.name not in manifest["completed_stages"]:
            manifest["completed_stages"].append(spec.name)
        if spec.kind == "denoise" and spec.step is not None:
            manifest["step_cursor"] = max(manifest["step_cursor"], spec.step)
        if spec.kind == "decode" and spec.frame is not None:
            manifest["frame_cursor"] = max(manifest["frame_cursor"], spec.frame)
        self._flush_manifest()
        return record

    def has_staged_artifact(self, stage_name: str) -> bool:
        return self.staging_artifacts is not None and os.path.isfile(
            os.path.join(self.staging_artifacts, f"{stage_name}.json")
        )

    def load_staged_artifact(self, stage_name: str) -> dict[str, Any] | None:
        if not self.has_staged_artifact(stage_name):
            return None
        path = os.path.join(self.staging_artifacts, f"{stage_name}.json")
        with open(path, "r", encoding="utf-8") as handle:
            envelope = json.load(handle)
        array_rel = envelope.get("array_file")
        if array_rel:
            envelope["array"] = np.load(
                os.path.join(self.staging_artifacts, array_rel)
            )
        return envelope

    def commit_stage(self) -> str:
        """阶段边界的整体原子提交：暂存目录整体替换并切换 current。

        强制终止可能发生在任意两次提交之间，恢复时只承认最后一次
        已完整替换的快照；未提交的 tmp-* 目录不会被读取。
        """
        if self.staging_path is None:
            raise SnapshotError("暂存区尚未打开")
        manifest = self._staging_manifest
        assert manifest is not None
        self._flush_manifest()
        version = manifest["global_version"]
        target = os.path.join(self.snapshot_root, f"v{version:06d}")
        if os.path.exists(target):
            shutil.rmtree(target)
        # 快照目录由暂存区硬链接克隆得到（零拷贝、同设备原子 rename 切换）。
        _hardlink_clone(self.staging_path, target)
        self._switch_current(target)
        # 重建暂存区为刚提交快照的独立硬链接副本，确保后续阶段原子替换
        # 断开的是暂存区 inode，已提交快照内容保持不可变。
        rebuild = f"{self.staging_path}.rebuild"
        if os.path.exists(rebuild):
            shutil.rmtree(rebuild)
        _hardlink_clone(target, rebuild)
        shutil.rmtree(self.staging_path)
        os.replace(rebuild, self.staging_path)
        self.staging_artifacts = os.path.join(self.staging_path, ARTIFACT_DIRNAME)
        return target

    def commit(self) -> str:
        if self.staging_path is None:
            raise SnapshotError("暂存区尚未打开")
        manifest = self._staging_manifest
        assert manifest is not None
        manifest["committed"] = True
        self._flush_manifest()

        version = manifest["global_version"]
        target = os.path.join(self.snapshot_root, f"v{version:06d}")
        if os.path.exists(target):
            shutil.rmtree(target)
        _hardlink_clone(self.staging_path, target)
        self._switch_current(target)
        staging = self.staging_path
        self.staging_path = None
        self.staging_artifacts = None
        self._staging_manifest = None
        if staging and os.path.isdir(staging):
            shutil.rmtree(staging)
        return target

    def discard_staging(self) -> None:
        if self.staging_path and os.path.isdir(self.staging_path):
            shutil.rmtree(self.staging_path)
        self.staging_path = None
        self.staging_artifacts = None
        self._staging_manifest = None

    def cleanup_orphans(self) -> list[str]:
        orphans = [
            name
            for name in os.listdir(self.snapshot_root)
            if name.startswith("tmp-")
            and os.path.isdir(os.path.join(self.snapshot_root, name))
        ]
        for name in orphans:
            shutil.rmtree(os.path.join(self.snapshot_root, name))
        return sorted(orphans)

    def _switch_current(self, target: str) -> None:
        link = self.current_path
        tmp_link = os.path.join(self.snapshot_root, "current.tmp")
        try:
            if os.path.islink(tmp_link) or os.path.exists(tmp_link):
                os.remove(tmp_link)
            os.symlink(os.path.basename(target), tmp_link)
            os.replace(tmp_link, link)
        except OSError:
            # 文件系统不支持符号链接时退化为目录替换（仍为原子 rename）。
            if os.path.isdir(link) and not os.path.islink(link):
                shutil.rmtree(link)
            os.replace(target, link)

    def _flush_manifest(self) -> None:
        assert self.staging_path is not None and self._staging_manifest is not None
        path = os.path.join(self.staging_path, MANIFEST_NAME)
        self._atomic_write_json(path, self._staging_manifest)

    @staticmethod
    def _atomic_write_json(path: str, payload: Mapping[str, Any]) -> None:
        tmp_path = f"{path}.tmp-{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex[:8]}"
        with open(tmp_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)

    @staticmethod
    def _atomic_write_npy(path: str, array: np.ndarray) -> None:
        tmp_path = f"{path}.tmp-{os.getpid()}-{threading.get_ident()}-{uuid.uuid4().hex[:8]}"
        with open(tmp_path, "wb") as handle:
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)


def _array_summary(array: np.ndarray) -> dict[str, Any]:
    cast = array.astype(np.float64)
    if not array.size:
        stats = {"min": 0.0, "max": 0.0, "mean": 0.0}
    else:
        stats = {
            "min": float(cast.min()),
            "max": float(cast.max()),
            "mean": float(cast.mean()),
        }
    return {"shape": list(array.shape), "dtype": str(array.dtype), **stats}


def _hardlink_clone(src: str, dst: str) -> None:
    """目录树硬链接克隆：同设备下零拷贝，原子替换前的快照与暂存区共享 inode。"""
    os.makedirs(dst)
    for root, dirs, files in os.walk(src):
        relative = os.path.relpath(root, src)
        target_dir = os.path.join(dst, relative) if relative != "." else dst
        os.makedirs(target_dir, exist_ok=True)
        for name in files:
            source_file = os.path.join(root, name)
            target_file = os.path.join(target_dir, name)
            os.link(source_file, target_file)
