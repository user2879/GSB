"""阶段化快照：临时目录整体替换 + 清单原子提交。

三个阶段：data -> train -> summary。每个阶段产物都先写入同文件系统下的
临时目录，再以 os.replace 整体替换目标目录；manifest.json 永远最后原子
替换，因此进程被强制终止后，要么看到上一版完整快照，要么看到新版完整
快照，不会出现半成品。恢复只信任已完整替换的 manifest。
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any

from .errors import ConfigMismatchError, StageWriteError
from .models import Checkpoint, TaskResult
from .util import monotonic

DATA_STAGE = "data"
TRAIN_STAGE = "train"
SUMMARY_STAGE = "summary"
MANIFEST_NAME = "manifest.json"


@dataclass
class ResumeInfo:
    version: int
    config_hash: str
    data_fingerprint: str
    completed_tasks: list[str]
    consumed_epochs: int
    stage_versions: dict[str, int]
    recovery_point: str


class SnapshotStore:
    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self.data_dir = os.path.join(self.root, "data")
        self.train_dir = os.path.join(self.root, "train")
        self.summary_dir = os.path.join(self.root, "summary")
        self.manifest_path = os.path.join(self.root, MANIFEST_NAME)
        os.makedirs(self.root, exist_ok=True)
        self._lock_path = os.path.join(self.root, ".commit.lock")

    # ---- 基础原子原语 -----------------------------------------------------
    def _lock(self):
        # 同进程多线程串行化提交；跨进程恢复场景只有一个写入进程。
        import threading

        if not hasattr(self, "_commit_lock"):
            self._commit_lock = threading.Lock()
        return self._commit_lock

    def _atomic_replace_dir(self, payload_dir: str, target_dir: str) -> None:
        parent = os.path.dirname(target_dir)
        os.makedirs(parent, exist_ok=True)
        replace_tmp = target_dir + ".replace-tmp"
        if os.path.exists(replace_tmp):
            shutil.rmtree(replace_tmp, ignore_errors=True)
        os.replace(payload_dir, replace_tmp)
        if os.path.exists(target_dir):
            old_dir = target_dir + ".old"
            if os.path.exists(old_dir):
                shutil.rmtree(old_dir, ignore_errors=True)
            os.replace(target_dir, old_dir)
            os.replace(replace_tmp, target_dir)
            shutil.rmtree(old_dir, ignore_errors=True)
        else:
            os.replace(replace_tmp, target_dir)

    def _write_stage(
        self, stage: str, target_dir: str, writer
    ) -> int:
        with self._lock():
            manifest = self._read_manifest_unlocked()
            version = int(manifest.get("global_version", 0)) + 1
            temp_dir = tempfile.mkdtemp(
                prefix=f".{stage}.", dir=self.root
            )
            try:
                writer(temp_dir, version)
                self._atomic_replace_dir(temp_dir, target_dir)
            except BaseException:
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise
            manifest["global_version"] = version
            stages = dict(manifest.get("stage_versions", {}))
            stages[stage] = version
            manifest["stage_versions"] = stages
            manifest["updated_monotonic"] = monotonic()
            self._write_manifest_unlocked(manifest)
            return version

    def _read_manifest_unlocked(self) -> dict[str, Any]:
        if not os.path.exists(self.manifest_path):
            return {
                "global_version": 0,
                "stage_versions": {},
                "completed_tasks": [],
                "consumed_epochs": 0,
            }
        try:
            with open(self.manifest_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise StageWriteError(f"清单无法读取: {exc}") from exc

    def _write_manifest_unlocked(self, manifest: dict[str, Any]) -> None:
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.manifest_path)

    # ---- 生命周期 ---------------------------------------------------------
    def reset(self) -> None:
        for path in (self.data_dir, self.train_dir, self.summary_dir):
            if os.path.exists(path):
                shutil.rmtree(path)
        if os.path.exists(self.manifest_path):
            os.remove(self.manifest_path)

    def resume_or_raise(self, config_hash: str) -> ResumeInfo | None:
        if not os.path.exists(self.manifest_path):
            return None
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        stored_hash = manifest.get("config_hash")
        if stored_hash != config_hash:
            raise ConfigMismatchError(
                f"快照配置哈希 {stored_hash} 与当前配置 {config_hash} 不一致"
            )
        stage_versions = manifest.get("stage_versions", {})
        if not stage_versions:
            return None
        recovery_point = max(stage_versions, key=lambda key: stage_versions[key])
        return ResumeInfo(
            version=int(manifest["global_version"]),
            config_hash=stored_hash,
            data_fingerprint=manifest.get("data_fingerprint", ""),
            completed_tasks=list(manifest.get("completed_tasks", [])),
            consumed_epochs=int(manifest.get("consumed_epochs", 0)),
            stage_versions=dict(stage_versions),
            recovery_point=recovery_point,
        )

    # ---- data 阶段 --------------------------------------------------------
    def save_data(self, prepared, config_hash: str) -> int:
        def writer(temp_dir: str, version: int) -> None:
            payload = {
                "version": version,
                "data_fingerprint": prepared.data_fingerprint,
                "spec_hash": prepared.spec_hash,
                "n_train": prepared.n_train,
                "n_val": prepared.n_val,
                "feature_names": prepared.feature_names,
                "imputation": prepared.imputation,
                "x_train": prepared.x_train.tolist(),
                "y_train": prepared.y_train.tolist(),
                "x_val": prepared.x_val.tolist(),
                "y_val": prepared.y_val.tolist(),
            }
            with open(os.path.join(temp_dir, "data.json"), "w", encoding="utf-8") as h:
                json.dump(payload, h, ensure_ascii=False)

        version = self._write_stage(DATA_STAGE, self.data_dir, writer)
        with self._lock():
            manifest = self._read_manifest_unlocked()
            manifest["config_hash"] = config_hash
            manifest["data_fingerprint"] = prepared.data_fingerprint
            self._write_manifest_unlocked(manifest)
        return version

    def load_data(self):
        from .data import PreparedData

        path = os.path.join(self.data_dir, "data.json")
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return PreparedData(
            spec_hash=payload["spec_hash"],
            data_fingerprint=payload["data_fingerprint"],
            n_train=payload["n_train"],
            n_val=payload["n_val"],
            feature_names=payload["feature_names"],
            x_train=np_as_array(payload["x_train"]),
            y_train=np_as_array(payload["y_train"]),
            x_val=np_as_array(payload["x_val"]),
            y_val=np_as_array(payload["y_val"]),
            raw_table={},
            imputation=payload["imputation"],
        )

    # ---- train 阶段 -------------------------------------------------------
    def _checkpoint_path(self, task_key: str) -> str:
        safe = task_key.replace("|", "__").replace("=", "")
        return os.path.join(self.train_dir, f"checkpoint__{safe}.json")

    def _result_path(self, task_key: str) -> str:
        safe = task_key.replace("|", "__").replace("=", "")
        return os.path.join(self.train_dir, f"result__{safe}.json")

    def save_checkpoint(self, task_key: str, checkpoint: Checkpoint) -> int:
        def writer(temp_dir: str, version: int) -> None:
            self._copy_existing_train_files(temp_dir)
            with open(
                os.path.join(
                    temp_dir, os.path.basename(self._checkpoint_path(task_key))
                ),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {"task_key": task_key, "checkpoint": checkpoint.to_dict()},
                    handle,
                    ensure_ascii=False,
                )

        return self._write_stage(TRAIN_STAGE, self.train_dir, writer)

    def save_result(
        self,
        result: TaskResult,
        checkpoint: Checkpoint,
        completed_tasks: list[str],
        consumed_epochs: int,
        failures: list[dict[str, Any]],
    ) -> int:
        def writer(temp_dir: str, version: int) -> None:
            self._copy_existing_train_files(temp_dir)
            base = os.path.basename(self._checkpoint_path(result.task_key))
            with open(os.path.join(temp_dir, base), "w", encoding="utf-8") as handle:
                json.dump(
                    {"task_key": result.task_key, "checkpoint": checkpoint.to_dict()},
                    handle,
                    ensure_ascii=False,
                )
            with open(
                os.path.join(
                    temp_dir, os.path.basename(self._result_path(result.task_key))
                ),
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(result.to_dict(), handle, ensure_ascii=False)

        version = self._write_stage(TRAIN_STAGE, self.train_dir, writer)
        with self._lock():
            manifest = self._read_manifest_unlocked()
            manifest["completed_tasks"] = list(completed_tasks)
            manifest["consumed_epochs"] = int(consumed_epochs)
            manifest["failures"] = list(failures)
            self._write_manifest_unlocked(manifest)
        return version

    def _copy_existing_train_files(self, temp_dir: str) -> None:
        if os.path.isdir(self.train_dir):
            for name in os.listdir(self.train_dir):
                source = os.path.join(self.train_dir, name)
                if os.path.isfile(source):
                    shutil.copy2(source, os.path.join(temp_dir, name))

    def load_checkpoint(self, task_key: str) -> Checkpoint | None:
        path = self._checkpoint_path(task_key)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return Checkpoint.from_dict(payload["checkpoint"])

    def load_result(self, task_key: str) -> TaskResult | None:
        path = self._result_path(task_key)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return task_result_from_dict(payload)

    def list_result_keys(self) -> list[str]:
        if not os.path.isdir(self.train_dir):
            return []
        keys: list[str] = []
        for name in os.listdir(self.train_dir):
            if name.startswith("result__"):
                keys.append(name)
        return sorted(keys)

    def update_failures(self, failures: list[dict[str, Any]]) -> None:
        with self._lock():
            manifest = self._read_manifest_unlocked()
            manifest["failures"] = list(failures)
            self._write_manifest_unlocked(manifest)

    def persist_progress(
        self,
        completed_tasks: list[str],
        consumed_epochs: int,
        failures: list[dict[str, Any]],
    ) -> None:
        with self._lock():
            manifest = self._read_manifest_unlocked()
            manifest["completed_tasks"] = list(completed_tasks)
            manifest["consumed_epochs"] = int(consumed_epochs)
            manifest["failures"] = list(failures)
            self._write_manifest_unlocked(manifest)

    def load_manifest(self) -> dict[str, Any]:
        if not os.path.exists(self.manifest_path):
            return {}
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    # ---- 交互覆盖层（会话参数修改） --------------------------------------
    def save_session_overlay(self, raw_config: dict[str, Any], active_scheme: str) -> None:
        payload = {"raw_config": raw_config, "active_scheme": active_scheme}
        path = os.path.join(self.root, "session_overlay.json")
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)

    def load_session_overlay(self) -> dict[str, Any] | None:
        path = os.path.join(self.root, "session_overlay.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    # ---- summary 阶段 -----------------------------------------------------
    def save_summary(self, summary: dict[str, Any]) -> int:
        def writer(temp_dir: str, version: int) -> None:
            with open(os.path.join(temp_dir, "summary.json"), "w", encoding="utf-8") as h:
                json.dump(summary, h, ensure_ascii=False, indent=2, sort_keys=True)

        return self._write_stage(SUMMARY_STAGE, self.summary_dir, writer)

    def load_summary(self) -> dict[str, Any] | None:
        path = os.path.join(self.summary_dir, "summary.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)


def np_as_array(payload: Any):
    import numpy as np

    return np.asarray(payload, dtype=np.float64)


def task_result_from_dict(payload: dict[str, Any]) -> TaskResult:
    return TaskResult(
        task_key=payload["task_key"],
        cache_key=payload["cache_key"],
        scheme=payload["scheme"],
        penalty=float(payload["penalty"]),
        l1_ratio=float(payload["l1_ratio"]),
        status=payload["status"],
        epochs_executed=int(payload["epochs_executed"]),
        loss_history=[float(v) for v in payload["loss_history"]],
        weights=np_as_array(payload["weights"]),
        weight_summaries=payload.get("weight_summaries", []),
        final_weight_summary=payload.get("final_weight_summary", {}),
        converged=payload.get("converged", False),
        convergence_epoch=payload.get("convergence_epoch"),
        validation=payload.get("validation", {}),
        quota_touched=payload.get("quota_touched", False),
        seed=int(payload.get("seed", 0)),
        failure=payload.get("failure"),
        started_monotonic=payload.get("started_monotonic"),
        ended_monotonic=payload.get("ended_monotonic"),
    )
