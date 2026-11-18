"""阶段化快照整体替换、清单管理与中断恢复。

每个阶段产物先写入 stages/<stage>.tmp/，完成后 os.replace 整体替换为
stages/<stage>/；强制中断只可能留下 .tmp 目录，重启时视为未提交并清理，
恢复仅依据完整替换后的目录与 manifest.json 进度记录。
"""

import json
import os
import signal

MANIFEST_NAME = "manifest.json"
STAGE_NAMES = ("validate", "extract", "align", "convert", "retrieve")


class ShutdownRequested(Exception):
    """收到可处理退出信号，用于在样本边界优雅退出。"""


class SignalGate:
    def __init__(self):
        self.requested = False
        self.signal_name = None
        self._prev = {}

    def _handler(self, signum, _frame):
        self.requested = True
        self.signal_name = signal.Signals(signum).name

    def install(self):
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._prev[sig] = signal.signal(sig, self._handler)
            except (ValueError, OSError):
                pass

    def restore(self):
        for sig, handler in self._prev.items():
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                pass

    def checkpoint(self):
        if self.requested:
            raise ShutdownRequested(self.signal_name or "SIGTERM")


class SnapshotStore:
    def __init__(self, root):
        self.root = root
        self.stages_dir = os.path.join(root, "stages")
        self.manifest_path = os.path.join(root, MANIFEST_NAME)
        os.makedirs(self.stages_dir, exist_ok=True)

    def stage_tmp(self, stage):
        path = os.path.join(self.stages_dir, stage + ".tmp")
        os.makedirs(path, exist_ok=True)
        return path

    def stage_dir(self, stage):
        return os.path.join(self.stages_dir, stage)

    def commit_stage(self, stage):
        """临时目录整体替换正式阶段目录（同文件系统 rename，原子）。"""
        tmp = self.stage_tmp(stage)
        final = self.stage_dir(stage)
        old = self.stages_dir + "/" + stage + ".old"
        if os.path.exists(old):
            _rmtree(old)
        if os.path.exists(final):
            os.rename(final, old)
        os.rename(tmp, final)
        if os.path.exists(old):
            _rmtree(old)
        return final

    def cleanup_tmp(self):
        removed = []
        for name in sorted(os.listdir(self.stages_dir)):
            if name.endswith(".tmp"):
                path = os.path.join(self.stages_dir, name)
                _rmtree(path)
                removed.append(name)
        return removed

    def load_manifest(self):
        if not os.path.isfile(self.manifest_path):
            return None
        with open(self.manifest_path, "rb") as fh:
            return json.loads(fh.read().decode("utf-8"))

    def write_manifest(self, manifest):
        tmp = self.manifest_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, sort_keys=True, ensure_ascii=False,
                      indent=2)
        os.replace(tmp, self.manifest_path)

    def recovery_point(self):
        """仅依据完整替换产物与清单返回恢复点信息。"""
        manifest = self.load_manifest()
        committed = []
        for stage in STAGE_NAMES:
            if os.path.isdir(self.stage_dir(stage)):
                committed.append(stage)
        tmp_stages = sorted(
            name[:-4] for name in os.listdir(self.stages_dir)
            if name.endswith(".tmp"))
        return {
            "manifest_present": manifest is not None,
            "global_version": (manifest or {}).get("global_version", 0),
            "committed_stages": committed,
            "uncommitted_tmp_stages": tmp_stages,
            "completed_samples": (manifest or {}).get("completed_samples",
                                                      []),
        }


def _rmtree(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


def new_manifest(config_hash, spec_hash, run_id):
    return {
        "global_version": 0,
        "run_id": run_id,
        "config_hash": config_hash,
        "spec_hash": spec_hash,
        "format_version": None,
        "completed_samples": [],
        "valid_samples": [],
        "stage_versions": {stage: None for stage in STAGE_NAMES},
        "keys": {},
    }
