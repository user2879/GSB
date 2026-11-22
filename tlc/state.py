"""阶段化快照：临时目录整体替换、恢复判定。

- 各阶段产物先写入与状态目录同级的 .staging 临时目录；
- 提交前比对清单中的基座层版本指纹与配置哈希，然后整体替换
  （rename 交换），不做逐文件覆盖；
- 强制终止后：若临时目录的 run_key 与当前输入一致则续跑，
  不一致（版本号/配置哈希变化）则输出 E_SNAPSHOT_STALE。
"""

import json
import os
import shutil

from .errors import BizError, E_SNAPSHOT_STALE
from .hashing import canonical, sha256_hex
from .loaders import encoder_from_dict
from .model import Snapshot

STAGES = ["A_forward", "B_decide", "C_train", "D_shares", "E_eval"]


def _read_json(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=1))


def staging_dir(state_dir):
    return state_dir.rstrip("/") + ".staging"


def backup_dir(state_dir):
    return state_dir.rstrip("/") + ".bak"


def load_snapshot(state_dir) -> Snapshot:
    """加载已提交快照；处理提交中断（.bak 残留）的恢复。"""
    bak = backup_dir(state_dir)
    if not os.path.isdir(state_dir) and os.path.isdir(bak):
        os.rename(bak, state_dir)
    manifest_path = os.path.join(state_dir, "manifest.json")
    snap_path = os.path.join(state_dir, "snapshot.json")
    if not (os.path.isfile(manifest_path) and os.path.isfile(snap_path)):
        return Snapshot()
    manifest = _read_json(manifest_path)
    data = _read_json(snap_path)
    return Snapshot(
        manifest=manifest,
        encoder=encoder_from_dict(manifest["base"]),
        adapters=data.get("adapters", {}),
        last_shares=data.get("last_shares", {}),
        pending=data.get("pending", []),
    )


def compute_run_key(manifest_fields) -> str:
    return sha256_hex(manifest_fields)


class Staging:
    """一次运行的暂存区：阶段产物写入、断点续跑、整体提交。"""

    def __init__(self, state_dir, run_key, manifest):
        self.state_dir = state_dir
        self.dir = staging_dir(state_dir)
        self.run_key = run_key
        self.manifest = manifest
        if os.path.isdir(self.dir):
            old = _read_json(os.path.join(self.dir, "staging_manifest.json"))
            if old.get("run_key") != run_key:
                raise BizError(
                    E_SNAPSHOT_STALE,
                    "暂存区基座层版本指纹/配置哈希与当前输入不一致，"
                    f"期望 {run_key}，实际 {old.get('run_key')}",
                )
            self.resumed = True
        else:
            os.makedirs(self.dir)
            _write_json(
                os.path.join(self.dir, "staging_manifest.json"),
                {"run_key": run_key, "manifest": manifest},
            )
            self.resumed = False

    def stage_done(self, stage):
        path = os.path.join(self.dir, f"stage_{stage}.json")
        return os.path.isfile(path)

    def load_stage(self, stage):
        return _read_json(os.path.join(self.dir, f"stage_{stage}.json"))["payload"]

    def save_stage(self, stage, payload):
        _write_json(
            os.path.join(self.dir, f"stage_{stage}.json"),
            {"run_key": self.run_key, "payload": payload},
        )

    def commit(self, snapshot_data):
        """整体替换提交：先比对清单指纹，再 rename 交换。"""
        manifest = dict(self.manifest)
        manifest["run_key"] = self.run_key
        # 提交前校验：清单中的基座版本指纹与配置哈希必须与本次运行一致
        if sha256_hex(manifest["base"]) != manifest["base_fingerprint"]:
            raise BizError(E_SNAPSHOT_STALE, "提交前基座层版本指纹校验失败")
        if sha256_hex(manifest["config"]) != manifest["config_hash"]:
            raise BizError(E_SNAPSHOT_STALE, "提交前配置哈希校验失败")
        _write_json(os.path.join(self.dir, "manifest.json"), manifest)
        _write_json(os.path.join(self.dir, "snapshot.json"), snapshot_data)
        bak = backup_dir(self.state_dir)
        if os.path.isdir(bak):
            shutil.rmtree(bak)
        if os.path.isdir(self.state_dir):
            os.rename(self.state_dir, bak)
        os.rename(self.dir, self.state_dir)
        if os.path.isdir(bak):
            shutil.rmtree(bak)


def discard_staging(state_dir):
    path = staging_dir(state_dir)
    if os.path.isdir(path):
        shutil.rmtree(path)


def dumps_canonical(obj) -> str:
    return canonical(obj)
