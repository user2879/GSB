"""训练状态快照：版本识别、原子持久化与整体替换恢复。

快照写入采用“临时文件 + fsync + os.replace”原子替换：强制终止后只有
完整写入并替换成功的快照可见，残留的 .tmp 文件被识别为未完成训练步，
恢复时标记为 unknown 并要求重跑。重新打开快照时若快照版本与当前会话
版本不一致，整体替换参数与训练状态，绝不做增量合并。
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import MIN_SNAPSHOT_VERSION, SESSION_VERSION, Config
from .digest import array_digest
from .errors import E_REPLAY, E_SNAPSHOT, GradCompError

SNAPSHOT_SUFFIX = ".snap"
TMP_SUFFIX = ".tmp"
PENDING_SUFFIX = ".pending"
PENDING_STEP = "unknown"


@dataclass
class Snapshot:
    snapshot_version: int
    session_version: int
    config_digest: str
    seed: int
    dimension: int
    workers: int
    scale_exponent: int
    truncation: str
    block_size: int
    compressed: bool
    parameters: np.ndarray
    parameter_version: int
    completed_steps: int
    shard_digests: list[list[str]]
    global_gradient_digest: str
    last_step_complete: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_version": self.snapshot_version,
            "session_version": self.session_version,
            "config_digest": self.config_digest,
            "seed": self.seed,
            "dimension": self.dimension,
            "workers": self.workers,
            "scale_exponent": self.scale_exponent,
            "truncation": self.truncation,
            "block_size": self.block_size,
            "compressed": self.compressed,
            "parameter_version": self.parameter_version,
            "completed_steps": self.completed_steps,
            "shard_digests": self.shard_digests,
            "global_gradient_digest": self.global_gradient_digest,
            "last_step_complete": self.last_step_complete,
            "parameters_b64": _encode_array(self.parameters),
            "parameters_digest": array_digest(self.parameters),
        }


def _encode_array(arr: np.ndarray) -> str:
    return np.ascontiguousarray(arr.astype(np.float64)).tobytes().hex()


def _decode_array(hex_str: str, dimension: int) -> np.ndarray:
    raw = bytes.fromhex(hex_str)
    arr = np.frombuffer(raw, dtype=np.float64)
    if arr.size != dimension:
        raise GradCompError(E_SNAPSHOT,
                            "快照参数维度与记录 dimension 不一致",
                            field="dimension",
                            details={"bytes": len(raw),
                                     "dimension": dimension})
    return arr.copy()


def snapshot_path(workdir: str | Path, step: int) -> Path:
    return Path(workdir) / f"snapshot_step{step:04d}{SNAPSHOT_SUFFIX}"


def pending_path(workdir: str | Path, step: int) -> Path:
    return Path(workdir) / f"snapshot_step{step:04d}{PENDING_SUFFIX}"


def write_snapshot(workdir: str | Path, cfg: Config, parameters: np.ndarray,
                   completed_steps: int, shard_digests: list[list[str]],
                   global_gradient_digest: str,
                   complete: bool = True) -> Path:
    """原子写入快照；complete=False 表示训练步中途的待决标记。"""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    snap = Snapshot(
        snapshot_version=SESSION_VERSION,
        session_version=SESSION_VERSION,
        config_digest=cfg.digest(),
        seed=cfg.seed, dimension=cfg.dimension, workers=cfg.workers,
        scale_exponent=cfg.scale_exponent, truncation=cfg.truncation,
        block_size=cfg.block_size, compressed=cfg.compressed,
        parameters=parameters,
        parameter_version=completed_steps,
        completed_steps=completed_steps,
        shard_digests=shard_digests,
        global_gradient_digest=global_gradient_digest,
        last_step_complete=complete)
    target = (pending_path(workdir, completed_steps) if not complete
              else snapshot_path(workdir, completed_steps))
    payload = json.dumps(snap.to_dict(), sort_keys=True, ensure_ascii=False)
    fd, tmp_name = tempfile.mkstemp(prefix=target.name, suffix=TMP_SUFFIX,
                                    dir=str(workdir))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, target)
        # 目录项 fsync，确保替换在崩溃后可见。
        dir_fd = os.open(str(workdir), os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except OSError as exc:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise GradCompError(E_SNAPSHOT, f"快照写入失败: {exc}")
    return target


def list_snapshots(workdir: str | Path) -> list[dict[str, Any]]:
    """按步数稳定排序列出完整快照，并报告残留的未完成文件。"""
    workdir = Path(workdir)
    items: list[dict[str, Any]] = []
    pending: list[str] = []
    if not workdir.exists():
        return {"snapshots": items, "pending_files": pending}
    for path in sorted(workdir.iterdir()):
        if path.name.endswith(TMP_SUFFIX) or path.name.endswith(PENDING_SUFFIX):
            pending.append(path.name)
        elif path.name.endswith(SNAPSHOT_SUFFIX):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pending.append(path.name)
                continue
            items.append({
                "file": path.name,
                "completed_steps": data.get("completed_steps"),
                "parameter_version": data.get("parameter_version"),
                "snapshot_version": data.get("snapshot_version"),
                "last_step_complete": data.get("last_step_complete"),
                "parameters_digest": data.get("parameters_digest"),
                "global_gradient_digest": data.get("global_gradient_digest"),
            })
    items.sort(key=lambda x: (x["completed_steps"], x["file"]))
    return {"snapshots": items, "pending_files": pending}


def load_latest(workdir: str | Path) -> tuple[Snapshot, list[str]]:
    """加载最新完整快照；返回 (快照, 未完成/待决文件名列表)。"""
    listing = list_snapshots(workdir)
    pending = list(listing["pending_files"])
    complete = [s for s in listing["snapshots"]
                if s["last_step_complete"]]
    if not complete:
        raise GradCompError(E_SNAPSHOT, "工作目录中没有完整快照可用于恢复",
                            details={"pending": pending})
    latest = complete[-1]
    path = Path(workdir) / latest["file"]
    snap = load_file(path)
    if not snap.last_step_complete:
        pending.append(path.name)
    return snap, pending


def load_file(path: str | Path) -> Snapshot:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GradCompError(E_SNAPSHOT, f"快照无法读取: {exc}")
    required = {"snapshot_version", "session_version", "config_digest",
                "dimension", "parameters_b64", "completed_steps",
                "parameter_version"}
    missing = sorted(required - set(data))
    if missing:
        raise GradCompError(E_SNAPSHOT, "快照缺少必需字段",
                            details={"missing": missing})
    params = _decode_array(data["parameters_b64"], data["dimension"])
    return Snapshot(
        snapshot_version=int(data["snapshot_version"]),
        session_version=int(data["session_version"]),
        config_digest=data["config_digest"], seed=int(data["seed"]),
        dimension=int(data["dimension"]), workers=int(data["workers"]),
        scale_exponent=int(data["scale_exponent"]),
        truncation=str(data["truncation"]),
        block_size=int(data["block_size"]),
        compressed=bool(data["compressed"]), parameters=params,
        parameter_version=int(data["parameter_version"]),
        completed_steps=int(data["completed_steps"]),
        shard_digests=data.get("shard_digests", []),
        global_gradient_digest=data.get("global_gradient_digest", ""),
        last_step_complete=bool(data.get("last_step_complete", True)))


def restore_state(cfg: Config, snap: Snapshot) -> dict[str, Any]:
    """版本识别 + 整体替换恢复；返回恢复决策说明。"""
    replaced_fields = ["parameters", "parameter_version", "completed_steps",
                       "shard_digests", "global_gradient_digest"]
    if snap.snapshot_version < MIN_SNAPSHOT_VERSION:
        raise GradCompError(E_SNAPSHOT, "快照版本低于最低可识别版本",
                            field="snapshot_version",
                            details={"snapshot_version": snap.snapshot_version,
                                     "min_supported": MIN_SNAPSHOT_VERSION})
    decision = {
        "snapshot_version": snap.snapshot_version,
        "session_version": SESSION_VERSION,
        "version_match": snap.session_version == SESSION_VERSION,
        "mode": "in_place_resume"
        if snap.session_version == SESSION_VERSION else "full_replace",
        "replaced_fields": replaced_fields,
        "incremental_merge": False,
    }
    # 无论版本是否一致，恢复都整体替换，绝不增量合并。
    return decision


def assert_replay_compatible(cfg: Config, snap: Snapshot) -> None:
    """恢复后续训前，校验影响可复现性的配置依据；不一致报 E_REPLAY。"""
    basis = {
        "seed": (cfg.seed, snap.seed),
        "dimension": (cfg.dimension, snap.dimension),
        "workers": (cfg.workers, snap.workers),
        "scale_exponent": (cfg.scale_exponent, snap.scale_exponent),
        "truncation": (cfg.truncation, snap.truncation),
        "block_size": (cfg.block_size, snap.block_size),
    }
    mismatches = {k: {"current": cur, "snapshot": old}
                  for k, (cur, old) in basis.items() if cur != old}
    if mismatches:
        reasons = []
        if "scale_exponent" in mismatches:
            reasons.append("缩放因子与快照不一致，量化结果不可复现")
        if "workers" in mismatches or "block_size" in mismatches \
                or "dimension" in mismatches:
            reasons.append("分片划分与快照不一致，环形归约不可复现")
        if not reasons:
            reasons.append("恢复配置与快照依据不一致")
        raise GradCompError(E_REPLAY, "；".join(reasons),
                            field=",".join(sorted(mismatches)),
                            details={"mismatches": mismatches})
    if cfg.digest() != snap.config_digest:
        raise GradCompError(
            E_REPLAY, "当前配置摘要与快照记录不一致，拒绝重放",
            field="config_digest",
            details={"current": cfg.digest(),
                     "snapshot": snap.config_digest})
