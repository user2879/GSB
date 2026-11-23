"""阶段快照：临时目录写入 + 整体替换提交 + 恢复判定。"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .errors import ErrorCollector, ErrorCode
from .fixedpoint import ARTIFACT_VERSION
from .hashing import fingerprint

SNAPSHOT_DIRNAME = "snapshots"
SNAPSHOT_MANIFEST = "manifest.json"


@dataclass
class SnapshotHandle:
    path: Path
    manifest: dict


def snapshot_root(output_dir: str | Path) -> Path:
    return Path(output_dir) / SNAPSHOT_DIRNAME


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, sort_keys=True, indent=2,
                              ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def commit_snapshot(output_dir: str | Path, stage: str, cursor: int,
                    config_hash: str, injection_hash: str, frac_bits: int,
                    quant_gear: int, artifacts: dict, ledger,
                    extra: dict | None = None) -> SnapshotHandle:
    root = snapshot_root(output_dir)
    staging = root / f".stage-{stage}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    manifest = {
        "artifact_version": ARTIFACT_VERSION,
        "stage": stage,
        "config_hash": config_hash,
        "injection_hash": injection_hash,
        "cursor_round": cursor,
        "frac_bits": frac_bits,
        "quant_gear": quant_gear,
        "artifacts_hash": fingerprint(
            {k: _jsonable(v) for k, v in artifacts.items()}),
    }
    if extra:
        manifest.update(extra)
    _atomic_write_json(staging / SNAPSHOT_MANIFEST, manifest)
    _atomic_write_json(staging / "artifacts.json",
                       {k: _jsonable(v) for k, v in artifacts.items()})
    # 快照账本独立落盘，不清空调用方内存账本的本次运行决策。
    import copy
    snapshot_ledger = copy.deepcopy(ledger)
    snapshot_ledger.save(staging / "ledger.json")

    final = root / stage
    if final.exists():
        shutil.rmtree(final)
    os.replace(staging, final)  # 同目录整体替换，原子提交
    return SnapshotHandle(final, manifest)


def load_snapshot(output_dir: str | Path, stage: str, expected_config_hash: str,
                  expected_injection_hash: str, errors: ErrorCollector
                  ) -> SnapshotHandle | None:
    path = snapshot_root(output_dir) / stage
    manifest_path = path / SNAPSHOT_MANIFEST
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.add(ErrorCode.E_SNAPSHOT_STALE, "snapshot",
                   f"snapshot manifest unreadable: {exc}", file=str(manifest_path))
        return None

    stale_reasons = []
    if manifest.get("artifact_version") != ARTIFACT_VERSION:
        stale_reasons.append(
            f"version {manifest.get('artifact_version')} != {ARTIFACT_VERSION}")
    if manifest.get("config_hash") != expected_config_hash:
        stale_reasons.append("config_hash mismatch")
    if manifest.get("injection_hash") != expected_injection_hash:
        stale_reasons.append("injection_hash mismatch")
    if stale_reasons:
        errors.add(ErrorCode.E_SNAPSHOT_STALE, "snapshot",
                   "; ".join(stale_reasons), file=str(manifest_path))
        return None
    return SnapshotHandle(path, manifest)


def _jsonable(value):
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value
