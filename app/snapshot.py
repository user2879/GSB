"""版本化快照：临时文件写盘 + 整体替换提交 + 启动校验。

清单文件记录：全局版本号、配置哈希、名称表哈希、IR 内容指纹、旁路元数据哈希。
启动时以清单版本号与当前内容指纹判定是否命中；版本落后或内容指纹不符时
输出 E_SNAPSHOT_STALE 并整体替换重建（而非逐文件覆盖）。
"""
from __future__ import annotations

import json
from pathlib import Path

from .errors import E_SNAPSHOT_STALE, Failure
from .ir import fingerprint

MANIFEST = "manifest.json"
TRACKED_FILES = ["ir.json", "sidecar.json", "ledger.json"]


class SnapshotManager:
    def __init__(self, snapshot_dir: Path):
        self.dir = Path(snapshot_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _write_atomic(self, name: str, content: str) -> None:
        tmp = self.dir / f"{name}.tmp"
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(self.dir / name)

    def commit(self, global_version: int, config_hash: str, name_map_hash: str,
               files: dict[str, str]) -> dict:
        for name in TRACKED_FILES:
            self._write_atomic(name, files[name])
        manifest = {
            "global_version": global_version,
            "config_hash": config_hash,
            "name_map_hash": name_map_hash,
            "ir_fingerprint": fingerprint(json.loads(files["ir.json"])),
            "sidecar_hash": fingerprint(json.loads(files["sidecar.json"])),
            "ledger_hash": fingerprint(json.loads(files["ledger.json"])),
        }
        self._write_atomic(MANIFEST, json.dumps(manifest, indent=1, sort_keys=True) + "\n")
        return manifest

    def read_manifest(self) -> dict | None:
        path = self.dir / MANIFEST
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

    def verify(self, expected_min_version: int = 0) -> tuple[bool, list[Failure]]:
        """校验快照内部一致性；任何不符都判定为整体过期。"""
        failures: list[Failure] = []
        manifest = self.read_manifest()
        if manifest is None:
            failures.append(Failure(None, "snapshot", E_SNAPSHOT_STALE,
                                    "ManifestMissing", "清单缺失或损坏"))
            return False, failures
        if manifest.get("global_version", 0) < expected_min_version:
            failures.append(Failure(None, "snapshot", E_SNAPSHOT_STALE,
                                    "VersionLag",
                                    f"清单版本 {manifest.get('global_version')} 落后 "
                                    f"{expected_min_version}"))
            return False, failures
        checks = {
            "ir.json": "ir_fingerprint",
            "sidecar.json": "sidecar_hash",
            "ledger.json": "ledger_hash",
        }
        for name, key in checks.items():
            path = self.dir / name
            if not path.exists():
                failures.append(Failure(None, "snapshot", E_SNAPSHOT_STALE,
                                        "FileMissing", f"快照文件 {name} 缺失"))
                return False, failures
            try:
                actual = fingerprint(json.loads(path.read_text(encoding="utf-8")))
            except (json.JSONDecodeError, OSError) as exc:
                failures.append(Failure(None, "snapshot", E_SNAPSHOT_STALE,
                                        type(exc).__name__, f"快照文件 {name} 损坏"))
                return False, failures
            if actual != manifest.get(key):
                failures.append(Failure(None, "snapshot", E_SNAPSHOT_STALE,
                                        "FingerprintMismatch",
                                        f"{name} 内容指纹与清单不符"))
                return False, failures
        return True, failures

    def load_files(self) -> dict[str, str]:
        return {name: (self.dir / name).read_text(encoding="utf-8")
                for name in TRACKED_FILES}
