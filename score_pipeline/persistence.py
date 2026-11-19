"""稳定状态持久化：原子提交、版本戳/哈希校验、孤儿回收、恢复。

目录布局：
  state/
    manifest.json        当前稳定清单（版本戳、分片哈希表、时间轴哈希）
    operations.jsonl     已提交操作序列（逐行原子追加，仅在清单提交后写入）
    blobs/<sha256>.json  内容寻址分片/快照（写后不可变）
    closures/v<N>.json   每次增量提交的失效闭包记录
    orphans/<name>.json  恢复时发现的未提交/无引用文件
    CLOSED               受控退出标记
    exit_report.json     受控退出报告（强制终止路径不会产生）
    tmp/                 暂存目录（未完成写入全部视为未提交）

提交协议（任一环节中断都不会推进 manifest 版本戳）：
  1. 写分片 blob（crash_after_blob 在此之后触发 -> blob 成孤儿）
  2. 暂存新 manifest 到 tmp/ 并 fsync（crash_after_journal 在此之后触发）
  3. os.replace 原子替换 manifest.json
  4. 追加 operations 与闭包记录
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from .errors import (
    E_SNAPSHOT_STALE,
    E_TIMELINE_HASH,
    E_VERSION_REGRESS,
    PipelineError,
)
from .injection import (
    FAULT_COMMIT_CRASH_AFTER_BLOB,
    FAULT_COMMIT_CRASH_AFTER_JOURNAL,
    HardExit,
)
from .model import canonical_bytes, content_hash, read_json, write_json


class Store:
    def __init__(self, root: str, injection=None):
        self.root = Path(root)
        self.blobs = self.root / "blobs"
        self.closures = self.root / "closures"
        self.orphans = self.root / "orphans"
        self.tmp = self.root / "tmp"
        for directory in (self.root, self.blobs, self.closures, self.orphans, self.tmp):
            directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.operations_path = self.root / "operations.jsonl"
        self.closed_marker = self.root / "CLOSED"
        self.exit_report_path = self.root / "exit_report.json"
        self.injection = injection

    # ---------- 清单与恢复 ----------
    def has_manifest(self) -> bool:
        return self.manifest_path.exists()

    def read_manifest(self) -> dict:
        return read_json(self.manifest_path)

    def write_blob(self, payload: dict) -> str:
        digest = content_hash(payload)
        target = self.blobs / f"{digest}.json"
        if not target.exists():
            write_json(target, payload)
        return digest

    def read_blob(self, digest: str) -> dict:
        path = self.blobs / f"{digest}.json"
        if not path.exists():
            raise PipelineError(
                E_TIMELINE_HASH, "清单引用的分片 blob 缺失", {"blob": digest}
            )
        data = read_json(path)
        if content_hash(data) != digest:
            raise PipelineError(
                E_TIMELINE_HASH, "blob 内容哈希与文件名不一致", {"blob": digest}
            )
        return data

    def recover(self) -> dict:
        """重启恢复：校验稳定清单、回收孤儿、返回恢复报告。"""
        report = {
            "recovered": False,
            "stable_version": None,
            "orphans": [],
            "missing_exit_report": False,
            "closed": self.closed_marker.exists(),
        }
        if not self.has_manifest():
            self._quarantine_tmp(report)
            self._quarantine_dangling_blobs(set(), report)
            return report

        manifest = self.read_manifest()
        referenced = set(manifest["shard_blobs"].values())
        self._verify_manifest_blobs(manifest)
        if manifest.get("timeline_hash") != self._compute_timeline_hash(manifest):
            raise PipelineError(
                E_TIMELINE_HASH,
                "稳定清单的时间轴哈希重算不一致",
                {"version": manifest["version"]},
            )
        self._quarantine_tmp(report)
        self._quarantine_dangling_blobs(referenced, report)
        report.update(
            recovered=True,
            stable_version=manifest["version"],
            missing_exit_report=not self.exit_report_path.exists(),
        )
        return report

    def _verify_manifest_blobs(self, manifest: dict) -> None:
        for shard_id, digest in sorted(manifest["shard_blobs"].items()):
            data = self.read_blob(digest)
            inner = content_hash({k: v for k, v in data.items() if k != "content_hash"})
            if data["content_hash"] != inner:
                raise PipelineError(
                    E_TIMELINE_HASH,
                    "分片内嵌内容哈希不自洽",
                    {"shard_id": shard_id, "blob": digest},
                )

    def _compute_timeline_hash(self, manifest: dict) -> str:
        hashes = manifest.get("shard_inner_hashes") or manifest["shard_blobs"]
        ordered = [hashes[key] for key in sorted(hashes)]
        return content_hash(
            {
                "schema": "timeline-hash/1",
                "shard_hashes": ordered,
                "pedals": manifest["pedals"],
            }
        )

    def _quarantine_tmp(self, report: dict) -> None:
        for path in sorted(self.tmp.glob("*")):
            if path.is_file():
                self._move_orphan(path, report, reason="tmp_uncommitted")

    def _quarantine_dangling_blobs(self, referenced: set, report: dict) -> None:
        for path in sorted(self.blobs.glob("*.json")):
            if path.stem not in referenced:
                self._move_orphan(path, report, reason="dangling_blob")

    def _move_orphan(self, path: Path, report: dict, reason: str) -> None:
        target = self.orphans / f"{reason}__{path.name}"
        if target.exists():
            target = self.orphans / f"{reason}__{path.stem}_{len(list(self.orphans.glob('*')))}.json"
        os.replace(path, target)
        report["orphans"].append(
            {"name": target.name, "origin": str(path.name), "reason": reason}
        )

    # ---------- 提交 ----------
    def commit(
        self,
        state,
        closure_record: dict | None,
        op_dict: dict | None,
    ) -> dict:
        """原子提交一个新稳定状态。返回提交记录。"""
        if self.has_manifest():
            current = self.read_manifest()
            if state.version != current["version"] + 1:
                raise PipelineError(
                    E_VERSION_REGRESS,
                    "版本戳必须严格 +1 递增",
                    {"current": current["version"], "incoming": state.version},
                )

        shard_blobs = {}
        shard_inner_hashes = {}
        for shard_id in sorted(state.shards):
            payload = state.shards[shard_id]
            shard_blobs[shard_id] = self.write_blob(payload)
            shard_inner_hashes[shard_id] = payload["content_hash"]
        if (
            self.injection is not None
            and self.injection.crash_at_version == state.version
            and self.injection.crash_point == FAULT_COMMIT_CRASH_AFTER_BLOB
        ):
            raise HardExit(FAULT_COMMIT_CRASH_AFTER_BLOB)

        new_manifest = {
            "schema": "manifest/1",
            "version": state.version,
            "parent_version": state.parent_version,
            "tempo": state.tempo,
            "dynamics": state.dynamics,
            "pedals": state.rendered_pedals,
            "shard_blobs": shard_blobs,
            "shard_inner_hashes": shard_inner_hashes,
            "timeline_hash": state.timeline_hash,
        }
        staged = self.tmp / f"manifest-v{state.version}.json"
        write_json(staged, new_manifest)
        if self.injection is not None and (
            self.injection.crash_at_version == state.version
            and self.injection.crash_point == FAULT_COMMIT_CRASH_AFTER_JOURNAL
        ):
            raise HardExit(FAULT_COMMIT_CRASH_AFTER_JOURNAL)

        os.replace(staged, self.manifest_path)  # 唯一的版本戳推进点

        if op_dict is not None:
            with open(self.operations_path, "ab") as handle:
                handle.write(canonical_bytes(op_dict) + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        if closure_record is not None:
            write_json(self.closures / f"v{state.version}.json", closure_record)
        return {"version": state.version, "timeline_hash": state.timeline_hash}

    def commit_initial(self, state) -> dict:
        return self.commit(state, closure_record=None, op_dict=None)

    def load_state_manifest(self) -> dict:
        return self.read_manifest()

    def load_operations(self) -> list[dict]:
        if not self.operations_path.exists():
            return []
        with open(self.operations_path, "rb") as handle:
            return [json.loads(line) for line in handle if line.strip()]

    def replace_snapshot(self, candidate_manifest: dict) -> None:
        """快照整体替换：版本戳落后或哈希不符一律拒绝（禁止盲目复用旧分片）。"""
        if not self.has_manifest():
            raise PipelineError(E_SNAPSHOT_STALE, "当前无稳定清单，无法比较快照新旧")
        current = self.read_manifest()
        if candidate_manifest["version"] < current["version"]:
            raise PipelineError(
                E_SNAPSHOT_STALE,
                "候选快照版本戳落后于当前稳定状态",
                {"current": current["version"], "candidate": candidate_manifest["version"]},
            )
        referenced = set(candidate_manifest["shard_blobs"].values())
        for digest in referenced:
            if not (self.blobs / f"{digest}.json").exists():
                raise PipelineError(
                    E_SNAPSHOT_STALE, "候选快照分片内容缺失", {"blob": digest}
                )
        candidate_hashes = (
            candidate_manifest.get("shard_inner_hashes")
            or candidate_manifest["shard_blobs"]
        )
        recomputed = content_hash(
            {
                "schema": "timeline-hash/1",
                "shard_hashes": [
                    candidate_hashes[key] for key in sorted(candidate_hashes)
                ],
                "pedals": candidate_manifest["pedals"],
            }
        )
        if recomputed != candidate_manifest["timeline_hash"]:
            raise PipelineError(
                E_SNAPSHOT_STALE,
                "候选快照时间轴哈希不自洽，拒绝替换",
                {"expected": candidate_manifest["timeline_hash"], "actual": recomputed},
            )
        write_json(self.tmp / "snapshot-candidate.json", candidate_manifest)
        os.replace(self.tmp / "snapshot-candidate.json", self.manifest_path)

    # ---------- 受控退出 ----------
    def write_controlled_close(self, report: dict) -> None:
        write_json(self.exit_report_path, report)
        with open(self.closed_marker, "wb") as handle:
            handle.write(b"closed\n")
            handle.flush()
            os.fsync(handle.fileno())

    def is_closed(self) -> bool:
        return self.closed_marker.exists()
