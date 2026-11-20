"""会话快照的持久化、版本识别与整体替换恢复。

快照为单个 JSON 文件，持久化：

* 段集合摘要（每段格式版本与块摘要列表）；
* 编码选择版本、预算参数版本、格式版本；
* 任务轨迹、结果摘要与扫描统计；
* 对上述全部内容计算的 ``content_digest``。

重新打开时，任一段摘要或三类版本与当前工作态不一致，都不会做增量合并，
而是整体替换：旧快照保留为 ``*.backup.json`` 只读备份，再写入新快照。
"""
from __future__ import annotations

import hashlib
import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

from .errors import SnapshotError
from .segment import Segment
from .versioning import ParameterVersion


@dataclass
class SnapshotBundle:
    segment_overview: list[dict]
    version: ParameterVersion
    trajectories: list[dict]
    result_digests: list[dict]
    statistics: dict
    content_digest: str = ""

    def to_dict(self) -> dict:
        return {
            "segment_overview": self.segment_overview,
            "version": self.version.to_dict(),
            "trajectories": self.trajectories,
            "result_digests": self.result_digests,
            "statistics": self.statistics,
            "content_digest": self.content_digest,
        }


def _canonical(bundle: SnapshotBundle) -> bytes:
    payload = {
        "segment_overview": bundle.segment_overview,
        "version": bundle.version.to_dict(),
        "trajectories": bundle.trajectories,
        "result_digests": bundle.result_digests,
        "statistics": bundle.statistics,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def build_overview(segments: dict[str, Segment]) -> list[dict]:
    return [segments[segment_id].digest_overview()
            for segment_id in sorted(segments)]


def save_snapshot(path: str | Path, segments: dict[str, Segment],
                  version: ParameterVersion, trajectories: list[dict],
                  result_digests: list[dict], statistics: dict,
                  *, replace: bool = True) -> SnapshotBundle:
    path = Path(path)
    bundle = SnapshotBundle(
        segment_overview=build_overview(segments), version=version,
        trajectories=sorted(trajectories,
                            key=lambda item: (item["segment_id"],
                                              item["row_group"],
                                              item["column"],
                                              item["block_index"])),
        result_digests=sorted(result_digests,
                              key=lambda item: (item["segment_id"],
                                                item["row_group"],
                                                item["column"])),
        statistics=statistics)
    bundle.content_digest = hashlib.sha256(
        _canonical(bundle)).hexdigest()

    if path.exists() and replace:
        backup = path.with_suffix(".backup.json")
        if backup.exists():
            os.chmod(backup, stat.S_IWRITE | stat.S_IREAD)
        backup.write_bytes(path.read_bytes())
        os.chmod(backup, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    path.write_text(json.dumps(bundle.to_dict(), ensure_ascii=False,
                               indent=2, sort_keys=True),
                    encoding="utf-8")
    return bundle


def load_snapshot(path: str | Path) -> SnapshotBundle:
    path = Path(path)
    if not path.exists():
        raise SnapshotError(f"快照文件不存在：{path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    version = ParameterVersion(**raw["version"])
    bundle = SnapshotBundle(
        segment_overview=raw["segment_overview"], version=version,
        trajectories=raw["trajectories"],
        result_digests=raw["result_digests"],
        statistics=raw["statistics"],
        content_digest=raw.get("content_digest", ""))
    actual = hashlib.sha256(_canonical(bundle)).hexdigest()
    if actual != bundle.content_digest:
        raise SnapshotError(
            "快照内容摘要不匹配，拒绝增量合并",
            {"expected": bundle.content_digest, "actual": actual,
             "path": str(path)})
    return bundle


def restore_snapshot(path: str | Path,
                     current_segments: dict[str, Segment],
                     current_version: ParameterVersion) -> dict:
    """重新打开会话：版本/摘要不一致时整体替换并保留只读备份。"""
    bundle = load_snapshot(path)
    current_overview = build_overview(current_segments)
    mismatch = {
        "segment_overview_changed":
            bundle.segment_overview != current_overview,
        "version_changed": bundle.version.key() != current_version.key(),
    }
    if any(mismatch.values()):
        backup = Path(path).with_suffix(".backup.json")
        if Path(path).exists():
            if backup.exists():
                os.chmod(backup, stat.S_IWRITE | stat.S_IREAD)
            backup.write_bytes(Path(path).read_bytes())
            os.chmod(backup, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
        return {
            "action": "REPLACED",
            "reason": mismatch,
            "backup": str(backup),
            "old_trajectory_count": len(bundle.trajectories),
            "old_version": bundle.version.to_dict(),
        }
    return {
        "action": "RESTORED",
        "reason": mismatch,
        "trajectory_count": len(bundle.trajectories),
        "version": bundle.version.to_dict(),
    }
