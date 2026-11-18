"""会话快照：版本识别、整体替换恢复与变化清单。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import canonical_bytes, sha256_hex
from .errors import SnapshotError
from .models import Rule

SNAPSHOT_REQUIRED_VERSIONS = ("ruleset_version", "graph_version", "plan_version")


@dataclass
class Snapshot:
    ruleset_version: str
    graph_version: str
    plan_version: int
    rules: list[dict[str, Any]]
    topo: list[str]
    artifact_hashes: dict[str, str]
    artifact_versions: dict[str, int]
    plan_hash: str
    invalidation_sequence: list[str]
    invalidation_trace: list[dict[str, Any]]
    observation_digest: dict[str, Any]
    change_summary: dict[str, list[str]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": "qos-snapshot/1",
            "ruleset_version": self.ruleset_version,
            "graph_version": self.graph_version,
            "plan_version": self.plan_version,
            "plan_hash": self.plan_hash,
            "rules": self.rules,
            "topo": self.topo,
            "artifact_hashes": self.artifact_hashes,
            "artifact_versions": self.artifact_versions,
            "invalidation_sequence": self.invalidation_sequence,
            "invalidation_trace": self.invalidation_trace,
            "observation_digest": self.observation_digest,
            "change_summary": self.change_summary,
        }

    def content_hash(self) -> str:
        return sha256_hex(self.to_dict())


def save_snapshot(path: str | Path, snap: Snapshot) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = snap.to_dict()
    data["content_hash"] = snap.content_hash()
    path.write_bytes(canonical_bytes(data))
    return str(path)


def load_snapshot(path: str | Path, *, require_versions: bool = True) -> Snapshot:
    """读取快照；require_versions=True 时缺失版本字段抛 E_SNAP。"""
    path = Path(path)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SnapshotError(f"快照文件不存在: {path}",
                            details={"path": str(path)}) from exc
    except json.JSONDecodeError as exc:
        raise SnapshotError("快照文件 JSON 损坏",
                            details={"path": str(path), "reason": exc.msg}) from exc

    if require_versions:
        missing = [key for key in SNAPSHOT_REQUIRED_VERSIONS if key not in data]
        if missing:
            raise SnapshotError(
                "快照缺少必需版本字段，拒绝恢复",
                details={"path": str(path), "missing_fields": missing},
            )
    try:
        return Snapshot(
            ruleset_version=data["ruleset_version"],
            graph_version=data["graph_version"],
            plan_version=int(data["plan_version"]),
            rules=data["rules"],
            topo=data["topo"],
            artifact_hashes=data["artifact_hashes"],
            artifact_versions=data["artifact_versions"],
            plan_hash=data["plan_hash"],
            invalidation_sequence=data["invalidation_sequence"],
            invalidation_trace=data["invalidation_trace"],
            observation_digest=data["observation_digest"],
            change_summary=data.get("change_summary", {}),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SnapshotError(f"快照结构不完整: {exc}",
                            details={"path": str(path)}) from exc


def diff_versions(snapshot: Snapshot, current_ruleset_version: str,
                  current_graph_version: str,
                  current_rules: list[Rule]) -> dict[str, Any]:
    """稳定比较快照与当前会话版本，输出规则级变化清单。"""
    same = (snapshot.ruleset_version == current_ruleset_version
            and snapshot.graph_version == current_graph_version)
    current_by_id = {r.rule_id: r for r in current_rules}
    snap_by_id = {item["rule_id"]: item for item in snapshot.rules}

    added = sorted(set(current_by_id) - set(snap_by_id))
    deleted = sorted(set(snap_by_id) - set(current_by_id))
    modified: list[str] = []
    for rid in sorted(set(current_by_id) & set(snap_by_id)):
        if canonical_bytes(current_by_id[rid].to_dict()) != \
                canonical_bytes(snap_by_id[rid]):
            modified.append(rid)
    changes = {"added": added, "modified": modified, "deleted": deleted}
    return {
        "version_match": same,
        "snapshot_ruleset_version": snapshot.ruleset_version,
        "current_ruleset_version": current_ruleset_version,
        "snapshot_graph_version": snapshot.graph_version,
        "current_graph_version": current_graph_version,
        "changes": changes,
        "restore_mode": "noop" if same else "full-replace",
    }
