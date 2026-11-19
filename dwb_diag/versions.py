"""页版本映射: 每页最新写入序号与权威摘要, 以及崩溃点标记。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import PHASES
from .util import append_fsynced, canonical_json, write_bytes_atomic


@dataclass
class PageVersion:
    page_no: int
    seq: int
    digest: str  # 权威页帧摘要(hex), 来自最新可用缓冲镜像

    def to_dict(self) -> dict[str, Any]:
        return {"page_no": self.page_no, "seq": self.seq, "digest": self.digest}


class VersionMap:
    """page_no -> PageVersion; 持久化为规范化 JSON。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.pages: dict[int, PageVersion] = {}

    @classmethod
    def load(cls, path: Path) -> "VersionMap":
        vm = cls(path)
        if path.exists():
            obj = json.loads(path.read_text(encoding="utf-8"))
            for item in obj.get("pages", []):
                vm.pages[int(item["page_no"])] = PageVersion(
                    int(item["page_no"]), int(item["seq"]), str(item["digest"])
                )
        return vm

    def save(self) -> None:
        payload = {
            "v": 1,
            "pages": [self.pages[k].to_dict() for k in sorted(self.pages)],
        }
        write_bytes_atomic(self.path, canonical_json(payload).encode("utf-8"))

    def upsert(self, page_no: int, seq: int, digest: str) -> None:
        self.pages[int(page_no)] = PageVersion(int(page_no), int(seq), digest)

    def get(self, page_no: int) -> PageVersion | None:
        return self.pages.get(int(page_no))

    def replace_all(self, pages: dict[int, PageVersion]) -> None:
        """快照整体替换入口(不做增量合并)。"""
        self.pages = {int(k): v for k, v in pages.items()}

    def as_dict(self) -> dict[str, dict[str, Any]]:
        return {str(k): self.pages[k].to_dict() for k in sorted(self.pages)}

    def digest(self) -> str:
        from .util import digest_of_json

        return digest_of_json([self.pages[k].to_dict() for k in sorted(self.pages)])


@dataclass
class CrashMarker:
    seq: int
    phase: str
    ts_ms: int
    commit_id: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "phase": self.phase,
            "ts_ms": self.ts_ms,
            "commit_id": self.commit_id,
        }


class MarkerLog:
    """三阶段崩溃点标记, JSONL 追加; 阶段名受 PHASES 约束。"""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.markers: list[CrashMarker] = []

    @classmethod
    def load(cls, path: Path) -> "MarkerLog":
        log = cls(path)
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                obj = json.loads(line)
                log.markers.append(
                    CrashMarker(
                        seq=int(obj["seq"]),
                        phase=str(obj["phase"]),
                        ts_ms=int(obj["ts_ms"]),
                        commit_id=int(obj["commit_id"]),
                    )
                )
        return log

    def append(self, marker: CrashMarker) -> None:
        if marker.phase not in PHASES:
            raise ValueError(f"unknown phase {marker.phase}")
        self.markers.append(marker)
        append_fsynced(self.path, canonical_json(marker.to_dict()))

    def phases_for_commit(self, commit_id: int) -> list[str]:
        return [m.phase for m in self.markers if m.commit_id == commit_id]

    def latest_phase(self) -> CrashMarker | None:
        return self.markers[-1] if self.markers else None
