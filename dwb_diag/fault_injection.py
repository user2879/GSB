"""四类异常注入、主文件损坏校验与快照重放比对(要求 8)。

注入类型 -> 业务码:
  main_corrupt   -> E_CORRUPT (校验主文件) / 恢复时修复
  buffer_truncate-> E_PAGE    (缓冲扫描结构性拒绝)
  depend_cycle   -> E_DEPEND  (依赖成环)
  snapshot_stale -> E_SNAPSHOT(快照缓冲摘要不符)
重放差异 -> E_REPLAY。
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .config import DataLayout
from .errors import (
    MainCorruptError,
    PageFormatError,
    ReplayError,
)
from .paging import Page
from .util import canonical_json, sha256_hex


# ---------- 主文件损坏校验 (E_CORRUPT) ----------
def find_first_corrupt_main_page(layout: DataLayout, page_size: int) -> dict[str, Any] | None:
    """对照页版本映射中的权威摘要, 返回首个损坏页; 完好返回 None。"""
    from .versions import VersionMap

    versions = VersionMap.load(layout.version_map_file)
    if not layout.main_file.exists():
        return None
    with open(layout.main_file, "rb") as fh:
        blob = fh.read()
    for page_no in sorted(versions.pages):
        start = page_no * page_size
        frame = blob[start : start + page_size]
        version = versions.pages[page_no]
        intact = (
            len(frame) == page_size
            and Page.structural_valid(frame, page_size)
            and Page.digest_hex(frame) == version.digest
        )
        if not intact:
            return {
                "page_no": page_no,
                "expected_digest": version.digest,
                "actual_digest": sha256_hex(frame) if len(frame) == page_size else None,
                "offset": start,
            }
    return None


def assert_main_intact(layout: DataLayout, page_size: int) -> None:
    bad = find_first_corrupt_main_page(layout, page_size)
    if bad is not None:
        raise MainCorruptError(
            f"main page {bad['page_no']} corrupt",
            detail=bad,
        )


# ---------- 注入 1: 主文件页内容损坏 ----------
def inject_main_corrupt(layout: DataLayout, page_no: int, page_size: int) -> dict[str, Any]:
    layout.ensure()
    with open(layout.main_file, "r+b") as fh:
        # 破坏负载区(偏移 16, 位于 8 字节页头之后的 JSON 内),
        # 保持帧结构合法、仅令摘要不匹配 -> digest_mismatch
        fh.seek(page_no * page_size + 16)
        original = fh.read(4)
        flipped = bytes(b ^ 0xFF for b in original)
        fh.seek(page_no * page_size + 16)
        fh.write(flipped)
        fh.flush()
    return {"injection": "main_corrupt", "page_no": page_no}


# ---------- 注入 2: 双写缓冲镜像截断 ----------
def inject_buffer_truncate(layout: DataLayout, keep_bytes: int | None = None) -> dict[str, Any]:
    layout.ensure()
    data = layout.buffer_file.read_bytes()
    if keep_bytes is None:
        # 默认截掉最后 16 字节, 制造尾部镜像 payload/header 截断
        keep_bytes = max(0, len(data) - 16)
    layout.buffer_file.write_bytes(data[:keep_bytes])
    return {"injection": "buffer_truncate", "keep_bytes": keep_bytes, "original_bytes": len(data)}


# ---------- 注入 3: 页版本映射/视图依赖成环 ----------
def inject_dependency_cycle(layout: DataLayout, cycle_views: tuple[str, str]) -> dict[str, Any]:
    layout.ensure()
    obj = json.loads(layout.views_file.read_text(encoding="utf-8"))
    a, b = cycle_views

    def ensure_rollup(name: str) -> dict[str, Any]:
        for view in obj.get("views", []):
            if view["name"] == name:
                if view["kind"] != "rollup":
                    view["kind"] = "rollup"
                    view["params"] = {"sources": list(view.get("params", {}).values())}
                return view
        view = {"name": name, "kind": "rollup", "params": {"sources": []}}
        obj.setdefault("views", []).append(view)
        return view

    va = ensure_rollup(a)
    vb = ensure_rollup(b)
    va["params"].setdefault("sources", [])
    vb["params"].setdefault("sources", [])
    if b not in va["params"]["sources"]:
        va["params"]["sources"].append(b)
    if a not in vb["params"]["sources"]:
        vb["params"]["sources"].append(a)
    va["params"]["sources"] = sorted(va["params"]["sources"])
    vb["params"]["sources"] = sorted(vb["params"]["sources"])
    for view in obj.get("views", []):
        view["params"] = dict(view.get("params", {}))
    layout.views_file.write_text(
        canonical_json(obj), encoding="utf-8"
    )
    return {"injection": "depend_cycle", "cycle": [a, b, a]}


# ---------- 注入 4: 快照缓冲摘要不符 ----------
def inject_snapshot_stale(layout: DataLayout) -> dict[str, Any]:
    """篡改快照里记录的缓冲摘要, 使重开时与磁盘真实摘要不符。"""
    import sqlite3

    layout.ensure()
    with sqlite3.connect(str(layout.snapshot_db)) as conn:
        row = conn.execute(
            "SELECT session_id, buffer_digest FROM sessions "
            "ORDER BY created_seq DESC, session_id ASC LIMIT 1"
        ).fetchone()
        if row is None:
            raise FileNotFoundError("no snapshot row to tamper")
        session_id, old_digest = row
        fake = ("0" * len(old_digest)) if old_digest else "0" * 64
        conn.execute(
            "UPDATE sessions SET buffer_digest=? WHERE session_id=?",
            (fake, session_id),
        )
    return {"injection": "snapshot_stale", "session_id": session_id, "fake_digest": fake}


# ---------- 重放校验 (E_REPLAY) ----------
def clone_layout(src: DataLayout, dst_root: Path) -> DataLayout:
    dst_root.mkdir(parents=True, exist_ok=True)
    for name in [
        src.buffer_file.name,
        src.main_file.name,
        src.version_map_file.name,
        src.views_file.name,
        src.markers_file.name,
    ]:
        src_path = src.root / name
        if src_path.exists():
            shutil.copy2(src_path, dst_root / name)
    return DataLayout(dst_root)


def replay_compare(
    replayed_digests: dict[int, str],
    snapshot_digests: dict[str, str],
    replayed_invalidation: list[dict[str, Any]],
    snapshot_invalidation: list[dict[str, Any]],
) -> None:
    """逐项比较重放结果与快照, 差异定位到首个不一致页号与字段, 抛 E_REPLAY。"""
    all_pages = sorted({int(k) for k in replayed_digests} | {int(k) for k in snapshot_digests})
    for page_no in all_pages:
        new = replayed_digests.get(page_no)
        old = snapshot_digests.get(str(page_no), snapshot_digests.get(page_no))
        if new != old:
            raise ReplayError(
                f"replay page digest mismatch at page {page_no}",
                page_no=page_no,
                field="digest",
            )
    if replayed_invalidation != snapshot_invalidation:
        first = min(len(replayed_invalidation), len(snapshot_invalidation))
        for idx in range(first):
            if replayed_invalidation[idx] != snapshot_invalidation[idx]:
                page_no = replayed_invalidation[idx].get("page_no")
                raise ReplayError(
                    f"replay invalidation trace mismatch at index {idx}",
                    page_no=page_no,
                    field="invalidation",
                )
        raise ReplayError(
            "replay invalidation trace length mismatch",
            page_no=None,
            field="invalidation_length",
        )


def inject_replay_tamper(layout: DataLayout, page_no: int, page_size: int) -> dict[str, Any]:
    """在重放工作副本上篡改主文件, 制造 E_REPLAY(不改原会话目录)。"""
    return inject_main_corrupt(layout, page_no, page_size)
