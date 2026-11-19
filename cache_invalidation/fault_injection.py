"""异常注入。

约束：注入只复制并修改输入文件或改变传入参数，绝不修改实现代码。
- annotation_cycle: 在临时副本中新增一条制造注释环的引用，期望 E_GRAPH。
- truncated_snapshot: 截断快照文件尾部，期望 E_SNAPSHOT。
- missing_inherit: 改写命名空间父级为不存在的上游，期望 E_INHERIT。
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class InjectedPaths:
    namespace_path: Path
    entry_path: Path
    snapshot_path: Path | None


def inject_annotation_cycle(namespace_path: str | Path,
                            entry_path: str | Path,
                            work_dir: str | Path) -> InjectedPaths:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    namespace_copy = work / Path(namespace_path).name
    entry_copy = work / Path(entry_path).name
    namespace_copy.write_text(
        Path(namespace_path).read_text(encoding="utf-8"), encoding="utf-8")

    entries_doc = json.loads(Path(entry_path).read_text(encoding="utf-8"))
    entries = entries_doc["entries"]
    first = entries[0]
    last = entries[-1]
    # 让最后一个条目反向注释第一个条目；样本边的方向是 first -> ... -> last，
    # 因而该新增引用必然闭合出一个环。
    last.setdefault("annotations", []).append({
        "namespace": first["namespace"],
        "key": first["key"],
    })
    entry_copy.write_text(
        json.dumps(entries_doc, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return InjectedPaths(namespace_copy, entry_copy, None)


def inject_truncated_snapshot(snapshot_path: str | Path,
                              work_dir: str | Path) -> Path:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    target = work / "truncated_snapshot.json"
    text = Path(snapshot_path).read_text(encoding="utf-8")
    keep = max(1, int(len(text) * 0.65))
    target.write_text(text[:keep], encoding="utf-8")
    return target


def inject_missing_inheritance(namespace_path: str | Path,
                               entry_path: str | Path,
                               work_dir: str | Path) -> InjectedPaths:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    entry_copy = work / Path(entry_path).name
    entry_copy.write_text(
        Path(entry_path).read_text(encoding="utf-8"), encoding="utf-8")

    namespace_doc = json.loads(
        Path(namespace_path).read_text(encoding="utf-8"))
    namespace_doc = copy.deepcopy(namespace_doc)
    target = namespace_doc["namespaces"][-1]
    missing_parent = "missing_upstream_namespace"
    parents = target.setdefault("parents", [])
    if missing_parent not in parents:
        parents.append(missing_parent)
    namespace_copy = work / Path(namespace_path).name
    namespace_copy.write_text(
        json.dumps(namespace_doc, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return InjectedPaths(namespace_copy, entry_copy, None)
