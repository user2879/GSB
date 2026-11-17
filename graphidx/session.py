"""索引维护会话：变更应用、失效推导、预算调度、快照与整体替换恢复。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .canon import digest as canonical_digest
from .errors import GraphError, IndexIntegrityError
from .graph import PropertyGraph
from .indexes import IndexDef, IndexEntry, IndexStore
from .query import QueryEngine
from .scheduler import BudgetScheduler, Task
from .structlog import JsonLogger
from .views import ViewDef, ViewState, ViewStore

SNAPSHOT_VERSION = 1


def _load_json_file(path: str | Path, code: str = "E_GRAPH") -> Any:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise GraphError(
            f"cannot read file {path}",
            [{"element": str(path), "field": "file", "message": str(exc)}],
        ) from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise GraphError(
            f"file {path} is not valid JSON",
            [{"element": str(path), "field": "json", "message": str(exc)}],
        ) from exc


def load_index_defs(path: str | Path) -> list[IndexDef]:
    data = _load_json_file(path)
    raw = data.get("indexes") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        raise IndexIntegrityError(
            "index definition file must contain an 'indexes' list",
            [{"element": str(path), "field": "indexes", "message": "expected list"}],
        )
    return [IndexDef.from_dict(item) for item in raw]


def load_view_defs(path: str | Path) -> list[ViewDef]:
    data = _load_json_file(path)
    raw = data.get("views") if isinstance(data, dict) else None
    if not isinstance(raw, list):
        raise GraphError(
            "view definition file must contain a 'views' list",
            [{"element": str(path), "field": "views", "message": "expected list"}],
        )
    return [ViewDef.from_dict(item) for item in raw]


class MaintenanceSession:
    """一次完整的索引维护会话。"""

    def __init__(
        self,
        graph: PropertyGraph,
        index_defs: list[IndexDef],
        view_defs: list[ViewDef],
        concurrency: int = 4,
        budget: int = 100,
        logger: JsonLogger | None = None,
    ):
        self.graph = graph
        self.indexes = IndexStore(index_defs)
        self.views = ViewStore(view_defs)
        self.concurrency = concurrency
        self.budget = budget
        self.invalidation_trace: list[dict[str, Any]] = []
        self.ledger: list[dict[str, Any]] = []
        self.applied_changesets: list[dict[str, Any]] = []
        self.base_graph_content = graph.to_content()
        self.logger = logger or JsonLogger()
        self.query = QueryEngine(self)

    # ------------------------------------------------------------------
    # 装载
    # ------------------------------------------------------------------
    @classmethod
    def from_files(
        cls,
        graph_path: str | Path,
        indexes_path: str | Path,
        views_path: str | Path,
        **kwargs: Any,
    ) -> "MaintenanceSession":
        graph = PropertyGraph.from_dict(_load_json_file(graph_path))
        index_defs = load_index_defs(indexes_path)
        view_defs = load_view_defs(views_path)
        session = cls(graph, index_defs, view_defs, **kwargs)
        session.views.validate_sources(session.graph)
        return session

    # ------------------------------------------------------------------
    # 全量构建
    # ------------------------------------------------------------------
    def build_indexes(self) -> dict[str, Any]:
        report = self.indexes.build_full(self.graph)
        report["views"] = self.views.compute_all(self.graph)
        self.logger.info(
            "I_BUILD",
            "full index build finished",
            graph_version=report["graph_version"],
            graph_digest=report["graph_digest"],
            index_def_version=report["index_def_version"],
        )
        return report

    # ------------------------------------------------------------------
    # 增量维护
    # ------------------------------------------------------------------
    def apply_changeset(self, changeset: dict[str, Any]) -> dict[str, Any]:
        change = self.graph.apply_changeset(changeset)

        affected = self.indexes.derive_affected(change)
        trace_entries = []
        for name, key in affected:
            entry = self.indexes.entries[name].get(key)
            trace_entries.append(
                {
                    "index": name,
                    "key": key,
                    "previous_version": entry.entry_version if entry else None,
                }
            )
            self.indexes.mark_invalid(name, key)
        invalid_views = self.views.invalidate(change)

        tasks: list[Task] = []
        seq = len(self.ledger)
        view_seq: dict[str, int] = {}
        for name, key in affected:
            tasks.append(
                Task(seq=seq, kind="index", target=f"{name}:{key}", fn=self._make_index_fn(name, key))
            )
            seq += 1
        for view_name in invalid_views:
            deps = tuple(view_seq[up] for up in self.views.defs[view_name].depends_on if up in view_seq)
            view_seq[view_name] = seq
            tasks.append(
                Task(seq=seq, kind="view", target=view_name, fn=self._make_view_fn(view_name), deps=deps)
            )
            seq += 1

        scheduler = BudgetScheduler(self.concurrency, self.budget)
        scheduled = scheduler.execute(tasks)
        self.ledger.extend(scheduled["ledger"])

        trace = {
            "changeset": change.changeset_id,
            "graph_version": change.version_after,
            "graph_digest": change.digest_after,
            "affected_entries": [[name, key] for name, key in affected],
            "invalid_views": list(invalid_views),
            "entries": trace_entries,
        }
        self.invalidation_trace.append(trace)
        self.applied_changesets.append(changeset)

        report = {
            "changeset": change.changeset_id,
            "graph_version": change.version_after,
            "graph_digest": change.digest_after,
            "affected_entries": trace["affected_entries"],
            "invalid_views": trace["invalid_views"],
            "recomputed": scheduled["recomputed"],
            "skipped": scheduled["skipped"],
            "ledger": scheduled["ledger"],
        }
        self.logger.info(
            "I_CHANGESET",
            "changeset applied and indexes maintained",
            changeset=change.changeset_id,
            graph_version=change.version_after,
            recomputed=scheduled["recomputed"],
            skipped=scheduled["skipped"],
        )
        return report

    def _make_index_fn(self, name: str, key: str):
        def fn() -> Any:
            return self.indexes.recompute(name, key, self.graph)

        return fn

    def _make_view_fn(self, view_name: str):
        def fn() -> Any:
            return self.views.recompute(view_name, self.graph)

        return fn

    # ------------------------------------------------------------------
    # 快照持久化
    # ------------------------------------------------------------------
    def snapshot_dict(self) -> dict[str, Any]:
        return {
            "snapshot_version": SNAPSHOT_VERSION,
            "params": {"concurrency": self.concurrency, "budget": self.budget},
            "base_graph": self.base_graph_content,
            "graph": {
                "version": self.graph.version,
                "digest": self.graph.content_digest(),
                "content": self.graph.to_content(),
            },
            "index_defs": [self.indexes.defs[n].to_dict() for n in sorted(self.indexes.defs)],
            "index_def_version": self.indexes.defs_digest(),
            "view_defs": [self.views.defs[n].to_dict() for n in sorted(self.views.defs)],
            "entries": {
                name: {
                    key: self.indexes.entries[name][key].to_dict()
                    for key in sorted(self.indexes.entries[name])
                }
                for name in sorted(self.indexes.entries)
            },
            "views": {name: self.views.states[name].to_dict() for name in self.views.order},
            "invalidation_trace": self.invalidation_trace,
            "recompute_ledger": self.ledger,
            "changesets": self.applied_changesets,
        }

    def save_snapshot(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.snapshot_dict(), ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.logger.info("I_SNAPSHOT", "snapshot saved", path=str(path))

    # ------------------------------------------------------------------
    # 快照装载与完整性校验（E_INDEX）
    # ------------------------------------------------------------------
    @staticmethod
    def load_snapshot(path: str | Path) -> dict[str, Any]:
        path = Path(path)
        try:
            snap = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IndexIntegrityError(
                f"snapshot {path} is unreadable",
                [{"element": str(path), "field": "file", "message": str(exc)}],
            ) from exc
        errors: list[dict[str, str]] = []
        graph = snap.get("graph", {})
        if canonical_digest(graph.get("content")) != graph.get("digest"):
            errors.append({"element": "graph", "field": "digest", "message": "graph content digest mismatch"})
        for name, entries in (snap.get("entries") or {}).items():
            for key, entry in entries.items():
                if canonical_digest(entry.get("content")) != entry.get("digest"):
                    errors.append(
                        {
                            "element": f"{name}:{key}",
                            "field": "digest",
                            "message": "index entry digest mismatch (tampered?)",
                        }
                    )
        for name, view in (snap.get("views") or {}).items():
            content = view.get("content")
            if content is not None and canonical_digest(content) != view.get("digest"):
                errors.append(
                    {"element": f"view:{name}", "field": "digest", "message": "view digest mismatch (tampered?)"}
                )
        if errors:
            raise IndexIntegrityError("snapshot integrity check failed", errors)
        return snap

    @classmethod
    def from_snapshot(cls, path: str | Path, logger: JsonLogger | None = None) -> "MaintenanceSession":
        snap = cls.load_snapshot(path)
        session = cls(
            PropertyGraph.from_content(snap["graph"]["content"], version=snap["graph"]["version"]),
            [IndexDef.from_dict(d) for d in snap["index_defs"]],
            [ViewDef.from_dict(d) for d in snap["view_defs"]],
            concurrency=snap["params"]["concurrency"],
            budget=snap["params"]["budget"],
            logger=logger,
        )
        session._adopt(snap)
        return session

    def _adopt(self, snap: dict[str, Any]) -> None:
        """以快照内容整体替换当前会话状态（不做增量合并）。"""
        self.graph = PropertyGraph.from_content(
            snap["graph"]["content"], version=snap["graph"]["version"]
        )
        self.indexes = IndexStore([IndexDef.from_dict(d) for d in snap["index_defs"]])
        for name, entries in snap["entries"].items():
            for key, entry in entries.items():
                self.indexes.entries[name][key] = IndexEntry.from_dict(entry)
        self.views = ViewStore([ViewDef.from_dict(d) for d in snap["view_defs"]])
        for name, view in snap["views"].items():
            self.views.states[name] = ViewState.from_dict(view)
        self.invalidation_trace = list(snap["invalidation_trace"])
        self.ledger = list(snap["recompute_ledger"])
        self.applied_changesets = list(snap["changesets"])
        self.base_graph_content = snap["base_graph"]
        self.concurrency = snap["params"]["concurrency"]
        self.budget = snap["params"]["budget"]
        self.query = QueryEngine(self)

    # ------------------------------------------------------------------
    # 恢复：版本识别 + 整体替换
    # ------------------------------------------------------------------
    def restore(self, snapshot_path: str | Path) -> dict[str, Any]:
        snap = self.load_snapshot(snapshot_path)
        in_sync = (
            snap["graph"]["digest"] == self.graph.content_digest()
            and snap["index_def_version"] == self.indexes.defs_digest()
        )
        if in_sync:
            report = {"replaced": False, "changed_entries": [], "changed_views": []}
            self.logger.info("I_RESTORE", "snapshot in sync, no replacement", **report)
            return report

        old_entries = {
            (name, key): (e.entry_version, e.digest)
            for name, entries in self.indexes.entries.items()
            for key, e in entries.items()
        }
        old_views = {
            name: (s.version, s.digest) for name, s in self.views.states.items()
        }
        self._adopt(snap)

        changed_entries = []
        new_keys = {
            (name, key)
            for name, entries in self.indexes.entries.items()
            for key in entries
        }
        for name, key in sorted(old_entries.keys() | new_keys):
            old = old_entries.get((name, key))
            new_entry = self.indexes.entries.get(name, {}).get(key)
            new = (new_entry.entry_version, new_entry.digest) if new_entry else None
            if old != new:
                changed_entries.append(
                    {
                        "index": name,
                        "key": key,
                        "old_version": old[0] if old else None,
                        "new_version": new[0] if new else None,
                    }
                )
        changed_views = []
        new_view_names = set(self.views.states)
        for name in sorted(old_views.keys() | new_view_names):
            old = old_views.get(name)
            state = self.views.states.get(name)
            new = (state.version, state.digest) if state else None
            if old != new:
                changed_views.append(
                    {
                        "view": name,
                        "old_version": old[0] if old else None,
                        "new_version": new[0] if new else None,
                    }
                )
        report = {
            "replaced": True,
            "changed_entries": changed_entries,
            "changed_views": changed_views,
        }
        self.logger.info(
            "I_RESTORE",
            "snapshot diverged, wholesale replacement applied",
            changed_entries=len(changed_entries),
            changed_views=len(changed_views),
        )
        return report
