"""恢复后的一致性重放校验。

从快照恢复后，按快照记录的变更集序列与重算参数重新推导受影响条目
与失效视图并重算，将受影响集合、各条目摘要、视图内容摘要与重算账本
同快照逐项比较；差异定位到首个不一致的条目标识与字段并以 E_REPLAY
报告。账本中的单调时钟毫秒不参与比较（每次运行必然不同）。
"""

from __future__ import annotations

from typing import Any

from .errors import ReplayError
from .graph import PropertyGraph
from .indexes import IndexDef
from .session import MaintenanceSession
from .structlog import JsonLogger
from .views import ViewDef


def _fail(entry: str, field: str, expected: Any, actual: Any) -> None:
    raise ReplayError(
        f"replay mismatch at {entry} field {field}",
        [{"element": entry, "field": field, "expected": expected, "actual": actual}],
    )


def _normalize_ledger(ledger: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {k: v for k, v in entry.items() if not k.endswith("_ms")}
        for entry in ledger
    ]


def replay_snapshot(path: str, logger: JsonLogger | None = None) -> dict[str, Any]:
    snap = MaintenanceSession.load_snapshot(path)
    session = MaintenanceSession(
        PropertyGraph.from_content(snap["base_graph"], version=0),
        [IndexDef.from_dict(d) for d in snap["index_defs"]],
        [ViewDef.from_dict(d) for d in snap["view_defs"]],
        concurrency=snap["params"]["concurrency"],
        budget=snap["params"]["budget"],
        logger=logger,
    )
    session.build_indexes()
    for changeset in snap["changesets"]:
        session.apply_changeset(changeset)

    checks = 0

    # 1. 图内容摘要
    if session.graph.content_digest() != snap["graph"]["digest"]:
        _fail("graph", "digest", snap["graph"]["digest"], session.graph.content_digest())
    checks += 1

    # 2. 失效轨迹（受影响集合与失效视图）
    expected_trace = snap["invalidation_trace"]
    if len(session.invalidation_trace) != len(expected_trace):
        _fail("invalidation_trace", "length", len(expected_trace), len(session.invalidation_trace))
    for got, exp in zip(session.invalidation_trace, expected_trace):
        cs = exp["changeset"]
        for field in ("affected_entries", "invalid_views"):
            if got[field] != exp[field]:
                _fail(f"trace:{cs}", field, exp[field], got[field])
        checks += 1

    # 3. 索引条目版本与摘要
    for name in sorted({*session.indexes.entries, *snap["entries"]}):
        got_entries = session.indexes.entries.get(name, {})
        exp_entries = snap["entries"].get(name, {})
        for key in sorted({*got_entries, *exp_entries}):
            ident = f"{name}:{key}"
            got = got_entries.get(key)
            exp = exp_entries.get(key)
            if got is None or exp is None:
                _fail(ident, "presence", exp is not None, got is not None)
            if got.entry_version != exp["entry_version"]:
                _fail(ident, "entry_version", exp["entry_version"], got.entry_version)
            if got.digest != exp["digest"]:
                _fail(ident, "digest", exp["digest"], got.digest)
            if got.status != exp["status"]:
                _fail(ident, "status", exp["status"], got.status)
            checks += 1

    # 4. 视图版本与内容摘要
    for name in sorted({*session.views.states, *snap["views"]}):
        ident = f"view:{name}"
        got = session.views.states.get(name)
        exp = snap["views"].get(name)
        if got is None or exp is None:
            _fail(ident, "presence", exp is not None, got is not None)
        if got.version != exp["version"]:
            _fail(ident, "version", exp["version"], got.version)
        if got.digest != exp["digest"]:
            _fail(ident, "digest", exp["digest"], got.digest)
        if got.status != exp["status"]:
            _fail(ident, "status", exp["status"], got.status)
        checks += 1

    # 5. 重算账本（忽略单调时钟毫秒）
    got_ledger = _normalize_ledger(session.ledger)
    exp_ledger = _normalize_ledger(snap["recompute_ledger"])
    if len(got_ledger) != len(exp_ledger):
        _fail("recompute_ledger", "length", len(exp_ledger), len(got_ledger))
    for i, (got, exp) in enumerate(zip(got_ledger, exp_ledger)):
        if got != exp:
            for field in sorted({*got, *exp}):
                if got.get(field) != exp.get(field):
                    _fail(f"ledger[{i}]:{exp.get('target')}", field, exp.get(field), got.get(field))
        checks += 1

    if logger:
        logger.info("I_REPLAY", "replay verification passed", checks=checks, snapshot=path)
    return {"status": "ok", "checks": checks, "snapshot": str(path)}
