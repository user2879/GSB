"""Parsing and validation layer.

Invalid rows never reach computation: they are recorded in the failure list
and excluded from the returned valid collections.
"""
from __future__ import annotations

from typing import Any

from .common import (
    CODE_DEF_CYCLE,
    CODE_DEF_PARAM,
    CODE_EVENT_INVALID,
    Failure,
    is_finite_number,
)
from .models import (
    Event,
    FeatureDef,
    InjectionRule,
    LabelDef,
    validate_agg,
    validate_side,
)


def _parse_context(raw: Any) -> dict[str, float]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            out[str(key)] = float(value)
        elif isinstance(value, float):
            out[str(key)] = value
    return out


def load_events(path: str, allowed_behaviors: list[str]) -> tuple[list[Event], list[Failure]]:
    from .common import read_jsonl

    failures: list[Failure] = []
    events: list[Event] = []
    seen_ids: set[str] = set()
    prev_ts: int | None = None

    for line_no, row in enumerate(read_jsonl(path), start=1):
        eid = str(row.get("event_id", ""))
        prefix = eid or f"line:{line_no}"

        def fail(detail: str) -> None:
            failures.append(Failure("load", prefix, CODE_EVENT_INVALID, detail))

        if not eid or eid in seen_ids:
            fail("missing or duplicate event_id")
            continue
        seen_ids.add(eid)

        uid = row.get("user_id")
        iid = row.get("item_id")
        if not isinstance(uid, str) or not uid or not isinstance(iid, str) or not iid:
            fail("user_id/item_id must be non-empty strings")
            continue

        ts = row.get("ts_ms")
        if isinstance(ts, bool) or not isinstance(ts, int) or ts < 0:
            fail("ts_ms must be a non-negative integer")
            continue
        if prev_ts is not None and ts <= prev_ts:
            fail("ts_ms must be strictly increasing")
            continue
        prev_ts = ts

        behavior = row.get("behavior")
        if not isinstance(behavior, str) or behavior not in allowed_behaviors:
            fail(f"behavior {behavior!r} not in enum")
            continue

        context = _parse_context(row.get("context", {}))
        bad_ctx = [k for k, v in context.items() if not is_finite_number(v)]
        if bad_ctx:
            fail(f"non-finite context values: {','.join(sorted(bad_ctx))}")
            continue

        events.append(Event(eid, uid, iid, behavior, ts, context))

    return events, failures


def load_feature_defs(path: str) -> tuple[list[FeatureDef], list[Failure]]:
    from .common import read_json

    failures: list[Failure] = []
    rows = read_json(path)
    defs: list[FeatureDef] = []
    if not isinstance(rows, list):
        failures.append(Failure("load", "feature_defs", CODE_DEF_PARAM,
                                "feature definition table must be a JSON array"))
        return defs, failures

    names: set[str] = set()
    for row in rows:
        name = str(row.get("name", ""))

        def fail(code: str, detail: str) -> None:
            failures.append(Failure("load", name or "<unnamed>", code, detail))

        if not name or name in names:
            fail(CODE_DEF_PARAM, "missing or duplicate feature name")
            continue
        names.add(name)

        side = row.get("side")
        if not validate_side(side):
            fail(CODE_DEF_PARAM, f"side must be user|item, got {side!r}")
            continue

        window = row.get("window_ms")
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            fail(CODE_DEF_PARAM, "window_ms must be a positive integer")
            continue

        agg = row.get("agg")
        if not validate_agg(agg):
            fail(CODE_DEF_PARAM, f"agg must be one of count/sum/mean/last, got {agg!r}")
            continue

        upstream = row.get("upstream", [])
        source_field = row.get("source_field")
        if not isinstance(upstream, list) or any(not isinstance(u, str) for u in upstream):
            fail(CODE_DEF_PARAM, "upstream must be a list of strings")
            continue
        if upstream:
            if source_field:
                fail(CODE_DEF_PARAM, "derived feature must not declare source_field")
                continue
        else:
            if not isinstance(source_field, str) or not source_field:
                fail(CODE_DEF_PARAM, "base feature requires source_field")
                continue

        empty_value = row.get("empty_value")
        if empty_value is not None and not isinstance(empty_value, (int, float, str)):
            fail(CODE_DEF_PARAM, "empty_value must be numeric or null")
            continue

        defs.append(FeatureDef(
            name=name,
            side=side,
            window_ms=window,
            agg=agg,
            source_field=source_field if not upstream else None,
            upstream=list(upstream),
            empty_value=None if empty_value is None else int(empty_value)
            if isinstance(empty_value, int) and not isinstance(empty_value, bool)
            else (float(empty_value) if isinstance(empty_value, (float, str)) else None),
        ))

    # Referential integrity + DAG acyclicity (Kahn's algorithm).
    valid_names = {d.name for d in defs}
    for d in defs:
        unknown = [u for u in d.upstream if u not in valid_names]
        if unknown:
            d.valid = False
            d.invalid_reason = f"unknown upstream: {','.join(unknown)}"
            failures.append(Failure("load", d.name, CODE_DEF_PARAM, d.invalid_reason))

    indegree = {d.name: 0 for d in defs}
    edges: dict[str, list[str]] = {d.name: [] for d in defs}
    for d in defs:
        for up in d.upstream:
            if up in indegree:
                edges[up].append(d.name)
                indegree[d.name] += 1
    ready = sorted(n for n, deg in indegree.items() if deg == 0)
    order: list[str] = []
    while ready:
        node = ready.pop(0)
        order.append(node)
        for child in sorted(edges[node]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()

    cyclic = [d for d in defs if d.name not in order]
    for d in cyclic:
        d.valid = False
        d.invalid_reason = "dependency graph contains a cycle"
        failures.append(Failure("load", d.name, CODE_DEF_CYCLE, d.invalid_reason))

    # Any feature depending (transitively) on an invalid feature is invalid too.
    by_name = {d.name: d for d in defs}
    changed = True
    while changed:
        changed = False
        for d in defs:
            if d.valid and any(not by_name[u].valid for u in d.upstream if u in by_name):
                d.valid = False
                d.invalid_reason = "transitive dependency is invalid"
                failures.append(Failure("load", d.name, CODE_DEF_PARAM, d.invalid_reason))
                changed = True

    # Deterministic topological ranks for the valid DAG.
    level: dict[str, int] = {}
    for name in order:
        d = by_name[name]
        level[name] = 0 if not d.upstream else 1 + max(level.get(u, 0) for u in d.upstream)
    for rank, name in enumerate(order):
        by_name[name].topo_rank = rank
        by_name[name].level = level[name]

    # Return in stable topological order; invalid defs stay at the end by rank -1.
    defs.sort(key=lambda d: (d.topo_rank < 0, d.topo_rank if d.topo_rank >= 0 else 0, d.name))
    return defs, failures


def load_label_defs(path: str, feature_names: set[str],
                    behaviors: list[str]) -> tuple[list[LabelDef], list[Failure]]:
    from .common import read_json

    failures: list[Failure] = []
    rows = read_json(path)
    labels: list[LabelDef] = []
    if not isinstance(rows, list):
        failures.append(Failure("load", "label_defs", CODE_DEF_PARAM,
                                "label definition must be a JSON array"))
        return labels, failures

    seen: set[str] = set()
    for row in rows:
        name = str(row.get("name", ""))

        def fail(detail: str) -> None:
            failures.append(Failure("load", name or "<unnamed>", CODE_DEF_PARAM, detail))

        if not name or name in seen:
            fail("missing or duplicate label name")
            continue
        seen.add(name)

        triggers = row.get("trigger_behaviors", [])
        if not isinstance(triggers, list) or not triggers or \
                any(t not in behaviors for t in triggers):
            fail("trigger_behaviors must be a non-empty subset of config behaviors")
            continue

        decision_feature = row.get("decision_feature")
        if decision_feature not in feature_names:
            fail(f"decision_feature {decision_feature!r} does not exist")
            continue
        observed = row.get("observed_features", [])
        if not isinstance(observed, list) or not observed or \
                any(f not in feature_names for f in observed):
            fail("observed_features must be an existing non-empty feature list")
            continue
        if decision_feature not in observed:
            fail("decision_feature must be included in observed_features")
            continue

        op = row.get("positive_op", "ge")
        if op not in ("ge", "gt", "le", "lt", "eq"):
            fail("positive_op must be one of ge/gt/le/lt/eq")
            continue
        threshold = row.get("threshold", 0)
        try:
            threshold_text = str(threshold)
            float(threshold_text)
        except (TypeError, ValueError):
            fail("threshold must be numeric")
            continue

        delay = row.get("delay_ms", 0)
        if isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
            fail("delay_ms must be a non-negative integer")
            continue

        labels.append(LabelDef(
            name=name,
            trigger_behaviors=list(triggers),
            decision_feature=decision_feature,
            threshold_raw=threshold_text,
            positive_op=op,
            observed_features=list(observed),
            delay_ms=delay,
        ))

    labels.sort(key=lambda x: x.name)
    return labels, failures


def load_injections(path: str) -> list[InjectionRule]:
    from .common import read_json

    raw = read_json(path)
    rules: list[InjectionRule] = []
    for row in raw.get("rules", []):
        rules.append(InjectionRule(
            injection_id=str(row["injection_id"]),
            event_id=str(row["event_id"]),
            stage=str(row["stage"]),
            kind=str(row["kind"]),
            context_field=str(row.get("context_field", "")),
            bad_ts_ms=int(row.get("bad_ts_ms", -1)),
        ))
    rules.sort(key=lambda r: r.injection_id)
    return rules
