"""Spec / source / window JSON loading and strong validation (requirement 1).

Any violation raises SpecError (E_SPEC) carrying the first offending stage id
(or None for non-stage sections) and the field name. No computation may start
until validate_spec returns a Spec.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from .errors import SpecError

SUPPORTED_SCALES = (1, 10, 100, 1000, 10000)
STAGE_KINDS = ("std_diff", "window_avg", "quantile_bucket", "threshold_norm")
RECORD_FORMATS = (1, 2, 3)

# Integer params each stage kind must declare.
STAGE_PARAMS = {
    "std_diff": (),
    "window_avg": ("window",),
    "quantile_bucket": ("buckets", "lo", "hi"),
    "threshold_norm": ("threshold", "span"),
}


@dataclass(frozen=True)
class Stage:
    id: str
    kind: str
    params: dict
    scale: int
    upstream: tuple
    order: int


@dataclass(frozen=True)
class Source:
    id: str
    seed: int
    record_count: int
    record_format: int
    metrics: tuple
    default_scale: int
    null_bitmap: tuple  # 1 means the record at that index is null


@dataclass(frozen=True)
class Window:
    id: str
    start: int
    end: int
    order: int


@dataclass(frozen=True)
class Spec:
    spec_version: int
    task_interval: tuple  # (start, end) over the merged observation sequence
    stages: tuple
    sources: tuple
    windows: tuple
    drift_threshold: int
    dominance_threshold_ppm: int  # dominant-stage share threshold, parts per million
    raw: dict = field(default_factory=dict, compare=False)


def _require_int(value):
    return isinstance(value, int) and not isinstance(value, bool)


def load_spec(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return validate_spec(data)


def validate_spec(data):
    if not isinstance(data, dict):
        raise SpecError(None, "root", "spec root must be a JSON object")

    spec_version = data.get("spec_version")
    if not _require_int(spec_version) or spec_version < 1:
        raise SpecError(None, "spec_version", "spec_version must be a positive integer")

    interval = data.get("task_interval")
    if (
        not isinstance(interval, dict)
        or not _require_int(interval.get("start"))
        or not _require_int(interval.get("end"))
        or interval["start"] < 0
        or interval["end"] <= interval["start"]
    ):
        raise SpecError(None, "task_interval", "task_interval must be {start,end} with 0 <= start < end")

    stages = _validate_stages(data.get("stages"))
    windows = _validate_windows(data.get("windows"), interval)
    sources = _validate_sources(data.get("sources"))

    drift_threshold = data.get("drift_threshold", 0)
    if not _require_int(drift_threshold) or drift_threshold < 0:
        raise SpecError(None, "drift_threshold", "drift_threshold must be a non-negative integer")
    dominance = data.get("dominance_threshold_ppm", 500000)
    if not _require_int(dominance) or not 0 < dominance <= 1000000:
        raise SpecError(None, "dominance_threshold_ppm", "dominance threshold must be in (0, 1000000]")

    return Spec(
        spec_version=spec_version,
        task_interval=(interval["start"], interval["end"]),
        stages=tuple(stages),
        sources=tuple(sources),
        windows=tuple(windows),
        drift_threshold=drift_threshold,
        dominance_threshold_ppm=dominance,
        raw=data,
    )


def _validate_stages(stages):
    if not isinstance(stages, list) or not stages:
        raise SpecError(None, "stages", "stages must be a non-empty list")
    seen = set()
    parsed = []
    for order, entry in enumerate(stages):
        if not isinstance(entry, dict):
            raise SpecError(None, "stages", "stage entry must be an object")
        stage_id = entry.get("id")
        if not isinstance(stage_id, str) or not stage_id:
            raise SpecError(str(stage_id), "id", "stage id must be a non-empty string")
        if stage_id in seen:
            raise SpecError(stage_id, "id", "duplicate stage id")
        seen.add(stage_id)
        kind = entry.get("kind")
        if kind not in STAGE_KINDS:
            raise SpecError(stage_id, "kind", "unsupported stage kind: %r" % (kind,))
        scale = entry.get("scale")
        if not _require_int(scale) or scale not in SUPPORTED_SCALES:
            raise SpecError(stage_id, "scale", "scale must be one of %s" % (SUPPORTED_SCALES,))
        params = entry.get("params", {})
        if not isinstance(params, dict):
            raise SpecError(stage_id, "params", "params must be an object")
        for name in STAGE_PARAMS[kind]:
            value = params.get(name)
            if not _require_int(value):
                raise SpecError(stage_id, "params.%s" % name, "param %s must be an integer" % name)
            if name in ("window", "buckets", "span") and value <= 0:
                raise SpecError(stage_id, "params.%s" % name, "param %s must be positive" % name)
        if kind == "quantile_bucket" and params["hi"] <= params["lo"]:
            raise SpecError(stage_id, "params.hi", "quantile_bucket requires hi > lo")
        upstream = entry.get("upstream", [])
        if not isinstance(upstream, list) or any(not isinstance(u, str) for u in upstream):
            raise SpecError(stage_id, "upstream", "upstream must be a list of stage ids")
        parsed.append(Stage(stage_id, kind, dict(params), scale, tuple(upstream), order))

    by_id = {s.id: s for s in parsed}
    for stage in parsed:
        for up in stage.upstream:
            if up not in by_id:
                raise SpecError(stage.id, "upstream", "unknown upstream stage: %s" % up)

    _check_acyclic(parsed)
    return parsed


def _check_acyclic(stages):
    """DFS in declaration order; report the first stage found on a cycle."""
    by_id = {s.id: s for s in stages}
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {s.id: WHITE for s in stages}

    def visit(node):
        color[node.id] = GRAY
        for up in node.upstream:
            parent = by_id[up]
            if color[parent.id] == GRAY:
                raise SpecError(parent.id, "upstream", "dependency cycle via stage %s" % node.id)
            if color[parent.id] == WHITE:
                visit(parent)
        color[node.id] = BLACK

    for stage in stages:
        if color[stage.id] == WHITE:
            visit(stage)


def topo_order(stages):
    """Deterministic topological order (declaration order among ready nodes)."""
    by_id = {s.id: s for s in stages}
    indegree = {s.id: 0 for s in stages}
    dependents = {s.id: [] for s in stages}
    for s in stages:
        for up in s.upstream:
            indegree[s.id] += 1
            dependents[up].append(s.id)
    ready = [s.id for s in stages if indegree[s.id] == 0]
    order = []
    while ready:
        node = ready.pop(0)
        order.append(by_id[node])
        for dep in dependents[node]:
            indegree[dep] -= 1
            if indegree[dep] == 0:
                ready.append(dep)
        ready.sort(key=lambda sid: by_id[sid].order)
    return order


def _validate_windows(windows, interval):
    if not isinstance(windows, list) or not windows:
        raise SpecError(None, "windows", "windows must be a non-empty list")
    parsed = []
    expected_start = interval["start"]
    for order, entry in enumerate(windows):
        if not isinstance(entry, dict):
            raise SpecError(None, "windows", "window entry must be an object")
        wid = entry.get("id")
        start, end = entry.get("start"), entry.get("end")
        if not isinstance(wid, str) or not _require_int(start) or not _require_int(end) or end <= start:
            raise SpecError(None, "windows", "window %r must have id and integer start < end" % (wid,))
        if start != expected_start:
            raise SpecError(None, "windows", "window %s does not start at %d (gap or overlap)" % (wid, expected_start))
        if end > interval["end"]:
            raise SpecError(None, "windows", "window %s ends past task interval end" % wid)
        parsed.append(Window(wid, start, end, order))
        expected_start = end
    if expected_start != interval["end"]:
        raise SpecError(None, "windows", "windows do not cover the task interval up to %d" % interval["end"])
    return parsed


def _validate_sources(sources):
    if not isinstance(sources, list) or not sources:
        raise SpecError(None, "sources", "sources must be a non-empty list")
    seen = set()
    parsed = []
    for entry in sources:
        if not isinstance(entry, dict):
            raise SpecError(None, "sources", "source entry must be an object")
        sid = entry.get("id")
        if not isinstance(sid, str) or not sid:
            raise SpecError(None, "sources.id", "source id must be a non-empty string")
        if sid in seen:
            raise SpecError(None, "sources.id", "duplicate source id %s" % sid)
        seen.add(sid)
        seed = entry.get("seed")
        count = entry.get("record_count")
        fmt = entry.get("record_format")
        metrics = entry.get("metrics")
        default_scale = entry.get("default_scale")
        bitmap = entry.get("null_bitmap")
        if not _require_int(seed):
            raise SpecError(None, "sources.seed", "source %s seed must be an integer" % sid)
        if not _require_int(count) or count < 0:
            raise SpecError(None, "sources.record_count", "source %s record_count must be >= 0" % sid)
        if fmt not in RECORD_FORMATS:
            raise SpecError(None, "sources.record_format", "source %s record_format must be in %s" % (sid, RECORD_FORMATS))
        if not isinstance(metrics, list) or not metrics or any(not isinstance(m, str) for m in metrics):
            raise SpecError(None, "sources.metrics", "source %s metrics must be a non-empty list" % sid)
        if not _require_int(default_scale) or default_scale not in SUPPORTED_SCALES:
            raise SpecError(None, "sources.default_scale", "source %s default_scale unsupported" % sid)
        if not isinstance(bitmap, list) or any(b not in (0, 1) for b in bitmap):
            raise SpecError(None, "sources.null_bitmap", "source %s null_bitmap must be a list of 0/1" % sid)
        if len(bitmap) != count:
            raise SpecError(
                None,
                "sources.null_bitmap",
                "source %s null_bitmap length %d != record_count %d" % (sid, len(bitmap), count),
            )
        parsed.append(Source(sid, seed, count, fmt, tuple(metrics), default_scale, tuple(bitmap)))
    return parsed
