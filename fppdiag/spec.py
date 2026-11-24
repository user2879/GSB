"""Pipeline / data-source / window specification model and strong validation.

Validation is intentionally strict and ordered: the *first* violation is
reported with :class:`SpecError` (code E_SPEC), carrying the offending stage or
data-source id and the offending field name.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .errors import SpecError
from .fixedpoint import ALLOWED_SCALE_EXPS, INT32_MAX, INT32_MIN
from .util import canonical_json, content_digest

STAGE_TYPES = ("standardize", "bucket", "log", "product", "truncate")
LOG_BASES = ("e", "2", "10")
# Stages that read directly from a data source rather than an upstream stage.
SOURCE_FED_TYPES = ("standardize", "log")


@dataclass(frozen=True)
class SourceSpec:
    sid: str
    kind: str                 # "integer" | "float"
    rows: int
    seed: int
    low: int
    high: int
    ingest_scale: int         # decimal scale exponent of generated ulps
    null_seed: int
    null_rate_bps: int        # 0..10000
    null_bitmap_len: int      # declared bitmap length; must equal rows


@dataclass(frozen=True)
class WindowSpec:
    ref_start: int
    ref_end: int               # exclusive
    obs_start: int
    obs_end: int               # exclusive


@dataclass(frozen=True)
class StageSpec:
    stage_id: str
    stype: str
    scale_exp: int
    source_id: str | None
    refs: tuple[str, ...]
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PipelineSpec:
    version: int
    session_scale_exp: int
    sources: dict[str, SourceSpec]
    window: WindowSpec
    stages: list[StageSpec]
    thresholds_bps: dict[str, int]
    block_size: int
    raw: dict[str, Any]

    def stage(self, stage_id: str) -> StageSpec:
        for st in self.stages:
            if st.stage_id == stage_id:
                return st
        raise KeyError(stage_id)

    def digest(self) -> str:
        return content_digest(self.raw)

    def sources_digest(self) -> str:
        payload = {sid: _source_payload(s) for sid, s in sorted(self.sources.items())}
        return content_digest(payload)


def _source_payload(s: SourceSpec) -> dict:
    return {"sid": s.sid, "kind": s.kind, "rows": s.rows, "seed": s.seed,
            "low": s.low, "high": s.high, "ingest_scale": s.ingest_scale,
            "null_seed": s.null_seed, "null_rate_bps": s.null_rate_bps}


def _require(cond: bool, location: str, field_name: str, message: str) -> None:
    if not cond:
        raise SpecError(message, location=location, field=field_name)


def _is_int(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _parse_source(raw: dict[str, Any]) -> SourceSpec:
    sid = str(raw.get("id", ""))
    _require(bool(sid), sid or "<source>", "id", "data source missing id")
    kind = raw.get("kind")
    _require(kind in ("integer", "float"), sid, "kind",
             f"source kind must be integer|float, got {kind!r}")
    rows = raw.get("rows")
    _require(_is_int(rows) and rows > 0, sid, "rows", "rows must be positive int")
    seed = raw.get("seed")
    _require(_is_int(seed) and seed >= 0, sid, "seed", "seed must be non-negative int")
    low = raw.get("low")
    high = raw.get("high")
    _require(_is_int(low), sid, "low", "low must be integer ulps")
    _require(_is_int(high), sid, "high", "high must be integer ulps")
    _require(high >= low, sid, "high", "high must be >= low")
    ingest_scale = raw.get("ingest_scale")
    _require(_is_int(ingest_scale) and ingest_scale in ALLOWED_SCALE_EXPS,
             sid, "ingest_scale", f"ingest_scale must be in {ALLOWED_SCALE_EXPS}")
    _require(INT32_MIN <= low <= INT32_MAX and INT32_MIN <= high <= INT32_MAX,
             sid, "low", "low/high must fit signed 32-bit envelope")
    null_seed = raw.get("null_seed")
    _require(_is_int(null_seed) and null_seed >= 0, sid, "null_seed",
             "null_seed must be non-negative int")
    null_rate = raw.get("null_rate_bps", 0)
    _require(_is_int(null_rate) and 0 <= null_rate <= 10000, sid,
             "null_rate_bps", "null_rate_bps must be in [0,10000]")
    bitmap_len = raw.get("null_bitmap_len", rows)
    _require(_is_int(bitmap_len), sid, "null_bitmap_len",
             "null_bitmap_len must be int")
    _require(bitmap_len == rows, sid, "null_bitmap_len",
             f"null bitmap length {bitmap_len} != rows {rows}")
    return SourceSpec(sid, kind, rows, seed, low, high, ingest_scale,
                      null_seed, null_rate, bitmap_len)


def _parse_window(raw: dict[str, Any], rows: int) -> WindowSpec:
    keys = ("ref_start", "ref_end", "obs_start", "obs_end")
    for k in keys:
        _require(_is_int(raw.get(k)), "<window>", k, f"window {k} must be int")
    w = WindowSpec(*(raw[k] for k in keys))
    _require(w.ref_start == 0, "<window>", "ref_start",
             "reference window must start at row 0")
    _require(w.ref_end == w.obs_start, "<window>", "obs_start",
             "window boundaries must be contiguous: ref_end == obs_start")
    _require(w.obs_end == rows, "<window>", "obs_end",
             f"observation window must end at row count {rows}")
    _require(w.ref_start < w.ref_end and w.obs_start < w.obs_end,
             "<window>", "ref_end", "each window must contain at least one row")
    return w


def _parse_stage(raw: dict[str, Any]) -> StageSpec:
    sid = str(raw.get("id", ""))
    _require(bool(sid), "<stage>", "id", "stage missing id")
    stype = raw.get("type")
    _require(stype in STAGE_TYPES, sid, "type",
             f"stage type must be one of {STAGE_TYPES}, got {stype!r}")
    scale_exp = raw.get("scale_exp")
    _require(_is_int(scale_exp) and scale_exp in ALLOWED_SCALE_EXPS, sid,
             "scale_exp", f"scale_exp must be in {ALLOWED_SCALE_EXPS}")
    source_id = raw.get("source")
    refs_raw = raw.get("refs", [])
    _require(isinstance(refs_raw, list) and all(isinstance(r, str) for r in refs_raw),
             sid, "refs", "refs must be a list of stage id strings")
    params = raw.get("params", {})
    _require(isinstance(params, dict), sid, "params", "params must be object")
    for pk, pv in params.items():
        if pk == "edges":
            _require(isinstance(pv, list) and pv and all(_is_int(e) for e in pv),
                     sid, "params.edges", "edges must be a non-empty int list")
            _require(all(INT32_MIN <= e <= INT32_MAX for e in pv),
                     sid, "params.edges", "edges must fit signed 32-bit envelope")
            continue
        _require(_is_int(pv), sid, f"params.{pk}",
                 f"param {pk} must be integer; fixed-point factors stay integer")
        _require(INT32_MIN <= pv <= INT32_MAX, sid, f"params.{pk}",
                 f"param {pk} must fit signed 32-bit envelope")
    if stype in SOURCE_FED_TYPES:
        _require(isinstance(source_id, str) and source_id, sid, "source",
                 f"{stype} stage must declare a data source")
        _require(not refs_raw, sid, "refs",
                 f"{stype} is source-fed and must not reference upstream stages")
    else:
        _require(source_id is None, sid, "source",
                 f"{stype} stage must be upstream-fed, not source-fed")
        _require(len(refs_raw) >= 1, sid, "refs",
                 f"{stype} stage requires at least one upstream ref")
    if stype == "product":
        _require(len(refs_raw) == 2, sid, "refs",
                 "product stage requires exactly two upstream refs")
    else:
        _require(len(refs_raw) <= 1, sid, "refs",
                         f"{stype} stage accepts at most one upstream ref")
    if stype == "log":
        base = params.get("base_flag")
        _require(base is not None, sid, "params.base_flag",
                 "log stage requires integer params.base_flag")
    return StageSpec(sid, stype, scale_exp, source_id, tuple(refs_raw),
                     dict(params))


def _topo_order(stages: list[StageSpec]) -> list[str]:
    """Kahn topological sort in declared order; raises on cycle/unknown ref."""
    ids = [s.stage_id for s in stages]
    known = set(ids)
    indeg = {s.stage_id: 0 for s in stages}
    adj: dict[str, list[str]] = {s.stage_id: [] for s in stages}
    for s in stages:
        for ref in s.refs:
            if ref not in known:
                raise SpecError(f"unknown upstream ref {ref!r}",
                                location=s.stage_id, field="refs")
            indeg[s.stage_id] += 1
            adj[ref].append(s.stage_id)
    ready = [sid for sid in ids if indeg[sid] == 0]
    order: list[str] = []
    while ready:
        cur = ready.pop(0)
        order.append(cur)
        for nxt in adj[cur]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                ready.append(nxt)
    if len(order) != len(ids):
        remaining = [sid for sid in ids if sid not in order]
        raise SpecError("stage dependency graph contains a cycle",
                        location=remaining[0], field="refs")
    return order


def load_spec(doc: dict[str, Any]) -> PipelineSpec:
    """Validate a raw JSON document and build an immutable PipelineSpec."""
    _require(isinstance(doc, dict), "<root>", "", "spec root must be object")
    version = doc.get("version")
    _require(_is_int(version) and version >= 1, "<root>", "version",
             "version must be a positive monotonically increasing integer")
    session_scale = doc.get("session_scale_exp")
    _require(_is_int(session_scale) and session_scale in ALLOWED_SCALE_EXPS,
             "<root>", "session_scale_exp",
             f"session_scale_exp must be in {ALLOWED_SCALE_EXPS}")
    block_size = doc.get("block_size", 64)
    _require(_is_int(block_size) and block_size >= 1, "<root>", "block_size",
             "block_size must be positive int")

    raw_sources = doc.get("sources")
    _require(isinstance(raw_sources, list) and raw_sources, "<root>", "sources",
             "sources must be a non-empty list")
    sources: dict[str, SourceSpec] = {}
    for rs in raw_sources:
        src = _parse_source(rs)
        _require(src.sid not in sources, src.sid, "id", "duplicate source id")
        sources[src.sid] = src
    row_counts = {s.rows for s in sources.values()}
    _require(len(row_counts) == 1, next(iter(sources)), "rows",
             "all sources must share the same row count for aligned windows")
    rows = row_counts.pop()

    win_raw = doc.get("window")
    _require(isinstance(win_raw, dict), "<root>", "window", "window required")
    window = _parse_window(win_raw, rows)

    raw_stages = doc.get("stages")
    _require(isinstance(raw_stages, list) and raw_stages, "<root>", "stages",
             "stages must be a non-empty list")
    stages: list[StageSpec] = []
    seen: set[str] = set()
    for rs in raw_stages:
        st = _parse_stage(rs)
        _require(st.stage_id not in seen, st.stage_id, "id",
                 "duplicate stage id")
        if st.source_id is not None:
            _require(st.source_id in sources, st.stage_id, "source",
                     f"unknown data source {st.source_id!r}")
        seen.add(st.stage_id)
        stages.append(st)
    _topo_order(stages)

    thresholds = doc.get("thresholds_bps")
    _require(isinstance(thresholds, dict), "<root>", "thresholds_bps",
             "thresholds_bps must be an object keyed by stage id")
    for st in stages:
        t = thresholds.get(st.stage_id)
        _require(_is_int(t), st.stage_id, "thresholds_bps",
                 f"missing or non-integer threshold for stage {st.stage_id}")
        _require(1 <= t <= 10000, st.stage_id, "thresholds_bps",
                 "threshold must be in [1,10000] basis points")
    extra = set(thresholds) - {s.stage_id for s in stages}
    _require(not extra, "<root>", "thresholds_bps",
             f"threshold declared for unknown stage: {sorted(extra)}")

    return PipelineSpec(version, session_scale, sources, window, stages,
                        dict(thresholds), block_size, doc)


def load_spec_file(path: str) -> PipelineSpec:
    import json
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    return load_spec(doc)


def spec_canonical_json(spec: PipelineSpec) -> str:
    return canonical_json(spec.raw)
