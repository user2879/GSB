"""Core domain models, business error codes, fixed-point helpers,
a minimal YAML-subset parser, and the loading/validation layer."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

# ---------------------------------------------------------------- error codes
E_DUP_REQUEST = "E_DUP_REQUEST"
E_DUP_CANDIDATE = "E_DUP_CANDIDATE"
E_BAD_ID = "E_BAD_ID"
E_BAD_SCORE = "E_BAD_SCORE"
E_BAD_TOPK = "E_BAD_TOPK"
E_INVALID_RATIO = "E_INVALID_RATIO"
E_TIER_OUT_OF_RANGE = "E_TIER_OUT_OF_RANGE"
E_EMPTY_CHANNEL = "E_EMPTY_CHANNEL"
E_UNKNOWN_CHANNEL = "E_UNKNOWN_CHANNEL"
E_BAD_CONFIG = "E_BAD_CONFIG"
E_BAD_JSON = "E_BAD_JSON"
E_POINT_BUDGET_PENDING = "E_POINT_BUDGET_PENDING"
E_INJECTED_FAULT = "E_INJECTED_FAULT"


class BizError(Exception):
    """Business-level validation error carrying an E_xxx code."""

    def __init__(self, code, message, request_id=None):
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message
        self.request_id = request_id


class CrashFault(Exception):
    """Injected forced termination; nothing is committed when raised."""


# ------------------------------------------------------------- fixed point
TIERS = {"Q8": 8, "Q12": 12, "Q16": 16}
REF_EXTRA = 8  # reference computation uses frac + REF_EXTRA fractional bits


def to_fixed(x, frac):
    """Deterministic round-half-up to a fixed-point integer."""
    return math.floor(x * (1 << frac) + 0.5)


def half_quantum_ref():
    """Half of one quantization step expressed in reference-bitwidth units."""
    return 1 << (REF_EXTRA - 1)


# ------------------------------------------------------------------ models
@dataclass
class Candidate:
    candidate_id: int
    score: float
    aux: "int | None"
    channel: str


@dataclass
class Request:
    request_id: str
    candidates: list  # list[Candidate]


@dataclass
class Config:
    topk: int
    sample_ratio_permille: int
    quant_tier: str
    norm_method: str
    channel_weights: dict
    output_version: str = "v2"
    batch_budget: int = 100000
    per_request_quota: int = 100000
    threads: int = 1
    tolerance_units: int = 1

    @property
    def frac(self):
        return TIERS[self.quant_tier]


# ------------------------------------------------------------- mini YAML
def _scalar(tok):
    t = tok.strip()
    if t in ("", "~", "null"):
        return None
    if t in ("true", "True"):
        return True
    if t in ("false", "False"):
        return False
    if len(t) >= 2 and t[0] == t[-1] and t[0] in ("'", '"'):
        return t[1:-1]
    try:
        return int(t)
    except ValueError:
        pass
    try:
        return float(t)
    except ValueError:
        pass
    return t


def parse_yaml(text):
    """Parse a small YAML subset: nested maps and scalar lists by indent."""
    lines = []
    for raw in text.splitlines():
        if not raw.strip() or raw.strip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        lines.append((indent, raw.strip()))
    if not lines:
        return {}

    def parse_block(i, indent):
        if lines[i][1].startswith("- "):
            out = []
            while i < len(lines) and lines[i][0] == indent and lines[i][1].startswith("- "):
                out.append(_scalar(lines[i][1][2:]))
                i += 1
            return out, i
        out = {}
        while i < len(lines) and lines[i][0] == indent and not lines[i][1].startswith("- "):
            key, _, rest = lines[i][1].partition(":")
            key, rest = key.strip(), rest.strip()
            i += 1
            if rest:
                out[key] = _scalar(rest)
            elif i < len(lines) and lines[i][0] > indent:
                out[key], i = parse_block(i, lines[i][0])
            else:
                out[key] = None
        return out, i

    obj, _ = parse_block(0, lines[0][0])
    return obj


# ------------------------------------------------------------------ config
def config_from_dict(raw):
    if not isinstance(raw, dict):
        raise BizError(E_BAD_CONFIG, "config must be a mapping")
    topk = raw.get("topk")
    if not isinstance(topk, int) or isinstance(topk, bool) or topk <= 0:
        raise BizError(E_BAD_TOPK, f"topk must be a positive integer, got {topk!r}")
    ratio = raw.get("sample_ratio_permille")
    if not isinstance(ratio, int) or isinstance(ratio, bool) or not (0 <= ratio <= 1000):
        raise BizError(E_INVALID_RATIO, f"sample ratio must be a permille integer in [0,1000], got {ratio!r}")
    tier = raw.get("quant_tier")
    if tier not in TIERS:
        raise BizError(E_TIER_OUT_OF_RANGE, f"quant tier must be one of {sorted(TIERS)}, got {tier!r}")
    method = raw.get("norm_method", "minmax")
    if method not in ("minmax", "zscore"):
        raise BizError(E_BAD_CONFIG, f"norm_method must be minmax|zscore, got {method!r}")
    weights = raw.get("channel_weights")
    if not isinstance(weights, dict) or not weights:
        raise BizError(E_EMPTY_CHANNEL, "channel_weights must be a non-empty mapping")
    for name, w in weights.items():
        if not isinstance(name, str) or not name:
            raise BizError(E_EMPTY_CHANNEL, "channel name must be non-empty")
        if not isinstance(w, (int, float)) or isinstance(w, bool) or not math.isfinite(w):
            raise BizError(E_BAD_CONFIG, f"channel weight for {name!r} must be finite")
    version = raw.get("output_version", "v2")
    if version not in ("v1", "v2"):
        raise BizError(E_BAD_CONFIG, f"output_version must be v1|v2, got {version!r}")
    def _pos_int(key, default):
        v = raw.get(key, default)
        if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
            raise BizError(E_BAD_CONFIG, f"{key} must be a positive integer, got {v!r}")
        return v
    return Config(
        topk=topk,
        sample_ratio_permille=ratio,
        quant_tier=tier,
        norm_method=method,
        channel_weights={k: float(v) for k, v in weights.items()},
        output_version=version,
        batch_budget=_pos_int("batch_budget", 100000),
        per_request_quota=_pos_int("per_request_quota", 100000),
        threads=_pos_int("threads", 1),
        tolerance_units=_pos_int("tolerance_units", 1),
    )


def load_config(path):
    with open(path, "r", encoding="utf-8") as fh:
        return config_from_dict(parse_yaml(fh.read()))


# ---------------------------------------------------------------- requests
def validate_request(raw, cfg):
    rid = raw.get("request_id") if isinstance(raw, dict) else None
    if rid is None or (isinstance(rid, str) and not rid.strip()):
        raise BizError(E_BAD_ID, "request_id missing or empty")
    rid = str(rid)
    cands_raw = raw.get("candidates")
    if not isinstance(cands_raw, list) or not cands_raw:
        raise BizError(E_BAD_ID, "request has no candidates", rid)
    cands, seen = [], set()
    for c in cands_raw:
        if not isinstance(c, dict):
            raise BizError(E_BAD_ID, "candidate must be an object", rid)
        cid = c.get("candidate_id")
        if not isinstance(cid, int) or isinstance(cid, bool) or cid <= 0:
            raise BizError(E_BAD_ID, f"candidate_id must be a positive integer, got {cid!r}", rid)
        if cid in seen:
            raise BizError(E_DUP_CANDIDATE, f"duplicate candidate_id {cid} in request", rid)
        seen.add(cid)
        score = c.get("score")
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score):
            raise BizError(E_BAD_SCORE, f"score must be a finite number, got {score!r}", rid)
        aux = c.get("aux")
        if aux is not None and (not isinstance(aux, int) or isinstance(aux, bool)):
            raise BizError(E_BAD_SCORE, f"aux must be an integer or null, got {aux!r}", rid)
        ch = c.get("channel")
        if not isinstance(ch, str) or not ch:
            raise BizError(E_EMPTY_CHANNEL, "channel must be a non-empty string", rid)
        if ch not in cfg.channel_weights:
            raise BizError(E_UNKNOWN_CHANNEL, f"unknown channel {ch!r}", rid)
        cands.append(Candidate(cid, float(score), aux, ch))
    return Request(rid, cands)


def load_requests(path, cfg):
    """Parse candidate JSONL (one request per line); failures never stop the rest."""
    requests, failures = {}, []
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                failures.append({"line": lineno, "request_id": None,
                                 "code": E_BAD_JSON, "message": str(exc)})
                continue
            rid = str(raw.get("request_id")) if isinstance(raw, dict) else None
            try:
                req = validate_request(raw, cfg)
                if req.request_id in requests:
                    raise BizError(E_DUP_REQUEST, f"duplicate request_id {req.request_id!r}", rid)
                requests[req.request_id] = req
            except BizError as exc:
                failures.append({"line": lineno, "request_id": rid,
                                 "code": exc.code, "message": exc.message})
    return requests, failures


def request_to_dict(req):
    return {
        "request_id": req.request_id,
        "candidates": [
            {"candidate_id": c.candidate_id, "score": c.score, "aux": c.aux, "channel": c.channel}
            for c in sorted(req.candidates, key=lambda c: c.candidate_id)
        ],
    }
