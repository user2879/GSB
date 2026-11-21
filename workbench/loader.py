"""Loading and validation layer (requirement 1).

Every file is parsed line by line / entry by entry.  Illegal entries emit an
E_LOAD_* code, are appended to the failure list (in a fixed order) and are
skipped; all remaining entries continue to be processed.

Failure list ordering is fixed: turns file (line order) -> acoustics file
(line order) -> tools file (entry order) -> config -> cross checks.
"""

from __future__ import annotations

import json
import math

import yaml

from . import codes

ALLOWED_SCHEMA_TYPES = {"number", "integer", "string"}


def _failure(code, source, line, detail, injection_id=None):
    entry = {"code": code, "source": source, "line": line, "detail": detail}
    if injection_id is not None:
        entry["injection_id"] = injection_id
    return entry


def _is_finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


# ---------------------------------------------------------------------------
# Turns JSONL
# ---------------------------------------------------------------------------

def load_turns(path, vocab, tool_schemas):
    """Parse the turns JSONL file.

    Returns (turns, failures).  Each turn: turn_id, speaker, terms,
    start_ms, end_ms, sentiment, tool_calls.
    """
    turns = []
    failures = []
    seen_ids = set()
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                failures.append(_failure(codes.E_LOAD_JSON, "turns", lineno, str(exc)))
                continue
            ok = True
            required = ("turn_id", "speaker", "terms", "start_ms", "end_ms", "sentiment")
            if not isinstance(rec, dict) or any(k not in rec for k in required):
                failures.append(_failure(codes.E_LOAD_TURN_SCHEMA, "turns", lineno,
                                         "missing required field"))
                continue
            turn_id = rec["turn_id"]
            if not isinstance(turn_id, str) or not turn_id:
                failures.append(_failure(codes.E_LOAD_TURN_SCHEMA, "turns", lineno,
                                         "turn_id must be a non-empty string"))
                continue
            if turn_id in seen_ids:
                failures.append(_failure(codes.E_LOAD_TURN_DUP, "turns", lineno,
                                         f"duplicate turn_id {turn_id}"))
                continue
            if not isinstance(rec["terms"], list) or not rec["terms"]:
                failures.append(_failure(codes.E_LOAD_TURN_SCHEMA, "turns", lineno,
                                         "terms must be a non-empty list"))
                continue
            bad_terms = [t for t in rec["terms"] if t not in vocab]
            if bad_terms:
                failures.append(_failure(codes.E_LOAD_VOCAB, "turns", lineno,
                                         f"terms outside controlled vocabulary: {bad_terms}"))
                ok = False
            if not (_is_finite_number(rec["start_ms"]) and _is_finite_number(rec["end_ms"])
                    and rec["end_ms"] > rec["start_ms"]):
                failures.append(_failure(codes.E_LOAD_TURN_SCHEMA, "turns", lineno,
                                         "start_ms/end_ms must be finite and end > start"))
                ok = False
            tool_calls = rec.get("tool_calls", [])
            if not isinstance(tool_calls, list):
                failures.append(_failure(codes.E_LOAD_TURN_SCHEMA, "turns", lineno,
                                         "tool_calls must be a list"))
                ok = False
                tool_calls = []
            for call in tool_calls:
                name = call.get("tool") if isinstance(call, dict) else None
                params = call.get("params", {}) if isinstance(call, dict) else {}
                if name not in tool_schemas:
                    failures.append(_failure(codes.E_LOAD_TOOL_PARAMS, "turns", lineno,
                                             f"unknown tool {name!r}"))
                    ok = False
                    continue
                err = _check_params(tool_schemas[name], params)
                if err:
                    failures.append(_failure(codes.E_LOAD_TOOL_PARAMS, "turns", lineno,
                                             f"tool {name}: {err}"))
                    ok = False
            if not ok:
                seen_ids.add(turn_id)  # still reserve the id to keep ordering stable
                continue
            seen_ids.add(turn_id)
            turns.append({
                "turn_id": turn_id,
                "speaker": rec["speaker"],
                "terms": list(rec["terms"]),
                "start_ms": rec["start_ms"],
                "end_ms": rec["end_ms"],
                "sentiment": rec["sentiment"],
                "tool_calls": tool_calls,
            })
    return turns, failures


def _check_params(schema, params):
    """Validate params against the declared parameter schema."""
    if not isinstance(params, dict):
        return "params must be an object"
    for key, kind in schema.items():
        if key not in params:
            return f"missing param {key!r}"
        value = params[key]
        if kind == "number" and not _is_finite_number(value):
            return f"param {key!r} must be a finite number"
        if kind == "integer" and not (isinstance(value, int) and not isinstance(value, bool)):
            return f"param {key!r} must be an integer"
        if kind == "string" and not isinstance(value, str):
            return f"param {key!r} must be a string"
    extra = [k for k in params if k not in schema]
    if extra:
        return f"unexpected params {extra}"
    return None


# ---------------------------------------------------------------------------
# Acoustic evidence JSONL
# ---------------------------------------------------------------------------

def load_acoustics(path):
    """Parse the acoustic evidence JSONL.  Returns (dict turn_id -> record, failures)."""
    acoustics = {}
    failures = []
    with open(path, "r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                rec = json.loads(raw)
            except json.JSONDecodeError as exc:
                failures.append(_failure(codes.E_LOAD_JSON, "acoustics", lineno, str(exc)))
                continue
            if not isinstance(rec, dict) or "turn_id" not in rec:
                failures.append(_failure(codes.E_LOAD_ACOUSTIC_SCHEMA, "acoustics", lineno,
                                         "missing turn_id"))
                continue
            turn_id = rec["turn_id"]
            numeric_fields = ("pitch_mean", "pitch_var")
            bad = [f for f in numeric_fields if not _is_finite_number(rec.get(f))]
            quantiles = rec.get("energy_quantiles")
            if not isinstance(quantiles, dict) or not quantiles:
                bad.append("energy_quantiles")
            else:
                for qk, qv in quantiles.items():
                    if not _is_finite_number(qv):
                        bad.append(f"energy_quantiles.{qk}")
            if bad:
                code = codes.E_LOAD_ACOUSTIC_NONFINITE
                failures.append(_failure(code, "acoustics", lineno,
                                         f"non-finite or missing acoustic statistics: {bad}",
                                         injection_id=rec.get("_injection_id")))
                continue
            acoustics[turn_id] = {
                "turn_id": turn_id,
                "pitch_mean": rec["pitch_mean"],
                "pitch_var": rec["pitch_var"],
                "energy_quantiles": dict(quantiles),
            }
    return acoustics, failures


# ---------------------------------------------------------------------------
# Tool manifest JSON
# ---------------------------------------------------------------------------

def load_tools(path):
    """Parse the tool manifest JSON.  Returns (dict name -> tool, failures)."""
    failures = []
    tools = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            doc = json.load(handle)
    except (json.JSONDecodeError, OSError) as exc:
        return {}, [_failure(codes.E_LOAD_JSON, "tools", 0, str(exc))]
    entries = doc.get("tools") if isinstance(doc, dict) else None
    if not isinstance(entries, list):
        return {}, [_failure(codes.E_LOAD_TOOL_SCHEMA, "tools", 0, "manifest must contain a tools list")]
    for idx, entry in enumerate(entries):
        line = idx + 1
        if not isinstance(entry, dict):
            failures.append(_failure(codes.E_LOAD_TOOL_SCHEMA, "tools", line, "entry must be an object"))
            continue
        name = entry.get("name")
        schema = entry.get("param_schema")
        formula = entry.get("formula")
        dimension = entry.get("dimension")
        if not isinstance(name, str) or not name:
            failures.append(_failure(codes.E_LOAD_TOOL_SCHEMA, "tools", line, "missing tool name"))
            continue
        if not isinstance(schema, dict) or any(v not in ALLOWED_SCHEMA_TYPES for v in schema.values()):
            failures.append(_failure(codes.E_LOAD_TOOL_SCHEMA, "tools", line,
                                     f"tool {name}: invalid param_schema"))
            continue
        if not isinstance(formula, str) or not formula:
            failures.append(_failure(codes.E_LOAD_TOOL_SCHEMA, "tools", line,
                                     f"tool {name}: missing formula"))
            continue
        if not isinstance(dimension, str) or not dimension:
            failures.append(_failure(codes.E_LOAD_TOOL_SCHEMA, "tools", line,
                                     f"tool {name}: missing dimension"))
            continue
        tools[name] = {
            "name": name,
            "param_schema": dict(schema),
            "formula": formula,
            "dimension": dimension,
        }
    return tools, failures


# ---------------------------------------------------------------------------
# Fusion config YAML
# ---------------------------------------------------------------------------

DEFAULT_ALLOWED_FRAC_BITS = (4, 8, 12, 16, 20, 24)
DEFAULT_ALLOWED_QUANTUM = (1, 2, 4, 8)


def load_config(path):
    """Parse and validate the fusion config YAML.  Returns (config, failures)."""
    try:
        with open(path, "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)
    except (yaml.YAMLError, OSError) as exc:
        return None, [_failure(codes.E_LOAD_CONFIG, "config", 0, str(exc))]
    failures = validate_config(cfg)
    if failures:
        return None, failures
    return cfg, failures


def validate_config(cfg):
    """Validate a config mapping; reused by the injection layer."""
    failures = []
    if not isinstance(cfg, dict):
        return [_failure(codes.E_LOAD_CONFIG, "config", 0, "config must be a mapping")]

    weights = cfg.get("weights")
    if not isinstance(weights, dict) or sorted(weights) != ["acoustic", "text", "tool"]:
        failures.append(_failure(codes.E_LOAD_CONFIG, "config", 0,
                                 "weights must define text/acoustic/tool"))
        weights = None
    elif not all(_is_finite_number(w) for w in weights.values()):
        failures.append(_failure(codes.E_LOAD_CONFIG, "config", 0,
                                 "weights must be finite numbers"))
        weights = None

    for key in ("temperature",):
        if not _is_finite_number(cfg.get(key)) or cfg.get(key) <= 0:
            failures.append(_failure(codes.E_LOAD_CONFIG, "config", 0,
                                     f"{key} must be a positive finite number"))

    precision = cfg.get("precision")
    if not isinstance(precision, dict):
        failures.append(_failure(codes.E_LOAD_PRECISION, "config", 0, "missing precision section"))
        precision = None
    else:
        allowed_bits = tuple(precision.get("allowed_frac_bits", DEFAULT_ALLOWED_FRAC_BITS))
        allowed_quantum = tuple(precision.get("allowed_quantum", DEFAULT_ALLOWED_QUANTUM))
        if precision.get("frac_bits") not in allowed_bits:
            failures.append(_failure(codes.E_LOAD_PRECISION, "config", 0,
                                     f"frac_bits {precision.get('frac_bits')} not in {allowed_bits}",
                                     injection_id=precision.get("_injection_id")))
        if precision.get("quantum") not in allowed_quantum:
            failures.append(_failure(codes.E_LOAD_PRECISION, "config", 0,
                                     f"quantum {precision.get('quantum')} not in {allowed_quantum}",
                                     injection_id=precision.get("_injection_id")))

    tolerances = cfg.get("tolerances")
    if not isinstance(tolerances, dict) or not all(
            _is_finite_number(tolerances.get(k)) and tolerances.get(k) > 0
            for k in ("abs", "rel")):
        failures.append(_failure(codes.E_LOAD_TOL, "config", 0,
                                 "tolerances.abs and tolerances.rel must be positive finite numbers"))

    for key in ("fusion_point_budget", "interactive_latency_ms", "concurrency"):
        value = cfg.get(key)
        if not (isinstance(value, int) and not isinstance(value, bool) and value > 0):
            failures.append(_failure(codes.E_LOAD_BUDGET, "config", 0,
                                     f"{key} must be a positive integer"))

    interval = cfg.get("confidence_interval")
    if not (isinstance(interval, list) and len(interval) == 2
            and all(_is_finite_number(v) for v in interval) and interval[0] < interval[1]):
        failures.append(_failure(codes.E_LOAD_CONFIG, "config", 0,
                                 "confidence_interval must be [lo, hi] with lo < hi"))

    return failures


# ---------------------------------------------------------------------------
# Cross checks
# ---------------------------------------------------------------------------

def cross_check(turns, acoustics):
    """Verify the 1:1 correspondence between turns and acoustic rows."""
    failures = []
    turn_ids = [t["turn_id"] for t in turns]
    for turn_id in turn_ids:
        if turn_id not in acoustics:
            failures.append(_failure(codes.E_LOAD_ACOUSTIC_MISSING, "cross", 0,
                                     f"no acoustic evidence for turn {turn_id}"))
    turn_id_set = set(turn_ids)
    for turn_id in sorted(acoustics):
        if turn_id not in turn_id_set:
            failures.append(_failure(codes.E_LOAD_ACOUSTIC_ORPHAN, "cross", 0,
                                     f"acoustic evidence without turn {turn_id}"))
    return failures


def load_vocabulary(path):
    with open(path, "r", encoding="utf-8") as handle:
        doc = json.load(handle)
    return set(doc["terms"]), doc.get("sentiment_lexicon", {})
