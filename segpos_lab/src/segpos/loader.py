"""加载与校验层。

非法条目不抛出异常，而是写入失败清单（含来源与行号），其余条目照常处理，
从而满足“失败可恢复、顺序可复现”的要求。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .errors import (
    E_COMMAND_INVALID,
    E_CONFIG_INVALID,
    E_LEXICON_INVALID,
    E_PROB_TABLE_INVALID,
    E_SENTENCE_INVALID,
    Failure,
)
from .models import (
    ALLOWED_FIXED_BITS,
    ALLOWED_QUANT_LEVELS,
    Config,
    Lexicon,
    LexiconEntry,
    ProbTable,
    Sentence,
)

_VALID_OOV_RULES = ("equal", "fixed_max")
_VALID_COMMANDS = ("add_entry", "remove_entry", "adjust_freq", "replace_prob_table", "replace_sentence")


def _is_finite_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def load_charset(path: str | Path) -> set[str]:
    text = Path(path).read_text(encoding="utf-8")
    return {ch for ch in text if not ch.isspace()}


def _validate_text(text: str, charset: set[str]) -> str | None:
    if not text:
        return "empty text"
    extra = sorted({ch for ch in text if ch not in charset})
    if extra:
        return "characters outside controlled set: " + "".join(extra)
    return None


def load_sentences(path: str | Path, charset: set[str]) -> tuple[list[Sentence], list[Failure]]:
    failures: list[Failure] = []
    sentences: list[Sentence] = []
    seen: set[str] = set()
    source = Path(path).name
    for line_no, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        parts = raw.split("\t", 1)
        if len(parts) != 2:
            failures.append(Failure(E_SENTENCE_INVALID, f"line{line_no}", "expected 'ID<TAB>text'", source, line_no))
            continue
        sid, text = parts[0].strip(), parts[1]
        key = sid or f"line{line_no}"
        if not sid:
            failures.append(Failure(E_SENTENCE_INVALID, key, "empty sentence id", source, line_no))
            continue
        if sid in seen:
            failures.append(Failure(E_SENTENCE_INVALID, sid, "duplicate sentence id", source, line_no))
            continue
        reason = _validate_text(text, charset)
        if reason is not None:
            failures.append(Failure(E_SENTENCE_INVALID, sid, reason, source, line_no))
            continue
        seen.add(sid)
        sentences.append(Sentence(sid=sid, text=text, source_line=line_no))
    return sentences, failures


def load_lexicon(path: str | Path, known_tags: set[str], source: str | None = None) -> tuple[Lexicon, list[Failure]]:
    entries: dict[str, LexiconEntry] = {}
    failures: list[Failure] = []
    src = source or Path(path).name
    for line_no, raw in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        parts = raw.split()
        if len(parts) != 3:
            failures.append(Failure(E_LEXICON_INVALID, f"line{line_no}", "expected 'word freq tag'", src, line_no))
            continue
        word, freq_raw, tag = parts
        if not word:
            failures.append(Failure(E_LEXICON_INVALID, f"line{line_no}", "empty word form", src, line_no))
            continue
        try:
            freq = float(freq_raw)
        except ValueError:
            freq = float("nan")
        if not _is_finite_number(freq) or freq <= 0:
            failures.append(Failure(E_LEXICON_INVALID, word, f"freq must be positive finite, got {freq_raw!r}", src, line_no))
            continue
        if tag not in known_tags:
            failures.append(Failure(E_LEXICON_INVALID, word, f"tag {tag!r} not declared in prob table", src, line_no))
            continue
        if word in entries:
            entries[word].freq += freq
            entries[word].sources.append(f"{src}:{line_no}")
        else:
            entries[word] = LexiconEntry(word=word, freq=freq, tag=tag, sources=[f"{src}:{line_no}"])
    return Lexicon(entries=entries), failures


def load_prob_table(path: str | Path, smoothing_hint: float | None = None) -> tuple[ProbTable | None, list[Failure]]:
    source = Path(path).name
    failures: list[Failure] = []
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return None, [Failure(E_PROB_TABLE_INVALID, source, f"unparseable json: {exc}", source, 0)]

    tags = data.get("tags")
    if not isinstance(tags, list) or not tags or any(not isinstance(t, str) for t in tags):
        failures.append(Failure(E_PROB_TABLE_INVALID, "tags", "tags must be a non-empty string list", source, 0))
        return None, failures
    tag_set = set(tags)
    if len(tag_set) != len(tags):
        failures.append(Failure(E_PROB_TABLE_INVALID, "tags", "duplicate tag declaration", source, 0))
        return None, failures

    start_logp, n1 = _validate_log_matrix(data.get("start_logp", {}), tag_set, source, "start")
    failures.extend(n1)
    transition_logp, n2 = _validate_nested_matrix(data.get("transition_logp", {}), tag_set, source, "transition")
    failures.extend(n2)
    emission_logp, n3 = _validate_nested_matrix(data.get("emission_logp", {}), tag_set, source, "emission")
    failures.extend(n3)

    smoothing = data.get("smoothing", smoothing_hint)
    if not _is_finite_number(smoothing) or float(smoothing) <= 0:
        failures.append(Failure(E_PROB_TABLE_INVALID, "smoothing", "smoothing must be positive finite", source, 0))
        return None, failures
    table = ProbTable(
        tags=sorted(tags),
        start_logp=start_logp,
        transition_logp=transition_logp,
        emission_logp=emission_logp,
        smoothing=float(smoothing),
    )
    return table, failures


def _validate_log_matrix(matrix, tag_set: set[str], source: str, name: str) -> tuple[dict, list[Failure]]:
    out: dict[str, float] = {}
    failures: list[Failure] = []
    if not isinstance(matrix, dict):
        return out, [Failure(E_PROB_TABLE_INVALID, name, "matrix must be an object", source, 0)]
    for key, value in matrix.items():
        if key not in tag_set:
            failures.append(Failure(E_PROB_TABLE_INVALID, f"{name}.{key}", "undeclared tag", source, 0))
            continue
        if not _is_finite_number(value):
            failures.append(Failure(E_PROB_TABLE_INVALID, f"{name}.{key}", "log prob must be finite", source, 0))
            continue
        out[key] = float(value)
    return out, failures


def _validate_nested_matrix(matrix, tag_set: set[str], source: str, name: str) -> tuple[dict, list[Failure]]:
    out: dict[str, dict[str, float]] = {}
    failures: list[Failure] = []
    if not isinstance(matrix, dict):
        return out, [Failure(E_PROB_TABLE_INVALID, name, "matrix must be an object", source, 0)]
    for outer, inner in matrix.items():
        if not isinstance(inner, dict):
            failures.append(Failure(E_PROB_TABLE_INVALID, f"{name}.{outer}", "inner matrix must be an object", source, 0))
            continue
        row: dict[str, float] = {}
        for tag, value in inner.items():
            if tag not in tag_set:
                failures.append(Failure(E_PROB_TABLE_INVALID, f"{name}.{outer}.{tag}", "undeclared tag", source, 0))
                continue
            if not _is_finite_number(value):
                failures.append(Failure(E_PROB_TABLE_INVALID, f"{name}.{outer}.{tag}", "log prob must be finite", source, 0))
                continue
            row[tag] = float(value)
        if row:
            out[outer] = row
    return out, failures


def load_config(path: str | Path) -> tuple[Config | None, list[Failure]]:
    source = Path(path).name
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return None, [Failure(E_CONFIG_INVALID, source, f"unparseable json: {exc}", source, 0)]

    failures: list[Failure] = []
    required = {
        "concurrency": int,
        "first_response_budget_ms": (int, float),
        "snapshot_commit_interval_commands": int,
        "low_confidence_threshold": (int, float),
        "fixed_point_bits": int,
        "quantization_levels": int,
        "smoothing": (int, float),
        "oov_rule": str,
        "oov_max_length": int,
    }
    for key, kind in required.items():
        if key not in data:
            failures.append(Failure(E_CONFIG_INVALID, key, "missing field", source, 0))
    if failures:
        return None, failures

    concurrency = data["concurrency"]
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency <= 0 or concurrency > 16:
        failures.append(Failure(E_CONFIG_INVALID, "concurrency", "must be integer in [1, 16]", source, 0))
    budget = data["first_response_budget_ms"]
    if not _is_finite_number(budget) or budget < 0:
        failures.append(Failure(E_CONFIG_INVALID, "first_response_budget_ms", "must be non-negative finite number", source, 0))
    interval = data["snapshot_commit_interval_commands"]
    if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
        failures.append(Failure(E_CONFIG_INVALID, "snapshot_commit_interval_commands", "must be positive integer", source, 0))
    threshold = data["low_confidence_threshold"]
    if not _is_finite_number(threshold) or not 0.0 <= float(threshold) <= 1.0:
        failures.append(Failure(E_CONFIG_INVALID, "low_confidence_threshold", "must be within [0, 1]", source, 0))
    bits = data["fixed_point_bits"]
    if bits not in ALLOWED_FIXED_BITS:
        failures.append(Failure(E_CONFIG_INVALID, "fixed_point_bits", f"allowed: {ALLOWED_FIXED_BITS}", source, 0))
    levels = data["quantization_levels"]
    if levels not in ALLOWED_QUANT_LEVELS:
        failures.append(Failure(E_CONFIG_INVALID, "quantization_levels", f"allowed: {ALLOWED_QUANT_LEVELS}", source, 0))
    smoothing = data["smoothing"]
    if not _is_finite_number(smoothing) or float(smoothing) <= 0:
        failures.append(Failure(E_CONFIG_INVALID, "smoothing", "must be positive finite", source, 0))
    oov_rule = data["oov_rule"]
    if oov_rule not in _VALID_OOV_RULES:
        failures.append(Failure(E_CONFIG_INVALID, "oov_rule", f"allowed: {_VALID_OOV_RULES}", source, 0))
    max_len = data["oov_max_length"]
    if not isinstance(max_len, int) or isinstance(max_len, bool) or max_len <= 0:
        failures.append(Failure(E_CONFIG_INVALID, "oov_max_length", "must be positive integer", source, 0))
    if failures:
        return None, failures

    return Config(
        concurrency=concurrency,
        first_response_budget_ms=float(budget),
        snapshot_commit_interval_commands=interval,
        low_confidence_threshold=float(threshold),
        fixed_point_bits=bits,
        quantization_levels=levels,
        smoothing=float(smoothing),
        oov_rule=oov_rule,
        oov_max_length=max_len,
        simulated_task_delay_ms=float(data.get("simulated_task_delay_ms", 0.0)),
    ), []


def load_commands(path: str | Path) -> tuple[list[dict], list[Failure]]:
    source = Path(path).name
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return [], [Failure(E_COMMAND_INVALID, source, f"unparseable json: {exc}", source, 0)]
    if not isinstance(data, list):
        return [], [Failure(E_COMMAND_INVALID, source, "command stream must be a list", source, 0)]

    commands: list[dict] = []
    failures: list[Failure] = []
    for idx, cmd in enumerate(data, start=1):
        if not isinstance(cmd, dict) or cmd.get("type") not in _VALID_COMMANDS:
            failures.append(Failure(E_COMMAND_INVALID, f"cmd{idx}", f"unknown command shape: {cmd!r}", source, idx))
            continue
        commands.append(dict(cmd, _index=idx))
    return commands, failures


def validate_sentence_text(text: str, charset: set[str]) -> str | None:
    return _validate_text(text, charset)
