"""种子、词表、配置、打分参数的加载与逐项校验层。

任何单条非法条目都转换为 Failure 进入失败清单，不中断其余条目加载。
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

from .errors import (
    E_CONFIG_INVALID,
    E_PARAMS_INVALID,
    E_SEED_INVALID,
    E_VOCAB_INVALID,
    Failure,
)
from .models import (
    BuildConfig,
    DimensionSpec,
    ScoreParams,
    Seed,
    SlotSource,
    Vocab,
)
from .schemas import (
    CONFIG_SCHEMA,
    SEED_SCHEMA,
    SCORE_PARAMS_SCHEMA,
    SLOT_SOURCE_SCHEMA,
    VOCAB_SCHEMA,
    schema_errors,
)
from .utils import is_nonneg_finite_number

DIMENSIONS = ("clarity", "correctness", "format", "length")


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", newline="\n") as handle:
        return handle.read()


def _read_json(path: str) -> Any:
    return json.loads(_read_text(path))


def _read_jsonl(path: str) -> list[tuple[int, str, Any]]:
    rows: list[tuple[int, str, Any]] = []
    for line_no, raw in enumerate(_read_text(path).splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            rows.append((line_no, stripped, json.loads(stripped)))
        except json.JSONDecodeError as exc:
            rows.append((line_no, stripped, {"__parse_error__": str(exc)}))
    return rows


def load_seeds(path: str, failures: list[Failure]) -> dict[str, Seed]:
    seeds: dict[str, Seed] = {}
    seen_ids: set[str] = set()
    for line_no, _raw, obj in _read_jsonl(path):
        ref = f"{os.path.basename(path)}:L{line_no}"
        if not isinstance(obj, dict) or "__parse_error__" in obj:
            message = obj.get("__parse_error__", "entry is not a JSON object") \
                if isinstance(obj, dict) else "entry is not a JSON object"
            failures.append(Failure(E_SEED_INVALID, path, ref, message))
            continue
        schema_problems = schema_errors(SEED_SCHEMA, obj)
        if schema_problems:
            failures.append(
                Failure(E_SEED_INVALID, path, ref, "; ".join(schema_problems))
            )
            continue
        problems = _validate_seed_shape(obj)
        if problems:
            failures.append(Failure(E_SEED_INVALID, path, ref, "; ".join(problems)))
            continue
        seed_id = str(obj["seed_id"])
        if seed_id in seen_ids:
            failures.append(
                Failure(E_SEED_INVALID, path, ref, f"duplicate seed_id {seed_id}")
            )
            continue
        seen_ids.add(seed_id)
        seeds[seed_id] = Seed(
            seed_id=seed_id,
            task_type=str(obj["task_type"]),
            template_id=str(obj["template_id"]),
            template=str(obj["template"]),
            slots=tuple(str(name) for name in obj.get("slots", [])),
            answer_template=str(obj["answer_template"]),
            line_no=line_no,
        )
    return seeds


def _validate_seed_shape(obj: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    for key in ("seed_id", "task_type", "template_id", "template", "answer_template"):
        if key not in obj or not isinstance(obj[key], str) or not obj[key].strip():
            problems.append(f"{key} missing or not a non-empty string")
    slots = obj.get("slots", [])
    if not isinstance(slots, list) or any(not isinstance(name, str) for name in slots):
        problems.append("slots must be a list of strings")
        return problems
    if len(set(slots)) != len(slots):
        problems.append("slot names must be unique")
    template = obj.get("template", "")
    answer = obj.get("answer_template", "")
    for name in slots:
        if "{" + name + "}" not in template:
            problems.append(f"slot {name} not present in template")
        if "{" + name + "}" not in answer:
            problems.append(f"slot {name} not present in answer_template")
    return problems


def load_slot_sources(path: str, failures: list[Failure]) -> dict[str, SlotSource]:
    sources: dict[str, SlotSource] = {}
    seen_ids: set[str] = set()
    for line_no, _raw, obj in _read_jsonl(path):
        ref = f"{os.path.basename(path)}:L{line_no}"
        if not isinstance(obj, dict) or "__parse_error__" in obj:
            message = obj.get("__parse_error__", "entry is not a JSON object") \
                if isinstance(obj, dict) else "entry is not a JSON object"
            failures.append(Failure(E_SEED_INVALID, path, ref, message))
            continue
        schema_problems = schema_errors(SLOT_SOURCE_SCHEMA, obj)
        if schema_problems:
            failures.append(
                Failure(E_SEED_INVALID, path, ref, "; ".join(schema_problems))
            )
            continue
        problems: list[str] = []
        for key in ("source_id", "task_type", "slot_name"):
            if not isinstance(obj.get(key), str) or not obj[key].strip():
                problems.append(f"{key} missing or not a non-empty string")
        values = obj.get("values")
        if not isinstance(values, list) or not values or any(
            not isinstance(value, str) for value in values
        ):
            problems.append("values must be a non-empty list of strings")
        if problems:
            failures.append(Failure(E_SEED_INVALID, path, ref, "; ".join(problems)))
            continue
        source_id = str(obj["source_id"])
        if source_id in seen_ids:
            failures.append(
                Failure(E_SEED_INVALID, path, ref,
                        f"duplicate source_id {source_id}")
            )
            continue
        seen_ids.add(source_id)
        sources[source_id] = SlotSource(
            source_id=source_id,
            task_type=str(obj["task_type"]),
            slot_name=str(obj["slot_name"]),
            values=tuple(str(value) for value in values),
        )
    return sources


def load_vocab(path: str, failures: list[Failure]) -> Vocab:
    try:
        obj = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(Failure(E_VOCAB_INVALID, path, os.path.basename(path), str(exc)))
        return Vocab(tokens=(), rules={})
    schema_problems = schema_errors(VOCAB_SCHEMA, obj)
    if schema_problems:
        failures.append(
            Failure(E_VOCAB_INVALID, path, os.path.basename(path),
                    "; ".join(schema_problems))
        )
        return Vocab(tokens=(), rules={})
    problems: list[str] = []
    tokens = obj.get("tokens")
    if not isinstance(tokens, list) or not tokens or any(
        not isinstance(token, str) or not token for token in tokens
    ):
        problems.append("tokens must be a non-empty list of non-empty strings")
    elif len(set(tokens)) != len(tokens):
        problems.append("tokens must be unique")
    rules = obj.get("rules", {})
    if not isinstance(rules, dict):
        problems.append("rules must be an object")
    else:
        if rules.get("match", "longest_prefix") != "longest_prefix":
            problems.append("rules.match must be longest_prefix")
        if rules.get("unknown", "char_fallback") != "char_fallback":
            problems.append("rules.unknown must be char_fallback")
    if problems:
        failures.append(
            Failure(E_VOCAB_INVALID, path, os.path.basename(path), "; ".join(problems))
        )
        return Vocab(tokens=(), rules={})
    ordered = sorted(tokens, key=lambda token: (-len(token), token))
    return Vocab(
        tokens=tuple(ordered),
        rules={"match": "longest_prefix", "unknown": "char_fallback", **rules},
    )


def load_config(path: str, failures: list[Failure]) -> BuildConfig | None:
    try:
        obj = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(
            Failure(E_CONFIG_INVALID, path, os.path.basename(path), str(exc))
        )
        return None
    schema_problems = schema_errors(CONFIG_SCHEMA, obj)
    if schema_problems:
        for problem in schema_problems:
            failures.append(
                Failure(E_CONFIG_INVALID, path, os.path.basename(path), problem)
            )
        return None
    problems = _validate_config(obj)
    if problems:
        for problem in problems:
            failures.append(
                Failure(E_CONFIG_INVALID, path, os.path.basename(path), problem)
            )
        return None
    task_types = tuple(sorted(str(name) for name in obj["task_types"]))
    return BuildConfig(
        task_types=task_types,
        target_counts={task: int(obj["target_counts"][task]) for task in task_types},
        template_weights={
            task: {
                str(tid): float(weight)
                for tid, weight in sorted(obj["template_weights"][task].items())
            }
            for task in task_types
        },
        slot_weights={
            task: {
                str(sid): float(weight)
                for sid, weight in sorted(obj["slot_weights"][task].items())
            }
            for task in task_types
        },
        dedup_threshold=float(obj["dedup_threshold"]),
        boundary_rule=str(obj.get("boundary_rule", "merge")),
        max_workers=int(obj["max_workers"]),
        snapshot_interval_batches=int(obj["snapshot_interval_batches"]),
    )


def _validate_config(obj: Any) -> list[str]:
    problems: list[str] = []
    if not isinstance(obj, dict):
        return ["config root must be an object"]
    task_types = obj.get("task_types")
    if not isinstance(task_types, list) or not task_types or any(
        not isinstance(name, str) for name in task_types
    ):
        problems.append("task_types must be a non-empty list of strings")
        return problems
    if len(set(task_types)) != len(task_types):
        problems.append("task_types must be unique")
    targets = obj.get("target_counts", {})
    if not isinstance(targets, dict):
        problems.append("target_counts must be an object")
    else:
        for task in task_types:
            value = targets.get(task)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                problems.append(
                    f"target_counts[{task}] must be a non-negative integer"
                )
    for section in ("template_weights", "slot_weights"):
        weights = obj.get(section)
        if not isinstance(weights, dict):
            problems.append(f"{section} must be an object")
            continue
        for task in task_types:
            table = weights.get(task)
            if not isinstance(table, dict) or not table:
                problems.append(f"{section}[{task}] must be a non-empty object")
                continue
            if any(not is_nonneg_finite_number(weight) for weight in table.values()):
                problems.append(
                    f"{section}[{task}] weights must be non-negative finite"
                )
                continue
            if sum(float(weight) for weight in table.values()) <= 0.0:
                problems.append(f"{section}[{task}] weights must sum to > 0")
    threshold = obj.get("dedup_threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool) or not (
        0.0 <= float(threshold) <= 1.0
    ):
        problems.append("dedup_threshold must be a finite number in [0.0, 1.0]")
    if obj.get("boundary_rule", "merge") not in ("merge", "split"):
        problems.append("boundary_rule must be merge or split")
    workers = obj.get("max_workers")
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        problems.append("max_workers must be a positive integer")
    interval = obj.get("snapshot_interval_batches")
    if not isinstance(interval, int) or isinstance(interval, bool) or interval < 1:
        problems.append("snapshot_interval_batches must be a positive integer")
    return problems


def load_score_params(path: str, failures: list[Failure]) -> ScoreParams | None:
    try:
        obj = _read_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(
            Failure(E_PARAMS_INVALID, path, os.path.basename(path), str(exc))
        )
        return None
    schema_problems = schema_errors(SCORE_PARAMS_SCHEMA, obj)
    if schema_problems:
        for problem in schema_problems:
            failures.append(
                Failure(E_PARAMS_INVALID, path, os.path.basename(path), problem)
            )
        return None
    problems: list[str] = []
    dims = obj.get("dimensions", list(DIMENSIONS))
    if dims != list(DIMENSIONS):
        problems.append(f"dimensions must be exactly {list(DIMENSIONS)}")
    specs_raw = obj.get("specs", {})
    specs: dict[str, DimensionSpec] = {}
    if not isinstance(specs_raw, dict):
        problems.append("specs must be an object")
    else:
        for dim in DIMENSIONS:
            spec = specs_raw.get(dim)
            if not isinstance(spec, dict):
                problems.append(f"specs.{dim} missing")
                continue
            weight = spec.get("weight")
            baseline = spec.get("baseline")
            if not is_nonneg_finite_number(weight):
                problems.append(f"specs.{dim}.weight must be non-negative finite")
            if not isinstance(baseline, (int, float)) or isinstance(baseline, bool) \
                    or not math.isfinite(float(baseline)):
                problems.append(f"specs.{dim}.baseline must be finite")
            else:
                specs[dim] = DimensionSpec(float(weight or 0.0), float(baseline))
        if all(
            isinstance(specs_raw.get(dim), dict)
            and is_nonneg_finite_number(specs_raw[dim].get("weight"))
            for dim in DIMENSIONS
        ) and sum(float(specs_raw[dim]["weight"]) for dim in DIMENSIONS) <= 0.0:
            problems.append("dimension weights must sum to > 0")
    lower = obj.get("length_lower")
    upper = obj.get("length_upper")
    if not isinstance(lower, int) or isinstance(lower, bool) or lower < 0:
        problems.append("length_lower must be a non-negative integer")
    if not isinstance(upper, int) or isinstance(upper, bool) or upper <= 0:
        problems.append("length_upper must be a positive integer")
    if isinstance(lower, int) and isinstance(upper, int) and lower > upper:
        problems.append("length_lower must be <= length_upper")
    if problems:
        for problem in problems:
            failures.append(
                Failure(E_PARAMS_INVALID, path, os.path.basename(path), problem)
            )
        return None
    return ScoreParams(
        dimensions=DIMENSIONS,
        specs=specs,
        length_lower=int(lower),
        length_upper=int(upper),
    )
