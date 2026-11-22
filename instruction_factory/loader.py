from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .constants import (
    ALLOWED_QUANTIZATION_STEPS,
    DEFAULT_INPUT_FILES,
    DIMENSION_ORDER,
    E_CONFIG_INVALID,
    E_SEED_INVALID,
    E_WEIGHT_INVALID,
    HARD_MAX_CONCURRENCY,
    MAX_DECIMAL_PLACES,
    MIN_DECIMAL_PLACES,
)
from .errors import FailureCollector
from .util import (
    as_decimal,
    finite_number,
    fingerprint_file,
    fingerprint_object,
    is_safe_id,
    is_number,
    parse_json_text,
)

PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


@dataclass(frozen=True)
class InputPaths:
    input_dir: Path
    seeds: Path
    lexicon: Path
    scoring: Path
    config: Path
    plan: Path

    @classmethod
    def from_dir(cls, input_dir: str | Path, overrides: dict[str, str | Path] | None = None) -> "InputPaths":
        root = Path(input_dir).resolve()
        overrides = overrides or {}
        values = {
            key: Path(overrides[key]).resolve() if key in overrides else (root / filename).resolve()
            for key, filename in DEFAULT_INPUT_FILES.items()
        }
        return cls(input_dir=root, **values)

    def as_dict(self) -> dict[str, str]:
        return {
            "seeds": str(self.seeds),
            "lexicon": str(self.lexicon),
            "scoring": str(self.scoring),
            "config": str(self.config),
            "plan": str(self.plan),
        }


@dataclass
class LoadedInputs:
    paths: InputPaths
    seeds_by_id: dict[str, dict[str, Any]]
    seeds_by_variant: dict[str, dict[str, dict[str, Any]]]
    seed_lines: dict[str, int]
    plan_templates: dict[str, dict[str, dict[str, Any]]]
    lexicon: dict[str, Any]
    scoring: dict[str, Any]
    config: dict[str, Any]
    plan: dict[str, Any]
    fingerprints: dict[str, str]
    failures: FailureCollector = field(default_factory=FailureCollector)

    @property
    def sources(self) -> dict[str, list[str]]:
        return self.lexicon["sources"]

    @property
    def tokenizer_rules(self) -> dict[str, Any]:
        return self.lexicon["subword"]


def _is_positive_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _is_nonnegative_number(value: Any) -> bool:
    return finite_number(value) and as_decimal(value) >= 0


def _load_json_document(
    path: Path,
    *,
    field_name: str,
    failures: FailureCollector,
    fatal: bool = True,
) -> Any | None:
    if not path.exists():
        failures.add(
            E_CONFIG_INVALID,
            f"required file does not exist: {path}",
            stage="loading",
            file=str(path),
            field=field_name,
            path=str(path),
            fatal=fatal,
        )
        return None
    try:
        return parse_json_text(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        line = getattr(exc, "lineno", None)
        column = getattr(exc, "colno", None)
        where = f"line {line}, column {column}" if line is not None else "document"
        failures.add(
            E_CONFIG_INVALID,
            f"invalid JSON in {path.name} at {where}: {exc}",
            stage="loading",
            file=str(path),
            line=line,
            field=field_name,
            path=str(path),
            fatal=fatal,
        )
        return None


def _weight_map(
    value: Any,
    *,
    field_path: str,
    file_path: Path,
    failures: FailureCollector,
    allow_empty: bool = False,
) -> dict[str, float] | None:
    if not isinstance(value, dict):
        failures.add(
            E_WEIGHT_INVALID,
            f"{field_path} must be an object mapping IDs to weights",
            stage="loading",
            file=str(file_path),
            field=field_path,
            path=field_path,
            fatal=True,
        )
        return None
    if not value and not allow_empty:
        failures.add(
            E_WEIGHT_INVALID,
            f"{field_path} must contain at least one weight",
            stage="loading",
            file=str(file_path),
            field=field_path,
            path=field_path,
            fatal=True,
        )
        return None
    result: dict[str, float] = {}
    total = as_decimal(0)
    for key, weight in value.items():
        item_path = f"{field_path}.{key}"
        if not is_safe_id(key):
            failures.add(
                E_CONFIG_INVALID,
                f"invalid ID {key!r} in weight map",
                stage="loading",
                file=str(file_path),
                field=item_path,
                path=item_path,
                fatal=True,
            )
            continue
        if not _is_nonnegative_number(weight):
            failures.add(
                E_WEIGHT_INVALID,
                f"weight at {item_path} must be a non-negative finite number",
                stage="loading",
                file=str(file_path),
                field=item_path,
                path=item_path,
                fatal=True,
            )
            continue
        result[key] = float(weight)
        total += as_decimal(weight)
    if result and total <= 0:
        failures.add(
            E_WEIGHT_INVALID,
            f"{field_path} weights must have a positive sum",
            stage="loading",
            file=str(file_path),
            field=field_path,
            path=field_path,
            fatal=True,
        )
        return None
    return result


def _template_placeholders(
    template: Any,
    *,
    file_path: Path,
    field_path: str,
    failures: FailureCollector,
    code: str = E_SEED_INVALID,
    fatal: bool = False,
) -> list[str] | None:
    if not isinstance(template, str) or not template.strip():
        failures.add(
            code,
            f"{field_path} must be a non-empty string",
            stage="loading",
            file=str(file_path),
            field=field_path,
            path=field_path,
            fatal=fatal,
        )
        return None
    placeholders = PLACEHOLDER_RE.findall(template)
    scrubbed = PLACEHOLDER_RE.sub("", template)
    if "{" in scrubbed or "}" in scrubbed:
        failures.add(
            code,
            f"{field_path} contains an invalid brace placeholder",
            stage="loading",
            file=str(file_path),
            field=field_path,
            path=field_path,
            fatal=fatal,
        )
        return None
    return sorted(set(placeholders))


def _validate_slots(
    value: Any,
    *,
    file_path: Path,
    field_path: str,
    failures: FailureCollector,
    code: str = E_SEED_INVALID,
    fatal: bool = False,
) -> list[str] | None:
    if not isinstance(value, list):
        failures.add(
            code,
            f"{field_path} must be an array of slot names",
            stage="loading",
            file=str(file_path),
            field=field_path,
            path=field_path,
            fatal=fatal,
        )
        return None
    result: list[str] = []
    seen: set[str] = set()
    for index, slot in enumerate(value):
        item_path = f"{field_path}[{index}]"
        if not is_safe_id(slot):
            failures.add(
                code,
                f"invalid slot name at {item_path}",
                stage="loading",
                file=str(file_path),
                field=item_path,
                path=item_path,
                fatal=fatal,
            )
            continue
        if slot in seen:
            failures.add(
                code,
                f"duplicate slot name {slot!r}",
                stage="loading",
                file=str(file_path),
                field=item_path,
                path=item_path,
                fatal=fatal,
            )
            continue
        seen.add(slot)
        result.append(slot)
    return result


def _parse_and_validate_seeds(
    path: Path,
    failures: FailureCollector,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, dict[str, Any]]], dict[str, int]]:
    seeds_by_id: dict[str, dict[str, Any]] = {}
    seeds_by_variant: dict[str, dict[str, dict[str, Any]]] = {}
    seed_lines: dict[str, int] = {}
    if not path.exists():
        failures.add(
            E_CONFIG_INVALID,
            f"required file does not exist: {path}",
            stage="loading",
            file=str(path),
            field="seeds",
            path=str(path),
            fatal=True,
        )
        return seeds_by_id, seeds_by_variant, seed_lines

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        failures.add(
            E_CONFIG_INVALID,
            f"cannot read seed file: {exc}",
            stage="loading",
            file=str(path),
            field="seeds",
            path=str(path),
            fatal=True,
        )
        return seeds_by_id, seeds_by_variant, seed_lines

    for line_number, raw_line in enumerate(lines, start=1):
        if not raw_line.strip():
            continue
        try:
            record = parse_json_text(raw_line)
        except ValueError as exc:
            failures.add(
                E_SEED_INVALID,
                f"invalid seed JSON: {exc}",
                stage="loading",
                file=str(path),
                line=line_number,
                field="line",
                path=f"line[{line_number}]",
            )
            continue
        if not isinstance(record, dict):
            failures.add(
                E_SEED_INVALID,
                "seed line must be a JSON object",
                stage="loading",
                file=str(path),
                line=line_number,
                field="line",
                path=f"line[{line_number}]",
            )
            continue

        seed_id = record.get("seed_id")
        task_type = record.get("task_type")
        variant_id = record.get("template_variant_id", seed_id)
        field_errors = 0
        for field_name, field_value in (("seed_id", seed_id), ("task_type", task_type), ("template_variant_id", variant_id)):
            if not is_safe_id(field_value):
                failures.add(
                    E_SEED_INVALID,
                    f"{field_name} must be a non-empty safe identifier",
                    stage="loading",
                    file=str(path),
                    line=line_number,
                    field=field_name,
                    path=field_name,
                )
                field_errors += 1
        if field_errors:
            continue
        if seed_id in seeds_by_id:
            failures.add(
                E_SEED_INVALID,
                f"duplicate seed_id {seed_id!r}; first seen on line {seed_lines[seed_id]}",
                stage="loading",
                file=str(path),
                line=line_number,
                field="seed_id",
                path="seed_id",
            )
            continue

        slots = _validate_slots(record.get("slots", []), file_path=path, field_path="slots", failures=failures)
        if slots is None:
            continue
        instruction_template = record.get("instruction_template")
        placeholders = _template_placeholders(
            instruction_template,
            file_path=path,
            field_path="instruction_template",
            failures=failures,
        )
        if placeholders is None:
            continue
        missing_slots = sorted(set(placeholders) - set(slots))
        if missing_slots:
            failures.add(
                E_SEED_INVALID,
                f"template placeholders are missing from slots: {', '.join(missing_slots)}",
                stage="loading",
                file=str(path),
                line=line_number,
                field="slots",
                path="slots",
            )
            continue
        reference_answer = record.get("reference_answer", "")
        if not isinstance(reference_answer, str):
            failures.add(
                E_SEED_INVALID,
                "reference_answer must be a string",
                stage="loading",
                file=str(path),
                line=line_number,
                field="reference_answer",
                path="reference_answer",
            )
            continue
        answer_placeholders = PLACEHOLDER_RE.findall(reference_answer)
        answer_scrubbed = PLACEHOLDER_RE.sub("", reference_answer)
        if "{" in answer_scrubbed or "}" in answer_scrubbed:
            failures.add(
                E_SEED_INVALID,
                "reference_answer contains an invalid brace placeholder",
                stage="loading",
                file=str(path),
                line=line_number,
                field="reference_answer",
                path="reference_answer",
            )
            continue
        unknown_answer_slots = sorted(set(answer_placeholders) - set(slots))
        if unknown_answer_slots:
            failures.add(
                E_SEED_INVALID,
                f"reference_answer uses slots not declared in slots: {', '.join(unknown_answer_slots)}",
                stage="loading",
                file=str(path),
                line=line_number,
                field="reference_answer",
                path="reference_answer",
            )
            continue

        normalized = {
            "seed_id": seed_id,
            "task_type": task_type,
            "template_variant_id": variant_id,
            "instruction_template": instruction_template,
            "slots": slots,
            "reference_answer": reference_answer,
            "line": line_number,
        }
        task_variants = seeds_by_variant.setdefault(task_type, {})
        if variant_id in task_variants:
            previous = task_variants[variant_id]
            failures.add(
                E_SEED_INVALID,
                f"duplicate template_variant_id {variant_id!r} for task {task_type!r}; "
                f"first seen on line {previous['line']}",
                stage="loading",
                file=str(path),
                line=line_number,
                field="template_variant_id",
                path="template_variant_id",
            )
            continue
        seeds_by_id[seed_id] = normalized
        seed_lines[seed_id] = line_number
        task_variants[variant_id] = normalized
    return seeds_by_id, seeds_by_variant, seed_lines


def _validate_lexicon(path: Path, raw: Any, failures: FailureCollector) -> dict[str, Any] | None:
    if not isinstance(raw, dict):
        failures.add(
            E_CONFIG_INVALID,
            "lexicon document must be a JSON object",
            stage="loading",
            file=str(path),
            field="lexicon",
            path="lexicon",
            fatal=True,
        )
        return None
    sources = raw.get("sources")
    if not isinstance(sources, dict) or not sources:
        failures.add(
            E_CONFIG_INVALID,
            "lexicon.sources must be a non-empty object",
            stage="loading",
            file=str(path),
            field="sources",
            path="sources",
            fatal=True,
        )
        return None
    normalized_sources: dict[str, list[str]] = {}
    for source_name, values in sources.items():
        source_path = f"sources.{source_name}"
        if not is_safe_id(source_name):
            failures.add(
                E_CONFIG_INVALID,
                f"invalid source name {source_name!r}",
                stage="loading",
                file=str(path),
                field=source_path,
                path=source_path,
                fatal=True,
            )
            continue
        if not isinstance(values, list) or not values:
            failures.add(
                E_CONFIG_INVALID,
                f"{source_path} must be a non-empty array of strings",
                stage="loading",
                file=str(path),
                field=source_path,
                path=source_path,
                fatal=True,
            )
            continue
        cleaned: list[str] = []
        seen: set[str] = set()
        for index, entry in enumerate(values):
            if not isinstance(entry, str) or not entry.strip():
                failures.add(
                    E_CONFIG_INVALID,
                    f"lexicon value at {source_path}[{index}] must be a non-empty string",
                    stage="loading",
                    file=str(path),
                    field=f"{source_path}[{index}]",
                    path=f"{source_path}[{index}]",
                    fatal=True,
                )
                continue
            if entry in seen:
                failures.add(
                    E_CONFIG_INVALID,
                    f"duplicate lexicon value at {source_path}[{index}]",
                    stage="loading",
                    file=str(path),
                    field=f"{source_path}[{index}]",
                    path=f"{source_path}[{index}]",
                    fatal=True,
                )
                continue
            seen.add(entry)
            cleaned.append(entry)
        if cleaned:
            normalized_sources[source_name] = cleaned

    subword = raw.get("subword")
    if not isinstance(subword, dict):
        failures.add(
            E_CONFIG_INVALID,
            "lexicon.subword must be an object",
            stage="loading",
            file=str(path),
            field="subword",
            path="subword",
            fatal=True,
        )
        return None
    lowercase = subword.get("lowercase")
    ngram_size = subword.get("ngram_size")
    drop_chars = subword.get("drop_chars", "")
    min_chars = subword.get("min_chars", 1)
    valid = True
    if not isinstance(lowercase, bool):
        valid = False
        failures.add(E_CONFIG_INVALID, "subword.lowercase must be boolean", stage="loading", file=str(path), field="subword.lowercase", path="subword.lowercase", fatal=True)
    if not isinstance(ngram_size, int) or isinstance(ngram_size, bool) or not 1 <= ngram_size <= 8:
        valid = False
        failures.add(E_CONFIG_INVALID, "subword.ngram_size must be an integer from 1 to 8", stage="loading", file=str(path), field="subword.ngram_size", path="subword.ngram_size", fatal=True)
    if not isinstance(drop_chars, str):
        valid = False
        failures.add(E_CONFIG_INVALID, "subword.drop_chars must be a string", stage="loading", file=str(path), field="subword.drop_chars", path="subword.drop_chars", fatal=True)
    if not isinstance(min_chars, int) or isinstance(min_chars, bool) or not 1 <= min_chars <= 8:
        valid = False
        failures.add(E_CONFIG_INVALID, "subword.min_chars must be an integer from 1 to 8", stage="loading", file=str(path), field="subword.min_chars", path="subword.min_chars", fatal=True)
    if not normalized_sources or not valid:
        return None
    return {
        "sources": normalized_sources,
        "subword": {
            "lowercase": lowercase,
            "ngram_size": ngram_size,
            "drop_chars": drop_chars,
            "min_chars": min_chars,
        },
    }


def _validate_scoring(path: Path, raw: Any, failures: FailureCollector) -> dict[str, Any] | None:
    start = len(failures.items)
    if not isinstance(raw, dict):
        failures.add(E_CONFIG_INVALID, "scoring document must be a JSON object", stage="loading", file=str(path), field="scoring", path="scoring", fatal=True)
        return None
    dimensions = raw.get("dimensions")
    if not isinstance(dimensions, dict):
        failures.add(E_CONFIG_INVALID, "scoring.dimensions must be an object", stage="loading", file=str(path), field="dimensions", path="dimensions", fatal=True)
        return None
    normalized_dimensions: dict[str, dict[str, float]] = {}
    weight_total = as_decimal(0)
    for dimension in DIMENSION_ORDER:
        item = dimensions.get(dimension)
        item_path = f"dimensions.{dimension}"
        if not isinstance(item, dict):
            failures.add(E_CONFIG_INVALID, f"{item_path} must be an object", stage="loading", file=str(path), field=item_path, path=item_path, fatal=True)
            continue
        weight = item.get("weight")
        baseline = item.get("baseline")
        if not _is_nonnegative_number(weight):
            failures.add(E_WEIGHT_INVALID, f"{item_path}.weight must be a non-negative finite number", stage="loading", file=str(path), field=f"{item_path}.weight", path=f"{item_path}.weight", fatal=True)
            weight = None
        else:
            weight_total += as_decimal(weight)
        if not finite_number(baseline) or not 0 <= as_decimal(baseline) <= 100:
            failures.add(E_CONFIG_INVALID, f"{item_path}.baseline must be a finite number in [0, 100]", stage="loading", file=str(path), field=f"{item_path}.baseline", path=f"{item_path}.baseline", fatal=True)
            baseline = None
        if weight is not None and baseline is not None:
            normalized_dimensions[dimension] = {"weight": float(weight), "baseline": float(baseline)}
    if weight_total <= 0:
        failures.add(E_WEIGHT_INVALID, "scoring dimension weights must have a positive sum", stage="loading", file=str(path), field="dimensions", path="dimensions", fatal=True)
    pass_threshold = raw.get("pass_threshold")
    if not finite_number(pass_threshold) or not 0 <= as_decimal(pass_threshold) <= 100:
        failures.add(E_CONFIG_INVALID, "pass_threshold must be a finite number in [0, 100]", stage="loading", file=str(path), field="pass_threshold", path="pass_threshold", fatal=True)
    ideal_length = raw.get("ideal_length_chars")
    if not _is_positive_int(ideal_length):
        failures.add(E_CONFIG_INVALID, "ideal_length_chars must be a positive integer", stage="loading", file=str(path), field="ideal_length_chars", path="ideal_length_chars", fatal=True)
    penalty = raw.get("length_penalty_scale")
    if not _is_nonnegative_number(penalty):
        failures.add(E_CONFIG_INVALID, "length_penalty_scale must be a non-negative finite number", stage="loading", file=str(path), field="length_penalty_scale", path="length_penalty_scale", fatal=True)
    dimension_order = raw.get("dimension_order", list(DIMENSION_ORDER))
    if dimension_order != list(DIMENSION_ORDER):
        failures.add(E_CONFIG_INVALID, f"dimension_order must be exactly {list(DIMENSION_ORDER)!r}", stage="loading", file=str(path), field="dimension_order", path="dimension_order", fatal=True)
    if len(normalized_dimensions) != len(DIMENSION_ORDER) or not all(name in normalized_dimensions for name in DIMENSION_ORDER):
        return None
    if any(item.fatal for item in failures.items[start:]):
        return None
    return {
        "dimension_order": list(DIMENSION_ORDER),
        "dimensions": normalized_dimensions,
        "pass_threshold": float(pass_threshold),
        "ideal_length_chars": ideal_length,
        "length_penalty_scale": float(penalty),
    }


def _validate_task_settings(
    task_type: Any,
    item: Any,
    *,
    path: Path,
    field_prefix: str,
    failures: FailureCollector,
    quota: Any | None = None,
) -> dict[str, Any] | None:
    if not is_safe_id(task_type):
        failures.add(E_CONFIG_INVALID, "task type must be a non-empty safe identifier", stage="loading", file=str(path), field=field_prefix, path=field_prefix, fatal=True)
        return None
    if not isinstance(item, dict):
        failures.add(E_CONFIG_INVALID, f"{field_prefix} must be an object", stage="loading", file=str(path), field=field_prefix, path=field_prefix, fatal=True)
        return None
    target = item.get("target_count")
    if not _is_positive_int(target):
        failures.add(E_CONFIG_INVALID, f"{field_prefix}.target_count must be a positive integer", stage="loading", file=str(path), field=f"{field_prefix}.target_count", path=f"{field_prefix}.target_count", fatal=True)
    template_weights = _weight_map(item.get("template_weights"), field_path=f"{field_prefix}.template_weights", file_path=path, failures=failures)
    slot_source_weights = _weight_map(item.get("slot_source_weights"), field_path=f"{field_prefix}.slot_source_weights", file_path=path, failures=failures)
    if quota is not None and not _is_positive_int(quota):
        failures.add(E_CONFIG_INVALID, f"{field_prefix}.task_quota must be a positive integer", stage="loading", file=str(path), field=f"{field_prefix}.task_quota", path=f"{field_prefix}.task_quota", fatal=True)
    if target is None or template_weights is None or slot_source_weights is None:
        return None
    result = {
        "target_count": target,
        "template_weights": template_weights,
        "slot_source_weights": slot_source_weights,
    }
    if quota is not None:
        result["task_quota"] = quota
    return result


def _validate_config(path: Path, raw: Any, failures: FailureCollector) -> dict[str, Any] | None:
    start = len(failures.items)
    if not isinstance(raw, dict):
        failures.add(E_CONFIG_INVALID, "config document must be a JSON object", stage="loading", file=str(path), field="config", path="config", fatal=True)
        return None
    tasks = raw.get("task_types")
    quotas = raw.get("task_quotas")
    if not isinstance(tasks, dict) or not tasks:
        failures.add(E_CONFIG_INVALID, "task_types must be a non-empty object", stage="loading", file=str(path), field="task_types", path="task_types", fatal=True)
        return None
    if not isinstance(quotas, dict):
        failures.add(E_CONFIG_INVALID, "task_quotas must be an object", stage="loading", file=str(path), field="task_quotas", path="task_quotas", fatal=True)
        quotas = {}
    normalized_tasks: dict[str, dict[str, Any]] = {}
    normalized_quotas: dict[str, int] = {}
    for task_type, item in tasks.items():
        prefix = f"task_types.{task_type}"
        quota = quotas.get(task_type)
        if task_type not in quotas:
            failures.add(E_CONFIG_INVALID, f"missing positive quota for task type {task_type!r}", stage="loading", file=str(path), field=f"task_quotas.{task_type}", path=f"task_quotas.{task_type}", fatal=True)
        settings = _validate_task_settings(task_type, item, path=path, field_prefix=prefix, failures=failures, quota=quota)
        if settings is not None:
            normalized_tasks[task_type] = settings
            normalized_quotas[task_type] = settings.pop("task_quota")
    for task_type, quota in quotas.items():
        if task_type not in tasks:
            if not _is_positive_int(quota):
                failures.add(E_CONFIG_INVALID, f"task_quotas.{task_type} must be a positive integer", stage="loading", file=str(path), field=f"task_quotas.{task_type}", path=f"task_quotas.{task_type}", fatal=True)
            else:
                normalized_quotas[task_type] = quota

    result: dict[str, Any] = {
        "task_types": normalized_tasks,
        "task_quotas": normalized_quotas,
    }
    threshold = raw.get("near_duplicate_threshold")
    if not finite_number(threshold) or not 0 <= as_decimal(threshold) <= 1:
        failures.add(E_CONFIG_INVALID, "near_duplicate_threshold must be a finite number in [0, 1]", stage="loading", file=str(path), field="near_duplicate_threshold", path="near_duplicate_threshold", fatal=True)
    else:
        result["near_duplicate_threshold"] = float(threshold)
    places = raw.get("fixed_decimal_places")
    if not isinstance(places, int) or isinstance(places, bool) or not MIN_DECIMAL_PLACES <= places <= MAX_DECIMAL_PLACES:
        failures.add(E_CONFIG_INVALID, f"fixed_decimal_places must be an integer in [{MIN_DECIMAL_PLACES}, {MAX_DECIMAL_PLACES}]", stage="loading", file=str(path), field="fixed_decimal_places", path="fixed_decimal_places", fatal=True)
    else:
        result["fixed_decimal_places"] = places
    step = raw.get("quantization_step")
    if not _is_nonnegative_number(step) or float(step) not in ALLOWED_QUANTIZATION_STEPS:
        failures.add(E_CONFIG_INVALID, f"quantization_step must be one of {list(ALLOWED_QUANTIZATION_STEPS)!r}", stage="loading", file=str(path), field="quantization_step", path="quantization_step", fatal=True)
    else:
        result["quantization_step"] = float(step)
    if isinstance(places, int) and not isinstance(places, bool) and _is_nonnegative_number(step):
        scaled = as_decimal(step) * (as_decimal(10) ** places)
        if scaled != scaled.to_integral_value():
            failures.add(E_CONFIG_INVALID, "quantization_step must be representable at fixed_decimal_places", stage="loading", file=str(path), field="quantization_step", path="quantization_step", fatal=True)
    concurrency = raw.get("max_concurrency")
    if not _is_positive_int(concurrency) or concurrency > HARD_MAX_CONCURRENCY:
        failures.add(E_CONFIG_INVALID, f"max_concurrency must be an integer in [1, {HARD_MAX_CONCURRENCY}]", stage="loading", file=str(path), field="max_concurrency", path="max_concurrency", fatal=True)
    else:
        result["max_concurrency"] = concurrency
    for name in (
        "global_score_budget",
        "per_batch_generation_points",
        "snapshot_commit_interval_batches",
    ):
        value = raw.get(name)
        if not _is_positive_int(value):
            failures.add(E_CONFIG_INVALID, f"{name} must be a positive integer", stage="loading", file=str(path), field=name, path=name, fatal=True)
        else:
            result[name] = value
    return result if not any(item.fatal for item in failures.items[start:]) else None


def _validate_plan_operation(
    operation: Any,
    *,
    op_path: str,
    path: Path,
    failures: FailureCollector,
    sources: set[str] | None,
) -> None:
    if not isinstance(operation, dict):
        failures.add(E_CONFIG_INVALID, f"{op_path} must be an object", stage="loading", file=str(path), field=op_path, path=op_path, fatal=True)
        return
    op = operation.get("op")
    known = {
        "add_task",
        "delete_task",
        "set_target",
        "set_template_weights",
        "replace_template",
        "delete_template",
        "set_slot_source_weights",
        "add_slot_source",
        "remove_slot_source",
        "set_score_weight",
        "set_quality_threshold",
        "set_near_duplicate_threshold",
    }
    if op not in known:
        failures.add(E_CONFIG_INVALID, f"unknown plan operation {op!r}", stage="loading", file=str(path), field=f"{op_path}.op", path=f"{op_path}.op", fatal=True)
        return
    if op == "add_task":
        task_type = operation.get("task_type")
        _validate_task_settings(
            task_type,
            operation,
            path=path,
            field_prefix=op_path,
            failures=failures,
            quota=operation.get("task_quota"),
        )
    elif op in {"delete_task", "set_target", "set_template_weights", "set_slot_source_weights"}:
        if not is_safe_id(operation.get("task_type")):
            failures.add(E_CONFIG_INVALID, f"{op_path}.task_type must be a safe identifier", stage="loading", file=str(path), field=f"{op_path}.task_type", path=f"{op_path}.task_type", fatal=True)
        if op == "set_target" and not _is_positive_int(operation.get("target_count")):
            failures.add(E_CONFIG_INVALID, f"{op_path}.target_count must be a positive integer", stage="loading", file=str(path), field=f"{op_path}.target_count", path=f"{op_path}.target_count", fatal=True)
        if op == "set_template_weights":
            _weight_map(operation.get("template_weights"), field_path=f"{op_path}.template_weights", file_path=path, failures=failures)
        if op == "set_slot_source_weights":
            weights = _weight_map(operation.get("slot_source_weights"), field_path=f"{op_path}.slot_source_weights", file_path=path, failures=failures)
            if weights is not None and sources is not None:
                for source in weights:
                    if source not in sources:
                        failures.add(E_CONFIG_INVALID, f"unknown slot source {source!r}", stage="loading", file=str(path), field=f"{op_path}.slot_source_weights.{source}", path=f"{op_path}.slot_source_weights.{source}", fatal=True)
    elif op in {"replace_template", "delete_template"}:
        if not is_safe_id(operation.get("task_type")):
            failures.add(E_CONFIG_INVALID, f"{op_path}.task_type must be a safe identifier", stage="loading", file=str(path), field=f"{op_path}.task_type", path=f"{op_path}.task_type", fatal=True)
        if not is_safe_id(operation.get("variant_id")):
            failures.add(E_CONFIG_INVALID, f"{op_path}.variant_id must be a safe identifier", stage="loading", file=str(path), field=f"{op_path}.variant_id", path=f"{op_path}.variant_id", fatal=True)
        if op == "replace_template":
            placeholders = _template_placeholders(
                operation.get("instruction_template"),
                file_path=path,
                field_path=f"{op_path}.instruction_template",
                failures=failures,
                code=E_CONFIG_INVALID,
                fatal=True,
            )
            reference = operation.get("reference_answer", "")
            if not isinstance(reference, str):
                failures.add(E_CONFIG_INVALID, f"{op_path}.reference_answer must be a string", stage="loading", file=str(path), field=f"{op_path}.reference_answer", path=f"{op_path}.reference_answer", fatal=True)
            slots = operation.get("slots")
            if slots is not None:
                normalized_slots = _validate_slots(
                    slots,
                    file_path=path,
                    field_path=f"{op_path}.slots",
                    failures=failures,
                    code=E_CONFIG_INVALID,
                    fatal=True,
                )
                if placeholders is not None and normalized_slots is not None:
                    missing = sorted(set(placeholders) - set(normalized_slots))
                    if missing:
                        failures.add(E_CONFIG_INVALID, f"replacement template placeholders are missing from slots: {', '.join(missing)}", stage="loading", file=str(path), field=f"{op_path}.slots", path=f"{op_path}.slots", fatal=True)
    elif op == "add_slot_source":
        if not is_safe_id(operation.get("task_type")):
            failures.add(E_CONFIG_INVALID, f"{op_path}.task_type must be a safe identifier", stage="loading", file=str(path), field=f"{op_path}.task_type", path=f"{op_path}.task_type", fatal=True)
        source = operation.get("source")
        if not is_safe_id(source):
            failures.add(E_CONFIG_INVALID, f"{op_path}.source must be a safe identifier", stage="loading", file=str(path), field=f"{op_path}.source", path=f"{op_path}.source", fatal=True)
        elif sources is not None and source not in sources:
            failures.add(E_CONFIG_INVALID, f"unknown slot source {source!r}", stage="loading", file=str(path), field=f"{op_path}.source", path=f"{op_path}.source", fatal=True)
        weight = operation.get("weight")
        if not _is_nonnegative_number(weight):
            failures.add(E_WEIGHT_INVALID, f"{op_path}.weight must be a non-negative finite number", stage="loading", file=str(path), field=f"{op_path}.weight", path=f"{op_path}.weight", fatal=True)
    elif op == "remove_slot_source":
        if not is_safe_id(operation.get("task_type")):
            failures.add(E_CONFIG_INVALID, f"{op_path}.task_type must be a safe identifier", stage="loading", file=str(path), field=f"{op_path}.task_type", path=f"{op_path}.task_type", fatal=True)
        if not is_safe_id(operation.get("source")):
            failures.add(E_CONFIG_INVALID, f"{op_path}.source must be a safe identifier", stage="loading", file=str(path), field=f"{op_path}.source", path=f"{op_path}.source", fatal=True)
    elif op == "set_score_weight":
        if operation.get("dimension") not in DIMENSION_ORDER:
            failures.add(E_CONFIG_INVALID, f"{op_path}.dimension must be one of {list(DIMENSION_ORDER)!r}", stage="loading", file=str(path), field=f"{op_path}.dimension", path=f"{op_path}.dimension", fatal=True)
        if not _is_nonnegative_number(operation.get("weight")):
            failures.add(E_WEIGHT_INVALID, f"{op_path}.weight must be a non-negative finite number", stage="loading", file=str(path), field=f"{op_path}.weight", path=f"{op_path}.weight", fatal=True)
    elif op == "set_quality_threshold":
        value = operation.get("pass_threshold")
        if not finite_number(value) or not 0 <= as_decimal(value) <= 100:
            failures.add(E_CONFIG_INVALID, f"{op_path}.pass_threshold must be a finite number in [0, 100]", stage="loading", file=str(path), field=f"{op_path}.pass_threshold", path=f"{op_path}.pass_threshold", fatal=True)
    elif op == "set_near_duplicate_threshold":
        value = operation.get("threshold")
        if not finite_number(value) or not 0 <= as_decimal(value) <= 1:
            failures.add(E_CONFIG_INVALID, f"{op_path}.threshold must be a finite number in [0, 1]", stage="loading", file=str(path), field=f"{op_path}.threshold", path=f"{op_path}.threshold", fatal=True)


def _validate_plan(path: Path, raw: Any, failures: FailureCollector, sources: set[str] | None) -> dict[str, Any] | None:
    start = len(failures.items)
    if not isinstance(raw, dict):
        failures.add(E_CONFIG_INVALID, "plan document must be a JSON object", stage="loading", file=str(path), field="plan", path="plan", fatal=True)
        return None
    batches = raw.get("batches")
    if not isinstance(batches, list):
        failures.add(E_CONFIG_INVALID, "plan.batches must be an array", stage="loading", file=str(path), field="batches", path="batches", fatal=True)
        return None
    normalized_batches: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for index, batch in enumerate(batches):
        batch_path = f"batches[{index}]"
        if not isinstance(batch, dict):
            failures.add(E_CONFIG_INVALID, f"{batch_path} must be an object", stage="loading", file=str(path), field=batch_path, path=batch_path, fatal=True)
            continue
        batch_id = batch.get("batch_id")
        if not is_safe_id(batch_id):
            failures.add(E_CONFIG_INVALID, f"{batch_path}.batch_id must be a safe identifier", stage="loading", file=str(path), field=f"{batch_path}.batch_id", path=f"{batch_path}.batch_id", fatal=True)
            continue
        if batch_id in seen_ids:
            failures.add(E_CONFIG_INVALID, f"duplicate batch_id {batch_id!r}", stage="loading", file=str(path), field=f"{batch_path}.batch_id", path=f"{batch_path}.batch_id", fatal=True)
            continue
        seen_ids.add(batch_id)
        operations = batch.get("operations")
        if not isinstance(operations, list) or not operations:
            failures.add(E_CONFIG_INVALID, f"{batch_path}.operations must be a non-empty array", stage="loading", file=str(path), field=f"{batch_path}.operations", path=f"{batch_path}.operations", fatal=True)
            continue
        for op_index, operation in enumerate(operations):
            _validate_plan_operation(
                operation,
                op_path=f"{batch_path}.operations[{op_index}]",
                path=path,
                failures=failures,
                sources=sources,
            )
        normalized_batches.append({"batch_id": batch_id, "operations": operations})
    if any(item.fatal for item in failures.items[start:]):
        return None
    return {"batches": normalized_batches}


def _collect_plan_templates(plan: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    templates: dict[str, dict[str, dict[str, Any]]] = {}
    for batch in plan["batches"]:
        for operation in batch["operations"]:
            if operation.get("op") != "replace_template":
                continue
            task_type = operation["task_type"]
            variant_id = operation["variant_id"]
            placeholders = sorted(set(PLACEHOLDER_RE.findall(operation["instruction_template"])))
            slots = operation.get("slots")
            if slots is None:
                slots = placeholders
            templates.setdefault(task_type, {})[variant_id] = {
                "seed_id": f"plan:{task_type}:{variant_id}",
                "task_type": task_type,
                "template_variant_id": variant_id,
                "instruction_template": operation["instruction_template"],
                "slots": list(slots),
                "reference_answer": operation.get("reference_answer", ""),
                "line": None,
            }
    return templates


def load_inputs(
    input_dir: str | Path,
    *,
    overrides: dict[str, str | Path] | None = None,
) -> LoadedInputs:
    paths = InputPaths.from_dir(input_dir, overrides)
    failures = FailureCollector()

    seeds_by_id, seeds_by_variant, seed_lines = _parse_and_validate_seeds(paths.seeds, failures)
    raw_lexicon = _load_json_document(paths.lexicon, field_name="lexicon", failures=failures)
    raw_scoring = _load_json_document(paths.scoring, field_name="scoring", failures=failures)
    raw_config = _load_json_document(paths.config, field_name="config", failures=failures)
    raw_plan = _load_json_document(paths.plan, field_name="plan", failures=failures)

    lexicon = _validate_lexicon(paths.lexicon, raw_lexicon, failures) if raw_lexicon is not None else None
    scoring = _validate_scoring(paths.scoring, raw_scoring, failures) if raw_scoring is not None else None
    config = _validate_config(paths.config, raw_config, failures) if raw_config is not None else None
    source_names = set(lexicon["sources"]) if lexicon else None
    plan = _validate_plan(paths.plan, raw_plan, failures, source_names) if raw_plan is not None else None

    plan_templates = _collect_plan_templates(plan) if plan else {}
    allowed_tasks = set(config["task_types"]) if config else set()
    if plan:
        for batch in plan["batches"]:
            for operation in batch["operations"]:
                if operation.get("op") == "add_task" and is_safe_id(operation.get("task_type")):
                    allowed_tasks.add(operation["task_type"])

    for seed_id, seed in list(seeds_by_id.items()):
        if seed["task_type"] not in allowed_tasks:
            failures.add(
                E_SEED_INVALID,
                f"task_type {seed['task_type']!r} is not present in the configured task enum",
                stage="loading",
                file=str(paths.seeds),
                line=seed["line"],
                field="task_type",
                path="task_type",
            )
            seeds_by_id.pop(seed_id, None)
            seeds_by_variant.get(seed["task_type"], {}).pop(seed["template_variant_id"], None)

    if config and lexicon:
        for task_type, settings in config["task_types"].items():
            for variant_id in settings["template_weights"]:
                if variant_id not in seeds_by_variant.get(task_type, {}) and variant_id not in plan_templates.get(task_type, {}):
                    failures.add(
                        E_CONFIG_INVALID,
                        f"template variant {variant_id!r} for task {task_type!r} has no seed or replacement definition",
                        stage="loading",
                        file=str(paths.config),
                        field=f"task_types.{task_type}.template_weights.{variant_id}",
                        path=f"task_types.{task_type}.template_weights.{variant_id}",
                        fatal=True,
                    )
            for source_name in settings["slot_source_weights"]:
                if source_name not in lexicon["sources"]:
                    failures.add(
                        E_CONFIG_INVALID,
                        f"slot source {source_name!r} is not present in lexicon.sources",
                        stage="loading",
                        file=str(paths.config),
                        field=f"task_types.{task_type}.slot_source_weights.{source_name}",
                        path=f"task_types.{task_type}.slot_source_weights.{source_name}",
                        fatal=True,
                    )
    if plan and lexicon:
        for batch in plan["batches"]:
            for op_index, operation in enumerate(batch["operations"]):
                if operation.get("op") == "add_task":
                    task_type = operation["task_type"]
                    prefix = f"batches.{batch['batch_id']}.operations[{op_index}]"
                    for variant_id in operation["template_weights"]:
                        if variant_id not in seeds_by_variant.get(task_type, {}) and variant_id not in plan_templates.get(task_type, {}):
                            failures.add(
                                E_CONFIG_INVALID,
                                f"add_task template variant {variant_id!r} for task {task_type!r} has no seed or replacement definition",
                                stage="loading",
                                file=str(paths.plan),
                                field=f"{prefix}.template_weights.{variant_id}",
                                path=f"{prefix}.template_weights.{variant_id}",
                                fatal=True,
                            )
                    for source_name in operation["slot_source_weights"]:
                        if source_name not in lexicon["sources"]:
                            failures.add(
                                E_CONFIG_INVALID,
                                f"add_task slot source {source_name!r} is not present in lexicon.sources",
                                stage="loading",
                                file=str(paths.plan),
                                field=f"{prefix}.slot_source_weights.{source_name}",
                                path=f"{prefix}.slot_source_weights.{source_name}",
                                fatal=True,
                            )

    fingerprints = {
        "seed_file": fingerprint_file(paths.seeds) if paths.seeds.exists() else "",
        "lexicon": fingerprint_object(lexicon) if lexicon else "",
        "scoring": fingerprint_object(scoring) if scoring else "",
        "config": fingerprint_object(config) if config else "",
        "plan": fingerprint_object(plan) if plan else "",
    }
    return LoadedInputs(
        paths=paths,
        seeds_by_id=seeds_by_id,
        seeds_by_variant=seeds_by_variant,
        seed_lines=seed_lines,
        plan_templates=plan_templates,
        lexicon=lexicon or {"sources": {}, "subword": {}},
        scoring=scoring or {},
        config=config or {},
        plan=plan or {"batches": []},
        fingerprints=fingerprints,
        failures=failures,
    )
