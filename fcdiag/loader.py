"""Loading and validation of samples, pipelines, configs and command streams.

Validation collects *every* violation and raises :class:`InputInvalidError`
once the whole document has been inspected, so callers receive a complete
list with locators rather than one error at a time.
"""

from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Tuple

from .errors import E_INPUT_INVALID, Violation, InputInvalidError
from .fixedpoint import FixedConfig
from .miniyaml import MiniYamlError, parse_yaml
from .models import Budget, Config, Pipeline, Policy, Sample, StageDef


def _raise(violations: List[Violation]) -> None:
    if violations:
        raise InputInvalidError(violations)


def load_samples(text: str) -> Tuple[Sample, ...]:
    violations: List[Violation] = []
    samples: List[Sample] = []
    seen: Dict[str, int] = {}
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        loc = f"samples.jsonl:L{line_no}"
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as exc:
            violations.append(Violation(E_INPUT_INVALID, loc, f"invalid JSON: {exc.msg}"))
            continue
        if not isinstance(obj, dict):
            violations.append(Violation(E_INPUT_INVALID, loc, "sample must be a JSON object"))
            continue
        sample_id = obj.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            violations.append(Violation(E_INPUT_INVALID, loc, "sample_id must be a non-empty string"))
        elif sample_id in seen:
            violations.append(
                Violation(
                    E_INPUT_INVALID,
                    loc,
                    f"duplicate sample_id {sample_id!r}, first seen at L{seen[sample_id]}",
                )
            )
        else:
            seen[sample_id] = line_no
        features = obj.get("features")
        parsed_features: List[float] = []
        if not isinstance(features, list) or not features:
            violations.append(Violation(E_INPUT_INVALID, loc, "features must be a non-empty array"))
        else:
            for idx, value in enumerate(features):
                value_loc = f"{loc}.features[{idx}]"
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    violations.append(
                        Violation(E_INPUT_INVALID, value_loc, "feature must be a number, not bool/null/string")
                    )
                elif not math.isfinite(value):
                    violations.append(Violation(E_INPUT_INVALID, value_loc, "feature must be finite"))
                else:
                    parsed_features.append(float(value))
        label = obj.get("label")
        if label is not None and (not isinstance(label, int) or isinstance(label, bool) or label not in (0, 1)):
            violations.append(Violation(E_INPUT_INVALID, loc, "label must be 0, 1 or null"))
        if sample_id and isinstance(features, list) and len(parsed_features) == len(features):
            samples.append(Sample(sample_id, tuple(parsed_features), label))
    if not samples:
        violations.append(Violation(E_INPUT_INVALID, "samples.jsonl", "no valid samples found"))
    _raise(violations)
    return tuple(samples)


def _topo_validate(stage_dicts: List[dict], violations: List[Violation]) -> Tuple[StageDef, ...]:
    names = [item.get("name") for item in stage_dicts]
    stages: List[StageDef] = []
    for order, item in enumerate(stage_dicts):
        loc = f"pipeline.yaml:stages[{order}]"
        name = item.get("name")
        kind = item.get("kind")
        if not isinstance(name, str) or not name:
            violations.append(Violation(E_INPUT_INVALID, loc, "stage name must be a non-empty string"))
            continue
        if names.count(name) > 1:
            violations.append(Violation(E_INPUT_INVALID, loc, f"duplicate stage name {name!r}"))
        if kind not in ("standardize", "bucketize", "interact", "train_view"):
            violations.append(Violation(E_INPUT_INVALID, loc, f"unsupported stage kind {kind!r}"))
        upstream = item.get("upstream", [])
        if not isinstance(upstream, list) or not all(isinstance(u, str) for u in upstream):
            violations.append(Violation(E_INPUT_INVALID, loc, "upstream must be a list of stage names"))
        else:
            for ref in upstream:
                if ref not in names:
                    violations.append(
                        Violation(E_INPUT_INVALID, f"{loc}.upstream", f"unknown upstream stage {ref!r}")
                    )
                if ref == name:
                    violations.append(Violation(E_INPUT_INVALID, f"{loc}.upstream", "stage cannot reference itself"))
        params = item.get("params", {})
        if not isinstance(params, dict):
            violations.append(Violation(E_INPUT_INVALID, loc, "params must be a mapping"))
            params = {}
        stages.append(StageDef(name, kind, tuple(upstream), params, order))

    by_name = {stage.name: stage for stage in stages}
    # DFS cycle detection with deterministic visiting order.
    state: Dict[str, int] = {}

    def visit(stage_name: str, path: List[str]) -> None:
        if stage_name not in by_name:
            return
        mark = state.get(stage_name, 0)
        if mark == 1:
            trail = path + [stage_name]
            start = trail.index(stage_name)
            cycle = " -> ".join(trail[start:])
            violations.append(
                Violation(E_INPUT_INVALID, f"pipeline.yaml:stages.{stage_name}", f"dependency cycle: {cycle}")
            )
            return
        if mark == 2:
            return
        state[stage_name] = 1
        for ref in by_name[stage_name].upstream:
            visit(ref, path + [stage_name])
        state[stage_name] = 2

    for stage in stages:
        if state.get(stage.name, 0) != 2:
            visit(stage.name, [])
    return tuple(stages)


def load_pipeline(text: str) -> Pipeline:
    violations: List[Violation] = []
    try:
        doc = parse_yaml(text)
    except MiniYamlError as exc:
        raise InputInvalidError(
            [Violation(E_INPUT_INVALID, f"pipeline.yaml:L{exc.line_no}", str(exc))]
        )
    if not isinstance(doc, dict):
        raise InputInvalidError([Violation(E_INPUT_INVALID, "pipeline.yaml", "top level must be a mapping")])
    name = doc.get("name")
    if not isinstance(name, str) or not name:
        violations.append(Violation(E_INPUT_INVALID, "pipeline.yaml:name", "pipeline name required"))
    feature_count = doc.get("feature_count")
    if not isinstance(feature_count, int) or isinstance(feature_count, bool) or feature_count <= 0:
        violations.append(Violation(E_INPUT_INVALID, "pipeline.yaml:feature_count", "must be a positive integer"))
    stage_dicts = doc.get("stages")
    if not isinstance(stage_dicts, list) or not stage_dicts:
        violations.append(Violation(E_INPUT_INVALID, "pipeline.yaml:stages", "at least one stage is required"))
        stage_dicts = []
    stages = _topo_validate(stage_dicts, violations)
    _validate_stage_params(stages, violations)
    if isinstance(feature_count, int) and feature_count > 0:
        for stage in stages:
            degree = stage.params.get("degree")
            if stage.kind == "interact" and isinstance(degree, int) and degree > feature_count:
                violations.append(
                    Violation(
                        E_INPUT_INVALID,
                        f"pipeline.yaml:stages[{stage.order}].params.degree",
                        "interaction degree cannot exceed feature_count",
                    )
                )
    _raise(violations)
    return Pipeline(name, feature_count, stages)


def _validate_stage_params(stages: Tuple[StageDef, ...], violations: List[Violation]) -> None:
    by_name = {stage.name: stage for stage in stages}
    for stage in stages:
        loc = f"pipeline.yaml:stages[{stage.order}].params"
        upstream_kinds = [
            by_name[ref].kind for ref in stage.upstream if ref in by_name
        ]
        if stage.kind == "standardize" and stage.upstream:
            violations.append(Violation(E_INPUT_INVALID, loc, "standardize reads raw samples and cannot declare upstream"))
        if stage.kind == "bucketize" and "standardize" not in upstream_kinds:
            violations.append(Violation(E_INPUT_INVALID, loc, "bucketize must reference a standardize stage"))
        if stage.kind == "interact" and "standardize" not in upstream_kinds:
            violations.append(Violation(E_INPUT_INVALID, loc, "interact must reference a standardize stage"))
        if stage.kind == "train_view" and "interact" not in upstream_kinds:
            violations.append(Violation(E_INPUT_INVALID, loc, "train_view must reference an interact stage"))
        if stage.kind == "bucketize":
            bucket_count = stage.params.get("bucket_count")
            if not isinstance(bucket_count, int) or isinstance(bucket_count, bool) or bucket_count <= 0:
                violations.append(Violation(E_INPUT_INVALID, loc, "bucket_count must be a positive integer"))
        elif stage.kind == "interact":
            degree = stage.params.get("degree")
            if not isinstance(degree, int) or isinstance(degree, bool) or degree <= 0:
                violations.append(Violation(E_INPUT_INVALID, loc, "interaction degree must be a positive integer"))
        elif stage.kind == "train_view":
            weights = stage.params.get("weights")
            if not isinstance(weights, list) or not weights:
                violations.append(Violation(E_INPUT_INVALID, loc, "weights must be a non-empty list"))
            else:
                for idx, weight in enumerate(weights):
                    if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                        violations.append(
                            Violation(E_INPUT_INVALID, f"{loc}.weights[{idx}]", "weight must be numeric")
                        )
                    elif not math.isfinite(weight):
                        violations.append(
                            Violation(E_INPUT_INVALID, f"{loc}.weights[{idx}]", "weight must be finite")
                        )
            threshold = stage.params.get("threshold")
            if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold):
                violations.append(Violation(E_INPUT_INVALID, loc, "threshold must be a finite number"))


def load_config(doc: Dict[str, Any]) -> Config:
    violations: List[Violation] = []
    fixed_doc = doc.get("fixed", {})
    frac_bits = fixed_doc.get("frac_bits")
    gear = fixed_doc.get("gear")
    if not isinstance(frac_bits, int) or isinstance(frac_bits, bool) or not 1 <= frac_bits <= 12:
        violations.append(Violation(E_INPUT_INVALID, "config.json:fixed.frac_bits", "integer in [1,12] required"))
        frac_bits = 4
    if not isinstance(gear, int) or isinstance(gear, bool) or gear <= 0:
        violations.append(Violation(E_INPUT_INVALID, "config.json:fixed.gear", "positive integer required"))
        gear = 1
    budget_doc = doc.get("budget", {})
    batch_points = budget_doc.get("batch_points")
    if not isinstance(batch_points, int) or isinstance(batch_points, bool) or batch_points <= 0:
        violations.append(Violation(E_INPUT_INVALID, "config.json:budget.batch_points", "positive integer required"))
        batch_points = 1
    quotas = budget_doc.get("stage_quotas", {})
    if not isinstance(quotas, dict):
        violations.append(Violation(E_INPUT_INVALID, "config.json:budget.stage_quotas", "mapping required"))
        quotas = {}
    clean_quotas: Dict[str, int] = {}
    for key, value in quotas.items():
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            violations.append(
                Violation(E_INPUT_INVALID, f"config.json:budget.stage_quotas.{key}", "positive integer required")
            )
        else:
            clean_quotas[key] = value
    policy_doc = doc.get("policy", {})
    degraded_reuse = policy_doc.get("degraded_reuse", "param_version")
    if degraded_reuse not in ("strict", "param_version"):
        violations.append(
            Violation(E_INPUT_INVALID, "config.json:policy.degraded_reuse", "must be 'strict' or 'param_version'")
        )
        degraded_reuse = "strict"
    seed = doc.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool):
        violations.append(Violation(E_INPUT_INVALID, "config.json:seed", "integer seed required"))
        seed = 0
    _raise(violations)
    return Config(
        FixedConfig(frac_bits, gear),
        Budget(batch_points, clean_quotas),
        Policy(degraded_reuse),
        seed,
    )


def load_commands(text: str) -> Tuple[dict, ...]:
    violations: List[Violation] = []
    commands: List[dict] = []
    allowed = {
        "replace_sample",
        "append_sample",
        "set_bucket_count",
        "set_interaction_degree",
        "set_threshold",
        "set_weight",
        "set_frac_bits",
        "set_gear",
        "set_receipt_version",
    }
    for line_no, raw in enumerate(text.splitlines(), start=1):
        if not raw.strip():
            continue
        loc = f"commands.jsonl:L{line_no}"
        try:
            command = json.loads(raw)
        except json.JSONDecodeError as exc:
            violations.append(Violation(E_INPUT_INVALID, loc, f"invalid JSON: {exc.msg}"))
            continue
        if not isinstance(command, dict) or command.get("op") not in allowed:
            violations.append(Violation(E_INPUT_INVALID, loc, f"op must be one of {sorted(allowed)}"))
            continue
        commands.append(command)
    _raise(violations)
    return tuple(commands)
