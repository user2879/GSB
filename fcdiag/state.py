"""Editable pipeline state and deterministic parameter version accounting."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

from .fingerprint import canonical_json, sha256_hex, short_digest
from .fixedpoint import FixedConfig
from .models import Config, Pipeline, Sample, StageDef


@dataclass
class WorkState:
    samples: Tuple[Sample, ...]
    pipeline: Pipeline
    config: Config
    receipt_version: int = 3
    param_versions: Dict[str, int] = field(default_factory=dict)
    sequence: int = 0

    @classmethod
    def initial(cls, samples: Tuple[Sample, ...], pipeline: Pipeline, config: Config) -> "WorkState":
        versions = {}
        for stage in pipeline.stages:
            for key in sorted(stage.params):
                versions[f"{stage.name}.{key}"] = 0
            versions[f"{stage.name}.fixed.frac_bits"] = 0
            versions[f"{stage.name}.fixed.gear"] = 0
        return cls(tuple(samples), pipeline, config, 3, versions, 0)

    def copy(self) -> "WorkState":
        return WorkState(
            self.samples,
            self.pipeline,
            self.config,
            self.receipt_version,
            dict(self.param_versions),
            self.sequence,
        )

    def sample_fingerprint(self) -> str:
        body = [sample.to_dict() for sample in self.samples]
        return short_digest(body, length=24)

    def config_hash(self) -> str:
        return sha256_hex(self.config.to_dict())

    def receipt_fingerprint(self) -> str:
        return short_digest({"receipt_version": self.receipt_version}, length=24)

    def global_param_version(self) -> int:
        return sum(self.param_versions.values()) + self.sequence


def _replace_stage(state: WorkState, stage_name: str, mutate) -> Pipeline:
    stages: List[StageDef] = []
    for stage in state.pipeline.stages:
        if stage.name == stage_name:
            params = copy.deepcopy(stage.params)
            mutate(params)
            stage = StageDef(stage.name, stage.kind, stage.upstream, params, stage.order)
        stages.append(stage)
    return Pipeline(state.pipeline.name, state.pipeline.feature_count, tuple(stages))


def _bump(state: WorkState, key: str) -> None:
    if key not in state.param_versions:
        state.param_versions[key] = 0
    state.param_versions[key] += 1


def apply_command(state: WorkState, command: dict) -> WorkState:
    """Return a new state; parameter version numbers only move on real edits."""

    next_state = state.copy()
    next_state.sequence += 1
    op = command["op"]
    if op in ("replace_sample", "append_sample"):
        new_sample = Sample(
            str(command["sample"]["sample_id"]),
            tuple(float(v) for v in command["sample"]["features"]),
            command["sample"].get("label"),
        )
        samples = list(next_state.samples)
        if op == "replace_sample":
            ids = [sample.sample_id for sample in samples]
            if new_sample.sample_id not in ids:
                raise KeyError(f"unknown sample_id {new_sample.sample_id}")
            samples[ids.index(new_sample.sample_id)] = new_sample
        else:
            if any(sample.sample_id == new_sample.sample_id for sample in samples):
                raise KeyError(f"duplicate append sample_id {new_sample.sample_id}")
            samples.append(new_sample)
        next_state.samples = tuple(samples)
        return next_state

    stage_name = command["stage"]
    if op == "set_bucket_count":
        value = int(command["value"])

        def mutate_bucket(params: dict) -> None:
            params["bucket_count"] = value

        next_state.pipeline = _replace_stage(next_state, stage_name, mutate_bucket)
        _bump(next_state, f"{stage_name}.bucket_count")
    elif op == "set_interaction_degree":
        value = int(command["value"])

        def mutate_degree(params: dict) -> None:
            params["degree"] = value

        next_state.pipeline = _replace_stage(next_state, stage_name, mutate_degree)
        _bump(next_state, f"{stage_name}.degree")
    elif op == "set_threshold":

        def mutate_threshold(params: dict) -> None:
            params["threshold"] = float(command["value"])

        next_state.pipeline = _replace_stage(next_state, stage_name, mutate_threshold)
        _bump(next_state, f"{stage_name}.threshold")
    elif op == "set_weight":
        index = int(command["index"])

        def mutate_weight(params: dict) -> None:
            weights = list(params["weights"])
            while len(weights) <= index:
                weights.append(0.0)
            weights[index] = float(command["value"])
            params["weights"] = weights

        next_state.pipeline = _replace_stage(next_state, stage_name, mutate_weight)
        _bump(next_state, f"{stage_name}.weights")
    elif op == "set_frac_bits":
        old = next_state.config
        next_state.config = Config(
            FixedConfig(int(command["value"]), old.fixed.gear),
            old.budget,
            old.policy,
            old.seed,
        )
        for stage in next_state.pipeline.stages:
            _bump(next_state, f"{stage.name}.fixed.frac_bits")
    elif op == "set_gear":
        old = next_state.config
        next_state.config = Config(
            FixedConfig(old.fixed.frac_bits, int(command["value"])),
            old.budget,
            old.policy,
            old.seed,
        )
        for stage in next_state.pipeline.stages:
            _bump(next_state, f"{stage.name}.fixed.gear")
    elif op == "set_receipt_version":
        version = int(command["value"])
        if version not in (1, 2, 3):
            raise ValueError("receipt version must be 1, 2 or 3")
        next_state.receipt_version = version
    else:
        raise ValueError(f"unknown op {op}")
    return next_state


def apply_commands(state: WorkState, commands: Tuple[dict, ...]) -> Tuple[WorkState, ...]:
    states = [state]
    for command in commands:
        states.append(apply_command(states[-1], command))
    return tuple(states)
