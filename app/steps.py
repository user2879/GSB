"""管道步骤的确定性执行。

确定性规则（显式声明）：
- 全部运算在 10^6 定点整数域进行，禁止浮点中间结果参与迭代。
- 运算顺序固定为实体 ID 升序、特征名升序。
- 缺失传播：任何依赖缺失输入的输出亦为缺失；步骤不为缺失条目造值。
- 分桶边界相等：归入较低分桶（bisect_left 语义），并记录决断。
- 交叉特征哈希冲突：同名交叉特征保留先出现的一对（keep_first），并记录决断。
- 同一输入与配置下，逐实体特征值逐字段可复现。
"""
from __future__ import annotations

import bisect
import hashlib
from dataclasses import dataclass, field

from .ir import SCALE, Step, fingerprint


def to_fp(value: float) -> int:
    return int(round(value * SCALE))


def from_fp(value: int) -> float:
    return value / SCALE


@dataclass
class StepResult:
    index: int
    name: str
    cache_key: str
    values: dict[str, dict[str, int]]
    missing: dict[str, set[str]]
    decisions: list[dict] = field(default_factory=list)
    summary: dict = field(default_factory=dict)
    reused: bool = False

    def summary_dict(self) -> dict:
        return {
            "index": self.index,
            "name": self.name,
            "cache_key": self.cache_key,
            "reused": self.reused,
            "summary": self.summary,
            "decisions": self.decisions,
        }


def _summarize(values: dict[str, dict[str, int]], missing: dict[str, set[str]]) -> dict:
    canonical = {
        "values": {e: dict(sorted(f.items())) for e, f in sorted(values.items())},
        "missing": {e: sorted(m) for e, m in sorted(missing.items())},
    }
    return {
        "n_entities": len(values),
        "n_features": len({f for feats in values.values() for f in feats}),
        "values_sha256": fingerprint(canonical),
    }


def _copy_state(values, missing):
    return ({e: dict(f) for e, f in values.items()},
            {e: set(m) for e, m in missing.items()})


def execute_step(index: int, step: Step, values: dict[str, dict[str, int]],
                 missing: dict[str, set[str]], cache_key: str) -> StepResult:
    values, missing = _copy_state(values, missing)
    decisions: list[dict] = []
    if step.name == "normalize":
        _exec_normalize(step, values, missing)
    elif step.name == "cross":
        _exec_cross(step, values, missing, decisions)
    elif step.name == "bucket":
        _exec_bucket(step, values, missing, decisions)
    else:
        raise ValueError(f"未知步骤类型: {step.name}")
    return StepResult(index=index, name=step.name, cache_key=cache_key,
                      values=values, missing=missing, decisions=decisions,
                      summary=_summarize(values, missing))


def _exec_normalize(step: Step, values, missing) -> None:
    features = sorted(step.params.get("features", []))
    for feature in features:
        present = [(e, values[e][feature]) for e in sorted(values)
                   if feature in values.get(e, {}) and feature not in missing.get(e, set())]
        if not present:
            continue
        vmin = min(v for _, v in present)
        vmax = max(v for _, v in present)
        span = vmax - vmin
        for entity, value in present:
            values[entity][feature] = 0 if span == 0 else (value - vmin) * SCALE // span


def _exec_cross(step: Step, values, missing, decisions) -> None:
    produced: dict[str, tuple[str, str]] = {}
    for pair in step.params.get("pairs", []):
        fa, fb = pair[0], pair[1]
        digest = hashlib.sha256(f"{fa}|{fb}".encode("utf-8")).hexdigest()
        name = f"cross_{int(digest[:8], 16) % 65536:04x}"
        if name in produced and produced[name] != (fa, fb):
            decisions.append({
                "type": "cross_hash_collision", "name": name,
                "kept": list(produced[name]), "dropped": [fa, fb],
                "rule": "keep_first",
            })
            continue
        produced[name] = (fa, fb)
        for entity in sorted(values):
            feats = values[entity]
            miss = missing.get(entity, set())
            if fa not in feats and fb not in feats:
                continue
            if fa in miss or fb in miss or fa not in feats or fb not in feats:
                missing.setdefault(entity, set()).add(name)
                feats.setdefault(name, 0)
                continue
            feats[name] = feats[fa] * feats[fb] // SCALE


def _exec_bucket(step: Step, values, missing, decisions) -> None:
    feature = step.params["feature"]
    output = step.params.get("output", f"{feature}_bucket")
    bounds = sorted(to_fp(float(b)) for b in step.params.get("boundaries", []))
    bound_set = set(bounds)
    for entity in sorted(values):
        feats = values[entity]
        if feature not in feats:
            continue
        if feature in missing.get(entity, set()):
            missing.setdefault(entity, set()).add(output)
            feats.setdefault(output, 0)
            continue
        value = feats[feature]
        if value in bound_set:
            decisions.append({
                "type": "bucket_boundary_equal", "entity": entity,
                "feature": feature, "value_fp": value, "rule": "lower_bucket",
            })
        feats[output] = bisect.bisect_left(bounds, value) * SCALE


@dataclass
class PipelineResult:
    step_results: list[StepResult]
    final_values: dict[str, dict[str, int]]
    final_missing: dict[str, set[str]]
    reuse: list[int]
    recompute: list[int]
    parse_key: str
    final_key: str

    def final_features_float(self) -> dict[str, dict[str, float]]:
        return {e: {f: from_fp(v) for f, v in sorted(feats.items())}
                for e, feats in sorted(self.final_values.items())}


def execute_pipeline(ir, cache, version_tag: str,
                     counter: dict | None = None) -> PipelineResult:
    """按 IR 步骤顺序执行；cache 为 None 时全量执行。

    缓存键派生规则：parse_key = H(来源内容指纹 | 版本号)；
    第 i 步 key = H(第 i-1 步 key | 步骤参数指纹 | 版本号)；
    最终特征集 key = H(末步 key | "final" | 版本号)。
    任一上游变化自动使该步骤及全部下游键失效。
    """
    def _chain(*parts: str) -> str:
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()

    parse_key = _chain("parse", ir.features.fingerprint(), version_tag)
    values = {e: {f: to_fp(v) for f, v in sorted(feats.items())}
              for e, feats in sorted(ir.features.values.items())}
    missing = {e: set(m) for e, m in ir.features.missing.items()}
    for e in values:
        missing.setdefault(e, set())

    step_results: list[StepResult] = []
    reuse: list[int] = []
    recompute: list[int] = []
    prev_key = parse_key
    for index, step in enumerate(ir.steps):
        key = _chain(prev_key, step.fingerprint(), version_tag)
        cached = cache.get(key) if cache is not None else None
        if cached is not None:
            result = StepResult(
                index=index, name=step.name, cache_key=key,
                values={e: dict(f) for e, f in cached["values"].items()},
                missing={e: set(m) for e, m in cached["missing"].items()},
                decisions=list(cached["decisions"]),
                summary=dict(cached["summary"]), reused=True)
            reuse.append(index)
        else:
            if counter is not None:
                counter[step.name] = counter.get(step.name, 0) + 1
            result = execute_step(index, step, values, missing, key)
            recompute.append(index)
            if cache is not None:
                cache.set(key, {
                    "values": result.values,
                    "missing": {e: sorted(m) for e, m in result.missing.items()},
                    "decisions": result.decisions,
                    "summary": result.summary,
                })
        values, missing = result.values, result.missing
        step_results.append(result)
        prev_key = key
    final_key = _chain(prev_key, "final", version_tag)
    return PipelineResult(step_results=step_results, final_values=values,
                          final_missing=missing, reuse=reuse, recompute=recompute,
                          parse_key=parse_key, final_key=final_key)
