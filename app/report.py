"""互操作诊断报告：跨版本偏差直方图、降级损失、复用/重算与失败清单。"""
from __future__ import annotations

import json
from pathlib import Path

from . import formats
from .ir import IR, FeatureSet

BINS = ["0", "(0,1e-6]", "(1e-6,1e-3]", ">1e-3"]


def _bin(value: float) -> str:
    if value == 0:
        return "0"
    if value <= 1e-6:
        return "(0,1e-6]"
    if value <= 1e-3:
        return "(1e-6,1e-3]"
    return ">1e-3"


def cross_version_deviation(final_features: dict[str, dict[str, float]],
                            final_missing: dict[str, set[str]],
                            steps, name_map, report_failures) -> dict:
    """把最终特征集序列化为三个版本再解析回来，逐实体统计偏差直方图。"""
    temp = IR(
        features=FeatureSet(
            values={e: dict(f) for e, f in final_features.items()},
            missing={e: set(m) for e, m in final_missing.items()},
            precision={e: {f: "double" for f in feats}
                       for e, feats in final_features.items()},
        ),
        steps=list(steps),
        source_version="v2",
    )
    per_entity: dict[str, dict] = {}
    versions = {
        "v1": lambda: formats.parse_v1(formats.serialize_v1(temp), report_failures),
        "v2": lambda: formats.parse_v2(formats.serialize_v2(temp), report_failures),
        "v3": lambda: formats.parse_v3(
            formats.serialize_v3(temp, name_map, report_failures),
            name_map, report_failures),
    }
    for version, loader in versions.items():
        back = loader()
        for entity in sorted(final_features):
            slot = per_entity.setdefault(entity, {
                "hist": {b: 0 for b in BINS},
                "max_deviation": {"feature": None, "value": 0.0, "version": None},
            })
            back_feats = back.features.values.get(entity, {})
            for feature, orig in sorted(final_features[entity].items()):
                if feature in final_missing.get(entity, set()):
                    continue
                if feature not in back_feats:
                    continue
                dev = abs(back_feats[feature] - orig)
                slot["hist"][_bin(dev)] += 1
                if dev > slot["max_deviation"]["value"]:
                    slot["max_deviation"] = {
                        "feature": feature, "value": dev, "version": version}
    return per_entity


def build_report(runner) -> dict:
    res = runner.last_result
    deviation = {}
    if res is not None:
        deviation = cross_version_deviation(
            res.final_features_float(), res.final_missing,
            runner.ir.steps, runner.name_map, runner.failures)
    return {
        "snapshot_status": runner.snapshot_status,
        "target_version": runner.target_version,
        "global_version": runner.global_version,
        "injected_hits": sorted(runner.injector.hits),
        "cross_version_deviation": deviation,
        "downgrade_ledger": runner.ledger,
        "reuse": sorted(res.reuse) if res else [],
        "recompute": sorted(res.recompute) if res else [],
        "step_execution_counts": runner.last_counter,
        "decisions": runner.decisions,
        "failures": [f.to_dict() for f in runner.failures],
    }


def write_report(runner, path: Path) -> dict:
    report = build_report(runner)
    path = Path(path)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=1,
                              sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return report
