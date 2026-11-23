"""异常注入：以独立注入配置施加真实输入故障并隔离。

四类故障（各带稳定注入项编号，映射到业务码）：
  INJ-01 非有限打分权重          -> E_CONFIG_INVALID
  INJ-02 桶数不符（buckets>dim） -> E_CONFIG_INVALID
  INJ-03 边界阈值越界（负数）    -> E_CONFIG_INVALID
  INJ-04 替换对 ID 不存在        -> E_PAIR_NOT_FOUND

注入通过“真实输入”实现（构造畸形配置/命令），而非 mock；故障被隔离进
失败清单，不影响其余对的处理顺序与结果。
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

from .commands import Command, apply_command
from .config import load_config
from .errors import FailureLog
from .pairs import PreferencePair


@dataclass(frozen=True)
class InjectionSpec:
    injection_id: str
    kind: str
    business_code: str
    description: str


INJECTION_SPECS = (
    InjectionSpec("INJ-01", "non_finite_weight", "E_CONFIG_INVALID",
                  "打分权重为 NaN/Inf（非有限）"),
    InjectionSpec("INJ-02", "bucket_mismatch", "E_CONFIG_INVALID",
                  "哈希桶数超过特征维度（桶数不符）"),
    InjectionSpec("INJ-03", "threshold_out_of_range", "E_CONFIG_INVALID",
                  "边界阈值为负数（越界）"),
    InjectionSpec("INJ-04", "replace_missing_pair", "E_PAIR_NOT_FOUND",
                  "替换/操作的对 ID 在数据中不存在"),
)


def build_injected_config(base_config_dict: dict[str, Any],
                          injection_id: str) -> dict[str, Any]:
    """返回施加了指定注入的配置 dict（真实畸形输入）。"""
    cfg = copy.deepcopy(base_config_dict)
    if injection_id == "INJ-01":
        cfg["weights"] = {"prompt": float("nan"), "chosen": 1.0, "rejected": 0.75}
    elif injection_id == "INJ-02":
        cfg["hash_buckets"] = cfg["feature_dim"] + 16
    elif injection_id == "INJ-03":
        cfg["boundary_threshold"] = -1.0
    return cfg


def build_injected_command(injection_id: str) -> Command | None:
    """INJ-04 通过命令流施加：操作一个不存在的对。"""
    if injection_id == "INJ-04":
        return Command(
            seq=9001, op="delete_pair",
            payload={"pair_id": "pair-DOES-NOT-EXIST"},
        )
    return None


def run_config_injections(config_path: str) -> list[dict[str, Any]]:
    """对配置类注入（INJ-01..03）逐一真实加载，确认命中业务码。"""
    import yaml
    with open(config_path, "r", encoding="utf-8") as fh:
        base = yaml.safe_load(fh)

    results = []
    for spec in INJECTION_SPECS:
        if spec.kind == "replace_missing_pair":
            continue
        injected = build_injected_config(base, spec.injection_id)
        tmp_path = config_path + f".{spec.injection_id}.tmp.yaml"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            yaml.safe_dump(injected, fh, allow_unicode=True, sort_keys=True)
        failures = FailureLog()
        cfg = load_config(tmp_path, failures)
        import os
        os.remove(tmp_path)
        codes = failures.codes()
        results.append({
            "injection_id": spec.injection_id,
            "kind": spec.kind,
            "expected_code": spec.business_code,
            "hit": cfg is None and spec.business_code in codes,
            "observed_codes": codes,
            "description": spec.description,
        })
    return results


def run_command_injection(pairs: dict[str, PreferencePair], cfg) -> dict[str, Any]:
    """INJ-04：对不存在的对施加命令，确认 E_PAIR_NOT_FOUND 且状态不变。"""
    spec = next(s for s in INJECTION_SPECS if s.injection_id == "INJ-04")
    failures = FailureLog()
    cmd = build_injected_command("INJ-04")
    before_ids = sorted(pairs.keys())
    new_pairs, new_cfg, changed = apply_command(
        cmd, pairs, cfg, cfg.normalize, failures,
    )
    after_ids = sorted(new_pairs.keys())
    return {
        "injection_id": spec.injection_id,
        "kind": spec.kind,
        "expected_code": spec.business_code,
        "hit": spec.business_code in failures.codes(),
        "observed_codes": failures.codes(),
        "state_unchanged": before_ids == after_ids and not changed,
        "description": spec.description,
    }
