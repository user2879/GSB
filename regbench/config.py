"""实验配置数据结构与全量校验。

校验采用“收集而非中断”的语义：每个非法条目都生成一条 E_CONFIG_INVALID
失败记录，随后继续校验其余条目，最后统一返回配置与失败清单。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from typing import Any

from . import yaml_lite
from .errors import E_CONFIG_INVALID, FailureRecord
from .util import fingerprint

SUPPORTED_SCHEMES = ("ridge", "lasso", "elasticnet")


@dataclass(frozen=True)
class NumericColumn:
    name: str
    mean: float
    std: float


@dataclass(frozen=True)
class CategoricalColumn:
    name: str
    cardinality: int


@dataclass(frozen=True)
class RedundantColumn:
    name: str
    source: str
    coefficient: float
    noise: float


@dataclass(frozen=True)
class TargetSpec:
    name: str
    noise_std: float


@dataclass(frozen=True)
class DataSpec:
    n_samples: int
    base_seed: int
    split_ratio: float
    missing_rate: float
    target: TargetSpec
    numeric: list[NumericColumn]
    categorical: list[CategoricalColumn]
    redundant: list[RedundantColumn]


@dataclass(frozen=True)
class OptimizerSpec:
    learning_rate: float
    max_epochs: int
    tol: float
    batch_size: int


@dataclass(frozen=True)
class SchedulerSpec:
    max_workers: int
    global_epoch_budget: int
    per_scheme_epoch_quota: int
    snapshot_interval: int


@dataclass(frozen=True)
class SchemeSpec:
    name: str
    penalties: tuple[float, ...]
    l1_ratios: tuple[float, ...]


@dataclass
class ExperimentConfig:
    name: str
    data: DataSpec
    optimizer: OptimizerSpec
    scheduler: SchedulerSpec
    schemes: list[SchemeSpec]
    seeds: tuple[int, ...]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)

    @property
    def config_hash(self) -> str:
        # 并发数属于执行资源而非实验语义，不参与配置哈希/缓存指纹，
        # 从而同一实验在不同 max_workers 下哈希一致、缓存可复用。
        semantic = copy.deepcopy(self.raw)
        if isinstance(semantic.get("scheduler"), dict):
            semantic["scheduler"].pop("max_workers", None)
        return fingerprint(semantic)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data": asdict(self.data),
            "optimizer": asdict(self.optimizer),
            "scheduler": asdict(self.scheduler),
            "schemes": [asdict(s) for s in self.schemes],
            "seeds": list(self.seeds),
        }


def load_config(path: str) -> tuple[ExperimentConfig | None, list[FailureRecord]]:
    try:
        raw = yaml_lite.load_file(path)
    except OSError as exc:
        return None, [
            FailureRecord(
                E_CONFIG_INVALID, "config_load", f"无法读取配置: {exc}", subject=path
            )
        ]
    except ValueError as exc:
        return None, [
            FailureRecord(
                E_CONFIG_INVALID, "config_load", f"YAML 解析失败: {exc}", subject=path
            )
        ]
    return parse_config(raw)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _as_float(value: Any, default: float) -> float:
    return float(value) if _is_number(value) else default


def parse_config(raw: dict[str, Any]) -> tuple[ExperimentConfig | None, list[FailureRecord]]:
    failures: list[FailureRecord] = []

    def invalid(message: str, subject: str | None = None) -> None:
        failures.append(
            FailureRecord(
                code=E_CONFIG_INVALID,
                stage="config_validation",
                message=message,
                subject=subject,
            )
        )

    if not isinstance(raw, dict):
        invalid("配置根节点必须是映射")
        return None, failures

    name = str(raw.get("experiment", {}).get("name", "unnamed")) if isinstance(
        raw.get("experiment"), dict
    ) else "unnamed"

    # ---- 数据段 -----------------------------------------------------------
    data_raw = raw.get("data")
    if not isinstance(data_raw, dict):
        invalid("缺少 data 映射段")
        data_raw = {}

    n_samples = data_raw.get("n_samples")
    if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples <= 0:
        invalid(f"n_samples 必须为正整数，实际: {n_samples!r}", "data.n_samples")
        n_samples = 200

    base_seed = data_raw.get("base_seed")
    if not isinstance(base_seed, int) or isinstance(base_seed, bool):
        invalid(f"base_seed 必须为整数，实际: {base_seed!r}", "data.base_seed")
        base_seed = 0

    split_ratio = data_raw.get("split_ratio")
    if not _is_number(split_ratio) or not 0.0 < float(split_ratio) < 1.0:
        invalid(
            f"split_ratio 必须位于 (0,1)，实际: {split_ratio!r}", "data.split_ratio"
        )
        split_ratio = 0.75

    missing_rate = data_raw.get("missing_rate", 0.0)
    if not _is_number(missing_rate) or not 0.0 <= float(missing_rate) < 1.0:
        invalid(
            f"missing_rate 必须位于 [0,1)，实际: {missing_rate!r}",
            "data.missing_rate",
        )
        missing_rate = 0.0

    target_raw = data_raw.get("target")
    target_name = ""
    target_noise = 0.1
    if not isinstance(target_raw, dict) or not target_raw.get("name"):
        invalid("目标列缺失：data.target.name 为空", "data.target")
        target_name = "target"
    else:
        target_name = str(target_raw["name"])
        target_noise = _as_float(target_raw.get("noise_std", 0.1), 0.1)
        if target_noise < 0:
            invalid("目标噪声不能为负", "data.target.noise_std")
            target_noise = 0.1
    target = TargetSpec(target_name, target_noise)

    numeric: list[NumericColumn] = []
    numeric_names: set[str] = set()
    for item in data_raw.get("numeric", []) or []:
        if not isinstance(item, dict) or not item.get("name"):
            invalid(f"数值列描述非法: {item!r}", "data.numeric")
            continue
        col_name = str(item["name"])
        mean = _as_float(item.get("mean", 0.0), 0.0)
        std = item.get("std", 1.0)
        if not _is_number(std) or float(std) < 0:
            invalid(f"数值列 {col_name} 的 std 类型不符或为负: {std!r}", col_name)
            std = 1.0
        numeric.append(NumericColumn(col_name, mean, float(std)))
        numeric_names.add(col_name)

    categorical: list[CategoricalColumn] = []
    categorical_names: set[str] = set()
    for item in data_raw.get("categorical", []) or []:
        if not isinstance(item, dict) or not item.get("name"):
            invalid(f"类别列描述非法: {item!r}", "data.categorical")
            continue
        col_name = str(item["name"])
        cardinality = item.get("cardinality")
        if not isinstance(cardinality, int) or isinstance(cardinality, bool) or cardinality <= 0:
            invalid(
                f"类别列 {col_name} 的 cardinality 必须为正整数，实际: {cardinality!r}",
                col_name,
            )
            cardinality = 2
        categorical.append(CategoricalColumn(col_name, cardinality))
        categorical_names.add(col_name)

    known_columns = numeric_names | categorical_names
    redundant: list[RedundantColumn] = []
    for item in data_raw.get("redundant", []) or []:
        if not isinstance(item, dict) or not item.get("name"):
            invalid(f"冗余列描述非法: {item!r}", "data.redundant")
            continue
        col_name = str(item["name"])
        source = item.get("source")
        if source not in known_columns:
            invalid(
                f"未知列：冗余列 {col_name} 引用了不存在或类型不符的源列 {source!r}",
                col_name,
            )
            continue
        coefficient = item.get("coefficient", 1.0)
        noise = item.get("noise", 0.05)
        if not _is_number(coefficient):
            invalid(f"冗余列 {col_name} coefficient 类型不符: {coefficient!r}", col_name)
            coefficient = 1.0
        if not _is_number(noise) or float(noise) < 0:
            invalid(f"冗余列 {col_name} noise 类型不符或为负: {noise!r}", col_name)
            noise = 0.05
        redundant.append(
            RedundantColumn(col_name, str(source), float(coefficient), float(noise))
        )

    data = DataSpec(
        n_samples=n_samples,
        base_seed=base_seed,
        split_ratio=float(split_ratio),
        missing_rate=float(missing_rate),
        target=target,
        numeric=numeric,
        categorical=categorical,
        redundant=redundant,
    )

    # ---- 优化器段 ---------------------------------------------------------
    opt_raw = raw.get("optimizer", {})
    if not isinstance(opt_raw, dict):
        invalid("optimizer 必须为映射段")
        opt_raw = {}
    learning_rate = opt_raw.get("learning_rate")
    if not _is_number(learning_rate) or float(learning_rate) < 0:
        invalid(
            f"学习率不能为负或类型不符，实际: {learning_rate!r}",
            "optimizer.learning_rate",
        )
        learning_rate = 0.01
    max_epochs = opt_raw.get("max_epochs")
    if not isinstance(max_epochs, int) or isinstance(max_epochs, bool) or max_epochs <= 0:
        invalid(f"max_epochs 必须为正整数，实际: {max_epochs!r}", "optimizer.max_epochs")
        max_epochs = 100
    tol = opt_raw.get("tol", 1e-6)
    if not _is_number(tol) or float(tol) < 0:
        invalid(f"收敛容差不能为负，实际: {tol!r}", "optimizer.tol")
        tol = 1e-6
    batch_size = opt_raw.get("batch_size")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        invalid(f"批量大小必须为正整数，实际: {batch_size!r}", "optimizer.batch_size")
        batch_size = 16
    optimizer = OptimizerSpec(
        float(learning_rate), max_epochs, float(tol), batch_size
    )

    # ---- 调度器段 ---------------------------------------------------------
    sch_raw = raw.get("scheduler", {})
    if not isinstance(sch_raw, dict):
        invalid("scheduler 必须为映射段")
        sch_raw = {}
    max_workers = sch_raw.get("max_workers")
    if not isinstance(max_workers, int) or isinstance(max_workers, bool) or max_workers <= 0:
        invalid(f"并发数必须为正整数，实际: {max_workers!r}", "scheduler.max_workers")
        max_workers = 2
    global_budget = sch_raw.get("global_epoch_budget")
    if not isinstance(global_budget, int) or isinstance(global_budget, bool) or global_budget <= 0:
        invalid(
            f"全局轮次预算必须为正整数，实际: {global_budget!r}",
            "scheduler.global_epoch_budget",
        )
        global_budget = 10_000
    quota = sch_raw.get("per_scheme_epoch_quota")
    if not isinstance(quota, int) or isinstance(quota, bool) or quota <= 0:
        invalid(
            f"单方案轮次配额必须为正整数，实际: {quota!r}",
            "scheduler.per_scheme_epoch_quota",
        )
        quota = max_epochs
    interval = sch_raw.get("snapshot_interval", 5)
    if not isinstance(interval, int) or isinstance(interval, bool) or interval <= 0:
        invalid(
            f"快照提交间隔必须为正整数，实际: {interval!r}",
            "scheduler.snapshot_interval",
        )
        interval = 5
    scheduler = SchedulerSpec(max_workers, global_budget, quota, interval)

    # ---- 正则化方案段 -----------------------------------------------------
    schemes: list[SchemeSpec] = []
    seen_scheme_names: set[str] = set()
    for item in raw.get("schemes", []) or []:
        if not isinstance(item, dict) or not item.get("name"):
            invalid(f"方案描述非法: {item!r}", "schemes")
            continue
        scheme_name = str(item["name"])
        if scheme_name not in SUPPORTED_SCHEMES:
            invalid(
                f"未知正则化方案 {scheme_name!r}，支持: {SUPPORTED_SCHEMES}",
                "schemes.name",
            )
            continue
        if scheme_name in seen_scheme_names:
            invalid(f"方案重复定义: {scheme_name}", "schemes.name")
            continue
        penalties = item.get("penalties", []) or []
        clean_penalties: list[float] = []
        for penalty in penalties:
            if not _is_number(penalty) or float(penalty) < 0:
                invalid(
                    f"方案 {scheme_name} 惩罚系数不能为负或类型不符: {penalty!r}",
                    f"schemes.{scheme_name}.penalties",
                )
                continue
            clean_penalties.append(float(penalty))
        if not clean_penalties:
            invalid(f"方案 {scheme_name} 没有合法惩罚系数", f"schemes.{scheme_name}")
            continue
        ratios_raw = item.get("l1_ratios", []) or []
        if scheme_name == "elasticnet":
            if not ratios_raw:
                invalid("弹性网络必须提供 l1_ratios 候选", "schemes.elasticnet")
                ratios_raw = [0.5]
        else:
            ratios_raw = [0.0]
        clean_ratios: list[float] = []
        for ratio in ratios_raw:
            if not _is_number(ratio) or not 0.0 <= float(ratio) <= 1.0:
                invalid(
                    f"方案 {scheme_name} 混合比例必须位于 [0,1]: {ratio!r}",
                    f"schemes.{scheme_name}.l1_ratios",
                )
                continue
            clean_ratios.append(float(ratio))
        if not clean_ratios:
            clean_ratios = [0.0]
        schemes.append(
            SchemeSpec(
                scheme_name,
                tuple(clean_penalties),
                tuple(clean_ratios),
            )
        )
        seen_scheme_names.add(scheme_name)
    if not schemes:
        invalid("没有任何合法正则化方案", "schemes")

    seeds_raw = raw.get("seeds", []) or []
    seeds: list[int] = []
    for seed in seeds_raw:
        if not isinstance(seed, int) or isinstance(seed, bool):
            invalid(f"随机种子必须为整数: {seed!r}", "seeds")
            continue
        seeds.append(seed)
    if not seeds:
        invalid("种子列表为空，回退为 [0]", "seeds")
        seeds = [0]

    if failures:
        # 仍返回尽力解析的配置，调用方可决定是否中止；关键非法已全部入清单。
        config = ExperimentConfig(
            name, data, optimizer, scheduler, schemes, tuple(seeds), raw
        )
        return config, failures

    config = ExperimentConfig(
        name, data, optimizer, scheduler, schemes, tuple(seeds), raw
    )
    return config, failures
