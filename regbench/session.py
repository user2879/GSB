"""交互命令会话与精确失效判定。

四类命令：
- switch <scheme>                切换当前方案视图（不失效任何结果）
- set-penalty <scheme> <value>   修改惩罚系数（仅失效该方案）
- set-ratio <scheme> <value>     修改弹性网络混合比例（仅失效该方案）
- set-lr <value>                 修改学习率（失效所有方案的训练缓存）
- set-workers <n>                调整并发数（不失效数值结果）
- show / replay / status         查看与回放辅助

缓存键由 TaskSpec 参数指纹 + 数据指纹派生；未命中且参数变化的任务进入
重算集合，命中的任务直接复用已持久化结果，绝不重新训练。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from . import grid
from .config import ExperimentConfig, SchemeSpec
from .data import prepare_data
from .errors import E_CONFIG_INVALID, FailureLog
from .models import make_cache_key, make_task_spec
from .runner import run_training
from .snapshot import SnapshotStore
from .util import fingerprint


@dataclass
class CommandTrace:
    raw: str
    command: str
    args: list[str]
    effect: str
    invalidated: list[str] = field(default_factory=list)
    recompute: list[str] = field(default_factory=list)
    reused: list[str] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "command": self.command,
            "args": self.args,
            "effect": self.effect,
            "invalidated": self.invalidated,
            "recompute": self.recompute,
            "reused": self.reused,
            "error": self.error,
        }


class InteractiveSession:
    def __init__(self, config: ExperimentConfig, snapshot_dir: str):
        self.base_config = config
        self.config = _clone_config(config)
        self.store = SnapshotStore(snapshot_dir)
        self.active_scheme = config.schemes[0].name if config.schemes else ""
        self.trace: list[CommandTrace] = []
        self.invalidated_keys: set[str] = set()
        self.reused_keys: list[str] = []
        self.recomputed_keys: list[str] = []
        self._last_outcome = None
        self._data = prepare_data(self.config)
        overlay = self.store.load_session_overlay()
        if overlay is not None:
            restored, failures = _parse_overlay(overlay, config)
            if restored is not None:
                self.config = restored
                self.active_scheme = overlay.get("active_scheme", self.active_scheme)
                self._data = prepare_data(self.config)

    # ---- 命令解析 ---------------------------------------------------------
    def execute_line(self, line: str) -> CommandTrace:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            return CommandTrace(stripped, "noop", [], "ignored")
        parts = stripped.split()
        command, args = parts[0], parts[1:]
        trace = CommandTrace(raw=stripped, command=command, args=args, effect="")
        try:
            if command == "switch":
                self._switch(args, trace)
            elif command == "set-penalty":
                self._set_penalty(args, trace)
            elif command == "set-ratio":
                self._set_ratio(args, trace)
            elif command == "set-lr":
                self._set_lr(args, trace)
            elif command == "set-workers":
                self._set_workers(args, trace)
            elif command == "status":
                trace.effect = self._status_text()
            elif command == "show":
                trace.effect = self._show_text()
            elif command == "replay":
                trace.effect = "replay 命令只能由 replay-file 入口驱动"
            else:
                raise ValueError(f"未知命令: {command}")
        except ValueError as exc:
            trace.error = str(exc)
            trace.effect = "rejected"
        self.trace.append(trace)
        return trace

    def replay_file(self, path: str) -> list[CommandTrace]:
        with open(path, "r", encoding="utf-8") as handle:
            lines = handle.read().splitlines()
        return [self.execute_line(line) for line in lines]

    # ---- 命令实现 ---------------------------------------------------------
    def _require_scheme(self, scheme: str) -> SchemeSpec:
        for candidate in self.config.schemes:
            if candidate.name == scheme:
                return candidate
        raise ValueError(f"未知方案: {scheme}")

    def _switch(self, args: list[str], trace: CommandTrace) -> None:
        if len(args) != 1:
            raise ValueError("用法: switch <ridge|lasso|elasticnet>")
        scheme = args[0]
        self._require_scheme(scheme)
        self.active_scheme = scheme
        trace.effect = f"active_scheme={scheme}"
        trace.invalidated = []

    def _set_penalty(self, args: list[str], trace: CommandTrace) -> None:
        if len(args) != 2:
            raise ValueError("用法: set-penalty <scheme> <penalty>")
        scheme, value = args[0], float(args[1])
        if value < 0:
            raise ValueError(f"{E_CONFIG_INVALID}: 惩罚系数不能为负")
        self._require_scheme(scheme)
        before = self._cache_keys_for_scheme(scheme)
        self._replace_scheme(scheme, penalties=(value,))
        after = self._cache_keys_for_scheme(scheme)
        self._mark_invalidated(before - after, trace, scheme)

    def _set_ratio(self, args: list[str], trace: CommandTrace) -> None:
        if len(args) != 2:
            raise ValueError("用法: set-ratio <elasticnet> <ratio>")
        scheme, value = args[0], float(args[1])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{E_CONFIG_INVALID}: 混合比例必须位于 [0,1]")
        self._require_scheme(scheme)
        before = self._cache_keys_for_scheme(scheme)
        self._replace_scheme(scheme, l1_ratios=(value,))
        after = self._cache_keys_for_scheme(scheme)
        self._mark_invalidated(before - after, trace, scheme)

    def _set_lr(self, args: list[str], trace: CommandTrace) -> None:
        if len(args) != 1:
            raise ValueError("用法: set-lr <learning_rate>")
        value = float(args[0])
        if value < 0:
            raise ValueError(f"{E_CONFIG_INVALID}: 学习率不能为负")
        before = self._all_cache_keys()
        self.config = _with_learning_rate(self.config, value)
        after = self._all_cache_keys()
        self._mark_invalidated(before - after, trace, "*")

    def _set_workers(self, args: list[str], trace: CommandTrace) -> None:
        if len(args) != 1:
            raise ValueError("用法: set-workers <n>")
        value = int(args[0])
        if value <= 0:
            raise ValueError(f"{E_CONFIG_INVALID}: 并发数必须为正整数")
        self.config = _with_workers(self.config, value)
        trace.effect = f"max_workers={value}（数值缓存不失效）"

    def _replace_scheme(
        self,
        scheme: str,
        penalties: tuple[float, ...] | None = None,
        l1_ratios: tuple[float, ...] | None = None,
    ) -> None:
        new_schemes: list[SchemeSpec] = []
        for candidate in self.config.schemes:
            if candidate.name == scheme:
                new_schemes.append(
                    SchemeSpec(
                        name=candidate.name,
                        penalties=penalties if penalties is not None else candidate.penalties,
                        l1_ratios=l1_ratios if l1_ratios is not None else candidate.l1_ratios,
                    )
                )
            else:
                new_schemes.append(candidate)
        self.config = _with_schemes(self.config, new_schemes)

    def _mark_invalidated(
        self, keys: set[str], trace: CommandTrace, scheme: str
    ) -> None:
        self.invalidated_keys |= keys
        trace.invalidated = sorted(keys)
        trace.effect = f"invalidated_scheme={scheme}"

    # ---- 缓存键与重算集合 -------------------------------------------------
    def _cache_keys_for_scheme(self, scheme: str) -> set[str]:
        keys: set[str] = set()
        for candidate in self.config.schemes:
            if candidate.name != scheme:
                continue
            ratios = candidate.l1_ratios if scheme == "elasticnet" else (0.0,)
            for penalty in candidate.penalties:
                for ratio in ratios:
                    spec = make_task_spec(scheme, penalty, ratio, self.config)
                    keys.add(make_cache_key(spec, self._data.data_fingerprint))
        return keys

    def _all_cache_keys(self) -> set[str]:
        keys: set[str] = set()
        for candidate in self.config.schemes:
            keys |= self._cache_keys_for_scheme(candidate.name)
        return keys

    def run(self) -> dict[str, Any]:
        """按当前配置运行；命中缓存的任务直接复用，未命中任务重算。"""

        data = prepare_data(self.config)
        tasks = grid.expand_grid(self.config, data)
        cache_by_key: dict[str, Any] = {}
        for file_name in self.store.list_result_keys():
            result = _load_result_by_file(self.store, file_name)
            if result is not None:
                cache_by_key[result.cache_key] = result
        reused: list[str] = []
        recompute: list[str] = []
        for task in tasks:
            if task.cache_key in cache_by_key:
                reused.append(task.task_key)
            else:
                recompute.append(task.task_key)
        failure_log = FailureLog()
        outcome = run_training(self.config, self.store, failure_log=failure_log)
        self.store.save_session_overlay(self.config.raw, self.active_scheme)
        self._last_outcome = outcome
        self.reused_keys = reused
        self.recomputed_keys = recompute
        if self.trace:
            self.trace[-1].reused = reused
            self.trace[-1].recompute = recompute
        return {
            "reused": reused,
            "recomputed": recompute,
            "invalidated": sorted(self.invalidated_keys),
            "outcome": outcome,
        }

    def cache_stats(self) -> dict[str, Any]:
        return {
            "reused": list(self.reused_keys),
            "recomputed": list(self.recomputed_keys),
            "invalidated": sorted(self.invalidated_keys),
        }

    # ---- 展示 -------------------------------------------------------------
    def _status_text(self) -> str:
        return (
            f"active={self.active_scheme}; workers={self.config.scheduler.max_workers}; "
            f"lr={self.config.optimizer.learning_rate}; "
            f"invalidated={len(self.invalidated_keys)}"
        )

    def _show_text(self) -> str:
        rows = []
        for scheme in self.config.schemes:
            ratios = scheme.l1_ratios if scheme.name == "elasticnet" else [0.0]
            rows.append(
                f"{scheme.name}: penalties={list(scheme.penalties)} ratios={list(ratios)}"
            )
        return " | ".join(rows)


def _clone_config(config: ExperimentConfig) -> ExperimentConfig:
    return ExperimentConfig(
        name=config.name,
        data=config.data,
        optimizer=config.optimizer,
        scheduler=config.scheduler,
        schemes=[
            SchemeSpec(item.name, item.penalties, item.l1_ratios)
            for item in config.schemes
        ],
        seeds=config.seeds,
        raw=copy.deepcopy(config.raw),
    )


def _with_learning_rate(config: ExperimentConfig, value: float) -> ExperimentConfig:
    clone = _clone_config(config)
    clone.optimizer = type(config.optimizer)(
        value,
        config.optimizer.max_epochs,
        config.optimizer.tol,
        config.optimizer.batch_size,
    )
    clone.raw = copy.deepcopy(clone.raw)
    clone.raw.setdefault("optimizer", {})["learning_rate"] = value
    return clone


def _with_workers(config: ExperimentConfig, value: int) -> ExperimentConfig:
    clone = _clone_config(config)
    clone.scheduler = type(config.scheduler)(
        value,
        config.scheduler.global_epoch_budget,
        config.scheduler.per_scheme_epoch_quota,
        config.scheduler.snapshot_interval,
    )
    clone.raw = copy.deepcopy(clone.raw)
    clone.raw.setdefault("scheduler", {})["max_workers"] = value
    return clone


def _with_schemes(
    config: ExperimentConfig, schemes: list[SchemeSpec]
) -> ExperimentConfig:
    clone = _clone_config(config)
    clone.schemes = schemes
    raw = copy.deepcopy(clone.raw)
    raw["schemes"] = [
        {
            "name": item.name,
            "penalties": list(item.penalties),
            "l1_ratios": list(item.l1_ratios),
        }
        for item in schemes
    ]
    clone.raw = raw
    return clone


def _load_result_by_file(store, file_name):
    import json
    import os

    from .snapshot import task_result_from_dict

    path = os.path.join(store.train_dir, file_name)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return task_result_from_dict(json.load(handle))
    except (OSError, KeyError):
        return None


def _parse_overlay(overlay, fallback_config):
    from .config import parse_config

    raw = overlay.get("raw_config")
    if not isinstance(raw, dict):
        return None, []
    return parse_config(raw)
