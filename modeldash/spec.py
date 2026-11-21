"""Loading and strict validation for versioned workbench specifications."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import SpecError
from .util import canonical_json, digest_json


KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,63}$")
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")
SUPPORTED_SCALARS = (int, float, bool, str)
FAULT_MODES = {"drop_before_handshake", "drop_before_reply", "error_reply"}


@dataclass(frozen=True)
class ConfigRecord:
    version: int
    priority: int
    target_nodes: tuple[str, ...]
    shards: int
    items: dict[str, int | float | bool | str]
    digest: str

    def shard_items(self, shard_index: int) -> dict[str, int | float | bool | str]:
        keys = sorted(self.items)
        return {key: self.items[key] for index, key in enumerate(keys) if index % self.shards == shard_index}


@dataclass(frozen=True)
class FaultPlan:
    shard_index: int
    remaining: int
    mode: str


@dataclass(frozen=True)
class NodeRecord:
    node_id: str
    host: str
    port: int
    quota: int
    shard_delay_ms: int
    startup_timeout_ms: int
    faults: tuple[FaultPlan, ...] = ()


@dataclass(frozen=True)
class BudgetRecord:
    global_concurrency: int
    timeout_ms: int
    retry_backoff_ms: tuple[int, ...]
    max_retries: int


@dataclass(frozen=True)
class TaskRecord:
    task_id: str
    config_version: int
    node_id: str
    priority: int
    release_after_ms: int


@dataclass(frozen=True)
class Spec:
    spec_version: int
    format_version: int
    seed: int
    input_size: int
    input_scale: int
    configs: tuple[ConfigRecord, ...]
    nodes: tuple[NodeRecord, ...]
    budget: BudgetRecord
    tasks: tuple[TaskRecord, ...]
    source_path: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, compare=False)
    digest: str = ""
    budget_digest: str = ""

    @property
    def current_version(self) -> int:
        return self.configs[-1].version

    def config(self, version: int) -> ConfigRecord:
        for item in self.configs:
            if item.version == version:
                return item
        raise KeyError(version)

    def node(self, node_id: str) -> NodeRecord:
        for item in self.nodes:
            if item.node_id == node_id:
                return item
        raise KeyError(node_id)

    def task(self, task_id: str) -> TaskRecord:
        for item in self.tasks:
            if item.task_id == task_id:
                return item
        raise KeyError(task_id)

    def as_dict(self) -> dict[str, Any]:
        return self.raw


def _fail(path: str, detail: str) -> None:
    raise SpecError(path, detail)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int(value: Any, path: str, *, minimum: int | None = None, maximum: int | None = None) -> int:
    if not _is_int(value):
        _fail(path, "must be an integer")
    if minimum is not None and value < minimum:
        _fail(path, f"must be >= {minimum}")
    if maximum is not None and value > maximum:
        _fail(path, f"must be <= {maximum}")
    return value


def _port(value: Any, path: str) -> int:
    return _int(value, path, minimum=1024, maximum=65535)


def _id(value: Any, path: str) -> str:
    if not isinstance(value, str) or not ID_RE.fullmatch(value):
        _fail(path, "must match ^[A-Za-z0-9][A-Za-z0-9_.:-]{0,95}$")
    return value


def _scalar(value: Any, path: str) -> int | float | bool | str:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        if value < -(1 << 63) or value > (1 << 63) - 1:
            _fail(path, "integer must fit signed 64-bit range")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            _fail(path, "float must be finite")
        return value
    if isinstance(value, str):
        if len(value) > 256:
            _fail(path, "string length must be <= 256")
        return value
    _fail(path, "supported values are int, finite float, bool and str")
    raise AssertionError("unreachable")


def validate_spec(raw: Any) -> Spec:
    if not isinstance(raw, dict):
        _fail("$", "top-level value must be an object")

    format_version = _int(raw.get("format_version"), "format_version", minimum=1)
    if format_version != 1:
        _fail("format_version", "only format version 1 is supported")
    spec_version = _int(raw.get("spec_version"), "spec_version", minimum=1)
    seed = _int(raw.get("seed"), "seed", minimum=0, maximum=(1 << 63) - 1)

    input_obj = raw.get("input")
    if not isinstance(input_obj, dict):
        _fail("input", "must be an object")
    input_size = _int(input_obj.get("size"), "input.size", minimum=1, maximum=256)
    input_scale = _int(input_obj.get("scale"), "input.scale", minimum=1, maximum=1_000_000)

    nodes_raw = raw.get("nodes")
    if not isinstance(nodes_raw, list) or not nodes_raw:
        _fail("nodes", "must be a non-empty array")
    nodes: list[NodeRecord] = []
    node_ids: set[str] = set()
    ports: set[int] = set()
    for index, node_raw in enumerate(nodes_raw):
        path = f"nodes[{index}]"
        if not isinstance(node_raw, dict):
            _fail(path, "must be an object")
        node_id = _id(node_raw.get("id"), f"{path}.id")
        if node_id in node_ids:
            _fail(f"{path}.id", "node id must be globally unique")
        node_ids.add(node_id)
        host = node_raw.get("host")
        if host != "127.0.0.1":
            _fail(f"{path}.host", "must be the literal 127.0.0.1")
        port = _port(node_raw.get("port"), f"{path}.port")
        if port in ports:
            _fail(f"{path}.port", "port must be globally unique")
        ports.add(port)
        quota = _int(node_raw.get("quota"), f"{path}.quota", minimum=1, maximum=64)
        shard_delay_ms = _int(node_raw.get("shard_delay_ms", 0), f"{path}.shard_delay_ms", minimum=0, maximum=60_000)
        startup_timeout_ms = _int(
            node_raw.get("startup_timeout_ms", 3_000),
            f"{path}.startup_timeout_ms",
            minimum=50,
            maximum=120_000,
        )
        faults_raw = node_raw.get("faults", [])
        if not isinstance(faults_raw, list):
            _fail(f"{path}.faults", "must be an array")
        faults: list[FaultPlan] = []
        for fault_index, fault_raw in enumerate(faults_raw):
            fault_path = f"{path}.faults[{fault_index}]"
            if not isinstance(fault_raw, dict):
                _fail(fault_path, "must be an object")
            shard_index = _int(fault_raw.get("shard_index"), f"{fault_path}.shard_index", minimum=0)
            remaining = _int(fault_raw.get("remaining"), f"{fault_path}.remaining", minimum=1)
            mode = fault_raw.get("mode")
            if mode not in FAULT_MODES:
                _fail(f"{fault_path}.mode", f"must be one of {sorted(FAULT_MODES)}")
            faults.append(FaultPlan(shard_index, remaining, mode))
        nodes.append(
            NodeRecord(
                node_id=node_id,
                host=host,
                port=port,
                quota=quota,
                shard_delay_ms=shard_delay_ms,
                startup_timeout_ms=startup_timeout_ms,
                faults=tuple(faults),
            )
        )

    configs_raw = raw.get("configs")
    if not isinstance(configs_raw, list) or not configs_raw:
        _fail("configs", "must be a non-empty array")
    configs: list[ConfigRecord] = []
    seen_versions: set[int] = set()
    previous_version: int | None = None
    for index, config_raw in enumerate(configs_raw):
        path = f"configs[{index}]"
        if not isinstance(config_raw, dict):
            _fail(path, "must be an object")
        version = _int(config_raw.get("version"), f"{path}.version", minimum=1)
        if version in seen_versions:
            _fail(f"{path}.version", "config version must be unique")
        if previous_version is not None and version <= previous_version:
            _fail(f"{path}.version", "config versions must be strictly increasing")
        seen_versions.add(version)
        previous_version = version
        priority = _int(config_raw.get("priority"), f"{path}.priority", minimum=0)
        shards = _int(config_raw.get("shards"), f"{path}.shards", minimum=1, maximum=1024)
        target_nodes_raw = config_raw.get("target_nodes")
        if not isinstance(target_nodes_raw, list) or not target_nodes_raw:
            _fail(f"{path}.target_nodes", "must be a non-empty array")
        target_nodes: list[str] = []
        for target_index, target in enumerate(target_nodes_raw):
            target_path = f"{path}.target_nodes[{target_index}]"
            target_id = _id(target, target_path)
            if target_id not in node_ids:
                _fail(target_path, f"unknown node id {target_id!r}")
            if target_id in target_nodes:
                _fail(target_path, "target node must not be duplicated")
            target_nodes.append(target_id)
        items_raw = config_raw.get("items")
        if not isinstance(items_raw, dict) or not items_raw:
            _fail(f"{path}.items", "must be a non-empty object")
        items: dict[str, int | float | bool | str] = {}
        for key in sorted(items_raw):
            key_path = f"{path}.items.{key}"
            if not KEY_RE.fullmatch(key):
                _fail(key_path, "invalid configuration item key")
            items[key] = _scalar(items_raw[key], key_path)
        configs.append(
            ConfigRecord(
                version=version,
                priority=priority,
                target_nodes=tuple(target_nodes),
                shards=shards,
                items=items,
                digest=digest_json({"version": version, "items": items, "shards": shards}),
            )
        )

    for index, config in enumerate(configs):
        for target in config.target_nodes:
            if target not in node_ids:
                _fail(f"configs[{index}].target_nodes", f"unknown node id {target!r}")

    budget_raw = raw.get("budget")
    if not isinstance(budget_raw, dict):
        _fail("budget", "must be an object")
    global_concurrency = _int(budget_raw.get("global_concurrency"), "budget.global_concurrency", minimum=1, maximum=256)
    timeout_ms = _int(budget_raw.get("timeout_ms"), "budget.timeout_ms", minimum=1, maximum=120_000)
    backoff_raw = budget_raw.get("retry_backoff_ms")
    if not isinstance(backoff_raw, list) or not backoff_raw:
        _fail("budget.retry_backoff_ms", "must be a non-empty array")
    backoff = tuple(
        _int(value, f"budget.retry_backoff_ms[{index}]", minimum=0, maximum=60_000)
        for index, value in enumerate(backoff_raw)
    )
    max_retries = budget_raw.get("max_retries", len(backoff))
    max_retries = _int(max_retries, "budget.max_retries", minimum=0, maximum=len(backoff))
    budget = BudgetRecord(
        global_concurrency=global_concurrency,
        timeout_ms=timeout_ms,
        retry_backoff_ms=backoff,
        max_retries=max_retries,
    )

    tasks_raw = raw.get("tasks")
    if not isinstance(tasks_raw, list) or not tasks_raw:
        _fail("tasks", "must be a non-empty array")
    tasks: list[TaskRecord] = []
    task_ids: set[str] = set()
    config_by_version = {config.version: config for config in configs}
    for index, task_raw in enumerate(tasks_raw):
        path = f"tasks[{index}]"
        if not isinstance(task_raw, dict):
            _fail(path, "must be an object")
        task_id = _id(task_raw.get("id"), f"{path}.id")
        if task_id in task_ids:
            _fail(f"{path}.id", "task id must be globally unique")
        task_ids.add(task_id)
        config_version = _int(task_raw.get("config_version"), f"{path}.config_version", minimum=1)
        config = config_by_version.get(config_version)
        if config is None:
            _fail(f"{path}.config_version", f"unknown config version {config_version}")
        node_id = _id(task_raw.get("node"), f"{path}.node")
        if node_id not in node_ids:
            _fail(f"{path}.node", f"unknown node id {node_id!r}")
        if node_id not in config.target_nodes:
            _fail(f"{path}.node", f"node {node_id!r} is not targeted by config version {config_version}")
        priority = _int(task_raw.get("priority"), f"{path}.priority", minimum=0)
        release_after_ms = _int(task_raw.get("release_after_ms", 0), f"{path}.release_after_ms", minimum=0, maximum=600_000)
        tasks.append(TaskRecord(task_id, config_version, node_id, priority, release_after_ms))

    budget_digest = digest_json(
        {
            "global_concurrency": budget.global_concurrency,
            "timeout_ms": budget.timeout_ms,
            "retry_backoff_ms": list(budget.retry_backoff_ms),
            "max_retries": budget.max_retries,
        }
    )
    digest = digest_json(raw)
    return Spec(
        spec_version=spec_version,
        format_version=format_version,
        seed=seed,
        input_size=input_size,
        input_scale=input_scale,
        configs=tuple(configs),
        nodes=tuple(nodes),
        budget=budget,
        tasks=tuple(tasks),
        raw=raw,
        digest=digest,
        budget_digest=budget_digest,
    )


def load_spec(path: str | Path) -> Spec:
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except FileNotFoundError:
        _fail("$", f"spec file not found: {source}")
    except json.JSONDecodeError as exc:
        _fail("$", f"invalid JSON at line {exc.lineno} column {exc.colno}: {exc.msg}")
    spec = validate_spec(raw)
    object.__setattr__(spec, "source_path", str(source))
    return spec


def spec_to_canonical_json(spec: Spec) -> str:
    return canonical_json(spec.raw)
