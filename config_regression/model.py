"""Typed configuration model, JSON loading and strong manifest validation."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from typing import Any

from .errors import ManifestError, ViolationCollector

KEY_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]{0,63}$")
SUPPORTED_STRATEGIES = ("full_mesh", "canary", "staged")
SUPPORTED_FAULT_MODES = ("none", "drop_first", "drop_every", "drop_all")


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class NodeEntry:
    node_id: str
    host: str
    port: int
    work_ms_min: int
    work_ms_max: int
    fault_mode: str
    fault_arg: int


@dataclass
class Registry:
    nodes: dict[str, NodeEntry]

    @property
    def node_ids(self) -> list[str]:
        return sorted(self.nodes)


def load_registry(data: dict[str, Any]) -> Registry:
    vc = ViolationCollector()
    if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
        raise ManifestError("registry must be an object with a 'nodes' list",
                            [("registry", "missing nodes list")])
    nodes: dict[str, NodeEntry] = {}
    for index, raw in enumerate(data["nodes"]):
        subject = f"registry.nodes[{index}]"
        if not isinstance(raw, dict):
            vc.add(subject, "node entry must be an object")
            continue
        node_id = raw.get("node_id")
        if not isinstance(node_id, str) or not KEY_PATTERN.fullmatch(node_id):
            vc.add(f"{subject}.node_id", "missing or illegal node_id")
            continue
        if node_id in nodes:
            vc.add(f"{subject}.{node_id}", "duplicate node_id")
            continue
        host = raw.get("host", "127.0.0.1")
        if not isinstance(host, str) or not host:
            vc.add(f"{subject}.{node_id}.host", "host must be a non-empty string")
        port = raw.get("port", 0)
        if not isinstance(port, int) or isinstance(port, bool) or not (0 <= port <= 65535):
            vc.add(f"{subject}.{node_id}.port", "port must be an int in [0, 65535]")
        wmin = raw.get("work_ms_min", 1)
        wmax = raw.get("work_ms_max", 4)
        for name, val in (("work_ms_min", wmin), ("work_ms_max", wmax)):
            if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                vc.add(f"{subject}.{node_id}.{name}", "must be a non-negative int")
        if isinstance(wmin, int) and isinstance(wmax, int) and wmin > wmax:
            vc.add(f"{subject}.{node_id}.work_ms", "work_ms_min > work_ms_max")
        fault_mode = raw.get("fault_mode", "none")
        if fault_mode not in SUPPORTED_FAULT_MODES:
            vc.add(f"{subject}.{node_id}.fault_mode",
                   f"unsupported fault mode {fault_mode!r}")
        fault_arg = raw.get("fault_arg", 0)
        if not isinstance(fault_arg, int) or isinstance(fault_arg, bool) or fault_arg < 0:
            vc.add(f"{subject}.{node_id}.fault_arg", "must be a non-negative int")
        nodes[node_id] = NodeEntry(node_id, host or "127.0.0.1",
                                   port if isinstance(port, int) else 0,
                                   wmin if isinstance(wmin, int) else 1,
                                   wmax if isinstance(wmax, int) else 4,
                                   fault_mode if fault_mode in SUPPORTED_FAULT_MODES else "none",
                                   fault_arg if isinstance(fault_arg, int) else 0)
    vc.check()
    return Registry(nodes)


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class DerivedEdge:
    parent: str
    coefficient: float


@dataclass(frozen=True)
class ManifestItem:
    key: str
    value: float
    targets: tuple[str, ...]
    strategy: str
    strategy_params: dict[str, Any]
    derived_from: tuple[DerivedEdge, ...]


@dataclass
class Manifest:
    manifest_id: str
    version: int
    predecessor_version: int | None
    items: dict[str, ManifestItem]
    order: list[str] = field(default_factory=list)

    def parents(self, key: str) -> list[str]:
        return [edge.parent for edge in self.items[key].derived_from]

    def coefficient(self, parent: str, key: str) -> float:
        for edge in self.items[key].derived_from:
            if edge.parent == parent:
                return edge.coefficient
        raise KeyError((parent, key))


def _is_real_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and not (isinstance(value, float) and not math.isfinite(value))


def load_manifest(data: dict[str, Any], registry: Registry,
                  label: str = "manifest") -> Manifest:
    """Validate one manifest and raise :class:`ManifestError` on any problem."""

    vc = ViolationCollector()
    if not isinstance(data, dict):
        raise ManifestError(f"{label} must be an object",
                            [(label, "top-level JSON value is not an object")])
    version = data.get("version")
    if not isinstance(version, int) or isinstance(version, bool):
        vc.add(f"{label}.version", "version must be an integer")
        version = -1
    predecessor = data.get("predecessor_version")
    if predecessor is not None and (
            not isinstance(predecessor, int) or isinstance(predecessor, bool)):
        vc.add(f"{label}.predecessor_version", "must be an integer or null")
    elif isinstance(predecessor, int) and isinstance(version, int) and version >= 0:
        if predecessor >= version:
            vc.add(f"{label}.version",
                   f"version {version} is not greater than predecessor "
                   f"{predecessor} (monotonic increase required)")
    raw_items = data.get("items")
    if not isinstance(raw_items, list) or not raw_items:
        vc.add(f"{label}.items", "items must be a non-empty list")
        raw_items = []

    items: dict[str, ManifestItem] = {}
    for index, raw in enumerate(raw_items):
        if not isinstance(raw, dict):
            vc.add(f"{label}.items[{index}]", "item must be an object")
            continue
        key = raw.get("key")
        subject = f"{label}.items[{index}].{key}"
        if not isinstance(key, str) or not KEY_PATTERN.fullmatch(key):
            vc.add(f"{label}.items[{index}].key={key!r}",
                   "missing or illegal key (^[A-Za-z_][A-Za-z0-9_.-]{0,63}$)")
            continue
        if key in items:
            vc.add(subject, "duplicate item key")
            continue
        value = raw.get("value")
        if not _is_real_number(value):
            vc.add(f"{subject}.value",
                   "value must be a finite int or float (booleans rejected)")
        targets = raw.get("targets")
        if not isinstance(targets, list) or not targets:
            vc.add(f"{subject}.targets", "targets must be a non-empty list")
            target_ids: tuple[str, ...] = ()
        else:
            if len(set(targets)) != len(targets):
                vc.add(f"{subject}.targets", "duplicate target node entries")
            unknown = [t for t in targets if t not in registry.nodes]
            for node_id in sorted(unknown):
                vc.add(f"{subject}.targets.{node_id}",
                       "target node is not registered")
            target_ids = tuple(sorted(set(targets)))
        strategy = raw.get("strategy", "full_mesh")
        if strategy not in SUPPORTED_STRATEGIES:
            vc.add(f"{subject}.strategy",
                   f"unsupported strategy {strategy!r}; expected one of "
                   f"{SUPPORTED_STRATEGIES}")
        strategy_params = raw.get("strategy_params", {})
        if not isinstance(strategy_params, dict):
            vc.add(f"{subject}.strategy_params", "must be an object")
            strategy_params = {}
        edges: list[DerivedEdge] = []
        raw_edges = raw.get("derived_from", [])
        if not isinstance(raw_edges, list):
            vc.add(f"{subject}.derived_from", "must be a list")
            raw_edges = []
        seen_parents: set[str] = set()
        for edge_raw in raw_edges:
            if not isinstance(edge_raw, dict):
                vc.add(f"{subject}.derived_from", "each edge must be an object")
                continue
            parent = edge_raw.get("key")
            coeff = edge_raw.get("coefficient")
            if not isinstance(parent, str):
                vc.add(f"{subject}.derived_from", "edge missing parent key")
                continue
            if parent == key:
                vc.add(f"{subject}.derived_from.{parent}", "self dependency")
            if parent in seen_parents:
                vc.add(f"{subject}.derived_from.{parent}", "duplicate parent edge")
            seen_parents.add(parent)
            if not _is_real_number(coeff):
                vc.add(f"{subject}.derived_from.{parent}.coefficient",
                       "coefficient must be a finite number")
                continue
            edges.append(DerivedEdge(parent, float(coeff)))
        items[key] = ManifestItem(
            key=key,
            value=float(value) if _is_real_number(value) else 0.0,
            targets=target_ids,
            strategy=strategy if strategy in SUPPORTED_STRATEGIES else "full_mesh",
            strategy_params=strategy_params,
            derived_from=tuple(edges),
        )

    # Reference integrity and DAG cycle detection (after all keys are known).
    known = set(items)
    for key in sorted(known):
        for edge in items[key].derived_from:
            if edge.parent not in known:
                vc.add(f"{label}.items.{key}.derived_from.{edge.parent}",
                       "derivation target does not exist in this manifest")
    order = _topological_order(items, vc, label)
    vc.check()
    return Manifest(
        manifest_id=str(data.get("manifest_id", label)),
        version=version,
        predecessor_version=predecessor,
        items=items,
        order=order,
    )


def _topological_order(items: dict[str, ManifestItem],
                       vc: ViolationCollector, label: str) -> list[str]:
    """Kahn's algorithm with lexicographic tie-breaking (deterministic)."""

    indegree = {key: 0 for key in items}
    children: dict[str, list[str]] = {key: [] for key in items}
    for key, item in items.items():
        for edge in item.derived_from:
            if edge.parent in items:
                indegree[key] += 1
                children[edge.parent].append(key)
    ready = sorted(key for key, deg in indegree.items() if deg == 0)
    order: list[str] = []
    while ready:
        key = ready.pop(0)
        order.append(key)
        for child in sorted(children[key]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if len(order) != len(items):
        remaining = sorted(set(items) - set(order))
        for key in remaining:
            vc.add(f"{label}.items.{key}",
                   "derivation graph contains a cycle involving this item")
    return order


def assert_version_monotonic(baseline: Manifest, candidate: Manifest) -> None:
    """Cross-manifest monotonicity: candidate version must be the newer one."""

    if candidate.version <= baseline.version:
        raise ManifestError(
            "candidate version is not greater than baseline version",
            [("manifest.version",
              f"baseline={baseline.version} candidate={candidate.version} "
              "requires candidate > baseline")],
        )


# --------------------------------------------------------------------------- #
# Session parameters
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SessionConfig:
    scale_exponent: int
    base_seed: int
    sample_count: int
    group_count: int
    aggregations: tuple[str, ...]
    global_concurrency: int
    node_concurrency: int
    socket_timeout_ms: int
    max_attempts: int
    retry_backoff_ms: int
    value_tolerance: float
    elapsed_tolerance_ms: int
    direct_threshold_permille: int
    coefficient_threshold_permille: int
    dispersion_threshold_permille: int
    ledger_format_version: int
    precision_scheme_version: int

    @staticmethod
    def from_dict(data: dict[str, Any]) -> "SessionConfig":
        defaults = SessionConfig.defaults()
        merged = {**defaults.__dict__, **(data or {})}
        allowed = set(defaults.__dict__)
        unknown = sorted(set(merged) - allowed)
        if unknown:
            raise ManifestError("unknown session parameters",
                                [(f"session.{name}", "unknown parameter")
                                 for name in unknown])
        cfg = SessionConfig(**merged)
        cfg.validate()
        return cfg

    @staticmethod
    def defaults() -> "SessionConfig":
        return SessionConfig(
            scale_exponent=6,
            base_seed=20261123,
            sample_count=2000,
            group_count=8,
            aggregations=("sum", "mean", "min", "max"),
            global_concurrency=4,
            node_concurrency=2,
            socket_timeout_ms=1500,
            max_attempts=3,
            retry_backoff_ms=20,
            value_tolerance=1e-9,
            elapsed_tolerance_ms=100000,
            direct_threshold_permille=600,
            coefficient_threshold_permille=600,
            dispersion_threshold_permille=600,
            ledger_format_version=1,
            precision_scheme_version=1,
        )

    def validate(self) -> None:
        vc = ViolationCollector()
        if not isinstance(self.scale_exponent, int) or not 0 <= self.scale_exponent <= 12:
            vc.add("session.scale_exponent", "must be int in [0, 12]")
        for name in ("base_seed", "sample_count", "group_count",
                     "global_concurrency", "node_concurrency",
                     "socket_timeout_ms", "max_attempts", "retry_backoff_ms",
                     "elapsed_tolerance_ms", "ledger_format_version",
                     "precision_scheme_version"):
            val = getattr(self, name)
            if not isinstance(val, int) or isinstance(val, bool) or val < 0:
                vc.add(f"session.{name}", "must be a non-negative int")
        if self.sample_count <= 0:
            vc.add("session.sample_count", "must be > 0")
        if self.group_count <= 0:
            vc.add("session.group_count", "must be > 0")
        if self.global_concurrency <= 0 or self.node_concurrency <= 0:
            vc.add("session.concurrency", "concurrency limits must be > 0")
        if self.max_attempts <= 0:
            vc.add("session.max_attempts", "must be > 0")
        if not self.aggregations or any(a not in ("sum", "mean", "min", "max")
                                        for a in self.aggregations):
            vc.add("session.aggregations",
                   "only sum/mean/min/max are supported and at least one required")
        if not _is_real_number(self.value_tolerance) or self.value_tolerance < 0:
            vc.add("session.value_tolerance", "must be a non-negative finite number")
        for name in ("direct_threshold_permille", "coefficient_threshold_permille",
                     "dispersion_threshold_permille"):
            val = getattr(self, name)
            if not isinstance(val, int) or not 0 <= val <= 1000:
                vc.add(f"session.{name}", "must be int in [0, 1000]")
        vc.check()


def load_json_file(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
