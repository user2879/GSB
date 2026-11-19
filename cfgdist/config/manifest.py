"""Versioned configuration manifests, node registry and hard validation."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

from ..errors import E_CYCLE, E_MANIFEST, ErrorDetail, PlatformError
from .grid import _to_fraction

_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{0,63}$")
_STRATEGIES = ("full", "shard")


@dataclass(frozen=True)
class Derivation:
    """``target = offset + sum(coef[upstream] * upstream)``."""

    target: str
    offset: Fraction
    coefficients: dict[str, Fraction]  # upstream -> exact rational coefficient

    def to_dict(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "offset": f"{self.offset}",
            "coefficients": {k: f"{v}" for k, v in sorted(self.coefficients.items())},
        }


@dataclass(frozen=True)
class Item:
    key: str
    value: Fraction | None          # None for derived items
    targets: frozenset[str]
    derivation: Derivation | None


@dataclass(frozen=True)
class ManifestVersion:
    version: int
    strategy: str
    items: dict[str, Item]
    order: tuple[str, ...]          # stable declaration order

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "strategy": self.strategy,
            "items": [
                {
                    "key": item.key,
                    "value": None if item.value is None else f"{item.value}",
                    "targets": sorted(item.targets),
                    "derivation": None
                    if item.derivation is None
                    else item.derivation.to_dict(),
                }
                for item in (self.items[k] for k in self.order)
            ],
        }


@dataclass(frozen=True)
class Manifest:
    path: Path
    versions: tuple[ManifestVersion, ...]


@dataclass(frozen=True)
class NodeSpec:
    node_id: str
    aggregate: str
    sample: list[float]
    port: int | None = None  # optional fixed port for E_PORT injection

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "aggregate": self.aggregate,
            "sample": list(self.sample),
            "port": self.port,
        }


@dataclass(frozen=True)
class Registry:
    path: Path
    nodes: dict[str, NodeSpec]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _load_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError as exc:
        raise PlatformError(
            E_MANIFEST,
            "manifest file not found",
            [ErrorDetail(reason=str(exc), source=str(path))],
        ) from exc
    except json.JSONDecodeError as exc:
        raise PlatformError(
            E_MANIFEST,
            "manifest file is corrupt and cannot be parsed as JSON",
            [
                ErrorDetail(
                    reason=f"JSON decode error at line {exc.lineno}: {exc.msg}",
                    source=str(path),
                )
            ],
        ) from exc
    except OSError as exc:
        raise PlatformError(
            E_MANIFEST,
            "manifest file unreadable",
            [ErrorDetail(reason=str(exc), source=str(path))],
        ) from exc


def _fraction(raw: Any, *, key: str, field_name: str,
              details: list[ErrorDetail], source: str) -> Fraction | None:
    if isinstance(raw, bool) or not isinstance(raw, (int, float, str)):
        details.append(
            ErrorDetail(
                reason=f"{field_name} must be a numeric value, got {type(raw).__name__}",
                key=key,
                source=source,
            )
        )
        return None
    try:
        return _to_fraction(raw)
    except (ValueError, ZeroDivisionError) as exc:
        details.append(
            ErrorDetail(reason=f"illegal numeric literal: {exc}", key=key, source=source)
        )
        return None


def load_registry(path: str | Path) -> Registry:
    registry_path = Path(path)
    data = _load_json(registry_path)
    details: list[ErrorDetail] = []
    if not isinstance(data, dict) or not isinstance(data.get("nodes"), list):
        details.append(
            ErrorDetail(reason="registry must contain a 'nodes' list", source=str(registry_path))
        )
        PlatformError.raise_if(E_MANIFEST, "invalid node registry", details)

    nodes: dict[str, NodeSpec] = {}
    for entry in data["nodes"]:
        if not isinstance(entry, dict):
            details.append(ErrorDetail(reason="node entry must be an object", source=str(registry_path)))
            continue
        node_id = entry.get("node_id")
        if not isinstance(node_id, str) or not _KEY_RE.match(node_id):
            details.append(
                ErrorDetail(reason="node_id missing or illegal identifier", source=str(registry_path))
            )
            continue
        if node_id in nodes:
            details.append(ErrorDetail(reason="duplicate node id", key=node_id, source=str(registry_path)))
            continue
        aggregate = entry.get("aggregate", "sum")
        if aggregate not in ("sum", "mean", "l2norm", "max"):
            details.append(
                ErrorDetail(
                    reason=f"unsupported aggregate '{aggregate}'",
                    key=node_id,
                    source=str(registry_path),
                )
            )
        sample = entry.get("sample", [])
        if not isinstance(sample, list) or not sample or not all(
            isinstance(x, (int, float)) and not isinstance(x, bool) for x in sample
        ):
            details.append(
                ErrorDetail(
                    reason="sample must be a non-empty list of finite numbers",
                    key=node_id,
                    source=str(registry_path),
                )
            )
            sample = []
        port = entry.get("port")
        if port is not None and (not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535):
            details.append(
                ErrorDetail(reason="port must be an integer in [1,65535]", key=node_id, source=str(registry_path))
            )
            port = None
        nodes[node_id] = NodeSpec(node_id=node_id, aggregate=aggregate, sample=list(sample), port=port)

    PlatformError.raise_if(E_MANIFEST, "invalid node registry", details)
    return Registry(path=registry_path, nodes=nodes)


def load_manifest(path: str | Path, registry: Registry) -> Manifest:
    manifest_path = Path(path)
    data = _load_json(manifest_path)
    source = str(manifest_path)
    details: list[ErrorDetail] = []

    if not isinstance(data, dict) or not isinstance(data.get("versions"), list):
        details.append(ErrorDetail(reason="manifest must contain a 'versions' list", source=source))
        PlatformError.raise_if(E_MANIFEST, "invalid configuration manifest", details)

    parsed_versions: list[ManifestVersion] = []
    seen_versions: set[int] = set()
    previous_version: int | None = None

    for version_entry in data["versions"]:
        if not isinstance(version_entry, dict):
            details.append(ErrorDetail(reason="version entry must be an object", source=source))
            continue
        version = version_entry.get("version")
        if isinstance(version, bool) or not isinstance(version, int):
            details.append(ErrorDetail(reason="version must be an integer", source=source))
            continue
        if version in seen_versions:
            details.append(ErrorDetail(reason="duplicate version number", key=str(version), source=source))
            continue
        if previous_version is not None and version <= previous_version:
            details.append(
                ErrorDetail(
                    reason="version numbers must be strictly increasing",
                    key=str(version),
                    source=source,
                )
            )
        seen_versions.add(version)
        previous_version = version

        strategy = version_entry.get("strategy")
        if strategy not in _STRATEGIES:
            details.append(
                ErrorDetail(
                    reason=f"strategy must be one of {_STRATEGIES}",
                    key=str(version),
                    source=source,
                )
            )
            strategy = "full"

        raw_items = version_entry.get("items")
        if not isinstance(raw_items, list) or not raw_items:
            details.append(ErrorDetail(reason="items must be a non-empty list", key=str(version), source=source))
            continue

        items: dict[str, Item] = {}
        order: list[str] = []
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                details.append(ErrorDetail(reason="item must be an object", key=str(version), source=source))
                continue
            key = raw_item.get("key")
            if not isinstance(key, str) or not _KEY_RE.match(key):
                details.append(
                    ErrorDetail(
                        reason="item key missing or illegal (^[A-Za-z_][A-Za-z0-9_]{0,63}$)",
                        source=source,
                    )
                )
                continue
            if key in items:
                details.append(ErrorDetail(reason="duplicate item key within version", key=key, source=source))
                continue

            targets_raw = raw_item.get("targets")
            if not isinstance(targets_raw, list) or not targets_raw:
                details.append(ErrorDetail(reason="targets must be a non-empty list", key=key, source=source))
                targets: frozenset[str] = frozenset()
            else:
                bad = [t for t in targets_raw if not isinstance(t, str) or t not in registry.nodes]
                for node_id in bad:
                    details.append(
                        ErrorDetail(
                            reason=f"target node '{node_id}' not present in registry",
                            key=key,
                            source=source,
                        )
                    )
                targets = frozenset(t for t in targets_raw if isinstance(t, str) and t in registry.nodes)

            raw_derivation = raw_item.get("derivation")
            derivation: Derivation | None = None
            value: Fraction | None = None
            if raw_derivation is not None:
                if not isinstance(raw_derivation, dict):
                    details.append(ErrorDetail(reason="derivation must be an object", key=key, source=source))
                else:
                    offset = _fraction(
                        raw_derivation.get("offset", 0),
                        key=key,
                        field_name="derivation.offset",
                        details=details,
                        source=source,
                    ) or Fraction(0)
                    raw_coeffs = raw_derivation.get("coefficients", {})
                    if not isinstance(raw_coeffs, dict) or not raw_coeffs:
                        details.append(
                            ErrorDetail(reason="derivation coefficients must be a non-empty object", key=key, source=source)
                        )
                    coefficients: dict[str, Fraction] = {}
                    for upstream, raw_coef in raw_coeffs.items() if isinstance(raw_coeffs, dict) else []:
                        if upstream == key:
                            details.append(ErrorDetail(reason="item cannot derive from itself", key=key, source=source))
                            continue
                        if not isinstance(upstream, str) or not _KEY_RE.match(upstream):
                            details.append(ErrorDetail(reason="illegal upstream key", key=key, source=source))
                            continue
                        coef = _fraction(raw_coef, key=key, field_name="coefficient",
                                         details=details, source=source)
                        if coef is not None:
                            coefficients[upstream] = coef
                    if coefficients:
                        derivation = Derivation(target=key, offset=offset, coefficients=coefficients)
            else:
                value = _fraction(raw_item.get("value"), key=key, field_name="value",
                                  details=details, source=source)
                if value is None:
                    details.append(
                        ErrorDetail(reason="non-derived item requires a numeric 'value'", key=key, source=source)
                    )

            items[key] = Item(key=key, value=value, targets=targets, derivation=derivation)
            order.append(key)

        # Upstream existence and acyclicity for this version.
        _validate_graph(items, order, source, details)
        parsed_versions.append(
            ManifestVersion(version=version, strategy=strategy, items=items, order=tuple(order))
        )

    PlatformError.raise_if(E_MANIFEST, "invalid configuration manifest", details)
    return Manifest(path=manifest_path, versions=tuple(parsed_versions))


def _validate_graph(
    items: dict[str, Item],
    order: list[str],
    source: str,
    details: list[ErrorDetail],
) -> None:
    # Upstream existence first; missing references are stripped before the DFS.
    edges: dict[str, list[str]] = {}
    for key in order:
        item = items[key]
        if item.derivation is None:
            edges[key] = []
            continue
        existing: list[str] = []
        for upstream in item.derivation.coefficients:
            if upstream not in items:
                details.append(
                    ErrorDetail(
                        reason=f"derivation references missing upstream item '{upstream}'",
                        key=key,
                        source=source,
                    )
                )
            else:
                existing.append(upstream)
        edges[key] = existing

    # Stable DFS cycle detection; report the actual cycle as E_CYCLE separately
    # from generic manifest problems.
    WHITE, GRAY, BLACK = 0, 1, 2
    color = {key: WHITE for key in order}
    cycle_path: list[str] = []

    def visit(node: str, stack: list[str]) -> bool:
        color[node] = GRAY
        stack.append(node)
        for upstream in edges[node]:
            if color[upstream] == GRAY:
                idx = stack.index(upstream)
                cycle_path.extend(stack[idx:] + [upstream])
                return True
            if color[upstream] == WHITE and visit(upstream, stack):
                return True
        stack.pop()
        color[node] = BLACK
        return False

    for key in order:
        if color[key] == WHITE and visit(key, []):
            raise PlatformError(
                E_CYCLE,
                "derivation graph contains a cycle",
                [
                    ErrorDetail(
                        reason="cycle: " + " -> ".join(cycle_path),
                        key=cycle_path[0],
                        source=source,
                    )
                ],
            )


def topo_order(version: ManifestVersion) -> list[str]:
    """Stable topological order: leaves first, derived items after upstreams."""
    indegree = {key: 0 for key in version.order}
    children: dict[str, list[str]] = {key: [] for key in version.order}
    for key in version.order:
        item = version.items[key]
        if item.derivation is not None:
            for upstream in item.derivation.coefficients:
                indegree[key] += 1
                children[upstream].append(key)
    ready = sorted(key for key in version.order if indegree[key] == 0)
    result: list[str] = []
    while ready:
        key = ready.pop(0)
        result.append(key)
        for child in sorted(children[key]):
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
                ready.sort()
    if len(result) != len(version.order):
        raise PlatformError(
            E_CYCLE,
            "cannot topologically sort a cyclic derivation graph",
            [ErrorDetail(reason="cycle detected during topo sort", source=str(version.version))],
        )
    return result
