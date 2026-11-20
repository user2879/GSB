"""Explicit, reproducible input-file fault injection for acceptance commands."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .errors import BusinessError, E_IO


def _read(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _walk_selector_keys(selector: Any, output: list[str]) -> None:
    if not isinstance(selector, dict):
        return
    labels = selector.get("matchLabels")
    if isinstance(labels, dict):
        output.extend(str(key) for key in labels)
    expressions = selector.get("matchExpressions")
    if isinstance(expressions, list):
        for expression in expressions:
            if isinstance(expression, dict) and isinstance(expression.get("key"), str):
                output.append(expression["key"])


def _node_reference_keys(pods_doc: dict[str, Any]) -> list[str]:
    keys: list[str] = []
    for pod in pods_doc.get("pods", []):
        if not isinstance(pod, dict):
            continue
        affinity = pod.get("nodeAffinity")
        if isinstance(affinity, dict):
            _walk_selector_keys(affinity.get("hard"), keys)
            for rule in affinity.get("soft", []) if isinstance(affinity.get("soft"), list) else []:
                if isinstance(rule, dict):
                    _walk_selector_keys(rule.get("selector"), keys)
        anti = pod.get("antiAffinity")
        if isinstance(anti, dict):
            for mode in ("hard", "soft"):
                rules = anti.get(mode, [])
                if not isinstance(rules, list):
                    continue
                for rule in rules:
                    if isinstance(rule, dict) and isinstance(rule.get("topologyKey"), str):
                        keys.append(rule["topologyKey"])
    return keys


def prepare_injected_inputs(
    nodes_path: str | Path,
    pods_path: str | Path,
    plugins_path: str | Path | None,
    workdir: str | Path,
    kind: str,
) -> tuple[str | Path, str | Path, str | Path | None, dict[str, Any]]:
    """Create modified copies under workdir; original manifests are untouched."""
    if kind in {None, "", "none"}:
        return nodes_path, pods_path, plugins_path, {"kind": "none"}
    if kind not in {"manifest", "ref"}:
        raise BusinessError(E_IO, f"unsupported manifest injection kind '{kind}'")
    try:
        nodes_doc = _read(nodes_path)
        pods_doc = _read(pods_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise BusinessError(E_IO, f"cannot prepare injection input: {exc}") from None
    injected_dir = Path(workdir) / "injected-inputs" / kind
    out_nodes = injected_dir / "nodes.json"
    out_pods = injected_dir / "pods.json"
    out_plugins = plugins_path
    meta: dict[str, Any] = {"kind": kind, "originalNodes": str(nodes_path), "originalPods": str(pods_path)}
    if kind == "manifest":
        if len(nodes_doc.get("nodes", [])) >= 2:
            nodes_doc["nodes"][1]["name"] = nodes_doc["nodes"][0]["name"]
            meta["mutation"] = "duplicate node name at nodes[1].name"
        elif nodes_doc.get("nodes"):
            nodes_doc["nodes"][0]["allocatable"]["cpu"] = 0
            meta["mutation"] = "zero node CPU at nodes[0].allocatable.cpu"
        else:
            nodes_doc["nodes"] = [{"name": "broken", "labels": {}, "allocatable": {"cpu": 0, "memory": 1}}]
            meta["mutation"] = "empty node list replaced by invalid node"
    elif kind == "ref":
        referenced = _node_reference_keys(pods_doc)
        available = {
            key
            for node in nodes_doc.get("nodes", [])
            if isinstance(node, dict) and isinstance(node.get("labels"), dict)
            for key in node["labels"]
        }
        target = next((key for key in referenced if key in available), None)
        if target is None:
            raise BusinessError(E_IO, "cannot inject E_REF: no node-affinity label key was found")
        for node in nodes_doc.get("nodes", []):
            if isinstance(node, dict) and isinstance(node.get("labels"), dict):
                node["labels"].pop(target, None)
        meta["mutation"] = f"removed node label key {target!r} referenced by pod constraints"
        meta["labelKey"] = target
    _write(out_nodes, nodes_doc)
    _write(out_pods, pods_doc)
    if plugins_path is not None:
        out_plugins = injected_dir / "plugins.json"
        _write(out_plugins, _read(plugins_path))
    meta.update({"nodes": str(out_nodes), "pods": str(out_pods), "plugins": str(out_plugins) if out_plugins else None})
    return out_nodes, out_pods, out_plugins, meta


def write_digest_mismatch_snapshot(payload: dict[str, Any], workdir: str | Path) -> Path:
    """Write an intentionally digest-corrupted snapshot input file."""
    altered = copy.deepcopy(payload)
    altered["contentDigest"] = "sha256:injected-content-digest-mismatch"
    target = Path(workdir) / "injected-inputs" / "snapshot-digest-mismatch.json"
    _write(target, altered)
    return target
