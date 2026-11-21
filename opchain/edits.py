"""Reversible-at-the-cache-layer chain/config edit helpers."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

from .errors import E_CHAIN_INVALID, BusinessError
from .util import atomic_write_json, read_json


EDIT_KINDS = {
    "set_params",
    "set_inplace",
    "move_node",
    "replace_subgraph",
    "set_precision",
    "set_quantization",
    "set_fusion",
    "set_config",
}


def _copy_chain(chain: dict[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(chain)


def _node_index(chain: dict[str, Any], node_id: str) -> int:
    for index, node in enumerate(chain.get("nodes", [])):
        if node.get("id") == node_id:
            return index
    raise BusinessError(E_CHAIN_INVALID, f"unknown node: {node_id}", stage="edit")


def set_node_params(chain: dict[str, Any], node_id: str, params: dict[str, Any], *, replace: bool = True) -> dict[str, Any]:
    updated = _copy_chain(chain)
    node = updated["nodes"][_node_index(updated, node_id)]
    if replace:
        node["params"] = copy.deepcopy(params)
    else:
        merged = dict(node.get("params", {}))
        merged.update(copy.deepcopy(params))
        node["params"] = merged
    return updated


def set_node_inplace(chain: dict[str, Any], node_id: str, inplace: bool) -> dict[str, Any]:
    updated = _copy_chain(chain)
    updated["nodes"][_node_index(updated, node_id)]["inplace"] = bool(inplace)
    return updated


def move_node(chain: dict[str, Any], node_id: str, target_index: int) -> dict[str, Any]:
    updated = _copy_chain(chain)
    source_index = _node_index(updated, node_id)
    if target_index < 0 or target_index >= len(updated["nodes"]):
        raise BusinessError(E_CHAIN_INVALID, f"target_index out of range: {target_index}", stage="edit")
    node = updated["nodes"].pop(source_index)
    updated["nodes"].insert(target_index, node)
    return updated


def replace_subgraph(
    chain: dict[str, Any],
    remove_ids: list[str],
    add_nodes: list[dict[str, Any]],
    *,
    add_edges: list[dict[str, Any]] | None = None,
    rewire: dict[str, str] | None = None,
) -> dict[str, Any]:
    updated = _copy_chain(chain)
    removed = set(remove_ids)
    rewired = {str(old): str(new) for old, new in (rewire or {}).items()}
    existing_ids = {node.get("id") for node in updated["nodes"]}
    missing = removed - existing_ids
    if missing:
        raise BusinessError(E_CHAIN_INVALID, f"subgraph removal references unknown nodes: {sorted(missing)}", stage="edit")
    updated["nodes"] = [node for node in updated["nodes"] if node.get("id") not in removed]
    updated["nodes"].extend(copy.deepcopy(add_nodes))
    replacement_ids = {node.get("id") for node in add_nodes}

    for node in updated["nodes"]:
        inputs = node.get("inputs")
        if isinstance(inputs, list):
            node["inputs"] = [rewired.get(str(item), str(item)) for item in inputs]

    # Keep the product-facing order field implicit (array index), and remove edges touching replaced nodes.
    if isinstance(updated.get("edges"), list):
        kept_edges: list[dict[str, Any]] = []
        for edge in updated["edges"]:
            endpoint_from = edge.get("from", edge.get("source"))
            endpoint_to = edge.get("to", edge.get("target"))
            if endpoint_from in removed and endpoint_from not in replacement_ids and endpoint_from not in rewired:
                continue
            if endpoint_to in removed and endpoint_to not in replacement_ids and endpoint_to not in rewired:
                continue
            edge = copy.deepcopy(edge)
            if "from" in edge:
                edge["from"] = rewired.get(str(edge["from"]), edge["from"])
            if "source" in edge:
                edge["source"] = rewired.get(str(edge["source"]), edge["source"])
            if "to" in edge:
                edge["to"] = rewired.get(str(edge["to"]), edge["to"])
            if "target" in edge:
                edge["target"] = rewired.get(str(edge["target"]), edge["target"])
            kept_edges.append(edge)
        updated["edges"] = kept_edges
    if add_edges:
        if not isinstance(updated.get("edges"), list):
            updated["edges"] = []
        updated["edges"].extend(copy.deepcopy(add_edges))
    return updated


def set_config_value(config: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    updated = copy.deepcopy(config)
    updated[key] = value
    return updated


def apply_edit(chain: dict[str, Any], config: dict[str, Any], edit: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    kind = edit.get("kind")
    if kind not in EDIT_KINDS:
        raise BusinessError(E_CHAIN_INVALID, f"unsupported edit kind: {kind!r}", stage="edit")
    if kind == "set_params":
        return set_node_params(chain, edit["node_id"], edit.get("params", {}), replace=edit.get("replace", True)), copy.deepcopy(config)
    if kind == "set_inplace":
        return set_node_inplace(chain, edit["node_id"], edit["inplace"]), copy.deepcopy(config)
    if kind == "move_node":
        return move_node(chain, edit["node_id"], int(edit["target_index"])), copy.deepcopy(config)
    if kind == "replace_subgraph":
        return replace_subgraph(
            chain,
            edit.get("remove", []),
            edit.get("add_nodes", []),
            add_edges=edit.get("add_edges"),
            rewire=edit.get("rewire"),
        ), copy.deepcopy(config)
    if kind == "set_precision":
        return copy.deepcopy(chain), set_config_value(config, "precision_tier", edit["precision_tier"])
    if kind == "set_quantization":
        return copy.deepcopy(chain), set_config_value(config, "quantization_tier", edit["quantization_tier"])
    if kind == "set_fusion":
        return copy.deepcopy(chain), set_config_value(config, "fusion_strategy", edit["fusion_strategy"])
    if kind == "set_config":
        merged = copy.deepcopy(config)
        merged.update(copy.deepcopy(edit.get("config", {})))
        return copy.deepcopy(chain), merged
    raise BusinessError(E_CHAIN_INVALID, f"unsupported edit kind: {kind!r}", stage="edit")


def apply_edit_files(
    chain_path: str | Path,
    config_path: str | Path,
    edit: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    chain = read_json(chain_path)
    config = read_json(config_path)
    updated_chain, updated_config = apply_edit(chain, config, edit)
    atomic_write_json(chain_path, updated_chain)
    atomic_write_json(config_path, updated_config)
    return updated_chain, updated_config
