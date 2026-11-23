"""Immutable logical plan trees, deterministic summaries and plan edits."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping, Sequence

from .canonical import canonical_bytes, digest, stable_sorted
from .errors import E_DIVERGE, E_SPEC, AppError, ref_error, spec_error
from .sql_parser import AGGREGATES, COMPARISONS, Condition, ParsedQuery, SelectItem


SUPPORTED_EDIT_TYPES = frozenset(
    {"push_filter", "prune_columns", "replace_scan_order", "modify_aggregate"}
)


@dataclass(frozen=True)
class TableSnapshot:
    name: str
    schema: tuple[tuple[str, str], ...]
    rows: tuple[dict[str, Any], ...]

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self.schema)

    @property
    def column_types(self) -> dict[str, str]:
        return {name: type_name for name, type_name in self.schema}

    @classmethod
    def from_table(cls, table: Any) -> "TableSnapshot":
        if isinstance(table, TableSnapshot):
            return table
        name = str(table.name).lower()
        schema = tuple((str(column.name).lower(), str(column.type).lower()) for column in table.columns)
        row_count = len(table.columns[0].values) if table.columns else 0
        rows = []
        for index in range(row_count):
            row: dict[str, Any] = {}
            for column in table.columns:
                row[str(column.name).lower()] = None if column.nulls[index] else column.values[index]
            rows.append(row)
        return cls(name=name, schema=schema, rows=tuple(rows))


@dataclass(frozen=True)
class PlanNode:
    id: str
    kind: str
    props: dict[str, Any]
    children: tuple["PlanNode", ...]
    version: int
    summary: str
    estimated_rows: float


@dataclass(frozen=True)
class PlanTree:
    root: PlanNode
    query_name: str
    table: TableSnapshot
    scale_factor: float

    @property
    def summary(self) -> str:
        return self.root.summary

    @property
    def nodes(self) -> dict[str, PlanNode]:
        result: dict[str, PlanNode] = {}

        def visit(node: PlanNode) -> None:
            result[node.id] = node
            for child in node.children:
                visit(child)

        visit(self.root)
        return result

    @property
    def estimated_rows(self) -> float:
        return self.root.estimated_rows

    def to_dict(self) -> dict[str, Any]:
        return {"format": 1, "root": node_to_dict(self.root)}

    def to_bytes(self) -> bytes:
        return canonical_bytes(self.to_dict())


def _node_summary(node_id: str, kind: str, props: Mapping[str, Any], children: Sequence[PlanNode]) -> str:
    return digest(
        {
            "id": node_id,
            "kind": kind,
            "props": props,
            "children": [{"id": child.id, "summary": child.summary} for child in children],
        }
    )


def make_node(
    node_id: str,
    kind: str,
    props: Mapping[str, Any],
    children: Sequence[PlanNode] = (),
    *,
    version: int = 1,
    estimated_rows: float = 0.0,
) -> PlanNode:
    child_tuple = tuple(children)
    copied_props = copy.deepcopy(dict(props))
    return PlanNode(
        id=node_id,
        kind=kind,
        props=copied_props,
        children=child_tuple,
        version=int(version),
        summary=_node_summary(node_id, kind, copied_props, child_tuple),
        estimated_rows=float(estimated_rows),
    )


def node_to_dict(node: PlanNode) -> dict[str, Any]:
    return {
        "id": node.id,
        "kind": node.kind,
        "props": node.props,
        "version": node.version,
        "summary": node.summary,
        "estimated_rows": node.estimated_rows,
        "children": [node_to_dict(child) for child in node.children],
    }


def _find_path(root: PlanNode, node_id: str) -> list[PlanNode] | None:
    if root.id == node_id:
        return [root]
    for child in root.children:
        path = _find_path(child, node_id)
        if path is not None:
            return [root] + path
    return None


def _replace_node(root: PlanNode, node_id: str, replacement: PlanNode | None) -> PlanNode | None:
    if root.id == node_id:
        return replacement
    changed = False
    children: list[PlanNode] = []
    for child in root.children:
        new_child = _replace_node(child, node_id, replacement)
        if new_child is not child:
            changed = True
        if new_child is not None:
            children.append(new_child)
    if not changed:
        return root
    return replace(root, children=tuple(children))


def _validate_query(parsed: ParsedQuery, table: TableSnapshot, json_path: str) -> None:
    if parsed.table != table.name:
        raise spec_error(f"unknown table {parsed.table!r}", json_path, position=parsed.table_pos)
    column_types = table.column_types
    column_names = set(column_types)
    alias_names: list[str] = []

    if any(item.kind == "star" for item in parsed.select) and len(parsed.select) != 1:
        raise spec_error("SELECT * must be the only selected item", json_path, position=parsed.select[0].pos)

    for item in parsed.select:
        alias_names.append(item.alias)
        if item.kind == "column" and item.column not in column_names:
            raise spec_error(
                f"unknown column {item.column!r}", json_path, position=item.pos
            )
        if item.kind == "aggregate":
            if item.function not in AGGREGATES:
                raise spec_error(
                    f"unsupported aggregate {item.function!r}", json_path, position=item.pos
                )
            if item.column != "*" and item.column not in column_names:
                raise spec_error(
                    f"unknown column {item.column!r}", json_path, position=item.pos
                )
            if item.column == "*" and item.function != "count":
                raise spec_error("only COUNT may use *", json_path, position=item.pos)
    if len(set(alias_names)) != len(alias_names):
        seen: set[str] = set()
        for item in parsed.select:
            if item.alias in seen:
                raise spec_error(f"duplicate output name {item.alias!r}", json_path, position=item.pos)
            seen.add(item.alias)

    for condition in parsed.where:
        if condition.column not in column_names:
            raise spec_error(
                f"unknown WHERE column {condition.column!r}", json_path, position=condition.pos
            )
        column_type = column_types[condition.column]
        for value in condition.values:
            if column_type in {"int", "float"} and not isinstance(value, (int, float)):
                raise spec_error(
                    f"literal {value!r} is not numeric for column {condition.column!r}",
                    json_path,
                    position=condition.pos,
                )
            if column_type == "int" and isinstance(value, float) and not value.is_integer():
                raise spec_error(
                    f"non-integer literal {value!r} for integer column {condition.column!r}",
                    json_path,
                    position=condition.pos,
                )

    for group_name, position in zip(parsed.group_by, parsed.group_positions):
        if group_name not in column_names:
            raise spec_error(f"unknown GROUP BY column {group_name!r}", json_path, position=position)

    group_set = set(parsed.group_by)
    if parsed.is_aggregate_query:
        for item in parsed.select:
            if item.kind == "star":
                raise spec_error(
                    "SELECT * cannot be combined with grouping or aggregates",
                    json_path,
                    position=item.pos,
                )
            if item.kind == "column" and item.column not in group_set:
                raise spec_error(
                    f"selected column {item.column!r} must appear in GROUP BY",
                    json_path,
                    position=item.pos,
                )
        if parsed.has_aggregate and not parsed.group_by:
            for item in parsed.select:
                if item.kind != "aggregate":
                    raise spec_error(
                        "aggregate queries without GROUP BY may only select aggregates",
                        json_path,
                        position=item.pos,
                    )

    output_names = set(alias_names)
    if any(item.kind == "star" for item in parsed.select):
        output_names = set(table.column_names)
    for item in parsed.order_by:
        if item.name not in output_names:
            raise spec_error(
                f"ORDER BY name {item.name!r} is not a projected output",
                json_path,
                position=item.pos,
            )


def compile_base(
    parsed: ParsedQuery,
    table_like: Any,
    scale_factor: float,
    query_name: str,
    *,
    json_path: str = "$.queries[0].sql",
) -> PlanTree:
    table = TableSnapshot.from_table(table_like)
    _validate_query(parsed, table, json_path)

    scan = make_node(
        "scan",
        "scan",
        {
            "table": table.name,
            "columns": list(table.column_names),
            "schema": [list(item) for item in table.schema],
            "pushed_filters": [],
        },
    )
    child = scan
    if parsed.where:
        child = make_node(
            "filter",
            "filter",
            {"predicate": [condition.canonical() for condition in parsed.where]},
            [child],
        )

    if parsed.is_aggregate_query:
        aggregates: list[dict[str, Any]] = []
        for item in parsed.select:
            if item.kind == "aggregate":
                aggregates.append(
                    {
                        "name": item.alias,
                        "function": item.function,
                        "column": item.column,
                    }
                )
        aggregates.sort(key=lambda entry: canonical_bytes(entry["name"]))
        child = make_node(
            "aggregate",
            "aggregate",
            {"group_by": sorted(parsed.group_by), "aggregates": aggregates},
            [child],
        )

    outputs = project_outputs(parsed, table)
    child = make_node("project", "project", {"outputs": outputs}, [child])
    if parsed.order_by:
        child = make_node(
            "sort",
            "sort",
            {"items": [item.canonical() for item in parsed.order_by]},
            [child],
        )

    tree = PlanTree(root=child, query_name=query_name, table=table, scale_factor=float(scale_factor))
    return evaluate_plan(tree)


def project_outputs(parsed: ParsedQuery, table: TableSnapshot) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for item in parsed.select:
        if item.kind == "star":
            for name in table.column_names:
                outputs.append({"name": name, "kind": "column", "source": name})
        elif item.kind == "column":
            outputs.append({"name": item.alias, "kind": "column", "source": item.column})
        elif item.kind == "aggregate":
            outputs.append({"name": item.alias, "kind": "aggregate", "source": item.alias})
        else:  # pragma: no cover - parser invariant
            raise AssertionError(item.kind)
    return outputs


def _matches_value(row_value: Any, literal: Any, op: str) -> bool:
    if row_value is None or literal is None:
        return False
    if op == "=":
        return row_value == literal
    if op in {"!=", "<>"}:
        return row_value != literal
    if op == "<":
        return row_value < literal
    if op == "<=":
        return row_value <= literal
    if op == ">":
        return row_value > literal
    if op == ">=":
        return row_value >= literal
    raise AssertionError(op)


def condition_matches(row: Mapping[str, Any], condition: Mapping[str, Any]) -> bool:
    value = row.get(condition["column"])
    if condition["op"] == "in":
        if value is None:
            return False
        return any(value == candidate for candidate in condition["values"] if candidate is not None)
    return _matches_value(value, condition.get("value"), str(condition["op"]))


def _rows_for_node(node: PlanNode, table: TableSnapshot) -> list[dict[str, Any]]:
    if node.kind == "scan":
        rows = [dict(row) for row in table.rows]
        for predicate in node.props.get("pushed_filters", []):
            rows = [row for row in rows if condition_matches(row, predicate)]
        return rows
    if node.kind == "filter":
        rows = _rows_for_node(node.children[0], table)
        predicates = node.props.get("predicate", [])
        return [row for row in rows if all(condition_matches(row, predicate) for predicate in predicates)]
    if node.kind == "aggregate":
        rows = _rows_for_node(node.children[0], table)
        grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
        group_by = node.props.get("group_by", [])
        for row in rows:
            key = tuple(row.get(column) for column in group_by)
            grouped.setdefault(key, []).append(row)
        result: list[dict[str, Any]] = []
        for key, group_rows in grouped.items():
            output = {column: value for column, value in zip(group_by, key)}
            for aggregate in node.props.get("aggregates", []):
                function = aggregate["function"]
                column = aggregate["column"]
                values = [row.get(column) for row in group_rows] if column != "*" else [1] * len(group_rows)
                non_null = [value for value in values if value is not None]
                if function == "count":
                    value = len(non_null)
                elif not non_null:
                    value = None
                elif function == "sum":
                    value = sum(non_null)
                elif function == "min":
                    value = min(non_null)
                elif function == "max":
                    value = max(non_null)
                else:  # pragma: no cover - validated
                    raise AssertionError(function)
                output[aggregate["name"]] = value
            result.append(output)
        return result
    if node.kind == "project":
        rows = _rows_for_node(node.children[0], table)
        projected: list[dict[str, Any]] = []
        for row in rows:
            projected.append(
                {output["name"]: row.get(output["source"]) for output in node.props.get("outputs", [])}
            )
        return projected
    if node.kind == "sort":
        rows = _rows_for_node(node.children[0], table)
        for item in reversed(node.props.get("items", [])):
            name = item["name"]
            reverse = item["direction"] == "DESC"
            rows.sort(
                key=lambda row: (0 if row.get(name) is None else 1, row.get(name) if row.get(name) is not None else 0),
                reverse=reverse,
            )
        return rows
    raise AssertionError(node.kind)


def _attach_estimates(node: PlanNode, table: TableSnapshot, scale: float) -> PlanNode:
    children = tuple(_attach_estimates(child, table, scale) for child in node.children)
    provisional = replace(node, children=children)
    rows = _rows_for_node(provisional, table)
    return replace(provisional, estimated_rows=round(len(rows) * scale, 9))


def evaluate_plan(tree: PlanTree) -> PlanTree:
    return replace(tree, root=_attach_estimates(tree.root, tree.table, tree.scale_factor))


def _effects_payload(
    edit_seq: int,
    operation: Mapping[str, Any],
    direct: Iterable[str],
    removed: Iterable[str],
    propagation: Sequence[Mapping[str, str]],
    affected: Iterable[str],
) -> dict[str, Any]:
    return {
        "edit_seq": edit_seq,
        "operation": operation.get("type"),
        "target": operation.get("target"),
        "direct_nodes": stable_sorted(direct),
        "removed_nodes": stable_sorted(removed),
        "propagation": [{key: entry[key] for key in ("node", "reason")} for entry in propagation],
        "affected_nodes": stable_sorted(affected),
    }


def _apply_transform(
    plan: PlanTree, operation: Mapping[str, Any], edit_seq: int
) -> tuple[PlanNode, set[str], set[str], list[dict[str, str]], dict[str, Any]]:
    node_type = operation.get("type")
    target = operation.get("target")
    if node_type not in SUPPORTED_EDIT_TYPES:
        raise spec_error(f"unsupported edit type {node_type!r}", f"$.edits[{edit_seq - 1}].type")
    if not isinstance(target, str) or not target:
        raise spec_error("edit target must be a non-empty string", f"$.edits[{edit_seq - 1}].target")
    path = _find_path(plan.root, target)
    if path is None:
        raise ref_error(f"plan node {target!r} does not exist", edit_seq=edit_seq, node_id=target)
    target_node = path[-1]
    ancestors = path[:-1]
    direct: set[str] = set()
    removed: set[str] = set()
    detail: dict[str, Any] = {}

    if node_type == "push_filter":
        if target_node.kind != "filter":
            raise ref_error(
                f"edit {node_type!r} requires a filter node, got {target_node.kind!r}",
                edit_seq=edit_seq,
                node_id=target,
            )
        if not target_node.children or target_node.children[0].kind != "scan":
            raise spec_error("filter can only be pushed into a directly scanned table", f"$.edits[{edit_seq - 1}]")
        scan = target_node.children[0]
        pushed = list(scan.props.get("pushed_filters", []))
        predicate = copy.deepcopy(target_node.props.get("predicate", []))
        if not predicate:
            raise spec_error("empty filter cannot be pushed", f"$.edits[{edit_seq - 1}]")
        pushed.extend(predicate)
        new_scan = make_node(
            scan.id,
            "scan",
            {**scan.props, "pushed_filters": pushed},
            scan.children,
            version=scan.version,
        )
        root = _replace_node(plan.root, target, new_scan)
        direct.add(scan.id)
        removed.add(target)
        detail = {"pushed_predicate": predicate, "into_scan": scan.id}
    elif node_type == "prune_columns":
        if target_node.kind != "project":
            raise ref_error(
                f"edit {node_type!r} requires a project node, got {target_node.kind!r}",
                edit_seq=edit_seq,
                node_id=target,
            )
        raw_columns = operation.get("columns")
        if not isinstance(raw_columns, list) or not raw_columns or not all(isinstance(item, str) for item in raw_columns):
            raise spec_error("prune_columns.columns must be a non-empty string array", f"$.edits[{edit_seq - 1}].columns")
        requested = {item.lower() for item in raw_columns}
        outputs = list(target_node.props.get("outputs", []))
        existing = {output["name"] for output in outputs}
        missing = requested - existing
        if missing:
            raise ref_error(
                f"project output(s) not found: {', '.join(stable_sorted(missing))}",
                edit_seq=edit_seq,
                node_id=target,
            )
        remaining = [output for output in outputs if output["name"] not in requested]
        if not remaining:
            raise spec_error("prune_columns cannot remove every projected column", f"$.edits[{edit_seq - 1}].columns")
        for node in _walk(plan.root):
            if node.kind == "sort" and any(item["name"] in requested for item in node.props.get("items", [])):
                raise spec_error(
                    "cannot prune an ORDER BY output without removing the sort",
                    f"$.edits[{edit_seq - 1}].columns",
                )
        new_project = make_node(
            target_node.id,
            "project",
            {**target_node.props, "outputs": remaining},
            target_node.children,
            version=target_node.version,
        )
        root = _replace_node(plan.root, target, new_project)
        direct.add(target)
        detail = {"removed_outputs": stable_sorted(requested)}
    elif node_type == "replace_scan_order":
        if target_node.kind != "scan":
            raise ref_error(
                f"edit {node_type!r} requires a scan node, got {target_node.kind!r}",
                edit_seq=edit_seq,
                node_id=target,
            )
        raw_columns = operation.get("columns", operation.get("order"))
        if not isinstance(raw_columns, list) or not raw_columns or not all(isinstance(item, str) for item in raw_columns):
            raise spec_error("replace_scan_order.columns must be a non-empty string array", f"$.edits[{edit_seq - 1}].columns")
        requested = [item.lower() for item in raw_columns]
        current = list(target_node.props.get("columns", []))
        if sorted(requested) != sorted(current) or len(set(requested)) != len(requested):
            raise spec_error(
                "replace_scan_order.columns must be a permutation of the scan columns",
                f"$.edits[{edit_seq - 1}].columns",
            )
        new_scan = make_node(
            target_node.id,
            "scan",
            {**target_node.props, "columns": requested},
            target_node.children,
            version=target_node.version,
        )
        root = _replace_node(plan.root, target, new_scan)
        direct.add(target)
        detail = {"scan_order": requested}
    elif node_type == "modify_aggregate":
        if target_node.kind != "aggregate":
            raise ref_error(
                f"edit {node_type!r} requires an aggregate node, got {target_node.kind!r}",
                edit_seq=edit_seq,
                node_id=target,
            )
        name = operation.get("name", operation.get("alias"))
        function = operation.get("function")
        column = operation.get("column")
        if not isinstance(name, str) or not name:
            raise spec_error("modify_aggregate.name must be a non-empty string", f"$.edits[{edit_seq - 1}].name")
        if not isinstance(function, str) or function.lower() not in AGGREGATES:
            raise spec_error("unsupported aggregate function", f"$.edits[{edit_seq - 1}].function")
        function = function.lower()
        if function == "count" and (column is None or column == "*"):
            column = "*"
        elif not isinstance(column, str) or column.lower() not in set(plan.table.column_names):
            raise ref_error(
                f"aggregate column {column!r} does not exist", edit_seq=edit_seq, node_id=target
            )
        else:
            column = column.lower()
        aggregates = copy.deepcopy(target_node.props.get("aggregates", []))
        found = False
        for aggregate in aggregates:
            if aggregate["name"] == name.lower():
                aggregate["function"] = function
                aggregate["column"] = column
                found = True
                break
        if not found:
            raise ref_error(
                f"aggregate output {name!r} does not exist", edit_seq=edit_seq, node_id=target
            )
        aggregates.sort(key=lambda entry: canonical_bytes(entry["name"]))
        new_aggregate = make_node(
            target_node.id,
            "aggregate",
            {**target_node.props, "aggregates": aggregates},
            target_node.children,
            version=target_node.version,
        )
        root = _replace_node(plan.root, target, new_aggregate)
        direct.add(target)
        detail = {"aggregate": name.lower(), "function": function, "column": column}
    else:  # pragma: no cover - guarded above
        raise AssertionError(node_type)

    if root is None:  # pragma: no cover - every edit keeps a root
        raise AssertionError("edit removed the plan root")

    affected_existing = set(direct)
    for ancestor in ancestors:
        affected_existing.add(ancestor.id)

    old_versions = {node.id: node.version for node in _walk(plan.root)}
    refreshed = _refresh_affected(root, old_versions, affected_existing)
    temp_tree = replace(plan, root=refreshed)
    temp_tree = replace(
        temp_tree,
        root=_attach_estimates_affected(refreshed, plan.table, plan.scale_factor, affected_existing),
    )

    propagation: list[dict[str, str]] = []
    path_ids = [node.id for node in path]
    for index in range(len(path_ids) - 2, -1, -1):
        child_id = path_ids[index + 1]
        if child_id in removed:
            child_id = next(iter(direct)) if len(direct) == 1 else child_id
        propagation.append({"node": path_ids[index], "reason": f"descendant_changed:{child_id}"})

    affected_all = affected_existing | set(removed)
    return temp_tree.root, direct, removed, propagation, detail


def _walk(node: PlanNode) -> Iterable[PlanNode]:
    yield node
    for child in node.children:
        yield from _walk(child)


def _refresh(node: PlanNode, old_versions: Mapping[str, int], affected: set[str]) -> PlanNode:
    children = tuple(_refresh(child, old_versions, affected) for child in node.children)
    previous_version = old_versions.get(node.id, 1)
    version = previous_version + 1 if node.id in affected and node.id in old_versions else previous_version
    return make_node(
        node.id,
        node.kind,
        node.props,
        children,
        version=version,
    )


def _refresh_affected(node: PlanNode, old_versions: Mapping[str, int], affected: set[str]) -> PlanNode:
    """Rebuild only nodes on the edited subtree's ancestor closure.

    Descendant subtrees outside ``affected`` are returned by identity, so their
    summaries, versions and serialised bytes cannot be accidentally rebuilt.
    """

    if node.id not in affected:
        return node
    children = tuple(_refresh_affected(child, old_versions, affected) for child in node.children)
    previous_version = old_versions.get(node.id, 1)
    version = previous_version + 1 if node.id in old_versions else previous_version
    return make_node(
        node.id,
        node.kind,
        node.props,
        children,
        version=version,
    )


def _attach_estimates_affected(
    node: PlanNode,
    table: TableSnapshot,
    scale: float,
    affected: set[str],
) -> PlanNode:
    if node.id not in affected:
        return node
    children = tuple(_attach_estimates_affected(child, table, scale, affected) for child in node.children)
    provisional = replace(node, children=children)
    rows = _rows_for_node(provisional, table)
    return replace(provisional, estimated_rows=round(len(rows) * scale, 9))


def plan_diff(old: PlanTree | None, new: PlanTree) -> dict[str, Any]:
    old_nodes = old.nodes if old is not None else {}
    new_nodes = new.nodes
    added = stable_sorted(set(new_nodes) - set(old_nodes))
    removed = stable_sorted(set(old_nodes) - set(new_nodes))
    changed = stable_sorted(
        node_id
        for node_id in set(old_nodes) & set(new_nodes)
        if old_nodes[node_id].summary != new_nodes[node_id].summary
    )
    return {
        "added": added,
        "removed": removed,
        "summary_changed": changed,
        "changed_details": [
            {
                "node_id": node_id,
                "old_summary": old_nodes[node_id].summary,
                "new_summary": new_nodes[node_id].summary,
                "old_version": old_nodes[node_id].version,
                "new_version": new_nodes[node_id].version,
            }
            for node_id in changed
        ],
    }


def apply_edit(plan: PlanTree, operation: Mapping[str, Any], edit_seq: int) -> tuple[PlanTree, dict[str, Any]]:
    old_root = plan.root
    new_root, direct, removed, propagation, detail = _apply_transform(plan, operation, edit_seq)
    new_tree = replace(plan, root=new_root)
    effect = _effects_payload(
        edit_seq,
        operation,
        direct,
        removed,
        propagation,
        set(direct) | set(removed) | {entry["node"] for entry in propagation},
    )
    effect["detail"] = detail
    effect["changed_nodes"] = plan_diff(replace(plan, root=old_root), new_tree)
    affected = set(effect["affected_nodes"])
    old_nodes = plan.nodes
    new_nodes = new_tree.nodes
    unchanged = [
        {
            "node_id": node_id,
            "summary": new_nodes[node_id].summary,
            "version": new_nodes[node_id].version,
            "instance_preserved": old_nodes[node_id] is new_nodes[node_id],
        }
        for node_id in stable_sorted(set(old_nodes) & set(new_nodes) - affected)
    ]
    for entry in unchanged:
        node_id = entry["node_id"]
        if old_nodes[node_id].summary != new_nodes[node_id].summary or old_nodes[node_id].version != new_nodes[node_id].version:
            raise AppError(
                E_DIVERGE,
                "unaffected subtree changed during incremental edit",
                edit_seq=edit_seq,
                node_id=node_id,
            )
        if not entry["instance_preserved"]:
            raise AppError(
                E_DIVERGE,
                "unaffected subtree instance was replaced during incremental edit",
                edit_seq=edit_seq,
                node_id=node_id,
            )
    effect["unchanged_nodes"] = unchanged
    return new_tree, effect


def plan_from_dict(data: Mapping[str, Any], query_name: str, table: TableSnapshot, scale_factor: float) -> PlanTree:
    def from_node(node_data: Mapping[str, Any]) -> PlanNode:
        children = tuple(from_node(child) for child in node_data.get("children", []))
        return PlanNode(
            id=str(node_data["id"]),
            kind=str(node_data["kind"]),
            props=copy.deepcopy(node_data.get("props", {})),
            children=children,
            version=int(node_data["version"]),
            summary=str(node_data["summary"]),
            estimated_rows=float(node_data.get("estimated_rows", 0.0)),
        )

    root_data = data.get("root", data)
    return PlanTree(root=from_node(root_data), query_name=query_name, table=table, scale_factor=scale_factor)


def _first_node_difference(left: PlanNode | None, right: PlanNode | None) -> dict[str, Any] | None:
    if left is None and right is None:
        return None
    if left is None:
        return {"node_id": right.id if right else "$", "reason": "node_missing_on_left", "left": None, "right": node_to_dict(right) if right else None}
    if right is None:
        return {"node_id": left.id, "reason": "node_missing_on_right", "left": node_to_dict(left), "right": None}
    if left.id != right.id:
        return {"node_id": left.id, "reason": "node_id", "left": left.id, "right": right.id}
    # Compare node-local structure before derived summaries. This locates the
    # edit source (for example ``scan``) rather than an ancestor whose only
    # difference is propagation of its child's changed summary/version.
    comparisons = (
        ("kind", left.kind, right.kind),
        ("props", left.props, right.props),
    )
    for field, left_value, right_value in comparisons:
        if left_value != right_value:
            return {
                "node_id": left.id,
                "reason": field,
                "left": left_value,
                "right": right_value,
            }
    shared = min(len(left.children), len(right.children))
    for index in range(shared):
        difference = _first_node_difference(left.children[index], right.children[index])
        if difference is not None:
            return difference
    if len(left.children) != len(right.children):
        return {
            "node_id": left.id,
            "reason": "child_count",
            "left": len(left.children),
            "right": len(right.children),
        }
    # Derived fields are checked after the entire child subtree. If only these
    # fields differ, the current node is the first inconsistent node.
    for field, left_value, right_value in (
        ("version", left.version, right.version),
        ("estimated_rows", left.estimated_rows, right.estimated_rows),
        ("summary", left.summary, right.summary),
    ):
        if left_value != right_value:
            return {
                "node_id": left.id,
                "reason": field,
                "left": left_value,
                "right": right_value,
            }
    return None


def first_plan_difference(left: PlanTree, right: PlanTree) -> dict[str, Any] | None:
    if left.to_bytes() == right.to_bytes():
        return None
    difference = _first_node_difference(left.root, right.root)
    return difference or {"node_id": "$", "reason": "plan_bytes", "left": left.summary, "right": right.summary}


def first_plan_bytes_difference(left_bytes: bytes, right_bytes: bytes) -> dict[str, Any] | None:
    if left_bytes == right_bytes:
        return None
    import json

    try:
        left_data = json.loads(left_bytes.decode("utf-8"))
        right_data = json.loads(right_bytes.decode("utf-8"))
        difference = _first_node_difference(_node_from_serialized(left_data["root"]), _node_from_serialized(right_data["root"]))
        return difference or {"node_id": "$", "reason": "plan_bytes"}
    except Exception as exc:  # corrupted bytes still need a useful location
        return {"node_id": "$", "reason": "plan_bytes_unreadable", "error": str(exc)}


def _node_from_serialized(node_data: Mapping[str, Any]) -> PlanNode:
    return PlanNode(
        id=str(node_data["id"]),
        kind=str(node_data["kind"]),
        props=copy.deepcopy(node_data.get("props", {})),
        children=tuple(_node_from_serialized(child) for child in node_data.get("children", [])),
        version=int(node_data.get("version", 0)),
        summary=str(node_data.get("summary", "")),
        estimated_rows=float(node_data.get("estimated_rows", 0.0)),
    )


def verify_incremental_against_full(incremental: PlanTree, full: PlanTree, edit_seq: int | None = None) -> dict[str, Any]:
    difference = first_plan_difference(full, incremental)
    if difference is not None:
        raise AppError(
            E_DIVERGE,
            "incremental plan differs from full recompilation",
            edit_seq=edit_seq,
            node_id=difference.get("node_id"),
            details={"first_divergence": difference, "incremental_summary": incremental.summary, "full_summary": full.summary},
        )
    return {
        "status": "MATCH",
        "incremental_summary": incremental.summary,
        "full_summary": full.summary,
        "node_count": len(incremental.nodes),
    }
