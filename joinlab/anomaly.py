from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from .errors import fail
from .logging_utils import atomic_write_text


def _read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        fail("E_TABLE", "异常注入读取输入失败", "workspace", file=str(path), reason=str(exc))


def clone_workspace(workspace_path: str | Path, output_dir: str | Path) -> Path:
    source = Path(workspace_path).resolve()
    document = _read_json(source)
    if not isinstance(document, dict):
        fail("E_TABLE", "工作区 JSON 顶层必须是对象", "$")
    target_dir = Path(output_dir).resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    target_workspace = target_dir / source.name
    atomic_write_text(target_workspace, json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    for table in document.get("tables", []):
        if not isinstance(table, dict) or not isinstance(table.get("data_file"), str):
            continue
        relative = Path(table["data_file"])
        source_data = (source.parent / relative).resolve()
        target_data = (target_dir / relative).resolve()
        target_data.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_data, target_data)
    return target_workspace


def _find_table(document: dict[str, Any], table_name: str) -> tuple[int, dict[str, Any]]:
    for index, table in enumerate(document.get("tables", [])):
        if isinstance(table, dict) and table.get("name") == table_name:
            return index, table
    fail("E_TABLE", "异常注入找不到目标表", "tables", table=table_name)
    raise AssertionError("unreachable")


def _key_column_index(table: dict[str, Any], key: str) -> int:
    for index, column in enumerate(table.get("columns", [])):
        if isinstance(column, dict) and column.get("name") == key:
            return index
    fail("E_TABLE", "异常注入找不到联接键列", "tables.columns", key=key)
    raise AssertionError("unreachable")


def inject_skew(workspace_path: str | Path, output_dir: str | Path) -> Path:
    target_workspace = clone_workspace(workspace_path, output_dir)
    document = _read_json(target_workspace)
    join = document.get("join")
    if not isinstance(join, dict):
        fail("E_TABLE", "异常注入缺少 join", "join")
    table_index, table = _find_table(document, join["right_table"])
    key_index = _key_column_index(table, join["right_key"])
    if isinstance(table.get("rows"), list):
        rows = table["rows"]
    else:
        data_path = (target_workspace.parent / table["data_file"]).resolve()
        row_document = _read_json(data_path)
        if isinstance(row_document, dict):
            rows = row_document["rows"]
        else:
            rows = row_document
    if not rows:
        fail("E_TABLE", "倾斜注入目标表为空", f"tables[{table_index}].rows")
    anchor = None
    for row in rows:
        value = row[key_index] if isinstance(row, list) else row[join["right_key"]]
        if value is not None:
            anchor = value
            break
    if anchor is None:
        fail("E_TABLE", "倾斜注入找不到非空键", f"tables[{table_index}].rows")
    for index, row in enumerate(rows):
        if index % 5 == 4:
            continue
        if isinstance(row, list):
            row[key_index] = anchor
        else:
            row[join["right_key"]] = anchor
    table["skew"] = True
    if not isinstance(table.get("rows"), list):
        if isinstance(row_document, dict):
            row_document["rows"] = rows
            row_document_to_write = row_document
        else:
            row_document_to_write = rows
        atomic_write_text(data_path, json.dumps(row_document_to_write, ensure_ascii=False, indent=2) + "\n")
    atomic_write_text(target_workspace, json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    return target_workspace


def inject_type_mismatch(workspace_path: str | Path, output_dir: str | Path) -> Path:
    target_workspace = clone_workspace(workspace_path, output_dir)
    document = _read_json(target_workspace)
    join = document.get("join")
    if not isinstance(join, dict):
        fail("E_TABLE", "异常注入缺少 join", "join")
    table_index, table = _find_table(document, join["left_table"])
    key_index = _key_column_index(table, join["left_key"])
    table["columns"][key_index]["type"] = "text"
    atomic_write_text(target_workspace, json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    return target_workspace


def inject_snapshot_missing_version(snapshot_path: str | Path, output_path: str | Path) -> Path:
    document = _read_json(Path(snapshot_path))
    if not isinstance(document, dict):
        fail("E_SNAP", "快照顶层必须是对象", "snapshot")
    document.pop("params_version", None)
    target = Path(output_path).resolve()
    atomic_write_text(target, json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    return target

