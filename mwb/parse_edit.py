"""编辑脚本解析：算子白名单与字段结构校验。

引用目标是否存在（音符 / 参数）在 :mod:`mwb.engine` 的干跑重演中
结合乐谱与表达参数做语义校验，非法时返回 E_EDIT_TARGET。
"""
from __future__ import annotations

from .atomicio import read_json
from .errors import E_EDIT_OP, E_SCHEMA, WorkbenchError
from .models import EDIT_OPS, Edit

_REQUIRED_ARGS = {
    "tempo_upsert": {"beat", "bpm"},
    "tempo_delete": {"beat"},
    "velocity_upsert": {"x", "y"},
    "velocity_delete": {"x"},
    "stretch_set": {"name", "factor"},
    "articulation_set": {"name", "gate"},
    "note_articulation_set": {"note_id", "articulation"},
    "pedal_add": {"pedal_id", "start_beat", "end_beat"},
    "pedal_remove": {"pedal_id"},
}


def parse_edit_script(path: str) -> tuple[list[Edit], int, int]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise WorkbenchError(E_SCHEMA, "编辑脚本根节点必须是对象")
    raw_edits = data.get("edits", [])
    if not isinstance(raw_edits, list):
        raise WorkbenchError(E_SCHEMA, "edits 必须为数组")

    edits: list[Edit] = []
    for index, raw in enumerate(raw_edits):
        if not isinstance(raw, dict):
            raise WorkbenchError(E_SCHEMA, f"edits[{index}] 必须是对象")
        op = raw.get("op")
        if op not in EDIT_OPS:
            raise WorkbenchError(E_EDIT_OP, "编辑算子类型不在允许集合内", {"index": index, "op": op})
        args = raw.get("args", {})
        if not isinstance(args, dict):
            raise WorkbenchError(E_SCHEMA, f"edits[{index}].args 必须是对象")
        missing = _REQUIRED_ARGS[op] - set(args)
        if missing:
            raise WorkbenchError(E_SCHEMA, f"算子 {op} 缺少参数", {"missing": sorted(missing)})
        edits.append(Edit(step=index, op=op, args=args))

    concurrency = data.get("concurrency", 1)
    if not isinstance(concurrency, int) or isinstance(concurrency, bool) or concurrency < 1:
        raise WorkbenchError(E_SCHEMA, "concurrency 必须为 >=1 的整数")
    memory_budget = data.get("memory_budget_chunks", concurrency)
    if not isinstance(memory_budget, int) or isinstance(memory_budget, bool) or memory_budget < 1:
        raise WorkbenchError(E_SCHEMA, "memory_budget_chunks 必须为 >=1 的整数")
    return edits, concurrency, memory_budget
