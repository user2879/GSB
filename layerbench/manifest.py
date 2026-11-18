"""多阶段构建清单的数据模型、JSON 加载与强校验。

校验失败统一抛 :class:`ManifestError`（业务码 E_MANIFEST），
实现上先收集全部问题再一次性抛出，保证「整体拒绝、不部分生效」。
出错定位精确到阶段名与指令序号（seq 从 1 开始，与清单中顺序一致）。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from . import digest
from .errors import ManifestError

INSTRUCTION_TYPES = ("copy", "run", "env", "label")
REQUIRED_FIELDS = {
    "copy": ("src", "dst"),
    "run": ("command",),
    "env": ("name", "value"),
    "label": ("name", "value"),
}
# 仅 copy / run 可能携带输入文件；env、label 为纯元数据
RUN_INPUTS_FIELD = "inputs"


@dataclass(frozen=True)
class Instruction:
    seq: int                      # 1-based，阶段内序号
    type: str
    params: dict                  # 已规范化的参数字典
    after: tuple[str, ...]        # 仅影响本阶段的前置指令 id
    id: str
    stage: str

    def loc(self) -> str:
        return f"stage={self.stage} seq={self.seq} id={self.id}"


@dataclass
class Stage:
    name: str
    depends_on: tuple[str, ...]
    quota: int
    instructions: list[Instruction] = field(default_factory=list)


@dataclass
class Manifest:
    version: str
    global_slots: int
    default_timeout_ms: int
    cache_capacity: int
    stages: list[Stage]
    workspace: str                 # 绝对路径，复制源的根（工作区）
    source_path: str
    stage_by_name: dict[str, Stage] = field(default_factory=dict)
    topo_order: list[str] = field(default_factory=list)

    def all_instructions(self) -> list[Instruction]:
        result = []
        for stage_name in self.topo_order:
            result.extend(self.stage_by_name[stage_name].instructions)
        return result


def _is_positive_int(value: Any) -> bool:
    # 显式拒绝 bool（bool 是 int 子类）
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _topological_sort(stage_by_name: dict[str, Stage]) -> list[str]:
    """Kahn 算法；同层候选按名字排序，得到确定的拓扑序。"""
    indegree = {name: 0 for name in stage_by_name}
    adjacency: dict[str, list[str]] = {name: [] for name in stage_by_name}
    for name, stage in stage_by_name.items():
        for dep in stage.depends_on:
            indegree[name] += 1
            adjacency[dep].append(name)
    ready = sorted(n for n, d in indegree.items() if d == 0)
    order: list[str] = []
    while ready:
        current = ready.pop(0)
        order.append(current)
        for nxt in adjacency[current]:
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                ready.append(nxt)
                ready.sort()
    if len(order) != len(stage_by_name):
        cyclic = sorted(n for n, d in indegree.items() if d > 0)
        raise ManifestError(
            "阶段依赖存在环", location="stages",
            problems=[f"阶段依赖环涉及: {cyclic}"],
        )
    return order


def load_manifest(path: str, workspace: str | None = None) -> Manifest:
    """加载并强校验清单；任何非法项均以 E_MANIFEST 整体拒绝。"""
    source_path = os.path.abspath(path)
    problems: list[str] = []
    if not os.path.isfile(source_path):
        raise ManifestError(
            "清单文件不存在", location=source_path,
            problems=[f"文件不存在: {source_path}"],
        )
    try:
        with open(source_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ManifestError(
            "清单不是合法 JSON", location=source_path,
            problems=[f"JSON 解析失败: {exc}"],
        ) from exc
    if not isinstance(raw, dict):
        raise ManifestError(
            "清单顶层必须是对象", location=source_path,
            problems=["顶层类型为 " + type(raw).__name__],
        )

    workspace_abs = os.path.abspath(workspace) if workspace else os.path.dirname(source_path)
    version = raw.get("version")
    if not isinstance(version, str) or not version:
        problems.append("顶层字段 version 必须为非空字符串")
        version = version if isinstance(version, str) else "invalid"

    global_slots = raw.get("global_slots")
    if not _is_positive_int(global_slots):
        problems.append("global_slots 必须为正整数（不能为 0）")
    default_timeout_ms = raw.get("default_timeout_ms")
    if not _is_positive_int(default_timeout_ms):
        problems.append("default_timeout_ms 必须为正整数")
    cache_capacity = raw.get("cache_capacity")
    if not _is_positive_int(cache_capacity):
        problems.append("cache_capacity 必须为正整数")

    stages_raw = raw.get("stages")
    if not isinstance(stages_raw, list) or not stages_raw:
        raise ManifestError(
            "stages 必须为非空数组", location=source_path,
            problems=problems + ["stages 缺失或为空"],
        )

    stages: list[Stage] = []
    stage_by_name: dict[str, Stage] = {}
    seen_names: set[str] = set()
    for idx, stage_raw in enumerate(stages_raw):
        stage = _parse_stage(stage_raw, idx, workspace_abs, seen_names,
                             stage_by_name, problems)
        if stage is not None:
            stages.append(stage)
            stage_by_name[stage.name] = stage

    # 跨阶段依赖引用与无环检查
    for stage in stages:
        for dep in stage.depends_on:
            if dep not in stage_by_name:
                problems.append(
                    f"stage={stage.name}: depends_on 引用了不存在的阶段 '{dep}'"
                )
    if not problems:
        topo = _topological_sort(stage_by_name)
    else:
        topo = sorted(stage_by_name)

    if problems:
        raise ManifestError(
            "清单校验失败，整体拒绝加载（未产生任何部分生效状态）",
            location=source_path, problems=sorted(set(problems)),
        )

    return Manifest(
        version=version,
        global_slots=global_slots,
        default_timeout_ms=default_timeout_ms,
        cache_capacity=cache_capacity,
        stages=stages,
        workspace=workspace_abs,
        source_path=source_path,
        stage_by_name=stage_by_name,
        topo_order=topo,
    )


def _parse_stage(stage_raw: Any, idx: int, workspace: str,
                 seen_names: set[str], stage_by_name: dict[str, Stage],
                 problems: list[str]) -> Stage | None:
    if not isinstance(stage_raw, dict):
        problems.append(f"stages[{idx}] 必须为对象")
        return None
    name = stage_raw.get("name")
    if not isinstance(name, str) or not name:
        problems.append(f"stages[{idx}].name 必须为非空字符串")
        return None
    loc = f"stage={name}"
    if name in seen_names:
        problems.append(f"{loc}: 阶段名重复")
    seen_names.add(name)

    depends_raw = stage_raw.get("depends_on", [])
    if not isinstance(depends_raw, list) or not all(isinstance(d, str) for d in depends_raw):
        problems.append(f"{loc}: depends_on 必须为字符串数组")
        depends_on: tuple[str, ...] = ()
    else:
        depends_on = tuple(depends_raw)

    quota = stage_raw.get("quota")
    if not _is_positive_int(quota):
        problems.append(f"{loc}: quota 必须为正整数")
        quota = quota if _is_positive_int(quota) else 1

    instructions_raw = stage_raw.get("instructions")
    if not isinstance(instructions_raw, list) or not instructions_raw:
        problems.append(f"{loc}: instructions 必须为非空数组")
        return Stage(name, depends_on, quota, [])

    stage = Stage(name, depends_on, quota, [])
    local_ids: set[str] = set()
    for seq, ins_raw in enumerate(instructions_raw, start=1):
        instruction = _parse_instruction(ins_raw, seq, stage, workspace,
                                         local_ids, problems)
        if instruction is not None:
            stage.instructions.append(instruction)
            local_ids.add(instruction.id)
    # after 引用闭合检查（必须引用本阶段已声明 id）
    for instruction in stage.instructions:
        for ref in instruction.after:
            if ref not in local_ids:
                problems.append(
                    f"{loc} seq={instruction.seq}: after 引用了本阶段不存在的指令 id '{ref}'"
                )
            if ref == instruction.id:
                problems.append(
                    f"{loc} seq={instruction.seq}: 指令不能把自身声明为前置"
                )
    return stage


def _parse_instruction(ins_raw: Any, seq: int, stage: Stage, workspace: str,
                       local_ids: set[str], problems: list[str]) -> Instruction | None:
    loc = f"stage={stage.name} seq={seq}"
    if not isinstance(ins_raw, dict):
        problems.append(f"{loc}: 指令必须为对象")
        return None
    ins_type = ins_raw.get("type")
    if ins_type not in INSTRUCTION_TYPES:
        problems.append(
            f"{loc}: type 必须是 {INSTRUCTION_TYPES} 之一，实际为 {ins_type!r}"
        )
        return None
    ins_id = ins_raw.get("id")
    if not isinstance(ins_id, str) or not ins_id:
        problems.append(f"{loc}: id 必须为非空字符串")
        ins_id = f"__anon_{seq}"
    elif ins_id in local_ids:
        problems.append(f"{loc}: 指令 id '{ins_id}' 在阶段内重复")

    after_raw = ins_raw.get("after", [])
    if not isinstance(after_raw, list) or not all(isinstance(a, str) for a in after_raw):
        problems.append(f"{loc}: after 必须为字符串数组")
        after: tuple[str, ...] = ()
    else:
        after = tuple(after_raw)

    params_raw = ins_raw.get("params")
    if not isinstance(params_raw, dict):
        problems.append(f"{loc}: params 必须为对象且字段完整")
        return Instruction(seq, ins_type, {}, after, ins_id, stage.name)

    params: dict[str, Any] = {}
    for field_name in REQUIRED_FIELDS[ins_type]:
        value = params_raw.get(field_name)
        if not isinstance(value, str) or not value:
            problems.append(f"{loc}: params.{field_name} 必须为非空字符串")
        else:
            params[field_name] = value

    # 规范化：只保留声明字段，剔除未知字段，保证摘要稳定
    normalized = {key: params[key] for key in sorted(params)}

    if ins_type == "copy":
        _check_copy_source(normalized, workspace, loc, problems)
    if ins_type == "run":
        extra = params_raw.get(RUN_INPUTS_FIELD)
        if extra is not None:
            if not isinstance(extra, list) or not all(isinstance(x, str) for x in extra):
                problems.append(f"{loc}: params.{RUN_INPUTS_FIELD} 必须为字符串数组")
            else:
                normalized[RUN_INPUTS_FIELD] = sorted(extra)
                for rel in normalized[RUN_INPUTS_FIELD]:
                    _check_workspace_file(rel, workspace, loc, problems)
        timeout = params_raw.get("timeout_ms")
        if timeout is not None and not _is_positive_int(timeout):
            problems.append(f"{loc}: params.timeout_ms 必须为正整数")
        elif isinstance(timeout, int):
            normalized["timeout_ms"] = timeout
    return Instruction(seq, ins_type, normalized, after, ins_id, stage.name)


def _check_copy_source(params: dict, workspace: str, loc: str,
                       problems: list[str]) -> None:
    src = params.get("src")
    if not src:
        return
    _check_workspace_file(src, workspace, loc, problems)


def _check_workspace_file(rel_path: str, workspace: str, loc: str,
                          problems: list[str]) -> None:
    # 禁止绝对路径与盘符，强制位于工作区内
    if os.path.isabs(rel_path) or (len(rel_path) >= 2 and rel_path[1] == ":"):
        problems.append(f"{loc}: 源路径必须是工作区内相对路径，非法值 '{rel_path}'")
        return
    candidate = os.path.abspath(os.path.join(workspace, rel_path))
    if os.path.commonpath([candidate, workspace]) != workspace:
        problems.append(f"{loc}: 源路径越出工作区边界 '{rel_path}'")
        return
    if not os.path.isfile(candidate):
        problems.append(f"{loc}: 复制源文件不存在 '{rel_path}'")


def input_file_digests(instruction: Instruction, workspace: str) -> dict[str, str]:
    """返回 {工作区相对路径: 文件 SHA256}，按路径稳定序排列。"""
    rel_paths: list[str] = []
    if instruction.type == "copy":
        rel_paths.append(instruction.params["src"])
    elif instruction.type == "run":
        rel_paths.extend(instruction.params.get(RUN_INPUTS_FIELD, []))
    result = {}
    for rel in sorted(set(rel_paths)):
        absolute = os.path.abspath(os.path.join(workspace, rel))
        result[rel] = digest.hash_file(absolute)
    return dict(sorted(result.items()))
