"""加载校验层：张量包、算子配置、编辑命令流与层间引用无环校验。"""

from __future__ import annotations

import math

from .errors import ErrorCode, FailureList, PipelineError
from .fixedpoint import ALLOWED_FRAC_BITS, TIER_GUARD_BITS
from .models import EditCommand, OpConfig, TensorPackage
from .models import NAMED_ROUNDING_MODES

VALID_COMMAND_KINDS = {
    "switch_version",
    "replace_layer_params",
    "replace_row_activation",
    "set_frac_bits",
    "set_tier",
}


def validate_tensor_package(
    package: TensorPackage,
    source: str,
    failures: FailureList,
    duplicate_injected: set[str] | None = None,
) -> bool:
    ok = True
    seen: set[str] = set()
    duplicate_injected = duplicate_injected or set()
    for layer, matrix in package.layers.items():
        if layer in seen or layer in duplicate_injected:
            failures.add(
                ErrorCode.E_DUPLICATE_LAYER,
                source,
                f"层名重复: {layer}",
                {"layer": layer},
            )
            ok = False
        seen.add(layer)
        if (
            not isinstance(matrix, list)
            or not matrix
            or any(not isinstance(row, list) or not row for row in matrix)
        ):
            failures.add(
                ErrorCode.E_SHAPE_INVALID, source,
                f"层 {layer} 的行列数必须为正整数", {"layer": layer},
            )
            ok = False
            continue
        width = len(matrix[0])
        for row_index, row in enumerate(matrix):
            if len(row) != width:
                failures.add(
                    ErrorCode.E_SHAPE_INVALID, source,
                    f"层 {layer} 第 {row_index} 行列数不一致",
                    {"layer": layer, "row": row_index},
                )
                ok = False
            for col_index, value in enumerate(row):
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    failures.add(
                        ErrorCode.E_NON_FINITE, source,
                        f"激活非有限值: 层 {layer} 行 {row_index} 列 {col_index}",
                        {"layer": layer, "row": row_index, "col": col_index},
                    )
                    ok = False
    ok = _validate_dependencies(package, source, failures) and ok
    return ok


def _validate_dependencies(
    package: TensorPackage, source: str, failures: FailureList
) -> bool:
    ok = True
    for layer, deps in package.depends_on.items():
        for dep in deps:
            if dep not in package.layers:
                failures.add(
                    ErrorCode.E_UNKNOWN_REF, source,
                    f"层 {layer} 引用了不存在的上游层 {dep}",
                    {"layer": layer, "ref": dep},
                )
                ok = False
    # 有环检测：三色 DFS
    color: dict[str, int] = {layer: 0 for layer in package.layers}

    def visit(node: str, stack: list[str]) -> bool:
        color[node] = 1
        stack.append(node)
        for dep in package.depends_on.get(node, []):
            if dep not in color:
                continue
            if color[dep] == 1:
                cycle = " -> ".join(stack[stack.index(dep):] + [dep])
                failures.add(
                    ErrorCode.E_LAYER_CYCLE, source,
                    f"层间引用存在环: {cycle}", {"cycle": cycle},
                )
                return False
            if color[dep] == 0 and not visit(dep, stack):
                return False
        stack.pop()
        color[node] = 2
        return True

    for layer in sorted(package.layers):
        if color[layer] == 0 and not visit(layer, []):
            ok = False
    return ok


def validate_op_config(config: OpConfig, source: str) -> None:
    if config.frac_bits not in ALLOWED_FRAC_BITS:
        raise PipelineError(
            ErrorCode.E_SHAPE_INVALID,
            f"定点小数位宽 {config.frac_bits} 超出 1..15",
            {"source": source},
        )
    if config.tier not in TIER_GUARD_BITS:
        raise PipelineError(
            ErrorCode.E_SHAPE_INVALID, f"未知量化档位 {config.tier}",
            {"source": source},
        )
    if config.rounding_mode not in NAMED_ROUNDING_MODES:
        raise PipelineError(
            ErrorCode.E_SHAPE_INVALID,
            f"未知舍入模式 {config.rounding_mode}",
            {"source": source},
        )
    if config.max_workers < 1 or config.element_budget < 1 or config.layer_row_quota < 1:
        raise PipelineError(
            ErrorCode.E_SHAPE_INVALID,
            "max_workers/element_budget/layer_row_quota 必须为正整数",
            {"source": source},
        )


def validate_commands(
    commands: list[EditCommand], source: str, failures: FailureList
) -> list[EditCommand]:
    valid: list[EditCommand] = []
    for command in commands:
        if command.kind not in VALID_COMMAND_KINDS:
            failures.add(
                ErrorCode.E_COMMAND_UNKNOWN, source,
                f"未知编辑命令 {command.kind}",
                {"seq": command.seq, "kind": command.kind},
            )
            continue
        valid.append(command)
    return valid
