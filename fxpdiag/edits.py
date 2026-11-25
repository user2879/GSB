"""编辑命令流应用：命令只使真正受影响的行缓存失效。"""

from __future__ import annotations

import math

from .errors import ErrorCode
from .models import (
    EditCommand,
    V1WeightRow,
    V2WeightRow,
)
from .pipeline import PipelineState


def apply_command(state: PipelineState, command: EditCommand) -> set[tuple[str, int]]:
    """应用命令并返回受影响行集合（账本据此区分复用与重算）。"""

    payload = command.payload
    if command.kind == "switch_version":
        target = payload["version"]
        if target not in ("v1", "v2"):
            state.failures.add(
                ErrorCode.E_COMMAND_UNKNOWN, "command-stream",
                f"未知交换版本 {target}", {"seq": command.seq},
            )
            return set()
        state.active_version = target
        # 版本切换影响全部行
        affected = {
            (layer, row_index)
            for layer, matrix in state.package.layers.items()
            for row_index in range(len(matrix))
        }
        return affected

    if command.kind == "replace_layer_params":
        layer = payload["layer"]
        affected = state.affected_rows(layer, None)
        rows = len(state.package.layers[layer])
        for row_index in range(rows):
            key = (layer, row_index)
            if state.active_version == "v1":
                state.weights.v1_rows[key] = V1WeightRow(
                    layer=layer,
                    row_index=row_index,
                    scale_permille=int(payload["scale_permille"]),
                    bias_code=int(payload.get("bias_code", 0)),
                    bit_width=int(payload["bit_width"]),
                )
            else:
                state.weights.v2_rows[key] = V2WeightRow(
                    layer=layer,
                    row_index=row_index,
                    scale=float(payload["scale"]),
                    scheme=payload["scheme"],
                    rounding_mode=payload["rounding_mode"],
                )
        return affected

    if command.kind == "replace_row_activation":
        layer = payload["layer"]
        row_index = int(payload["row_index"])
        new_row = [float(value) for value in payload["activation"]]
        if any(not math.isfinite(value) for value in new_row):
            state.failures.add(
                ErrorCode.E_NON_FINITE, "command-stream",
                "替换的激活含非有限值",
                {"seq": command.seq, "layer": layer, "row": row_index},
            )
            return set()
        if len(new_row) != state.row_width(layer):
            state.failures.add(
                ErrorCode.E_SHAPE_INVALID, "command-stream",
                "替换激活列数与原张量不一致",
                {"seq": command.seq, "layer": layer, "row": row_index},
            )
            return set()
        state.package.layers[layer][row_index] = new_row
        return state.affected_rows(layer, row_index)

    if command.kind == "set_frac_bits":
        state.config.frac_bits = int(payload["frac_bits"])
        return _all_rows(state)

    if command.kind == "set_tier":
        state.config.tier = payload["tier"]
        return _all_rows(state)

    state.failures.add(
        ErrorCode.E_COMMAND_UNKNOWN, "command-stream",
        f"未知编辑命令 {command.kind}", {"seq": command.seq},
    )
    return set()


def _all_rows(state: PipelineState) -> set[tuple[str, int]]:
    return {
        (layer, row_index)
        for layer, matrix in state.package.layers.items()
        for row_index in range(len(matrix))
    }


def run_command_sequence(
    state: PipelineState,
    commands: list[EditCommand],
    runner_factory,
    parallel: bool = True,
) -> list[dict]:
    """逐条施加命令并增量运行，返回每步快照（含账本与失败清单顺序）。"""

    snapshots: list[dict] = []
    runner = runner_factory(state)
    snapshots.append({"seq": 0, "event": "initial", **runner.execute(parallel=parallel)})
    for command in commands:
        apply_command(state, command)
        runner = runner_factory(state)
        artifacts = runner.execute(force_recompute=False, parallel=parallel)
        snapshots.append({"seq": command.seq, "event": command.kind, **artifacts})
    return snapshots
