"""层摘要的确定性计算（调度器与重放共用的唯一规则来源）。

层摘要分两段，组合规则全部定义在 :mod:`layerbench.digest`：

precursor（前驱指纹，执行前可算）
    hash(
      ("type", 指令类型),
      ("params", 规范化参数的稳定 JSON),
      ("inputs", 按路径稳定序的 {路径: 文件摘要} 集合),
      ("predecessors", 按 node_uid 稳定序的前驱层摘要集合),
    )

layer_digest（层摘要）
    - 执行类 run：hash(("precursor", precursor), ("result", 命令结果摘要))
    - 其余指令：hash(("precursor", precursor), ("result", "no-command-result"))

命令结果摘要
    hash(("exit_code", 退出码), ("stdout", 标准化换行后的 stdout 摘要))

前置集合包含：阶段内显式 after、阶段根指令对上游阶段根的跨阶段依赖。
所有集合先排序再哈希，故结果与线程交错、就绪顺序、缓存命中与否无关。
"""
from __future__ import annotations

from . import digest
from .manifest import Manifest, Instruction, input_file_digests

NO_COMMAND_RESULT = "no-command-result"


def command_result_digest(exit_code: int, stdout_digest: str) -> str:
    return digest.combine([
        ("exit_code", int(exit_code)),
        ("stdout", stdout_digest),
    ])


def precursor_digest(instruction: Instruction, manifest: Manifest,
                     predecessor_layers: dict[str, str]) -> tuple[str, dict]:
    """计算前驱指纹；同时返回输入文件摘要集合（稳定序）。"""
    inputs = input_file_digests(instruction, manifest.workspace)
    predecessor_part = sorted(
        (uid, layer) for uid, layer in predecessor_layers.items()
    )
    precursor = digest.combine([
        ("type", instruction.type),
        ("params", instruction.params),
        ("inputs", inputs),
        ("predecessors", predecessor_part),
    ])
    return precursor, inputs


def layer_digest_non_run(precursor: str) -> str:
    return digest.combine([
        ("precursor", precursor),
        ("result", NO_COMMAND_RESULT),
    ])


def layer_digest_run(precursor: str, result_digest: str) -> str:
    return digest.combine([
        ("precursor", precursor),
        ("result", result_digest),
    ])
