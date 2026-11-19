# 技术说明

## 模块边界

| 模块 | 职责 |
|---|---|
| `story_workbench/spec.py` | UTF-8 YAML 解析、重复键拒绝、严格字段校验、表达式/分量/阈值校验、DAG 检查 |
| `story_workbench/expressions.py` | 受限表达式词法/语法、变量白名单、规范渲染、系数定位、binary64 求值 |
| `story_workbench/engine.py` | 条件裕度、分支状态、效果组裁决、逐分量因果链、节点出边投影 |
| `story_workbench/timeline.py` | 不可变 `Snapshot`、时间轴推进、版本戳、状态/重放哈希 |
| `story_workbench/merge.py` | 所有简单路径枚举、独立累积、逐分量成对差异、首个分歧定位 |
| `story_workbench/dependency.py` | 参数所有者、反向依赖、精确闭包、最小重算、复用版本记录 |
| `story_workbench/injection.py` | 可序列化注入、规格副本修补、条件翻转定位 |
| `story_workbench/samples.py` | 固定样本与注入配置生成 |
| `story_workbench/demo.py` | 一键演示、逐节点哈希清单、两次构建比较、报告输出 |
| `story_workbench/cli.py` | 命令行分发与结构化错误 |
| `story_workbench/gui.py` | 只读版本化快照的 Tkinter 时间轴 |

## 数据流

```text
YAML bytes
  -> duplicate-key-safe SafeLoader
  -> strict schema validation
  -> expression AST + variable whitelist
  -> DAG validation
  -> immutable NarrativeSpec
  -> SequenceSample
  -> effect evaluation / last-writer arbitration
  -> immutable versioned Snapshot chain
  -> timeline / merge / dependency / injection reports
  -> canonical JSON + SHA-256 manifests
```

核心模型均为 frozen dataclass，状态、因果链、分支、裁决和快照使用 tuple。GUI 和报告不持有可变的运行时字典作为状态真源。

## 效果执行顺序

对一个效果组（正常边执行时是一组一条边；节点投影时是多条出边）：

1. 按边声明顺序、边内效果声明顺序枚举效果。
2. 所有 RHS 针对同一份执行前状态求值。
3. 对每个目标分量记录最后一个写入者。
4. 按同一枚举顺序，仅将生效写入者加到旧值。
5. 为目标分量追加 `CausalContribution`。
6. 为每个被覆盖效果生成 `Arbitration`。

该规则同时避免 YAML 映射键顺序依赖、字典迭代顺序依赖和非生效效果污染因果链。

## 条件裕度

```python
signed_margin = value - threshold
directed_margin = signed_margin       # >, >=
directed_margin = -signed_margin     # <, <=
boundary = abs(signed_margin) < boundary_tolerance
```

分支状态：

- 任一条件 `unsatisfied`：`unreachable`
- 无硬失败但任一条件 `boundary`：`boundary`
- 全部满足且无边界：`reachable`

边界清单不是旁路日志，而是快照内容哈希的一部分。快照另外暴露同内容、同版本戳和同内容哈希的 `boundary_manifest`，便于验收时直接关联。

## 哈希定义

- 规格哈希：`NarrativeSpec.canonical_dict()` 的规范化 JSON 的 SHA-256。
- 状态哈希：按 `components` 声明顺序，将 UTF-8 分量名、NUL 分隔符与 8 字节大端 binary64 依次喂入 SHA-256。
- 边序哈希：边 ID 以换行连接后的 UTF-8 SHA-256。
- 重放哈希：边序哈希 + `NUL` + 状态哈希字节流的 SHA-256。
- 快照内容哈希：快照核心 payload（不含内容哈希与版本戳自身）的规范化 JSON SHA-256。
- 版本戳：`state-v{四位版本}:{内容哈希前16位}`。
- 节点内容哈希：节点值规范化 JSON 的 SHA-256。

规范化 JSON 使用 UTF-8、键排序、无多余空白、禁止 NaN/Infinity。所有输出文件固定 `\n` 换行且无时间戳。

## 多路径核对

对每个入度大于 1 的节点，从 `start` 枚举所有不重复节点的简单路径，按边声明序号排序。每条路径独立从初值执行效果，保存：

- 完整边序列与边序号；
- 每一步执行前状态、效果候选、生效效果和裁决；
- 路径条件可满足性；
- 终态、状态哈希与因果链。

只对条件可行的路径做数值比较，但报告保留全部结构路径。所有可行路径两两比较；对每对路径按 `components` 声明顺序报告差异。首个超差记录包含分量、两条边序列、两侧状态、绝对差、容差和重算公式。报告不选“最后路径”或“最长路径”作为结论。

## 依赖图模型

每个序列单独构图。节点 ID 稳定且带上下文，例如：

- `edge_effect:e_intro`
- `raw_state:normal_path:1`
- `condition:normal_path:1:e_ally:c_trust_ally`
- `path_state:normal_path:1`
- `ending:normal_path`

边效果节点拥有系数键与效果集合键；条件节点拥有阈值键；初值节点拥有初值键。`path_state` 依赖原始状态、该步所有条件判定和该节点所有出边效果节点，因此快照中的裕度与出边投影都能被精确失效。后续原始状态依赖所选边的效果、所选边的条件与前一状态。

## 闭包算法

```text
owners = key_owners[trigger_key]
closure = reachable(owners, reverse_dependencies)  # 包含 owners
for node in topological_order where node in closure:
    recompute(node)
assert set(actual_recomputed) == set(closure)
```

图拓扑只在初始化时构建一次。参数修改只更新规格值并重算闭包，不重建拓扑、不重置闭包外缓存。报告的 `reused_sources` 列出被闭包节点读取但未重算的依赖及其版本/哈希。

## 错误码

| 错误码 | 含义 |
|---|---|
| `E_SPEC_INVALID` | YAML/字段/变量/分量/成环/非有限阈值或初值错误，位置包含节点/边/效果/条件序号 |
| `E_SPEC_IO` | 规格读取失败 |
| `E_SEQUENCE_INVALID` | 选择序列结构不连续 |
| `E_STATE_NONFINITE` | 执行中产生非有限值 |
| `E_INJECTION_INVALID` | 注入配置或目标无效 |
| `E_DEPENDENCY_INVALID` | 未知触发键或依赖图不一致 |
| `E_NONDETERMINISTIC` | 两次构建字节不一致 |
| `E_GUI_UNAVAILABLE` | 当前 Python 无 Tkinter |

## 验证

测试集中在 `tests/`，使用标准库 `unittest`，不要求第三方测试框架。关键断言包括：

- 未知分量、未知变量、非有限阈值、成环、重复 YAML 键均结构化失败并带位置。
- 两次重放的状态字节、边序哈希、状态哈希、重放哈希和快照哈希完全一致。
- 同边最后写者、预执行 RHS 求值和逐分量因果链。
- 一致汇合与超差汇合的首个分歧、两侧边序列和可复算步骤。
- 边界判定的单独状态、版本戳和内容哈希清单。
- 四种注入均可序列化、仅修改副本、可重放；阈值注入首个翻转定位。
- 系数/阈值闭包精确、实际重算集合等于闭包、闭包外哈希和版本不变。
- 两次样本生成、两次报告构建、两个输出目录的全部报告文件逐字节一致。
