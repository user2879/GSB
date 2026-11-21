# opchain：确定性增量激活/重排算子链编辑器

本项目实现一个纯本地、无网络、无 GPU 的 Python 应用，用于编辑和增量执行确定性合成张量上的激活与转置重排算子链。项目包含可导入核心库、命令行入口、样本 JSON、快照/缓存工作区以及完整 pytest 验收测试。

## 快速开始

环境要求：Python >= 3.10、NumPy。测试需要 pytest。

```bash
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt

# 校验样本
python3 -m opchain validate \
  --chain samples/chain.json --input samples/input.json \
  --config samples/config.json --workspace .opchain-workspace

# 首次执行/编辑后重复执行；使用同一 workspace 启用增量缓存与续跑
python3 -m opchain run \
  --chain samples/chain.json --input samples/input.json \
  --config samples/config.json --workspace .opchain-workspace

# 端到端验收集
python3 -m pytest -q
python3 -m opchain demo --workspace .opchain-demo
```

安装为本地包后也可使用 `opchain` 命令。项目不访问网络、不使用容器、GPU 或外部账号。

## 目录

- `opchain/chain.py`：链 JSON 解析、拓扑检查、参数检查和形状/置换推导。
- `opchain/numeric.py`：四种激活、定点量化、精度舍入、融合边界和逐元素顺序元数据。
- `opchain/engine.py`：增量失效、缓存复用、稳定拓扑调度、预算、线程池汇总和结果生成。
- `opchain/cache.py`：内容寻址节点产物缓存。
- `opchain/snapshot.py`：阶段化快照、临时目录写入、整体替换和恢复判定。
- `opchain/edits.py`：参数、顺序、子图、精度、融合和原地标记编辑。
- `samples/`：本地样本链、输入张量描述和配置。
- `tests/`：错误码、数值、融合、增量、预算、快照和并发验收。

## JSON 输入

### 算子链

节点输出张量 ID 等于节点 ID。节点通过 `inputs`/`input` 或顶层 `edges` 指定输入；两种表示可同时使用，但每个节点必须恰好有一个输入张量。节点顺序数组还作为稳定拓扑排序的并列裁决顺序。

```json
{
  "input": "x",
  "outputs": ["perm1"],
  "nodes": [
    {"id": "relu1", "op": "relu", "params": {}, "inplace": false},
    {"id": "perm1", "op": "permute", "params": {"perm": [1, 0]}}
  ],
  "edges": [
    {"from": "x", "to": "relu1"},
    {"from": "relu1", "to": "perm1"}
  ]
}
```

支持算子：

- `relu`：可选 `negative_slope` 或 `alpha`，必须非负。
- `gelu`：`approximation` 为 `erf`（默认）或 `tanh`。
- `sigmoid`、`tanh`：无参数。
- `permute`：`perm`/`axes`/`dims` 给出 0 起始、无重复、覆盖全部维度的置换。
- `transpose`：二维时省略置换等价于 `[1, 0]`；也可显式给出完整置换。

### 输入张量描述

固定种子生成输入，再按索引写入特殊值。支持 `uniform`、`normal`、`integer`、`zeros`、`ones` 分布。

```json
{
  "name": "x",
  "seed": 20261122,
  "shape": [3, 4],
  "dtype": "float64",
  "distribution": {"kind": "uniform", "low": -2.0, "high": 2.0},
  "special_values": [
    {"index": [0, 0], "value": "nan"},
    {"index": [1, 1], "value": 0.0312500004}
  ]
}
```

### 执行配置

```json
{
  "precision_tier": "fp64",
  "quantization_tier": "q8",
  "thread_pool_size": 4,
  "node_recompute_budget": 100,
  "snapshot_interval": 1,
  "tolerance": 1e-9,
  "fusion_strategy": "auto"
}
```

- 精度档位：`fp64`（9 位十进制保留）、`fp32`（6 位）、`fp16`（3 位）。
- 定点档位：`none`、`q8`（scale 16）、`q12`（256）、`q16`（4096）、`int8`、`int16`。
- 融合策略：`none` 或 `auto`/`fused`。直接相连且激活只有一个重排消费者时，`auto` 融合。
- `node_recompute_budget`/`recompute_budget` 为节点级重算次数预算。

## 执行语义

### 激活与近似记录

ReLU、Sigmoid、Tanh、GELU 均按配置 dtype 和固定运算顺序逐元素执行。GELU 的 `erf` 路径使用固定系数的 Abramowitz–Stegun 7.1.26 多项式；`tanh` 路径使用固定 tanh 公式。Sigmoid 使用 `exp`，GELU 使用 `sqrt`/`erf` 或 `sqrt`/`tanh`。每个节点元数据显式记录：

- `approximation.exp`、`approximation.sqrt`、`approximation.erf`、`approximation.tanh` 的实现与版本；
- `precision.tier`、`precision.dtype`、`precision.decimal_places`；
- `quantization.tier`、`quantization.scale`；
- 逐元素 `element_order`（累积操作标识）与 `round_count`（中间舍入次数）。

ReLU/GELU 对 `NaN`、`+inf`、`-inf` 做了确定性保护，不会因 `0 * inf` 产生伪 NaN。

### 转置与原地写回

重排结果形状为 `tuple(input.shape[axis] for axis in perm)`，通过 `numpy.transpose` 实现，输出按输出索引字典序排列。原地节点静态检查其输入张量的所有消费者和链输出引用；存在其他引用时产生 `E_INPLACE_CONFLICT`，并在 `details.conflict_pairs` 记录有序节点对。合法的写回会更新结果中的 `tensor_registry`，展示输入张量已被原地产物覆盖。

### 融合路径与可归因差异

非融合路径在每个节点边界依次执行 `fixed quantization -> decimal rounding`；融合路径把激活和重排合并，在最终边界执行 `decimal rounding -> fixed quantization`。两种顺序不交换，因此同一链可能产生真实差异。样本中的特殊值 `0.0312500004` 在 `fp64 + q8` 下产生：

- 非融合：`0.0625`；
- 融合：`0.0`；
- 最大绝对差：`0.0625`。

融合节点元数据记录 `fusion.fused=true`、配对激活节点、`rounding_order=["decimal","fixed"]`、非融合对照顺序、舍入次数增量和逐元素累积操作标识；非融合产物记录 `rounding_order=["fixed","decimal"]`。样本中非融合重排的逐元素累计舍入次数为 4，融合重排为 2。

## 增量缓存、预算与一致性

缓存键由节点 ID/参数指纹、输入内容指纹、精度档位、量化档位、原地标记、算子实现版本和融合上下文派生。编辑后引擎比较前一状态的节点描述、节点顺序、输入描述和执行配置：

- 参数修改：该节点及全部下游失效；
- 顺序调整：使用序列最长公共子序列识别实际移动节点，标记该节点及下游；
- 子图替换：新增/移除节点、被重连下游失效；
- 精度/量化切换：全部节点失效；
- 未失效节点直接从内容寻址缓存加载，不重新计算。

调度器按稳定键 `[topological_rank, node_id]` 排序；线程池任务完成后仍按该键汇总。预算按稳定顺序消耗，超出预算的节点留在 `pending` 并在 `diagnostics` 中记录 `E_RECOMPUTE_BUDGET_PENDING`。下次运行优先处理待办节点。预算耗尽不会改动已复用产物。

`compare_execution_artifacts()` 对增量结果和从空工作区全量执行结果的逐节点 `values`、`values_hash`、`cache_key`、形状、dtype、全部累积顺序元数据、失败清单、待办清单和 `tensor_registry` 做逐字段比较。`invalidated`/`reused` 是本次运行的缓存调度诊断：冷全量执行必然将所有节点列为失效且复用为空，增量执行则只列出受影响节点；两者内容按需求定义本就不同，但分别以稳定顺序输出并可由测试断言。最终执行产物与失败语义保持一致。

## 快照与恢复

一次运行划分为链校验、逐节点执行、产物汇总阶段。每个快照先写入 `workspace/tmp/snapshot-*`，完成后通过目录重命名整体替换 `workspace/snapshots/current`，保留 `current.previous` 作为崩溃恢复备选。`manifest.json` 记录：

- `global_version`；
- `config_hash`；
- `chain_hash`、`input_hash`；
- `completed_nodes`；
- `artifact_versions`（节点 -> 缓存键）；
- `pending_nodes`、`status`、失败数量。

启动时比较期望全局版本号和配置哈希。匹配的未完成快照可续跑；不匹配时普通运行记录 `E_SNAPSHOT_STALE` 诊断并从缓存重建，`--strict-resume` 直接失败并输出该业务码。工作区使用 `fcntl` 文件锁串行化多进程写操作。

## CLI 编辑示例

```bash
# 修改 GELU 参数
python3 -m opchain edit \
  --chain samples/chain.json --config samples/config.json \
  --edit '{"kind":"set_params","node_id":"gelu1","params":{"approximation":"tanh"}}'

# 切换精度并再次运行
python3 -m opchain edit \
  --chain samples/chain.json --config samples/config.json \
  --edit '{"kind":"set_precision","precision_tier":"fp32"}'
python3 -m opchain run --chain samples/chain.json --input samples/input.json \
  --config samples/config.json --workspace .opchain-workspace --no-values

# 比较融合与非融合路径
python3 -m opchain compare-fusion \
  --chain samples/chain.json --input samples/input.json \
  --config samples/config.json --workspace .opchain-fusion-check
```

链配置编辑 API 还可执行 `set_inplace`、`move_node`、`replace_subgraph`、`set_quantization`、`set_fusion` 和 `set_config`。

## 业务码

| 业务码 | 含义 |
| --- | --- |
| `E_CHAIN_INVALID` | 节点/边结构、ID、生产者和激活参数不合法 |
| `E_CYCLE_DETECTED` | 算子链含环 |
| `E_PERM_INVALID` | 置换长度、范围、唯一性或维度覆盖不合法 |
| `E_INPLACE_CONFLICT` | 原地写回会破坏其他消费者的输入引用 |
| `E_RECOMPUTE_BUDGET_PENDING` | 重算预算耗尽，节点保留待办 |
| `E_SNAPSHOT_STALE` | 快照版本号或配置哈希不一致 |
| `E_ACTIVATION_FAILED` | 激活阶段运行时异常 |
| `E_PERMUTE_FAILED` | 重排阶段运行时异常 |
| `E_UPSTREAM_FAILED` | 上游节点失败导致下游隔离失败 |
| `E_SNAPSHOT_IO_ERROR` | 缓存/快照写入诊断失败 |

## 实际验证结果

验证日期：2026-11-22；环境：Python 3.11.0rc1、NumPy 1.26.4。

```text
$ python3 -m pytest -q
.......................                                                  [100%]
23 passed in 14.04s
```

`python3 -m opchain demo --workspace .runtime/demo-root3` 的关键结果：

```text
initial.status = complete
parameter_edit.invalidated = [gelu1, sigmoid1, tanh1, perm2]
parameter_edit.reused = [relu1, perm1]
parameter_edit.full_equality.equal = true
precision_switch.invalidated = [relu1, perm1, gelu1, sigmoid1, tanh1, perm2]
precision_switch.full_equality.equal = true
fusion_difference_perm1_max_abs = 0.0625
budget limited codes = [E_RECOMPUTE_BUDGET_PENDING]
snapshot resumed status = complete
ok = true
```

测试还覆盖：非法链/环/非法置换错误码、四种激活确定性、精度档位真实变化、原地冲突节点对、融合/非融合产物与元数据差异、参数/顺序/子图/精度/原地标记连续编辑后与冷全量逐字段一致、预算待办续跑、快照恢复与过期判定、串行与四线程成功/异常路径一致。

## 说明与边界

- 当前算子均为单输入张量算子，符合题目中的激活与重排节点形态；多输入图结构会被 `E_CHAIN_INVALID` 拒绝。
- `values` 默认以嵌套数组输出，并同时提供 `values_hash`；大张量可使用 `--no-values`。
- 快照包含完整产物与阶段清单，测试样本很小；生产规模可替换为引用计数/增量打包，不影响缓存键和恢复协议。
- 全部验证为本地确定性样本，不使用未授权目标。
