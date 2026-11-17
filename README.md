# Softmax / LayerNorm 确定性增量推理组件

这是一个纯 Python + NumPy 的可集成领域组件与完整示例。它从固定种子在本地生成
`float32` 张量，执行数值稳定或不稳定的 Softmax、带均值/方差统计的 LayerNorm，并支持
交互式参数编辑、行级缓存复用、真实单调时钟延迟预算、原子阶段快照、断点恢复和
增量/全量逐字节一致性验证。

项目不依赖 Docker、外部数据或网络服务。

## 快速开始

```bash
# Linux / WSL
python3 -m pip install -r requirements.txt
PYTHONPATH=src python3 -m pytest -q
PYTHONPATH=src python3 -m softmax_layernorm.cli demo
```

`demo` 会依次执行：

1. 从固定种子生成 `6 x 8` 的 `float32` 张量。
2. 执行包含 epsilon、Softmax 轴、LayerNorm 轴、稳定化开关和行区间替换的 9 条命令。
3. 写完整快照和逐命令报告。
4. 从最终输入与最终参数从零全量重算。
5. 对 Softmax、LayerNorm、均值、方差、epsilon 标记逐字节比较，并比较摘要和失败清单。

输出位置：

- 增量报告：`output/demo/incremental/reports/latest.json`
- 失败清单：`output/demo/incremental/failures/latest.json`
- 最终快照：`output/demo/incremental/current.json` 指向的 `snapshots/v.../`
- 全量结果：`output/demo/full/results.npz`、`output/demo/full/summary.json`
- 一致性报告：`output/demo/verification.json`

也可运行内置样例但不做全量比较：

```bash
PYTHONPATH=src python3 -m softmax_layernorm.cli run --fresh
```

## Windows / Python 环境

要求 Python 3.10 或更高版本，以及可用的 NumPy、pytest：

```powershell
py --version
py -m pip install -U pip
py -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
py -m pytest -q
py -m softmax_layernorm.cli demo
```

也可安装为可编辑包，从而不需要手动设置 `PYTHONPATH`：

```powershell
py -m pip install -e ".[test]"
py -m pytest -q
softmax-layernorm demo
```

仓库提供了 `scripts/check_windows.ps1`，可在 PowerShell 中执行上述检查。

本次实际执行环境是 WSL2 Linux、Python 3.11.0rc1、NumPy 2.4.6、pytest 9.2.0。
实现只使用跨平台标准库的原子替换、临时文件、线程池和 `pathlib`；Windows 上目录
`fsync` 会被安全跳过。POSIX 信号测试在 Windows 上自动跳过。

## 项目结构

```text
samples/
  calls.json                  # 算子调用描述
  tensor.json                 # 固定种子张量描述
  commands.json               # 交互命令流
  config.json                 # 延迟、并发、快照间隔模板
src/softmax_layernorm/
  loader.py                   # JSON 解析、校验、确定性张量生成
  ops.py                      # 固定累加顺序的 Softmax / LayerNorm
  cache.py                    # 内容指纹、行/通道缓存键、失效判断
  compute.py                  # 线程池逐 lane 计算与稳定排序聚合
  engine.py                   # 命令状态机、预算返回、快照与报告
  snapshots.py                # 整目录原子提交、清单校验、恢复
  reference.py                # 可选 float64 高精度参考档位
  summary.py                  # 最新可用摘要
  cli.py                      # run / verify / demo
tests/                        # 算子、校验、增量、并发、预算、恢复、信号、CLI
output/                       # 快照、失败清单、报告预留目录
scripts/check_windows.ps1
scripts/run_demo.py
```

## 输入契约

### 张量描述

```json
{
  "shape": [6, 8],
  "seed": 20261118,
  "distribution": {
    "kind": "normal",
    "loc": 0.0,
    "scale": 1.5
  },
  "special_values": [
    {"row": 0, "col": 7, "value": 120.0},
    {"row": 3, "col": 2, "value": "-inf"}
  ]
}
```

- `shape` 必须是两个正整数。
- 支持 `normal`、`uniform`、`constant` 分布，使用 `numpy.random.PCG64(seed)`。
- `special_values` 按列出的顺序精确覆盖指定单元格；字符串 `"nan"`、`"inf"`、
  `"-inf"` 可表示非有限值。
- 生成结果始终为本地 `float32` 张量。加载层不做 NaN/Inf 清理、裁剪、替换或平滑。

### 调用描述

```json
{
  "calls": [
    {
      "call_id": "call-row-axis",
      "sequence": 1,
      "rows": 6,
      "cols": 8,
      "softmax_axis": 1,
      "layernorm_axis": 1,
      "epsilon": 0.00001,
      "stabilize": true,
      "layernorm_scale": 1.0,
      "reference_precision": "float64"
    }
  ]
}
```

- 当前输入张量为二维，因此轴允许 `-2..1`，并规范化为 `0` 或 `1`。
- `axis=1` 表示逐行归约；`axis=0` 表示逐列归约。缓存 lane 数与归约轴对应。
- `epsilon` 必须是有限正数。配置/加载流程按业务要求拒绝 0。
- `stabilize=false` 会走不减最大值的对照路径，并可能产生 `E_EXP_OVERFLOW`。
- `layernorm_scale` 可逐调用覆盖运行配置中的缩放。
- `reference_precision="float64"` 时，快照额外生成 float64 参考数组和误差比较。

### 交互命令

```json
{
  "commands": [
    {"command_id": "1", "type": "set_epsilon", "epsilon": 0.000001, "call_ids": ["call-row-axis"]},
    {"command_id": "2", "type": "set_softmax_axis", "axis": 0},
    {"command_id": "3", "type": "set_layernorm_axis", "axis": 1},
    {"command_id": "4", "type": "set_stabilize", "enabled": false},
    {
      "command_id": "5",
      "type": "replace_rows",
      "start": 1,
      "end": 3,
      "values": [[1, 2, 3], [4, 5, 6]]
    }
  ]
}
```

- `call_id` 或 `call_ids` 省略时作用于全部调用。
- `replace_rows` 的区间是 `[start, end)`，替换行数必须等于 `end-start`，列数必须完全一致。
- 输入张量由本运行中的调用共享；`replace_rows` 修改共享张量并影响全部调用，调用选择只用于
  参数类命令。
- 行替换也支持由固定种子生成的 `replacement` 对象（`normal` / `uniform` / `constant`）。
- 命令参数非法时不会先改一半状态；整条命令失败并记录业务码。

### 运行配置

```json
{
  "interaction_budget_ms": 100.0,
  "thread_pool_workers": 4,
  "snapshot_interval_commands": 3,
  "layernorm_scale": 1.0,
  "snapshot_root": "output",
  "failure_dir": "output/failures",
  "report_dir": "output/reports",
  "resume": true
}
```

- 并发数必须是正整数；预算必须是非负有限数；快照间隔必须是正整数。
- 配置哈希同时覆盖运行配置、初始调用描述和初始张量描述，避免混用旧产物。

## 数值语义

### 固定 float32 累加顺序

Softmax 指数求和、LayerNorm 均值和偏差平方和都使用显式下标递增的 `np.float32`
逐项累加，而不是把求和顺序交给底层并行归约或成对求和。因此同一 lane 在任意线程数、
任意调度顺序下都得到相同位模式。

Softmax 流程：

1. 稳定档：沿归约轴求最大值并从每个元素减去。
2. 对结果逐元素执行自然指数。
3. 按固定下标顺序累加。
4. 逐元素相除。
5. 不稳定档不减最大值；若指数或总和为 Inf，则产生 `E_EXP_OVERFLOW`，并记录 lane、
   行号（逐行轴）或列号（逐列轴）。

LayerNorm 流程：

1. 固定顺序求均值。
2. 计算 `x - mean`，固定顺序累加偏差平方，再除以特征数，得到总体方差。
3. 分母为 `sqrt(variance + epsilon)`。
4. 输出 `(x - mean) / denominator * scale`。
5. 同时输出与输入同形的归一化结果、均值、方差和 epsilon 标记。

`epsilon_applied=true` 表示该 lane 的方差为零且正 epsilon 激活了分母保护。该标记按
元素存储，因此 `axis=1` 时可直接按行读取，`axis=0` 时同列行共享该标记。

加载/命令校验要求 `epsilon > 0`，所以 `E_ZERO_VARIANCE` 的零 epsilon 对照路径通过
公开的单 lane API 验证：

```python
from softmax_layernorm.ops import layer_norm_lane
result = layer_norm_lane([2.0, 2.0], epsilon=0.0)
assert result.failure.code == "E_ZERO_VARIANCE"
```

### 行级缓存与失效矩阵

每个调用、阶段和 lane 的缓存键由以下内容派生：

- 该 lane 当前的 `float32` 内容 SHA-256；
- 调用序号、调用 ID、lane 号；
- 该阶段实际依赖的归约轴、epsilon、缩放和稳定化开关。

失效规则：

| 命令 | Softmax | LayerNorm |
| --- | --- | --- |
| 修改 epsilon | 全部复用 | 全部重算 |
| 修改稳定化开关 | 全部重算 | 全部复用 |
| 修改 Softmax 轴 | 全部重算 | 全部复用 |
| 修改 LayerNorm 轴 | 全部复用 | 全部重算 |
| `axis=1` 行替换 | 仅替换行重算 | 仅替换行重算 |
| `axis=0` 行替换 | 所有列 lane 重算 | 所有列 lane 重算 |

`axis=0` 时某一行的数据参与每一列的均值/方差，因此替换任意行会改变所有输出行；
这是数学上必要的保守失效，而不是重新计算全部无关 lane。`axis=1` 时未替换行直接复用
持久化数组，不提交计算任务。

每个快照的 `reuse_recompute.json` 记录每条命令、每个调用、每个阶段的有序复用与重算
lane/行清单。

## 交互预算与并发

`Engine.handle_command()` 提交命令后，使用 `time.monotonic()` 计算真实截止时间。它会
在剩余预算内等待完成事件，但绝不越过截止时间无限阻塞。

- 最新参数下的计算在预算内完成：返回 `status="ok"`，携带最新摘要。
- 预算耗尽仍未完成：立即返回 `status="pending"`、业务码
  `E_INTERACT_BUDGET_PENDING`、最近一次已完成摘要、预览受影响行、待处理命令队列。
- 已收到但未完成的命令保留在命令队列中，后续线程继续执行。
- 快照 I/O 不占用状态锁；交互读取不会因写盘冻结。

逐 lane 任务同时携带 `(调用序号, lane, 阶段, 调用 ID)` 稳定排序键。无论工作线程以
什么顺序结束，最终写入结果、失败排序和复用/重算清单都按稳定键聚合。

## 快照、恢复与退出

每次提交的流程：

1. 在 `snapshots/.tmp-*` 建立完整临时目录。
2. 写 `input.npz`、`results.npz`、`state.json`、`reuse_recompute.json`、失败清单，
   可选写 `reference.npz`。
3. 计算每个阶段文件的 SHA-256，最后写 `manifest.json`。
4. 对临时目录执行一次整体目录重命名。
5. 最后原子替换 `current.json` 指针。

因此 `current.json` 只会指向完整快照。强制终止最多留下未被引用的临时/完整目录，不会
把半写阶段混入恢复结果。

`manifest.json` 包含：

- 全局版本号与快照版本号；
- 配置哈希；
- 初始张量指纹与当前张量指纹；
- 命令流摘要；
- 输入、结果、状态、复用/重算、失败、可选参考各阶段版本和 SHA-256。

启动时如果快照存在但配置哈希、源输入指纹、命令流摘要、格式版本或阶段校验不一致，
会抛出 `E_CONFIG_MISMATCH` 并拒绝继续，不覆盖旧快照。配置完全一致时从
`next_command_index` 续跑。

收到 SIGINT/SIGTERM 时，CLI 设置停止请求，等待已接收的当前命令结束，提交完整快照和
报告，然后以退出码 `0` 返回。强制终止不承诺生成退出报告。

## 报告

`reports/latest.json` 包含：

- 每条命令的真实响应耗时 `response_latency_ms` 与 `over_budget`；
- 初始响应状态（`ok`、`error`、`pending`）和最终完成状态；
- 复用/重算 lane 数与去重行数、受影响行；
- 完成行数、lane 进度、各调用最大减法使用情况；
- 累计业务码计数和当前活动业务码计数；
- 当前活动失败、命令级失败和统一失败清单；
- 所有命令 Records、复用/重算历史、高精度参考误差；
- 最终全局版本、快照版本、配置哈希和输入指纹。

## 业务码

| 业务码 | 触发条件 |
| --- | --- |
| `E_CONFIG_INVALID` | 非法 JSON、配置类型/取值非法、非法命令 |
| `E_SHAPE_MISMATCH` | 调用形状、特殊值位置、替换行形状不匹配 |
| `E_AXIS_OUT_OF_RANGE` | 归约轴不在二维张量范围内 |
| `E_EPS_INVALID` | epsilon 非正、非有限或类型非法 |
| `E_EXP_OVERFLOW` | 关闭最大减法后指数或总和溢出 |
| `E_ZERO_VARIANCE` | 单 lane API 上方差为零且 epsilon 为零 |
| `E_INTERACT_BUDGET_PENDING` | 预算内未得到最新参数结果 |
| `E_CONFIG_MISMATCH` | 旧快照与当前配置/输入/命令/阶段校验不一致 |
| `E_INTERNAL` | 防御性线程或文件系统边界的内部错误 |

## 集成 API

```python
from softmax_layernorm import Engine, load_commands_file, load_run

loaded = load_run("samples/calls.json", "samples/tensor.json", "samples/config.json")
_, commands, digest = load_commands_file("samples/commands.json")
engine = Engine(loaded, commands, digest, output_dir="output/api-run", fresh=True)

for command in commands:
    response = engine.handle_command(command)
    print(response["status"], response["latest_completed_summary"])

report = engine.finalize()
```

直接算子 API：

```python
from softmax_layernorm.ops import layer_norm_lane, softmax_lane

softmax_result = softmax_lane([1, 2, 3], stabilize=True)
layernorm_result = layer_norm_lane([1, 2, 3], epsilon=1e-5, scale=1.0)
```

## 实际验证结果

本次在当前工作区实际执行：

```text
PYTHONPATH=src python3 -m pytest -q
25 passed

PYTHONPATH=src python3 -m softmax_layernorm.cli demo
ok: true
arrays_equal: true
failures_equal: true
summary_equal: true
active_business_codes_equal: true
exit code: 0
```

覆盖验证包括：

- 稳定 Softmax 归一化与重复运行位级一致；
- 不稳定 Softmax 溢出及 lane/row 记录；
- LayerNorm 均值、总体方差、epsilon 标记和零方差零 epsilon 路径；
- 非法 JSON、形状、轴、epsilon、并发数业务码；
- epsilon、稳定化、两个归化轴各自的精确失效集合；
- 行区间替换只重算目标行，未替换行位级复用；
- 1 线程与 8 线程结果、统计量、失败顺序一致；
- 多 lane 溢出失败严格按 lane 升序排列；
- 任意编辑序列最终增量与全量逐字节一致；
- 0ms 延迟预算返回 pending 且保留队列；
- 快照续跑、配置哈希拒绝、阶段文件和指针校验；
- SIGTERM 后完成快照并以 0 退出；
- 运行期非法 epsilon、轴和命令类型分别返回预期业务码；
- 内置样例的 CLI 增量/全量验证。
