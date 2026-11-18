# 混合精度推理显存管理诊断台

面向 Windows 本地 Python 3.11 + Tkinter 的确定性诊断应用。它在一个**纯本地、无 GPU、无网络**的推理环境中，模拟块级显存预算、两种精度路径、批次调度、逐层误差归因、版本替换和界面查询延迟。

## 快速开始（Windows）

```bat
py -3.11 --version
py -3.11 -m pip install -r requirements.txt
py -3.11 -c "import tkinter as tk; root=tk.Tk(); print('Tk', tk.TkVersion); root.destroy()"
```

确认当前目录可写后启动界面或运行无界面演示：

```bat
scripts\run_ui.bat
scripts\demo.bat --expected-manifest configs\digest_manifest.json
py -3.11 -m unittest discover -s tests -v
```

PowerShell 等价入口为 `scripts\run_ui.ps1` 与 `scripts\demo.ps1`。也可直接使用：

```bat
py -3.11 -m mixedprec_diag.cli demo --config-dir configs --report-dir reports --snapshot-dir snapshots --expected-manifest configs/digest_manifest.json
```

运行后：

- `reports/current.json`：当前配置版本的完整报告，不混入旧版本数据。
- `reports/version_000N.json`：每个配置版本的当前/历史报告。
- `reports/session.json`：版本历史、陈旧状态与当前摘要。
- `reports/demo_summary.json`：三次演示的摘要、摘要校验、重放与替换结论。
- `snapshots/version_000N_stale.json`：被整体替换的旧版本快照，显式标记 `STALE`。

## 演示流程

`scripts/demo.py` 在同一个会话中顺序执行：

1. `configs/float32.json`：float32 参考路径、按批连续预分配。
2. `configs/fixed.json`：定点路径、块池分配，配置版本递增。
3. `configs/scaled.json`：缩放因子扰动、批大小和预算变化，版本再次递增。

每次运行都会重新生成同一随机种子下的数据集和模型参数；配置版本变化时旧版本批次输出、显存统计和误差结论被标为陈旧并从当前视图/当前聚合中排除。

额外探针配置：

- `configs/latency_strict.json`：将 UI 查询预算设为 0.001 ms，便于观察 `E_LATENCY`。
- `configs/pressure_probe.json`：在 1 MiB 块、2 MiB 预算下令单批占用全部空闲块，触发 `E_PRESSURE`。
- `configs/oom_probe.json`：模型常驻后没有样本工作区，触发 `E_OOM`。

## 配置格式

配置必须是本地 JSON 对象，字段强校验，未知字段也整体拒绝。示例：

```json
{
  "schema_version": 1,
  "memory_budget_mib": 1,
  "block_size_kb": 16,
  "precision": "fixed",
  "quant_scale_power": 4,
  "batch_size_limit": 8,
  "preallocation": "block_pool",
  "budget_rounding": "reject",
  "pressure_threshold_ms": 25.0,
  "ui_query_budget_ms": 16.0
}
```

| 字段 | 规则 |
| --- | --- |
| `schema_version` | 当前仅接受 `1` |
| `memory_budget_mib` | 正整数 MiB |
| `block_size_kb` | 受支持整数集合：4/8/16/32/64/128/256/512/1024/1536/3072 |
| `precision` | `float32` 或 `fixed` |
| `quant_scale_power` | `10` 的整数幂指数，安全范围 `[-9,9]` |
| `batch_size_limit` | 正整数，批次可因容量下降但不会越过该上限 |
| `preallocation` | `per_batch` 或 `block_pool` |
| `budget_rounding` | `reject` 或 `floor_to_block`；预算非块整数倍时只能显式向下取整 |
| `pressure_threshold_ms` | 容量耗尽持续阈值，非负有限数 |
| `ui_query_budget_ms` | 单次 UI 查询预算，正有限数 |

已校验字段：

```bat
py -3.11 -m mixedprec_diag.cli validate-config --config configs\fixed.json
```

## 确定性、摘要与 E_DATA

本地生成器使用 `numpy.random.PCG64(seed)`。在相同 Python/NumPy 平台和相同 `--seed --samples --input-dim --hidden-dim` 下，输入矩阵和各层权重/偏置逐字节一致。报告包含：

- `data_sha256`：样本矩阵的 dtype、shape 和 C-order little-endian float32 字节摘要。
- `params_sha256`：每层权重/偏置的名称、dtype、shape 和规范化字节摘要。

生成并校验摘要清单：

```bat
py -3.11 -m mixedprec_diag.cli manifest --output configs\digest_manifest.json --seed 184 --samples 48 --input-dim 32 --hidden-dim 128
py -3.11 -m mixedprec_diag.cli run --config configs\fixed.json --expected-manifest configs\digest_manifest.json
```

摘要不一致时返回 `E_DATA`，附带 `field_path` 与 `expected/actual`，不启动推理。会话同时写出 `reports/aborted_E_DATA.json` 和对应快照，其中 `inference_started=false`。

## 显存模型与调度

本地 CPU 推理不使用真实 GPU 显存；显存池是**可重放的块级会计模型**：

- 预算先按块尺寸折算为整数块；模型参数常驻占用连续块。
- `per_batch`：每批请求一个连续 arena，批量释放。
- `block_pool`：输入激活、逐层输出激活和定点累加 scratch 分别从空闲块池分配，逐层释放前一阶段。
- 调度按 `batch_size_limit`、空闲块数和连续空洞数决定可容纳样本数；容量不足的剩余样本进入队列，不会越过预算。
- `fragmentation_blocks = free_blocks - largest_contiguous_free_extent`，完全由当前块池状态推导。
- 每次分配/释放/OOM 均记录序号、块号集合、请求字节数、所属批次、阶段、分配后的完整空闲段和单调时钟。
- 报告中的 `replay_events` 从初始块池重放日志，并校验 no-overlap、越界、释放匹配、逐事件 used/free/extent/fragmentation 状态。

`E_OOM` 记录所需块数和当前空闲块数；`E_PRESSURE` 由事件流中“剩余空闲块不足以容纳一个样本”的持续时间与阈值推导。

## 两条数值路径

float32 路径用 NumPy 前向计算作为参考。定点路径：

- 权重、输入、激活和偏置量化为 signed int32；缩放因子为 `S = 10^power`，负幂以精确整数分子/分母表示。
- 舍入为最近整数，恰好 0.5 时远离零。
- 点积使用 Python 大整数精确累加，再执行 `S² -> S` 的舍入，最后显式饱和到 int32；不会依赖静默整数回绕。
- 报告记录权重/逐层激活饱和计数、最近舍入上界 `500 milli-int = 0.5 int`，以及每层保守整数误差界。

逐样本诊断独立执行参考与定点前向，批次只影响调度和输出摘要，不影响单样本中间量、最大偏差层或误差归因。因此同一个样本落在任意批次和任意批内位置，归因结果一致。报告同时给出各层最大误差、绝对误差总和、误差占比、累计占比、主导层和最终一致性结论。

## 配置版本与陈旧防护

`batch_size_limit`、`precision`、`quant_scale_power`、`memory_budget_mib`、`block_size_kb` 或 `preallocation` 变化时版本号递增。替换行为：

- `replaced_whole=true`，当前报告只包含新版本批次、显存统计和数值结论。
- `changed_fields` 排序后稳定输出。
- `affected_batches`、`stale_batch_ids` 排序后稳定输出。
- 旧版本快照标记 `STALE`，并声明 `excluded_from_current_report=true`；当前报告不合并旧批次、旧曲线或旧误差结论。

## UI 查询延迟

界面查询使用 `time.perf_counter_ns` 单调时钟，逐次保存 `start_mono_ns`、`end_mono_ns` 和整数微秒 `duration_us`。报告声明真值来源为 `ui_latency.queries`，并给出：

- 按批次和整体的均值、P95（nearest-rank）、最大值，全部以整数微秒定点表示。
- 原始量化误差上界 `0.0005 ms`；均值重新取整附加 `0.001 ms` 上界。
- 单次查询超过配置 `ui_query_budget_ms` 时记录 `E_LATENCY`，界面以红色延迟徽标和状态码标注。

界面包含配置校验/执行、三版本演示、批次列表与进度、显存占用/碎片曲线、块池事件明细、逐层误差界、逐样本归因、查询延迟真值表和最终一致性结论。运行任务在后台线程执行，Tk 主线程只消费事件队列。

## 错误码

| 错误码 | 含义 |
| --- | --- |
| `E_CONFIG` | JSON 语法、重复键、未知字段、字段类型/取值或预算折算规则非法；整体拒绝 |
| `E_DATA` | 本地数据/参数摘要与清单不一致；不启动推理并写出出错位置 |
| `E_OOM` | 块预算或连续空洞不足；报告所需块数和空闲块数 |
| `E_PRESSURE` | 容量耗尽持续超过阈值；报告事件区间 |
| `E_LATENCY` | 单次 UI 查询超过预算 |
| `E_IO` | 报告目录/图形环境等本地 I/O 或环境错误 |

## 测试与当前验证

```bat
py -3.11 -m unittest discover -s tests -v
```

当前环境实际执行结果：**34 tests OK**。覆盖内容包括配置强校验、重复键、摘要拒绝位置、块池重叠/OOM/碎片/重放、int32 饱和与半远离零舍入、两路径重复性、批次与位置无关归因、容量拆分与压力区间、OOM 报告、版本整体替换、陈旧快照、E_DATA 中止报告、UI 延迟定点聚合。

正式演示结果（seed=184，48 samples，input=32，hidden=128）：

| 版本 | 路径/策略 | 批次 | 峰值块 | 输出摘要（前 16 位） | 结论 |
| --- | --- | ---: | ---: | --- | --- |
| v1 | float32 / per_batch | 6 | 7 | `9c5795a72a1ccfa4` | PASS |
| v2 | fixed / block_pool | 6 | 9 | `441ad585bbcd9376` | PASS |
| v3 | fixed / block_pool / scale 10^5 | 8 | 9 | `66dd630be6de3323` | PASS |

演示摘要：数据 `36c21d83c44e75425469d80ef8c4eb541a4b7ba17abecf29baaaa47f9d3d2842`，参数 `bdb82888a59179f08d2c2388cddf7bf051a03f016303aff5f9b32f455d5c3ad8`；所有版本块池重放通过、版本严格递增、整体替换为真。最终 v3 单次 UI 查询整体 P95 约 `4.185 ms`、最大值约 `4.185 ms`，低于 16 ms 预算。

## 目录

```text
mixedprec_diag/
  config.py         配置解析与强校验
  deterministic.py  PCG64 样本/模型与 SHA-256 摘要
  memory.py         块池、碎片、OOM、重放
  quantization.py   定点舍入、饱和、整数点积
  inference.py      两条前向路径与误差归因
  engine.py         调度、报告、版本化会话
  versioning.py     整体替换与陈旧防护
  latency.py        UI 查询单调时钟与定点聚合
  ui.py             Tkinter 交互诊断台
  cli.py            命令行入口
configs/             三组演示配置与诊断探针
scripts/             Windows/PS/演示脚本
tests/               34 项本地测试
```

## 环境说明与限制

当前代码交付验证环境为 Linux Python 3.11 + NumPy；该容器没有 `_tkinter`/Tk 动态库，因此这里完成了 UI 源码编译检查和“缺失 Tkinter 时 E_IO 诊断”验证，但未能在容器内打开真实窗口。Windows 桌面上的启动确认命令已列在快速开始中。应用自身不读取网络、GPU 或外部数据集；显存数据是本地确定性块级会计仿真，不声称测量真实显卡 VRAM。
