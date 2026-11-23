# mixedmem 混合精度显存租约诊断工作台

本项目是一个完全本地、确定性的混合精度推理诊断工作台。它从零实现以下闭环：

1. 严格校验本地 JSON 推理计划；固定整数种子生成输入与权重。
2. 同时执行 NumPy float32 参考路径和显式定点量化路径。
3. 通过带 TTL 的块池租约，把批次推理投递到真实本地线程池。
4. 在层依赖图上只重算失效子集，并原子替换版本租约账本。
5. 将逐层结论、租约、失效轨迹与误差归因写入 SQLite。
6. 保存带版本和内容摘要的会话快照，支持整体替换恢复。
7. 对强制终止后的账本做未知状态隔离、重放校验与剩余批次续跑。
8. 用 Tkinter 展示依赖图、租约/调度时间线、偏差归因和快照列表。

项目不依赖外部服务、真实账号、容器或网络资源。唯一运行时第三方依赖是 `numpy==1.26.4`。

## Windows 快速开始

使用 Windows 官方 Python 3.11 或 3.12 安装包，安装时保留 Tcl/Tk 组件：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m unittest discover -s tests -t . -v
```

运行完整本地验收场景：

```powershell
python -m mixedmem demo --workspace artifacts\demo --plan sample_plan.json --workers 2
Get-Content artifacts\demo\summary.txt
```

启动诊断界面：

```powershell
python -m mixedmem gui `
  --plan artifacts\demo\plan-final.json `
  --db artifacts\demo\ledger.db `
  --snapshots artifacts\demo\snapshots
```

界面刷新按钮只以 SQLite 只读连接读取数据，不写入、更新或 checkpoint 账本。

## 目录

```text
mixedmem/
  schema.py       计划解析、强校验、占用派生、原子多层编辑
  graph.py        激活/显存依赖图、稳定拓扑、失效传播、删除保护
  samples.py      固定种子输入与权重生成、批切分
  quant.py        定点量化、饱和、float32 定序归约、误差界
  inference.py    双路径推理、逐层/逐样本摘要、误差归因
  lease.py        块池、独占区间、单调时钟 TTL、幂等归还、撤销
  scheduler.py    真实 ThreadPoolExecutor 调度、预算排队观测
  ledger.py       SQLite 事务账本、格式/内容摘要、崩溃恢复视图
  snapshot.py     版本快照、内容校验、差异列表、整体替换
  runtime.py      全量运行、失效子集重算、版本替换、重放续跑
  gui.py          Tkinter 只读诊断界面
  demo.py         一键验收场景与报告生成
  cli.py          `python -m mixedmem` 命令行入口
tests/            30 个 unittest 用例
sample_plan.json  固定种子本地声明样本
artifacts/demo/   实际运行生成的账本、快照、最终计划和验收报告
```

## 计划 JSON

`sample_plan.json` 使用固定种子 `20261124`。核心字段如下：

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 当前为整数 `1` |
| `version` | 计划版本，变更时单调递增 |
| `seed` | 非负整数随机种子 |
| `supported_quant_scales` | 受支持量化缩放因子集合，升序 |
| `quantization_scale_version` | 量化口径版本，用于快照版本对齐 |
| `block_size_bytes` | 显存块字节数 |
| `memory_budget_bytes` | 显存总量预算 |
| `batch_size_max` | 单批样本上限 |
| `lease_ttl_ms` | 租约有效期，正整数毫秒 |
| `layers[]` | 层声明 |

层声明字段：

```json
{
  "id": "hidden",
  "input_dim": 6,
  "output_dim": 5,
  "param_count": 35,
  "precision": "fixed",
  "scale": 128,
  "depends_on": ["embed"],
  "memory_derivation": {
    "base_bytes": 32,
    "terms": [{"layer": "embed", "coefficient": 0}]
  }
}
```

`depends_on` 同时表示激活引用和推理结论依赖。有引用时，`input_dim` 必须等于被引用层输出维度按声明顺序拼接后的总和。`memory_derivation.terms` 额外建立显存占用依赖边；系数可为 0，但引用关系仍参与失效传播和环检测。

层块占用计算：

```text
raw = base_bytes
    + param_count * weight_bytes
    + output_dim * actual_batch_size * activation_bytes
    + Σ(coefficient * referenced_layer_memory)

bytes = align_up(max(raw, 1), block_size_bytes)
blocks = bytes / block_size_bytes
```

`float32` 每元素按 4 字节、`fixed` 每元素按 2 字节计入占用。`actual_batch_size` 是本批实际样本数，因此最后一批可少于 `batch_size_max`。

### 强校验与错误定位

加载时任一非法都会整体拒绝。错误对象稳定包含：

```json
{
  "ok": false,
  "code": "E_SCHEMA",
  "message": "scale must belong to supported_quant_scales",
  "layer_id": "compress",
  "field": "scale",
  "details": {"supported": [64, 128, 256, 512]}
}
```

校验内容包括层 ID 唯一、维度/参数量/预算/TTL 为正整数、精度集合、缩放因子集合、引用目标存在、激活维度拼接一致、派生图无环和未知字段拒绝。删除仍被引用的层返回 `E_DEPEND`，并稳定列出 `referrers`。

## 确定性样本与双路径

输入和权重只依赖以下内容：

- 输入：`seed + layer.id + input_dim + sample_id`
- 权重：`seed + layer.id + input_dim + output_dim + param_count`

`precision`、`scale`、计划版本和批次切分不进入权重种子，因此某层量化参数变化不会无故改变其他层张量。每个样本独立生成，再按样本 ID 规范排序聚合。

float32 路径使用固定升序标量归约：

```text
acc = float32(0)
for k ascending:
    acc = float32(acc + float32(w[j,k] * x[k]))
out = float32(acc + bias[j])
```

它不依赖 BLAS 的批形状相关归约，因此批大小和执行顺序不会改变输出位模式。

定点路径策略明确记录在归因结果中：

- 舍入：`round_half_away_from_zero`。
- 权重、激活、偏置：饱和到 signed int24。
- 点积累加：逐步饱和到 signed int64。
- 归约顺序：输入索引升序。
- 反量化：`accumulator / scale**2`，最后转 float32。
- 租约重排不参与任何数值归约，其数值贡献恒为 0。

层摘要按样本 ID 排序后对 `<f4` 小端连续字节做 SHA-256；另有逐样本 float/quant 摘要。改变批大小只改变批次边界，不改变逐层汇总摘要或归因结果。

## 块池与租约状态机

块编号是半开区间 `[block_start, block_start + block_count)`。租约期限使用进程单调时钟整数毫秒：

```text
expires_ms = granted_ms + lease_ttl_ms
```

状态包括：

- `active`：有效并参与当前块占用视图。
- `released`：提前归还，块已释放。
- `expired`：真实单调时钟到期后自动归还。
- `revoked`：计划版本演进时整体撤销，记录 `E_REVOKE` 依据。
- `unknown`：强制终止后无法确认的旧租约，要求重跑。
- `assigned`：版本替换账本中的新版本视图记录。

关键行为：

- `acquire_bundle` 对所有层请求做 all-or-nothing 授予。
- 若预算不足，调用线程在条件变量上真实等待；租约释放或到期会唤醒重排。
- 同一时刻活动区间两两不重叠，活动块并集不超过预算。
- 提前归还必须同时匹配租约 ID 和批次 ID，可选匹配层 ID；不符返回 `E_LEASE`。
- 重复归还返回第一次归还的同一结果，不重复释放。
- 到期由前台检查和后台 reaper 共同触发，并通知等待线程。

## 线程池调度

调度器使用 `concurrent.futures.ThreadPoolExecutor`。主调度线程在授予租约后提交工作线程，工作线程执行逐层计算；没有用 `sleep` 伪造租约等待或计算耗时。

每个批次的观测字段包括：

- `queue_start_ms`：开始申请预算的单调整数毫秒。
- `queue_end_ms`：工作线程真正启动时刻。
- `compute_start_ms` / `end_ms`。
- `worker_name`：真实线程名。
- 每个层的租约 ID 与块区间。

若租约在计算中到期，worker 在下一层提交前通过 `assert_active` 检测并失败；不会继续使用已归还块。

## 失效传播与版本账本替换

依赖图把 `depends_on` 建成实线激活边，把 `memory_derivation.terms` 建成虚线占用派生边。拓扑排序以声明顺序优先、层 ID 为稳定次级键。

单层变更：

```python
runtime.apply_change("hidden", {"scale": 256})
```

需要同时调整上下游维度时使用原子多层编辑：

```python
runtime.apply_changes({
    "hidden": {"output_dim": 6},
    "compress": {"input_dim": 6},
})
```

计划版本只递增一次。沿依赖图求出的直接和间接失效层按稳定拓扑序重算；未失效层复用原结果对象，块区间、float/quant 摘要和层版本保持不变。

版本演进时：

1. 识别并真实撤销当前块池中旧计划版本的未到期租约。
2. 将撤销结果、释放区间和依据写入账本。
3. 先完整构建新计划、新结果映射和新租约视图。
4. 成功后一次性替换运行时状态，不逐条修补旧账本。
5. 已归还、已到期和已撤销记录保留为只读历史，不进入当前活动视图。

`verify_incremental()` 会从零重算全部层，并逐字段比较数值摘要、偏差、误差界和占用；首个不一致定位到层和字段。

## 数值偏差归因

每层从逐样本误差统计开始，以样本 ID 排序并用 `math.fsum` 聚合，避免批次分组导致浮点末位差异。输出包含：

- 每层最大/平均绝对偏差、相对偏差、累计误差占比。
- 每层量化截断、累加顺序、租约重排三段误差界。
- 总误差界的各段贡献和占比。
- 最大偏差层、并列候选和 `max_deviation_error_affected` 标记。

截断项来自操作数和反量化半单位误差；累加顺序项来自 float32 参考路径的定序归约舍入界；租约重排项为 0。主导段规则：

```text
share >= 0.60 且领先第二名 >= 0.10 -> 该段为主导
否则 -> mixed
```

若前两层经验偏差差落在二者合成误差界内，按层 ID 升序稳定选择，并标记判定受误差影响。

## SQLite 账本与事务

数据库启用 WAL 与 `synchronous=FULL`。主要表：

- `batches`：批次、样本、状态、排队/启动/结束时刻、worker 名。
- `leases`：租约、块区间、期限、状态与撤销依据。
- `layer_results`：逐层版本、摘要、偏差、误差界、占用、租约归属。
- `progress`：每次完成的层批次游标。
- `invalidations`：失效传播完整依据。
- `attributions`：逐层归因和总体判定。
- `lease_events`：grant/release/expire/revoke/assign/unknown 事件。
- `meta`：整数账本格式版本与全库内容摘要。

每次完成一个“层 × 批次”，`layer_results` 与 `progress` 在同一个 `BEGIN IMMEDIATE ... COMMIT` 事务中写入，并同步更新内容摘要。账本重新打开时先核对格式版本和全表内容摘要，任何篡改或不一致返回 `E_REPLAY`。

## 快照、整体恢复与强制终止

快照文件包含：

- 完整计划声明、计划版本、量化口径版本。
- 样本数、数据集摘要。
- 租约集合与到期毫秒。
- 逐层输出摘要与层版本。
- 已处理批次游标、账本内容摘要。
- 快照格式版本与快照自身内容摘要。

载入时依次校验格式、自摘要、计划可解析性和量化口径版本。版本无法对齐返回 `E_SNAPSHOT`。可对齐但与当前计划/数据集/账本/租约不同，则稳定列出差异项，并通过 `InferenceRuntime.restore_snapshot()` 整体替换内存计划、租约集合与逐层摘要，绝不增量合并。

强制终止测试通过子进程在完成第 5 个层事务后执行 `os._exit(91)`：

- 已提交的 `b0000` 保留。
- 计算中的 `b0001` 标记为 `unknown`，其活动租约标记 `unknown`。
- 报告明确 `uncommitted_exit_report_available: false`。
- 重启只重放校验完整提交行，不信任半批次数据。
- 已提交层逐项比较输出摘要、偏差、误差界、租约 ID/区间；归因随后重算并写入完整账本。
- 缺失批次重新调度，完成后 `complete_after_resume: true`。

## CLI

```text
python -m mixedmem validate --plan sample_plan.json

python -m mixedmem run --plan sample_plan.json --db run.db \
  --snapshots snapshots --samples 8 --batch-size 4 --workers 2 \
  --report run-report.json

python -m mixedmem invalidate --plan sample_plan.json --db run.db \
  --snapshots snapshots --layer hidden --updates '{"scale":256}' \
  --plan-out plan-v2.json --report invalidation.json

python -m mixedmem recover --plan sample_plan.json --db crashed.db \
  --samples 8 --batch-size 4 --workers 2 --report recovery.json

python -m mixedmem snapshots --snapshots snapshots

python -m mixedmem snapshot-load --snapshot SNAPSHOT.json \
  --plan plan-v2.json --db run.db

python -m mixedmem demo --workspace artifacts/demo --plan sample_plan.json

python -m mixedmem gui --plan artifacts/demo/plan-final.json \
  --db artifacts/demo/ledger.db --snapshots artifacts/demo/snapshots
```

所有错误都以 JSON 输出，进程返回非零码；不会只打印堆栈后留下部分状态。

## Tkinter 界面

界面包含五个稳定排序页签：

1. 层依赖图：实线激活依赖、虚线显存派生依赖、稳定拓扑序。
2. 块池/租约时间线：按授予时刻、批次、层、租约 ID 排序，显示块区间和到期。
3. 批调度时间线：排队段与计算段、真实 worker 名、批次状态。
4. 数值偏差/误差归因：逐层偏差、误差界、累计占比、饱和次数和主导段规则。
5. 版本/快照：当前计划版本、量化口径版本、账本版本/摘要和快照列表。

刷新使用 SQLite URI `mode=ro`，不会修改账本；自动化测试比较了刷新前后的账本内容摘要。

## 实际验证记录

本工作区使用 Python 3.11 与 `numpy==1.26.4` 验证：

```text
python -m unittest discover -s tests -t .
Ran 30 tests
OK
```

`python -m mixedmem demo --workspace artifacts/demo --plan sample_plan.json --workers 2` 全部通过，关键结果：

```text
overall: PASS
plan_version: 2
dataset_digest: 6ac697e75b835fe163f3c6dd84bf0ff6163d4699c6dc174d2b520adee91776a7
invalidated: hidden, compress, head
unaffected: embed
revoked leases: 4
new version leases: 8
batch-size invariant: True
attribution invariant: True
max observed budget queue wait: 50 ms
forced termination return code: 91
recovery unknown batches: 1
recovery resumed batches: b0001
```

完整证据位于：

- `artifacts/demo/report.json`
- `artifacts/demo/summary.txt`
- `artifacts/demo/ledger.db`
- `artifacts/demo/snapshots/`
- `artifacts/demo/plan-final.json`
- `artifacts/demo/crash/ledger.db`

本次执行环境缺少 `libtk8.6.so`，因此只验证了 Tkinter 数据装载、只读刷新和图形布局数据，未在此容器内显示窗口；在带 Tk 的 Windows 官方 Python 中可直接运行 GUI 命令。缺失 Tk 时 CLI 会稳定返回 `E_IO`，不会影响推理、账本与恢复流程。
