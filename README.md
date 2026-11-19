# VecFlow

VecFlow 是一个面向本地 Windows 环境的向量化执行领域数据编辑与操作应用。它读取本地 JSON 列式数据块和表达式任务，将任务编译为确定性的行区间分片，在全局资源单位预算与每任务并发配额的双重约束下，把分片分配到本地工作线程池，并在所有分片完成后按显式聚合口径合流。

项目不访问网络、不需要端口或外部服务，不使用 Docker。唯一第三方运行依赖是 NumPy；桌面界面使用 Python 官方安装包通常自带的 Tkinter。

## 主要能力

- 列式数据块与任务定义的加载、编辑和强校验。
- 算术、比较、逻辑、三元条件表达式解析、规范化与结构摘要。
- 显式分片或按行数生成分片；首尾相接、无重叠、无遗漏。
- 全局资源预算与每任务并发配额硬约束。
- NumPy 逐算子向量求值，空值与除零显式标记。
- 按行区间顺序和 sum/concat/count 等口径确定性合流。
- 参数版本递增、旧运行陈旧隔离、整体替换结果。
- SQLite 执行账本、批次事务提交、会话快照整体恢复。
- 强制终止后的 unknown 标记与逐分片一致性重放。
- Tkinter 数据编辑、任务/预算编辑、时间线、资源曲线、结果与账本视图。
- `E_DATA`、`E_RANGE`、`E_NULLMAP`、`E_BUDGET`、`E_REPLAY` 稳定业务码。

## 环境准备（Windows）

要求 Python 3.11 或更高版本，并确保 `pip` 可用：

```powershell
py -3.11 --version
py -3.11 -m pip --version
py -3.11 -m pip install -r requirements.txt
```

Windows 官方 Python 安装包通常自带 Tk。可用以下命令确认：

```powershell
py -3.11 -m vecflow preflight --log-dir logs
```

从源码安装命令入口：

```powershell
py -3.11 -m pip install -e .
vecflow --help
```

不安装也可以使用 `py -3.11 -m vecflow ...` 或 `py -3.11 app.py ...`。

## 快速演示

```powershell
py -3.11 tools/generate_samples.py
py -3.11 -m vecflow preflight --log-dir logs
py -3.11 -m vecflow validate --log-dir logs
py -3.11 -m vecflow run --reset-state --log-dir logs
py -3.11 -m vecflow gui --log-dir logs
```

`--reset-state` 清空默认账本 `samples/ledger.sqlite` 并删除默认快照 `samples/snapshots/session.json`，适合重复演示。

完整 Windows 演示脚本：`scripts/demo_windows.ps1`。它会生成样本、预检、正常执行、强杀恢复/重放，最后打开 Tkinter 工作台。

## 本地样本

`samples/` 中的全部文件由本地脚本确定性生成：

- `samples/data_blocks.json`：两个列式数据块，包含整数、浮点、布尔、空值位图、除零点。
- `samples/tasks.json`：三个任务，覆盖算术、比较、逻辑、三元、显式分片、生成分片、`sum`/`concat`/`count`。
- `samples/inject_range/`：分片终点越界，业务码 `E_RANGE`。
- `samples/inject_nullmap/`：空值位图缩短，业务码 `E_NULLMAP`。
- `samples/inject_budget/`：全局预算为 0，真实调度失败并返回 `E_BUDGET`。

重新生成：

```powershell
py -3.11 -m vecflow generate-samples
```

## 数据与任务格式

### 列式数据

```json
{
  "format_version": 1,
  "blocks": [
    {
      "id": "orders",
      "row_count": 4,
      "columns": [
        {
          "name": "qty",
          "type": "integer",
          "values": [1, 2, 3, 4],
          "null_bitmap": [0, 1, 0, 0]
        },
        {
          "name": "price",
          "type": "float",
          "values": [10.5, 11.0, 11.5, 12.0],
          "null_bitmap": [0, 0, 0, 0]
        },
        {
          "name": "active",
          "type": "boolean",
          "values": [true, false, true, true],
          "null_bitmap": [0, 0, 0, 0]
        }
      ]
    }
  ]
}
```

校验规则包括：

- 数据块标识、列名、任务标识唯一。
- 列类型仅接受 `integer`/`float`/`boolean`（兼容 `int`、`double`、`bool` 等别名）。
- `row_count`、每列 `values`、每列 `null_bitmap` 长度一致。
- 空值位图只允许 0/1；位图为 0 的位置不能是 JSON `null`。
- 任务引用数据块、表达式引用列都必须存在。
- UI 编辑保存失败时整体拒绝为 `E_DATA`，内存中已加载状态和文件均不被覆盖。

### 任务与预算

```json
{
  "format_version": 1,
  "parameter_version": 1,
  "global_budget": 4,
  "threads": 4,
  "tasks": [
    {
      "id": "revenue_guard",
      "block": "orders",
      "expression": "if(active, (price * qty) / (discount + 1.0), 0.0)",
      "quota": 2,
      "aggregation": "sum",
      "shards": [
        {"start": 0, "end": 2, "resources": 2},
        {"start": 2, "end": 4, "resources": 3}
      ]
    }
  ]
}
```

也可用确定性行数切分：

```json
{
  "id": "generated",
  "block": "orders",
  "expression": "qty + 1",
  "quota": 1,
  "aggregation": "concat",
  "shard_rows": 100,
  "resources_per_shard": [1, 2, 1]
}
```

每个任务必须二选一声明 `shards` 或 `shard_rows`。显式分片会按 `start` 规范排序，并要求 `[0, row_count)` 首尾相接、不重叠、不遗漏；越界返回 `E_RANGE`。资源单位和配额必须是正整数，全局预算可以是 0（用于预算异常样本）。

## 表达式语言

支持：

- 算术：`+ - * / %`
- 比较：`< <= > >= == !=`
- 逻辑：`and or not`，也接受 `&& || !`
- 三元：`cond ? when_true : when_false`
- 等价函数式三元：`if(cond, when_true, when_false)`
- 整数、浮点、布尔、`null` 字面量，列引用和括号

空白和冗余括号会被消除。例如：

```text
((price * qty) + 1)  ==  price*qty+1
if(a > 2, b + 1, c * 2) == a>2 ? b+1 : c*2
```

规范化文本和 AST 结构 SHA-256 摘要共同标识表达式语义。非冗余括号会保留，例如 `a - (b - c)`。

### 空值与除零规则

- 一元、二元、比较和逻辑操作的任一空值输入都传播为空值。
- 除数非空且等于 0 时，结果为空值，同时置位 `division_by_zero_mask`。
- 三元条件为空时结果为空；只传播被选中分支的空值和除零位图，未选中分支不会泄漏异常。
- 空值位置的占位值固定为 0/0.0/false，规范结果用 `values`、`null_mask`、`division_by_zero_mask` 三列表达。
- 聚合跳过空值（包括除零产生的空值），但保留空值数量和除零数量。

聚合口径：`concat`、`sum`、`count`、`min`、`max`、`mean`、`any`、`all`、`first`、`last`。

合流始终按分片行区间排序后拼接，不按线程完成顺序。浮点结果使用规范 JSON 序列化；相同输入、相同分片计划的任意线程完成顺序得到逐字节一致的任务结果和摘要。

## 并行调度

调度器在一次启动决策中同时检查：

```text
所有运行分片资源之和 + 新分片资源 <= global_budget
该任务运行分片数 + 1 <= task.quota
```

所有分片进入稳定队列，记录：

- `queued_at_ms`
- `started_at_ms`
- `ended_at_ms`
- `wait_duration_ms`
- `resources`

时钟为 `time.monotonic_ns() // 1_000_000` 的整数毫秒。运行过程中会记录资源开始/结束事件和峰值。预算或配额持续不足时不会绕过上限，而是返回 `E_BUDGET` 并列出所有阻塞分片。

## 版本、陈旧防护与整体替换

- 全局预算、线程数、任务配额、表达式或数据内容发生语义变化时，`parameter_version` 单调递增。
- 仅格式/空白变化且语义摘要不变时，不递增版本。
- 修改发生时，正在运行和排队的分片会被稳定列为 affected 分片，旧运行被标记为 stale。
- 旧版本的轨迹、结果和统计不参与当前结果展示/汇总。
- 新一轮执行构建全新的 RunReport 并整体替换当前运行，不按分片增量合并。

## SQLite 账本

默认账本：`samples/ledger.sqlite`。账本携带整数格式版本 `format_version=1` 和内容摘要 `content_digest`。

核心表：

- `runs`：批次、参数版本、内容摘要、计划摘要、状态、结果摘要、累计摘要、退出报告许可、重放状态。
- `shard_records`：分片标识、任务、行区间、状态、结局、资源单位、排队/开始/结束毫秒、等待时长、结果摘要、错误码。
- `ledger_meta`：整数格式版本和最近内容摘要/参数版本。

开始批次时先提交运行头和全部计划分片。分片开始前提交 `running` 标记；每次完成批次在同一 SQLite 事务中写入该批分片结果并重算累计摘要。正常完成批次在同一事务中写入最终结果摘要和退出报告许可。

### 强制终止与恢复

强杀演示：

```powershell
py -3.11 -m vecflow run --ledger artifacts/crash_ledger.sqlite --snapshot-dir artifacts/crash_snapshots --log-dir logs --reset-state --force-crash-after 2
py -3.11 -m vecflow recover --ledger artifacts/crash_ledger.sqlite --log-dir logs
py -3.11 -m vecflow replay --ledger artifacts/crash_ledger.sqlite --snapshot-dir artifacts/crash_snapshots --log-dir logs
```

强制终止后：

- 只保留已完整提交的分片结果。
- `running`/`queued` 但未提交的分片标记为 `unknown`，要求重跑。
- 不生成强杀时的退出报告，`exit_report_available=false`。

### 一致性重放

`replay` 从账本保存的 `state_json` 重建当时参数版本，不依赖当前数据文件。重放按稳定顺序逐分片比较：

1. `outcome`
2. `result_digest`
3. `resources`

首个差异会写入 `E_REPLAY.details`，包含 `shard_id`、`field`、`expected`、`actual` 和比较依据；最后再比较任务总结果摘要。

## 异常注入

```powershell
py -3.11 -m vecflow validate --inject-range
py -3.11 -m vecflow validate --inject-nullmap
py -3.11 -m vecflow run --inject-budget --reset-state
```

也接受 `--inject range|nullmap|budget`。注入只修改输入参数副本，不伪造错误结果：

- 越界分片由计划器校验，返回 `E_RANGE`。
- 错位空值位图由数据校验器发现，返回 `E_NULLMAP`。
- 全局预算 0 由真实调度器发现，返回 `E_BUDGET`。

三类异常都写入 JSON-lines 结构化日志 `logs/vecflow.jsonl`。

## Tkinter 工作台

```powershell
py -3.11 -m vecflow gui --log-dir logs
```

界面包含：

- 数据块摘要和完整 JSON 编辑器；保存前重新校验，失败统一 `E_DATA`。
- 任务规范化表达式、分片摘要、任务 JSON、全局预算/线程编辑和单任务配额/表达式快捷编辑。
- 稳定排序的分片排队/运行时间线。
- 活动资源单位阶梯曲线和全局预算线。
- 任务结果规范 JSON、总摘要、运行状态。
- 账本格式版本、参数版本、最近批次状态和恢复检查。

无显示环境可做构造冒烟：

```powershell
py -3.11 -m vecflow gui --smoke --smoke-ms 300
```

## 测试与验收

```bash
python3 -m unittest discover -s tests -v
```

测试覆盖校验回滚、表达式规范化、区间切分、预算/配额峰值、异常位图、确定性合流、陈旧失效、SQLite 账本、快照整体恢复、真实子进程强杀恢复、`E_REPLAY` 首差异定位、CLI 注入与 GUI 冒烟。

本次实际验证结果见 [VALIDATION.md](VALIDATION.md)，机器可读结果见 `artifacts/verification.json`。

## 目录

```text
vecflow/             核心包（模型、表达式、计划、引擎、调度、服务、账本、CLI、GUI）
tests/               标准库 unittest 测试
samples/             本地确定性样本与注入样本
tools/               样本生成与最终验收脚本
scripts/             Windows/POSIX 演示脚本
artifacts/           验收输出（不包含外部数据）
```
