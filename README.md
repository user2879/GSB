# 服务调用链错误预算与熔断联动诊断工作台

这是一个仅依赖 Python 标准库的本地诊断工作台，面向 Windows 10+ / Python 3.11+。它按 NDJSON 调用顺序自底向上推进每层服务的整数错误预算与熔断状态，并把“深层服务的预算耗尽或熔断打开是否导致浅层上游快速失败”输出为可审计的显式决策。

## 交付内容

- `srebudget/`：可集成的 Python 包，含校验、状态机、SQLite 快照/批次、重放比较和 Tkinter 界面。
- `data/`：脚本生成的确定性服务链、调用流、SQLite 会话库与演示报告。
- `examples/generate_samples.py`：生成离线样本。
- `examples/run_demo.py`：完整演示，含快照重放、整状态恢复、真实子进程强杀恢复与三类异常注入。
- `examples/crash_worker.py`：在已提交批次尝试后执行 `os._exit(86)`，模拟强制终止。
- `tests/`：18 项标准库 `unittest`，覆盖核心验收路径。
- `run_demo.bat`、`open_ui.bat`、`verify.bat`：Windows 命令行入口。

## 快速开始

在 Windows 项目根目录执行：

```bat
python --version
python -c "import tkinter"
python examples\run_demo.py
python -m unittest discover -v
```

等价地，可直接双击或运行：

```bat
run_demo.bat
verify.bat
open_ui.bat
```

演示报告写入 `data/demo_report.json`。界面只打开本地 Tk 窗口，不监听网络端口。

## 服务链声明

`data/chain.json` 由 `examples/generate_samples.py` 生成，结构如下：

```json
{
  "version": "chain-2026.11.20-r1",
  "services": [
    {
      "id": "api",
      "upstream": ["gateway"],
      "timeout_ms": 1800,
      "budget": {
        "window_hours": 1,
        "ratio_bp": 5000,
        "min_samples": 3
      },
      "circuit": {
        "failure_threshold": 3,
        "cooldown_ms": 2000,
        "half_open_probes": 1
      }
    }
  ]
}
```

`upstream` 表示调用该服务的上游调用者。调用流 `path` 按“由深到浅”排列，例如 `["db", "orders", "api", "gateway"]`。因此 `db` 的状态触发会依次使 `orders`、`api`、`gateway` 快速失败。

强校验包括服务标识唯一、上游引用存在、声明图无环、整数类型约束、超时/冷却/最小采样数/阈值/试探数为正整数、预算比例位于 `0..10000`。错误统一为 `E_CHAIN`，并带有如 `$.services[2].budget.ratio_bp` 的字段路径。NDJSON 校验时间戳单调不减、路径服务存在、结局属于 `success/failure/timeout`、耗时为非负整数；错误为 `E_TRACE` 并带物理行号。

## 状态推进规则

### 错误预算

每个服务独立维护窗口事件与窗口内采样/失败计数。对时间戳 `t`，窗口为 `[t - window_hours * 3600000, t]`。所有计算均为整数：

```text
budget_total   = window_samples * ratio_bp // 10000
raw_remaining  = budget_total - failures
remaining      = max(0, raw_remaining)
exhausted      = window_samples >= min_samples and raw_remaining < 0
```

`raw_remaining` 为负时，余量截断为零并记录 `budget_exhausted` 事件；窗口滑出或采样降至最小值以下后可记录 `budget_recovered`。不使用浮点累加。

### 熔断

成功将连续失败计数清零；`failure/timeout` 递增连续失败计数。达到阈值后进入 `open`，记录打开次数和单调时钟打开时刻。冷却结束后转为 `half_open`，成功试探关闭熔断并清零计数，失败或超时试探重新打开并递增打开次数。顺序流中每次试探会立即得到结局；`half_open_probes` 作为半开放行上限参与门控。所有迁移事件包含触发调用序号、连续计数、冷却剩余毫秒与原因。

生产默认使用 `time.monotonic_ns()` 派生的毫秒时钟。为让样本、测试和快照回放逐字节可复现，CLI 的 `--clock trace`、演示脚本和测试使用受控逻辑单调时钟；界面恢复会话时沿用快照记录的时钟游标。该时钟仍只增不减，且状态机从不直接读取墙上时间。

### 快速失败

每次调用按 `path` 自底向上处理。遇到以下任一条件即停止继续向浅层消耗：

- 当前服务熔断为 `open` 且冷却未结束；
- 当前服务在本次采样后预算首次变为耗尽；
- 当前服务在本次失败/超时试探后重新打开熔断。

触发源自身若实际被采样，则消费一次；触发源之上的所有服务标记为 `fast_failed`，不消耗预算、不改变熔断计数。每个快速失败结果都输出 `trigger_source`、`trigger_reason`（`budget_exhausted` / `circuit_open`）和 `propagation_depth`。若预算与熔断同一调用同时触发，固定选择 `budget_exhausted`，避免依赖字典遍历顺序。

## 批次、快照与恢复

- 每个批次先以独立事务写入 `started` 尝试记录；随后在同一事务中写入批次内全部调用结论、完整预算/熔断状态、内容摘要和已提交游标。
- 进程在事务提交前终止时，重启只加载已完整提交批次；`started` 批次被标记为 `unknown`，范围内未提交调用写入 `unknown_calls` 并要求重跑。
- 快照写入 SQLite，包含服务链版本/摘要、窗口事件与计数、余量、连续失败数、熔断相态/冷却起点、打开次数、事件游标、事件流、结论摘要和 SHA-256 内容摘要。
- 打开快照先校验内容摘要，不符报告 `E_SNAPSHOT`。服务链版本、链摘要或调用流摘要不一致时，不进行字段级增量合并，而是将旧结论全部作废、旧批次标记为被替换并重置为新链的完整状态，稳定返回受影响服务与作废条数。
- 同链快照恢复执行整状态替换，回退到快照游标，并作废该游标之后的结论；恢复检查点保证后续批次不会重新加载更靠后的旧状态。

## 重放校验

`ReplayVerifier` 先独立推进前半段作为基线，再从快照状态继续推进剩余调用。它逐调用比较：

- 预算窗口采样数、失败数、预算总量、原始余量、截断后余量；
- 连续失败数、熔断相态、冷却剩余毫秒、打开次数；
- 快速失败触发源、原因、传递深度与传递链。

首个差异以 `E_REPLAY` 返回 `call_index`、`service`、`field`、`expected`、`actual` 和依据。服务链版本变化、窗口参数变化或调用流内容变化属于不可复现依据，返回 `kind=incompatible` 与单独 `basis`，不与实现发散混同。

## 命令行

```bat
python -m srebudget.cli generate --data-dir data

python -m srebudget.cli run --data-dir data --session-id demo-session ^
  --batch-size 4 --clock trace --snapshot-at 5 --show-decisions

python -m srebudget.cli replay --data-dir data --session-id demo-session

python -m srebudget.cli inject chain_corrupt --data-dir data
python -m srebudget.cli inject timestamp_regression --data-dir data
python -m srebudget.cli inject snapshot_digest_mismatch --data-dir data

python -m srebudget.cli ui --data-dir data --session-id demo-session
```

异常注入只创建或修改输入文件/参数：损坏的链副本、时间戳倒退的调用流副本、内容摘要被篡改的 SQLite 副本。事件写入 `data/logs/events.ndjson`，业务码分别为 `E_CHAIN`、`E_TRACE`、`E_SNAPSHOT`。

注入命令退出码为 20/21/22；重放发散为 23。命令行始终输出结构化 JSON，便于本地脚本集成。

## Tkinter 诊断界面

界面固定展示：

- 服务依赖边，按服务标识和上游标识排序；
- 各服务窗口采样、失败数、预算总量、余量、连续失败数；
- 熔断相态、冷却剩余毫秒、打开次数；
- 最近结论中的快速失败传递链、触发源、原因和深度；
- 当前服务链版本/摘要、会话游标与快照列表。

界面可推进一批、保存快照、对选中快照执行整状态替换恢复。Tkinter 随标准 Windows CPython 安装；当前 Linux 验证容器未安装 Tk，因此这里完成了模块编译检查，实际窗口需在 Windows 目标环境运行。

## 验证结果

实际运行结果见 `VERIFICATION.md` 和 `data/demo_report.json`。本次验证包含：

- 18 项 `unittest` 全部通过；
- 12 条确定性样本调用，产生 10 条快速失败调用，预算触发 6 次、熔断打开触发 4 次；
- 第 5 条调用后的快照恢复并逐调用比较剩余 7 条，全部一致；
- 子进程以退出码 86 强制终止后，识别批次 2 的调用 5–8 为未知并重跑成功；
- 三类注入分别报告 `E_CHAIN`、`E_TRACE`、`E_SNAPSHOT` 并写入 NDJSON 日志。
