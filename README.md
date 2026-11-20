# Trace Diag

面向 Windows 本地环境的确定性日志与追踪采样诊断组件。应用从本地 JSON 读取服务样本声明和候选采样策略，生成逐字节可复现的记录流，完成头部/尾部采样、定点分位数、逐桶误差传播、参数翻转诊断，并把完整批次写入 SQLite 账本。Tkinter 工作台可编辑策略参数和服务描述、查看样本、逐桶统计、误差热力、归因链、翻转与账本批次。

运行时不访问网络、不依赖容器或外部服务。样本、策略、局部 SQLite 账本和结构化 JSONL 日志都保存在项目目录。

## 快速开始（Windows）

要求 Python 3.11 或更高版本，并选择带 Tcl/Tk 的官方安装包。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -V
pip -V
pip install -r requirements.txt
python -c "import tkinter; print('tkinter', tkinter.TkVersion)"
```

生成完全本地的确定性样例：

```powershell
python -m trace_diag generate-samples --data-dir data --records 600
```

执行分析、提交 SQLite、深重放并写出诊断报告：

```powershell
python -m trace_diag demo
```

打开桌面工作台：

```powershell
python -m trace_diag gui
```

也可以安装为可编辑包后使用 `trace-diag` 命令：

```powershell
pip install -e .
trace-diag demo
trace-diag gui
```

默认文件位置：

```text
data/services.json                  服务与记录字段声明
data/policies.json                  候选采样策略
data/ledger/trace_diag.sqlite       SQLite 账本
data/logs/trace_diag.jsonl          结构化日志
data/diagnostic_report.json         最近一次 demo 的完整报告
```

## 常用命令

```powershell
python -m trace_diag analyze --services data/services.json --policies data/policies.json --db data/ledger/trace_diag.sqlite --log-dir data/logs --report-out data/report.json
python -m trace_diag status --db data/ledger/trace_diag.sqlite --log-dir data/logs
python -m trace_diag verify --db data/ledger/trace_diag.sqlite --log-dir data/logs
python -m trace_diag recover --db data/ledger/trace_diag.sqlite --log-dir data/logs
```

`analyze` 的流程是“校验全部输入 → 生成并统计分析 → 创建新 run → 每服务一个事务提交 → 最后激活新 run”。内容相同的当前 run 会被复用；内容改变后由账本分配单调递增的样本版本或策略版本。

### 异常注入

```powershell
python -m trace_diag analyze --inject sample
python -m trace_diag analyze --inject policy-ratio
python -m trace_diag analyze --inject snapshot
```

- `sample`：内存中破坏服务声明，整体拒绝并报告 `E_SPEC`，例如 `$.services[0].id`。
- `policy-ratio`：内存中把首条策略头部比例改为 10001，以诊断用途报告 `E_BUCKET`；正常输入校验仍严格使用 `E_SPEC`。
- `snapshot`：先确保存在有效账本，再临时篡改 run 快照摘要以触发 `E_SNAPSHOT`，检测后恢复原摘要。

每次注入都会追加到 `data/logs/trace_diag.jsonl`，包含 `code`、`path`、`event` 和 `context`。CLI 退出码：`0` 成功，`2` 规格/记录/桶错误，`3` 重放不一致，`4` 快照摘要不符。

## JSON 声明

服务文件示例：

```json
{
  "version": 1,
  "services": [
    {
      "id": "api-checkout",
      "seed": 1103,
      "record_count": 600,
      "description": "Checkout API",
      "fields": {
        "timestamp_ms": {"type": "int", "start": 1735689600000, "max_step_ms": 41},
        "magnitude": {"type": "int", "min": 1, "max": 5},
        "client_key": {"type": "string", "key_space": 173},
        "duration_us": {"type": "int", "min_us": 900, "max_us": 180000},
        "trace_id": {"type": "string"},
        "sampled": {"type": "bool", "rate_bp": 2000}
      }
    }
  ]
}
```

字段必须完整声明 `timestamp_ms`、`magnitude`、`client_key`、`duration_us`、`trace_id`、`sampled`。未知字段、重复 JSON 键、重复服务标识、非正种子/记录数、越界参数都会在加载阶段整体拒绝，并给出字段路径。

策略文件示例：

```json
{
  "version": 1,
  "policies": [
    {
      "id": "head-balanced",
      "head_ratio_bp": 4200,
      "tail_duration_threshold_us": 50000,
      "tail_ratio_bp": 1800,
      "scale_power": 1,
      "bucket_count": 32
    }
  ]
}
```

约束：两类比例均为 `0..10000` 整数，尾部阈值为正整数，`scale_power` 为 `0..6` 整数，`bucket_count` 为 `2..256` 整数。每条策略可带可选 `version` 和 `description`。

## 统计口径

- 定点粒度为 `quantum_us = 10^scale_power` 微秒；记录值按 `floor(duration_us / quantum_us)` 向下截断，输出值再乘回粒度，因此单记录最大量化误差是 `quantum_us - 1`。
- 分位桶按排序后的耗时选择分位切点。桶区间为 `[lower_us, upper_us_exclusive)`，最后一个桶包含最大值；切点相等的记录归入右侧桶。归属使用排序值和 `bisect_right`，不依赖输入遍历顺序。
- 分位数使用整数排序秩上的线性插值，最终向下取整到定点粒度。采样估计使用重标定权重 `n / sample_hits` 计算各桶估计占用，再把桶中点向下取整到定点粒度。
- 归因把逐桶误差精确拆为 `sampling_sparse + fixed_truncation + bucket_boundary`，每段给出有符号贡献、上界、占比和主导段。主导规则是“占比至少 1/2 且严格高于其他段”，否则为 `balanced`。
- 分位数的合成误差界由“采样 CDF 最大秩偏差形成的可能带 + 双侧定点截断 + 选中桶半宽”组成，确保 `estimated_value_in_error_bound` 是严格最坏情况判定，而不是只比较近似值。
- 服务排序：估计 p95、服务标识、策略版本、候选策略版本、候选策略标识依次决胜。候选策略排序：估计均值、候选策略版本、候选策略标识依次决胜。
- 差值不超过双方误差界之和时标记“受误差界影响”。参数敏感性扫描逐项改变 `scale_power ± 1` 或 `bucket_count ± 1`，保持版本不变，输出排序或候选策略选择的实际翻转。

完整数学定义见 `docs/DESIGN.md`。

## SQLite 账本与恢复

账本格式版本为整数 `1`，存于 `meta` 表，并写入每个 run。主要表为 `runs`、`batches`、`sample_summaries`、`bucket_stats`、`error_bounds`、`attribution_chain`、`flips`。

- 创建 run 时先写入 `pending` 批次并立即把旧 run 标记为 `stale`。
- 每完成一个服务，在一个 SQLite 事务中写入样本摘要、全部逐桶统计、定点误差、归因、翻转、批次摘要、进度游标和 `committed` 状态。
- 全部批次提交后，在同一收尾事务中计算并写入 run 快照摘要，设为唯一活动 run；旧版本不再参与当前展示。
- 进程强制终止后，重启会把未提交批次标记为 `unknown`，并把 run 置为 `recovering`。`recover` 只复用 `committed` 批次，重新执行 `unknown` 批次，然后整体激活。
- `verify` 独立读取账本中的服务/策略快照，按记录版本和缩放因子重算样本流、分位统计、逐桶归因和翻转，与账本逐项比较；首个不一致会报告服务标识和字段路径。

## Tkinter 工作台

工作台展示服务列表、策略参数与版本、逐桶记录数/采样命中、定点分位统计和误差界热力、逐桶归因链、误差界内判定与参数翻转、run 历史和当前批次。

- “刷新当前视图”只读取活动账本，不写入账本。
- “重算并整体替换”提交新 run。
- 策略参数编辑会校验后递增策略版本并整体替换。
- 服务描述可编辑，保存时递增样本版本。
- 样本页支持客户端键/追踪标识过滤、按序号/耗时/客户端键排序和分页。
- “重放校验”调用独立账本重算器。

## 测试

```powershell
python -m pytest -q
```

测试覆盖：规格字段路径与整体拒绝、重复键、记录确定性、时间戳倒退/耗时越界定位、头部尾部去重、分位桶归属、定点截断、归因可加性与顺序无关、误差界内判定、参数翻转稳定性、SQLite 原子批次、强制终止恢复、版本替换、快照注入、深重放和 CLI 三类异常。

本次实际验证结果见 `docs/VALIDATION.md`。

## 目录

```text
trace_diag/spec.py              JSON 强校验
trace_diag/generator.py         确定性记录流与摘要
trace_diag/sampling.py          稳定分区与头尾采样
trace_diag/stats.py             分桶、定点分位、估计与归因
trace_diag/diagnostics.py       排序、误差界判定与参数翻转
trace_diag/ledger.py            SQLite 账本、事务、版本和恢复
trace_diag/replay.py            独立重算校验
trace_diag/workflow.py          分析到账本的编排
trace_diag/gui.py               Tkinter 工作台
trace_diag/cli.py               命令入口和异常注入
tests/                          自动化验收测试
docs/                           设计与验证记录
```
