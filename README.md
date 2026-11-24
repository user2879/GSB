# 迟到事件位移与水位账本诊断台

这是一个完全在 Windows 本地运行的确定性事件时间诊断工作台。它不只展示延迟，而是把以下判定收敛为可复核的 SQLite 账本：

- 同一事件流在 `source_max`、`partition_min`、`fixed_step` 三种水位推进策略下何时定稿；
- 同一窗口在迟到预算恰好耗尽时接纳还是丢弃；
- 迟到接纳或派生边界变化后，哪些下游窗口被位移、如何按稳定拓扑序重算；
- 定点聚合误差怎样拆成量化截断、迟到接纳、边界归属三段；
- 快照整体替换、强制终止后的未知窗口、恢复后的逐项重放定位。

不使用 Docker、容器或外部服务。事件源、窗口规则和派生/预算声明均来自本地 UTF-8 JSON。

## Windows 准备

```powershell
python --version                    # 需要 3.11 或更高
python -c "import tkinter; print(tkinter.TkVersion)"
python -m pip install "numpy>=1.26"
python -m pip install -e .
```

Windows 官方 Python 安装包默认包含 Tk。若 `tkinter` 导入失败，重新运行安装包并勾选 Tcl/Tk。SQLite 使用 Python 标准库，不需要额外服务。

## 快速开始

在仓库根目录运行：

```powershell
python -m lateness_ledger validate `
  --sources samples/sources_source_max.json `
  --rules samples/window_rules.json `
  --budgets samples/derivation_budget.json

python -m lateness_ledger run `
  --sources samples/sources_source_max.json `
  --rules samples/window_rules.json `
  --budgets samples/derivation_budget.json `
  --ledger .\work\source_max.sqlite

python -m lateness_ledger replay `
  --sources samples/sources_source_max.json `
  --rules samples/window_rules.json `
  --budgets samples/derivation_budget.json `
  --ledger .\work\source_max.sqlite
```

其余两种策略只需替换 `--sources`：

- `samples/sources_partition_min.json`
- `samples/sources_fixed_step.json`

三份源声明除会话标识和策略外完全一致，并用固定种子生成同一事件流；样本同时覆盖 `out_of_order_ppm=0`（`alpha_clean`、`gamma_paced`）和 `9500`（`beta_bursty`）。

## 常用命令

```powershell
# 快照
python -m lateness_ledger snapshot --ledger .\work\source_max.sqlite --out .\work\snapshot.sqlite

# 整体替换恢复；输出 added / removed / scale_upgraded 稳定清单
python -m lateness_ledger restore --ledger .\work\target.sqlite --snapshot .\work\snapshot.sqlite

# 强制终止恢复分类。未提交窗口标为 unknown，不生成退出报告
python -m lateness_ledger recover --ledger .\work\source_max.sqlite

# 只读状态汇总
python -m lateness_ledger status --ledger .\work\source_max.sqlite

# Tkinter 诊断界面
python -m lateness_ledger ui --ledger .\work\source_max.sqlite

# 真实回环套接字采样；server-delay-ms 大于 timeout-ms 时按次数真实超时重试
python -m lateness_ledger probe `
  --sources samples/sources_source_max.json `
  --rules samples/window_rules.json `
  --budgets samples/derivation_budget.json `
  --timeout-ms 30 --attempts 3 --server-delay-ms 200
```

同一账本已包含完整结论时，重新运行需显式 `--reset`；崩溃留下的未完成账本应先 `recover`，或用新账本重跑。

## 异常注入

```powershell
python -m lateness_ledger run ... --inject order       # E_ORDER，退出码 3
python -m lateness_ledger run ... --inject budget      # E_BUDGET，退出码 4
python -m lateness_ledger run ... --inject watermark   # E_WATERMARK，退出码 5
```

每次运行默认在账本旁写 `<ledger>.events.jsonl`，也可用 `--log-json` 指定。异常记录包含稳定错误码、事件/窗口证据和 UTC/单调时钟时间。

## 输入 JSON 口径

- **源声明**：`session_id`、`scale`（0..6 整数）、`strategy`、`sources[]`。每个源包含唯一的 `source_id`、非负整数 `seed`/`event_count`/`start_ms`、正整数 `pace_ms`、`0..10000` 的 `out_of_order_ppm`、`partition_values` 以及十进制量度范围。
- **窗口规则**：`rules_version` 和 `rules[]`。每条规则包含正整数 `width_ms`、`allowed_lateness_ms`，非空 `partition_keys`，`aggregate_functions` 子集（`count`、`sum`、`mean`、`weighted_mean`），以及量度/权重列名。
- **派生与预算**：`derivations[]` 指定父/子规则与 `offset_ms`；子窗口结束边界等于父窗口结论结束边界加偏移。`late_budgets` 支持 default、规则覆盖和窗口覆盖；绝对预算使用会话缩放因子下的定点整数。

任一重复 ID、负数/非整数值、越界 ppm、缺失分区列、未知派生目标或多父、派生环都会在写账本前以 `E_SPEC` 整体拒绝，并报告首个非法源或窗口。

完整字段和语义见 [docs/DESIGN.md](docs/DESIGN.md)。

## 确定性、定点与并发

- 事件流按排序后的源标识和源内序号轮转生成；所有窗口、分区、水位轨迹、派生边和拓扑序都使用显式稳定排序，不依赖字典/哈希遍历。
- 滚动窗口的定稿条件是 `end_ms <= global_watermark - allowed_lateness_ms`。流尾以逻辑 EOS 水位排空；`fixed_step` 仍按配置步长逐级追赶。
- 固定点：`trunc(10^scale * value)`，向零截断；整数累加按 int64 饱和；除法向零截断。窗口行显式携带 `truncation_direction`、`overflow_rule`、单值量子、聚合上界和均值上界。
- 三个误差段以定点整数记录：量化截断、迟到接纳绝对量、派生边界新增事件绝对量。主导段阈值为 `1/2`；均以精确整数比较。
- 定稿调度使用真实线程池和单调时钟。全局并发上限与每分区配额同时生效，排队任务记录真实等待开始/结束；超时任务记录实际尝试耗时并重试。

## SQLite 与恢复

- 每个窗口的定点统计、迟到证据、三段误差、进度游标和位移轨迹在**同一事务**中写入 `windows`，并携带整数格式版本 `1` 和内容摘要。
- `run_plans` 在运行前提交全部窗口计划；每个窗口提交后更新计划状态。进程被强制终止时 SQLite 回滚未提交事务，重启只看到完整提交行，其余计划稳定标为 `unknown`。
- 快照是完整的 SQLite 备份。恢复永远整体替换窗口结论、迟到证据、位移、计划和分区水位，不做增量合并；差异按 `added`、`removed`、`scale_upgraded` 排序输出。
- 重放按账本记录的规则版本、缩放因子、策略和事件游标重新推进；首先比较元数据与水位轨迹，再按窗口 ID 比较边界、定稿次序、触发序号、定点统计、迟到裁决、误差和位移。首个差异返回分区键、窗口边界、字段、期望值与实际值并报 `E_REPLAY`。

## 界面

`python -m lateness_ledger ui ...` 打开只读工作台，页签包含：

- 水位轨迹及画布折线；
- 定稿窗口及策略/触发序号；
- 逐事件迟到接纳/丢弃证据；
- 位移传播链、路径、依据和版本变化；
- 三段定点误差与主导段；
- 当前策略、规则版本、账本状态和快照列表。

刷新只重新读取 SQLite，不写账本；所有渲染集合按窗口定稿序号、分区标识和事件序号稳定排序。

## 验证

```powershell
python -m unittest discover -s tests -v
python scripts/acceptance.py --output artifacts/acceptance.json
```

最近一次本机实测：36 个单元/集成测试全部通过；三策略各 36 个窗口且相同窗口键集；`source_max` 接纳 48/丢弃 972、`partition_min` 接纳 46/丢弃 947、`fixed_step` 接纳 47/丢弃 928；三次重放均 `MATCH`；三类异常分别报 `E_ORDER`/`E_BUDGET`/`E_WATERMARK`；强制终止后 2 个已提交、34 个未知。详见 [artifacts/verification.md](artifacts/verification.md) 和 [artifacts/acceptance.json](artifacts/acceptance.json)。

当前 Linux 验证容器缺少 `libtk8.6.so`，因此本机只验证了 Tk 不可用时的 `E_UI`、只读投影稳定性和刷新不写账本；Windows 官方 Python 的 Tk 实际窗口需在目标机执行上述 `ui` 命令。
