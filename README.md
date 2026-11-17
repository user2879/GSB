# 弹性伸缩指标驱动诊断工作台

面向本地 Windows 环境的确定性云原生弹性伸缩诊断工作台。输入是一份本地 JSON 工作负载声明，输出是逐分钟可解释决策、稳定 SQLite 账本、恢复重放校验、双路径一致性证据，以及 Tkinter/HTML 诊断视图。

工作台只访问本地文件，不使用 Docker、外部账号或在线资源。核心算法只依赖 Python 标准库与整数算术，`numpy` 与 `tkinter` 作为本机环境能力由 `doctor` 检查。

## 快速开始（Windows）

推荐 PowerShell 或 `cmd.exe`，在仓库根目录执行：

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python check_env.py
python -m unittest discover -s tests -v
python demo.py artifacts\demo
```

打开诊断界面：

```bat
python -m autoscale_diag ui --db artifacts\demo\workbench.db --run-id demo
```

也可以直接双击/执行：

- `run_demo.bat`：重建三场景演示与报告。
- `run_ui.bat`：打开账本诊断窗口。
- `run_ui_check.bat`：无交互检查视图模型与 Tk 可用性。
- `run_tests.bat`：运行全部测试。

若 Python 安装时未勾选 Tcl/Tk，`tkinter` 不可用；请在 Python 官方 Windows 安装器中启用 `tcl/tk and IDLE` 后重装。SQLite、JSON 是标准库组件。`numpy` 由 `requirements.txt` 安装。

## 交付结构

```text
autoscale_diag/
  spec.py           声明加载与 E_SPEC 强校验
  aggregation.py    全量窗口与增量环形缓冲聚合器
  engine.py         规则估算、预算裁剪、冷却与决策依据链
  ledger.py         每分钟事务账本、恢复标记、整体替换
  replay.py         已完成分钟逐项重放与首个差异定位
  reporting.py     稳定 JSON、快照与自包含 HTML
  viewmodel.py      稳定渲染顺序的 Tk 无关视图模型
  ui.py             Tkinter 诊断工作台
  cli.py            统一命令入口与业务码
  demo.py           三场景可重复演示
samples/
  demo_cluster.json              预算 6、12 分钟、full 自动路径
  demo_cluster_budget_v2.json    预算 5、incremental 自动路径
tests/                            38 项单元/集成测试
artifacts/demo/                   实际运行账本、报告、快照与场景证据
```

## 声明格式

完整样本见 `samples/demo_cluster.json`。核心字段如下：

```json
{
  "declaration_version": "demo-v1",
  "global_replica_budget": 6,
  "aggregation_threshold": 20,
  "window_size": 3,
  "workloads": [
    {
      "id": "checkout",
      "concurrency_per_replica": 25,
      "min_replicas": 1,
      "max_replicas": 4,
      "cooldown_ms": 180000,
      "rules": [
        {
          "id": "checkout-requests",
          "metric": "requests",
          "target_value": 60,
          "per_replica_capacity": 50
        }
      ],
      "observations": [
        {"minute": 0, "requests": 10, "avg_latency_ms": 90, "errors": 0}
      ]
    }
  ]
}
```

强校验规则：

- `declaration_version` 非空；工作负载 ID 全局唯一；规则 ID 在工作负载内唯一。
- 每个工作负载分钟时间戳必须严格连续、每次加 1，不得重复；所有工作负载共享同一分钟网格。
- `requests`、`errors`、`avg_latency_ms` 为非负整数，`errors <= requests`；布尔值不算整数。
- `min_replicas`、`max_replicas`、`concurrency_per_replica` 为正整数，且 `min_replicas <= max_replicas`。
- `cooldown_ms` 为正整数毫秒；`target_value` 为 1..100 的整数利用率百分比。
- 全局预算为正整数，且不小于所有工作负载最小副本数之和，否则约束本身不可满足。
- 任一错误整体拒绝加载，输出 `E_SPEC` 和精确字段路径，例如 `workloads[1].observations[2].errors`。

## 指标与整数估算

每分钟对每个工作负载计算尾部滑动窗口，窗口容量为全局 `window_size`。冷启动时窗口只含当前已有样本：

- 请求指标：`ceil(window_requests_sum / samples)`。
- 延迟指标：`ceil(window_latency_sum / samples)`。
- 错误率：`ceil(window_errors_sum * 1_000_000 / window_requests_sum)`，请求总数为 0 时为 0。

每条规则的估算契约：

```text
effective_capacity = per_replica_capacity * target_value / 100
unclamped = ceil(observed_value / effective_capacity)
          = ceil(observed_value * 100 / (per_replica_capacity * target_value))
expected  = clamp(unclamped, min_replicas, max_replicas)
```

`target_value` 表示目标利用率百分比，`per_replica_capacity` 表示单副本承载的指标单位。输出保留规则 ID、观测值、目标值、承载量、精确分子分母、未夹取需求、夹取后需求，以及整数化量化误差上界：

`concurrency_per_replica` 作为工作负载级并发容量被强校验、保存在声明/账本中并在依据链中展示；具体伸缩公式使用规则自带的 `per_replica_capacity`，从而允许同一工作负载针对不同指标声明不同承载量。

- `quantization_error_upper_bound_milli_replicas = ceil(1000 * remainder / denominator)`，范围为 0..1000。
- `quantization_error_upper_bound_metric_units = ceil(remainder / 100)`。

一个工作负载可声明多条规则：规则按 ID 稳定排序并逐条估算，取夹取后最大值作为期望副本数；最大值并列时选择字典序最小规则 ID。因此重复估算、规则顺序变化和哈希遍历顺序都不会改变结果。

## 预算、冷却与决策

同一分钟的处理顺序固定为：

1. 各工作负载独立按窗口统计和规则计算期望副本数；该步骤无共享副作用，可并行化。
2. 先做一次不含冷却锁定的临时预算分配，用于记录冷却抑制的“尝试目标”。
3. 所有处于冷却期的工作负载先锁定到上轮实际副本数。
4. 在锁定集合之外执行统一硬预算分配，提交全工作负载最终副本数。
5. 比较最终副本数与上轮实际副本数，判定 `expand`、`shrink` 或 `hold`。

预算分配策略为 `proportional-largest-remainder-v1`，全程整数算术：

- 所有 ID 先排序；先为每个可变工作负载保留最小副本数。
- 剩余预算按 `期望副本数 - 最小副本数` 的权重比例分配。
- 整数余数按“余数降序、工作负载 ID 升序”补足。
- 冷却锁定负载先计入占用，再分配其他负载；因此冷却不会突破全局预算。
- 被裁剪工作负载记录裁剪量、依据、受影响集合和完整分配快照。

冷却判定使用分钟差值对应的 60,000 毫秒：

```text
(current_minute - last_change_minute) * 60000 < cooldown_ms
```

冷启动初始实际副本数为 `min_replicas`，没有历史冷却触发点。冷却中的变更被抑制时，最终副本保持上轮值，记录 `E_COOLDOWN`、触发分钟、冷却结束分钟、尝试副本数、最终副本数和依据链。由于上轮实际状态与最近变更分钟均来自账本，进程被强杀或异常后恢复也不能绕过冷却。

## 双聚合路径

- 总分钟数 `< aggregation_threshold`：`full` 路径。保留历史观测，每分钟按尾部窗口全量重算。
- 总分钟数 `>= aggregation_threshold`：`incremental` 路径。单遍扫描，每个工作负载使用固定长度 `deque` 环形缓冲并维护整数总和。
- 可用 `run --path full|incremental` 强制路径，用于同一输入的一致性对照。
- 两个聚合器调用同一整数统计函数；路径决策摘要（排除路径元数据本身）按 SHA-256 比较，必须逐字节一致。
- 报告输出所用路径、窗口容量和峰值内存估计。全量路径按历史行数估计，增量路径按固定窗口行数估计。

样本 12 分钟全量/增量摘要均为：

```text
fd26cacba76eafbe7b976a25d4fa9d1eb83fbdb204ee182724f39cdf3a93095d
```

## 账本与恢复

SQLite 默认使用 WAL 与 `synchronous=FULL`。每个已完成分钟在一个事务中写入：

- `minute_commits`：分钟、路径、窗口容量、峰值内存估计、预算占用、预算分配 JSON、摘要。
- `workload_decisions`：观测、窗口统计、规则中间项、期望/尝试/预算目标/最终副本、决策、冷却和依据链。
- `cooling_suppressions`：独立 `E_COOLDOWN` 记录。
- `runs` 元数据更新时间。

提交成功后该分钟才可见；事务中任一步失败则全部回滚。最后一个分钟的提交与 `run.status=complete` 的最终确认是分开的，因此强杀后：

- 只有已提交分钟可用于恢复。
- 第一个未提交分钟写入 `UNKNOWN_MINUTE_REQUIRES_RERUN`，状态置为 `interrupted`，必须重跑。
- 所有分钟行存在但缺少最终确认时写入 `COMPLETION_UNCONFIRMED`，仍不得声称可生成退出报告；恢复并验证后才可完成。

`--replace` 会先用 SQLite backup 生成 `<db>.bak`，再整体替换同一 `run_id` 的账本内容；旧恢复事件也会清理，避免跨预算版本污染。

## 重放校验

恢复前先核对账本的声明版本、策略参数和声明 SHA-256。随后按账本中的路径与已提交观测序列从第一分钟重新推进，逐项比较：

- 观测与窗口统计。
- 规则中间项、所选规则与期望副本数。
- 尝试、预算目标和最终副本数。
- 决策类型、`E_COOLDOWN` 状态、触发分钟和冷却说明。

比较顺序稳定为分钟、工作负载 ID、字段。首个差异输出 `E_REPLAY`，并包含 `minute_index`、`minute`、`workload_id`、`field`、账本值、重放值和不可复现依据。声明版本、预算/窗口等参数、观测或规则内容不匹配会在重放前直接指出具体字段。

## 业务码与退出码

| 业务码 | 含义 | CLI 退出码 |
|---|---|---|
| `OK` | 成功，或完整账本可出退出报告 | 0 |
| `E_SPEC` | 声明整体非法，附字段路径 | 2 |
| `E_REPLAY` | 声明不可复现或重放首个差异 | 3 |
| `E_LEDGER` | 账本不完整、事务失败或不能出退出报告 | 4 |
| `E_USAGE` | 本地命令/界面环境错误 | 2 |
| `E_COOLDOWN` | 非致命业务诊断码，表示冷却抑制 | 不单独退出 |

CLI 以单行稳定 JSON 输出结果；失败 JSON 写入标准错误。

## 常用命令

```bat
python -m autoscale_diag validate samples\demo_cluster.json
python -m autoscale_diag run samples\demo_cluster.json --db artifacts\demo\workbench.db --run-id demo --out-dir artifacts\demo\v1
python -m autoscale_diag replay samples\demo_cluster.json --db artifacts\demo\workbench.db --run-id demo
python -m autoscale_diag compare-paths samples\demo_cluster.json
python -m autoscale_diag report --db artifacts\demo\workbench.db --run-id demo --json-out out\report.json --html-out out\report.html --snapshot-out out\snapshot.json
python -m autoscale_diag doctor
python -m autoscale_diag ui --check --db artifacts\demo\workbench.db --run-id demo
```

强杀恢复演示：

```bat
python -m autoscale_diag run samples\demo_cluster.json --db artifacts\recover.db --run-id r1 --stop-after 4
python -m autoscale_diag resume samples\demo_cluster.json --db artifacts\recover.db --run-id r1
python -m autoscale_diag replay samples\demo_cluster.json --db artifacts\recover.db --run-id r1
```

`--stop-after` 在提交指定新分钟数后调用 `os._exit(137)`，不执行清理、最终确认或退出报告。

## 演示产物

`python demo.py artifacts\demo` 顺序完成：

1. 冷启动并在第 4 个已提交分钟后强杀，`01_cold_start.json` 显示仅 4 分钟可见。
2. 恢复：登记分钟 4 为未知、重放 4 个已提交分钟、继续到 12 分钟；`02_cooldown_suppression.json` 记录 8 次冷却抑制及完整重放。
3. 修改预算 6→5、声明版本 `demo-v1`→`demo-v2`，备份旧账本并整体替换，自动路径 full→incremental；`03_budget_replacement.json`。
4. `04_path_parity.json` 证明全量与增量摘要相同。
5. `05_replay_guard.json` 用旧声明重放新账本，稳定返回 `E_REPLAY` / `declaration_version`。
6. `v1/`、`v2/` 下各有稳定 JSON 报告、自包含 HTML 和快照；`workbench.db.bak` 保存替换前账本。

可直接用浏览器打开 `artifacts/demo/v1/report.html` 或 `artifacts/demo/v2/report.html` 查看领域可视化。

## 测试

```bat
python -m unittest discover -s tests -v
```

测试覆盖：强校验字段路径、规则整数推导与量化误差、预算顺序无关与硬上限、冷却触发回溯、双路径逐字节一致、每分钟事务回滚、未知分钟登记、恢复续跑一致、声明/参数/观测失配、首个差异定位、替换清理、三场景端到端和视图模型稳定顺序。

## 本次实际验证

验证见 `VALIDATION.md`。核心结论：

- 38 项测试全部通过。
- 三场景演示全部检查通过，`summary.json` 中 `ok=true`。
- 强制终止只留下 4 个完整分钟；恢复时分钟 4 标记未知，重放 12 分钟通过。
- 预算 6→5 整体替换成功，自动路径 full→incremental。
- 全量/增量决策摘要一致，账面预算每分钟不超限。
- 旧声明重放新账本返回 `E_REPLAY`，字段为 `declaration_version`。
- `numpy 2.4.6` 已在用户态安装并确认可用。默认 Linux 解释器未内置 Tk；验证时解包本地 Tk 8.6 运行库后，`doctor --strict` 为 `full_ready=true`，`ui --check --require-tk` 成功实例化完整窗口。核心与 CLI 不依赖两者，可独立运行；Windows 官方 Python 安装时启用 Tcl/Tk 即可。
