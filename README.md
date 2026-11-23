# 声明式配置漂移收敛诊断组件

这是一个面向 Windows 本地环境的、零第三方依赖的声明式配置漂移诊断实现。组件只读取本地 JSON/NDJSON 和 SQLite 账本，不连接集群、不执行任何真实变更。核心目标是：把 v1/v2/v3 历史声明归一为唯一目标状态，根据观测事件流推进实际状态，逐字段给出确定性漂移结论，并提供并发调度、快照替换、崩溃恢复、重放校验和只读 Tkinter 诊断界面。

## 1. 环境与快速开始

要求 Python 3.11+，使用标准库 `sqlite3` 与 `tkinter`，无需网络、Docker 或其他服务。

```powershell
# 工作区根目录执行
python -m unittest discover -v
python scripts/demo.py

# 完整诊断并持久化账本
python -m drift_diag run `
  --declaration samples/declaration-v3.json `
  --events samples/events.ndjson `
  --db .demo-work/ledger.sqlite3 `
  --tolerance 1/6 `
  --global-limit 3 `
  --per-workload-quota 1 `
  --batch-events 4

# Windows 上打开只读诊断界面
python -m drift_diag gui --db .demo-work/ledger.sqlite3
```

`python -m drift_diag check` 会报告 Python、`sqlite3` 与 `tkinter` 可用性。当前交付验证容器缺少 `libtk8.6.so`，因此这里只验证了 Tk 启动依赖探测和 Tk 之外的稳定文本渲染；Windows 官方 Python 安装包通常自带 Tk，应在 Windows 上执行 `check` 和 `gui` 做最终桌面验收。

## 2. 目录

```text
drift_diag/       核心包：加载、归一、漂移、定点量度、传播、调度、账本、重放、CLI、只读 UI
samples/          v1/v2/v3 声明、收敛/边界/发散事件流、变更集样本
tests/            57 个本地单元与集成测试
scripts/demo.py   一键生成账本并做完整重放
README.md         本技术说明与实测记录
```

## 3. 输入格式

### 3.1 期望声明

声明顶层只允许 `format_version` 和 `workloads`。未知顶层字段、未知版本、额外工作负载字段、缺字段、重复工作负载名、重复 JSON 键都会以 `E_FORMAT` 拒绝。

| 版本 | 工作负载必需字段 |
|---|---|
| v1 | `name`, `replicas` |
| v2 | `name`, `replicas`, `image`, `resource_limit` |
| v3 | `name`, `replicas`, `image`, `resource_limit`, `probe_ms`, `labels`, `parent_ref` |

规则：`replicas` 为非负整数；`resource_limit` 为正整数；`probe_ms` 为非负整数；`image` 为非空字符串；`labels` 是字符串到字符串的对象，键按 Kubernetes 风格标签键校验；`parent_ref` 为 `null` 或已有工作负载名。悬空引用和循环引用均拒绝。

v3 示例：

```json
{
  "format_version": 3,
  "workloads": [
    {
      "name": "worker",
      "replicas": 2,
      "image": "registry.local/worker:3.2.1",
      "resource_limit": 2048,
      "probe_ms": 750,
      "labels": {"app": "worker", "env": "prod"},
      "parent_ref": "api-gateway"
    }
  ]
}
```

v1/v2 缺少的更高版本字段由显式、版本化默认规则补齐。每个目标字段都带 `provenance`，默认字段还带 `rule_id`、`defaults_version` 和 `default_value`。

| 字段 | `defaults-v1` | `defaults-v2` |
|---|---|---|
| `image` | `registry.local/default:1.0` | `registry.local/default:2.0` |
| `resource_limit` | `1024` | `2048` |
| `probe_ms` | `500` | `750` |
| `labels` | `{"defaulted":"v1"}` | `{"defaulted":"v2"}` |
| `parent_ref` | `null` | `null` |

v1 会补齐后五项，v2 会补齐 `probe_ms`、`labels`、`parent_ref`。统一目标始终按固定字段顺序保存：`replicas`, `image`, `resource_limit`, `probe_ms`, `labels`, `parent_ref`；`name` 是工作负载标识，不作为漂移字段。

### 3.2 观测事件流

NDJSON 每行一个事件：

```json
{"seq":7,"workload":"worker","field":"replicas","value":2,"at_ms":700}
```

- `seq`：从 1 开始、连续且唯一的整数。文件行可以乱序，加载后按 `seq` 排序推进。
- `workload`：必须存在于归一目标中。
- `field`：必须是该工作负载已知六个字段之一。
- `value`：整数字段拒绝布尔值；`labels` 必须是合法标签对象；`parent_ref` 为 `null` 或字符串。
- `at_ms`：非负整数；按 `seq` 排序后必须单调不减。

未知事件键、未知字段路径、未知工作负载、类型冲突、序号重复/缺口、时间倒退均以 `E_EVENT` 拒绝。未观测字段仍参与漂移字段占比，并记录为 `missing_observation`；缺失数值不虚构观测值，因此不进入数值距离量度。

## 4. 归一、替换与字段版本

### 4.1 整体替换

SQLite 快照签名包含：声明格式版本集合、声明内容摘要、统一目标摘要、默认规则版本、定点缩放口径版本/指数、收敛容差。任一不一致时：

- 整体替换全部目标状态，不做旧/新字段增量合并；
- 同一事务清除不匹配的旧批次与判定结论；
- 旧声明完整写入 `declaration-backups/<declaration_digest>.json`，并设为只读；
- 输出稳定排序的 `added`、`removed`、`upgraded` 字段路径和详细差异。

“升级字段”包括目标值变化或字段来源/默认规则/格式版本等 provenance 变化。

### 4.2 确定性字段版本

每条判定的 `field_version` 是由“工作负载、字段、目标值、provenance、观测是否存在、观测值”派生的稳定正整数。输入未变则版本逐字节不变；目标或观测变化则会变化。它不依赖调度线程、完成顺序或计时。

## 5. 漂移判定与定点量度

每个字段输出目标值、观测值、是否观测、`drift`、比较依据、原因、字段版本和可选定点量度。比较依据：

- 整数：`integer_fixed_point`
- 镜像：`string_exact`
- 标签：`label_map_canonical_exact`
- 父引用：`nullable_reference_exact`
- 未观测：`observation_presence_exact`

数值量度使用整数定点：

```text
factor = 10^scale_exp
q = saturate_int64(truncate_toward_zero((target - observed) * factor, 1))
numeric_drift_metric_units = sum(abs(q))
```

- 截断方向：向零截断；当前输入均为整数，故实际不会发生小数截断。
- 溢出：饱和到有符号 64 位范围。
- 单字段量化误差上界：整数输入为 `0`。
- 工作负载误差上界：所有已测数值字段单字段误差之和。
- 收敛：`drift_field_count / field_count <= tolerance`，使用整数交叉乘法精确比较。
- 边界相等时采用包含式决胜：等于容差即收敛；输出 `boundary_sensitive`、`boundary_rule_applied`、`error_affected` 和固定决胜规则文本。

例如 6 个字段中 1 个漂移、容差 `1/6` 时，占比恰在边界，稳定判为收敛并标注边界敏感。样本 `events-boundary.ndjson` 复现此情况；`events-diverge.ndjson` 则有 2 个字段漂移，不收敛。

## 6. 失效传播与局部重算

`propagate` 支持变更集：

```json
{
  "target_updates": [{"workload": "worker", "field": "replicas", "value": 3}],
  "format_upgrade": {"from": 2, "to": 3},
  "default_rules_change": {"to": "defaults-v2"},
  "delete_workloads": ["legacy"]
}
```

- 单字段目标修改：只影响该工作负载该字段。
- 格式升级：所有字段 provenance 受影响，稳定重算全部目标字段。
- 默认规则调整：只影响当前 provenance 为 `default` 的字段；显式声明字段保持原判定与版本。
- 工作负载删除：移除该工作负载的所有判定；若仍被其他 `parent_ref` 引用，以 `E_REF` 返回引用者列表。
- 报告按工作负载名、固定字段顺序输出传播原因，并逐项列出受影响/未受影响/已删除判定；`unaffected_byte_stable` 校验未受影响判定完全不变。

## 7. 并发调度

每个工作负载字段是一个真实判定任务，提交给 `ThreadPoolExecutor` 工作线程。任务按字段优先、工作负载名排序后分轮提交；每轮最多 `global_limit` 个任务，线程池也是同一上限。每个工作负载另有 `threading.Semaphore` 配额。只有获得配额后任务才计入运行数并执行比较。

调度轨迹包括：

- 每轮任务分配、工作负载、字段、工作线程名与线程标识；
- `queue_wait_start_ms` / `queue_wait_end_ms` / `queue_wait_ms`；
- 配额 `quota_wait_start_ms` / `quota_wait_end_ms` / `quota_wait_ms`；
- 执行起止毫秒、并发进入/退出轨迹、全局与单工作负载峰值；
- `assignments_all_on_workers`、`quota_respected`、`global_limit_respected`。

计时来自 `time.monotonic_ns()`，因此轨迹本身不承诺跨运行逐字节相同；最终判定先按工作负载名、再按固定字段顺序排序，所以并发布局和完成顺序不会改变漂移集合或收敛结论。

## 8. SQLite 账本、快照与崩溃恢复

账本包含 `snapshot`、`replacements`、`batches`、`decisions`、`inflight`、`unknown_workloads` 和整数格式版本元数据。每个批次在**一个 SQLite 事务**内写入：

- 该批全部逐字段判定；
- 工作负载定点量度、误差界与收敛结论；
- 完整调度轨迹；
- 事件进度游标和首事件序号；
- 批次内容摘要。

读取时会用存储内容重算摘要；任何内容篡改产生 `E_SNAPSHOT`。

批次开始前会单独提交 `inflight` 标记。进程被强制终止时，未提交批次自动回滚；下次打开账本先执行恢复，把 inflight 中的工作负载写入 `unknown_workloads`，只从最后一个完整提交游标继续并将这些工作负载要求重跑。强制终止时不会生成或伪装退出报告；`--crash-after-attempt` 直接以退出码 91 结束进程，用于复现测试。成功提交后对应 unknown 标记清除。

## 9. 重放校验

每次 `run` 结束都会重放账本中的全部已提交批次；`verify` 可单独执行。重放使用快照中的声明、默认规则、缩放口径与容差，从空实际状态按事件游标推进，重算逐字段判定和批次量度，再与账本逐项比较。

首个差异报告 `E_REPLAY`，定位到：

- 工作负载；
- 字段；
- 字段类别：`target`、`observed`、`drift`、`reason`、`basis`、`metric`、`field_version`、`membership`、`convergence`；
- 批次键、事件游标、账本值与重放值。

账本摘要被篡改会先以 `E_SNAPSHOT` 拒绝，不会进入重放比较。

## 10. 异常注入与业务码

```powershell
# 声明 JSON 损坏：E_FORMAT
python -m drift_diag run ... --inject declaration-corrupt

# 观测事件字段路径未知：E_EVENT
python -m drift_diag run ... --inject unknown-event-field

# 已有账本决策内容被篡改、摘要不符：E_SNAPSHOT
python -m drift_diag run ... --inject ledger-digest
```

| 业务码 | 含义 | CLI 退出码 |
|---|---|---|
| `E_FORMAT` | 声明/变更集/配置格式非法 | 2 |
| `E_EVENT` | 事件序号、工作负载、字段或类型非法 | 3 |
| `E_REF` | 父引用悬空、循环或删除仍被引用工作负载 | 4 |
| `E_SNAPSHOT` | 账本/快照版本、摘要或内容不一致 | 5 |
| `E_REPLAY` | 重放首个差异 | 6 |
| `E_UI` | 当前 Python 无法加载 Tkinter | 7 |

错误 JSON 包含 `code`、`message`、`item`、`field`、`details`。非法条目会指出稳定顺序中的首个条目与字段。

## 11. Tkinter 诊断界面

`gui` 通过 SQLite 只读 URI 打开账本，刷新只重建文本，不创建表、不更新元数据、不写批次。界面展示：

- 归一目标、来源格式、provenance、被补齐字段及规则；
- 最新批次逐字段目标/观测/漂移/依据/字段版本/定点量度；
- 各工作负载占比、收敛结论、误差上界与边界标注；
- 调度轮次、线程、配额等待与执行时长；
- 全部账本批次、摘要、完整性校验与快照替换历史。

渲染顺序固定：工作负载名排序、固定字段顺序、账本次序。无显示环境也可调用 `drift_diag.dashboard.render_dashboard_text()` 得到完全相同且只读的稳定文本。

## 12. 实测结果

在本交付容器使用 Python 3.11.0rc1、SQLite 3.37.2 执行：

```text
python -m unittest discover -v
Ran 57 tests in 5.474s
OK
```

覆盖：三版本强校验、首个非法字段、乱序事件按序号归一、默认规则追溯、整体替换与只读备份、确定性漂移、定点饱和/截断、边界决胜、传播局部重算、E_REF、真实线程与双重配额、批次事务、摘要篡改、容差变更替换、崩溃 inflight 恢复、E_REPLAY 首字段定位、三类 CLI 注入、只读界面刷新不变。

`python scripts/demo.py` 实测输出：

```text
ledger=.../.demo-work/ledger.sqlite3
batches=6 latest_cursor=18
overall_converged=True
replay_ok=True decisions_checked=108
```

`events-boundary.ndjson` + `--tolerance 1/6`：`cache/probe_ms` 单字段漂移，占比 `1/6`，边界敏感且收敛。`events-diverge.ndjson`：`cache/resource_limit` 与 `cache/probe_ms` 两个字段漂移，不收敛。`--crash-after-attempt` 实测退出码 91，重启后恢复三个 unknown 工作负载并只从最后提交批次继续。

## 13. 边界与假设

- 不执行真实集群操作；观测值仅来自本地事件样本。
- 一个本地 SQLite 账本对应一个当前快照；快照变化会整体替换旧判定结论，但旧声明和替换审计保留。
- 标签作为完整映射字段比较，不拆成 `labels.<key>` 事件路径。
- 数值输入必须是整数；因此整数定点缩放的单字段量化误差为 0。边界标记仍保留，以避免策略边界被隐式改变。
- 缩放指数限定为 `0..18`，超过有符号 64 位的量度按饱和规则截断。
- 调度计时天然受操作系统线程调度影响；确定性和逐字节一致性只承诺用于漂移集合、字段版本、量度与收敛结论，不承诺计时轨迹相同。
