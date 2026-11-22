# 多源慢查询证据诊断工作台

本项目从零实现一个面向本地 Windows 环境的慢查询诊断工作台。它不连接真实数据库，而是把慢查询诊断视为**异源证据合并**问题：应用日志摘要、库执行统计、慢日志摘录分别以不同格式版本、上报口径和采样周期描述同一批查询；系统按显式规则选出采纳证据，再把结论迁移到另一来源复核，并将冲突归因到口径、格式版本与采样周期。

所有样本、证据、快照和日志均为本地确定性数据。实际处理耗时使用 `time.perf_counter_ns()` 单调时钟计为整数毫秒，只进入运行指标与结构化日志，不进入要求逐字节稳定的结论视图。

## 快速开始

### Windows

```bat
setup_windows.bat
run_demo_windows.bat
run_workbench_windows.bat
```

也可以直接执行：

```bat
py -3 -m pip install -r requirements.txt
py -3 scripts\validate_environment.py --workspace workbench
py -3 -m slowdiag demo --workspace workbench --pretty
py -3 -m slowdiag gui --workspace workbench
```

### Linux / macOS 开发验证

```bash
python3 -m pip install -r requirements.txt
python3 scripts/validate_environment.py --workspace workbench --json
python3 -m slowdiag demo --workspace workbench --pretty
python3 -m unittest discover -s tests -v
```

`demo` 会依次完成工作区初始化、确定性样本生成、重复摘要校验、证据合并、跨源迁移、冲突归因、快照创建和快照恢复复核。

## 工作区布局

```text
workbench/
  slow-sources.json              # 观测源声明
  sources/<source_id>/records.sqlite3
  ledger/ledger.sqlite3          # 证据账本、当前视图、运行指标、失效历史
  snapshots/*.json               # 诊断会话快照
  reports/latest.json            # 当前确定性视图
  logs/slowdiag.jsonl            # 结构化 JSONL 日志
  _injection/                    # 命令行异常注入的临时输入副本
```

默认样例包含四个源：

| 源标识 | 来源类型 | 声明版本 | 记录版本 | 样本行数 | 周期 | 耗时口径 | 扫描口径 |
|---|---|---:|---|---:|---:|---|---|
| `app-prod` | 应用日志摘要 | 2 | 1,2 | 24 | 100 ms | p95 | estimated |
| `db-prod` | 库执行统计 | 2 | 1,2 | 24 | 250 ms | p95 | exact |
| `slow-prod` | 慢日志摘录 | 2 | 1,2 | 24 | 500 ms | max | exact |
| `slow-canary` | 慢日志摘录 | 1 | 1 | 6 | 1000 ms | max | sampled |

`slow-canary` 只覆盖 `q001–q006`，用于验证局部失效传播：修改它时，`q007/q008` 的结论必须逐字节保持不变。

## 观测源声明

完整示例见 `examples/slow-sources.example.json`。核心字段如下：

```json
{
  "schema": "slowdiag.sources/v1",
  "dataset": {
    "query_count": 8,
    "seed": 20261122,
    "slow_thresholds": {
      "elapsed_ms": 250,
      "scanned_rows": 12000,
      "rule": "either"
    }
  },
  "migration": {
    "replay_count": 3,
    "reducer": "median",
    "trim_percent": 20,
    "min_evidence": 2,
    "value_tolerance_pct": 18
  },
  "sources": [
    {
      "id": "app-prod",
      "type": "app_log_summary",
      "format_version": 2,
      "sample_rows": 24,
      "sample_period_ms": 100,
      "elapsed_semantics": "p95",
      "scan_semantics": "estimated",
      "data_path": "sources/app-prod/records.sqlite3"
    }
  ]
}
```

来源类型固定为三类：`app_log_summary`、`db_exec_stats`、`slow_log_excerpt`，同时接受 `应用日志摘要`、`库执行统计`、`慢日志摘录` 别名。合法口径组合如下：

| 来源类型 | 耗时口径 | 扫描口径 |
|---|---|---|
| 应用日志摘要 | `p95`, `mean` | `estimated`, `sampled` |
| 库执行统计 | `p95`, `mean`, `p50` | `exact` |
| 慢日志摘录 | `max`, `p95`, `mean` | `exact`, `sampled` |

声明按固定字段顺序强校验：源 ID、类型、格式版本、样本行数、采样周期、耗时口径、扫描口径、数据路径。任何非法声明整体以 `E_SOURCE` 拒绝，并输出首个违规字段路径，例如 `sources[0].sample_rows`。布尔值不会被当作整数接受。源 ID 必须唯一，数据路径必须唯一、位于工作区内，且不得进入 `ledger/` 或 `snapshots/`。

## 多版本编解码与归一

每条记录存储在源自己的 SQLite `records` 表中，载荷是 UTF-8 规范 JSON：

```json
{"body":{"query_id":"q001","latency_p95_ms":120},"length":44,"version":2}
```

`length` 是 `body` 规范 JSON 的 UTF-8 字节数。解码依次检查 UTF-8、JSON、信封字段、整数版本、已知版本、声明版本上限、长度自洽、必需字段、字段冲突、查询标识和非负整数指标。错误以 `E_FORMAT` 报告 `source_id`、`row_no` 与字段路径。

三种来源都实现 v1/v2 两套字段集。历史 v1 记录缺少 `sample_period_ms` 时，归一化表示会在 `padded_fields` 中显式记录该补齐字段；v2 记录自带周期并可校验其与声明一致。归一后的每条证据包含来源标识、来源类型、记录版本、声明版本、原始字段集合、补齐字段集合、原始上报值、归一值、口径和采样周期。

口径到规范值的校准因子（上报值 = 规范值 × 因子）：

| 耗时口径 | 因子 | 扫描口径 | 因子 |
|---|---:|---|---:|
| p95 | 1.00 | exact | 1.00 |
| mean | 1.30 | estimated | 1.05 |
| p50 | 1.75 | sampled | 1.20 |
| max | 0.90 | | |

解码时用整数基点运算反向归一，避免浮点累加差异。

## 证据合并仲裁

每个来源先按声明的 `replay_count` 选取稳定复核样本，再用 `median` 或 `trimmed_mean` 归约为每个查询、每个指标的一条来源候选证据。仲裁键固定为：

1. 来源类型偏置降序：库执行统计 `300` > 慢日志摘录 `200` > 应用日志摘要 `100`。
2. 来源声明格式版本降序。
3. 源标识升序。

每条查询的耗时和扫描行分别仲裁。采纳项输出源标识、值、格式版本集合、记录版本集合、复核行号、口径与周期；被否决项保留在 `comparisons` 中，并记录 `lower_source_type_bias`、`lower_format_version` 或 `source_id_order` 否决理由。所有候选先排序再判定，因此同一证据集合无论输入加载顺序如何，规范 JSON 输出逐字节一致。

慢查询诊断阈值为耗时 `>= 250 ms` 或扫描行 `>= 12000`（`rule=either`），也可配置为 `both`。诊断结果会逐条给出 `slow`、指标状态与比较依据。

## 跨源迁移复核

默认对四个源生成全部有向源对，也可在 `migration.pairs` 中显式声明。对源 A 的每条慢查询结论，系统：

1. 在 A、B 各自的全部样本上按 `replay_count` 与 reducer 得到稳定值。
2. 用 B 的归一值和阈值重新判定。
3. 计算每个指标的绝对偏差和百分比偏差基点：`abs(a-b) / max(a,b)`。

显式判定规则：

- `insufficient`：A 或 B 任一指标的唯一证据行数少于 `min_evidence`。
- `not_migratable`：慢查询状态不同，或任一指标偏差超过 `value_tolerance_pct`。
- `migratable`：慢查询状态相同，且全部指标偏差不超过容差。

输出包含源 A 结论、源 B 重判结论、状态、冲突指标、偏差量、容差和判定依据。默认使用 3 次复核和中位数；若改为 `trimmed_mean`，`trim_percent=20` 表示两侧各截去 20% 后取整数均值。

## 冲突归因

对每个 `not_migratable` 查询，系统把每个冲突指标分解为三类结构性贡献：

- `semantics`：A/B 校准因子差。
- `format_version`：记录版本均值差与补齐字段差。
- `sample_period`：采样周期相对差。

权重归一为 10000 个基点，并用最大余数法保证三类份额精确相加为 10000。贡献以 `delta × share` 的千分之一单位输出。指标冲突按 `conflict_score_milli_units` 降序、指标名升序稳定排序；同类因子的输出顺序始终为 `semantics`、`format_version`、`sample_period`。归因只依赖冻结的迁移结果与有序集合，不依赖字典/哈希遍历顺序。

## 失效传播与整体替换

账本保存上一方案版本的源参数指纹与策略摘要。任一源的 `sample_rows`、`sample_period_ms`、`format_version`、耗时/扫描口径，或全局诊断/迁移策略改变时，方案版本号加 1。失效范围沿“查询 → 指标 → 证据引用 → 迁移源对 → 归因结果”推导：

- 受影响查询写入 `affected_queries` 与 `stale_scope.merged_queries`。
- 实际引用改动源的迁移结果写入 `stale_scope.migration_results`。
- 实际引用改动源证据的归因结果写入 `stale_scope.attribution_results`。
- 未受影响查询写入逐项 SHA-256 校验；结论必须与上一版本完全相同。

当前结论通过同一个 `current_view` 单行原子替换，并同步删除/重建证据、合并、迁移、归因与失效标记表，不做增量拼接。账本保留 `invalidation_history` 供审计。

修改声明后需重新执行样本生成，再运行诊断。例如：

```bash
python3 -m slowdiag generate --workspace workbench
python3 -m slowdiag run --workspace workbench
```

## 快照与重放

快照格式 `slowdiag.snapshot`、整数 `format_version=1`，包含：

- 快照内容摘要 `content_sha256`；
- 方案版本与完整源声明；
- 各源内容摘要、声明格式版本与记录版本集合；
- 合并结论、迁移结论、归因结论、失效轨迹。

恢复时先验证快照摘要、格式版本、源声明、源内容摘要与格式版本集合，稳定排序差异项；当前视图整体替换。随后按快照记录的方案版本从本地源 SQLite 重新合并与迁移，按“查询 ID 升序 → 指标固定顺序 → 字段固定顺序”定位首个不一致，并以 `E_REPLAY` 输出 `query_id`、`metric`、`field`、期望值与实际值。即使重放失败，账本中也已经完成当前视图的整体替换。

```bash
python3 -m slowdiag snapshot --workspace workbench --name session-a.json
python3 -m slowdiag restore --workspace workbench --snapshot workbench/snapshots/session-a.json
```

## 异常注入与业务码

注入只改输入文件或命令行参数，不修改源码：

```bash
python3 -m slowdiag run --workspace workbench --inject source   # E_SOURCE
python3 -m slowdiag run --workspace workbench --inject format   # E_FORMAT
python3 -m slowdiag run --workspace workbench --inject empty    # E_EMPTY
```

- `source`：把源声明副本的首个源类型改为不支持值，输出 `sources[0].type`。
- `format`：复制首个源 SQLite，并在首行注入不属于该版本的字段，输出源、行号与字段路径。
- `empty`：在证据合并后以参数清空查询集合，输出 `E_EMPTY`。

所有命令都追加写入 `workbench/logs/slowdiag.jsonl`，每条日志含 UTC 时间、单调时钟纳秒、`run_id`、事件、严重级别、业务码和详情。

## 常用命令

```bash
python3 -m slowdiag init     --workspace workbench
python3 -m slowdiag generate --workspace workbench --check-repeat
python3 -m slowdiag run      --workspace workbench --pretty
python3 -m slowdiag show     --workspace workbench --full
python3 -m slowdiag snapshot --workspace workbench
python3 -m slowdiag restore  --workspace workbench --snapshot <path>
python3 -m slowdiag events   --workspace workbench --limit 20
python3 -m slowdiag gui      --workspace workbench
```

Tk 工作台提供观测源、合并结论、迁移复核、冲突归因和失效传播五个视图；可一键初始化、生成样本、运行诊断、创建快照、恢复最新快照，并在下方查看选中记录的结构化详情。

## 实际验证结果

已验证环境：Python `3.11.0rc1`、SQLite `3.37.2`、tkinter `8.6`、NumPy `1.26.4`，工作区可写。

`python3 -m unittest discover -s tests -v`：**24 项全部通过**。

`python3 scripts/verify_acceptance.py`：**全部通过**，结果保存在 `verification/acceptance.json`。关键实跑结果：

- 样本重复生成摘要一致；四个源摘要分别稳定为 `1e94ff…`、`fe5133…`、`6805a8…`、`918965…`。
- 合并后慢查询为 `q002`、`q003`、`q005`、`q006`。
- 迁移判定计数：可迁移 `36`、不可迁移 `12`、证据不足 `48`。
- 归因：不可迁移查询 `12` 组，指标冲突 `22` 个，三类贡献份额均精确合计 10000 基点。
- 修改 `slow-canary.sample_period_ms: 1000 → 1200` 后方案版本 `1 → 2`；`q001–q006` 失效，`q007/q008` 前后 SHA-256 完全一致。
- 快照即时恢复差异 `0`，恢复后视图摘要 `cd4a4bd0bb8870f722416a4077a85bda97a1f01fa4eefdf4e072c8c167a115ea`。
- 篡改 `q003.scanned_rows.adopted.value` 后恢复以 `E_REPLAY` 定位到 `query_id=q003`、`metric=scanned_rows`、`field=adopted.value`。
- 三类注入实跑分别得到 `E_SOURCE`（`sources[0].type`）、`E_FORMAT`（`app-prod` 第 1 行）、`E_EMPTY`。

工作区中已保留实际产物：`workbench/reports/latest.json`、`workbench/snapshots/demo-session.json`、`workbench/ledger/ledger.sqlite3`、`workbench/logs/slowdiag.jsonl`，以及 `verification/` 下的验收输出。

## 设计说明

更详细的模块职责、账本表结构、稳定排序不变量、恢复比较顺序和验收映射见 `docs/design.md`。

本系统只处理本地生成数据，不包含 Docker、容器运行时、网络访问或真实数据库连接。
