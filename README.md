# MVCC 可见性诊断工作台

面向 Windows 本地 Python 环境的 MVCC 快照可见性诊断工具。输入是本地 JSON 或 NDJSON
规格，输出是可复核的逐查询判定账本、理由链摘要、算法选择依据、双配额线程池调度轨迹、
SQLite 原子轮次、版本化会话快照，以及 Tkinter 可视化诊断界面。

本项目不依赖 Docker、数据库服务、网络或外部账号。

## 快速启动（Windows）

建议使用 Python 3.11/3.12 官方安装包，并保留安装包默认勾选的 `tcl/tk`。

```powershell
py -3.11 -m pip install -r requirements.txt
py -3.11 -c "import tkinter, numpy, sqlite3; print(tkinter.TkVersion, numpy.__version__, sqlite3.sqlite_version)"
py -3.11 -m mvcc_workbench init
py -3.11 -m mvcc_workbench validate --spec data/demo_large.json
py -3.11 -m mvcc_workbench diagnose --spec data/demo_large.json --ledger data/ledger.sqlite3 --report-out artifacts/diagnostic.json
py -3.11 -m mvcc_workbench gui --spec data/demo_large.json --ledger data/ledger.sqlite3
```

也可直接双击或运行：

```powershell
run_gui.bat
run_tests.bat
```

## 目录

```text
mvcc_workbench/       领域模型、校验、算法、调度、账本、快照、CLI 与 Tk 界面
data/                 确定性 JSON、乱序 JSON、NDJSON 样本
tests/                unittest 自动化测试
scripts/              run_acceptance.py 独立验收脚本
docs/                 技术设计与实际验证记录
artifacts/            诊断结果、账本、快照、异常样本与界面截图
```

## 输入规格

完整规格示例如 `data/demo_large.json`。顶层字段固定为：

```json
{
  "format": "mvcc-diagnostic-spec",
  "format_version": 1,
  "name": "sample",
  "transactions": [],
  "records": [],
  "read_points": [],
  "queries": [],
  "budget": {"global_concurrency": 4, "per_read_point_concurrency": 2}
}
```

事务、版本、读点与查询字段：

```json
{"id":"T1","start_seq":10,"outcome":"committed","commit_seq":20}
{"row_key":"row:001","tx_id":"T1","version":1,"value_digest":"sha256:...","prev_version":null}
{"id":"RP1","snapshot_seq":15,"tx_id":"T1"}
{"id":"Q1","read_point_id":"RP1","row_key":"row:001","predicate":{"op":"exists"}}
```

外部规格只接受 `committed` 与 `rolledback` 两种结局。`pending` 仅存在于失效演进的内部
状态模型中，用于验证“未决变为提交/回滚”，不会被本地规格加载器接受。

支持的谓词：`exists`、`version_eq`、`digest_eq`、`tx_eq`。谓词在可见版本确定后求值，
不会绕过 MVCC 可见性规则。

NDJSON 每行一个对象，`kind` 为 `meta`、`transaction`、`record`、`read_point` 或
`query`；`data/demo_large.ndjson` 可直接加载。

## 强校验与业务错误

加载器会稳定排序后再检查，因此数组排列不影响首错结论或最终摘要。主要校验包括：

- 事务标识唯一；开始/提交序号为非负整数；结局合法；回滚事务不得携带非空提交序号；
  已提交事务的提交序号不得早于开始序号。
- 版本记录事务存在，行键内版本号唯一；`prev_version` 必须引用同一行键中存在的版本；
  版本引用无环。
- 读点标识唯一，快照序号非负，读事务存在。
- 查询标识唯一，读点和行键存在，谓词结构合法。
- 未知字段和 NaN/Infinity 也会被拒绝。

非法规格返回 `E_SPEC`，并尽量在 `details.entity_id` 中给出首个非法事务标识或行键；
版本链成环单独返回 `E_CYCLE`。

## 可见性规则

版本链从最大版本号链头开始，沿 `prev_version` 回溯；不在链头路径中的碎片按版本号稳定
补齐，保证确定顺序。对每个候选版本：

1. 读事务自身尚未提交的 `pending` 版本对自身可见，依据记为 `SELF_UNCOMMITTED`。
2. 其他未决事务版本被排除，理由为 `WRITER_PENDING_NOT_SELF`。
3. 已回滚事务版本被排除，理由为 `WRITER_ROLLED_BACK`。
4. 已提交但 `commit_seq > snapshot_seq` 的版本被排除，理由为
   `COMMIT_AFTER_SNAPSHOT`。
5. 只有已提交且 `commit_seq <= snapshot_seq` 的版本可见，依据记为
   `COMMITTED_AT_OR_BEFORE_SNAPSHOT`。

每条查询记录可见版本、回退步数、检查版本数、逐步排除理由、谓词结果、终止码与
`reason_chain_digest`。摘要使用按键排序、无多余空白的 UTF-8 规范 JSON，输入数组任意
重排会得到逐字节一致的结论与理由摘要。

## 双路径与阈值选路

两条算法路径：

- `per_query`：每条查询独立沿版本链回溯。
- `materialized`：先按 `(read_point, row_key)` 只物化一次可见版本，再由查询复用。

规划器显式阈值：

| 参数 | 阈值 |
|---|---:|
| 查询数 `Q` | 32 |
| 行键数 `R` | 12 |
| 平均链长 `L` | 5.0 |

至少 2 个阈值跨界时，规模先验路径为 `materialized`，否则为 `per_query`。工作量估计为：

```text
per_query_work    = Q * (L + 1.0) + 0.5 * Q
materialized_work = U * (L + 2.35) + 0.20 * Q
```

其中 `U` 是唯一 `(read_point, row_key)` 组合数。相对差不超过 10% 时进入交界带，采用
`threshold-majority-v1` 稳定决胜，并设置 `threshold_influenced=true`。计划输出同时包含
实际规模、阈值、距离、未触发上界与两条路径估算值；可复核字段见
`artifacts/tie_band_plan.json`。

每次诊断默认同时执行两条路径并逐查询比较可见版本、排除理由序列与理由链摘要。首个不一致
以 `E_DIVERGE` 返回查询标识、行键和字段，不依赖字典或哈希遍历顺序。

## 并行调度

任务真实提交到 `ThreadPoolExecutor`。中央派发器同时维护全局活动数和每读点活动数，只有
两类配额都允许时才提交任务；任务完成后再补齐。生产路径没有 `sleep` 或伪超时。

调度轨迹记录：

- `queued_at_ms`、`quota_wait_start_ms`、`quota_wait_end_ms`、`quota_wait_ms`
- 工作线程内的 `started_at_ms`、`finished_at_ms`、`thread_name`、`thread_id`
- 阻塞来源 `blocked_by_global` / `blocked_by_read_point`
- 实测全局与每读点并发峰值、最大等待、线程数、约束是否满足
- 每个读点实际分配任务数与被配额阻塞的任务数

短小判定任务的实际峰值可能低于上限；`tests/test_scheduler.py` 使用受控阻塞任务验证
全局 3、每读点 2 的硬上限和真实多线程并发。

## 账本、崩溃恢复与快照

每轮先以独立事务写入 `pending_rounds(status=building)`，然后在一个 SQLite 事务中写入：
逐查询结论、理由链摘要、调度轨迹、预算分配、进度游标和内容摘要。格式版本为整数 `1`。

进程强制终止后：

- 只有完整提交的事务可被恢复；
- 残留 `building` 轮次不会产生半成品结论；
- 其查询被标记为 `UNKNOWN_RERUN_REQUIRED` 并要求重跑；
- 最近完整轮次仍可正常读取并校验内容摘要。

会话快照保存事务集合版本、版本链摘要、预算版本、读点集合版本、结论摘要、整数账本游标、
完整规格与结论，并携带 `content_digest`。恢复时逐项比较版本；不一致返回 `E_SNAPSHOT`
并要求 `REPLACE_ALL` 整体替换，绝不增量合并。替换后从快照规格重放全部判定并核对结论
摘要。

## 失效传播

事务结局变化按“事务 → 版本记录 → 读点 → 查询”推导；追加版本按“行键 → 查询”推导。
受影响的全部查询整体标记为 `STALE`，证据按 `query_id, cause_type, cause_id, row_key`
稳定排序。重算后整体替换受影响结论，并逐字节核验未受影响结论保持不变。

## 常用 CLI

```powershell
python -m mvcc_workbench init --data-dir data
python -m mvcc_workbench validate --spec data/demo_large.json
python -m mvcc_workbench diagnose --spec data/demo_large.json --ledger data/ledger.sqlite3 --snapshot-out data/session_snapshot.json --report-out artifacts/diagnostic.json
python -m mvcc_workbench consistency --spec data/demo_large.json --compare-spec data/demo_large_shuffled.json
python -m mvcc_workbench recover --ledger data/ledger.sqlite3 --report-out artifacts/recovery.json
python -m mvcc_workbench snapshot-replay --snapshot data/session_snapshot.json
python -m mvcc_workbench evolution --spec data/demo_small.json --append-row row:001:T0001:3:2 --report-out artifacts/invalidation.json
python -m mvcc_workbench inject-anomaly --kind cycle --spec data/demo_small.json --out artifacts/cycle.json
python -m mvcc_workbench inject-anomaly --kind snapshot --spec data/demo_small.json --out artifacts/bad_snapshot.json
python -m mvcc_workbench gui --spec data/demo_large.json --ledger data/ledger.sqlite3
```

开发验证可运行：

```powershell
python -m unittest discover -v
python scripts/run_acceptance.py
```

## Tk 诊断界面

界面包含五个页签：诊断概览、版本链结构、读点与判定理由、调度时间线与预算、快照/账本。
界面上的按钮支持加载、运行诊断、保存快照、快照恢复/重放、账本恢复，以及注入
`E_CYCLE` 与 `E_SNAPSHOT` 两类异常。

界面截图：

- `artifacts/gui.png`：选路、阈值边界与一致性概览
- `artifacts/gui_chains.png`：版本链树与单版本详情
- `artifacts/gui_reasons.png`：读点快照线、查询列表与逐步理由链
- `artifacts/gui_schedule.png`：双配额预算与真实调度时间线
- `artifacts/gui_state.png`：版本识别、内容摘要、账本与一致性状态

## 业务码

| 业务码 | 含义 |
|---|---|
| `OK` | 校验、诊断、两路径一致性或重放成功 |
| `E_SPEC` | 规格非法，含首个非法事务/行键定位 |
| `E_CYCLE` | 版本引用成环 |
| `E_DIVERGE` | 两路径结论不一致，含首个查询/行键/字段 |
| `E_SNAPSHOT` | 快照摘要或版本不一致 |
| `E_LEDGER` | 账本版本或内容摘要不一致 |
| `E_SCHEDULER` | 调度约束或任务执行失败 |
| `E_ARGUMENT` | CLI 参数组合非法 |
| `E_INTERNAL` | 异常注入未按预期触发或未分类内部错误 |
| `E_GUI` | Tk 窗口不可用 |
| `UNKNOWN_RERUN_REQUIRED` | 崩溃前未提交轮次中的查询，必须重跑 |
| `OK_RERUN_REQUIRED` | 已恢复最近完整轮次，但仍有未知轮次需重跑 |

实际验证摘要、摘要值和并发探针结果见 `docs/VALIDATION.md`；设计与实现细节见
`docs/TECHNICAL.md`。

## 运行说明

当前判定内核只使用 Python 标准库；`numpy>=1.26,<3` 按任务环境要求安装并验证可导入。
Windows 官方 Python 自带 Tk；无显示服务器的 CI 环境不能启动 GUI，但所有领域、算法、
调度、账本、快照与 CLI 测试均可无头运行。
