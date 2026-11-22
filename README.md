# 云原生多租户资源配额抢占与回收诊断工作台

这是一个完全本地运行的 Windows 诊断工作台，使用 Python 3.11+、Tkinter 与 SQLite 标准库实现。它从本地 JSON/NDJSON 读取租户、工作负载和资源压力事件，执行整数定点份额分摊、确定性逐副本抢占、收敛终止、事务账本、强制终止恢复、快照整体替换与逐字节重放校验。

项目不依赖网络、Docker、容器、第三方 Python 包或外部服务。所有样本和异常注入均在仓库内本地生成。

## 目录

```text
app.py                         CLI/Tkinter 入口
demo.py                        三场景可重复演示入口
run_demo.bat                   Windows 演示脚本
run_ui.bat                     Windows 诊断界面脚本
quota_workbench/
  allocation.py                整数定点分摊与中间项
  canonical.py                 规范 JSON 与 SHA-256
  cli.py                       命令行、异常注入、恢复、重放
  demo.py                      三场景演示实现
  engine.py                    压力、收敛、逐副本抢占
  errors.py                    业务码与退出码
  ledger.py                    SQLite 事务账本与恢复标记
  session.py                   会话、崩溃恢复、重放定位
  snapshot.py                  只读快照与整体替换
  spec.py                      JSON/NDJSON 强校验
  ui.py                        Tkinter 诊断界面
samples/
  spec.json                    确定性租户/工作负载声明
  spec_context_changed.json    用于演示快照口径变化与整体替换
  normal.ndjson                常规抢占事件流
  unsat.ndjson                 缺口不可满足事件流
  replay.ndjson                快照恢复后继续推进的事件流
  injections.json              E_QUOTA/E_RESERVE/E_SNAPSHOT 注入清单
tests/                         35 项标准库 unittest 回归用例
demo_output/                   demo.py 生成的账本、快照、日志和稳定结果
```

## Windows 环境准备

```bat
py -3.11 -m venv .venv
.venv\Scripts\activate
set PYTHONUTF8=1
python -c "import tkinter, sqlite3; print(tkinter.TkVersion, sqlite3.sqlite_version)"
```

若 `tkinter` 导入失败，请在 Windows 安装器中勾选 “tcl/tk and IDLE”，或使用官方 python.org 的完整 Python 3.11+ 安装包。随后可直接运行：

```bat
run_demo.bat
run_ui.bat
```

也可以手动执行：

```bat
python demo.py
python app.py --ui --spec samples/spec.json --events samples/normal.ndjson --db demo_output/ui.sqlite3
python -m unittest discover -v
```

## 输入契约

### 声明 JSON

```json
{
  "cluster_total_units": 120,
  "tenants": [
    {
      "tenant_id": "tenant-alpha",
      "quota_bp": 5000,
      "reserve_units": 10,
      "priority": 10
    }
  ],
  "workloads": [
    {
      "workload_id": "wl-alpha-batch",
      "tenant_id": "tenant-alpha",
      "requested_replicas": 2,
      "units_per_replica": 7,
      "preemptible": true,
      "startup_order": 20
    }
  ]
}
```

强校验规则：

- `cluster_total_units` 为正整数。
- 租户标识唯一且非空；`quota_bp` 为正整数；全部租户 `quota_bp` 之和不超过 `10000`；`reserve_units` 为非负整数；`priority` 为整数。
- 保留配额总和超过集群总资源时整体拒绝，返回 `E_RESERVE`。
- 工作负载标识唯一；所属租户必须存在；请求副本数与每副本资源单位为正整数；`preemptible` 必须是 JSON 布尔值；`startup_order` 为整数且全局唯一。
- 非配额/保留专项错误以 `E_SPEC` 整体拒绝，并报告首个违规对象、标识和字段。配额和保留专项使用更明确的 `E_QUOTA`、`E_RESERVE`。

### 抢占触发 NDJSON

每行一个 JSON 对象，事件号从 `1` 开始连续递增，`time_ms` 非负且单调不减：

```json
{"event_id":1,"time_ms":1000,"type":"TENANT_PRESSURE","target_tenant":"tenant-alpha","pressure_units":35}
{"event_id":2,"time_ms":2000,"type":"WORKLOAD_PRESSURE","target_workload":"wl-beta-batch-b","pressure_units":15}
```

支持类型：

- `TENANT_PRESSURE`：要求 `target_tenant`，压力计入该租户的持续压力桶。
- `WORKLOAD_PRESSURE`：要求 `target_workload`，压力计入该工作负载所属租户。

压力值是会话内持续累加的外部占用压力，不会在下一事件自动释放。初始时所有工作负载均按请求副本数运行；抢占会减少运行副本数，后续事件继承该状态。该语义使每个事件后的运行状态、分摊矩阵和账本字节都可独立复核。

## 确定性分摊算法

所有计算均为整数，不使用浮点数。设集群总资源为 `C`、租户份额为 `q_i`、保留配额为 `r_i`。

1. `floor_i = C * q_i // 10000`，向下截断。
2. `shared_budget = C * sum(q_i) // 10000`。
3. 将 `shared_budget - sum(floor_i)` 个单位按最大余数分配；余数为 `C * q_i % 10000`。并列时按 `quota_bp` 降序，再按 `tenant_id` 字典序升序。
4. 令份额应得为 `share_i`。先提升到 `max(share_i, r_i)`。若提升后总和超过 `C`，只从高于保留配额的可缩减部分中回收溢出；缩减量使用同样的最大余数方法，保证每租户最终 `>= r_i` 且总和 `<= C`。
5. 若 `sum(q_i) < 10000`，未声明的份额形成 `unassigned_units`，不会强行分给租户。

矩阵字段包括：份额下限、余数分子、余数补位、份额应得、保留抬升、容量缩减、最终应得、1e-4 精度量化误差、工作负载占用、持续压力占用、实际占用、超额量、分摊偏差、剩余保留空间。

### 超额与逐副本抢占

存在以下任一条件时继续收敛：集群实际占用超过 `C`，或某租户实际占用超过其最终应得。

候选选择规则完全显式且与输入数组顺序无关：

1. 只考虑当前存在超额的租户；租户按 `priority` 升序，若相同按 `tenant_id` 升序。
2. 只考虑 `preemptible=true` 且仍有运行副本的工作负载；不可抢占工作负载不会被回收。
3. 同一租户内按 `startup_order` 降序，再按 `workload_id` 升序。
4. 每轮只抢占一个副本。
5. 若 `租户当前占用 - 每副本单位 < reserve_units`，禁止抢占，避免跨入保留配额。

每次抢占记录：轮次、租户、工作负载、每副本单位、回收量、租户抢占前后占用、集群抢占前后占用、抢占前后矩阵摘要和选择依据。每轮均保存完整矩阵快照。

终止原因：

- `NO_EXCESS`：全部租户均不超过应得，且集群不超总资源。
- `NO_PREEMPTIBLE_REPLICA`：仍有缺口，但没有可安全抢占的副本，事件结果为 `E_UNSATISFIED`。

`E_UNSATISFIED` 结果仍会完整提交到账本，并包含 `remaining_gap_units`、涉及租户、阻塞工作负载、不可抢占原因和被拒绝的抢占请求。CLI 对该结果返回退出码 `13`。

## SQLite 事务账本与强制终止恢复

每次处理事件前先提交一个独立的 `event_attempts` 意图记录，状态为 `IN_PROGRESS`。事件的分摊矩阵、逐轮迭代、逐次抢占、回收量、终止原因、运行状态和进度游标在同一个 `BEGIN IMMEDIATE` 事务内写入。`ledger_info.format_version` 是 SQLite 整数列，当前值为 `1`。

账本完整性：

- 每个事件有规范 JSON 字节、`record_digest = SHA-256(record_json)`。
- 事件链摘要为 `SHA-256(previous_head + ":" + record_digest)`。
- `meta.head_digest` 保存账本头摘要。
- 重放校验同时比较规范记录字节和规范化矩阵/抢占表。

若进程在收敛中被强制终止且结果事务未提交，重启只能恢复已完整提交的事件；未完成尝试会被标记为 `UNKNOWN_REQUIRES_RERUN`。工作台明确要求重跑该事件，不会伪造终止时刻的退出报告。已提交事件不受影响。

## 快照与整体替换

快照保存：

- 格式版本、租户集合版本、工作负载声明版本、份额口径版本、集群总资源。
- 完整声明、全部事件记录、全部迭代分摊矩阵、抢占记录、运行状态、账本游标/头摘要。
- 规范化载荷的 `content_digest`。

快照采用写一次语义：目标已存在即返回 `E_SNAPSHOT`；成功写入后设置为只读。恢复目标必须是全新数据库，快照中的声明和状态整体替换当前上下文，绝不增量合并。

加载时验证内容摘要、声明版本、事件记录摘要、链摘要、运行状态和最终矩阵。与当前声明比较后稳定输出：

- `mode: WHOLE_REPLACE`
- `changed_fields`
- `affected_tenant_ids`
- 快照版本与当前版本

内容摘要不符返回 `E_SNAPSHOT`。声明版本/总资源/事件流不同导致无法复现时使用 `E_CONTEXT` 并标记 `unreproducible=true`，不会混同为 `E_REPLAY`。

## 重放校验

`verify_replay()` 从初始运行状态开始，按事件游标重新执行已提交事件，并逐项比较：

- 事件后的完整分摊矩阵与每个矩阵的摘要/规范化字段。
- 被抢占副本集合、每副本回收量、抢占前后占用和选择依据。
- 每轮迭代轨迹、剩余缺口和终止原因。
- 运行副本状态与压力桶。

第一处差异会报告事件号、租户标识、字段、期望值、实际值和依据，然后抛出 `E_REPLAY`。

从快照恢复后，可继续推进剩余事件；演示会与从初始声明全量运行的参考会话比较账本头摘要、状态摘要和逐事件记录摘要，三项均一致。

## Tkinter 诊断界面

```bat
python app.py --ui --spec samples/spec.json --events samples/normal.ndjson --db demo_output/ui.sqlite3
```

界面包含六个稳定排序页签：

- 租户分摊：份额下限/补位/应得、保留抬升、最终应得、工作负载与压力占用、超额量、偏差、量化误差。
- 本轮抢占：副本、回收量、租户/集群前后占用、选择依据。
- 收敛轨迹：每轮矩阵摘要、集群/租户超额、剩余缺口、涉及租户、终止原因。
- 工作负载状态：请求/运行副本与占用。
- 账本事件：事件号、时刻、目标、压力、结论、轮次、回收量和记录摘要。
- 声明与恢复：版本、恢复状态、账本头、运行状态和完整账本事件。

「运行下一事件」「运行全部」「刷新」「重放校验」「保存快照」「恢复快照」均为本地操作。刷新仅查询账本，不修改账本内容。

## CLI 示例

常规抢占：

```bat
python app.py --spec samples/spec.json --events samples/normal.ndjson --db demo_output/normal.sqlite3 --run-all --verify-replay
```

缺口不可满足（预期退出码 13）：

```bat
python app.py --spec samples/spec.json --events samples/unsat.ndjson --db demo_output/unsat.sqlite3 --run-all
```

三类异常注入：

```bat
python app.py --inject quota-overflow
python app.py --inject reserve-overflow
python app.py --snapshot demo_output/session.cursor-2.snapshot.json --inject snapshot-digest-mismatch
```

分别返回 `E_QUOTA`、`E_RESERVE`、`E_SNAPSHOT`，并追加规范 JSON 到 `demo_output/structured.ndjson`（或 `--log` 指定文件）。

保存快照、整体恢复并继续重放：

```bat
python app.py --events samples/replay.ndjson --db demo_output/source.sqlite3 --max-events 2 --save-snapshot demo_output/session.snapshot.json
python app.py --events samples/replay.ndjson --restore-snapshot demo_output/session.snapshot.json --restore-db demo_output/restored.sqlite3 --current-spec samples/spec_context_changed.json --run-all --verify-replay
```

## 可重复演示

```bat
python demo.py
```

脚本按顺序执行：

1. 常规抢占：4 个事件、4 次副本抢占、回收 25 单位，最终 `NO_EXCESS`。
2. 缺口不可满足：`tenant-gamma` 剩余缺口 12 单位，`wl-gamma-core` 因不可抢占被拒绝，返回 `E_UNSATISFIED`。
3. 快照保存与恢复重放：游标 2 保存只读快照；用变化后的声明触发整体替换报告；恢复后继续事件 3/4；与全量参考运行比较账本头、状态和逐事件摘要。
4. 额外异常自检：验证 `E_QUOTA`、`E_RESERVE`、`E_SNAPSHOT` 与三条结构化日志。

结果写入 `demo_output/results.json`，同时输出稳定排序的规范 JSON。演示测试连续运行两次，`results.json` 字节完全一致。

## 本次实际验证结果

环境：Python `3.11.0rc1`，SQLite 标准库可用，Linux 反馈容器。

```text
python -m unittest discover -v
Ran 35 tests in 3.594s
OK
```

`python demo.py` 实测：

```text
status                 PASS
summary_digest         ac6b085b56b237f15e3b3f56641740ea38550f5a17ce9e5445c6fd6cecb23ea2
normal preemptions     4
normal reclaimed       25
normal ledger head     9771d3751362093d64eecc8c3194482752c0c279b575399e41378b4ea0399b9a
unsat code             E_UNSATISFIED
unsat gap              12 units / tenant-gamma
replay head equal      true
replay state equal     true
record digests equal   true
anomaly codes          E_QUOTA, E_RESERVE, E_SNAPSHOT
```

强制终止测试通过子进程在事件 1 的意图写入后执行 `os._exit(137)`：重启游标仍为 0，`event_attempts` 报告 `UNKNOWN_REQUIRES_RERUN` 和事件 1，未声称存在退出报告；随后重跑成功提交。

当前反馈容器缺少 `libtk8.6.so`，因此无法在此 Linux 环境真正打开 Tk 窗口。`quota_workbench.ui` 采用惰性 Tk 导入，已通过模块导入和 `py_compile` 验证；在无 Tk 环境中执行 `--ui` 会稳定返回 `E_IO`/退出码 `18`，不会输出堆栈。窗口应在目标 Windows Python 3.11+ 环境中通过 `run_ui.bat` 进行最终显示验收。此限制不影响核心引擎、SQLite、快照、重放和 CLI 验证。

## 业务码

| 业务码 | 含义 | CLI 退出码 |
|---|---|---:|
| `E_SPEC` | 首个通用声明/事件格式违规 | 10 |
| `E_QUOTA` | 份额万分比总和超过 10000 | 11 |
| `E_RESERVE` | 保留配额总和超过集群总资源 | 12 |
| `E_UNSATISFIED` | 无可抢占副本，仍有缺口 | 13 |
| `E_SNAPSHOT` | 快照摘要/版本/目标冲突 | 14 |
| `E_REPLAY` | 账本重放内容不一致 | 15 |
| `E_CONTEXT` | 声明、总资源或事件流口径变化，不可复现 | 16 |
| `E_LEDGER` | 账本结构/事务/版本错误 | 17 |
| `E_IO` | 本地文件读写错误 | 18 |
| `E_INTERNAL` | 未预期中断 | 19 |
