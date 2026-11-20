# SQLite 事务隔离与锁等待实验台

本项目是一个本地、确定性、零第三方依赖的事务隔离实验台。它使用 Python 3.11+ 标准库
`sqlite3` 保存行数据、提交版本、事件账本与快照，在应用层实现行级共享锁/排他锁、FIFO
等待队列、锁等待预算、死锁仲裁、隔离级别语义、快照整体替换和重放校验。Tkinter 界面和
命令行脚本共享同一套核心引擎。

## 快速开始

Windows PowerShell 中可将 `python3` 替换为 `py -3`。

```bash
python3 --version                 # 需要 3.11+
python3 -m isolation_lab.cli validate --script samples/current-v3.json
python3 -m isolation_lab.cli run \
  --script samples/current-v3.json \
  --db runtime/lab.sqlite \
  --snapshots runtime/snapshots.sqlite \
  --snapshot-id normal
python3 -m isolation_lab.cli snapshot-replay \
  --snapshots runtime/snapshots.sqlite --snapshot-id normal
python3 -m isolation_lab.cli ledger --db runtime/lab.sqlite
```

四个场景的确定性演示：

```bash
python3 -m isolation_lab.cli demo --workdir runtime/demo
# 或
python3 run_demo.py --workdir runtime/demo
```

报告写入 `runtime/demo/demo_report.txt`，同时打印到标准输出。相同工作目录重复运行时，
报告除固定工作目录文本外保持一致。

Tkinter 工作台：

```bash
python3 -m isolation_lab.cli gui \
  --db runtime/lab.sqlite \
  --snapshots runtime/snapshots.sqlite
```

运行全部测试：

```bash
python3 -m unittest discover -s tests -v
```

## 样本

| 文件 | 版本 | 用途 |
|---|---:|---|
| `samples/current-v3.json` | 3 | 三个会话、全局并发上限 2 的正常并发编辑 |
| `samples/legacy-v1.json` | 1 | 仅会话标识和有序读写操作 |
| `samples/legacy-v2.json` | 2 | 隔离级别、锁等待预算及 RR 读/阻塞写 |
| `samples/lock-timeout-v3.json` | 3 | `W` 等待预算 3 ms，超时回滚 |
| `samples/deadlock-v3.json` | 3 | 相反加锁顺序的等待环，稳定牺牲 `D1` |

## 版本与强校验

- `format_version=1`：会话仅含 `session_id`、有序 `operations`，可选派生的 `lock_order`。
- `format_version=2`：每个会话增加 `isolation_level` 与整数 `lock_wait_budget_ms`。
- `format_version=3`：每个会话再增加 `concurrency` 与 `operation_timeout_ms`。
- 加载时先检查版本、字段集合与类型。未知版本、缺字段、类型不符、重复 `session_id`、
  重复/乱序 `op_no`、非法隔离级别都会以 `E_SCRIPT` 整体拒绝，并输出字段路径与会话标识。
- 低版本在内存中归一为 v3，顶层保留 `source_format_version` 与 `upgrade_changes`。
  `upgrade` 命令输出的文件可以再次被 `run`/`validate` 加载。
- 语义摘要 `script_digest` 不包含来源版本。相同脚本升级前后使用相同调度参数时，
  `run_id`、提交版本、逐会话结局、等待毫秒与最终行值一致。

默认值：

| 字段 | 默认 |
|---|---|
| `isolation_level` | `read_committed` |
| `lock_wait_budget_ms` | `1000` |
| `concurrency` | `1` |
| `operation_timeout_ms` | `1000` |

## 调度与锁语义

调度器采用确定性逻辑时钟（默认 1 ms/轮）推进会话。真实单调时钟毫秒也记录在每个操作和
账本事件中，但预算、超时、等待时长和重放比较使用逻辑时钟，因此同一脚本/参数可逐项复现。

- 会话按脚本顺序启动，同时处于运行或等待状态的会话数不超过 `global_concurrency_limit`。
- 每会话按 `op_no` 顺序执行；共享锁用于读，排他锁用于写。`concurrency` 是会话内允许的最大
  在途操作数，会话核心始终不超过该值。
- 锁兼容规则：`S+S` 兼容，`X` 与其他任何持有者冲突。请求不能绕过已有队列；等待队列按全局
  `request_seq` 稳定升序。每次授予、等待、升级、释放、取消都写入账本，包含持有者、队列、
  申请序号、授予序号、阻塞原因和隔离级别依据。
- `repeatable_read`：首次读取得事务启动快照并保留共享锁直到提交；同一行重复读得到相同值
  与版本，自己的待提交写优先于快照。
- `read_committed`：每次读取最新已提交值；读取期间的共享锁在操作结束后立即释放，即“短暂
  共享”。未提交写仍由排他锁隔离。
- 写操作先进入会话私有写集，只有提交时才写入 SQLite。提交按稳定顺序分配一次单调递增的
  `data_version`；同一会话多行写共享该版本。超时/死锁会话的写集全部丢弃。
- 单次等待上限为 `min(剩余锁预算, operation_timeout_ms)`。累计等待耗尽预算或单操作超时
  都产生 `E_LOCKTIMEOUT`，会话终止，后续操作不再执行。
- 等待图只包含当前持有者。发现环后按 `min(start_seq, session_id)` 选择牺牲者，产生
  `E_DEADLOCK`，释放其全部锁和队列项；循环仲裁直到无环。相同输入五次重复测试牺牲者一致。
- 最终行状态会按提交顺序重放校验；不一致产生 `E_ISOLATION`，包含会话、行键和两次值。
  RR 同会话重复读取的不一致也产生 `E_ISOLATION`。

## 存储、账本与快照

主数据库 `lab.sqlite` 包含：

- `rows(row_key, value_json, data_version)`：提交后的当前行状态。
- `commits(run_id, commit_seq, session_id, data_version, virtual_ms, writes_json)`：提交顺序。
- `ledger(...)`：跨版本事件账本，记录来源版本、归一版本、事件序号、虚拟/单调毫秒和 payload。
- `meta`：当前脚本摘要、参数摘要、会话与结局、全局数据版本等。

账本读取支持 `source_version=1/2/3`。同一数据库可依次运行 v1、v2、v3 脚本，读取时保留各自
来源版本并按 `event_id` 稳定排序。

快照保存到独立 SQLite 文件 `snapshots.sqlite`，payload 包含来源/归一版本、脚本、脚本摘要、
参数版本与摘要、锁表、等待队列、执行轨迹、逐会话结局、提交记录、最终行状态和全局版本。
恢复时比较当前脚本摘要/参数摘要，但无论是否相同都执行整体状态替换，而不是增量合并。报告
稳定列出变化会话、变化参数项、被清空锁条目数量、轨迹条数和恢复后的数据版本。

`snapshot-replay` 在全新内存 SQLite 中按快照记录的归一语义和参数重调度，比较：

1. 每个会话的结局、业务码、提交版本、等待预算、虚拟起止时间；
2. 每个操作的 `status/value/data_version/grant_seq/wait_ms/wait`；
3. 确定性轨迹（排除真实单调时钟）；
4. 最终行值与全局数据版本。

差异返回 `E_REPLAY`，并定位首个不一致的 `session_id`、`op_no` 和字段路径。测试通过篡改
快照中的 `N1` 首操作值验证了该定位。

## 业务错误码

| 业务码 | 触发条件 | 关键定位字段 |
|---|---|---|
| `E_SCRIPT` | 格式版本未知、缺字段、类型/范围/结构不合法、JSON 解析失败 | `path`、`session_id`、`op_no` |
| `E_LOCKTIMEOUT` | 累计锁预算耗尽或单操作等待超时 | `session_id`、`op_no`、等待请求序号 |
| `E_DEADLOCK` | 等待图成环并完成稳定仲裁 | 牺牲会话、环路径、仲裁规则 |
| `E_ISOLATION` | 最终状态异于提交顺序重放，或 RR 重复读不一致 | `session_id`、`row_key`、两次值 |
| `E_SNAPSHOT` | 快照不存在、payload 损坏或缺字段 | `snapshot_id`、payload 路径 |
| `E_REPLAY` | 快照与重新调度结果不一致 | 首个不一致 `session_id`、`op_no`、字段 |

## 命令行

| 命令 | 说明 |
|---|---|
| `validate --script FILE [--json]` | 强校验并展示归一结果 |
| `upgrade --script FILE [--output FILE]` | 输出可重新加载的 v3 归一脚本 |
| `run --script FILE --db DB [--snapshot-id ID]` | 调度脚本并可选保存快照 |
| `ledger --db DB [--run-id ID] [--json]` | 读取跨版本事件账本 |
| `snapshot-list --snapshots DB` | 列出快照 |
| `snapshot-replay --snapshots DB --snapshot-id ID` | 重放并检查不可复现原因 |
| `snapshot-restore --snapshots DB --snapshot-id ID --db DB` | 整体替换目标状态 |
| `equivalence --left A --right B` | 比较两脚本的最终行、版本、结局与等待 |
| `demo --workdir DIR` | 执行四场景演示 |
| `gui --db DB --snapshots DB` | 打开 Tkinter 工作台 |

全局调度参数通过 `--global-concurrency` 和 `--clock-quantum` 传入；当前参数结构版本为 3，
写入快照并与脚本摘要一起参与恢复比较。

## 界面

Tkinter 工作台提供：加载/校验脚本、执行、保存快照、恢复、重放、查看账本；会话表显示阶段、
隔离级别、已用/总等待预算、持有锁、等待对象、业务码与提交版本；锁表显示持有者、FIFO 队列、
申请与授予序号；事件区显示锁授予、等待、超时和死锁仲裁依据。执行在后台线程中进行，界面通过
事件队列实时刷新。

## 实际验证记录

本仓库在 Python 3.11.0rc1、SQLite 3.37.2 环境中完成验证：

```text
python3 -m unittest discover -s tests -v
Ran 28 tests
OK
```

覆盖内容包括：三版本加载与归一、未知版本/类型/缺字段的 `E_SCRIPT` 定位、锁兼容/FIFO/
升级、全局并发上限、预算与操作超时、稳定死锁牺牲、RR/RC 语义、v1/v2 升级等价、五样本
快照重放、锁表整体清空、篡改快照的 `E_REPLAY` 定位、跨版本账本、CLI 和可重复演示。

演示实际结果要点：

```text
[1] normal concurrent editing  data_version=3, N1/N2/N3 committed 1/2/3, replay=REPRODUCIBLE
[2] lock-wait timeout          H committed v1; W E_LOCKTIMEOUT wait=3ms; replay=REPRODUCIBLE
[3] deadlock arbitration       D1 E_DEADLOCK; D2 committed v1; replay=REPRODUCIBLE
[4] legacy v1/v2 upgrade       upgrade_equivalence=OK for both
```

当前 Linux 验证机未安装 `python3-tk`，因此 Tkinter 只完成语法编译与缺库错误路径测试；
Windows 官方 Python 默认包含 Tkinter，可直接运行 `gui`。若 Linux 需要界面：

```bash
sudo apt-get install python3-tk
```

## 设计边界

- 实验采用协作式确定性调度，不是操作系统线程级并发；这保证预算和仲裁可重复。
- SQLite 负责持久化已提交状态；行锁、等待队列、隔离快照和死锁图由应用层维护。
- 每会话操作序列按声明顺序执行，所以 `concurrency` 作为上界和展示字段参与约束，实际事务内
  最多只有一个在途有序操作，不会并发执行相互依赖的操作。
