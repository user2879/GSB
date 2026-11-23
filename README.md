# WAL 崩溃恢复诊断应用（Windows 本地版）

一个不依赖网络服务、容器或第三方运行时库的数据库预写日志（WAL）崩溃恢复诊断工具。应用使用 Python 3.10+、SQLite、版本化本地页文件和 Tkinter，提供可重复的加载、预算内并行重放、检查点、强制终止、断点恢复、恢复校验、快照整体替换与异常注入演示。

## 快速开始（Windows）

1. 安装 [python.org](https://www.python.org/) 的 Python 3.10 或更新版本，并保留安装器中的 **tcl/tk and IDLE** 选项。`sqlite3`、`tkinter`、`hashlib` 等均为标准库，无需 pip 安装运行时依赖。
2. 在项目目录打开 PowerShell 或 `cmd`：

```bat
cd /d <项目目录>
py -3.10 demo.py
```

`demo.py` 会清理并重建相对工作区 `demo_workspace` 与 `demo_alt_workspace`，然后依次验证正常重放、检查点、快照、强制终止、断点恢复、恢复校验、快照差异和三类异常。成功时最后一行是 `all_assertions=passed`。Windows 也可执行 `run_demo.bat`。

启动交互界面：

```bat
py -3.10 -m walcrash --workspace demo_workspace --gui
```

界面首次打开时若工作区尚未初始化，会提示先加载样本。点击“加载样本…”选择 `samples/sample.json`，再按“执行重放”。所有刷新操作均为只读查询，不会修改页版本或账本。

运行测试（仅测试需要 pytest）：

```bat
py -3.10 -m pip install -r requirements-dev.txt
py -3.10 -m pytest -q
```

## 交付内容

```text
walcrash/                 应用包
  validation.py           页/WAL 强校验与目标版本计算
  content.py              跨平台确定性页字节与摘要
  storage.py              SQLite、页文件、原子批次、哈希链账本
  engine.py               预算调度、并行准备、单线程 LSN 定序提交
  verification.py         独立顺序基线、E_REPLAY 首个差异定位
  snapshots.py            自摘要快照、整体替换与页差异
  gui.py                  Tkinter 专业诊断界面（延迟导入）
  cli.py / __main__.py    命令行入口
samples/                  有效样本、差异样本与异常样本
tests/                    24 项自动化测试
demo.py                   可重复端到端演示
run_demo.bat              Windows 演示启动脚本
docs/TECHNICAL.md         存储模型、算法和技术说明
docs/VALIDATION.md        实际验证结果
```

运行后相对工作区中的产物：

```text
<workspace>/state.sqlite3             页表、WAL 元数据、任务、检查点、批次、账本
<workspace>/pages/<token>/v*.bin      版本化页内容文件
<workspace>/inputs/sample.json        规范化输入副本
<workspace>/inputs/wal.ndjson         活动 WAL 内容
<workspace>/snapshots/*.json          会话快照
<workspace>/logs/app.ndjson           结构化操作/异常日志
<workspace>/logs/injections.ndjson    三类异常注入日志
<workspace>/ledger.json               独立复核导出（运行 demo 或 export-ledger）
```

## 输入格式

`samples/sample.json`：

```json
{
  "format_version": 1,
  "pages": [
    {"page_id": "p1", "size_bytes": 256, "initial_version": 1},
    {"page_id": "p2", "size_bytes": 512, "initial_version": 3},
    {"page_id": "p3", "size_bytes": 128, "initial_version": 0}
  ],
  "budget": {
    "global_slots": 3,
    "per_tx_quota": 2,
    "page_byte_budget": 512,
    "checkpoint_interval": 4
  },
  "log_file": "sample.ndjson"
}
```

也可在 JSON 中直接使用 `"wal": [...]`，但随附演示使用外部 NDJSON。`samples/sample.ndjson` 的每一行为：

```json
{"lsn":1,"txid":"tx-001","page_id":"p1","content_digest":"<sha256>","commit":false,"persist":false}
```

字段与校验规则：

- `page_id`：非空字符串或非负整数，规范化后必须唯一。
- `size_bytes`：受支持的 2 的整数幂，范围 64 B 到 1 MiB。
- `initial_version`：非负整数。
- `lsn`：正整数且唯一，首条之后必须逐 1 递增；任何重复、跳号或倒退整体拒绝。随附样本从 1 开始。
- `txid`：同一事务的日志必须形成一个连续组；组标识不能在其他位置再次出现。
- `commit` / `persist`：必须是 JSON 布尔值。提交标记只能出现在事务组最后一条日志。
- `content_digest`：64 位十六进制 SHA-256，必须等于按“页标识 + 目标页大小 + 目标版本 + LSN”生成的确定性内容摘要。
- `global_slots`、`per_tx_quota`、`page_byte_budget`、`checkpoint_interval`：均至少为 1；页字节预算不得小于最大页。

任一条目不合法都会整体拒绝，不创建部分状态。页声明问题返回 `E_PAGE`，顺序问题返回 `E_ORDER`，其余首个坏日志返回 `E_LOG`，上下文包含 `first_invalid_lsn` 和 `field`。

## 关键语义

### 崩溃点与可见性

1. 事务组只有在最后一条为 `commit=true` 时才算已提交；尾部允许存在一个未提交事务组。
2. 已提交事务组内出现任一 `persist=true` 时，该组末端成为候选崩溃点。
3. 崩溃点是最后一个“已提交且有持久化标记”的完整事务组末端；在它之后的完整事务也视为尚未进入持久前缀。
4. LSN 小于等于崩溃点的页版本 `visible=1`。崩溃点之后的页版本仍作为诊断记录写入并保留，但 `visible=0`，不参与恢复后页视图。
5. 随附样本崩溃点为 LSN 10；LSN 11、12 属于尾随未提交事务，页版本 6/3 被标记不可见。

### 并行重放与单线程日志语义

- 日志按“事务 → 页”切分任务，再按页字节预算与检查点间隔切块。
- 全局槽位限制同时运行任务数；每事务配额限制同一事务同时运行任务数。
- 任务在工作线程中生成确定性页字节和摘要，完成顺序允许乱序。
- 实际页版本、摘要、任务结局与游标只由单线程 LSN 定序器按日志顺序提交，因此线程完成顺序不会污染最终页状态。
- 同一页同一事务的多次写入在定序器中按 LSN 串行；跨页任务可以并行准备，提交顺序显式固定为 LSN 升序。
- 被槽位/配额挡住的任务记录 `wait_started_ms`、`wait_ended_ms`、`started_ms`、`finished_ms`，时钟为进程单调毫秒。

### 批次、检查点与崩溃恢复

- 定序器按 `checkpoint_interval` 形成批次，每批在同一个 SQLite 事务中写入页版本、摘要、任务结局、预算消耗、检查点、账本事件和进度游标。
- 页内容文件先以版本化路径原子写入；随后 SQLite 批次提交。崩溃发生在两者之间时，文件只是未被引用的孤儿，SQLite 游标才代表已提交状态。
- 每批都生成检查点；另可执行手工检查点。检查点包含可见已提交页状态及账本游标。
- 恢复从账本游标继续，不重放已提交页版本，并校验当前预算参数与账本记录一致。
- 强制终止后，`queued/running/staged/partial` 任务转为 `unknown` 并要求重跑；中断运行标为 `interrupted_unknown`，绝不伪造退出报告。只有完整提交的批次可信。

### 隔离级别不改变结果

同一 WAL 在 `global_slots=3, per_tx_quota=2` 和 `global_slots=1, per_tx_quota=1` 下重放后，页版本号、摘要、页文件和可见页视图逐字节一致；测试对此进行了直接比较。

### 快照

- 快照是带整数 `snapshot_format_version` 的 JSON，包含日志摘要、页表版本、预算参数版本、检查点集合、账本游标、全部相关表、页文件内容、活动 WAL 和输入的规范化副本。
- 顶层 `content_digest` 对除自身外的完整快照载荷做规范化 SHA-256；不一致立即 `E_SNAPSHOT`，且不修改当前状态。
- 打开快照时比较快照的 WAL 摘要与当前磁盘 WAL，以及当前页版本/页文件摘要。无论差异来自日志还是页，恢复都执行整表、整账本、整检查点替换，不做增量合并。
- 返回稳定排序的 `added`、`deleted`、`changed` 页集合；`changed` 中同时展示当前页文件摘要与快照摘要。

### 恢复校验

`--action verify` 从 `inputs/sample.json` 重新执行独立的单线程顺序基线，不依赖调度器完成顺序，逐项比较：

- 每个 LSN 的页标识、版本号、内容摘要、可见性；
- SQLite WAL 元数据；
- 磁盘页文件摘要；
- 已持久可见事务集合；
- 崩溃点；
- 每个检查点的页状态集合；
- 账本哈希链与预算摘要。

首个差异以 `E_REPLAY` 报告 `first_mismatch_lsn`、`page_id`、`field`、期望/实际值，并附覆盖该 LSN 或最近的批次账本记录（`batch_id`、区间、检查点、账本哈希）。

## 命令行

所有 `--snapshot-file`、`--output` 相对路径都相对 `--workspace`；`--input` 相对当前目录。工作区默认 `demo_workspace`。

```bat
:: 加载（会重建工作区状态，保留诊断日志）
py -3.10 -m walcrash --workspace demo_workspace --action load --input samples/sample.json

:: 重放（有未完成批次时也是断点恢复入口）
py -3.10 -m walcrash --workspace demo_workspace --action run --delay-ms 30
py -3.10 -m walcrash --workspace demo_workspace --action resume

:: 手工检查点、校验、状态
py -3.10 -m walcrash --workspace demo_workspace --action checkpoint
py -3.10 -m walcrash --workspace demo_workspace --action verify
py -3.10 -m walcrash --workspace demo_workspace --action status

:: 快照
py -3.10 -m walcrash --workspace demo_workspace --action snapshot --snapshot-file snapshots/session.json
py -3.10 -m walcrash --workspace demo_workspace --action restore  --snapshot-file snapshots/session.json

:: 可独立复核的账本导出
py -3.10 -m walcrash --workspace demo_workspace --action export-ledger --output ledger.json

:: 重置工作区
py -3.10 -m walcrash --workspace demo_workspace --action reset
```

强制终止演练：子进程在首批提交后直接 `os._exit(91)`。退出码 91 是故意的，不是应用崩溃报告；重启后应看到 `running` 运行、活动任务和仅一个已提交批次。

```bat
py -3.10 -m walcrash --workspace crash_workspace --action load --input samples/sample.json
py -3.10 -m walcrash --workspace crash_workspace --action run --delay-ms 90 --crash-after-batches 1
echo %ERRORLEVEL%
py -3.10 -m walcrash --workspace crash_workspace --action resume
py -3.10 -m walcrash --workspace crash_workspace --action verify
```

## 显式异常注入

```bat
py -3.10 -m walcrash --workspace demo_workspace --inject page_size
py -3.10 -m walcrash --workspace demo_workspace --inject order
py -3.10 -m walcrash --workspace demo_workspace --inject snapshot_digest
```

预期退出码与业务码：

| 注入 | 业务码 | 进程退出码 | 首个字段 |
| --- | --- | ---: | --- |
| 页大小非法 | `E_PAGE` | 10 | `size_bytes` |
| 日志序号倒退/重复 | `E_ORDER` | 11 | `lsn` |
| 快照内容摘要不符 | `E_SNAPSHOT` | 13 | `content_digest` |

三类异常都会追加到 `<workspace>/logs/injections.ndjson`，完整错误同时写入 `logs/app.ndjson`。`snapshot_digest` 注入会在需要时自动准备并重放样本，再生成有效快照和篡改副本；它不会把篡改状态合并进当前账本。

## 业务码

| 业务码 | 含义 | 退出码 |
| --- | --- | ---: |
| `E_PAGE` | 页标识/大小/初始版本错误 | 10 |
| `E_ORDER` | WAL 序号不连续、重复或倒退 | 11 |
| `E_LOG` | WAL 结构、页引用、事务分组或内容摘要错误 | 12 |
| `E_SNAPSHOT` | 快照版本、自摘要、内部一致性或页文件错误 | 13 |
| `E_REPLAY` | 恢复结果与独立顺序基线不一致 | 14 |
| `E_BUDGET` | 恢复预算参数非法 | 15 |
| `E_STATE` | 工作区/动作状态错误 | 16 |
| `E_GUI` | 当前 Python 缺少可用 Tkinter/显示 | 17 |

## Tkinter 界面

界面包含七个稳定排序的页签：

- **总览**：游标、崩溃点、页/版本/批次/检查点/未知任务、页表/预算/账本版本与账本 HEAD。
- **页列表与版本**：初始版本、可见版本、已应用版本、摘要和“存在不可见新版本”状态。
- **WAL 日志区间**：按起止 LSN 查询，显示目标版本、提交/持久化/可见标记。
- **重放调度时间线**：每任务排队、等待、运行单调毫秒、结局、预算字节、消耗与批次；顶部显示总预算与累计字节。
- **检查点 · 账本 · 快照版本**：检查点集合、原子批次、哈希链事件。
- **崩溃点与不可见版本**：崩溃点和所有 `visible=0` 页版本及原因。
- **快照**：版本化快照列表与整体替换差异。

刷新只执行 `SELECT` 查询，不使用恢复流程，也不写账本。界面支持加载样本、重放/断点恢复、校验、手工检查点、创建/恢复快照和强制终止演练；长操作在后台线程执行，Tk 主线程仅负责渲染。

## 技术说明与验证

- 详细数据结构、算法、事务边界与设计取舍见 `docs/TECHNICAL.md`。
- 本次实际执行结果见 `docs/VALIDATION.md`：24 项 pytest 全部通过、端到端演示全部断言通过、GUI 在 Xvfb + Tk 8.6 下实际启动，三类异常均返回预期业务码。

## 约束与已知边界

- 这是本地确定性诊断应用，不连接真实数据库引擎；页字节由公开、可复算的规则生成。
- 页内容文件与 SQLite 无法组成一个跨介质事务，因此采用“文件先写、SQLite 批次提交为准、孤儿文件忽略”的安全顺序。
- 本容器基线为 Linux Python 3.11.0rc1；代码只使用 Python 3.10+ 标准库与跨平台路径/原子替换 API，目标运行环境为 Windows 本地 Python 3.10+。Windows 验证步骤见 `docs/VALIDATION.md`。
- 不使用 Docker、容器、网络服务、真实账号或 GPU。
