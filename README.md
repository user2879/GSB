# 增量物化视图维护工作台

面向 Windows 本地 Python 环境的增量物化视图维护工作台。项目只依赖 Python 标准库（`sqlite3`、`tkinter`、`threading`、`json` 等），不访问网络端口，不依赖 Docker，也不使用第三方包。

目标是把两条路径严格分开：

- **界面快速视图**：只读取增量结果缓存，按视图、分组键、租约状态和不一致标记渲染，不触发全量重算。
- **权威重算**：从初始基表和 NDJSON 变更流逐行重放，独立重建全部视图，并与增量缓存做逐字节判定；首个差异定位到视图、分组键和字段。

## 快速开始（Windows）

要求 Python 3.10+，并确认解释器包含 `sqlite3` 和 `tkinter`：

```bat
python --version
python -c "import sqlite3, tkinter; print(sqlite3.sqlite_version, tkinter.TkVersion)"
```

在项目目录双击或运行：

```bat
run_gui.bat
```

界面默认使用 `samples/` 中的确定性样本，账本、快照和结构化日志写入 `artifacts/gui-workbench/`。界面按钮支持推进批次、推进全部、权威重算、一致性重放、重建所选视图、模拟租约过期、热替换视图定义和三类异常注入。

无界面验收演示：

```bat
run_demo.bat
```

等价的命令行形式：

```bat
python -m imv.cli demo --data-dir artifacts\demo-workbench --json
python -m imv.cli validate --json
python -m unittest discover -v
```

`demo` 每次默认重建数据目录，因此可以直接重复执行。成功时退出码为 `0`，同时生成 `artifacts/demo-workbench/demo_report.json`。

## 命令行

```text
python -m imv.cli validate   [--base ...] [--views ...] [--changes ...] [--json]
python -m imv.cli demo       [--data-dir ...] [--batch-size N] [--inject ...] [--json]
python -m imv.cli maintain   [--steps N] [--batch-size N] [--reset] [--json]
python -m imv.cli gui        [--data-dir ...] [--reset]
```

异常注入：

```bat
python -m imv.cli demo --inject lease_conflict
python -m imv.cli demo --inject base_corruption
python -m imv.cli demo --inject snapshot_tamper
```

分别报告 `E_LEASE`、`E_DATA`、`E_SNAPSHOT`。`snapshot_tamper` 会篡改落盘的增量结果摘要，关闭并重新打开快照，验证检测、整体恢复和恢复后的重放校验。所有业务事件同时写入 `data-dir/logs/structured.ndjson`。

## 输入文件契约

### 基表 `samples/base_table.json`

```json
{
  "format": "imv.base.v1",
  "table": "sales_daily",
  "row_id_column": "sale_id",
  "columns": [
    {"name": "sale_id", "type": "integer", "nullable": false},
    {"name": "region_id", "type": "integer", "nullable": true}
  ],
  "null_bitmap": [
    {"row_id": 3, "columns": ["region_id"]}
  ],
  "rows": [
    {"sale_id": 3, "region_id": null}
  ]
}
```

- 支持列类型：`integer`、`float`、`text`、`boolean`；题目要求的整数/浮点类型均已支持。
- 行标识列必须是 `integer` 或 `text`，不可空且唯一。
- `null_bitmap` 可用数组或对象形式声明空值。数组项为 `{"row_id": ..., "columns": [...]}`，对象形式为 `{"3": ["region_id"]}`。
- 位图与行值必须一致：位图声明为空时不得出现非空值；非空列漏填或出现 `null` 但未登记位图，均以 `E_SCHEMA` 拒绝。
- 浮点值必须是有限值；`NaN`/`Infinity` 不允许。

### 视图定义 `samples/views.json`

```json
{
  "format": "imv.views.v1",
  "session": {
    "scale_factor": 100,
    "truncation": "toward_zero",
    "overflow": "saturate_int64",
    "global_concurrency": 2,
    "batch_size": 2,
    "default_lease_ttl_ms": 900,
    "default_lease_renew_ms": 200
  },
  "views": [
    {
      "id": "v_region",
      "group_by": ["region_id"],
      "aggregates": [
        {"func": "count", "column": "*", "alias": "row_count"},
        {"func": "sum", "column": "amount", "alias": "amount_sum"},
        {"func": "min", "column": "amount", "alias": "amount_min"},
        {"func": "max", "column": "qty", "alias": "qty_max"}
      ],
      "filter": {"op": "eq", "column": "active", "value": 1},
      "refresh_mode": "incremental",
      "per_view_quota": 1,
      "lease_ttl_ms": 900,
      "lease_renew_ms": 200
    }
  ]
}
```

- 聚合函数集合固定为 `count`、`sum`、`min`、`max`。`count` 支持 `column: "*"` 或指定列（后者只计非空值）；数值聚合只允许数值列。
- 过滤条件支持 `eq`、`ne`、`gt`、`gte`、`lt`、`lte`、`in`、`not_in`、`is_null`、`is_not_null`、`and`、`or`、`not`。符号别名 `= != > >= < <=` 也可用。
- `refresh_mode` 支持 `incremental`（仅重算受影响分组）和 `full`（每次批次全量重建该视图）。
- `per_view_quota` 是单视图同时运行作业数上限；调度器始终不会超过该值与全局并发上限。
- 租约续约间隔必须严格小于租约时长。

### 变更流 `samples/changes.ndjson`

```json
{"seq":1,"op":"update","row_id":1,"row":{"sale_id":1,"region_id":10,"product_id":101,"amount":130.55,"qty":3,"active":1,"channel_id":1}}
{"seq":2,"op":"insert","row_id":13,"row":{"sale_id":13,"region_id":20,"product_id":202,"amount":88.80,"qty":4,"active":1,"channel_id":2}}
{"seq":3,"op":"delete","row_id":4}
```

- `op` 支持 `insert`、`update`、`delete`；也兼容 `type` 字段。
- `seq` 从 `1` 开始连续递增且不得重复。
- `insert`/`update` 必须提供完整变更后行；`delete` 只需行标识。
- `update`/`delete` 的行标识必须存在于当时基表状态；`insert` 引入新行且不得重复已有标识。该规则同时覆盖初始行和同流中先前的插入。
- 所有输入先完整校验再进入内存状态；任一错误都以 `E_SCHEMA` 整体拒绝，并给出文件、行号或 JSON 路径、首个违规字段。

## 定点聚合与确定性

所有聚合输出统一使用会话整数缩放因子 `scale_factor`，结果表示为有符号 64 位定点整数：

- `count`：`行数 / 非空计数 × scale_factor`。
- `sum`：先对每个非空输入按 `Decimal(str(value)) × scale_factor` 量化，再精确整数求和，最终一次性饱和到 `[-9223372036854775808, 9223372036854775807]`。
- `min`/`max`：对每个非空输入量化，取最小/最大值。
- `truncation`：`toward_zero`（默认）、`floor`、`ceil`，作用于缩放后的数值。
- `overflow`：`saturate_int64`。
- 空集合：`sum = 0`；`min`/`max = null`。
- 分组键中任一列为 `null` 的行显式排除；过滤条件中的普通比较遇到 `null` 按不匹配处理，需要空值判断时使用 `is_null`/`is_not_null`。

结果使用规范化 JSON 字节：UTF-8、键排序、紧凑分隔符、分组按规范分组键排序。增量路径和权威路径使用同一规范化器，因此 `result_json.encode("utf-8")` 可以逐字节比较。`sum` 的“精确累加后一次饱和”不依赖行处理顺序；因此视图并行顺序、批次边界和相同变更集不会改变最终字节。

## 租约、调度与恢复

- 每次刷新作业在真实工作线程上执行，并向 `LeaseManager` 申请租约：`view_id`、`refresh_version`、`granted_mono_ms`、`expires_mono_ms`、TTL 和续约间隔。
- 作业运行期间有独立续约线程按 `lease_renew_ms` 续约；续约必须提交正确的 `refresh_version`，版本不符立即抛出 `E_LEASE`。
- 到期判定使用 `time.monotonic_ns()` 的真实单调时钟，不使用墙上时钟回拨量。
- 租约到期且未续约时，视图立即标记为不一致，并从快速视图隐藏。租约监视器触发全量重建；重建作业先取得新租约，从初始基表和变更流完整重算，再在一个 SQLite 事务中原子替换该视图结果。
- 进程重启时，旧进程遗留的 `active` 租约无法证明仍有续约者，因此统一标记过期并进入重建流程。
- 调度器使用真实 `ThreadPoolExecutor` 工作线程和独立派发线程。全局活跃数与单视图活跃数同时受限；不满足条件的作业保持 `queued`，记录入队、开始、结束单调毫秒和等待毫秒。调度器没有超时杀线程逻辑，不会把超时描述成已终止作业。
- 作业失败写入结构化日志和调度轨迹业务码，后续刷新版本可以重试。

## 账本、快照与一致性

每个数据目录包含：

```text
data-dir/
  ledger.sqlite3
  ledger.sqlite3-wal / -shm
  logs/structured.ndjson
  demo_report.json
```

SQLite 表包括：

- `meta`：整数格式版本、参数版本、游标、快照版本与摘要、当前基表摘要等。
- `param_versions` / `view_defs`：历史与当前参数版本、视图定义摘要。
- `view_results`：每视图增量结果字节、摘要、游标、陈旧/不一致标记。
- `leases`：租约完整载荷与状态。
- `scheduler_jobs`：调度轨迹与业务码。
- `batches`：每个已完成变更批次的整数 `ledger_format_version`、内容摘要、快照版本、游标。
- `snapshots`：完整会话快照载荷与内容摘要。
- `rebuild_traces`：租约过期重建轨迹。
- `definition_replacements`：视图整体替换前后摘要。
- `authority_runs`：权威重算证据。

每完成一个变更批次，增量结果摘要、租约状态、调度轨迹、基表状态、进度游标、批次账本与快照在**同一个 SQLite 事务**中提交。

重新打开快照时：

1. 校验快照格式版本、储藏摘要和快照载荷自身摘要。
2. 从当前结果、租约、游标、基表与重建轨迹重建“当前状态摘要”。
3. 若与最新快照不一致，抛出/记录 `E_SNAPSHOT`，在事务中整体替换当前参数版本的增量结果、租约、游标和基表状态，而不是局部修补。

## 视图整体替换

视图定义、聚合口径或缩放因子改变后，参数版本单调递增。替换流程：

- 旧参数版本的 `view_results`、`leases` 全部标记 `stale=1`，`param_versions.is_current=0`。
- 新参数版本在当前基表游标上一次性重建全部当前视图，随后整体替换，不做增量合并。
- 替换摘要稳定列出 `added`、`removed`、`changed`、`unchanged`，并附带替换前后的指纹、视图数、视图摘要、分组键和缩放参数。
- 快照、快速视图和汇总只读取当前参数版本；旧版本结果不参与展示。

## 权威重算与重放校验

- `authoritative_recompute()` 从初始基表和已提交游标前的全部变更逐行重放，直接全量构建所有视图；不读取增量结果作为中间状态。
- 对每个视图比较规范化结果字节。若不同，报告 `E_PREVI`，包含首个差异的 `view_id`、`group_key`、`field`、缓存值与权威值。
- `replay_verify()` 按账本参数版本和游标重新执行增量维护，将每个视图的分组聚合值、租约状态、不一致标记和重建轨迹逐项与最新账本快照比较。首个差异以 `E_REPLAY` 报告视图、分组键和字段。

## 界面

界面采用 Tk/ttk，包含五个工作区：

- **总览**：参数版本、变更游标、可见视图、运行/排队、快照版本；视图健康度与租约表。
- **聚合结果**：从快速缓存读取当前视图分组聚合值；不一致或租约过期时显示隐藏原因。
- **调度轨迹**：真实工作线程、入队/开始/结束单调毫秒、等待/运行时间、业务码。
- **账本与重放**：SQLite 批次账本、快照证据、替换摘要和重建轨迹。
- **诊断日志**：NDJSON 结构化事件。

所有耗时操作在后台线程执行，Tk 主线程只刷新界面。推导/重算结果通过线程安全队列回到主线程；用户提交后界面先更新状态，不等待批次完成。

## 演示与测试

```bat
python -m unittest discover -v
python -m imv.cli demo --json
python -m imv.cli demo --inject lease_conflict
python -m imv.cli demo --inject base_corruption
python -m imv.cli demo --inject snapshot_tamper
```

验证覆盖：

- 输入强校验和首个违规位置；
- 任意批次边界、任意视图调度顺序下最终结果字节一致；
- 增量缓存与权威重算逐字节一致；
- `E_PREVI`、`E_REPLAY` 的首个差异定位；
- 租约乐观续约、过期隐藏、全量重建与恢复后重放；
- 真实线程全局/单视图并发上限、等待时长、续约与失败重试；
- 参数版本整体替换与旧结果/租约陈旧标记；
- 快照篡改检测、原子恢复和三类异常业务码。

实际验证结果见 [`VERIFICATION.md`](VERIFICATION.md)。

## 目录结构

```text
imv/
  model.py        # 输入加载、强校验、规范化数据模型
  fixed.py        # 定点量化与聚合语义
  engine.py       # 增量维护、权威重放、逐字节差异定位
  lease.py        # 单调时钟租约与乐观续约
  scheduler.py    # 真实线程派发、全局/单视图配额、调度轨迹
  store.py        # SQLite 账本、事务批次、快照检测与恢复
  workbench.py    # 会话状态机、批次、重建、替换、重放、注入
  gui.py          # Tk/ttk 桌面工作台
  cli.py          # 校验、维护、演示、GUI 启动
samples/          # 确定性基表、视图定义、变更流、替换版本
tests/            # 18 项标准库 unittest
run_gui.bat / run_demo.bat
run_gui.py / run_demo.py
artifacts/        # 账本、快照、演示报告与验证证据
```

## 说明与边界

- 一个数据目录绑定一组“初始基表 + 变更流”。变更流在会话中按游标推进；若需要修改样本或从新变更流开始，请使用新的数据目录或 `--reset`。
- 账本格式版本与快照格式版本均为整数 `1`。
- 界面和 CLI 均为本地文件操作，不使用网络端口。
- 样本的 `views_v2.json` 用于演示缩放因子、视图新增/删除/变更后的整体替换。
