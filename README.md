# COLMV：本地列式物化视图增量聚合组件

这是一个面向 Windows 本地环境的纯 Python 组件。它从本地 JSON 读取视图定义与基表目录，从本地二进制列式页读取基表，按 NDJSON 变更流维护物化视图分组，并在整数缓存槽位预算下确定性选择“槽位驻留增量收敛”或“受影响页重扫收敛”。两条路径的最终物化视图结果按规范化 JSON 逐字节一致。

项目不依赖 Docker、数据库服务或网络。唯一外部运行依赖是 `numpy>=1.24`，账本和会话快照使用 Python 标准库 `sqlite3`。

## 交付内容

- `colmv/`：可集成领域组件与 CLI。
- `samples/`：24 行、3 个列式页、3 个视图、20 条 NDJSON 变更的确定性样本。
- `samples/views_v2.json`：用于演进/失效传播示例的新版视图定义。
- `scripts/generate_samples.py`：可重复生成全部样本与页文件。
- `examples/in_memory.py`：直接调用领域组件的内存集成示例。
- `tests/`：31 项标准库 `unittest` 测试。
- `docs/ARCHITECTURE.md`：数据格式、确定性规则、事务与恢复设计。
- `docs/VERIFICATION.md`：本次实际执行命令和验证结果。

## Windows 快速开始

在项目根目录执行：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install "numpy>=1.24"
python -m colmv doctor
```

`doctor` 会检查 Python >= 3.11、NumPy >= 1.24、SQLite3，并输出实际版本。样本已经提交；也可以重新生成：

```powershell
python scripts/generate_samples.py
```

直接调用组件 API：

```powershell
python examples/in_memory.py
```

## 五分钟示例

强校验样本：

```powershell
python -m colmv validate `
  --catalog samples/base/catalog.json `
  --views samples/views.json
```

同一变更流分别跑驻留路径与重扫路径，并逐字节比较结果：

```powershell
python -m colmv verify-paths `
  --catalog samples/base/catalog.json `
  --views samples/views.json `
  --changes samples/changes.ndjson `
  --budget 1 `
  --batch-plan 3,6,3,6,2
```

输出包含 `equal_bytes`、`materialized_digest`、`byte_length`、两条路径的命中/置换/重扫计数，以及每批槽位分区互补检查。当前样本结果为 `equal_bytes=true`、`17017` 字节、摘要 `d7467813ac4d0180dca66d388090665088193cfce010e1f57103964fad902723`。

写入 SQLite 账本：

```powershell
python -m colmv run `
  --catalog samples/base/catalog.json `
  --views samples/views.json `
  --changes samples/changes.ndjson `
  --ledger demo.sqlite `
  --budget 1 `
  --batch-plan 3,6,3,6,2
```

样本批次规模 `3,6,3,6,2` 与预算 `1` 会得到路径序列 `resident,rescan,resident,rescan,resident`，切换历史逐批留痕。每批在同一 SQLite 事务中提交分组结局、置换/重扫事件、中间状态、固定点口径版本和游标。

重放校验：

```powershell
python -m colmv replay `
  --catalog samples/base/catalog.json `
  --ledger demo.sqlite `
  --changes samples/changes.ndjson
```

扫描并修复未提交批次：

```powershell
python -m colmv recover --ledger demo.sqlite
python -m colmv run `
  --catalog samples/base/catalog.json `
  --views samples/views.json `
  --changes samples/changes.ndjson `
  --ledger demo.sqlite `
  --budget 1 --resume --batch-size 4
```

视图定义/分组键/过滤条件/预算演进：

```powershell
python -m colmv evolve `
  --catalog samples/base/catalog.json `
  --old-views samples/views.json `
  --new-views samples/views_v2.json `
  --old-budget 1 --new-budget 1 `
  --changes samples/changes.ndjson `
  --ledger demo.sqlite
```

演进会输出稳定的受影响分组集合、受影响页集合和每步依据；旧方案批次/分组/事件整体标记为陈旧，快照清空，要求整体重算，不做逐组修补。

## 本地数据契约

### 基表目录

```json
{
  "format_version": 1,
  "table": {
    "name": "sales_events",
    "row_count": 24,
    "row_key": "row_id",
    "columns": [
      {"name": "row_id", "type": "int"},
      {"name": "region_id", "type": "int"},
      {"name": "amount", "type": "float"}
    ]
  },
  "pages": [
    {"page_id": "p000", "file": "pages/p000.col", "start_row": 0, "end_row": 8, "row_count": 8}
  ]
}
```

约束：

- 列类型只允许 `int`/`float`，整数以有符号 64 位存储，浮点以 IEEE-754 `float64` 存储。
- `row_key` 必须是整数列且值唯一、非空。
- 页目录区间必须从 0 开始、首尾相接、不重叠、最终覆盖 `row_count`。
- 任一页头必须与目录中的 `page_id`、`start_row`、`end_row` 一致。
- 每个页内每列有空值位图；位图一个字节一行，长度必须严格等于页行数。
- 浮点列所有编码值必须是有限数（NaN/Inf 被拒绝）。
- 语义契约问题返回 `E_VIEW`，物理文件截断/魔数/长度损坏返回 `E_INTEGRITY`。

### 列式页文件

页格式是确定性二进制：

```text
b"COLMVPG1\n" | uint32 header_length | canonical JSON header | payload
```

payload 按列依次存放“每行 1 字节空值位图”和“每行 8 字节小端数据”。header 给出每列偏移、长度、dtype 和列数据 SHA-256。详细布局见 `docs/ARCHITECTURE.md`。

### 视图定义

```json
{
  "format_version": 1,
  "scale_power": 3,
  "views": [
    {
      "view_id": "sales_by_region",
      "group_by": ["region_id"],
      "aggregates": ["count", "sum", "avg"],
      "value_column": "amount",
      "filters": [{"column": "status", "op": "eq", "value": 1}],
      "refresh_mode": "auto"
    }
  ]
}
```

- 支持聚合：`count`、`sum`、`avg`（`mean` 归一化为 `avg`）。
- `sum`/`avg` 必须声明数值型 `value_column`；`count` 统计过滤后行数。
- 支持过滤操作：`eq`、`ne`、`gt`、`gte`、`lt`、`lte`、`in`、`not_in`、`is_null`、`is_not_null`。
- `refresh_mode` 支持 `auto`、`resident`、`rescan`；兼容别名 `incremental`、`page_rescan`。仅当所有视图声明同一个非 `auto` 模式时，视图声明强制路径，否则由自动规则决定。
- 视图标识唯一、分组列存在、聚合受支持、过滤列存在；任意非法定义整体拒绝为 `E_VIEW`，并给出首个非法 `view_id` 与 `field`。

### NDJSON 变更流

```json
{"seq":1,"op":"update","row_id":2,"row":{"row_id":2,"region_id":10,"product_id":101,"amount":8.25,"qty":1,"status":1}}
{"seq":2,"op":"delete","row_id":6}
{"seq":3,"op":"insert","row_id":1001,"row":{"row_id":1001,"region_id":10,"product_id":100,"amount":13.579,"qty":2,"status":1}}
```

- `seq` 必须严格递增，`row_id` 必须为整数并等于 `row` 中 `row_key` 列的值。
- `insert`/`update` 的 `row` 必须包含全部基表列；`delete` 可省略 `row`。
- 更新不得改变 `row_key`；插入不得复用历史 `row_id`。
- 插入行归入虚拟页 `delta`；页文件保持不可变，重扫时读取原页并叠加当前已提交行状态。

## 路径选择规则

每个批次按以下确定性规则选择路径。令：

- `C` = 本批变更条数
- `P` = 本批受影响页数
- `V` = 视图数
- `B` = 整数槽位预算（必须 > 0）
- `R = C / B`

规则：

1. `R < 7/2`：选择 `resident`。
2. `R > 9/2`：选择 `rescan`。
3. `7/2 <= R <= 9/2`：标记 `boundary_affected=true`，比较整数工作量：
   - `resident_work = C * V`
   - `rescan_work = P * V * 8 + C`
   - `resident_work <= rescan_work` 选 `resident`，否则选 `rescan`；精确相等时稳定选 `resident`。

输出包含 `reason`、`change_count`、`affected_page_count`、`view_count`、`budget`、比例分数、决策阈值、边界上下界、到阈值距离、投影工作量和是否受边界影响。每批路径变化都会写入 `switch_history`。

## 槽位、命中与重扫

- 槽位预算是全视图共享的整数上限。置换使用整数单调时钟 LRU；若时钟相同，按 `(view_id, typed group key)` 稳定决胜，不比较浮点值。
- 分组首次进入空闲槽位不计命中；同一批或后续驻留期内再次变更计一次命中。
- 满槽时输出置换事件：`event_seq`、被置换分组键、占用该槽位的分组键、槽位号和原因。
- 本批被置换的分组不重新进入本批驻留集合；未获槽位的受影响分组进入重扫集合。每批严格满足：
  `resident_affected ∪ rescan_affected = affected` 且两者不相交。
- `rescan` 路径对重扫集合按当前引用页逐页重读本地页文件，叠加已提交行状态，重建中间结果；每个 `(group, page)` 输出 `event_seq`、分组键、页标识和原因。重扫分组不进入驻留集合，也不计命中。

## 整数定点与误差

- 会话缩放因子为 `10^scale_power`，样本为 `10^3`。
- 浮点输入在摄入时向零截断为整数定点单位；整数输入精确。
- 均值除法同样向零截断。
- 最终 `count`、`sum`、`avg` 都提供整数定点值，并显式给出 `fixed-v1;scale=10^k;trunc=toward-zero;saturate=int64` 口径。
- 内部累加使用 Python 任意精度整数，输出时饱和到有符号 64 位；每个聚合附带 `saturated` 标志。
- 累加顺序为 `change_seq, initial_row_order, row_id`；分组按带类型的分组键稳定排序，与驻留或重扫时机无关。
- 每个分组输出 `error_bounds_fp_units`：`count=0`；`sum` 上界为浮点非空值个数；`avg` 上界为 2（有浮点输入）或 1（仅整数）。全局输出单值量化误差上界、误差段列表、`share_ppm` 和占比最高的误差段。

## 账本、快照与恢复

SQLite 账本格式版本为整数 `1`，包含：

- `meta`：账本格式版本、快照格式版本、链式内容摘要、最后批次和游标。
- `schemes`：方案版本、fingerprint、规范化视图、预算、固定点口径、是否当前、被替代版本。
- `batch_attempts` / `attempt_groups`：批次意图及计划分组。
- `batches`：每批路径、决策、事件、分组结局、物化摘要、物化结果、固定点版本和游标。
- `group_states` / `events`：可逐项重放的逐组中间状态和事件。
- `snapshot`：单行会话快照，包含格式版本、方案版本、摘要和完整 payload。
- `recovery_log` / `invalidation_runs`：恢复与演进审计。

处理每批前先独立提交批次意图；成功批次在**同一事务**中写入批次、分组状态、事件、固定点口径版本、游标、链式摘要和快照。被强杀后：

- 已完整提交批次可恢复并继续；
- 处于未提交批次中的计划分组被标记为 `unknown`，必须重跑；
- `recover` 明确返回 `exit_report_available=false`，不会伪造未完成批次的退出报告；
- 打开快照时校验格式版本、方案版本和内容摘要；任一不一致会从最后完整提交批次整体替换快照。

## 错误码

| 错误码 | 退出码 | 含义 |
|---|---:|---|
| `E_USAGE` | 2 | 命令行/API 用法错误 |
| `E_VIEW` | 3 | 视图、基表目录或页语义契约非法；带首个 `view_id`/`field` |
| `E_BUDGET` | 4 | 槽位预算 <= 0 |
| `E_INTEGRITY` | 5 | 页文件截断、魔数、长度或列摘要损坏 |
| `E_REPLAY` | 6 | 重放某分组/字段/事件与账本不一致 |
| `E_DIVERGE` | 7 | 两条路径物化结果字节不一致；带首个差异分组键 |
| `E_CHANGE` | 8 | NDJSON 变更流序号、操作或行形状非法 |
| `E_LEDGER` | 9 | 账本状态/版本/恢复条件冲突 |

异常注入示例：

```powershell
python -m colmv validate --catalog samples/base/catalog.json --views samples/views.json --inject truncate-page
python -m colmv run --catalog samples/base/catalog.json --views samples/views.json --changes samples/changes.ndjson --ledger x.sqlite --budget 1 --inject zero-budget
```

分别返回 `E_INTEGRITY` 和 `E_BUDGET`。

## 测试

```powershell
python -m unittest discover -s tests -v
```

测试覆盖强校验、错误码、固定点截断/饱和、双路径逐字节一致、严格槽位互补、边界决胜、跨 `PYTHONHASHSEED` 确定性、账本事务与快照替换、强杀恢复、重放差异定位、演进失效和 `E_DIVERGE` 定位。

实际验证结果与命令见 `docs/VERIFICATION.md`。
