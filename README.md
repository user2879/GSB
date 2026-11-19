# BucketPack：确定性样本特征分桶打包组件

本项目面向 Windows 本地环境，从零实现了“样本生成 → 规格校验 → 桶级依赖图 → 单趟展开 → 预算打包 → 并行调度 → 增量重算 → SQLite 账本恢复”的完整链路。所有样本均来自整数随机种子和本地 JSON 文件，不访问真实账号、线上服务或外部数据集，也不依赖 Docker/容器。

本项目不是只演示最终结果：增量重算会与同一变更集的全量重建做逐字段比较；未受影响桶列、包摘要和包版本会单独保留并逐项校验。

## 1. 环境与安装

要求 Python 3.10+。当前沙箱实际验证环境为 Python 3.11.0rc1、SQLite 3.37.2、NumPy 2.4.6；当前 Linux 沙箱未安装 `tkinter`，因此 GUI 只做无桌面兼容与视图模型测试。Windows 官方 Python 安装器默认包含 Tkinter。

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import sqlite3, tkinter, numpy; print(sqlite3.sqlite_version, tkinter.TkVersion, numpy.__version__)"
```

Linux/无桌面环境只需命令行走纯 Python 回退即可运行；NumPy 缺失时 `bincount/sum_counts` 自动使用语义等价的纯 Python 实现。

```bash
python3 -m pip install -r requirements.txt   # 可选，有 NumPy 时走数值快路径
python3 -m bucketpack preflight --sqlite-path artifacts/ledger.sqlite
```

`preflight` 检查 Python、SQLite、Tkinter、NumPy、`127.0.0.1` 本地端口绑定以及 SQLite/日志路径写入能力。

## 2. 快速运行

```bash
# 重新生成两个数据源的本地确定性样本（已随仓库提供）
python tools/generate_samples.py

# 规格和图校验
python -m bucketpack validate --spec samples/spec.json --out artifacts/validate.json

# 全量运行并写入 SQLite 账本
python -m bucketpack run --spec samples/spec.json \
  --ledger artifacts/ledger.sqlite --result-out artifacts/run_result.json

# 值改写/空值位图调整后的增量运行；自动与全量重建比对
python -m bucketpack incremental --spec samples/spec.json \
  --changes samples/changes_value.json \
  --result-out artifacts/incremental_result.json \
  --report-out artifacts/incremental_report.json

# 账本恢复：只接受完整提交的包，未知包逐包重放
python -m bucketpack recover --spec samples/spec.json \
  --ledger artifacts/ledger.sqlite --batch-id <batch-id>

# 完整验收演示：增量、崩溃恢复、版本替换、四类异常注入
python -m bucketpack demo --spec samples/spec.json --out-dir artifacts/demo

# 可选桌面检查器（需要 Tkinter 和桌面会话）
python -m bucketpack gui --spec samples/spec.json \
  --changes samples/changes_value.json \
  --ledger artifacts/ledger.sqlite \
  --log-file artifacts/logs/demo_final.jsonl

# 41 项自动化测试
python -m unittest discover -s tests -t . -v
```

Windows 可执行 `scripts\check.ps1` 一次完成样本重生成、测试和演示。

## 3. 仓库结构

```text
bucketpack/             核心实现
  spec.py               强校验规格与规范化
  graph.py              桶级依赖图、环检测、E_REF
  source.py             确定性样本生成/加载、空值位图
  numeric.py            Decimal 定点量化、可选 NumPy 计数
  engine.py             单趟展开、打包、统计合并、成本账本
  scheduler.py          全局列宽预算 + 数据源配额线程池调度
  changes.py            变更集强校验
  incremental.py        失效传播、最小重算、增量/全量比对
  ledger.py             SQLite 逐包事务、恢复、E_REPLAY
  cli.py / __main__.py  命令行与异常注入
  view.py / gui.py      稳定视图模型与 Tkinter 界面
samples/                本地确定性规格、数据文件和变更集
tests/                  41 项标准库 unittest
tools/generate_samples.py
docs/DESIGN.md          算法口径与数据模型
docs/VALIDATION.md      本次实际验证结果
```

## 4. 规格口径

规格为本地 JSON。完整字段见 `samples/spec.json` 和 `docs/DESIGN.md`。关键规则如下。

- 数据源 `id` 唯一；列名仅允许 `[A-Za-z_][A-Za-z0-9_]*`；类型仅 `int/float`。
- 每个 `null_bitmaps` 条目长度必须等于 `rows`，元素仅 `0/1`；未列出的列默认全非空。
- 桶规则边界必须在**量化后的整数域**严格升序且不重复。浮点列先按 `scale` 定点化。
- `scale` 必须为 0..9 的整数；`rows_per_pack`、`pack_width_budget`、`global_width_budget`、`pack_quota_per_source`、`workers` 均为正整数。
- 任一不合法立即整体拒绝，返回稳定首个字段路径，例如 `$.sources[0].null_bitmaps.spend`。
- JSON 对象重复键也会被 `E_SPEC` 拒绝。

### 定点量化

```text
q = trunc_toward_zero(Decimal(str(value)) * 10^scale)
if q < -2^63: q = -2^63
if q >  2^63-1: q = 2^63-1
bucket = bisect_right(boundaries, q)
```

因此桶 0 是 `(-inf, b0)`，中间桶是 `[b[i-1], b[i])`，最后桶是 `[b[-1], +inf)`。浮点列未发生饱和时误差上界为 `10^-scale`；发生 int64 饱和时输出该列实际观测到的最大误差作为可核验上界。

### 空值填充与列宽

空值先用列的 `null_fill` 填充后参与量化与分桶，同时保留独立空值位图供统计。每行经过的每个桶列固定占用：

```text
cell_width = width_bits + 1
width_bits = user width_bits or max(1, ceil(log2(bucket_count)))
```

其中 `+1` 是存在/空值位。空值单元仍占用完整固定槽位，因此包边界和行宽不会因空值位置变化而漂移；空值数量单独计入逐包摘要与合并统计。单行行宽超过 `pack_width_budget` 时，在切包前返回 `E_WIDTH`，包含主键和行宽。

### 打包与调度

- 包按数据源内主键顺序切分，行区间首尾相接、无重叠、无遗漏，`rows_per_pack` 是上限。
- 包宽度为该包所有行固定槽位宽度之和。
- 就绪包按 `pack_id` 稳定排序；调度扫描同一顺序，不依赖字典/哈希遍历。
- 同一时刻运行包的 `width_used` 之和不超过 `global_width_budget`，每个数据源活动包数不超过 `pack_quota_per_source`。
- 每包记录 `queue_start_ms/start_ms/end_ms/wait_ms`，使用 `time.perf_counter_ns()` 单调时钟毫秒。

## 5. 依赖图、增量与一致性

桶规则既可直接引用数据源列，也可引用此前桶列。统计规则通过 `refs` 引用桶列计数，形成“规则边 + 统计派生引用”。

- 目标不存在、跨源引用或引用目标边界缺失：`E_REF`。
- 存在环：`E_CYCLE`，`cycle` 列表按字典序最小旋转稳定输出。
- 删除仍被规则或统计引用的桶列：`E_REF`，列出引用者、字段和路径。

变更集支持：

- `values`：改写列值；
- `nulls`：调整空值位图；
- `rules`：调整桶边界；
- `delete_rules`：删除桶规则（仅允许未被引用）。

传播规则：

- 值/空值变更：从直接输入该列的桶规则出发，沿依赖图传播到全部间接桶列；只失效包含变更行的包。
- 边界变更：失效该规则及全部下游桶列，并重算该数据源全部包。
- 规则删除：若未被引用则删除；行宽或规则集合变化导致该数据源重新切包。
- 每个触发变更产生稳定 `trace`，可回溯 `trigger → bucket_column → path/depth`。

未失效桶列直接复用原对象；包摘要中的 `bucket_versions` 是**包内行切片版本**而不是整列版本。这样同一桶列在其他行变化时，未受影响包仍保持逐字节不变；整列版本保留在数据源展开结果中。

增量结束后自动执行同变更集的全量重建，并比较图、展开结果、包、合并统计、派生统计、成本账本和策略字段。首个差异以 `E_DIVERGE` 报告首个差异桶列和包标识。

## 6. SQLite 账本与恢复

账本格式版本为整数 `1`。每个包在一笔 SQLite 事务中写入包标识、数据源、逐包摘要、列宽、成本、起止时钟和内容摘要；批次内容摘要和整账本内容摘要同事务更新。

恢复流程：

1. 只读取状态为 `COMMITTED` 的包。
2. 批次清单中存在但未提交的包标记为 `UNKNOWN`，输出并要求重跑。
3. 逐包比较摘要、宽度、成本、包摘要和内容摘要；不比较跨进程无意义的单调时钟。
4. 格式版本、规格摘要、清单、包内容或账本摘要不一致时，整体替换账本状态，然后以 `E_REPLAY` 报明首个差异包。
5. 正常前缀恢复后，CLI 重跑全部未知包并完成批次。

## 7. 业务码与异常注入

| 业务码 | 场景 |
|---|---|
| `E_SPEC` | 规格校验失败，稳定首个字段路径 |
| `E_REF` | 引用缺失或删除仍被引用的桶列 |
| `E_CYCLE` | 依赖图成环，列出环上桶列 |
| `E_WIDTH` | 单行/包/全局列宽预算无法满足 |
| `E_BUCKET` | 样本值与列类型冲突 |
| `E_LEDGER` | SQLite 账本打开/写入/事务失败 |
| `E_REPLAY` | 版本或内容回放不一致，账本已整体替换 |
| `E_DIVERGE` | 增量与全量、直接与合并统计不一致 |

命令行可重复注入：

```bash
python -m bucketpack run --spec samples/spec.json --inject width
python -m bucketpack run --spec samples/spec.json --inject bucket
python -m bucketpack run --spec samples/spec.json --inject cycle
python -m bucketpack run --spec samples/spec.json --inject ledger
```

异常和正常批次事件均写入 JSON Lines 结构化日志；每条记录含序号、UTC 时间、级别、业务码、消息和字段。

## 8. GUI

`python -m bucketpack gui` 使用 Tkinter 展示：

- 桶列依赖边；
- 本轮失效桶列、受影响包和传播轨迹；
- 包行区间、列宽占用、成本、排队/开始/结束毫秒；
- 账本批次、格式版本、已提交/未知数量及结构化日志业务码。

所有表格按 `pack_id`、桶列、批次或日志序号稳定排序。无桌面/无 Tkinter 时 CLI、演示、恢复与测试均不受影响。

## 9. 实际验证

详见 `docs/VALIDATION.md`。本次执行结果摘要：

- 41 项 unittest 全部通过（NumPy 2.4.6 快路径与纯 Python 回退路径均验证）。
- 样例 56 行、11 包，最大并发列宽 152 ≤ 160，成本 5600 µ ≤ 1,000,000 µ。
- 值/空值变更仅重算 6 个桶列和 4 个包，7 个包逐字节保持；增量与全量完全一致。
- 边界变更和规则删除分别触发对应数据源全量重包，另一数据源未受影响包保持。
- 强制终止 `os._exit(17)` 后，仅 2 个已提交包通过校验，9 个未知包被逐包重放并完成。
- 账本格式版本篡改触发 `E_REPLAY`，账本被整体替换。
- 四类注入分别稳定得到 `E_WIDTH/E_BUCKET/E_CYCLE/E_LEDGER`。

完整演示报告：`artifacts/demo/demo_report.json`。
