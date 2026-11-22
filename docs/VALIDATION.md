# 实际验证记录

验证日期：2026-11-23（Asia/Shanghai）。

## 环境

| 项目 | 实测值 |
|---|---|
| Python | `3.11.0rc1` |
| SQLite | `3.37.2` |
| NumPy | `1.26.4` |
| 操作系统验证容器 | Linux/WSL（Windows 兼容代码路径） |
| Tkinter | 动态库 `libtk8.6.so` 缺失；GUI 导入给出明确 `E_GUI` 提示，未能在本容器显示窗口 |

Windows 正式验收需使用带 Tcl/Tk 组件的 Python 安装，并运行 `python -c "import tkinter; print(tkinter.TkVersion)"` 与 `run_gui.bat`。

## 自动测试

命令：

```bash
python -m unittest discover -s tests -v
```

结果：

```text
Ran 17 tests in 1.591s

OK
```

覆盖项：

- `E_SPEC` 首个违规位置、重复索引列、未知查询表、非法算子、非法 `between`。
- 精确/抽样统计同时归档、样本量与误差界、直方图中间项。
- 四类路径生成、路径估算连续两次逐字节稳定。
- 候选声明顺序反转后选择结果与估算摘要不变。
- `E_SEARCH` 在贪心与穷举审计真值不同时报告差异索引。
- 三类偏差归因分支。
- 方案版本递增、受影响查询、旧快照/批次陈旧、当前视图整体替换。
- 账本篡改检测和自动整体替换。
- 强制终止后恢复、未知批次、默认拒绝部分复核、`--allow-partial` 复核已提交批次。
- 运行后独立复核全部路径。
- 账本会话权重覆盖无需在复核命令重复传入。

## 样本规模与运行

样本：`examples/sample_spec.json`。

| 表 | 行数 | 自适应口径 | 实际样本量 |
|---|---:|---|---:|
| `products` | 350 | `exact` | 350 |
| `customers` | 20,000 | `sampled` | 2,048 |
| `orders` | 120,000 | `sampled` | 2,048 |

运行命令：

```bash
time python -m qplan.cli run \
  --spec examples/sample_spec.json \
  --workspace "$WORKSPACE" \
  --batch-size 2 \
  --out "$WORKSPACE/report.json"
```

本次工作目录由 `mktemp -d /tmp/qplan-final3.XXXXXX` 创建，实测为
`/tmp/qplan-final3.7Pkg00`。

实测墙钟时间：

```text
real    0m16.277s
user    0m15.221s
sys     0m2.668s
```

结果：

- 总体状态：`OK_WITH_DIAGNOSTICS`
- 查询：4
- 候选索引：11
- 真实执行路径：27
- 路径类型：全表扫描、单列索引、多列联合索引、索引覆盖扫描
- 所有 27 条路径结果摘要均与全表扫描基线一致
- 索引搜索主模式：`greedy`
- 审计真值：`exhaustive`（2,048 个子集）
- 贪心评估：31 次
- 选中索引：`idx_customers_region_segment_id`、`idx_products_category_price_id`
- `E_SEARCH`：无（两种搜索结果一致）
- `E_BIAS`：4 条查询均触发，均归因为 `selectivity_error`
- 快照版本/方案版本：`1 / 1`
- 快照内容摘要：`1fdec87ed03ae5da3cbee908750174cd500ad95d65ce0e27ad78556f9ed7aa47`
- 估算内容摘要：`c3552a16bb71d374933aac2a4e27360193563aea66c485598f66115925e6ac9f`
- 两个批次摘要前缀：`1af3420bfbc6`、`ed104dbe750a`

查询结果摘要：

| 查询 | 选中路径 | 估算代价 | 估算行 | 实际行 | 实测中位耗时 ms | 累计耗时 ms | 摘要一致 |
|---|---|---:|---:|---:|---:|---:|---|
| `Q_status_region` | `idx_customers_region_segment_id:cover` | 102,868 | 569 | 3,616 | 5.112413 | 10.224826 | 是 |
| `Q_product_category` | `idx_products_category_price_id:cover` | 24,514 | 1,112 | 1,118 | 10.240772 | 20.481543 | 是 |
| `Q_amount_range` | `scan:customers` | 192,117 | 584 | 3,570 | 63.699613 | 127.399226 | 是 |
| `Q_customers_range` | `idx_customers_region_segment_id:composite` | 11,637 | 446 | 508 | 2.210808 | 4.421615 | 是 |

耗时口径：`time.perf_counter_ns` 包住 `cursor.execute()` 与 `fetchall()`；
正式值取一次预热后两次测量的中位数，累计耗时是两次测量之和。结果摘要由完整
行集合规范化排序后计算 SHA-256，因此与返回行序无关。

偏差指标：

| 查询 | 校准代价比 | 行数比 | 归因 |
|---|---:|---:|---|
| `Q_status_region` | 0.718532049314 | 0.157356194690 | `selectivity_error` |
| `Q_product_category` | 0.085481722151 | 0.994633273703 | `selectivity_error`（代价比 fallback） |
| `Q_amount_range` | 0.107701259195 | 0.163585434174 | `selectivity_error` |
| `Q_customers_range` | 0.187967075865 | 0.877952755906 | `selectivity_error` |

## 独立复核

完整账本：

```bash
python -m qplan.cli verify \
  --spec examples/sample_spec.json \
  --ledger "$WORKSPACE/qplan_ledger.sqlite"
```

实测结果：

```json
{
  "code": "OK",
  "declaration_session_field_differences": {},
  "execution_integrity_issues": [],
  "first_mismatch": null,
  "partial": false,
  "paths_checked": 27,
  "plan_version": 1,
  "queries_checked": 4,
  "scope": "recomputed estimates from declaration plus archived statistics only; generated SQLite data was not read",
  "session_config_source": "ledger snapshot for stats/search/weights/execution/bias; declaration for tables and queries",
  "session_id": "161a776059a545bea17036fe57959c72",
  "skipped_queries": [],
  "snapshot_digest": "1fdec87ed03ae5da3cbee908750174cd500ad95d65ce0e27ad78556f9ed7aa47",
  "snapshot_version": 1,
  "status": "VERIFIED",
  "unknown_batches": [],
  "verification_model_version": "qplan-independent-verify-v1"
}
```

为验证“只读账本与声明，不读样本数据库”，将账本用 SQLite backup API
复制到隔离目录，只复制声明 JSON，不复制 `sample.sqlite`：

```text
/tmp/qplan-verify-isolated.OISbnj/ledger.sqlite   520192 bytes
/tmp/qplan-verify-isolated.OISbnj/spec.json          7092 bytes
```

在该目录执行 `verify` 仍返回 `OK / VERIFIED`，检查 4 条查询、27 条路径估算。

## 强制终止与恢复

命令：

```bash
python -m qplan.cli run \
  --spec examples/sample_spec.json \
  --workspace "$CRASH_WORKSPACE" \
  --batch-size 2 \
  --crash-after-batches 1
```

本次工作目录由 `mktemp -d /tmp/qplan-crash-final.XXXXXX` 创建，实测为
`/tmp/qplan-crash-final.d9hE6Z`。

进程在第 1 个批次提交后以退出码 `75` 强制退出。恢复：

```bash
python -m qplan.cli recover --ledger "$CRASH_WORKSPACE/qplan_ledger.sqlite"
```

实测恢复结果：

```json
{
  "code": "E_RECOVERY",
  "status": "recovered_partial",
  "committed_batches": [{"batch_no": 1, "content_digest": "1c3e39e5c7e83c8af4715d1bd3584ec67cdbaaeb48fa71398a183d81b2ff627b"}],
  "unknown_batches": [{
    "batch_no": 2,
    "query_ids": ["Q_amount_range", "Q_customers_range"],
    "status": "unknown",
    "reason": "not committed before process termination"
  }]
}
```

- 不带 `--allow-partial` 的 `verify` 返回 `E_RECOVERY`，退出码 2。
- 带 `--allow-partial` 的 `verify` 检查已提交的 2 条查询、18 条路径，返回 `OK / VERIFIED`；两条未提交查询列入 `skipped_queries`。

## 失效与替换

自动测试中各执行两次小规模完整管线：

1. 修改 `users.region` 选择性提示后重新运行。
2. 旧快照版本从 `1` 变为 `stale`，新快照/方案版本递增到 `2`。
3. `Q_events`、`Q_users` 均列入受影响查询，原因包含 `selectivity_hint_changed:users.region`。
4. 当前视图仅包含新快照的新会话和两个查询，没有合并旧快照结论。
5. 新账本通过独立复核。

另一测试篡改批次 JSON 中某个路径代价但不更新批次摘要：

- `verify` 返回 `E_LEDGER`。
- 再次运行管线时检测到旧账本存储摘要损坏，创建新快照并设置替换原因 `snapshot_or_digest_mismatch`。
- 替换后的账本通过独立复核。

## GUI 状态

代码编译通过，CLI 的 GUI 入口在当前缺少 Tk 动态库的环境中返回：

```json
{
  "code": "E_GUI",
  "location": "tkinter",
  "message": "Tkinter is unavailable in this Python environment: libtk8.6.so: cannot open shared object file: No such file or directory. Install the OS Tk libraries and retry."
}
```

Windows 验收应在具备 Tcl/Tk 的 Python 3.11+ 上启动 `run_gui.bat`。窗口包含 Summary、Queries、Index Search、Diagnostics、JSON 和 Ledger 六个分页。
