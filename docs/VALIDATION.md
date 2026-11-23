# 实际验证记录

验证日期：2026-11-24

## 1. 环境

| 项 | 实际值 |
|---|---|
| Python | 3.11.0rc1 |
| NumPy | 1.26.4 |
| SQLite | 3.37.2 |
| Tk | 8.6 |
| 平台 | Linux 验证容器；目标运行环境为 Windows 本地 Python |

`numpy`、`sqlite3` 可直接导入。当前验证容器的系统 Python 缺少 `libtk8.6.so`，因此 GUI 烟雾
与截图验证时临时提供了本机 Tk 动态库和中文字体；代码未依赖这些临时路径。Windows 官方
Python 安装包默认包含 Tcl/Tk，无需该变通。

执行命令：

```bash
python -m unittest discover -v
python scripts/run_acceptance.py
python scripts/gui_smoke.py --spec data/demo_large.json --ledger artifacts/gui_ledger.sqlite3 --output artifacts/gui_smoke.json --screenshot artifacts/gui.png
```

## 2. 自动化测试

结果：`Ran 23 tests ... OK`。

覆盖点：

- 重复事务、提交序号规则、回滚事务提交序号、缺失前版本、成环、查询引用、数组重排。
- 两条算法路径一致、输入排列字节一致、自事务未决版本可见、`E_DIVERGE` 精确字段定位。
- 小/大样本选路、10% 交界带稳定决胜与 `threshold_influenced`。
- 全局 3 / 每读点 2 双配额、真实多线程、等待时间线与硬上限。
- SQLite 完整轮次恢复、强制终止后的未知轮次、内容摘要篡改。
- 快照保存/加载/重放、摘要篡改、版本不一致的 `REPLACE_ALL`。
- 追加版本失效传播、pending→committed 结局传播、未受影响结论字节不变。

## 3. 规模与算法选择

### 小样本 `data/demo_small.json`

| 指标 | 值 |
|---|---:|
| 事务 / 版本 / 查询 | 8 / 8 / 6 |
| 行键 | 4 |
| 平均链长 | 2.0 |
| 唯一读点/行组合 | 6 |
| 选路 | `per_query` |
| 工作量估计 | per_query 21.0，materialized 27.3 |
| 阈值跨界数 | 0 / 3 |

结论摘要：

```text
sha256:c93a26ba214b64c0e7ad891034c1d05c8bef312321c4311fef07154d5a2feb59
```

### 大样本 `data/demo_large.json`

| 指标 | 值 |
|---|---:|
| 事务 / 版本 / 查询 | 90 / 90 / 60 |
| 行键 | 20 |
| 平均链长 | 4.5 |
| 唯一读点/行组合 | 24 |
| 选路 | `materialized` |
| 工作量估计 | per_query 360.0，materialized 176.4 |
| 相对差 | 0.51 |
| 阈值跨界数 | 2 / 3（查询数、行键数） |

未触发阈值边界也完整输出，例如：

```text
avg_chain_length actual=4.5 threshold=5.0 triggered=false max_untriggered_lt=5.0
query_count      actual=60  threshold=32  triggered=true  max_untriggered=31
row_key_count    actual=20  threshold=12  triggered=true  max_untriggered=11
```

结论摘要：

```text
sha256:bf97d75172270f09de338a7b0dcf3e1519d285b8cf20dd4772b25f421b2156a6
```

两路径比较：`OK`，比较 60 条，`materialized` 与 `per_query` 完全一致。

### 交界带样本

构造 1 行、8 版本、1 查询：

```text
per_query_work=9.5, materialized_work=10.55
relative_delta=0.09952607 <= 0.10
threshold_crossed_count=1/3
selected_path=per_query
default_path=per_query
threshold_influenced=true
reason_codes=[ESTIMATE_TIE_BAND, THRESHOLD_MAJORITY]
```

完整计划见 `artifacts/tie_band_plan.json`。

## 4. 输入排列确定性

同一逻辑大样本分别以：

- 原始 JSON：`data/demo_large.json`
- 乱序 JSON：`data/demo_large_shuffled.json`
- 乱序 NDJSON：`data/demo_large.ndjson`

加载与判定后，三者的规格摘要均为：

```text
sha256:76bdafee83d9f7744a1128a4f627cde0d87574c60b6893059f97fa3d31c8bda7
```

三者的结论摘要均为：

```text
sha256:bf97d75172270f09de338a7b0dcf3e1519d285b8cf20dd4772b25f421b2156a6
```

规范 JSON 结果字节比较：`byte_identical=true`。

## 5. 并行调度

### 大样本生产诊断

- 物化任务 24 个，线程数 4。
- 全局配额 4，每读点配额 2，`constraints_ok=true`。
- 20 个任务经历配额等待，最大等待 4.453 ms。
- 每个读点分配 4 个任务；等待数分别为 RP001/RP002 各 2，RP003-RP006 各 4。

判定任务很短，实测实际并发峰值为 1；这不代表串行，线程名/ID 与线程数证明任务真实进入
工作线程池。需要用受控阻塞任务才能稳定观察并行峰值，因此另有下述探针。

### 双配额压力探针

16 个受控任务，全局上限 3、每读点上限 2：

```text
observed_peaks.global = 3
observed_peaks.per_read_point = {RP0: 2, RP1: 2, RP2: 2, RP3: 2}
thread_count = 3
blocked_task_count = 13
constraints_ok = true
```

这同时验证了全局和每读点两类硬上限。

## 6. 账本、强制终止与恢复

先提交 `acceptance-small` 与 `acceptance-large` 两轮，账本游标为 2；随后在
`--crash-before-commit` 模式中登记 pending 后调用 `os._exit(91)`。

实际结果：

```text
forced termination returncode = 91
recovery code = OK_RERUN_REQUIRED
latest complete round = acceptance-large
latest complete format_version = 1
latest complete conclusion_count = 60
ledger_cursor = 2
unknown round = acceptance-crashed-round
unknown status = UNKNOWN_RERUN_REQUIRED
unknown query count = 60
```

最近完整轮次结论摘要：

```text
sha256:bf97d75172270f09de338a7b0dcf3e1519d285b8cf20dd4772b25f421b2156a6
```

未提交轮次没有产生可读的半成品结论；查询集合被明确要求重跑。完整内容见
`artifacts/recovery_report.json`。

## 7. 快照与重放

保存的快照 identity：

```text
snapshot_id              = acceptance-session-snapshot
transaction_set_version  = v1:txset:ee2adf7a723b879d230ff8590a9e1326bf409723acd763c721bc1707242a38f5
chain_digest             = sha256:863aade07571e7aa0f75d16b4998952fa05b88241b5ba2f6ab058e161672798e
budget_version           = v1:budget:0efa9365da137f6170d35aa5e593ba24f1c7a0fff69a6e78f6bce4afc58759f7
read_point_set_version   = v1:rpset:ae47986b6ae7331eddf7953e2a3baa955b5f42b69c99ad129251914c4dd7c9c1
conclusion_digest        = sha256:bf97d75172270f09de338a7b0dcf3e1519d285b8cf20dd4772b25f421b2156a6
ledger_cursor            = 2
content_digest           = sha256:ab9709c7925b8dc9acca134c26554ca2d0e3048a69e570477351d3c676ca0f69
```

重放 60 条查询，期望与实算摘要一致，业务码为 `OK`。

注入摘要不符后返回：

```text
code = E_SNAPSHOT
anomaly = SNAPSHOT_DIGEST_MISMATCH
```

测试同时验证：identity 不一致时要求 `REPLACE_ALL`；显式整体替换后读取快照自身的整数游标，
而非增量合并当前状态。

## 8. 失效传播与整体替换

对小样本 `row:001` 追加版本 3：

- 受影响：`Q0001`, `Q0005`
- 未受影响：`Q0002`, `Q0003`, `Q0004`, `Q0006`
- 证据条数：2，按稳定键排序
- 未受影响结论逐字节一致：`true`
- 整体替换结果数：6
- 新结论摘要：`sha256:fe09979ef2974294a08261b5c511512a555afb7151e02d8b4f2f2e1cc6f25551`

另以内部 pending 事务验证“未决 → committed”传播，受影响查询正确重算为可见版本 1。

## 9. 业务码注入结果

| 场景 | 业务码 | 定位 |
|---|---|---|
| 提交序号 9 < 开始序号 10 | `E_SPEC` | `T0001.commit_seq` |
| 版本环 1↔2 | `E_CYCLE` | `row:001`, cycle `[1,2]` |
| 两路径可见版本 5 vs 999999 | `E_DIVERGE` | `Q0008`, `row:004`, `visible_version` |
| 快照内容摘要不符 | `E_SNAPSHOT` | `SNAPSHOT_DIGEST_MISMATCH` |
| 崩溃恢复 | `OK_RERUN_REQUIRED` | `acceptance-crashed-round` |

## 10. Tk 界面结果

GUI 烟雾结果 `artifacts/gui_smoke.json`：

```text
code = OK
status = 业务码: OK
selected_path = materialized
query_count = 60
version_chain_root_count = 20
schedule_trace_count = 24
budget_constraints_ok = true
```

界面截图：

- `artifacts/gui.png`：计划、阈值边界、规模与一致性
- `artifacts/gui_chains.png`：版本链树与详情
- `artifacts/gui_reasons.png`：读点快照线、查询、可见版本与逐步排除理由
- `artifacts/gui_schedule.png`：全局/读点预算条和真实调度时间线
- `artifacts/gui_state.png`：版本识别、内容摘要、账本与两路径一致性

截图由 Tk 实际渲染窗口后抓取，不是静态示意图。

## 11. 复核产物

| 产物 | 内容 |
|---|---|
| `artifacts/validation_report.json` | 本轮全部业务码、计划、摘要、调度、恢复与异常注入 |
| `artifacts/diagnostic_result.json` | 大样本 60 条完整判定与调度轨迹 |
| `artifacts/conclusions.ndjson` | 每条查询一行 NDJSON 结论 |
| `artifacts/recovery_report.json` | 崩溃后的完整恢复对象 |
| `artifacts/session_snapshot.json` | 带内容摘要与版本 identity 的快照 |
| `artifacts/tie_band_plan.json` | 交界带稳定决胜计划 |
| `artifacts/injected_cycle.json` | 成环注入样本 |
| `artifacts/injected_snapshot.json` | 摘要不符注入快照 |
| `artifacts/injected_spec.json` | 非法规格注入样本 |
| `artifacts/gui_smoke.json` | Tk 界面运行结果 |

注意：结论摘要与规格摘要可逐次复现；账本轮次内容摘要和快照内容摘要包含墙钟时间或实际调度
时间，因此每次运行可能不同，但恢复时都会重算并校验自身摘要。
