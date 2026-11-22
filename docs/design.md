# 技术设计

## 1. 数据流

```text
sources.json
  -> config.py            强校验与路径隔离
  -> generator.py         确定性 SQLite 样本 + 内容摘要
  -> storage.py           逐条解码、归一、内容摘要复核
  -> merge.py             来源级归约 + 稳定仲裁
  -> migration.py         A→B 迁移重判
  -> attribution.py       口径/版本/周期贡献分解
  -> invalidation.py      方案版本、失效范围、未变字节校验
  -> pipeline.py          构建完整视图并原子替换账本
  -> snapshots.py         摘要、恢复、重放差异定位
```

`cli.py` 只负责参数、业务码、输出和结构化日志；`gui.py` 复用同一套 API，不另实现业务规则。

## 2. 确定性不变量

- 所有规范 JSON 使用 `sort_keys=True`、无多余空白、`ensure_ascii=False`。
- 查询 ID、源 ID、迁移源对、比较证据、失效范围、快照差异均排序后输出。
- 证据合并先对来源排序再按仲裁键排序，不依赖输入数组顺序。
- 归因以固定因子顺序计算，使用整数基点与最大余数法分配份额。
- 实际耗时只写入 `run_metrics` 与 `logs/slowdiag.jsonl`，不进入 `current_view`。
- 样本摘要基于元数据和逐行载荷 SHA-256 的规范结构，不依赖 SQLite 文件物理字节。

## 3. 源数据库

每个源使用独立 SQLite 文件：

```sql
metadata(key TEXT PRIMARY KEY, value TEXT NOT NULL)
records(row_no INTEGER PRIMARY KEY,
        format_version INTEGER NOT NULL,
        payload BLOB NOT NULL,
        payload_sha256 TEXT NOT NULL)
```

`metadata.content_sha256` 是 `metadata`（排除自身）与有序记录摘要的 SHA-256。读取时先逐行解码并检查记录级长度/字段/数值，再核对整源摘要，因此注入字段冲突时能得到具体行号，而不是笼统的摘要错误。

## 4. 账本

`ledger/ledger.sqlite3` 主要表：

- `plan_state`：单调方案版本、源参数指纹、策略摘要。
- `current_view`：单行完整视图与视图 SHA-256。
- `source_snapshots`：源内容摘要、格式版本集合、参数指纹。
- `evidence`：归一后的逐行证据。
- `merged_conclusions`：查询×指标的采纳源、值、对照集合和仲裁依据。
- `migration_results`：每个有向源对×查询的迁移文档。
- `attribution_results`：不可迁移查询的冲突归因。
- `invalidation_marks`、`invalidation_history`：当前失效标记与历史轨迹。
- `run_metrics`、`events`：单调时钟运行指标与结构化事件副本。

`replace_view()` 在一个 `BEGIN IMMEDIATE` 事务中删除并重建当前结论相关表，然后替换 `current_view`；不会对旧结论做增量拼接。

## 5. 仲裁与审查链

来源级证据先按 `replay_count` 做稳定抽样；样本数不足时使用全部样本。中位数对偶数个值取中间两数四舍五入整数均值；截断均值两侧各截去 `floor(n × trim_percent / 100)`。

仲裁键：

```text
(-source_type_bias, -declared_format_version, source_id)
```

被否决证据保留完整候选信息。比较理由由第一个不同仲裁维度决定，因此不需要依赖排序实现细节。

## 6. 迁移阈值

对指标 m：

```text
delta_bp = round(10000 × |A_m - B_m| / max(A_m, B_m, 1))
```

`delta_bp > value_tolerance_pct × 100` 即超限。状态或任一指标冲突则为 `not_migratable`；证据行数不足 `min_evidence` 优先判为 `insufficient`。

## 7. 归因公式

三类权重均以基点表示：

- 口径：`|factor_A - factor_B| × 10000 / max(factor_A, factor_B)`。
- 版本：记录版本均值差（基点尺度）与补齐字段数差，封顶 10000。
- 周期：`|period_A - period_B| × 10000 / max(period_A, period_B)`。

份额使用整数除法和按权重降序、固定原因顺序的最大余数分配，确保总和 10000。贡献 `round(delta × share_bp / 10)` 表示千分之一指标单位。

## 8. 快照比较顺序

恢复时稳定差异项先按 `path` 排序。重放比较顺序：

1. 查询 ID 升序。
2. 指标顺序固定为 `elapsed_ms`、`scanned_rows`。
3. 指标字段：采纳源、采纳值、采纳格式版本、对照集合、仲裁依据。
4. 查询诊断字段。
5. 迁移源对、查询、状态、一致标记、冲突与阈值。
6. 归因结果。

首个差异形成 `E_REPLAY` 的 `query_id`、`metric`、`field`、`expected`、`actual`。视图替换先于异常抛出，异常详情标记 `replacement_mode=whole_view`。

## 9. 验收映射

| 要求 | 实现 | 测试/产物 |
|---|---|---|
| E_SOURCE 首错字段 | `config.py` | `tests/test_config.py` |
| 多版本编解码与 E_FORMAT | `codecs.py`、`storage.py` | `tests/test_codecs.py` |
| 稳定仲裁 | `merge.py` | `test_merge_and_attribution_are_order_independent` |
| 迁移三类判定 | `migration.py` | 完整流水线测试与 `verification/acceptance.json` |
| 冲突归因 | `attribution.py` | 贡献守恒与重复归因测试 |
| 失效传播与整体替换 | `invalidation.py`、`storage.py` | 版本 1→2、q007/q008 字节校验 |
| 快照与 E_REPLAY | `snapshots.py` | 快照恢复、篡改重放、版本差异测试 |
| 三类注入 | `cli.py`、`pipeline.py` | `tests/test_cli.py`、`verification/injection-*.json` |

## 10. 已知边界

- 这是确定性诊断工作台，不是真实 SQL 性能采集代理。
- 修改源参数后需要重新生成该工作区的确定性样本；运行前会校验源元数据与声明一致。
- 快照保存结论与源摘要，不内嵌完整源数据库；恢复重放读取快照声明所指向的本地 SQLite。
- GUI 需要可用的本地显示会话；无显示环境可直接使用 CLI 与测试。
