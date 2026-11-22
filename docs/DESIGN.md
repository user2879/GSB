# 设计与算法说明

## 模块划分

| 模块 | 职责 |
|---|---|
| `qplan/spec.py` | 声明解析、强校验、规范化与错误定位 |
| `qplan/data.py` | SplitMix64 确定性数据、SQLite 原子物化、跨平台 URI |
| `qplan/stats.py` | 精确/抽样统计、直方图、频数、Wilson 区间、选择性估计 |
| `qplan/paths.py` | 四类访问路径生成与参数化 SQL |
| `qplan/cost.py` | 访问行数/比较/回表/IO 估算及整数权重代价 |
| `qplan/search.py` | 稳定索引候选搜索、穷举/贪心切换、`E_SEARCH` |
| `qplan/executor.py` | 路径真实执行、摘要、计时、`EXPLAIN QUERY PLAN` |
| `qplan/bias.py` | 基线比、代价/行数偏差、三类归因 |
| `qplan/invalidation.py` | 内容差异、受影响查询和失效依据 |
| `qplan/ledger.py` | 快照/会话/批次事务账本、恢复和摘要校验 |
| `qplan/pipeline.py` | 端到端编排与批次提交 |
| `qplan/verify.py` | 只读独立重算与首个字段定位 |
| `qplan/cli.py` / `qplan/gui.py` | CLI 与 Tkinter 桌面端 |

## 确定性数据

每个值使用如下稳定输入：

```text
splitmix64(seed, table, column, row_index[, nullable/sample role])
```

字符串通过 BLAKE2b-64 转成整数，避免 Python 进程级字符串哈希随机化。外键值映射到目标 `unique_int` 起点与行数范围。数据库先写入同目录临时文件，事务提交并关闭后通过 `os.replace` 原子发布，避免强制终止留下半成品。

样本数据库元表记录生成指纹；表结构、行数、生成器、种子和候选索引不变时复用数据库，否则重建。

## 统计口径

### 精确统计

对每列执行完整行扫描，记录：

- 行数、NULL 数、有效值数、不同值数、最小值、最大值；
- 等宽直方图计数与比例；
- 不同值未超过 `value_count_limit` 时的完整值频数。

### 抽样统计

确定性系统抽样索引：

```text
base_i = floor(i * N / n)
index_i = sort((base_i + offset) mod N) + 1
offset = splitmix64(seed, "systematic-sample", table) mod max(1, floor(N/n))
```

`n = min(sample_size, N)`。同一表所有列使用同一批行索引，样本索引摘要进入归档。

抽样选择性的误差界使用 Wilson 区间，再乘以有限总体修正：

```text
fpc = sqrt((N - n) / (N - 1))
```

若 `n == N`，误差界为 0。

### 选择性估计

1. 等值谓词优先使用归档值频数；频数截断时使用不同值倒数，高基数抽样列结合表行数修正。
2. 范围谓词使用直方图累积分布；整数边界取半格修正。
3. 提示选择性与直方图结果混合：

```text
selectivity = 0.75 * histogram_selectivity + 0.25 * hint
```

4. 过滤条件按独立性相乘。
5. 自适应口径：`rows < row_threshold` 为精确，否则为抽样。
6. 若精确与抽样混合估计差不超过抽样误差界，仍采用自适应口径，稳定规则为“优先声明口径，再按方法名”，并标记 `tiebreak_applied=true`、`error_affected=true`。

每个谓词都保留 `exact`、`sampled`、最终口径、样本量、置信度、误差上界、直方图版本和统计口径版本。

## 访问路径

对每条查询生成：

- 每个查询表作为驱动表的全表扫描路径；
- 以查询表候选索引为前缀的单列/联合索引路径；
- 当索引覆盖该表所有过滤、联接和输出列时的覆盖路径。

驱动表可重排到 `FROM` 首位，其余表保持查询声明顺序；使用 `CROSS JOIN` 阻止 SQLite 重排。路径 SQL 通过 `INDEXED BY` 强制指定驱动表索引，全表扫描路径对驱动表使用 `NOT INDEXED`。

所有声明候选索引物理存在，以便每类路径都能真实执行和采集摘要。全局索引选择集合决定查询可使用的显式索引路径；联接探测由 SQLite 基于物理索引决定。该边界在 README 中明确说明。

## 代价模型

每条路径输出：

- `access_rows`：驱动访问路径访问行数；
- `join_probe_rows`：联接探测次数；
- `comparisons`：访问比较与联接比较；
- `lookups`：非覆盖索引回表及联接探测；
- `io_pages`：表页、索引页和回表页估计；
- `estimated_output_rows`：最终结果行估计。

索引访问行数：

```text
access_rows = N * selectivity(first index key predicates)
```

当首列没有可用谓词时为完整索引扫描。非覆盖路径按匹配行估计回表。联接选择性使用两侧不同值估计：

```text
join_selectivity = 1 / max(distinct_left, distinct_right, 1)
```

总代价：

```text
cost = w_rows * total_access_rows
     + w_comparisons * comparisons
     + w_lookups * lookups
     + w_io_pages * io_pages
```

权重必须为非负整数，`cost` 为整数。模型版本、权重、所有中间量和统计来源一并进入账本与摘要。

## 索引搜索

候选索引先按 `(表顺序, 索引名)` 稳定排序。任意子集的负载代价为各查询可选路径最小估算代价之和；平局依次比较候选集合大小和字典序，因此结果与候选声明顺序无关。

- 候选数 `<= exhaustive_max_candidates`：主搜索为穷举。
- 候选数 `> exhaustive_max_candidates`：主搜索为贪心，每步加入使总代价下降最多的索引。
- 候选数 `<= oracle_max_candidates`：额外计算穷举真值。
- 贪心与穷举结果不同：发 `E_SEARCH`，列出对称差索引，声明 `truth_source=exhaustive`，并采用穷举结果。

输出包含选中索引、未选候选及理由、两种搜索结果、评估次数和逐步贪心轨迹。

## 真实执行与偏差

每条路径在真实 SQLite 连接上：

1. 记录 `EXPLAIN QUERY PLAN`；
2. 按配置预热；
3. 重复执行并抓取全部行；
4. 对行做类型标记、规范 JSON 编码和稳定排序；
5. 计算摘要并比较所有重复执行摘要；
6. 输出实测中位耗时与累计耗时。

全表扫描基线用于：

- 校验所有路径结果摘要；
- 求全局 `unit_ms_per_cost = sum(baseline_ms) / sum(baseline_estimated_cost)`；
- 计算最优路径的校准估算毫秒与真实毫秒比、估算输出行与实际行比。

超过阈值的偏差按以下优先级归因：

1. `index_coverage_decision_error`：路径覆盖标记与 `EXPLAIN QUERY PLAN` 的 `USING COVERING INDEX` 观测不一致；
2. `selectivity_error`：驱动表估算过滤行/实际过滤行比超阈值；
3. `join_order_deviation`：计划表顺序与实际观测顺序不一致且存在行/代价偏差。

无更强证据但仍有代价偏差时归入 `selectivity_error`，并注明 fallback 依据。

## 失效与版本

比较上一快照中的规范声明与归档统计，检测：

- 表行数/物化行数变化；
- 列类型、选择性提示、列增删；
- 统计口径或配置变化；
- 候选索引集合/列序变化；
- 权重、搜索配置、偏差阈值变化；
- 查询定义增删改。

变化时创建新快照和方案版本（整数+1），逐查询记录原因。旧快照设为 `stale`，旧批次设为 `stale` 或 `superseded`；当前视图只读取最新 `current` 快照的最新会话，不合并旧查询。

预期快照版本/摘要不匹配，或运行前检测到已有账本摘要损坏时，整体替换为新版本，替换原因写入快照。

## 账本与恢复

一个批次事务插入的 JSON 包含该批查询：

- 路径规格和全部估算中间项；
- 全局索引搜索结论和查询选择；
- 每条路径真实执行摘要、耗时与计划；
- 偏差归因和结果摘要一致性。

批次摘要绑定：

```text
sha256(session_id, snapshot_version, plan_version, batch_no, payload)
```

开始运行前先提交会话清单（预期批次数和每批查询 ID）。强制终止时，未提交事务回滚；恢复只读取已提交批次，将清单中缺失的批次标为 `unknown`，不推测其结果。

## 独立复核

复核器：

1. 以声明文件校验表、查询、提示、候选索引等数据契约；
2. 从账本读取会话权重、统计口径、搜索配置和归档统计；
3. 重算每条路径的访问行、比较、回表、IO、输出行和总代价；
4. 重算索引搜索与每查询最优路径；
5. 逐字段与账本比较，首个差异返回查询 ID 和字段路径。

复核不打开样本数据库，不依赖当前进程缓存，也不修改账本。

