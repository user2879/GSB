# 设计与计算口径

## 1. 确定性记录流

服务声明只保存生成参数，记录在运行时由整数种子、记录序号和字段名生成：

```text
u64(seed, index, field) =
    first_8_bytes(SHA-256(seed || 0x1F || index || 0x1F || field))
```

每个字段独立取值，因此记录不依赖 Python `random` 的状态或执行历史。追踪标识由 `service_id/seed/index` 的 SHA-256 前缀生成。时间戳从声明的起点开始，按每条记录的整数步长累加；步长非负，所以服务内时间戳单调不减。

记录流摘要定义为一个规范化 JSON 对象一行，以 UTF-8 编码并用换行连接：

```text
stream_bytes = join(canonical_json(record_i) + "\n")
stream_digest = SHA-256(stream_bytes)
```

规范化 JSON 使用键排序、紧凑分隔符和 UTF-8，不依赖字典插入顺序。相同种子、记录数和字段声明必须得到相同 `stream_bytes` 和摘要。

## 2. 分区采样

客户端键按以下输入计算稳定分区：

```text
bucket = first_8_bytes(SHA-256(
    "trace-diag-partition-v1" || strategy_version || scale_power || client_key
)) mod 10000
```

判定顺序：

```text
head_hit = bucket < head_ratio_bp
tail_hit = (not head_hit)
           and duration_us >= tail_duration_threshold_us
           and bucket < tail_ratio_bp
```

头部和尾部互斥。采样摘要保存头命中数、尾命中数、跳过数、完整逐条判定摘要和采样集合摘要。重放会重新生成全部判定并比较摘要。

## 3. 定点表示

策略中的 `scale_power = p` 定义：

```text
quantum_us = 10^p
fixed(d) = floor(d / quantum_us)
value_us = fixed(d) * quantum_us
```

截断方向为负无穷；输入耗时非负，因此等价于向下截断。单条记录最大量化损失为 `quantum_us - 1`。

分位数使用有序定点刻度的 type-7 秩插值：

```text
position = (n - 1) * percent / 100
tick = tick[floor(position)] * (1 - frac(position))
     + tick[ceil(position)]  * frac(position)
fixed_quantile = floor(tick) * quantum_us
```

结果始终是 `quantum_us` 的整数倍。

## 4. 等深分位桶

对排序后的耗时 `s[0..n-1]` 和桶数 `k`，内部切点为：

```text
rank_i = ceil(i * n / k)              i = 1..k-1
cut_i  = s[min(n-1, rank_i - 1)]
```

第 `j` 桶为：

```text
j = bisect_right(cuts, duration_us)
bucket_j = [lower_j, upper_j_exclusive)
```

其中第一桶下界为最小值，最后一桶上界为最大值加一。值等于切点时归入右侧桶。由于整数耗时不会落在两个整数切点之间，浮点插值只存在于分位值计算，不参与桶成员判定。重复切点会产生空桶，但归属完全由值和切点决定。

## 5. 重标定与归因

记号：

```text
n       全量记录数
H       总采样命中数
r       n / H
S_b     桶 b 的全量真实耗时和
A_b     桶 b 命中记录的真实耗时和
Q_b     桶 b 命中记录按定点粒度重建后的耗时和
h_b     桶 b 命中数
M_b     桶边界中点
```

最终桶估计取“命中数 × 桶中点 × 重标定系数”：

```text
E_b = h_b * M_b * r
```

误差被精确拆为：

```text
sampling_sparse = A_b * r - S_b
fixed_truncation = (Q_b - A_b) * r
bucket_boundary = (h_b * M_b - Q_b) * r

E_b - S_b = sampling_sparse + fixed_truncation + bucket_boundary
```

对应保守上界：

```text
sampling_bound  = |sampling_sparse|
fixed_bound     = h_b * (quantum_us - 1) * r
boundary_bound  = sum(|fixed(d_i) * quantum_us - M_b| for i in sampled bucket) * r
total_bound     = sampling_bound + fixed_bound + boundary_bound
```

三角不等式保证 `|E_b - S_b| <= total_bound`。总计值是逐桶有符号贡献和与逐桶上界和，故归因结果不依赖记录注入顺序或桶遍历顺序。

占比定义为各段上界除以总上界。主导段必须同时满足：

```text
share >= 1/2
share > share(other segment) 对所有其他段成立
```

否则为 `balanced`；总上界为零时为 `none`。

分位估计使用重标定后的桶占用权重 `h_b * n / H`。找到目标秩所在桶后，以该桶中点重建采样估计分位值。

令重标定 CDF 相对全量 CDF 的最大绝对秩偏差为 `D`：

```text
D = max_i | sum_{b<=i}(h_b * n / H - count_b) |
```

目标秩 `(n-1)*p/100` 的可能带为 `[floor(rank-D), ceil(rank+D)]`。分位估计的采样秩界取估计值到该带两端距离的较大者减去选中桶半宽。总界为：

```text
sampling_rank_bound + 2*(quantum_us-1) + selected_bucket_half_width
```

`2*(quantum_us-1)` 同时覆盖全量分位与估计分位的定点截断，桶半宽覆盖中点归属；因此报告中的 `estimated_value_in_error_bound` 使用严格最坏带上界。

## 6. 判定与翻转

基准状态中：

- 每个服务按估计均值选择候选策略，平局依次用策略版本、策略标识决胜。
- 服务按选中策略的估计 p95 排序，平局依次用服务标识、策略版本、选中策略版本、选中策略标识决胜。
- 任意两服务的 p95 差、或任意两候选策略的均值差不超过双方误差界之和时，标记 `affected_by_error_bound=true`。

参数敏感性扫描对每条策略分别尝试：

```text
scale_power - 1, scale_power + 1
bucket_count - 1, bucket_count + 1
```

只改变一个参数并保持版本不变。扫描后重新计算完整统计、排序和策略选择。若服务顺序或任一服务的候选策略选择改变，就产生翻转事件。事件包含翻转前后的服务顺序、所选策略、p95 误差界、均值胜者区间和触发参数，并按 `(policy_id, parameter, old_value, new_value, flip_key)` 稳定排序。

## 7. 账本

SQLite 账本格式版本为 `1`。核心表：

| 表 | 内容 |
|---|---|
| `meta` | 格式版本、创建时间 |
| `runs` | 样本/策略版本、文档快照、状态、活动/陈旧标记、进度、快照摘要 |
| `batches` | 每服务批次状态、内容摘要、进度游标 |
| `sample_summaries` | 流摘要、采样摘要、每策略结果摘要 |
| `bucket_stats` | 每服务/策略/桶的边界、计数、命中与桶摘要 |
| `error_bounds` | 定点界、分位、均值界 |
| `attribution_chain` | 每桶及总计的三段贡献、上界、占比、主导段 |
| `flips` | 误差界内判定与参数翻转记录 |

写入协议：

1. 新 run 在一个事务中插入，并把旧 run 全部标为 `stale`；批次为 `pending`。
2. 一个服务的所有表和 `committed` 状态、批次摘要、run 游标在同一事务中提交。
3. 所有批次完成后，计算只包含不可变内容（文档、版本、批次载荷、翻转和进度）的快照摘要，并在收尾事务中激活新 run。
4. 旧 run 的 `active=0, stale=1` 不改变其快照摘要，因此仍可审计和重放，但不会进入当前视图。
5. 重启时 `pending` 被改为 `unknown`，run 被改为 `recovering`。恢复只跳过 `committed`，重算并提交 `unknown`。

“当前视图整体替换”通过查询唯一 `active=1, stale=0, status='complete'` 的 run 实现，不合并旧 run 的内存数据或数据库行。

## 8. 深重放

重放器不信任账本保存的统计值，而是：

1. 读取账本保存的服务和策略文档。
2. 按记录的样本/策略版本重新生成记录流。
3. 按每条策略的 `scale_power`、`bucket_count` 和采样参数重算摘要、分位、逐桶统计、归因与翻转。
4. 逐项与账本比较；按服务声明顺序、策略/桶稳定顺序返回首个不一致的服务标识和字段路径。
5. 最后重算 run 快照摘要并比较。

账本内容摘要不符使用 `E_SNAPSHOT`；统计不可复现使用 `E_REPLAY`，并带 `service_id` 和 `field`。
