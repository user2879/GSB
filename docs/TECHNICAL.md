# 技术设计与口径

## 1. 状态模型

主库日志由 `master_log` 表持久化，主键为连续提交序号。SQLite `CHECK` 与插入触发器共同保证：

- `commit_seq` 从 1 开始连续；
- `txn_id` 全局唯一且非空；
- `commit_ts_ms >= 0`；
- `txn_bytes > 0`；
- 回放游标不得超过当前日志末尾。

应用层在写库前校验完整批次。任何一条非法即整批抛出 `E_LOG`，携带实际出错序号；不会出现“合法前缀已写入、非法后缀失败”的半批状态。

配置中的回放参数与估算口径共同生成 `param_version`。参数变化后创建独立运行和样本集合，窗口只在一个 `param_version` 内聚合。

## 2. 确定性时钟与回放

### 2.1 时钟

- 内部单位：整数纳秒。
- 时钟类型：`virtual_monotonic_ns`。
- 原点：`clock_base_ms`。
- 主库提交时间与从库完成时间使用同一会话原点。
- 真值：`replay_samples.replay_completed_ns`，每条记录独立持久化。
- 展示：完成毫秒由纳秒整数格式化为三位小数，不参与计算。

使用虚拟单调时钟是历史重放逐字节一致的必要条件。真实 `time.monotonic_ns()` 只能在一个进程内单调，无法在重启后复现同一完成值。配置、快照和报告均声明该口径，避免将虚拟时刻误认为 wall clock。

### 2.2 批次回放

设：

- `R = rate_rps`，聚合记录/秒；
- `C = concurrency`，单批最大记录数；
- 第 b 个批次（从 0 开始）覆盖全局位置 `[b*C, (b+1)*C)`；
- 批次结束位置 `e = min((b+1)*C, N)`；
- 批次内全部记录在批次屏障完成。

```text
completed_ns(b) = base_ms * 1_000_000 + ceil(e * 1_000_000_000 / R)
delay_ns(seq)   = completed_ns(b) - commit_ts_ms(seq) * 1_000_000
```

完成时刻只依赖全局结束位置，不依赖进程启动、恢复点或已提交批次数，因此强杀后续跑仍可得到原来的样本。

并发度对误差的影响通过实际完成簇体现：同一批记录共享 `completed_ns`，从而形成 `K_obs` 大小的簇。聚合器不读取并发元数据，只读取真值样本，保证“同样本集合、不同并发元数据”逐字节一致。

## 3. 窗口与定点

窗口按提交序号升序，每 `window_size` 条记录为一窗；最后一个窗口允许不足。样本先按提交序号规范化排序，再进入聚合。

缩放因子：

```text
S = 10^scale_power
```

发布单位为一个定点单位 `1/S ms`，截断方向为 `floor`。

设窗口保留样本为 `d_i` 纳秒：

```text
mean_fixed = floor(sum(d_i) * S / (m * 1_000_000))
max_fixed  = floor(max(d_i) * S / 1_000_000)
p95_fixed  = floor(ordered[ceil(0.95*m)-1] * S / 1_000_000)
```

`m` 是截断后保留的样本数。

### 3.1 三层量化上界

1. 单样本量化：整数纳秒样本落到定点毫秒网格时，误差严格小于 `1/S ms`，报告保守上界 1 个定点单位。
2. 窗口聚合：均值先保持精确整数有理数，最后只做一次下取整，窗口聚合上界为 1 个定点单位；P95/最大值从精确纳秒样本取值后同样只做一次下取整。
3. 分位数取值：P95 的公开定义是 `nearest_rank_ceil`。为了与连续 Type-7 分位区间对齐，计算选中次序统计量到 Type-7 插值区间的距离，作为取值差上界，并计入定点段。该值由实际相邻样本决定，不使用浮点。

## 4. 采样截断与不可判定

窗口保留提交序号最小的 `K = max_samples_per_window` 条样本，尾部截断数量为 `omitted = n - m`。设显式延迟上界为 `L` 纳秒：

- 均值采样上界：`ceil(omitted * L * S / (n * 1_000_000))` 定点单位；
- 最大值采样上界：`ceil((L - max_kept) * S / 1_000_000)`；
- P95 采样上界：保守取 `ceil(L * S / 1_000_000)`。

均值上界基于“未观察样本均落在 `[0,L]`”的最坏情况；最大值基于完整最大值不小于已观察最大值；P95 使用完整取值域的保守绝对差。它们不声称是最紧界，只声称是可审计上界。

以下任一条件触发 `UNDECIDABLE`：

- `MISSING_LATENCY_SAMPLE`：窗口期望序号缺样本；
- `SAMPLE_COUNT_BELOW_MIN`：保留样本数小于 `min_samples`；
- `DUPLICATE_LATENCY_SAMPLE`：同序号重复；
- `UNEXPECTED_COMMIT_SEQ`：样本序号超出运行日志域；
- `LATENCY_CEILING_EXCEEDED`：实际延迟超过声明上界。

不可判定窗口不发布统计值，所有统计字段为 `null`，并保留 `expected_commit_seqs`、`observed_commit_seqs`、`missing_commit_seqs`、计数、阈值与截断策略。

## 5. 误差分段与归因

每个统计量分别输出：

```text
sampling_truncation          = 第 4 节采样上界
fixed_point_truncation       = 最终下取整上界 + P95 取值差上界
replay_rate_discretization   = ceil(K_obs * ceil(1_000_000_000 / R) * S / 1_000_000)
total_upper_bound            = 三段之和
```

`K_obs` 是当前窗口真实样本里，相同 `replay_completed_ns` 的最大簇大小。该定义把并发度的影响绑定到已持久化真值，而不是未验证配置。

每段输出整数上界与向下取整的 ppm 占比。主导规则：

```text
若 max_segment * 100 >= total * dominant_threshold_pct，则为主导段；
否则为 MIXED。
```

并列时按固定段顺序 `sampling_truncation`、`fixed_point_truncation`、`replay_rate_discretization` 稳定决策。报告不包含参数版本、并发度、时间戳等易变元数据，最终指纹只覆盖规范化估算结论。

## 6. SQLite 原子性

批次生命周期：

1. 事务 A：插入 `replay_batches(status='RUNNING')`，确保未完成批次可见。
2. 计算该批样本（确定性，无数据库写入）。
3. 事务 B：更新批次为 `COMPLETE`，插入全部样本，更新进度游标与完成批次数。

若进程在 A、B 之间被强杀，数据库中只有 `RUNNING` 批次，没有该批样本和游标推进。重启时：

- `RUNNING -> UNKNOWN`；
- 对应进度 `-> REQUIRES_RERUN`；
- 下一个 attempt 从最后完整游标继续。

强杀后的退出报告不可获得，恢复报告显式设置 `exit_report_available: false`。

## 7. 快照

快照 envelope 版本：`snapshot_format_version=1`；DB schema：1；估算口径：`fixed-window-v1`。内容包括：

- 当前日志计数、首尾序号与规范内容 SHA-256；
- 回放参数、参数版本、估算口径与 digest；
- 进度、完整批次、逐条样本；
- 样本序列与摘要、统计/误差结果及摘要。

保存时原子写临时文件后替换，最后设置只读。恢复流程在单个事务内：

```text
DELETE replay_samples
DELETE replay_progress
DELETE replay_batches
DELETE replay_runs
INSERT snapshot run / batches / samples / progress
```

不做逐行 upsert，因此旧当前状态不会残留。报告包含每个字段替换前后的计数、序号范围、进度、只读状态和剩余快照文件。

日志摘要不匹配但样本序号仍兼容时，恢复可完成但状态为 `RESTORED_STALE_LOG_MISMATCH`。这是“整体替换”而不是“宣布可复现”。若快照进度超前当前日志或样本序号不存在，事务内校验失败，不会留下部分状态。

## 8. 历史重放

校验顺序固定：

1. 快照 envelope、DB schema、估算版本与 digest；
2. 回放参数版本与估算绑定；
3. 当前日志 digest；
4. 用保存参数重新构建批次计划；
5. 逐提交序号比较 `replay_completed_ns`、`delay_ns`、`fixed_delay`；
6. 重新聚合，比较完整 estimation 对象（窗口、统计、不可判定依据、三段误差、总上界、主导段、指纹）；
7. 比较样本序列摘要。

样本差异报告 `commit_seq + sample.<field>`；统计/误差差异报告字段路径。参数、格式版本或日志摘要不一致返回 `E_REPLAY` 并列出具体依据。

## 9. 确定性与稳定性

- 所有数值结论使用整数运算。
- 规范 JSON 使用 `sort_keys=True`、紧凑分隔符和 UTF-8。
- 输入样本排序后再分窗；参数版本分隔旧样本。
- 误差占比用整数 ppm。
- 输出树/Tkinter 列表按序号、窗口号、更新时间+参数版本或文件名排序。
- 历史重放不读取当前回放元数据以外的可变进程时钟。
