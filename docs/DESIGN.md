# 技术设计

## 1. 处理模型

一次运行分为以下稳定阶段：

1. 读取 UTF-8 JSON、拒绝重复键、校验源/规则/派生/预算，构造规范化 `SessionBundle` 和摘要。
2. 按固定种子生成每个源的原始事件，按 `(source_local_index, source_id)` 轮转写入全局序号；乱序比例通过源内时间戳重排实现并再次校验观测回归率不超过声明 ppm。
3. 预规划全部窗口。基础窗口起点为 `origin + floor((t-origin)/width)*width`；派生窗口与父窗口同起点，终点为父窗口终点加声明偏移。窗口计划先于执行写入账本，用于崩溃后未知窗口标注。
4. 按每个事件更新所有适用分区水位，并执行无依赖窗口的并行定稿。派生窗口只有在父窗口完整提交后才进入可定稿集合。
5. 已定稿窗口收到迟到事件时，从原始事件、原定稿触发序号和当前游标重建窗口，按事件序号稳定裁决预算。
6. 接纳导致内容变化时，构造依赖闭包、稳定拓扑重算，并把轨迹嵌回每个变化窗口的同一最终事务。
7. 流尾以 EOS 逻辑水位排空；`fixed_step` 逐级追赶，其余策略直接推进到可排空全部计划窗口的水位。

## 2. 水位策略

所有分区从 0 开始，事件只推进其所属源/规则分区水位。

- `source_max`：全局水位为所有分区水位最大值。
- `partition_min`：全局水位为所有分区水位最小值。
- `fixed_step`：目标为分区最小值，全局每次最多推进 `step_ms`。

窗口定稿条件统一为：

```text
window.end_ms <= global_watermark - rule.allowed_lateness_ms
```

并附加“父窗口已定稿”约束。EOS 轨迹单独标记 `phase=eos`，不伪装成普通事件触发。

## 3. 同事件流不变性

- 源声明在公共摘要和事件生成中按 `source_id` 排序。
- 全局事件序号由 `(local_index, source_id)` 决定，不依赖 JSON 参数顺序。
- 窗口键为 `rule_id | canonical_quote(partition_json) | start_ms`。
- 定稿批按 `(end_ms, rule_depth, partition_label, start_ms, window_id)` 排序；任务可并行执行，但提交与定稿序号按该稳定序完成。
- 位移闭包按 `(start_ms, rule_depth, rule_id, window_id)` 稳定拓扑排序。
- SQLite 查询和界面投影均带显式 `ORDER BY`。

因此哈希/字典遍历、线程完成先后、声明数组顺序不会改变账本内容或定稿次序。内部只存在“规范化事件流”的概念；外部到达顺序必须先映射回稳定事件序号。

## 4. 迟到预算裁决

每个窗口初始预算为：

```json
{"event_count_limit": 2, "absolute_measure_limit_fixed": 1000000}
```

对定稿后到达且事件时间仍落在 `[start_ms, end_ms)` 的事件，按 `(event_seq)` 升序：

```text
accept iff remaining_count >= 1 AND remaining_abs >= abs(fixed(measure))
```

恰好把任一剩余量减到 0 时本条仍接纳；下一条在对应资源为 0 时稳定丢弃。拒绝原因区分：

- `discarded_after_event_count_budget`
- `discarded_after_absolute_measure_budget`

证据登记窗口标识、事件序号、剩余计数、剩余绝对量、累计拒绝绝对量。预算覆盖优先级为 `window_overrides > rule_overrides > default`。

## 5. 依赖与增量重算

依赖图有两类边：

- `derivation`：父窗口到同分区同起点子窗口，子终点从父结论派生；
- `shared_partition_successor`：同一规则、同一分区键序列中的相邻后继窗口。

从变化根开始取直接与间接后继闭包。对闭包内已定稿窗口重算，闭包外窗口绝不打开、不写、不升级版本。闭包内版本只在结论摘要变化时递增；未变化项仍列入位移集合并标注 `changed=false`。每条步骤包含：

- 根到该窗口的 `path`；
- 每段 `basis`（派生 ID 或共享分区依据）；
- 重算前后的内容签名和版本。

位移轨迹同时写入每个实际变化窗口的 `content_json.displacement_trajectory`，和该窗口统计、迟到证据、误差、进度游标处于同一 SQLite 事务。

## 6. 整数定点

会话缩放因子 `scale ∈ [0,6]`，量子 `q = 10^-scale`：

```text
fixed(value) = trunc_toward_zero(value * 10^scale)
```

原始整数量度和权重按各自十进制位数先转成定点整数。累加使用 int64 饱和；均值和加权均值除法向零截断。规则只输出其声明集合中的聚合字段。

每窗口误差界：

```text
quantization_truncation = included_event_count              # 每值少计 < 1 fixed unit
late_admission          = Σ abs(accepted_late.measure_fixed)
boundary_attribution    = Σ abs(measure_fixed of events t >= nominal_end)
aggregation_bound       = quantization + late + boundary
mean_bound              = trunc(aggregation_bound / count) + 1
```

主导段采用精确整数比较：候选贡献占比需至少 `1/2`；并列时按 `quantization_truncation > late_admission > boundary_attribution` 的稳定顺序。

## 7. SQLite 账本

主要表：

- `meta`：格式版本、会话、策略、规则版本、缩放因子、规格摘要、运行状态、事件游标、恢复/恢复报告。
- `run_plans`：预提交的全部窗口计划与状态。
- `watermark_points`：初始化、逐事件、EOS 水位轨迹。
- `windows`：窗口完整 JSON、摘要、版本、定稿序号、触发事件、状态、进度游标。
- `late_evidence`：逐事件裁决的可查询副本。
- `displacement_runs` / `window_displacements`：位移链的可查询副本。
- `snapshot_records`：快照路径、创建时间、规格/内容摘要。

每个窗口使用 `BEGIN IMMEDIATE ... COMMIT`。SQLite 配置 WAL、`synchronous=FULL`、5 秒 busy timeout。`--crash-after N` 在第 N 个窗口提交后立即 `os._exit(137)`，因此可以验证真实事务边界而非模拟状态。

## 8. 快照与恢复

快照使用 SQLite backup API 生成完整副本。恢复时先计算当前与快照窗口键集：

- `added`：快照有、当前无；
- `removed`：当前有、快照无；
- `scale_upgraded`：共有窗口的 scale/rules_version/format_version 不同。

然后在一个 `BEGIN IMMEDIATE` 中删除全部窗口、计划、迟到、位移和水位表，并从 `ATTACH` 的快照逐表复制；同时整体替换 `meta`。因此不存在“部分窗口合并”。报告原因按格式版本、规格摘要、内容摘要不匹配依次归类。

## 9. 崩溃与未知窗口

运行开始时写入 `run_status=active` 与全部计划。每个窗口提交同时把计划标为 finalized。重启除完整提交记录外，只读取到 `planned` 行：

1. 为每个未提交计划写 `status=unknown`、`digest=UNKNOWN`、无内容；
2. 运行状态改为 `incomplete_recovered`；
3. 输出未知窗口列表和 `required_action=rerun`；
4. 不生成 `summary_json`，不声称存在退出报告。

## 10. 重放比较

重放要求 `run_status=complete` 且无 unknown 窗口。顺序：

1. 格式版本、规则版本、缩放因子、策略、规格摘要、事件游标；
2. 完整水位轨迹，按原始插入顺序；
3. 窗口存在性，再按窗口 ID 比较格式/规则/缩放、分区、边界、父窗口、定稿序号、触发序号/事件、策略、版本、摘要、进度游标、统计、迟到裁决、误差、位移轨迹；
4. 最终定稿顺序列表。

任一项不同即抛出 `E_REPLAY`，`first_mismatch` 固定包含 `partition_label`、`window_id`、`window_boundary`、`field`、`expected`、`actual`、`evidence`（水位项无窗口边界时对应值为空）。

## 11. 调度与套接字

`BoundedFinalizationScheduler` 启动不超过全局上限的工作线程；每个任务仅在全局槽和该分区槽同时可用时出队。等待开始为批次提交单调时钟，等待结束为实际出队单调时钟。每次尝试的真实耗时会记录；超过 `task_timeout_ms` 或抛出真实 socket timeout 时按 retries 重试。

回环探针由本地子进程监听 `127.0.0.1:0`，父进程通过真实 TCP 连接发送请求并设置 socket timeout。超时来自操作系统套接字，不使用人为 sleep 伪造；重试退避也是实际单调时间。

## 12. 界面刷新

界面模型只以 SQLite `mode=ro` 打开账本，构造稳定排序的展示投影。刷新重新读取并替换控件内容，不执行 UPDATE/INSERT/DELETE，也不调用崩溃恢复写路径。
