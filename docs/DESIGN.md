# 技术设计

## 1. 模块边界

| 模块 | 职责 |
| --- | --- |
| `spec.py` | 本地 JSON 加载、结构校验、引用校验、规范化领域模型 |
| `chain.py` | 按环境、键、来源构建生效链并应用逐键绑定 |
| `fixedpoint.py` | 向零截断量化、64 位饱和、整数比较与中点运算 |
| `arbitration.py` | 三类仲裁、采纳/否决、误差带决胜和归因 |
| `engine.py` | 视图指纹、逐键依赖、版本递增、失效传播、逐视图提交 |
| `ledger.py` | SQLite 事务、格式版本、内容摘要、历史和陈旧记录 |
| `snapshot.py` | 快照摘要、整体替换、重放校验与剩余视图恢复 |
| `diagnostics.py` | 无写入的 UI 投影 |
| `ui.py` | Tkinter 编辑与诊断界面 |
| `cli.py` | 命令、异常注入、退出码和 JSONL 审计日志 |

## 2. 规格模型

环境文档要求 `schema=config-arbiter.environments/v1`，视图文档要求 `schema=config-arbiter.views/v1`。环境必须包含唯一 `environment_id`、非负整数 `depth`、可用来源类型集合、来源条目和可选的逐键绑定。来源必须具有环境内唯一 `source_id`、受支持 `source_type` 和整数 `priority_bias`。观测必须具有非空 `key`、有限 JSON 数字 `value`、非负整数 `observed_seq` 和 `captured_at_ms`。

视图必须包含唯一 `view_id`、参与环境、完整环境优先级排列、环境整数权重及逐键策略。策略只接受：

- `latest`
- `weighted_majority`
- `interval_midpoint`

结构、类型、唯一性和参与环境存在性失败抛 `E_SPEC`，并带首个违规路径。优先级/权重引用不存在的环境属于引用完整性错误，抛 `E_REF`。

## 3. 链与顺序无关性

同一环境、同一键的候选按以下元组排序：

```text
(-priority_bias, source_type_rank, source_id)
```

来源类型序号来自全局 `source_type_order` 的显式排列。绑定条件为来源类型属于 `required_source_types`，且 `observed_seq >= min_observed_seq`（若给出）。链上首个满足条件的步骤成为环境生效值。

所有输入列表都在规范化投影中排序；视图处理按 `view_id` 排序；候选和拒绝项按稳定环境顺序、标识和定点值排序。因此环境、来源、观测或 JSON 对象书写顺序变化不会改变结论字节。

## 4. 定点与误差模型

对输入十进制数 `x`，缩放幂 `k`，缩放因子 `S=10^k`：

```text
scaled = x × S
q = trunc_toward_zero(scaled)
q_final = saturate_i64(q)
```

量化误差以定点单位表示：

```text
e = |scaled - q_final|
e_upper = ceil(e)
```

每个键中：

```text
tolerance_fixed = saturate_i64(tolerance × S)
comparison_bound = saturate_i64(tolerance_fixed + 2 × max(e_upper))
```

`error_model` 同时记录每个候选的精确分数误差、`scale_policy_version`、`arbitration_policy_version`、截断方向和溢出规则。视图结果记录逐键比较界、逐键聚合上界和视图级聚合上界，因此仅修改口径版本也能形成新的输入指纹和视图版本。

## 5. 三类仲裁

### latest

按 `captured_at_ms` 降序、`observed_seq` 降序和环境优先级选出名义候选。所有与名义候选定点差在 `comparison_bound` 内的候选进入误差带；若超过一个，按稳定决胜键重选。最终值来自最终候选来源。

### weighted_majority

按完全相同定点值分组。每组权重为成员量化权重的饱和和。先按支持度降序、再按稳定决胜键排序；误差带内的其他分组参与稳定决胜。采纳最终分组的全部成员，权威值使用该分组中稳定序第一个来源的值。

### interval_midpoint

每个候选区间为：

```text
[q - (tolerance_fixed + e_upper), q + (tolerance_fixed + e_upper)]
```

全部区间有交集时取交集；无公共交集时选择最大重叠候选簇，并以最窄交集、稳定候选顺序决胜。权威值为交集上下界之和向零除二，并记录向零截断余量。

## 6. 版本与失效传播

视图输入指纹由规范化会话版本、视图范围、逐键口径和所有参与环境的完整规范化来源条目组成。链规则依赖 `view_scope`、`view_key_policy`、环境深度/可用来源/绑定以及该键实际引用的来源观测。

启动裁决时：

1. 指纹未变：动作 `unchanged`，不写账本。
2. 指纹变化：版本 = 当前版本 + 1。
3. 对每个键比较旧/新依赖投影，生成触发原因和变更路径。
4. 受影响键重算；未受影响键复制上一版本键结果的规范化字节。
5. 事务内将旧运行标为 `stale`，写新运行、逐键结果、依赖和 `stale_findings`。

## 7. SQLite 账本

表结构：

- `ledger_meta`：单行整数 `format_version`、内容摘要、版本上下文和时间；
- `views`：每视图当前版本、指纹、状态和结果摘要；
- `view_runs`：每视图版本完整结果、依赖清单和状态；
- `view_keys`：逐键结论、摘要、依赖和陈旧标志；
- `stale_findings`：视图/旧版本/新版本/键/原因/传播轨迹。

内容摘要对全部运行、逐键记录、陈旧记录和视图索引按规范 JSON 排序计算。每次提交在写完全部记录后计算摘要，并仍在同一事务内更新 `ledger_meta`。

## 8. 崩溃与快照恢复

SQLite 使用 WAL 和 `synchronous=FULL`。视图事务未执行 `COMMIT` 时不会留下半套结论。重启后账本仅提供完整提交的运行；声明视图若无当前指纹的完整提交，则判为 `unknown` 并要求重跑。

快照是原子写入的 JSON 文件，`content_digest` 覆盖除自身外的整个载荷，`state_digest` 覆盖嵌入的环境和视图文档。`restore` 先校验快照摘要和账本摘要，再比较当前状态与版本上下文：

- 不一致：验证快照嵌入规格，原子写回两份声明，清空账本结论，重新裁决全部视图。
- 一致：先逐视图重算并与账本比较权威值、采纳来源、比较误差界和聚合误差界；任何差异抛 `E_REPLAY`，不写账本。全部通过后才提交未知视图并刷新快照。

## 9. 安全与边界

- 不访问网络，不启动端口。
- 注入只修改加载参数或预期摘要参数，不修改账本或样本文件。
- 有限数字、布尔拒绝、64 位饱和、固定截断方向和规范序列化共同消除平台浮点差异。
- NumPy 按任务环境要求安装；核心仲裁刻意不依赖浮点数组，以保证跨平台整数可复现性。
