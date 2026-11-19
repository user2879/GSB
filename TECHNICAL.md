# 技术说明

## 1. 处理模型

权威状态由 `factions` 和 `event_cursors` 两部分组成。每个阵营状态包含资源、FIFO 队列、累计征收、累计消耗、累计生产和累计调度。队列项为 `{unit, remaining}`；tick 开始时先递减所有队列项的剩余建造 tick，再执行该 tick 事件。

事件提交序固定为 `(tick, actor, local_seq)`。合成器先在不修改状态的情况下验证所有 `local_seq` 与入口游标连续，再逐 tick 应用事件。这样 `E_SEQ_GAP` 在状态被污染前即可终止。

每个 tick 记录包含完整状态快照、各阵营仅含资源与队列的哈希、整个快照的内容哈希，以及该 tick 的参与事件键。

## 2. 随机数派生链

版本标识为 `sha256-u64-v1`：

```text
payload = "sha256-u64-v1|<seed>|<tick>|<rng_seed_delta>"
value_u64 = big_endian_uint64(SHA256(payload)[0:8])
roll_10000 = value_u64 % 10000
```

派生没有计数器、全局随机对象或线程局部状态，因此与遍历顺序、线程调度、块划分无关。账本同时保存 `seed`、`tick`、`rng_seed_delta`、`value_u64`、十六进制值和 roll，第三方可独立复算。账本加载时先复算 RNG，再检查自哈希，以便把漂移优先归类为 `E_RNG_DRIFT`。

## 3. 越界处理

事件执行前先计算候选状态：

- `collect`：候选资源超过 `resource_cap` 时拒绝；
- `produce`：候选资源小于 0 或队列超过容量时拒绝；
- `dispatch`：目标单位已就绪数量不足时拒绝。

被拒绝事件不改状态，但保留派生随机值、拒绝原因、事件序号和候选资源，随后继续审计。整体结果含 `E_RESOURCE_OVERFLOW`。预算报告按阵营给出峰值实际资源、峰值候选资源、峰值队列、累计消耗、征收、生产、调度、越界次数和首个越界事件序号。NumPy 用于峰值数组统计，缺失时使用等价标准库路径。

## 4. 分块与并行正确性

块边界只落在 tick 边界，不拆分 tick。构建清单前先做确定性边界扫描，得到每个 tick 后的状态快照。每块携带入口完整快照（阵营资源、队列和事件游标）、出口完整快照、范围内事件及入口/出口/内容哈希。

块工作线程从入口快照开始独立重放。块内仍严格串行；块间只共享只读规则和不可变块内容。出口必须与清单声明的出口哈希一致。最终归并再检查块区间连续、入口哈希等于上一块出口哈希、所有 tick 完整且无重复。

块长不整除时使用 `end = min(start + chunk_ticks - 1, tick_end)`，尾部不补事件。空 tick 也会产生状态记录，因此队列推进不会因分块改变。

`business_hash` 覆盖游戏、规则/事件哈希、所有 tick/事件/RNG/越界/预算内容，但不覆盖调度器报告、块清单哈希、文档类型和账本自哈希。不同线程数、预算降级、串行模式及不同合法块长应得到同一 `business_hash`。

## 5. 预算调度器

单块内存估算为：

```text
4096 + 4 * (events_bytes + entry_snapshot_bytes + exit_snapshot_bytes)
```

调度器采用保守的最大单块预算：`memory_capacity_workers = memory_budget // max_block_bytes`。降级优先级固定为：串行模式强制并行度 1；不超过块数量；线程上限为 0 时拒绝；内存不足单块时全部拒绝，否则内存容量低于目标并行度时降到容量值。

每次降级记录 `before_parallelism`、`after_parallelism` 和原因。拒绝时每个块都有 `chunk_id`、tick 区间、所需/可用内存和 `E_BUDGET_DENIED`。完成顺序不进入结果；合并前按 `(start_tick, chunk_id)` 排序。

## 6. 一致性核对

核对不是只比较事件条数，而是比较规则/事件/业务哈希、完整 tick 集合、每个 `(faction, tick)` 的完整状态与资源/队列哈希、整体 tick 状态哈希、入口/出口快照、逐事件业务字段与随机派生、越界记录及预算报告。

首个分歧的证据包含该 tick 的双方参与事件、所在块的入口/出口快照及并行调度降级记录。无分歧时明确输出 `FULL_MATCH` 和实际比较的 tick、阵营/tick 对、事件及 RNG 记录数。

## 7. 原子持久化与恢复

所有 JSON 文件先写入目标目录内的临时文件，`flush`、`fsync` 后用 `os.replace` 原子替换；目录 `fsync` 在 Windows 不支持时安全忽略。账本自身含 `ledger_hash`，另含 tick、阵营、RNG 和 `business_hash` 校验。退出报告最后写，内容引用账本、核对和产物清单哈希。

重跑开始时只撤下旧的退出/恢复/错误标记，不先删除已有账本。中途终止时临时文件不会被读取；完整旧账本仍可作为最近一次已提交状态恢复。`recover` 只读取 `output/parallel_ledger.json`，验证后写 `recovery_report.json`，永不重建退出报告。损坏或半写账本会返回 `E_LEDGER_HASH`。

## 8. 测试覆盖

`tests/` 覆盖同种子生成一致性、真值与 RNG 复算、tick 因果、队列推进、乱序不变性、`E_SEQ_GAP`、分块复现、尾部实际区间、块清单损坏、1/2/4/8 线程与多内存预算、`E_BUDGET_DENIED`、全量一致结论、首个分歧、`E_RNG_DRIFT`/`E_LEDGER_HASH` 区分、仅从账本恢复、孤儿临时文件忽略、损坏账本拒绝恢复，以及四场景完整样本双跑字节复现。

```bat
py -3.10 -m unittest discover -v
```

