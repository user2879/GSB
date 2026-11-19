# 设计说明

## 1. 分层

```text
shardrep.spec          严格声明解析与 DAG 校验
shardrep.content       确定性字节与摘要
shardrep.registry      物理副本状态、注入、E_INTEGRITY
shardrep.propagation   逻辑分片失效闭包与稳定轨迹
shardrep.packing       打包任务、产物版本、原子发布
shardrep.scheduler     双重预算、线程池、真实超时与重试
shardrep.repair        多数选源、修复任务、三分类报告
shardrep.snapshot      版本快照、整体替换、恢复重放比较
shardrep.ledger        SQLite claim/commit 与崩溃恢复
shardrep.audit         JSONL/SQLite 结构化事件
shardrep.gui           Tkinter 诊断台
shardrep.demo          固定三段演示
```

物理 `shard_id` 与逻辑 `logical_id` 分离。物理标识用于在节点上定位文件和记录字节；逻辑标识用于跨副本选源、引用数据块和沿镜像图传播。这样不会把“多节点副本”误当成多个不同逻辑分片。

## 2. 无部分生效的规格加载

`load_spec` 在内存中完成全部校验后才构造 `Spec`。节点与分片唯一性、正字节数、路径边界、逻辑元数据一致性、块区间覆盖、块校验摘要、关系引用与 DAG 检查任一失败都会抛出 `SpecValidationError("E_SPEC")`。错误携带首个确定性的分片标识和 JSON 路径。关系环由稳定 Kahn 算法发现，并附加 `cause_code=E_CYCLE`。

## 3. 失效传播

传播图只包含逻辑分片。边为 `source -> target`；拼接关系把每个来源连到目标。对每个种子：

1. 使用 `(距离, 路径字典序, 节点)` 最小堆求最短路径。
2. 同一距离下保留字典序最小路径。
3. 合并所有种子的轨迹，按 `(seed, path)` 排序。
4. 受影响节点按全局稳定拓扑顺序输出。
5. 包含任一受影响逻辑分片的数据块进入待修复块集合。

节点数组与关系数组的声明顺序不会影响结果；调度结果也不依赖字典迭代顺序。

## 4. 增量重打包

`build_pack_tasks` 只接受调用者给出的受影响逻辑分片，并按拓扑顺序生成每分片一个任务。节点选择采用 `(已分配任务数, node_id)` 最小者，且要求节点持有该逻辑副本并有正配额。

每个打包产物的 `content_digest` 由组版本、逻辑分片、序号、字节数和期望内容摘要决定，不包含版本号。产物文件独立保存，受影响任务仅重写自身文件；未受影响文件字节不变。清单中未受影响条目的 `content_digest` 与 `version` 也不变。

`compare_artifact_sets` 按逻辑标识排序比较 `content_digest`、`bytes`、`expected_digest`、`seq`，并比较块副本摘要。首个差异抛出 `E_INCR`，字段为具体属性名。

## 5. 并行调度与超时

调度器自行维护待运行堆和每节点运行计数，只有同时满足以下条件才提交线程池：

- 当前已提交且未完成任务数 `< global_slots`。
- 该节点当前已提交且未完成任务数 `< node_quotas[node]`。

`ThreadPoolExecutor` 的最大线程数同样设为全局槽位，形成第二道硬约束。任务完成前，其节点名额不释放；因此配额约束覆盖实际运行与排队边界。

工作函数按 8 个进度块处理字节，每块用 `time.monotonic()` 检查截止时间并最多睡到截止时间。超时返回已处理字节；调度器为超时提交账本行，再在未超过 `max_retries` 时重新入队。重试重新经过同一双重预算闸门。

每个调度条目包含排队、开始、结束的单调时钟毫秒值、排队时长、执行时长、完成字节、产物版本、摘要、预算消耗与跳过原因。全局峰值和逐节点峰值由工作线程实测。

## 6. 修复决策

对每个块、每个逻辑分片：

1. 收集全部物理副本状态。
2. 仅把长度和摘要都与声明一致的副本视为可用。
3. 若没有可用副本，登记缺失序号和全部已尝试副本。
4. 否则选择 `(node_id, shard_id)` 字典序最小可用副本作为修复源。
5. 对每个损坏/缺失目标生成修复任务，记录源、目标、字节数与期望摘要。
6. 目标节点 `repair_enabled=false` 时保留损坏副本，块降级为部分修复。

报告按块输出完全、部分、不可修复三类，并包含副本总数、逐状态计数、修复任务结果与计数自洽标志。不可修复块写 `E_UNRECOVERABLE`。

## 7. 快照

快照内容摘要覆盖除自身之外的全部字段。读取时先验证整数 `format_version=1`、正整数 `snapshot_version` 和内容摘要；内部摘要不符立即 `E_SNAPSHOT`。

恢复时比较：

- `group_version`
- `mirror_graph_version`
- `spec_digest`
- 当前副本清单的磁盘摘要

任一不同即把快照中的副本清单与镜像图整体写入恢复状态文件，并替换运行中的副本清单，不执行条目级合并。当前快照先复制为 `0444` 备份。

重放使用干净样本、记录中的注入参数与预算重新执行；比较受影响序列、打包摘要和版本、稳定调度事实、修复分类、副本状态摘要与块副本摘要。首个差异以 `E_REPLAY` 报告字段。

## 8. SQLite 提交点

每个尝试先插入 `task_claims(status=running)`。完成时在单个 `BEGIN IMMEDIATE` 事务中：

1. 读取并递增该运行的累计字节和游标。
2. 写入/更新 `task_commits`。
3. 把 claim 改为 `committed`。
4. 更新 `run_progress`。

强制退出后，`recover` 只把“仍为 running 且没有 commit”的 claim 改成 `unknown_requires_rerun`。已提交成功任务可依据提交行恢复，未发布产物从 staging 原子提升。恢复结果明确 `exit_report_available=false`。

