# 技术设计

## 1. 模块边界

| 模块 | 职责 |
| --- | --- |
| `gsb.spec` | 行号感知 JSON 扫描、结构/引用/DAG 校验 |
| `gsb.params` | 稳定参数、定点前向、损失摘要、量化误差界 |
| `gsb.worker` | 每分片独立子进程、真实 TCP JSONL 服务、子进程租约版本表 |
| `gsb.protocol` / `gsb.process` | 有超时的套接字客户端、启动/握手/停止生命周期 |
| `gsb.checkpoint` | 提交冻结、回滚版本、世代号、确定性基线 |
| `gsb.leases` | 单调时钟租约与乐观续约 |
| `gsb.scheduler` | 三配额准入、误差决胜、真实线程池、等待记录 |
| `gsb.ledger` | SQLite 原子事务、快照替换、恢复与逐字段重放 |
| `gsb.invalidation` | 配置差异、依赖链失效传播、全量重建对比 |
| `gsb.engine` / `gsb.cli` | 到达序分段编排、事件屏障与命令入口 |

## 2. 确定性参数与前向

参数种子熵为 SHA-256 派生，而不是 Python 内置哈希：

```text
seed, shard_id, parameter_version, layer_position,
layer_id, input_dim, output_dim, scale, depends_on
```

用 `numpy.random.SeedSequence` 和 `PCG64` 生成 `int64` 权重/偏置。权重和偏置使用小端连续字节、形状和类型共同参与层摘要；参数总摘要再绑定分片、种子、版本、样本维度和按稳定拓扑排序的层摘要。

前向使用 NumPy 的 `object` dtype 承载任意精度 Python 整数，避免量化链在 `int64` 上溢出；矩阵乘、偏置相加和逐元素整除仍由 NumPy 完成。样本在计算前按 `sample_id` 排序，因此参与集合相同即结果逐位相同。

损失载荷：

```json
{
  "shard_id": "...",
  "batch_id": "...",
  "sample_ids": [1, 2],
  "parameter_version": 2,
  "parameter_digest": "...",
  "loss": 123,
  "sample_digests": {"1": "...", "2": "..."}
}
```

`loss_digest = sha256(canonical_json(payload))`。规范 JSON 使用排序键、无空白、UTF-8 和禁止 NaN。

## 3. 量化误差界

每层以最大绝对权重的列和传播整数误差上界：

```text
err_root   = 1
err_layer  = floor(sum(dep_err * dep_max_abs_weight) / scale) + 1
magnitude  = floor((sum(dep_magnitude * dep_max_abs_weight) + bias_bound) / scale) + 1
loss_bound = samples * output_dim * (2 * magnitude * err + err^2)
```

实时损失误差可能远小于该保守界。该界只用于调度器的合成误差判定：当

```text
remaining_budget - candidate.resource_units <= candidate.error_bound
```

或两个候选的 `waterline` 差不超过双方误差界之和时，候选集合被标记为受误差影响。最终放行仍严格按 `(shard_id, batch_order, batch_id)` 的显式规则决定。

## 4. 检查点状态机

每个分片维护：

- `current_parameter_version`
- `generation`
- 每个检查点的声明版本、实际冻结版本、参数摘要、损失基线摘要、提交序号和批次序列。

提交时，损失基线只包含到达序号小于提交事件序号的批次，并按批次到达序排列，因此完成顺序与线程调度不会改变检查点字节。回滚只恢复目标检查点的参数版本并令 `generation += 1`；之后的新批次读取恢复后的版本。

引擎按到达序把工作分为“并发批次段 → 控制事件屏障”。屏障保证提交/回滚不会越过尚未完成的声明前驱，同时保留段内多分片并发。

## 5. 租约

租约记录包含标识、分片、授予/到期单调毫秒、续约次数、乐观版本和状态。主进程和子进程各持版本表：

1. 主进程先检查活动状态与本地版本。
2. 主进程携带调用方给出的版本向子进程发起续约。
3. 子进程版本严格等于请求版本才自增并延长。
4. 任一版本检查失败都直接报 `E_LEASE`，不写新到期时间。

到期由真实 `time.monotonic_ns()` 判定，测试可注入时钟。调度循环在每次准入前执行 `prune()`；过期租约立即不再占用分片租约额度。

## 6. 调度不变量

任一时刻同时满足：

```text
sum(inflight.resource_units) <= global_budget
active_leases(shard) <= shard.lease_limit
active_batches(shard) <= shard.concurrency_limit
```

准入、额度预占和 Future 提交在同一调度线程完成；执行本身只发生在 `ThreadPoolExecutor`。每个结果记录 `thread_name`，测试可验证不在主线程运行。排队记录从调度器接收到批次时开始，到实际放行时结束；无法满足的批次结束为空并报 `E_QUOTA`。

## 7. 账本

SQLite 使用 WAL、`synchronous=FULL`、`BEGIN IMMEDIATE`。主要表：

- `batches`：序号、批次/分片归属、参数版本与摘要、整数格式版本、损失值/摘要、样本集合、租约归属、检查点状态摘要、进度游标、行内容摘要、提交单调时间。
- `attempts`：提交前先写 `pending`；完成后同事务改 `committed`；崩溃恢复将遗留 `pending` 改为 `unknown`。
- `leases`：活动、释放或回收状态。
- `checkpoints`、`progress`、`runtime_state`、`events`。
- `snapshot`：格式版本、分片集合版本、初始参数版本映射、全账本内容摘要、替换世代。

打开账本时依次检查格式版本、分片集合版本、参数版本映射、行汇总内容摘要与 meta 摘要。任一不符即清空全部表并初始化新快照，不做增量合并。

批次提交先验证快照摘要与当前已提交状态一致，再在一个事务内写批次、attempt、progress、lease、checkpoint，并刷新全内容摘要。任何异常由上下文管理器整体回滚。

恢复重放按 `batch_order, shard_id, batch_id` 排序。依次验证行内容摘要、检查点摘要、批次归属、样本集合、参数摘要、损失值和损失摘要；首个差异以 `E_REPLAY` 返回 `batch_id`、`shard_id`、`field`、`expected`、`actual`。

## 8. 失效传播

审计首先比较分片种子、样本维度、初始参数版本以及每层维度、缩放和依赖；然后按事件流计算每个批次实际观察到的参数版本。直接失效来自：

- 批次新增/删除、分片/样本/检查点引用变化；
- 有效参数版本变化；
- 分片参数派生字段变化；
- 损失摘要变化。

随后沿批次 `depends_on_batches` 与检查点引用传播间接失效。检查点若位于失效分片、声明变化，或其提交序号前包含失效批次，则失效。未失效批次分别从旧、新声明独立重算，并要求规范 JSON 字节完全相等。

`rebuild-check` 不读取既有批次摘要作为输入，而是从声明、事件流和参数派生重新计算所有作者摘要与检查点管理器快照，再与账本逐字段比较。

