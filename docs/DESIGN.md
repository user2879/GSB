# 设计说明

## 模块边界

- `leasebench/validation.py`：严格 JSON/NDJSON 校验，所有失败统一为 `E_DECL` 并保留首个字段路径。
- `leasebench/engine.py`：确定性租约、借用、抢占、队列、版本失效与定点水位快照。
- `leasebench/fixedpoint.py`：NumPy `int64` 截断定点计算、误差段与稳定排序。
- `leasebench/ledger.py`：SQLite 批次事务、内容/链摘要、pending 意向、恢复与篡改检测。
- `leasebench/service.py`：从账本恢复、增量推进、声明版本切换和逐批重放比较。
- `leasebench/ui.py`：Tk 诊断界面和与 Tk 无关的稳定文本渲染器。
- `leasebench/cli.py`：校验、运行、重放、查看、UI 与异常注入入口。

## 租约状态机

```text
APPLICATION -> GRANTED -> ACTIVE -> RELEASED
                       -> QUEUED  -> GRANTED
ACTIVE -> STALE         (声明引用闭包失效)
```

租约硬截止为 `grant_ms + lease_ms`；计划完成点为 `grant_ms + expected_occupancy_ms`。实际调度取两者及最早借入归还截止的最小值。释放动作只执行一次，之后再次释放会抛错。

借入记录属于借用租约。释放借用租约时，借入记录先在覆盖容量中撤销，再释放函数已用量和全局预算。借入记录有唯一 `returned_ms` 和处置结论，不会与同一租约的到期结论重复记账。

计划完成、硬到期、借入归还截止、抢占边界和显式释放共用同一幂等释放路径。同一时刻若有多个触发条件，按固定类别优先级选择唯一结论。显式释放由 `LeaseEngine.release_lease(...)` 提供，测试覆盖其预算释放和借入归还；普通样本事件流使用占用完成与硬到期路径。

## 全局预算、函数配额与借出

全局已用量是所有活跃租约并发数之和。函数有效容量为：

```text
effective_capacity(f) = max(f) + borrowed_in(f) - lent_out(f)
```

借出者只能借出其最小配额之上的空闲部分：

```text
lendable(f) = max(0, max(f) - max(used(f), min(f)) - lent_out(f))
```

这保证借出不侵占声明的最小配额。借用关系必须直接存在于声明中，且高优先级严格大于低优先级。

## 抢占边界

`preemption_checkpoint_ms` 定义会话安全检查点网格。对当前时刻 `t`：

```text
next_boundary = (floor(t / checkpoint) + 1) * checkpoint
```

候选租约必须满足 `next_boundary < planned_release_ms`，因此不会把一个即将自然完成的调用当作挂起对象。抢占对象按以下稳定键选择：

```text
(victim_priority, next_boundary, lease_id)
```

高优先级调用需求由多份租约满足时可选择多个对象，但同一受害者不会因同一受益事件重复预约。受益事件在边界前从其他容量获得授予时，未到边界的抢占预约会取消并写入 `PREEMPT_CANCEL`。

挂起时把原 `event_seq`、剩余占用时长和并发数重新入队；恢复等待时间由对应关系的归还截止决定。抢占记录包含受害者、受益事件、已占用并发、挂起边界、状态和重排队列标识。

## 水位误差

函数水位分母是函数有效容量，整体水位分母是全局预算。固定点乘积使用 `numpy.int64`，超出范围时拒绝计算，避免平台整数溢出差异。

比较两个函数时：

```text
quantization_bound = 2  # 每个单值 < 1 fp 单位
borrow_delay_bound = ceil(near_deadline_borrowed * scale / capacity)
combined_bound = quantization_bound + borrow_delay_bound
```

差值在 `combined_bound` 内仍使用 `function_id` 字典序作为决胜键，并标注误差影响。主导段是数值较大的误差来源；相等时标注并列。

## 声明版本演进

变化检测覆盖函数全部公开属性、全局预算和抢占关系的端点与归还截止。受影响函数包括：

- 属性变化函数；
- 变化抢占关系的两个端点；
- 全局预算变化时的全部函数。

活跃借入若任一端受影响，则其借用租约受影响。受影响活跃租约变为 `STALE` 并按剩余占用重排；受影响 pending 队列项作废后重排；未受影响队列项迁移到新版本。计划中的抢占若涉及受影响端点则作废。每个失效对象写入不可变失效表，包含旧/新版本、原因和事件序号。

当前视图从新版本状态整体生成；旧批次水位和旧状态只保留审计价值，不进入当前租约、队列或水位集合。

## SQLite 账本

关键表：

- `meta`：账本格式版本、游标、最后批次、链摘要；
- `declarations`：每个声明版本原文与摘要；
- `batches`：批次结果、完整状态快照、内容摘要、前链/本链摘要、游标；
- `pending_batches`：提交前意向；
- `unknown_leases`：恢复时发现的未提交候选租约；
- `batch_operations`：申请、授予、回收、借入、归还、抢占、失效等操作流；
- `watermarks`：逐批固定点水位；
- `invalidations`：版本失效对象。

批次提交前计算：

```text
content = sha256(canonical(batch_payload))
chain   = sha256(previous_chain + content)
```

状态快照摘要也写入批次。恢复先验证所有已提交批次的内容、状态和链，再加载最后状态。`pending_batches` 中的批次没有权威状态，只生成未知租约清单；重算成功后与同批次提交一起清除。

## 重放比较

重放不信任账本状态快照：它从首个声明版本和第一批事件开始重建引擎。遇到版本切换时按账本声明原文重新执行版本失效逻辑。每个批次重算后与账本 `operations`、`watermarks`、摘要和游标逐项比较。比较顺序固定为操作流、其他负载字段；首个操作差异会从租约、释放、借入、队列或抢占嵌套对象提取事件序号与租约标识。

## 界面

Tk 界面包含函数、租约、队列三个 `Treeview` 和借入/抢占/水位/失效文本页。所有刷新读取账本最后完整状态，不对账本写入。文本渲染器使用相同排序键，可用无头模式做逐字节稳定性验证。

