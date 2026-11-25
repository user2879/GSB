# 技术设计

## 1. 规格与稳定序列化

所有 JSON 文件使用 UTF-8、重复键拒绝、`NaN/Infinity` 拒绝和严格字段集合。稳定序列化
使用：

```text
json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
```

摘要均为该字节序列的 SHA-256。请求按 `arrival_seq` 排序；工作器和批次按
`(worker_id, batch_seq)` 排序。批标识为：

```text
"batch:" + sha256(canonical({
  "schema_version": 1,
  "worker_id": worker_id,
  "batch_seq": batch_seq,
  "arrival_seqs": [参与请求的到达序号]
}))
```

## 2. 确定性推理与摘要

工作器参数由 `numpy.random.Generator(numpy.random.PCG64(seed))` 生成 4 个 float64
数组。参数摘要覆盖算法名、种子、输入/隐藏维度、数组形状和小端连续字节。前向计算为：

```text
hidden = tanh(input @ W1 + b1)
output = hidden @ W2 + b2
```

请求输出摘要覆盖输出算法名和输出 float64 字节。批次输出摘要再绑定：

```text
请求标识集合 + 参数版本序号 + 参数摘要 + 每个请求的输出摘要
```

批内请求先按到达序号排列；摘要和账本比较不受线程完成顺序影响。

## 3. 回执格式与归一

| 格式 | 字段 |
| --- | --- |
| r1 | `batch_id`, `request_id`, `output_digest` |
| r2 | r1 + `param_version_seq`, `latency_ms`, `worker_id` |
| r3 | r2 + `rank_in_batch`, `superseded_request_ids`, `param_digest` |

`merge_rules.json` 为缺字段声明唯一允许的补齐策略：

- r1：工作器和参数来自批次目录，耗时补 0，名次来自到达顺序，重复引用补空数组。
- r2：参数摘要来自批次目录，名次来自到达顺序，重复引用补空数组。
- r3：不补齐。

每条归一记录保留 `source_format`、`source_line`、`filled_fields` 和 `fill_basis`。
格式字段、耗时、名次和引用分别按 `E_FIELD`/`E_REF` 校验，位置包含文件行号。

## 4. 失效依赖链

节点依赖关系为工作器参数签名 → 工作器批次 → 批内请求。

- 工作器种子、输入维度、隐藏维度、参数版本序号或批大小变化时，该工作器旧计划和新
  计划的所有批次为直接失效批。
- 直接失效批中的请求为间接失效请求。
- 若请求级绑定变化，包含它的批次进入间接失效批闭包。
- 其他工作器批次不重算；账本行保持原始字节不变。

报告按工作器标识、批序号稳定排序，并给出旧计划/新计划存在性，以覆盖批大小变化导致
的旧批消失和新批出现。

## 5. 配额调度

调度器使用真实 `ThreadPoolExecutor(max_workers=global_concurrency)`。协调循环仅向
线程池提交同时满足全局槽位和该工作器配额的就绪批，因此等待某工作器配额的批不会占住
全局线程槽导致其他工作器饥饿。

每个批次记录：

- `queue_started_ms`：进入待调度队列。
- `queue_ended_ms`：分配到全局槽位和工作器配额。
- `run_started_ms` / `run_ended_ms`：实际工作线程执行起止。
- `wait_ms`：排队开始到配额获得。

时间全部来自 `time.monotonic_ns()`，不以 sleep 代替配额控制。

## 6. SQLite 账本

账本格式版本为整数 `1`。每个批次行包含工作器/批序归属、请求集合、参数版本、参数摘要、
计划摘要、回执摘要、归一记录和失效轨迹。初始计划行状态为 `pending`。

`complete_batch` 在一个 `BEGIN IMMEDIATE` 事务内更新全部归属和回执字段、失效轨迹、
元数据进度游标和账本内容摘要。SQLite 使用 WAL 与 `synchronous=FULL`。

恢复状态只把 `completed` 行视为可信；`pending` 行进入 `unknown_batch_ids` 并要求重跑。
恢复报告固定返回 `interrupted_process_exit_report: null`，不会推断强杀时的退出状态。

## 7. 快照与重放

快照包含工作器集合版本、每工作器参数版本与参数摘要、账本格式、账本内容摘要、进度游标、
完整批次行以及快照内容摘要。恢复时：

1. 校验快照内容摘要、行结构和账本摘要。
2. 若工作器集合版本、参数版本、参数摘要、目标账本摘要或游标任一不一致，使用临时
   SQLite 文件重建后原子 `os.replace`，不进行增量合并。
3. 对恢复后的 `pending` 批执行正常调度重算。
4. 以计划顺序逐批比较参考账本与实际账本；首个差异以 `E_REPLAY` 返回批标识和字段。

比较忽略工作器实际耗时毫秒，因为这是运行时观测值，不影响归属、参数绑定或回执摘要；
其余字段，包括状态、归属、请求集合、参数、回执摘要、归一记录和失效轨迹均比较。

## 8. 关键不变量

- `worker.port` 必须是 `127.0.0.1` 且同一全局声明内唯一。
- 同一批的归一记录必须一致地指向同一工作器、参数版本和参数摘要。
- 已完成行的回执摘要必须非空；待完成行不得声称拥有回执。
- 账本进度游标必须等于已完成行数。
- 快照游标必须等于快照中已完成行数。
- 计划、账本和快照内容摘要均不依赖 Python 字典遍历顺序。
