# 设计说明

## 1. 核心不变量

1. 调度配额按“正在占用配额的任务”计数，而不是按线程数或套接字连接数猜测。
2. 任务只有在 `PENDING/PAUSED` 且同时获得全局和节点配额后才进入 `RUNNING`。
3. 退避状态 `BACKOFF` 不占用配额；退避计时结束后重新进入 `PENDING`，重新竞争配额。
4. 分片检查点只能在收到完整、与请求语义摘要一致的节点响应后追加。
5. 分片索引必须从 `0` 连续增长；恢复只信任已提交检查点。
6. 每个调度事件与派生表更新在同一 SQLite 事务中提交。
7. 任何派生统计只读取当前 `is_stale=0` 会话；旧会话不能参与当前展示。
8. 回执审计和节点排序是回执集合的纯函数，以显式排序消除插入/执行顺序影响。

## 2. 节点协议

客户端在每次分片下发时建立新的回环连接：

```text
hello(protocol=1) -> hello_ok(protocol,node_id,pid,seed)
infer(request)    -> result(...) | error(code,detail)
```

`infer` 包含 `request_id`、任务、节点、配置版本、分片索引/总数、切片配置和确定性输入参数。节点计算：

1. 对 `request_id + 语义摘要` 做幂等缓存；
2. 从 SHA-256 生成输入向量；
3. 从种子、版本和分片生成权重矩阵；
4. 对排序后的配置切片执行整数 NumPy 乘加和矩阵点积；
5. 对元数据与输出字节计算 SHA-256 摘要；
6. 返回输出摘要、输入摘要和整数毫秒耗时。

所有算术使用无符号 64 位回绕与掩码，避免浮点平台差异；浮点配置值先按其 IEEE-754 位模式转成整数。

## 3. 调度状态机

```text
WAITING_RELEASE -> PENDING -> RUNNING -> DONE
                         |       |  \-> PREEMPT_REQUESTED -> PAUSED -> PENDING
                         |       +----> BACKOFF -> PENDING
                         |       +----> EXHAUSTED / FAILED
                         +------------> EXHAUSTED / FAILED
```

协调线程只负责优先级选择、配额分配、抢占请求与最终状态检查；每个运行任务由工作线程逐个分片执行。任务在多个分片之间保持配额，在分片响应提交后才允许挂起，因此不会产生半完成分片。

### 抢占选择

对最高优先级就绪任务：

1. 若目标节点配额耗尽，从该节点的较低优先级运行任务中选择；
2. 若全局配额仍耗尽，从剩余较低优先级运行任务中选择；
3. 候选排序为 `(priority, completed_shard_count, task_id)`；
4. 被选任务不立即释放配额，而是在当前分片提交后进入 `PAUSED`；
5. 调度器等到配额实际释放后才启动高优先级任务。

同一轮还会审计所有较低优先级的退避、终结和挂起任务，并记录不抢占的依据。

## 4. 事务与恢复

每个调度事件执行以下原子过程：

```text
BEGIN IMMEDIATE
  INSERT rounds(round_seq,cursor,kind,payload_json,content_digest)
  UPDATE sessions(state_json,content_digest)
  UPSERT tasks
  UPSERT attempts
  UPSERT preemptions / preemption_decisions
  UPSERT receipts / node_untrust
COMMIT
```

`rounds.payload_json` 保存事件和完整调度状态，`content_digest` 是规范化 JSON 的 SHA-256。恢复时：

- 打开账本校验格式版本；
- 读取当前会话及其状态摘要；
- 对最后状态中 `RUNNING/PREEMPT_REQUESTED` 的任务记录未知分片，清除在途请求并回到 `PENDING`；
- 对已提交 `PAUSED` 的任务保留检查点，恢复后可直接续发；
- 未提交在途请求没有检查点，因此从最后提交的 `completed_shards` 边界重跑。

## 5. 回执依赖

回执比较槽位是 `(node_id, shard_index, input_digest, shard_count)`。直接失配沿两类引用传播：

- 同节点；
- 共享配置项键。

传播只作用于生成版本早于当前版本的证据。相同槽位若存在多个当前版本输出且不一致，所有冲突回执都记为不一致，防止“任选一个”掩盖节点非确定性。

节点汇总只累计不可采信状态回执的整数分数：

```text
node_score = Σ untrust_score(untrusted receipts)
rank order = (-node_score, node_id)
```

## 6. 版本替换

会话由 `spec_version + spec_digest` 标识。新规格满足以下条件才允许替换：

- `spec_version` 严格增大；
- 已提交的配置版本不得以不同内容复用；
- 当前配置版本不得倒退。

替换事务先归档旧会话和全部子表，再创建空的新会话与演进记录。旧快照也被标记陈旧；恢复请求不会把它们合并进新视图。

快照自身有格式版本和内容摘要。读取时不一致会返回 `REPLACED`；`restore_snapshot_or_replace` 会归档当前会话并创建空会话，符合“整体替换而非增量合并”。

## 7. 回放

回放包含两层：

1. **离线不变量复算**：轮次/游标单调、分片连续唯一、任务输出聚合、等待窗口、抢占位置、回执与节点排序；
2. **实时重发**：启动相同节点，从快照确定剩余任务，重发账本中的回执请求，逐项比较输入摘要和输出摘要。

比较顺序固定：任务标识升序、回执键升序、节点排名升序。首个差异立即抛出 `E_REPLAY`，避免后续差异掩盖根因。

## 8. 可验证性边界

- 节点耗时字段是整数毫秒，但具体数值会受操作系统调度影响；确定性比较使用输出摘要、回执结论和排序，不使用耗时。
- 等待时长由真实单调时钟记录，回放验证其与窗口和一致；不要求跨进程重现完全相同的墙钟数值。
- Tkinter 仅做只读查询。刷新不写入调度、回执或快照表。
