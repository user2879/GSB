# 设计说明

## 1. 规范化与确定性边界

计划读取后先将层按标识排序，引用目标去重排序，所有摘要使用 UTF-8 规范 JSON（键排序、紧凑分隔符）。拓扑排序使用最小堆，因此同层可执行集合始终按字典序出队。所有失效源、参考映射和轨迹路径也按字典序决胜。

样本生成使用 `numpy.random.default_rng(fixed_seed)`。层权重的 RNG 种子由 `SHA-256(seed | layer_id | weights | input_dim | output_dim)` 派生，避免 Python 哈希随机化和文件顺序影响。每条样本单独执行前向，输出以 little-endian float32 字节计算 SHA-256。

## 2. 依赖图与失效

节点为层。边有两类：

- `output`：`producer -> consumer`，来自 `consumers[producer]`。
- `memory`：`prerequisite -> owner`，来自 `memory_derived_from[owner]`。

构建时合并重复边并做 Kahn 拓扑；剩余节点说明存在环，使用稳定 DFS 返回闭环。增量传播使用修改前后两张图的并集，因此引用删除会同时使旧消费者失效。对每个源执行最短路径 BFS，路径比较键为 `(深度, 节点序列)`；这一步产生每一步的触发源、直接上游 `via`、依据和完整路径。最终失效集合按新图稳定拓扑输出。

## 3. 增量计算与版本

失效层的块先释放，再按稳定拓扑重新分配。数值计算只遍历失效层；父层实际/浮点缓存分别保存，未失效层直接复用。

未失效层结果不重建，保留版本、输出摘要、逐样本摘要、块号、分配模式、请求字节和误差字段。失效层版本加一。全量数值 oracle 会重算所有层的数值与误差，但继承失效前未失效层的块位锚点，因为块号本质上是历史状态；这样可逐字段严格比较两个世界而不破坏“未失效块号不变”约束。

## 4. 浮点与定点路径

浮点：

```text
y_float = tanh(float32(x_float @ W_float + b_float))
```

定点：

```text
qx = sat_i32(rne(x * 2^s))
qw = sat_i32(rne(W * 2^s))
qb = sat_i32(rne(b * 2^s))
acc = sat_i64(sum(qx * qw)) + sat_i64(qb * 2^s)
pre_q = sat_i32(rne(acc / 2^s))
qy = sat_i32(rne(tanh(pre_q / 2^s) * 2^s))
y_fixed = qy / 2^s
```

`rne` 为 round-to-nearest, ties-to-even。整数除法通过 floor-division、余数与商奇偶判断实现，避免大整数转浮点丢精度。显式饱和边界为 `[-2^31, 2^31-1]` 和 `[-2^63, 2^63-1]`。

## 5. 误差归因

每层保存实际混合路径与完整 float32 参考路径，逐元素计算最大绝对偏差。三个误差界分量：

- 量化截断：层本地量化预算与上游量化预算的组合；若实际观测偏差更大，则把差值归入该段，保证合成界覆盖实际偏差。
- 累加顺序：同一整数矩阵乘分别使用向量化累加和逐项饱和累加，比较累加器差值并折算回实数尺度。
- 显存重排：失效层的块分配段数带来的附加界，连续分配时为 0。

总排名按实际最大偏差降序、层标识升序。若相邻层误差区间 `[实际偏差, 合成界]` 重叠，则两者标为 `tie_affected`，顶层判定另标 `top_decision_tie_affected`。累计占比按稳定拓扑顺序累计误差界并除以总和。

## 6. 块池

块池保存 `layer -> Allocation(blocks, mode, runs)`。分配候选为最低地址连续段；`auto` 无连续段时选最低编号的单块集合。失败不修改块池，抛 `E_OOM`。碎片量定义为：

```text
fragmentation = free_blocks - largest_contiguous_free_blocks
```

每批的峰值块数取该批提交时块池占用，碎片量取该批块池状态。请求字节按实际批样本数计算；层块数与重分配按 `max_batch_size` 工作区上界计算。

## 7. 状态摘要与 SQLite

内容摘要覆盖规范化计划、样本摘要、所有层结果、块池、批次、失效轨迹和归因，不包含时间戳和运行缓存。SQLite 包含：

- `ledger_meta`：整数格式版本、计划版本/摘要、当前内容摘要、完整当前状态 JSON。
- `plan_versions`、`sessions`：计划与样本的不可变历史；增量会话保存基线快照和变更源。
- `layer_commits`、`batch_commits`：逐层事务结果；每层实际/浮点批次摘要、块序、请求、峰值和碎片。
- `invalidation_events`、`unknown_layers`、`ledger_events`：轨迹、未知批次和审计事件。
- `current_layers`、`current_batches`、`current_invalidations`、`current_attribution`：可整体替换的当前领域表。
- `snapshots`：快照目录索引与内容摘要。

每次 `commit_layer` 在一个写事务中插入/更新该层结论、全部批次、移除该层 unknown 标记并更新 cursor/completed_layers。强制 `os._exit` 不会提交尚未开始的事务。

## 8. 快照与恢复状态机

快照目录采用临时目录写入、`fsync`、原子 rename。写入完成后文件和目录移除写位。文件名包含计划版本和内容摘要前缀，避免覆盖不同内容。

恢复比较三元组 `(plan_version, plan_digest, content_digest)`。不同时先固化当前状态为备份快照，再单事务清空当前领域表并从快照重建。会话历史与审计事件保留，不作为当前状态合并来源。

会话状态：

```text
running -> (process killed)
        -> needs_rerun  [recover; unknown_layers populated]
        -> running      [resume: verify committed, rerun unknown]
        -> complete     [current state replaced]
```

`replay` 从 `sessions.plan_json + samples_json` 重算；不需要当前运行缓存。重放比较顺序为账本层索引顺序，批次按 `batch_id`。任一差异生成 `E_REPLAY`，`basis` 携带会话、计划版本、记录层版本和提交序号。

## 9. GUI 视图

`viewmodel.build_view_model` 是无 Tk 依赖的纯函数，输出稳定拓扑层列表、按块号排序的块池、按排名排序的误差链、批次和快照。Tkinter 层只负责渲染与命令调用；因此无图形环境仍可测试领域可视化数据。所有 Treeview、Canvas、文字区和列表均按固定键排序，不依赖字典插入顺序。
