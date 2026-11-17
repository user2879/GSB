# 架构与不变量

## 1. 两条读取路径

系统刻意保留两个不共享缓存的决策路径：

```text
UI 增量路径
  commit -> 受影响 NodeCandidate -> Ledger 原子事务 -> 新 TreeSnapshot
         -> IncrementalViewModel 逐集群行更新 -> 单一新根绑定

权威路径
  ledger root -> load all nodes <= root -> rebuild histories/effective tree
              -> per-node simulator recompute -> compare canonical summary,
                 content digest, parent, commits, root pointer, task ownership
```

UI 路径可以用增量结果快速反馈，但任何“当前”统计只能绑定一个根。权威路径不信任 UI 节点，也不复用列表演示数据。

## 2. 版本树模型

每个修订只写受该修订任务影响的集群节点。节点主键为 `(cluster_id, revision)`。对集群 C 的节点链：

```text
C@1 -> C@3 -> C@7
```

表示 C 在修订 3、7 没有被写入时仍沿用最近节点。根修订 7 的有效节点是 C@7。完整树是 root 之前所有节点链的并集。

这避免了把未受影响集群重复写成没有语义的新节点，同时仍能由账本从任意旧根精确重建旧版本树。

## 3. 提交事务

`Ledger.commit_revision` 在 `BEGIN IMMEDIATE` 后执行：

1. 读取单例 `roots.root_revision`，要求其等于调用方预期父修订。
2. 拒绝已存在的 target revision 或重复集群候选。
3. 为每个候选查询该集群 `revision < target` 的最大节点作为父引用。
4. 写入全部节点。
5. 计算新根有效节点集并验证所有声明集群齐全。
6. 写入 `commits` 记录与根快照摘要。
7. `UPSERT` 根指针。
8. `COMMIT`。

在 SQLite 事务提交前进程终止，节点、提交记录和根指针均不可见；提交后三者同时可见。不存在“根已经切换但部分节点缺失”的可持久状态。

## 4. 视图换根

换根分为两个原子概念：

- 数据根：`IncrementalViewModel._active_snapshot = new_snapshot`，只来自账本重建。
- 行更新：只对目标修订影响集合或实际摘要变化的集群更新节点列。

旧根快照进入 `root_history`，此后只读。旧 `ViewItem` 的 `root_revision` 永远不等于新根，因此 `is_stale` 为真。当前统计先按 `bound_root_revision` 过滤，再排除陈旧项。

换根报告包含：

- `affected_clusters`：修订任务直接命中集合。
- `changed_clusters`：有效节点修订或内容摘要实际变化集合，含旧/新摘要。
- `stale_nodes`：旧根全部行绑定。
- `incremental_rows_touched`：本次允许 UI 重绘的集群行。

## 5. 内容与树摘要

节点内容摘要：

```text
sha256(canonical({
  schema, task_id, cluster_id, revision,
  parent_revision, aggregate_summary
}))
```

根快照摘要：

```text
sha256(canonical({
  schema, root_revision,
  sorted_effective_nodes:[cluster_id, revision, content_digest]
}))
```

完成时刻、服务耗时和查询延迟不参与内容摘要，但作为账本观测字段独立保存。

## 6. 延迟统计

UI 查询的测量区间只覆盖查询本身（账本一致读取与节点读取）；写入延迟事件发生在区间结束之后，避免把记账成本算进交互时间。权威路径覆盖整棵树的读取、模拟服务重算和比较。

统计完全使用整数：

```text
elapsed_us = round((ended_ns - started_ns) / 1000)
elapsed_ms_x1000 = elapsed_us
mean = round_half_up(sum_elapsed_us / count)
p95 = sorted_elapsed_us[ceil(0.95 * count) - 1]
max = max_elapsed_us
```

量化误差上界为 1 个 `ms_x1000` 单位。

## 7. 恢复状态机

启动时：

1. 打开 SQLite，严格验证节点自摘要、节点链、提交链和根指针。
2. 从账本当前根重建不可变快照。
3. 读取规格中当前根之后的修订，将其节点标记为 `unknown/rerun`。
4. 读取旧会话快照并比较根、节点摘要集合、会话 schema、延迟口径版本。
5. 匹配则采用；不匹配则从账本建立全新状态并整体覆盖快照。
6. 延迟统计真值始终从账本重新聚合，快照值用于重放核对。

重放启动后再次严格重建当前根，随后对每个历史节点调用模拟服务。任何差异抛出 `E_REPLAY`，不会用快照值覆盖账本或服务结果。

## 8. 异常注入边界

- `--inject-cycle`：只在内存中改写输入规格的最后一个 parent，随后走正常强校验与 `E_REV` 路径。
- `--inject-port-conflict`：父进程先持有规格声明的首个回环端口，再走正常端口检查。
- `--inject-ledger-corruption`：已在账本打开后、权威读取前篡改一条节点摘要；正常摘要校验报 `E_LEDGER`。

三者都不连接外部主机。
