# 架构与确定性不变量

## 数据流

```text
JSON 服务图 / 根因规则 ──┐
                        ├─> 强校验 ─> 归一(v1/v2/v3) ─> 窗口聚合 ─> 根因关联
NDJSON 混合告警 ────────┘        │             │             │
                                 └─────────────┴─────────────┴─> 批次快照
                                                                    │
                                                         SQLite 原子事务
                                                                    │
                                   ┌────────────────────────────────┴──────────┐
                                   │                                          │
                              当前视图/导出                             恢复与重放校验
```

`spec.py` 是唯一外部输入入口。图、规则和告警全部通过校验后，`engine.py` 才创建
批次任务。每个批次依次运行 `normalize`、`deduplicate`、`correlate` 三个阶段；
阶段之间存在数据依赖，阶段内任务可并行。`storage.py` 只在批次结束后提交一个
完整事务。

## 确定性不变量

1. 服务解析顺序不依赖输入顺序。拓扑排序使用最小堆，邻接表按
   `(拓扑秩, 服务标识)` 排序。
2. 告警组按 `(service_id, fingerprint)` 排序；组内按
   `(timestamp_ms, line_number)` 排序。
3. 聚合、候选和账本在输出前均使用显式稳定键排序。
4. 所有集合在进入输出前转为排序 tuple/list，避免哈希种子影响。
5. 内容摘要基于 `sort_keys=true`、无多余空白、固定 UTF-8 的规范 JSON。
6. 调度耗时与业务结论分离。`conclusions.canonical.json`、批次摘要、聚合和账本
   不包含单调时钟值。
7. 规则、依赖图和输入文件的 SHA-256 进入语义摘要。语义变化但声明版本未递增时
   直接拒绝。

## 调度约束

调度器只有一个条件变量状态：

```text
running_global < global_limit
AND
running_by_service[service] < per_service_limit
```

两个条件在同一临界区内同时检查并原子占用，因此不会出现“持有全局槽位等待服务
槽位”的额外阻塞。任务释放时 `notify_all`，所有等待任务重新竞争。峰值仅在成功
占用后更新，不会超过两个上限。

线程池工作线程数等于全局上限；线程池排队时间与配额排队时间都从任务提交时刻
计入 `wait_ms`。

## 代际状态机

```text
absent ──(更高声明版本)──> building ──(全批次提交)──> complete/current
                               │
                               ├─(强制终止)─> building + unknown attempt
                               └─(恢复)────> replay committed prefix

旧 complete/current ──(新代际)──> complete/stale
```

创建新代际和把旧代际标记陈旧在同一事务中完成。旧聚合、账本与批次统计同步写入
`is_stale=1`。新代际在完成前不是 current，因此旧数据不会被部分新结果混合展示。

## 批次事务

1. 独立提交 `batch_attempts(status=unknown)` 意图记录。
2. 计算该批次并得到完整当前聚合/账本快照。
3. 单事务写入 `batches`、`batch_aggregates`、`batch_ledger`、`aggregates`、
   `ledger`、`progress`，并把尝试改为 `committed`，旧 unknown 改为
   `superseded`。
4. 提交失败或进程被杀时，第 3 步整体回滚；第 1 步的 unknown 保留，恰当地表达
   “该批次结果未知，必须重跑”。

## 重放比较

`recover` 先校验当前输入与代际 `semantic_digest`、声明版本和种子一致，然后以
串行确定性模式从第一个已提交批次开始重建。每个批次比较：

- 聚合映射（含抑制数、参与行、来源版本、窗口版本）；
- 关联账本（含候选、路径、规则/图版本）；
- 批次结论摘要与批次摘要。

比较集合按 `alert_id` 排序，字段差异采用深度优先的稳定路径，例如
`aggregate.suppressed_count` 或 `ledger.candidates[0].path[1]`。首个不一致即抛
`E_REPLAY`，不继续生成伪完整报告。

## 验证器独立性

`verifier.py` 不调用 `engine.py` 的批次状态，也不信任数据库聚合内容。它直接：

```text
dataset.alerts ─> deduplicate_serial ─> build_ledger ─> compare current views
```

数据库只作为“待验证账本”读取，因此能发现主表、内容摘要和原始输入之间的不一致。
