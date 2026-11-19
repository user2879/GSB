# 设计与一致性说明

## 1. 进程与文件布局

```text
client (Tkinter / CLI / JSON scenario)
          │  JSON lines over 127.0.0.1 TCP
          ▼
StorageService (ThreadingTCPServer)
          │  UploadThrottle: global + per-session quota
          ▼
Workbench (single SQLite connection + engine RLock)
          ├── ledger.sqlite3        会话/分片/版本/图/审计/调度
          ├── data/chunks/<sid>/    持久化分片
          ├── data/objects/...      已提交对象文件
          ├── data/tmp/             合并临时文件
          └── logs/                 结构化 JSONL
```

服务端只绑定调用方指定的回环地址，默认是 `127.0.0.1`。一个请求使用一条 TCP 连接和一帧 canonical JSON。对同一 JSON 值排序键并使用紧凑分隔符，保证摘要输入与输出稳定。

## 2. SQLite 事实源

主要表：

- `sessions`: 会话标识、对象键、分片大小、总大小、摘要算法、标签、当前状态、位图、完成标记、提交版本、恢复分类。
- `session_events`: 每次状态迁移的 `from_state`、`to_state`、位图、原因和单调时钟毫秒。
- `chunks`: 以 `(session_id, chunk_index)` 为主键的当前登记；重传使用 UPSERT 覆盖。
- `object_versions`: 对象键与单调版本号联合主键，保存元数据摘要、清单摘要、分片集合摘要、内容摘要、标签、可见性、状态和对象路径。
- `graph_nodes` / `graph_edges` / `graph_invalidation_events`: 持久化 DAG、节点缓存版本和失效审计。
- `audit_runs`: 每次审计的结论、首个偏差及稳定摘要。
- `scheduler_events`: 每片的排队时长、开始/结束单调时钟和限制值。

部分唯一索引保证每个对象键最多只有一个 `active` 版本。历史 `expired` 和 `deleted` 行保留，供版本识别和只读历史审计使用。

## 3. 确定性摘要口径

分片摘要：对分片原始字节使用会话声明的哈希算法。

分片清单摘要：canonical JSON `chunk-manifest/v1`，包含对象键、会话标识、分片大小、分片数、算法和按序 `{chunk_index, digest, size}`。

分片集合摘要：canonical JSON `chunk-set/v1`，包含分片数和按序二元组。

对象内容摘要：将有序分片字节依次交给同一哈希算法。

对象元数据摘要：canonical JSON `object-metadata/v1`，包含对象键、版本、会话、分片形状、算法、标签、内容摘要、清单摘要和分片集合摘要。

前缀计数视图摘要：canonical JSON，包含前缀、计数和排序后的 `{object_key, version, metadata_digest, chunk_set_digest}`。

所有口径都包含 schema 域字符串，避免不同对象类型出现相同字节输入语义。

## 4. 状态机与提交原子性

```text
created ──first chunk──▶ uploading ──commit begin──▶ verifying ──success──▶ committed
   │                        │                           │
   │                        ├──── abort ──▶ aborted ──resume──▶ uploading
   │                        │                           └─E_INTEGRITY/E_VERSION─▶ uploading
   └──── abort ─────────────┘
```

`committed` 是终态，拒绝续传、重传、中止和再次提交。

提交步骤：

1. 写入 `verifying` 迁移。
2. 逐片核对位图、登记行、文件存在性、文件大小和文件摘要。
3. 计算内容、清单和分片集合摘要；失败时恢复 `uploading`，保留会话和分片。
4. 写对象临时文件，逐片重新读取并再次校验，`flush + fsync`。
5. 计算新版本号和全部元数据摘要。
6. 在 SQLite 事务中使旧活动版本过期、插入新活动版本、把会话置为 `committed` 并写完成标记。
7. 在事务提交前执行 `os.replace(temp, final)`；若进程在替换后、事务提交前终止，文件是孤立且未被元数据引用，不会被视为可见对象。

因此，可见对象只能由完整校验、完整对象文件和已提交 SQLite 元数据共同构成。

## 5. 依赖图与稳定失效

边方向为 `dependency -> dependent`。节点种类和典型关系：

```text
session-state ─┐
               ├─▶ manifest ─▶ stage-metadata ─▶ session-summary
chunk(s) ──────┘                ▲
object-key ─────────────────────┘

manifest ─▶ committed metadata ─▶ object-key ─▶ prefix-view(s)
```

触发器包括分片登记/覆盖、`session-state` 的位图迁移、会话中止、对象删除和版本替换。

算法：

1. 从触发节点集合做 BFS，求全部可达后代和最短深度。
2. 对受影响子图做 Kahn 拓扑排序；同层按 `(depth, kind_rank, node_id)` 建堆，得到稳定顺序。
3. 对每个节点执行 `entry_version += 1`、`cached_digest = NULL`、`valid = 0`。
4. 将触发源、原因、上下文、完整受影响列表和全局 `graph_version` 写入事件表。
5. 提交/删除等操作随后按需刷新新版本节点、对象键节点和前缀视图；未受影响节点不写库。

添加边前检查两端节点存在，并检查反向后继可达性以拒绝环；启动时执行全图引用完整性和无环校验。

## 6. 前缀视图

对象键 `team/docs/a.txt` 生成 `*`、`team`、`team/docs` 视图。视图节点依赖本前缀内所有历史对象键节点；对象键节点依赖当前活动元数据节点。删除和版本替换会沿 `metadata -> key -> view` 传播。刷新时只重算涉及当前对象键的视图，因此其他前缀视图的版本和摘要保持不变。

## 7. 多版本替换

版本号取该对象键历史最大值加一。提交新版本时：

- 旧活动行改为 `expired`、`visible=0` 并记录替换时刻。
- 新行由当前会话字段完全构造，不继承旧标签、清单或摘要。
- 字段差异按字段名排序输出，包含前后值和摘要。
- 旧元数据节点作为失效源，沿 `object-key` 和前缀视图传播。
- 旧对象文件不删除，供只读历史与审计使用。

`expected_base_version` 在提交点比较当前活动版本；不一致返回 `E_VERSION`，会话退回 `uploading`，不改变元数据。

## 8. 审计

对每个活动对象按对象键排序：

1. 会话存在、状态 `committed`、完成标记为真、版本链接一致。
2. 分片登记索引严格为 `0..n-1`，磁盘目录文件集合与登记集合完全一致。
3. 每片大小正确，磁盘摘要等于登记摘要。
4. 重算分片集合摘要、清单摘要和元数据摘要，与对象版本对应字段一致。
5. 重算可见对象文件内容摘要。

遇到第一个偏差立即保存失败审计并返回 `E_AUDIT`，细节包含对象键、分片序号、偏差种类和期望/实际值。

## 9. 快照与恢复

快照是 canonical JSON，包含：

- 全部核心表行和依赖图节点/边；
- 图版本、状态摘要、对象版本签名、文件清单；
- 最近审计结论。

恢复流程先比较快照与当前对象的活动版本、全部版本记录和分片集合摘要，输出 `metadata_version_changes` 与 `chunk_set_changes`。随后在一个事务中删除当前工作台表并重建快照状态，明确标记 `mode=whole-state-replacement`、`incremental_merge=false`。若文件清单不一致，恢复结果要求重新审计。

服务端 `--snapshot-on-start` 可在接受请求前执行同一整体替换。

## 10. 强制终止恢复

服务启动时扫描 `state='verifying' AND completed=0` 的会话。此类会话表示上次进程可能在合并中终止，因此：

- 设置 `commit_unknown=1` 和 `recovery_status='unknown_needs_rerun'`；
- 写一条 `verifying -> verifying` 恢复事件；
- 清理合并临时文件；
- 不创建对象版本，不推断提交结果，不提供退出报告。

再次调用提交时会重新执行完整校验和合并；也可以保留分片后人工处理。只有 `committed + completed` 的会话和对象元数据被认可。

## 11. 并发模型

服务端 `UploadThrottle` 在 `chunk upload` 请求进入业务引擎前获取配额，释放发生在引擎调用结束后。每个结果返回排队时间和单调起止时间，并写入 `scheduler_events`。

业务引擎使用可重入锁使“文件写入 + SQLite UPSERT + 位图更新 + 依赖失效”成为线性化区间。因此同一分片并发重传最终只有一个登记行，且该行摘要与最后一个原子文件写入一致。客户端使用线程池并行发出请求，但服务端配额是最终强制边界。

