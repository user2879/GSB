# 设计说明

## 1. 核心状态

每个会话由以下持久状态组成：

- `volume.bin`：固定块卷镜像。
- `session.sqlite3`：快照、位图、撤销引用、不可变 checkpoint 视图、操作日志和审计结论的唯一原子元数据源。
- `manifest.json`：当前规范化 m3 清单，可由其他历史版本归一/整体替换。
- `objects/<sha256>.blk`：内容寻址块对象；同一内容跨快照共享。
- `backups/objects/`：对象只读副本，用于块损坏修复。
- `backups/manifests/`：每次整体替换前的只读清单。
- `backups/initial-volume.bin`：会话初始卷只读副本。
- `tmp/<operation>.image`、`tmp/<operation>.json`：回滚临时镜像与持久标记。
- `logs/diagnostics.jsonl`：结构化事件日志。

## 2. 快照模型

快照同时维护两种视图：

1. **不可变 checkpoint 视图** `snapshot_views`：创建快照时记录每个块的完整引用。由于内容寻址，未变化块只增加引用，不重复占用数据空间。该视图让任一快照都能独立复核 checkpoint，即使当前卷随后被回滚到更早状态。
2. **撤销位图** `snapshots.bitmap` + `snapshot_blocks`：快照创建后，每次块改写若该快照尚未捕获该块，则保存旧块对象、置位、登记摘要并递增 `snapshot_version`。所有这些更新与全局 `state_version` 在一个 SQLite 事务中提交。

写入顺序：

```text
read old block
  -> store old object + read-only backup
  -> BEGIN IMMEDIATE
       for every active snapshot without this undo block:
           insert old ref/digest
           set bitmap bit
           update checksum + snapshot_version
       increment state_version
     COMMIT
  -> write new block in place + fsync
  -> export canonical m3 manifest
```

若进程在 SQLite 提交与原位写之间退出，撤销引用仍指向当前旧字节；后续回滚是幂等且安全的。回滚卷切换不使用原位写，严格使用临时镜像和 `os.replace`。

## 3. 清单归一

`manifest.decode_manifest` 先完成格式内强校验，再通过 `ManifestResolver` 从 SQLite 补齐历史字段：

- m1：摘要表、数据引用、父引用、位图校验、内容摘要、快照版本。
- m2：父引用、数据引用、内容摘要、顶层清单摘要。
- m3：不需要补齐。

每个 `SnapshotManifest` 保留 `source_format/filled_fields/fill_basis`。所有版本最终进入同一内部结构，`fingerprint()` 对位图、摘要、引用、父链和快照版本做规范化 SHA-256，因此可以证明三版本归一结果一致。

## 4. 整体替换

导入清单时：

1. 解析并归一来源清单。
2. 计算当前 DB 与来源的 `added/removed/replaced`。
3. 将当前 `manifest.json` 复制到只读备份目录。
4. 单事务删除全部 `snapshot_blocks`、`snapshot_views`、`snapshots`，按父先子后顺序重建。
5. 已有快照保留 checkpoint 视图和创建序号；新增 m3 快照通过撤销引用重建 checkpoint。
6. 从 DB 重新导出唯一规范 m3 清单。

不存在新旧条目合并逻辑。若只是清单格式/内容与磁盘不同，`reconcile-manifest` 以 DB 为权威整体重写清单，并列出差异。

## 5. 回滚算法

`plan_rollback` 输出精确分数比例与阈值，避免浮点边界不稳定。`ratio == threshold` 固定选择 full。

- bitmap 路径：读取目标快照已置位块，按升序验证引用对象摘要并写入临时镜像。
- full 路径：从最新后代到目标逐级应用撤销引用，再应用目标引用；随后扫描所有块统计差异。最终仍报告目标位图画布块集合，以保证两条路径结果字段一致。

两条路径都在替换前把结果镜像 SHA-256 与目标 `checkpoint_digest` 比较，防止算法或状态不一致时误提交。

## 6. 操作恢复状态机

```text
prepared/incomplete
  | volume == expected -> recovered_completed/completed
  | volume == prior    -> rolled_back/incomplete_unknown (requires retry)
  | otherwise          -> unknown (requires retry)
prepared + failpoint    -> process exits, persistent state remains
completed               -> terminal
superseded              -> explicit retry accepted for an incomplete op
```

`recover_session` 可重复执行。`incomplete_unknown` 会持续阻止新的回滚，直到调用方显式传 `--retry`。

## 7. 审计与版本

每次审计持久化单调递增 `audit_version`，同时保存：

- 全局 `state_version`。
- 每个快照的 `snapshot_version`。
- 当前规范清单摘要。
- 逐快照位图/引用一致性、对象摘要、完整 checkpoint 重建、根快照到初始卷、清单版本匹配结果。

旧的审计行从不更新。读取最新结论时，如果全局状态或快照版本已经变化，则抛 `E_VERSION` 并给出差异；必须执行新审计产生新版本。

## 8. 异常注入

- 块损坏：确定性翻转目标块对象的一个字节，保留只读备份。回滚预检按块号升序返回第一个失败块、期望摘要和实际摘要。
- 清单截断：先备份，再保留首个 `snapshot_id` 后截断，保证 `E_FORMAT` 可指明快照与 `json` 字段。
- 版本冲突：备份后将合法 m3 文档的 `format_version` 改为其他已知历史版本；会话版本门禁在解析前报 `E_VERSION`。

修复后执行 `repair-and-verify`：恢复对象、以 DB 重写清单、回滚目标快照并逐块比较完整期望摘要表。

