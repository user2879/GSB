# 技术设计

## 1. 模块边界

| 模块 | 职责 |
|---|---|
| `cacheproof/spec.py` | JSON 强校验、视图版本顺序、初始摘要解析 |
| `cacheproof/content.py` | 确定性内容块、条目摘要、临时文件原子落盘 |
| `cacheproof/proof.py` | 容量证明构造与证明自摘要校验 |
| `cacheproof/db.py` | SQLite 架构、WAL、账本内容摘要 |
| `cacheproof/store.py` | 写入/删除事务、恢复、证明复核、视图原子重建、快照恢复 |
| `cacheproof/snapshot.py` | 会话快照封印、稳定差异、游标后重放与首差异定位 |
| `cacheproof/diagnostics.py` | 只读仪表盘、文本渲染、异常注入 |
| `cacheproof/ui.py` | 可选 Tkinter 界面 |
| `cacheproof/cli.py` | 命令行、退出码、JSON 日志 |

## 2. SQLite 模型

- `ledger_state`：单行；整数格式版本、批次、段集合版本、完成游标、声明摘要、账本内容摘要。
- `segments`：段容量、已用字节、条目数、段版本、规范化证明 JSON、证明摘要。
- `entries`：已提交条目键、字节数、内容种子、条目摘要、相对文件路径、提交游标。
- `operations`：意图/审计；`prepared`、`committed`、`rejected`、`unknown`，记录期望摘要、文件路径、耗时和业务码。
- `unknown_entries`：恢复时观测到的未知条目；主键包含操作 ID 和实际文件路径。
- `read_views`：整个视图快照为一行，带视图版本、快照摘要和上次重建差异。
- `spec_meta`：声明摘要副本。

账本内容摘要覆盖格式、批次、段集合版本、完成游标、声明摘要、段证明摘要、条目元数据、视图摘要、未知条目；排除时间、UUID 和原始操作审计行，方便重放阶段独立比较操作业务码。

## 3. 关键不变量

1. 对每个段：`used_bytes == sum(entry.size_bytes)`。
2. 对每个段：`used_bytes + remaining_bytes == capacity_bytes`。
3. 段证明条目严格按 UTF-8 键序排列，无重复键。
4. 每个已提交条目的文件大小和重算摘要与账本一致。
5. 证明内容摘要覆盖除自身外的全部证明字段。
6. 视图快照是单行整体对象，不存在条目级修补。
7. `operations.committed_cursor` 唯一且接受操作连续递增。
8. 未知文件不进入容量证明；只有已提交行恢复证明。

## 4. 写入状态机

```text
BEGIN IMMEDIATE
  检查段与容量
  INSERT operations(status=prepared)
COMMIT

写 workspace/tmp/.cacheproof-write-*.tmp
  流式生成内容 + 流式 SHA-256
  flush + fsync
  临时文件摘要校验
  os.replace(tmp, entries/.../<digest>.bin)
  fsync parent dir（非 Windows）

BEGIN IMMEDIATE
  再次检查容量与并发提交
  INSERT entries
  UPDATE segments(segment_version + 1)
  UPDATE ledger_state(segment_set_version + 1, completed_cursor + 1)
  重建所有段证明（统一携带新段集合版本）
  UPDATE operations(status=committed, cursor, elapsed)
  更新账本内容摘要
COMMIT

原子保存会话快照（最佳努力）
```

崩溃矩阵：

| 崩溃点 | 账本可见状态 | 重启处理 |
|---|---|---|
| prepared 前 | 无操作 | 无事可做 |
| prepared 后、临时写中 | prepared | 标 unknown；清理临时文件 |
| 文件已替换、证明事务前 | prepared + 文件 | 标 unknown；记录观测摘要 |
| 证明事务中 | prepared（SQLite 回滚） | 标 unknown |
| 证明事务后 | committed | 正常恢复；快照可能变旧 |
| 删除提交后、旧文件移除前 | committed delete | 恢复时清理无引用旧文件 |

## 5. 容量证明

证明构造只依赖段的容量、段版本、段集合版本和按 UTF-8 排序的条目元数据，因此同一账本状态不受输入加载顺序、字典插入顺序或文件系统枚举顺序影响。

独立复核执行：

- 从 `entries` 重新构造证明并与存储证明逐字节比较。
- 重算证明摘要。
- 检查已用/剩余/容量恒等式。
- 流式重算每个文件摘要与大小。
- 扫描段目录，拒绝未在账本或未知清单中出现的内容文件。

`segment_set_version` 每次接受写入/删除后递增。为保证所有段证明都携带同一当前集合版本，系统在同一事务中刷新所有段证明；段自身 `segment_version` 只在所属段变更时递增。

## 6. 读视图整体替换

视图差异比较以“旧快照”和“当前引用段全量条目”为输入。任一引用段的版本或条目摘要集合变化即视为过期。更新语句一次性写入：

- 新视图版本（旧版本 + 1）
- 新段集合版本
- 完整新快照 JSON
- 快照摘要
- 稳定差异 JSON

SQLite WAL 让其他连接在写事务提交前继续读取旧行。因此不会暴露半重建状态。

## 7. 快照与重放

会话快照包含：

- 格式版本、段集合版本、账本批次、完成游标、最后操作序号
- 逐段容量证明
- 每个读视图的完整快照与版本
- 未知条目
- 整个对象的内容摘要

恢复算法：

1. 校验顶层摘要、格式版本、证明摘要和逐条恒等式。
2. 快照与当前账本一致：直接返回。
3. 快照过旧：先令实际视图收敛，抑制自动快照保存。
4. 取 `operation_seq > snapshot.last_operation_seq` 的账本操作重放：
   - committed 写入/删除必须产生连续游标并修改重放状态。
   - rejected 操作只比较推断业务码，不改变状态。
   - unknown 操作不进入证明，只标注业务码。
5. 重建所有段证明，按段 ID/键排序逐项比较容量、字节数、条目摘要。
6. 从快照视图出发重演一次整体重建，并比较实际视图版本、快照摘要和差异 JSON。
7. 全部一致后，以当前账本状态生成新快照并原子整体替换旧文件。
8. 首个不一致抛出 `E_REPLAY`，带上 `segment_id`、`entry_key`、`field`、期望值与实际值。

## 8. 诊断日志与注入

日志是追加式 JSON Lines。每条包含：

- `log_format`
- `event_seq`
- `operation_seq`
- `monotonic_ns`
- `event`
- `business_code`
- 可选的段、条目、字段和详情

注入映射：

| 注入 | 改变内容 | 预期业务码 |
|---|---|---|
| `truncate` | 已提交内容文件少 1 字节 | `E_INTEGRITY` |
| `capacity` | 仅给复核器传入更小容量参数 | `E_CAPACITY` |
| `ledger-digest` | 修改 SQLite 的账本内容摘要字段 | `E_SNAPSHOT` |

## 9. Windows 注意事项

- 所有路径组件经 base64url 编码，避开 Windows 非法文件名字符。
- 工作区拒绝盘符根目录、`Windows`、`Program Files`、`ProgramData` 以及 `CON`、`NUL`、`COM1` 等保留名。
- 原子替换使用 `os.replace`，同一工作区内完成，不跨卷。
- Windows 不执行目录 `fsync`；文件本身仍执行 flush/fsync。
- Tkinter 导入或 Tcl 初始化失败映射为 `E_TK_UNAVAILABLE`，核心逻辑继续可用。
