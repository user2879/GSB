# 技术架构

## 模块边界

```text
config_diag/
  validation.py    严格 JSON 解析、字段校验、版本序列模拟、环检测
  model.py         规范化不可变声明/变更/客户端模型
  graph.py         稳定拓扑、后代集合、传播依据、GUI 分层
  compute.py       定点公式、逐键摘要、账本摘要、整体视图
  ledger.py        SQLite 事务、元数据、journal、快照、恢复
  engine.py        初始装载、增量提交、过期回收、诊断状态
  subscriptions.py 固定种子轮询与整视图替换缓存
  compare.py       增量路径 vs 全量重建路径
  recovery.py      已提交 journal 的独立重放校验
  injections.py    三类异常样本与日志
  gui.py           Tkinter 延迟加载界面
  cli.py           无 GUI 命令行入口
```

## 规范 JSON

所有摘要都基于 `canonical_json`：

- UTF-8
- key 排序
- 紧凑分隔符
- `ensure_ascii=False`
- 禁止 NaN/Infinity

逐键摘要覆盖 key、原始值、生效值、键级版本、refs、formula、TTL、单调过期时刻、状态和派生依据。账本摘要覆盖格式版本、账本版本、集合版本、派生口径版本、游标和全部键的规范记录。

## 数据库表

| 表 | 用途 |
|---|---|
| `ledger_meta` | 单行账本元数据与摘要 |
| `ledger_keys` | 当前逐键记录，主键 key |
| `change_journal` | pending/committed/unknown 变更意图、游标和轨迹 |
| `snapshots` | 每次完整提交后的可验证快照 |
| `views` | current/stale 整体视图 |
| `stale_traces` | 旧视图对应失效轨迹的陈旧标记 |

键表主键避免同一键产生重复记录。journal 的 `operation_id` 唯一；相同操作重放返回原 journal 结果和 `idempotent=true`。

## 稳定性

- 所有对外集合按 key 排序。
- 拓扑排序使用最小堆作为平局规则。
- 回收先排序到期键，再求后代。
- 视图从完整记录重建。
- snapshot 排序 `ledger_version DESC, snapshot_id DESC`。
- 日志每行独立 JSON 并 `fsync`。
- 同层 GUI 节点按 key 排序。

## 崩溃窗口

```text
[提交 intent: pending]  <-- crash -> unknown, ledger unchanged
          |
[BEGIN IMMEDIATE]
  UPSERT changed keys
  UPDATE ledger_meta + digest
  INSERT current view; mark old stale
  UPDATE journal committed
  INSERT snapshot
[COMMIT]                <-- crash before commit -> SQLite rollback + unknown
```

`after-intent` 和 `before-commit` 都会留下已提交的 pending intent，但不会留下半成品键记录。重启后 intent 转 unknown，恢复重放跳过它；重跑时复用同一 journal id。

## 恢复重放隔离

恢复不信任当前视图或当前派生值，而是：

1. 从 `ledger_meta.initial_spec_json` 和 `initial_clients_json` 还原初始声明。
2. 在 `runtime/replay-*` 创建临时 SQLite 账本。
3. 使用原始 `initialized_mono_ns` 初始化。
4. 仅按 committed journal 逐操作重放。
5. 过期操作使用 journal 记录的 reclaim 时刻和回收集合。
6. 比较逐键字段、账本摘要、current view 和轨迹。
7. 删除临时目录。

因此即使有人手工修改当前键值并同步修改账本摘要，重放仍会报 `E_REPLAY`。

## 设计取舍

- 过期键保留为 `expired` 行而不物理删除，便于审计、版本追溯和订阅 unavailable 列表；生效视图会排除。
- “版本号”与“键级版本号”统一为 `ledger_keys.version`；直接编辑使用变更声明的整数版本，间接受影响键每次传播递增 1。
- 根键不可通过变更转为派生键；派生键的 `value` 只读，口径通过 `refs + formula` 修改。
- 全量比较路径从空账本装载并逐步执行全键重算，不复用增量数据库。
