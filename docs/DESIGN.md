# 设计说明

## 1. 模块边界

| 模块 | 职责 |
| --- | --- |
| `edgecache/spec.py` | 读取并强校验本地规格，稳定选择首个错误 |
| `edgecache/model.py` | 内容/策略模型、摘要、缓存键、依赖图与稳定拓扑序 |
| `edgecache/cache.py` | 单节点容量缓存、TTL 递减、逐出事件、条目版本 |
| `edgecache/protocol.py` | JSONL over TCP、真实套接字超时、服务端循环 |
| `edgecache/source_service.py` | 确定性内容源与 `reload` |
| `edgecache/node_service.py` | 节点配置、拉取、传播应用、整体恢复 |
| `edgecache/orchestrator.py` | 端口/子进程编排、轮次执行、强制终止与恢复 |
| `edgecache/ledger.py` | SQLite 轮次账本、版本查询、事务提交 |
| `edgecache/snapshot.py` | 快照读写、摘要/归属校验、重放、整体恢复 |
| `edgecache/gui.py` | 无副作用视图模型与 Tkinter 界面 |
| `edgecache/cli.py` | 验证、三场景演示、异常注入、查看与 GUI 入口 |

运行时仅使用 Python 标准库。`tools/generate_samples.py` 通过 NumPy 官方 wheel 和固定种子生成样本数值。

## 2. 强校验与稳定首错

内容记录先按 `path` 排序，再按固定字段顺序检查；节点的字段路径按 `node_id` 排序后生成。因此同一无效输入即使 JSON 数组顺序不同，也会稳定返回同一个首个 `path` 与 `field_path`。

结构或语义错误返回 `E_SPEC`。内容字节/摘要/派生拼接不一致返回 `E_CONTENT`。派生图环在加载层整体拒绝；普通加载返回 `E_SPEC`，异常注入场景显式使用 `E_CYCLE` 并在 `details.spec_rejection` 中保留 `E_SPEC` 依据。

只有全部校验通过后才构造 `LoadedSpec`，不存在部分加载。

## 3. 缓存不变量

- 条目身份由策略声明的缓存键字段构成。
- 条目记录 ETag、全局内容版本、原始写入单调时钟毫秒、原始 TTL、当前剩余 TTL 和条目版本号。
- `get` 基于两次访问之间的单调时钟差递减剩余 TTL；减到 0 时删除并记录 `expired` 事件。
- 容量超过限制时，逐出排序只读取不可变的写入时刻、绝对过期时刻和缓存键，不读取命中顺序。因此同一组条目在不同访问顺序下产生相同逐出集合。
- 条目写覆盖同一键时递增条目版本。
- 节点拒绝与当前账本版本不一致的读写，返回 `E_STALE`。

## 4. 派生失效

依赖图方向为 `derived_from -> dependent`。对触发路径执行可达性搜索后，在诱导子图上执行以最小堆为就绪队列的 Kahn 拓扑排序。排序只依赖路径字典序与图结构，不依赖 JSON 输入顺序、邻接构建顺序或集合迭代顺序。

传播计划逐行保存：

```text
seq, trigger_path, affected_path, depth, content_version, policy_version
```

一次内容更新轮次中，源路径与全部下游路径在各节点先失效再写入新 ETag；未受影响但仍保留的条目被重绑定到同一轮的全局 `content_version`。账本提交前再次验证所有条目只引用该轮内容版本。

## 5. 双版本账本

`ledger_versions` 保存轮次级版本引用和完整内容/策略 JSON；`entries`、`evictions`、`propagation` 均以 `ledger_version` 为作用域。`meta.current_ledger_version` 只指向最新 `complete` 轮次。

每轮流程：

1. 先提交一条 `pending` 轮次记录。
2. 更新源和节点，收集状态。
3. 在单个 `BEGIN IMMEDIATE` 事务中插入全部条目、逐出、轨迹和游标，更新为 `complete` 并移动当前版本指针。
4. 若进程中断，启动时 `recover_pending()` 将遗留 `pending` 标记为 `unknown`；恢复只读取最后的 `complete` 版本。

历史版本本身和未进入当前统计的版本视图均标记 `stale=true`。

## 6. 快照与整体替换

快照文件包含：

- `ledger_version`、`content_version`、`policy_version`。
- `content_versions`、`content_digest`、`policy_digest`。
- 完整 `content_records` 与派生 `graph`。
- 当前节点条目、逐出事件、传播轨迹与进度游标。

加载时依次验证 JSON、内容记录、派生内容、内容摘要、策略摘要、版本集合、图、缓存键、节点归属和每个条目的 ETag/内容版本。之后按快照图重新计算每个触发源的稳定下游集合，与保存轨迹逐项比较。

若重放不一致，`E_REPLAY` 的 `details` 包含 `trigger_path`、`first_mismatch_index`、`affected_path`、`expected_affected_path` 和 `actual_affected_path`。

版本集合不一致时，恢复创建新的完成轮次，只写入快照条目和快照派生图。旧轮次保留审计但全部陈旧，不做条目合并。保存后续快照时，先前的快照文件变为只读备份。

快照文件保留原始写入单调时钟基准和当时剩余 TTL。整体恢复时以恢复时刻作为新的单调时钟基准、保持快照剩余 TTL，避免离线时间使恢复结果不可控；崩溃恢复则直接使用账本中的原始基准，因此停机时间照常计入 TTL。

## 7. 子进程、端口与终止

- 编排器按排序后的 `node_id` 分配 `port_base + 1 ...`，源使用 `port_base`。
- 启动前逐个独占探测回环端口；活动监听或普通占用均报告 `E_PORT`。
- 子进程通过 `sys.executable -u -m edgecache.worker` 启动，不使用 shell。
- Windows 使用 `CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW`；日志重定向到本地文件。
- 握手读取使用 `socket.settimeout`，超时由真实套接字产生并映射为 `E_TIMEOUT`。
- 强制终止调用进程级 `kill()`。终止结果固定包含 `exit_report=null`、`exit_report_available=false`，不会读取日志伪造退出报告。
- 恢复时重放内容源到账本当前内容，重启节点并从账本条目整体恢复。

## 8. 工作台

`build_view_model` 只执行只读查询并按 `(node_id, path, etag, cache_key)`、`(node_id, event_seq)`、`seq`、`ledger_version` 稳定排序。Tkinter 四个页签分别展示条目、逐出、失效链和版本；刷新只替换界面数据，不改变账本内容。
