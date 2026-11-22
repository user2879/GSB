# 技术设计

## 1. 领域与首错校验

规范化后的 `Spec` 只保留稳定排序：集群按 ID，修订按 `(cluster_id, version)`，任务按 ID。校验器按“结构 -> 修订图 -> 预算 -> 任务”的顺序执行，任何失败抛出一个 `AppError`，携带 `code`、`field`、`location` 和可选 `details`。

修订图先做 DFS 环检测。若父引用形成环，直接返回 `E_CYCLE`；其他父引用缺失、父版本不小于当前版本、重复版本等返回 `E_SPEC`。这让“修订链注入成环”具有独立错误分类，同时保留字段级定位。

## 2. 稳定计划

`canonical_json` 使用 `sort_keys=True`、紧凑分隔符、UTF-8 和禁止 NaN。计划摘要为不含 `plan_digest` 的计划体 SHA-256；写入产物时再规范化整套对象。

分片 ID 为 `cluster_id/task_id/0000`。计划按 `(cluster_id, task_id, shard_index)` 排序，配置键排序。每个分片携带：

- 稳定 ID、集群、任务、分片序号；
- 引用配置键、对应当前修订的配置值；
- 资源单位需求、来源修订版本；
- 集群配额和任务并发上限。

因此，JSON 数组换序、配置键换序、任务目标换序都不会改变计划字节或分片序列。

## 3. 回环节点与确定性聚合

每个目标集群启动一个 `python -m configdist.node` 子进程。节点先绑定声明端口，再通过 TCP JSON 行协议服务。请求/响应均为单行 UTF-8 JSON。

握手：

```json
{"op":"handshake","cluster_id":"edge-a","revision_version":2}
```

分片：

```json
{"op":"dispatch","cluster_id":"edge-a","task_id":"deploy-core",
 "shard_id":"edge-a/deploy-core/0000","config_keys":["mode","region"],
 "resource_units":2,"revision_version":2}
```

节点校验修订后，用固定 `simulation_seed`、集群、任务、分片、修订和配置值派生 NumPy RNG 种子。样本数量由同样材料稳定映射到 `128..256`。引用配置键决定分组，逐组计算 `count/sum/min/max`。返回：

```json
{"ok":true,"event":"aggregate","aggregate":{
  "scan_rows":141,"groups":[...],"byte_digest":"...","summary_digest":"..."
},"elapsed_ms":0}
```

`elapsed_ms` 来自 `time.perf_counter_ns()` 的整数除法，不注入人为延迟。端口活跃监听为 `E_PORT`；连接成功但协议握手超时为 `E_HANDSHAKE`。测试钩子只让真实已绑定节点暂停协议响应，不伪造连接失败。

节点监视父进程 stdin；协调进程被强制终止时管道关闭，节点自行退出，避免残留监听。

## 4. 三重预算调度器

协调线程维护以下运行计数：

- `running_resource` 与 `global_resource_units`；
- `running_cluster[c]` 与集群 `concurrency`；
- `running_task[t]` 与任务 `max_parallel`。

只有三项检查同时通过才把分片提交给线程池。启动动作与计数更新均在协调线程内完成，因此不依赖工作线程锁顺序。所有不可运行情形都可判定：

- 配额/任务上限为 0；
- 单分片资源超过全局预算；
- 仍有运行分片时，不足条件会随完成事件重新评估；
- 没有运行分片且队列仍不可启动时，整批标记为 `E_BUDGET`。

每片轨迹含 `wait_started_ms`、`wait_ended_ms`、`started_ms`、`finished_ms`、`elapsed_ms`、结局、错误和聚合摘要。结果同时给出全局、各集群、各任务的峰值占用。

## 5. Generation 整体替换

每次运行创建新 generation，并在单一账本事务中：

1. 将旧当前运行标为 `current=0, stale=1`；
2. 将旧分片全部标为 `stale=1`；
3. 插入新运行及完整预期分片清单。

因此旧轨迹和摘要不会逐片修补，也不参与当前预览/汇总。替换报告区分：

- `changed_shards`：相对旧计划新增、删除或指纹变化的片；
- `affected_shards`：本次整体替换涉及的全部新旧分片 ID；
- `reasons`：修订版本变化、任务目标变化、分片数变化等稳定依据。

## 6. SQLite 账本

账本表为 `meta`、`runs`、`run_shards`、`shards`。`meta.format_version` 以 SQLite 整数保存；`meta.content_digest` 是运行、预期分片、已提交分片和除摘要外元数据的规范 SHA-256。

每完成一个分片，在一个事务中：

- 递增 `runs.completed_cursor`；
- 写入/更新 `shards` 的 ID、结局、资源单位、聚合 JSON、扫描行数、字节摘要、摘要摘要、进度游标和轨迹时间；
- 将 `run_shards.state` 标为 `committed`；
- 重算并更新账本内容摘要。

打开已有账本时不会静默修复缺失摘要；缺失或错误摘要要求显式校验并报 `E_LEDGER`。

强制终止后，`run_shards` 中未对应的完整 `shards` 记录全部标为 `unknown`，恢复结果只列出已提交片和待重跑片，并明确 `exit_report_generated=false`。

## 7. 缓存预览与权威重算

`preview` 只读当前未陈旧 generation，按集群和任务聚合已提交进度；预期但未提交的片显示为 `unknown`，总数来自完整 `run_shards`。

`verify` 有两种权威来源：

1. 本地 `compute_aggregate` 重算；
2. 可选地把每片重新下发至对应回环节点。

节点结果先与本地 NumPy 权威结果比对，再与账本缓存比对。首差按 `(cluster_id, task_id, shard_index, shard_id)` 定位，`E_PREVIEW.details` 同时提供 `scan_rows` 和 `byte_digest` 的 cache/authoritative 对照。

## 8. 快照与恢复重放

快照保存三组版本（集群、修订链、预算）、完整规范清单、计划、分片轨迹、聚合摘要、未知片集合和账本版本，并对除 `content_digest` 之外的整份内容做 SHA-256。

恢复策略固定为：

```json
{"mode":"whole_replace","incremental_merge":false}
```

若调用方提供当前版本，版本差异只产生替换依据，不做合并。内容摘要不符或格式错误直接 `E_SNAPSHOT`。恢复时还会重新校验快照内清单，因此注入修订环返回 `E_CYCLE`。

重放按已完成的稳定分片顺序重新下发。逐字段比较 `scan_rows`、`byte_digest`、`summary_digest`、`groups`；首个差异返回 `E_REPLAY`，包含首个分片 ID、字段、期望值、实际值和已检查片数。

## 9. 测试矩阵

`tests/test_acceptance.py` 覆盖：

- 首错字段/位置、`E_SPEC`、`E_CYCLE`；
- 编辑顺序置换后的逐字节计划一致；
- 调度启动瞬间的三项预算不变量、峰值与阻塞判定；
- 真实子进程、JSON 行协议、账本、权威重算、快照重放；
- `E_PREVIEW` 的首片、扫描行数与字节摘要对照；
- 快照摘要错误、重放首差、快照清单成环；
- 修订/目标变化后的整体陈旧化；
- `E_PORT`、真实握手超时 `E_HANDSHAKE`；
- 提交 2 片后 `os._exit(17)`，恢复得到 2 个已提交片和 16 个未知片，再运行可重新完成。

## 10. 平台边界

代码使用标准库 TCP、SQLite、subprocess 和 pathlib，不依赖 Unix-only 能力；Windows 官方 Python 3.11+ 可直接运行。当前自动验证环境为 Linux/WSL，未安装 Tkinter，因此无头核心、真实回环节点、账本、快照和 CLI 均已实际执行，窗口层只做静态实现审查。Windows 上可用 `run_gui.bat` 启动窗口。
