# 技术说明

## 1. 架构

```text
JSON spec + NDJSON requests
        |
        v
严格校验 / 规范化模型
        |
        +--> 确定性参数摘要 ----+
        |                       |
        v                       v
稳定哈希分流 -> 确定性批次规划 -> 配额调度器 -> ThreadPoolExecutor
                                           |
                                  JSON Lines over 127.0.0.1
                                           |
                                NumPy 推理子进程（每后端一个）
                                           |
                                           v
                        单事务：batch + receipt + quota + cursor + digest
                                           |
                          +----------------+----------------+
                          |                                 |
                          v                                 v
                     快速视图缓存                    权威重算 / 恢复重放
                  （单次 JOIN 读取）              （真实重新分流与推理）
```

主要模块：

- `model.py`：数据类、字段别名、强校验、规范 JSON、请求流摘要。
- `parameters.py`：SHA-256 参数流、量化、参数摘要、NumPy 前向推理、输出摘要。
- `routing.py`：稳定分桶、规则匹配、批次规划。
- `worker.py`：回环 TCP JSON-lines 推理子进程。
- `backends.py`：子进程生命周期、启动就绪行、TCP 握手、回执验证。
- `scheduling.py`：全局/后端双重配额与真实线程池调度。
- `ledger.py`：SQLite 账本、快速视图、版本替换、恢复状态、内容摘要。
- `engine.py`：端到端执行、权威校验、重放和恢复编排。
- `cli.py` / `gui.py`：无界面验收入口与交互操作台。

## 2. 强校验与首个非法项

校验严格按文件顺序执行。错误对象同时带有 `entry`（如 `backends[1]`、`rules[0]`、`requests[3]`）与 `field`（如 `port`、`version`、`input`），因此在同一处即可定位首个错误。

校验覆盖：

- 根对象、字段类型、正整数/整数判定，显式排除 Python 布尔值。
- 后端 ID、规则 ID 唯一；端口全局唯一。
- 主机只能是字面量 `127.0.0.1`（缺省值相同）。
- 量化口径受支持集合限制。
- 权重万分比闭区间、批大小与规则版本正整数。
- 每个规则目标后端存在。
- 请求 ID 唯一、输入有限且长度等于实际分流后后端维度。
- 并发上限与每后端配额正整数，且配额键必须指向已声明后端。

请求规则选择先支持显式 `prefix`/`request_prefix`/`path`/`rule_id`，否则按客户端键最长前缀匹配；单规则场景提供显式回退，避免未声明路径字段时无法运行。

## 3. 确定性参数与推理

参数流按以下规则生成：

1. 命名空间包含 `backend_id`、`model_dim`、`hidden_dim`、`seed` 与量化版本。
2. 对 `SHA-256(namespace + "\0" + tensor_name + counter_be64)` 取 32 字节，解释为四个大端 `uint64`，映射到 `[-1,1)`。
3. 权重乘固定尺度，构造输入/输出矩阵与偏置。
4. `fp32` 保持单精度；`int8` 使用对称 127 级；`int4` 使用对称 7 级，量化值为 `round(clip(x*L,-L,L))/L`。
5. 前向计算：`h=tanh(xW1+b1)`，`y=hW2+b2`，内部使用 `float64`，输出规范化为小端 `float32`。

参数摘要由元数据摘要和四个数组的规范化字节摘要共同得到。元数据包含后端标识、模型维度、隐藏维度、种子、量化口径与量化版本。`parameter_change_sources()` 可定位 `backend_id`、`model_dim`、`hidden_dim`、`seed`、`quantization`、`quantization_version` 或参数数组变化。

输出摘要绑定后端标识、请求序号和小端 `float32` 字节；批次输出摘要绑定批次 ID 和按请求序号排序的请求摘要。

## 4. 稳定分流与批次规划

```text
material = UTF8("gateway-route-v1\0" + rule_id + "\0" + client_key)
bucket   = be_uint64(SHA256(material)[0:8]) mod 10000
target   = bucket < canary_weight_bp ? canary : stable
```

10,000 个半开分桶编号为 `0..9999`，与万分比阈值直接比较。权重 `0` 和 `10000` 分别退化为全稳定和全灰度，没有边界漏洞。

分桶只依赖规则 ID 和客户端键，不依赖 Python `hash()`、字典插入顺序或进程哈希种子。

批次规划按请求到达时刻与请求序号稳定排序；按 `(rule_id, backend_id)` 分组；按规则批大小切块。批次 ID 由 revision、规则/后端联合摘要和组内序号组成。所有组和批次最终按明确键排序，调度输入顺序确定。

## 5. 子进程协议

主进程为每个后端启动：

```text
python -u -m gateway_diag.worker --backend-id ... --host 127.0.0.1 --port ...
```

启动协议：

1. 子进程在 NumPy 中独立构造参数，绑定 `127.0.0.1:port`，监听后向 stdout 输出单行 `ready`，包含后端、端口和参数摘要。
2. 主进程在超时内读取 `ready`；缺失或非法为 `E_HANDSHAKE`，端口绑定失败行为 `E_PORT`。
3. 主进程再通过真实 TCP 发送 `hello`，子进程回 `hello_ack`；失败仍为 `E_HANDSHAKE`。
4. 握手摘要必须与主进程独立计算值一致。

推理协议为单连接单行 JSON：

```json
{"type":"infer","backend_id":"stable","batch_id":"b...","items":[{"request_id":1,"input":[0.1,0.2]}]}
```

回执包含后端、端口、参数摘要、批次 ID、请求序号集合、逐请求输出摘要、批次输出摘要、耗时和工作进程 PID。主进程校验：

- 必填字段存在且类型正确。
- 后端、批次、参数摘要匹配。
- 请求序号集合无重复且精确等于提交集合。
- 逐请求输出覆盖同一集合，摘要为 64 位小写十六进制。
- 回执耗时和 PID 合法。

任何缺失/非法字段为 `E_RECEIPT`，并定位后端、端口、批次和字段。

子进程用 stdin 管道作为父进程存活信号；父进程被强制终止导致管道 EOF 后，子进程关闭监听套接字并退出，避免遗留回环进程。

## 6. 配额调度

调度器维护三类状态：

- 待调度批次。
- 全局运行集合。
- 每后端运行计数。

主协调循环只在“全局运行数 < 全局上限”且“目标后端运行数 < 该后端配额”时提交批次到 `ThreadPoolExecutor`，否则保留在队列。工作确实运行在线程池命名线程 `gateway-batch_*` 中。任务完成后才减少计数。

每个批次的单调时钟字段：

- `wait_start_ms`：进入调度队列。
- `wait_end_ms`：获得双重配额并开始提交。
- `exec_start_ms`：提交时刻。
- `exec_end_ms`：工作线程完成并返回回执。
- `elapsed_ms = exec_end_ms - exec_start_ms`。

实际观测的最大全局并发和每后端并发随运行报告返回，测试验证其不超限。批次完成后，协调线程以单个事务提交账本。

## 7. SQLite 账本

主要表：

- `meta`：`schema_version`、`ledger_format_version`、`ledger_content_digest`、当前 revision。
- `ledger_header`：单行整数 `format_version`、当前 `content_digest` 与 `active_revision`，用于强类型账本头部校验。
- `spec_versions`：规格快照、指纹、请求流摘要、格式版本、内容摘要、整体替换差异。
- `batches`：批次、回执摘要/原文、参数摘要、输出摘要、单调时钟、线程名、配额、陈旧标志、批次内容摘要。
- `ledger_entries`：逐请求规则、去向、后端、分桶、权重、权重版本、路由算法版本、批次归属、输出摘要、耗时、陈旧标志、条目内容摘要。
- `quota_usage`：每批次全局/后端槽位消耗与限制。
- `run_progress`：期望/已提交批次数与请求数、最后批次、最后请求、更新时间。
- `recovery_unknowns`：启动恢复发现的未提交批次，状态 `UNKNOWN`/`RESOLVED`。

批次事务写入批次、全部请求账本、回执摘要、配额、进度游标和新的账本内容摘要。SQLite 使用 WAL 与 `synchronous=FULL`，因此进程强制终止后只可能看到完整提交批次。

账本内容摘要覆盖当前 revision 的全部批次、请求条目与配额行，使用规范 JSON 和固定前缀做 SHA-256；任意已提交字段被修改会导致摘要不一致。

## 8. 快速视图与权威重算

### 快速视图

`load_preview_cache()` 先确定当前 revision，然后执行一次请求账本与批次表的 JOIN 读取，构造不可变 `PreviewCache`。界面查询只访问内存映射，不启动后端、不重新分流、不重新推理、不执行全量聚合。

缓存包含逐请求：

- 规则、去向、后端、分桶、权重与权重版本。
- 批次归属、输出摘要、参数摘要、回执摘要。
- 批次等待耗时与执行耗时。

并包含请求/去向/批次计数、等待与执行总/最大耗时、按后端和按规则汇总、稳定排序的批次成员清单。

### 权威重算

权威命令从当前规格和请求流重新计算稳定分桶与批次规划，启动真实回环后端重新执行全部批次，取得新的输出摘要，再与快速缓存逐请求比较：

1. 客户端键、到达时刻、规则、后端、去向。
2. 分桶、权重、权重版本、路由算法版本。
3. 批次归属。
4. 输出摘要与参数摘要。
5. 等待/执行耗时与聚合统计。
6. 最后比较完整的规范 JSON 字节。

首个差异抛出 `E_PREVIEW`，带请求序号、后端、批次与字段。耗时作为已提交观测事实从账本重算；新一次重放的真实墙钟耗时不会替换原批次耗时，否则“耗尽时间”本身不可逐字节复现。重放是否真实执行推理则独立体现在输出摘要校验。

### `E_PREVIEW` 的可测试性

`DiagnosticEngine.verify_authoritative(preview_cache)` 接受显式缓存快照。自动化测试会篡改去向或聚合耗时，验证错误精确落在首个请求及 `target`/`stats.total_exec_ms` 字段。

## 9. 版本演进与陈旧隔离

- 任意规格内容变化要求 `revision` 递增；同 revision 内容变化为 `E_SPEC`。
- 同规则内容变化要求规则 `version` 严格递增；否则 `E_SPEC` 定位到规则索引与 `version`。
- 新 revision 激活时，旧 revision 的全部批次和请求条目在同一事务内标记 `stale=1`。
- 当前快速视图只 JOIN 当前 revision 且 `stale=0` 的行；旧行不参与展示或统计。
- `spec_versions.diff_json` 稳定列出后端/规则的新增、删除和版本/内容变化，视图为整体替换，不是增量合并。
- 请求流摘要绑定当前 revision，防止复用旧账本写入不同请求集。

## 10. 崩溃恢复

运行开始时，根据确定计划与已提交批次生成恢复视图。缺失批次写入 `recovery_unknowns`，状态 `UNKNOWN`，`needs_rerun=true`。已完整提交批次保留。

恢复步骤：

1. 校验当前规格指纹、revision 与请求流摘要。
2. 只依据完整提交批次建立已提交集合。
3. 对缺失批次启动真实后端重跑并逐批事务提交。
4. 将对应 unknown 行改为 `RESOLVED`。
5. 从真实后端重新执行当前计划的全部批次，按请求序号逐项比较规则、去向、分桶、权重版本、批次、输出摘要和参数摘要。
6. 任一差异抛 `E_REPLAY`，含请求序号、后端、批次与字段。
7. 校验最终账本内容摘要。

强制终止测试使用独立 CLI 进程、每批延迟和 `SIGKILL`，轮询确认至少一个批次已提交但并非全部提交后终止；随后验证 unknown 标记、缺失重跑、账本完整性和重放一致性。

## 11. Windows 注意事项

- 子进程参数通过 `sys.executable -u -m gateway_diag.worker` 构造，不依赖 shell 引号。
- Windows 使用 `CREATE_NO_WINDOW` 避免 GUI 拉起控制台窗口。
- Windows 监听套接字优先使用 `SO_EXCLUSIVEADDRUSE`，防止共享端口。若独占绑定失败，先探测端口是否仍接受 TCP 连接：接受则判为真实占用并返回 `E_PORT`；拒绝则视为重启残留 `TIME_WAIT`，改用 `SO_REUSEADDR` 重试。子进程在写回应答后等待客户端先关闭，进一步降低服务端端口进入 `TIME_WAIT` 的风险。
- 仅使用标准库 `tkinter`、`sqlite3`、`socket`、`subprocess`、`threading` 与 NumPy。
- 回环通信不需要公网访问；防火墙只需允许本地 Python 监听高位端口。

当前交付验证在 Linux/WSL2 完成；Windows 分支已按上述方式实现，但应在最终 Windows 目标机重新执行测试矩阵，尤其确认 Tk 主题、防火墙首次提示、强制终止命令和固定样例端口可用性。

## 12. 验收测试矩阵

| 范围 | 关键断言 |
|---|---|
| 规格校验 | 唯一性、端口、主机、类型、量化、权重、批次、目标、limits |
| 请求校验 | ID 唯一、有限输入、路由后维度、前缀缺失 |
| 参数 | 同声明逐位一致、跨进程一致、变化来源可定位 |
| 路由 | 分桶范围/稳定、0/10000 边界、输入顺序无关 |
| 子进程 | 真实握手、回执集合、同端口重启、`E_PORT`、`E_HANDSHAKE`、`E_RECEIPT` |
| 调度 | 全局 2/后端 1 真实线程池上限、单调等待字段、完成回调 |
| 一致性 | 快速视图规范字节摘要等于权威重算、`E_PREVIEW` 首个字段 |
| 版本 | 旧行 stale、当前视图整体替换、版本变更清单、非法规则版本 |
| 恢复 | SIGKILL 后只恢复完整批次、unknown 状态、重跑、`E_REPLAY` 定位 |
| CLI | JSON 输出、退出码、run/preview/verify 真实链路 |
