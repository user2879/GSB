# 技术说明

## 1. 执行模型

`MeshRuntime` 的启动顺序：

1. 校验声明，选择活动路由修订。
2. 打开或初始化 SQLite 账本，设置活动修订、实例集合版本、配置摘要、全局并发和每服务配额元数据。
3. 若存在快照，验证格式、摘要、规则修订、实例版本、预算/游标；不一致则用账本权威状态整体替换。
4. 逐实例预检端口，使用 `subprocess.Popen([sys.executable, "-u", scripts/upstream_instance.py, ...])` 启动 Windows 子进程。Windows 设置 `CREATE_NO_WINDOW`；POSIX 设置新会话。
5. 通过真实 TCP 发送 `ping`，校验 `pong` 的 `protocol`、`instance_id`、`service_id` 和 `pid`。
6. 运行请求批次；每个请求在线程池中独立执行尝试与重试。
7. 全部完成后写快照，终止上游。

上游使用 stdin EOF 看门狗：父进程被强制终止后，写端关闭，子进程设置停止事件并退出 accept 循环。

## 2. JSON 行协议

握手请求：

```json
{"protocol":1,"type":"ping"}
```

握手应答必需字段：

```json
{"protocol":1,"type":"pong","instance_id":"checkout-a","service_id":"checkout","pid":1234}
```

尝试请求：

```json
{"protocol":1,"type":"request","request_id":"r-001","service_id":"checkout","instance_id":"checkout-a","attempt_no":1,"selection_seq":1000001,"route_revision":1,"payload":{}}
```

尝试应答必需字段：

```json
{"protocol":1,"request_id":"r-001","attempt_no":1,"instance_id":"checkout-a","status":"ok","latency_class":"fast","delay_ms":5}
```

协议层校验全部标识和字段类型。缺失/非法/不匹配返回 `E_REPLY`。客户端套接字显式使用 `settimeout(attempt_timeout_ms / 1000)`；超时来自真实 `socket.timeout`，不使用线程 join 或人工 sleep 伪装。

## 3. 子进程确定性

上游先把 `{"seed", "request_id", "attempt_no"}` 规范化为 JSON 字节，做 SHA-256，取前 8 字节对 100 取模：

- 桶小于 `success_percent`：快速成功；
- 其次小于 `success_percent + slow_percent`：慢速成功；
- 剩余：错误。

声明 `rules` 按列表顺序匹配 `request_prefix`，可覆盖结果与 `fast/slow` 延迟类。实际处理延迟为 `fast_ms` 或 `slow_ms`，由上游真实 `time.sleep` 产生。相同声明、请求 ID 与尝试号在任何平台给出相同的类别和声明延迟。

## 4. 权重选择

有效权重按服务与活动修订计算。实例对按声明顺序生成累计区间。选择位置：

```text
digest = SHA256(service_id || 0x1f || request_id || 0x1f || attempt_no)
position = uint64_be(digest[0:8]) % total_weight
```

位置落入第一个累计区间。该方法显式、稳定、无字典顺序依赖，并允许权重 0 的实例保持 UI 可见但永不被选择。

## 5. 双重预算调度

`BudgetScheduler` 在单一条件锁下维护等待表、全局在飞数、服务在飞数和峰值。等待者包含服务与稳定优先级 `(request_seq, attempt_no)`。派发时按优先级扫描，只有同时满足全局与服务配额才授予槽位。

授予时间写入 `wait_end_mono_ms`，加入时间写入 `wait_start_mono_ms`。槽位从授予持续到套接字调用返回。退避 sleep 在释放槽位并提交尝试之后执行。测试 `test_retry_backoff_does_not_hold_global_budget` 使用全局 1、220 ms 退避，验证另一服务尝试在退避间隙内实际运行。

## 6. 熔断状态机

- CLOSED：失败计数递增；成功清零；达到阈值进入 OPEN。
- OPEN：记录单调 `open_until`。冷却内 `before_attempt` 拒绝，记录 `circuit_open` 快速失败且 `budget_used=0`。
- HALF_OPEN：冷却到期后首次选择触发；每代最多放行 `half_open_probes` 个探测。成功数达到试探数则 CLOSED；任一探测失败则 OPEN 并重新计时。

许可中携带 generation，过期探测结果不会污染新一代状态。持久化时记录墙钟 `open_until_wall_ms`，恢复时折算为新的单调时钟截止时间；运行期冷却计时始终使用 `time.monotonic()`。

## 7. SQLite 账本

表：

- `ledger_meta(key,value)`：整数格式版本、内容摘要、活动修订、实例版本、配置摘要、全局/服务限制等。
- `requests`：请求序号、服务、payload、状态、重跑 epoch、最终结局。
- `attempt_intents`：已获得预算、正在真实套接字尝试但尚未提交的意图。
- `attempts`：尝试轨迹、选择序号、结局、错误码、预算等待、在飞数、熔断后状态。
- `budget_usage`：每次尝试的全局/服务槽位、限制、累计预算用量、等待和运行时间。
- `outcomes`：最终请求结局与尝试次数。
- `unknown_attempts`：恢复时由遗留意图转换的未知尝试。

`commit_attempt` 在一个 `BEGIN IMMEDIATE` 事务中插入 attempt、budget、可选 outcome，删除 intent，更新 request，并重算摘要。进程在事务中崩溃时 SQLite 回滚，重启只看见完整提交。

内容摘要覆盖格式版本、除自身外的全部 meta、所有表的有序规范行。打开账本时重算并与存储摘要比较；不一致返回 `E_SNAPSHOT`。

## 8. 快照

快照 envelope：

```json
{"format_version":1,"created_wall_ms":0,"digest":"...","payload":{}}
```

payload 包含路由修订、实例集合版本、预算、熔断、完成游标、账本格式版本、账本摘要和请求序列摘要。摘要为 payload 规范 JSON 的 SHA-256。文件名使用毫秒时间戳与摘要前缀；若同毫秒碰撞则递增时间戳，确保 `load_latest()` 按名称获得真正最新快照。

验证失败时：

1. 记录结构化 `E_SNAPSHOT`。
2. 从账本已提交尝试重建权威状态。
3. 写出全新快照。
4. 对旧/新状态输出稳定排序的差异列表。

验证通过时，正常重开会把快照中的预算和熔断状态装回运行时；崩溃恢复路径则明确传入 `apply_state=false`，只使用已提交账本重建。摘要不符时，旧 payload 仅以“未验证只读解析”用于生成差异，绝不作为权威状态。不进行逐项增量合并。

## 9. 强制终止恢复

恢复调用 `Ledger.recover()`：

1. `attempt_intents` 中每行插入/保留 `unknown_attempts` 并置 `state=unknown`。
2. 非完成请求置 unknown。
3. 预算按 `attempts.budget_used=1` 计数重建。
4. 每个实例取最后提交的 `breaker_after_json`；OPEN 剩余冷却按墙钟差换算。
5. 完成游标为最大 `attempt_id`。
6. 返回 `exit_report_available=false`。

恢复重跑使用请求新 `run_epoch`，保留旧已提交历史并避免唯一键冲突。账本中记录的实例版本与配置摘要会与当前声明比对，不一致以 `E_SPEC` 拒绝解释旧账本。

## 10. 一致性重放

`compare_ledgers` 按源账本请求序号逐项比较：

- 请求存在性、服务、序号；
- 最终结局、错误码、尝试次数；
- 每次尝试的 `attempt_no`、`instance_id`、`select_seq`、`outcome`、`error_code`、`budget_used`、`route_revision`、`upstream_delay_ms`、`final_attempt`。

不比较墙钟、单调时间、实际网络耗时等非确定字段。首个不一致抛出 `E_REPLAY`，包含请求 ID、服务、字段路径、尝试号与期望/实际值。

## 11. 整数定点与排名的误差语义

所有值在存储与比较层保持整数；界面只在显示时把定点值除以缩放因子。P95 使用 nearest-rank，不做插值。两两比较对所有服务对生成稳定顺序记录；差值小于等于合成误差界时，以服务 ID 升序作为决胜，并标记 `within_error_bound`。排名分组也采用稳定服务 ID 次序。

## 12. UI 只读刷新

`build_view_model` 只调用 SELECT 和纯内存函数，读取开始/结束 `content_digest` 并返回 `ledger_refresh.unchanged`。Tk 界面每秒刷新时重新构建视图，不写 SQLite、不写快照、不改变熔断状态。只有点击运行、写快照或关闭进程才产生显式写操作。

## 13. Windows 细节

- 端口预检与上游绑定在 Windows 使用 `SO_EXCLUSIVEADDRUSE`，避免共享监听；POSIX 使用 `SO_REUSEADDR` 以允许快速重启。
- 上游命令行以 UTF-8 环境启动，行为 JSON 通过参数传递。
- 实验样本端口为 `47321` 至 `47324`，均由 `check-ports` 预先验证。
- 不使用 Docker、WSL 专属 API 或容器运行时。
