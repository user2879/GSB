# 设计、预算与恢复不变量

## 1. 组件分层

```text
JSON config --strict--> TunnelConfig
                         |-- Route listeners (real TCP on 127.0.0.1)
                         |-- SequenceAllocator (per tunnel, lock protected)
                         |-- Envelope encoder / MTU fragmenter
                         |-- BudgetManager (two concurrency gates + two token buckets)
                         |-- Ledger (exactly-once, canonical JSON digest)
                         |-- SnapshotStore (checksum + fsync + os.replace)
                         `-- ReplayVerifier (reencode/reassemble/reapply behavior)

inbound TCP client --> engine session thread --> outbound TCP peer
                                                       |
                                              decode/reassemble/digest
                                                       |
                                                  JSON response
```

一个入站 TCP 连接对应一个会话和一个逻辑消息。客户端写完负载后半关闭写方向；引擎读取到 EOF 后分配/确认序号、封装、按 MTU 分片并转发。服务端回送 JSON 结局，引擎同时写账本与结构化日志。

## 2. 四重预算不变量

任一时刻：

```text
active_global <= global.max_sessions
active_route[t] <= route[t].max_sessions
global_token_units >= 0
route_token_units[t] >= 0
```

并发许可与两级令牌扣减都由同一把锁保护。带宽申请先把全局与隧道桶在同一临界区刷新，再原子判断/扣减；不能只扣一个桶。任一桶不足时不预留令牌，计算到下一次足够的时间并等待，随后重试。

令牌桶用 `TOKEN_SCALE=1_000_000` 的整数微字节单位，补充依据 `time.monotonic_ns()`。容量为配置速率的一秒字节数，因此允许最多一秒突发，但长期速率不超过配额。引擎按 `min(全局速率, 隧道速率)` 切分网络写块，保证单次申请不超过任一桶容量。

每次真正等待都会产生 `QueueEvent(start_ms, end_ms, duration_ms)`；并发许可、全局带宽、隧道带宽引起的等待都计入会话 `queue_count` 与 `total_queue_ms`。

## 3. 账本与定点统计

- 账本键是 `session_id`；第二次写入抛 `E_STATE`。
- 记录按 `(start_ms, session_id)` 规范排序后再计算摘要，避免插入顺序影响。
- `payload_bytes` 是输入负载大小；`forwarded_bytes` 是完整送达对端的逻辑负载大小；`wire_bytes` 是实际写出的线字节（含 4 字节长度前缀）。
- 吞吐只使用 `forwarded_bytes`，避免把连接失败会话的输入算作已转发。
- 吞吐 `throughput_kbps_fixed = round_half_up(forwarded_bytes * 1_000_000 / (1024 * span_ms))`，单位 milli-KB/s。
- 平均排队 `average_queue_ms_fixed = round_half_up(total_queue_ms * 1000 / session_count)`，单位 milli-ms。
- `scale=1000`；每个固定点结果的四舍五入最大误差为 `0.5 / 1000 = 0.0005` 对应单位。

## 4. 快照与整体替换

快照 payload 包括：

- `route_version` 与完整规范化配置；
- 每隧道封装序号水位；
- 全局与每隧道预算已用量及令牌余额；
- 账本记录与账本摘要；
- 每会话状态、负载 Base64、队列事件、字节数、结局与错误；
- 已计算统计结论、对端行为定义与每会话预算用量视图。

文件外层包含格式名、版本与 payload SHA-256；写入使用同目录临时文件、`flush + fsync`、`os.replace`，再原子更新 `latest.json`。只有完整文件会被视为可恢复快照，`.part` 残留不会被列出或加载。

恢复时：

1. 校验外层摘要、配置、账本摘要和统计重算值；
2. 把所有 `RUNNING/QUEUED` 或缺少结束时间的会话改为 `UNKNOWN` 且 `requires_rerun=true`；
3. 校验未知会话没有账本记录、结束会话必须有账本记录；
4. 校验账本序号不超过快照水位；
5. 比较当前配置与快照配置，输出稳定排序的路由/配额变化；
6. **始终整体采用快照配置、预算、账本与会话**，不向当前进程账本增量合并。

## 5. 严格重放

重放以快照的路由版本、负载、序号、时钟观测和对端行为为输入：

- 重新编码每个负载并验证帧长不超过 MTU；
- 解码每帧、校验摘要和序号，再重组；
- 根据快照中的对端行为重演 `OK/E_PEER/E_TIMEOUT/E_DIGEST`；
- 使用记录的 start/end/queue 观测重建新账本并重算统计；
- 依次比较会话结局、输入字节、已转发字节、封装摘要、分片数、账本字段和全局/隧道统计。

第一处差异返回 `E_REPLAY`，并填充 `tunnel_id`、`session_id`、`field`、`expected`、`actual` 与判定依据。物理网络时间不参与严格比较，因此重启后结果逐字节稳定；真实套接字行为由总演示和集成测试单独验收。

## 6. 错误映射

- 配置阶段所有错误统一为 `E_CONFIG`，字段路径精确到 `routes[i].mtu` 等。
- 帧解析失败统一为 `E_ENCAP`，尽量携带可解析出的隧道 ID 和首个出错偏移。
- 超时缺片为 `E_FRAG`，同时给出缺失偏移列表。
- 对端连接/写入失败为 `E_PEER`；等待回送超时为 `E_TIMEOUT`；回送摘要不符为 `E_DIGEST`。
- 快照/账本一致性失败为 `E_STATE`；重放差异为 `E_REPLAY`。
