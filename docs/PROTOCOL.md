# 协议与确定性规则

本文定义实现使用的逐字节规则。所有多字节整数均为大端序；字符串使用 UTF-8；摘要使用 SHA-256。

## 1. 通用分片头

`common` 字段序列：

| 字段 | 类型 | 说明 |
|---|---:|---|
| `tunnel_id_len` | `u16` | 隧道标识 UTF-8 字节长度 |
| `tunnel_id` | bytes | 隧道标识 |
| `fragment_index` | `u32` | 从 0 开始的分片序号 |
| `fragment_offset` | `u32` | 内层载荷字节偏移 |
| `payload_length` | `u32` | 本片载荷长度 |

数据报前缀为：

```text
b"TUNF" || format_version:u8 || common
```

`format_version`：f1=`1`，f2=`2`，f3=`3`。未知值返回 `E_FORMAT`。

## 2. 线格式

### f1

```text
prefix || payload
```

只携带隧道标识、分片序号、分片偏移和本片载荷。载荷长度由通用头显式携带，实际载荷占满剩余数据报。

### f2

```text
prefix
  || total_fragments:u32
  || message_id:32 bytes
  || generated_at_ms:u64
  || checksum:32 bytes
  || payload
```

f2 校验码规则：

```text
checksum = SHA256(
  b"TUNF-F2-CHECKSUM\0"
  || common
  || total_fragments:u32
  || message_id:32
  || generated_at_ms:u64
  || payload
)
```

### f3

```text
prefix
  || total_fragments:u32
  || message_id:32 bytes
  || generated_at_ms:u64
  || checksum:32 bytes
  || encapsulation_version:u16
  || retry_count:i32
  || fragment_header_digest:32 bytes
  || payload
```

f3 保留同样的 f2 校验码，并增加分片头摘要：

```text
header_digest = SHA256(
  b"TUNF-F3-HEADER\0"
  || common
  || total_fragments:u32
  || message_id:32
  || generated_at_ms:u64
  || checksum:32
  || encapsulation_version:u16
  || retry_count:i32
)
```

`retry_count` 读取为有符号整数，负数返回 `E_FIELD`。`encapsulation_version` 必须大于零。

## 3. 字段与一致性校验

- 未知格式版本、错误 magic、尾部多余字节：`E_FORMAT`。
- 必需字段截断、零总片数、序号不小于总片数、负重传计数、校验码或头摘要不匹配：`E_FIELD`。
- 载荷长度超过端点有效载荷上限 `mtu - encapsulation_overhead`：`E_FIELD`。
- 载荷长度超过 `fragment_size` 档位：`E_FIELD`。
- 偏移必须按 `fragment_size` 对齐：`E_FIELD`。
- `offset + payload_length` 必须不超过消息重组上限；当总片数已知时还必须不超过 `total_fragments * fragment_size` 以及 `total_fragments * effective_payload_limit`：`E_FIELD`。

`WireContext` 提供 `fragment_size`、`effective_payload_limit`、`max_message_bytes`、消息标识、生成时刻、总片数和封装版本。f1 非零偏移归一必须提供消息清单上下文。

## 4. 统一内部视图与补齐

统一视图字段：

```text
tunnel_id, fragment_index, fragment_offset, payload, total_fragments,
message_id, generated_at_ms, checksum, encapsulation_version,
fragment_header_digest, retry_count, source_format,
padded_fields, padding_basis
```

来源格式缺少字段时的补齐规则：

| 来源 | 补齐字段 | 依据 |
|---|---|---|
| f1 | 总片数、报文标识、生成时刻、校验码、f3 三字段 | 消息清单上下文、f1 单片段推断、epoch `0` 默认、显式校验规则计算 |
| f2 | 封装版本、分片头摘要、重传计数 | 默认封装版本 `1`、默认重传 `0`、按 f3 摘要规则计算 |
| f3 | 无 | 原生完整性 |

迁移后的 `semantic_key()` 包含所有归一关键字段；f1/f2/f3 变体的该对象必须逐字段相等。`semantic_key()` 的规范化 JSON 摘要用于 `semantic_signature`。

## 5. 确定性内层样本

路由按以下稳定键排序：

```text
(IP version, network address integer, prefix length, tunnel_id, next_hop, priority, prefix text)
```

采样种子：

```text
material = canonical_json({
  "domain": "tunnel-interop/sample-seed/v1",
  "endpoint_id": ...,
  "endpoint_seed": endpoint.seed,
  "tunnel_id": ...,
  "destination_prefix": ...
})
rng_seed = first_16_bytes(SHA256(material)) interpreted as big-endian integer
```

载荷按固定版本 NumPy 规则生成：

```python
rng = numpy.random.Generator(numpy.random.PCG64(rng_seed))
payload = rng.integers(0, 256, size=n, dtype=numpy.uint8).tobytes()
```

默认 `n = 3 * fragment_size + 29`；声明中的 `sample.bytes_per_message` 可覆盖。该规则可由 `rng_seed`、长度和 NumPy `1.26.4` 逐字节复算。

报文标识：

```text
message_id = SHA256(canonical_json({
  "domain": "tunnel-interop/message-id/v1",
  "tunnel_id": ...,
  "destination_prefix": ...,
  "next_hop": ...,
  "priority": ...,
  "payload_sha256": ...,
  "payload_length": ...,
  "fragment_size": ...
}))
```

生成时刻为固定样本纪元加稳定偏移：

```text
generated_at_ms = 1700000000000 + (first_12_hex_chars(SHA256(message_id)) mod 86400000)
```

这使 `message_id`、生成时刻、片偏移、片长度、三种格式封包和调度决策均不依赖端点加载顺序或 Python 字典遍历顺序。

## 6. 分片与重组

- `total_fragments = max(1, ceil(payload_length / fragment_size))`。
- `fragment_index = k` 的偏移为 `k * fragment_size`。
- 每片长度不超过档位，最后一片允许更短。
- 每片生成 f1/f2/f3 三个封包变体，主格式由 `runtime.fragment_format` 选择。
- 重组以偏移严格升序进行；重复偏移只有在长度和载荷字节一致时才去重。
- 偏移必须从 0 连续覆盖，分片索引集合必须完整，载荷摘要必须等于固定样本摘要。

## 7. 失效传播

传播图：声明字段 → 端点/路由/策略 → 内层消息 → 分片 → 重组结果。

- 端点 `mtu`、`encapsulation_overhead`、`seed`：关联路由的消息、全部分片和重组结果失效。
- 端点 `port`、`host`、`replay_window`：线上分片字节可保留，仅投递/重组结果失效。
- 路由 `prefix`、`next_hop`、`priority`、`tunnel_id`：路由消息、分片和重组失效。
- 路由选择结果变化：新旧被选消息均失效。
- `fragment_size`、`max_retries`：全部分片和重组失效。
- `reassembly_timeout_ms`：分片字节保留，仅重组结果失效。
- `runtime.fragment_format`、样本策略或派生消息标识变化：相关分片与重组失效。

失效分片的 `fragment_version` 在重算时递增；未失效分片直接复用旧封包对象，封包 SHA-256 和版本号保持不变。传播步骤按声明、端点、路由、策略、消息、分片的稳定顺序输出。

## 8. 调度口径

调度器采用确定性准入波次：每轮从稳定排序队列中尽可能多地准入满足以下条件的任务：

```text
global_active < global_inflight_limit
and tunnel_active < per_tunnel_inflight_quota
```

任务提交给真实 `ThreadPoolExecutor`，本轮全部完成后才开始下一轮准入。因此实际并发不会超过两个上限；配额不足的任务保持排队。

每次准入记录真实 `time.monotonic_ns()/1e6` 的队列开始、派发时刻和耗时，写入 `real_trace`，用于审计。可复现统计使用逻辑单调时钟：初始队列时刻为 `0.0`，第 `w` 个准入波次派发时刻为 `w.0` 毫秒，等待时长为 `w.0` 毫秒。`deterministic_statistics` 包含全局和单隧道峰值、排队分片数、每波峰值、每任务逻辑等待和自身 SHA-256 摘要。相同输入与相同上限下该对象逐字节一致。

正常与重复场景使用声明并发上限；反序验收场景将该批次有效全局和单隧道并发都设为 1，从而由真实串行收发产生稳定反序到达，不改变声明文件，也不以人为睡眠伪造到达顺序。

## 9. 账本与恢复

SQLite 表：

- `meta`：账本格式版本、活动 generation、声明快照、账本内容摘要。
- `batches` / `planned_fragments`：批次 intent 和计划分片。
- `fragments`：分片归属、偏移、格式、重传计数、分片版本、封包 Base64 与 SHA-256。
- `scheduling`：确定性统计和真实调度事件。
- `reassembly`：重组载荷与摘要。
- `progress`：进度游标。
- `snapshot_history`：快照整体替换历史。

账本摘要只覆盖 `status='committed'` 的批次，因此新 intent 不会污染已提交内容摘要。完成批次在一个 `BEGIN IMMEDIATE ... COMMIT` 中写入所有完成记录并更新摘要。重启时：

1. 验证整数账本格式版本。
2. 重算并比较账本内容摘要；不符返回 `E_SNAPSHOT`。
3. 比较声明版本、端点/路由/分片指纹和声明内容摘要。
4. 不一致时在一个事务中提升 generation、归档旧快照并创建全新的空活动快照。
5. `intent`/`failed` 批次转为 `unknown`；列出计划分片，要求重跑，禁止声称可生成退出报告。

重放按批次标识和分片序号的稳定顺序重新生成固定样本、对应格式封包和重组结果，逐字段比较。第一处差异以 `E_REPLAY` 返回，路径形如：

```text
batches[tun-alpha].fragments[2].wire_sha256
```

## 10. 错误码

| 业务码 | 含义 |
|---|---|
| `E_SPEC` | 声明损坏、字段非法、版本不单调 |
| `E_REF` | 路由下一跳引用悬空（注入审计模式） |
| `E_FORMAT` | 未知格式版本、magic 或线结构错误 |
| `E_FIELD` | 必需字段、字段值、校验码、头摘要或回执字段错误 |
| `E_TIMEOUT` | 真实 socket 超时或真实重组截止期耗尽 |
| `E_SNAPSHOT` | 账本格式、内容摘要或快照元数据不可信 |
| `E_REPLAY` | 恢复重放首处不一致 |
