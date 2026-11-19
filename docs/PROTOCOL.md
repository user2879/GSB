# 本地 TCP 协议

## 帧格式

- 传输：TCP，默认 `127.0.0.1:8765`。
- 每帧一行 UTF-8 JSON，以 `\n` 结束，最大 32 MiB。
- 协议名：`chunkdeps-jsonl/v1`。
- 客户端可为一帧新建连接；服务端支持并发连接。

请求：

```json
{"protocol":"chunkdeps-jsonl/v1","request_id":"req-...","op":"create_session","params":{}}
```

成功响应：

```json
{"request_id":"req-...","ok":true,"result":{}}
```

失败响应：

```json
{"request_id":"req-...","ok":false,"error":{"code":"E_CHUNK","message":"...","details":{}}}
```

所有服务端输出 JSON 使用排序键和紧凑分隔符。客户端校验 `request_id` 后解包。

## 操作

### `ping`

返回服务版本、端口、运行时间、并发限制、峰值与恢复会话数。

### `create_session`

```json
{
  "object_key": "archive/demo/sample.bin",
  "chunk_size": 1024,
  "total_size": 8249,
  "digest_algorithm": "sha256",
  "tags": {"team": "storage"},
  "expected_base_version": 0
}
```

返回会话标识、位图、连续缺失索引和依赖图节点标识。空文件或非法分片大小返回 `E_CHUNK`。

### `upload_chunk`

```json
{
  "session_id": "s-...",
  "chunk_index": 0,
  "data_b64": "AAECAw...",
  "digest": "sha256-hex",
  "inject": null
}
```

成功结果包含当前会话、覆盖标记、依赖失效事件和服务端调度记录：

```json
{
  "schedule": {
    "queue_ms": 7,
    "start_mono_ms": 123456,
    "end_mono_ms": 123463,
    "duration_ms": 7,
    "global_limit": 3,
    "session_limit": 2
  }
}
```

### `resume_session`

返回 `skipped_indices`、`missing_indices` 和当前位图。`aborted -> uploading` 会产生新的会话事件；`committed` 返回 `E_STATE`。

### `abort_session`

```json
{"session_id":"s-...","reason":"operator aborted session"}
```

分片登记和文件保留。返回 `chunks_retained` 与暂存依赖失效事件。

### `commit_session`

```json
{"session_id":"s-...","expected_base_version":1,"inject":null}
```

`inject` 可选 `corrupt-metadata` 或 `version-conflict`：前者演示登记摘要损坏导致的 `E_INTEGRITY`，后者强行使用过期基础版本并返回 `E_VERSION`，会话保持 `uploading`。成功结果包含新版本、旧版本只读视图、字段差异、前后摘要、视图失效快照和失效事件。

### `delete_object`

```json
{"object_key":"archive/demo/sample.bin"}
```

将活动版本改为 `deleted`，沿 `metadata -> key -> prefix-view` 失效。历史过期对象文件仍保留。

### 查询

- `get_session`: 会话、分片、事件与调度历史。
- `list_sessions`: 可按 `state`、`object_key` 过滤。
- `list_objects`: 活动对象、全部版本与计数。
- `list_views`: 全部前缀视图或指定 `prefix`。
- `get_graph`: 节点、边、全局图版本和无环/引用校验；可带事件。
- `invalidations`: 最近失效事件。
- `schedules`: 按会话或全局读取调度记录。
- `status`: 路径、计数、图版本、恢复摘要。
- `recovery_status`: 未知合并、已提交会话和退出报告可用性。

### `audit`

成功返回 `passed=true`、对象/分片计数、对象级摘要和审计摘要。失败返回 `E_AUDIT`，`details.first_deviation` 给出首个偏差对象和分片。

### `snapshot_save` / `snapshot_restore`

`path` 可省略。保存返回状态摘要和计数。恢复返回：

- `mode="whole-state-replacement"`；
- `incremental_merge=false`；
- `comparison.metadata_version_changes`；
- `comparison.chunk_set_changes`；
- 文件清单差异和是否要求重新审计。

## 本地场景 JSON

```json
{
  "defaults": {"concurrency": 3, "commit": true},
  "steps": [
    {
      "op": "upload_file",
      "object_key": "archive/demo/sample.bin",
      "source": "windows-sample.bin",
      "chunk_size": 1024,
      "tags": {"case": "v1"}
    },
    {"op": "audit"},
    {"op": "snapshot_save"},
    {"op": "delete", "object_key": "$last_object"},
    {"op": "snapshot_restore"}
  ]
}
```

相对 `source` 以场景文件所在目录为基准。步骤结果会自动更新 `$last_session`、`$last_object` 和 `$last_version`；也可用 `as` 指定变量。`rpc` 步骤可透传 `rpc_op` 与 `params`。
