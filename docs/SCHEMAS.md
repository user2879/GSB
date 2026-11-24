# 数据格式与业务码

## 主配置 `samples/config.json`

```json
{
  "schema_version": "mmledger-config-v1",
  "workspace_root": "..",
  "seed": 20261125,
  "model": {
    "input_dim": 16,
    "hidden_dim": 8,
    "quantization_bits": 8
  },
  "limits": {
    "batch_size": 3,
    "normalization_budget": 64,
    "worker_quota_points": 20
  },
  "port_range": {"min": 18800, "max": 19400},
  "parameters": [
    {"parameter_version": 3, "digest": "sha256:pv3-alpha"}
  ],
  "workers": [
    {
      "worker_id": "w-alpha",
      "host": "127.0.0.1",
      "port": 18900,
      "parameter_version": 3,
      "parameter_digest": "sha256:pv3-alpha",
      "quota_points": 20
    }
  ],
  "request_stream": "requests.json",
  "merge_rules": "merge_rules.json",
  "receipt_glob": "receipts/*.json",
  "fault_injections": "fault_injections.json",
  "fault_receipt_glob": "fault_receipts/*.json"
}
```

相对路径均相对于配置文件所在目录；`workspace_root` 决定 `snapshots/` 与 `reports/` 的根。

## 请求流

```json
{
  "schema_version": "request-stream-v1",
  "requests": [
    {
      "sequence": 1,
      "request_id": "req-001",
      "kind": "image_text",
      "arrival_ms": 7,
      "batch_id": "batch-001",
      "worker_id": "w-alpha",
      "input_seed": 1001,
      "overwritten_ref": null
    }
  ]
}
```

约束：`sequence` 严格递增、`arrival_ms >= 0`、`batch_size > 0`、同批请求数不超过批大小、`kind` 为 `image_text|image_only`、工作进程存在。`request_id` 可跨批重复，以表达重复投递。

## 归并规则

```json
{
  "schema_version": "merge-rules-v1",
  "worker_sets": {"active": ["w-alpha", "w-beta"]},
  "parameter_version_sets": {"production": [3, 4]},
  "active_worker_set": "active",
  "active_parameter_version_set": "production",
  "worker_priority": ["w-alpha", "w-beta"]
}
```

所有集合名称、集合内工作进程和参数版本都必须存在。

## r1 / r2 / r3 回执

文件可以是单条对象、列表或信封。推荐信封：

```json
{
  "version": "r3",
  "receipts": [
    {
      "batch_id": "batch-003",
      "request_id": "req-007",
      "output_digest": "sha256:...",
      "parameter_version": 3,
      "duration_ms": 12,
      "worker_id": "w-alpha",
      "stable_rank": 1,
      "overwritten_ref": null,
      "parameter_digest": "sha256:pv3-alpha"
    }
  ]
}
```

| 版本 | 必需字段 |
| --- | --- |
| r1 | `batch_id`, `request_id`, `output_digest` |
| r2 | r1 + `parameter_version`, `duration_ms`, `worker_id` |
| r3 | r2 + `stable_rank`, `overwritten_ref`, `parameter_digest` |

字段约束：

- 标识字段和摘要为非空字符串；
- `parameter_version` 为正整数且必须存在于配置；
- `duration_ms` 为非负整数；`NaN/Infinity` 产生 `E_DURATION_NONFINITE`；
- `stable_rank` 为正整数；
- `overwritten_ref` 为 `null` 或非空字符串；
- 转换产物中的未提供字段可写 `{"$missing": true}`。

## 转换旁路

`samples/conversion/sidecar.json`：

```json
{
  "entries": {
    "batch-003|req-007|sha256:...": {
      "batch_id": "batch-003",
      "request_id": "req-007",
      "output_digest": "sha256:...",
      "origin_uid": "receipts/r3_batch_003.json#0",
      "restored_fields": {
        "duration_ms": 12,
        "parameter_version": 3,
        "stable_rank": 1,
        "overwritten_ref": null,
        "parameter_digest": "sha256:pv3-alpha",
        "worker_id": "w-alpha"
      }
    }
  }
}
```

转换账本 `conversion_journal.json` 记录每次 `downgrade/upgrade/rewrite`、降级字段、恢复字段、不可逆字段、缺失字段和完整性。

## 账本字段

`reports/ledger.json` 的关键区段：

| 字段 | 含义 |
| --- | --- |
| `counts` | 有效回执、权威请求、批次、失败、待办、配额阻塞数 |
| `failures` | 稳定排序的失败清单 |
| `ledger` | 每条请求的权威归属、候选、裁决依据、名次冲突 |
| `batches` | 归一后的批次候选、冲突和依赖指纹 |
| `resources` | 每批整数 MAC、量化 MAC、利用率、耗时观测/缺失 |
| `quality` | 图文/纯图请求计数和耗时分布 |
| `dependency_invalidation` | 参数版本 → 批次 → 请求的最终依赖边界 |
| `invalidation_reuse_plan` / `cache_plan` | 最终输入派生的规范化失效/复用清单 |
| `scheduler` | 预算、tickets、待办、配额阻塞、逐点扣减 trace |
| `ledger_digest` | 除自身外全部账本字段的 SHA-256 |

## 错误码

| 业务码 | 场景 |
| --- | --- |
| `E_CONFIG_JSON` / `E_CONFIG_SCHEMA` | 配置 JSON/结构错误 |
| `E_WORKER_DUPLICATE_ID` | 工作进程标识重复 |
| `E_WORKER_PORT_RANGE` / `E_WORKER_PORT_DUPLICATE` | 回环端口越界/重复 |
| `E_WORKER_HOST` | 未绑定 `127.0.0.1` |
| `E_WORKER_PARAMETER_UNKNOWN` | 工作进程引用未知参数版本 |
| `E_MODEL_DIMENSION` / `E_QUANTIZATION_BITS` | 模型维度/量化位宽非法 |
| `E_REQUEST_SEQUENCE` / `E_REQUEST_ARRIVAL` | 请求序号/到达时间非法 |
| `E_REQUEST_BATCH_SIZE` / `E_REQUEST_KIND` / `E_REQUEST_WORKER_UNKNOWN` | 批次大小/请求类型/工作进程非法 |
| `E_MERGE_*` | 归并规则集合或成员不存在 |
| `E_LIMIT_INVALID` | 批大小、预算或配额非法 |
| `E_RECEIPT_JSON` / `E_RECEIPT_ENCODING` | JSON/编码错误 |
| `E_TRUNCATED_RECEIPT` | 截断 JSON |
| `E_RECEIPT_VERSION` | 版本标记缺失、未知或文件内混用 |
| `E_RECEIPT_LAYOUT` / `E_RECEIPT_FIELD_TYPE` | 字段布局/类型错误 |
| `E_UNKNOWN_PARAM_VERSION` | 回执引用未知参数版本 |
| `E_DURATION_NONFINITE` | 回执耗时为非有限值 |
| `E_DUPLICATE_BATCH_ID` | 重复投递身份 |
| `E_WORKER_QUOTA_EXCEEDED` | 工作进程累计点数超过配额 |
| `E_NORMALIZE_BUDGET_PENDING` | 预算不足，批次保留待办 |
| `E_CONVERSION_TARGET` | 转换目标版本非法 |
| `E_FRAME_INVALID` | 回执与请求/批次上下文不一致 |
