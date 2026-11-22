# 非法输入与故障注入

所有故障均由外部文件触发，不需要修改源码。退出码 `1` 表示输入校验失败，`3` 表示运行完成但存在孤立失败实体，`2` 表示运行被前置条件拒绝。

## 独立非法输入

| 目录 | 单独触发项 | 预期码 |
| --- | --- | --- |
| `data/illegal/duplicate_row` | 重复 `row_id` | `E_ROW_INVALID` |
| `data/illegal/nonfinite_row` | JSON 常量 `NaN` | `E_ROW_INVALID` |
| `data/illegal/timestamp_reversal` | 同实体时间戳回退 | `E_ROW_INVALID` |
| `data/illegal/bad_rows_all` | 三类坏行组合，用于观察隔离汇总 | `E_ROW_INVALID` |
| `data/illegal/bad_dag` | 特征依赖环 | `E_FEATURE_DAG_INVALID` |
| `data/illegal/missing_upstream` | `source_feature` 不存在 | `E_FEATURE_DAG_INVALID` |
| `data/illegal/missing_raw_field` | `source_field` 不在声明字段中 | `E_FEATURE_DAG_INVALID` |
| `data/illegal/bad_config_quota` | 单批点数配额为 0 | `E_CONFIG_INVALID` |
| `data/illegal/bad_config_lease_ttl` | 租约 TTL 为 0 | `E_CONFIG_INVALID` |
| `data/illegal/bad_plan` | 未知计划操作 | `E_PLAN_INVALID` |

单独运行示例：

```bash
.venv/bin/python -m feature_backfill validate \
  --samples data/illegal/duplicate_row/samples.jsonl \
  --features data/illegal/duplicate_row/features.json \
  --config data/illegal/duplicate_row/config.yaml \
  --plan data/illegal/duplicate_row/plan.json
```

## 运行时注入清单

清单可直接是数组，也可为 `{"injections": [...]}`。字段：

```json
{
  "id": "INJ-001-NONFINITE",
  "type": "nonfinite_value",
  "stage": "window_features",
  "entity_id": "E003",
  "row_id": "R-E003-002",
  "field": "amount",
  "expected_code": "E_ROW_INVALID"
}
```

| 文件 | 类型 | 目标 | 预期码 | 实际作用点 |
| --- | --- | --- | --- | --- |
| `injections/nonfinite.json` | `nonfinite_value` | `E003` | `E_ROW_INVALID` | 复制行写入 `NaN` 后调用正式行解析器 |
| `injections/timestamp_reversal.json` | `timestamp_reversal` | `E004` | `E_ROW_INVALID` | 复制行改写时间戳后调用严格递增校验 |
| `injections/lease_ttl_zero.json` | `lease_ttl_zero` | `E005` | `E_CONFIG_INVALID` | 租约签发前以 TTL=0 调用真实租约授权 |
| `injections/cache_budget_zero.json` | `cache_budget_lt_block` | `E006` | `E_CONFIG_INVALID` | 单块预算下注入容量 0 的运行时缓存故障 |
| `injections/reference.json` | 四类组合 | `E003`–`E006` | 分别映射 | 验证失败清单稳定排序 |

运行单点注入（使用空计划，避免后续批次重试同一实体）：

```bash
.venv/bin/python -m feature_backfill run \
  --samples data/default/samples.jsonl \
  --features data/default/features.json \
  --config data/default/config.yaml \
  --plan data/scenarios/empty_plan.json \
  --injections injections/nonfinite.json \
  --snapshot-dir results/inj-nonfinite/snapshots \
  --report results/inj-nonfinite/report.json
```

退出码为 `3`。报告中对应实体的 `injection_hits[].injection_id`、`observed_code` 和 `failures[].injection_id` 必须一致；未命中其他实体，`pending` 为空，其余实体完成。

## 租约演示

```bash
.venv/bin/python -m feature_backfill lease-demo --ttl-ms 60
```

输出包含 `lease_conflict`、`lease_reclaimed` 和第二次 `lease_granted`。竞争线程在条件变量上等待单调时钟越过 `expires_ns`，不是固定睡眠后直接覆盖。

## 自定义注入

支持四种 `type`：`nonfinite_value`、`timestamp_reversal`、`lease_ttl_zero`、`cache_budget_lt_block`。`id` 全局唯一，会出现在 `summary.json`、完整报告的 `injection_hits` 与 `failures` 中。注入运行时按注入 ID 排序，多个实体并发时报告顺序仍稳定。
