# 诊断报告结构

每次 `matgen run` 生成 `diagnostic_report.json`。数值数组同时以完整嵌套列表写入报告（`embed_pixel_diffs=true` 时），并另存为 NPZ，便于逐像素复核。

## 顶层字段

| 字段 | 含义 |
| --- | --- |
| `schema_version` | 报告结构版本，当前为 `1`。 |
| `report_type` | 固定为 `texture_material_generation_diagnostic`。 |
| `status` | `ok`、`partial` 或 `failed`。 |
| `input` | 输入目录、配置哈希、纹理/材质/命令数量、定点与调度参数。 |
| `codes` | 去重后的业务码。 |
| `issues` | 隔离后的错误清单，含 `code/message/source/item/details`。 |
| `layers[]` | 每层键、纹理引用、参数、量化步长、整数范围、残差统计、内容哈希。 |
| `materials[]` | 每材质最终通道摘要、边界样本、贡献层列表。 |
| `final` | 最终键、总摘要和逐材质逐通道摘要。 |
| `error_trace` | 两档位误差传播、首次越界环节、归因与逐像素差值。 |
| `full_rebuild` | 增量结果与空白全量重建的逐通道逐像素一致性。 |
| `cache_ledger` | 纹理/层/通道/最终合成缓存项、复用与重算列表。 |
| `scheduler` | 批次、预算扣减序列、`E_POINT_BUDGET_PENDING` 事件。 |
| `interactive` | 每条命令延迟、stale 标记、摘要修订、进度和恢复信息。 |
| `injections` | 注入编号、预期码、观测码、触发次数和匹配结果。 |
| `snapshot` | 恢复判定、最终提交、清单版本、阶段哈希和产物版本。 |
| `artifacts` | 报告、NPZ 与预览 PNG 路径。 |

## 误差追踪

```json
{
  "configured_bit_width": 20,
  "high_bit_width": 24,
  "stage_order": ["noise_composition", "quantization", "material_mix", "channel_synthesis"],
  "stages": {
    "noise_composition": {
      "tolerance": 0.00001,
      "max_abs_diff": 0.0,
      "mean_abs_diff": 0.0,
      "nonzero_pixels": 0,
      "over_tolerance_pixels": 0,
      "first_over_tolerance": null
    }
  },
  "first_exceeded_stage": "quantization",
  "final_consistent": false,
  "final_mismatch_pixels": 1024,
  "first_exceeded_stage_by_pixel": [["quantization", "..."]],
  "attribution": {
    "category": "quantization_truncation",
    "final_mismatch_pixels": 1024,
    "quantization_over_tolerance_pixels": 1024,
    "rounding_boundary_mismatch_pixels": 0,
    "mix_order": {
      "supported": false,
      "max_abs_diff": 0.0,
      "per_channel": {}
    },
    "evidence": {
      "quantization_max_abs_diff": 0.0012,
      "material_mix_max_abs_diff": 0.0024,
      "channel_synthesis_max_abs_diff": 0.0024,
      "final_consistency_tolerance": 0.00001,
      "reverse_order_max_abs_diff": 0.0,
      "boundary_mismatch_pixels": 0
    }
  }
}
```

`attribution.category` 取值为：

- `quantization_truncation`：量化环节已有超容差差值并传播到最终通道。
- `rounding_boundary`：最终不一致像素与半值/权重边界样本重合。
- `mix_order`：顺序敏感混合模式的反序反事实差值超过最终容差。
- `none`：最终通道在容差内一致。

完整证据始终保留，便于审计分类优先级。

## 交互响应

每条响应至少包含：

- `command_id`、`requested_revision`、`summary_revision`
- `stale`、`within_budget`、`elapsed_ms`、`latency_budget_ms`
- `completed_layers`、`total_layers`、`progress`、`requested_progress`
- `affected_layers`、`affected_channels`
- `final_channel_summary`、`revision_digest`
- `codes`，stale 时包含 `E_RESPONSE_STALE`

## 注入预期结构

独立预期文件见 `samples/injected/expected_diagnostic_structure.json`。验证时应检查：

1. `injections.events[].id` 与注入配置 ID 一致。
2. 触发项的 `expected_code` 出现在 `observed_codes` 中。
3. `all_triggered_codes_matched` 为 `true`。
4. 未被本次场景触发的注入出现在 `untouched_injections`，不得伪造观测码。
5. 单张坏纹理、单个材质失败不会导致其他有效产物的清单缺失。
