# 输入格式

所有输入均为本地 UTF-8 文本。配置与注入清单必须是 JSON 对象；样本文件为每行一个 JSON 对象的 JSONL。

## 1. 位置编码配置

示例：`samples/position_config.json`。

```json
{
  "schemes": ["RoPE", "LinearInterpolation", "NTK-aware"],
  "base": 10000.0,
  "train_context_length": 32,
  "extrapolation_target_length": 96,
  "dimension": 16,
  "scaling_factor": 2.0,
  "precision": "float64",
  "concurrency": 4,
  "tolerances": {
    "position_vector": 1e-6,
    "dot_score": 1e-6,
    "normalized_attention_weight": 1e-6
  },
  "sensitivity": {
    "base_values": [0.0, 1e-300, 1000.0, 5000.0, 10000.0, 20000.0, 1e300],
    "scaling_factor_values": [0.0, 0.5, 1.0, 2.0, 4.0, 1e300]
  }
}
```

字段要求：

| 字段 | 要求 |
| --- | --- |
| `schemes` | 非空字符串或字符串列表；支持 `RoPE`、`LinearInterpolation`、`NTK-aware`，并兼容 `线性内插`、`NTK 感知缩放` 等别名。 |
| `base` | 必填、有限且大于 0；缺失或非法进入 `E_CONFIG_INVALID`。 |
| `train_context_length` | 必填正整数。 |
| `extrapolation_target_length` | 必填正整数，必须大于训练长度；合法位置范围为 `[0, target_length)`。 |
| `dimension` | 必填、正偶数；奇数进入 `E_CONFIG_INVALID`。 |
| `scaling_factor` | 必填、有限且大于 0；不得静默填充默认值。 |
| `precision` | `float64`/`fp64` 或 `float32`/`fp32`；报告中的三阶段偏差始终以 float64 计算。 |
| `concurrency` | 必填正整数，即线程池上限。 |
| `tolerances` | `position_vector`、`dot_score`、`normalized_attention_weight` 三项均必填且大于 0。 |
| `sensitivity.base_values` | 绝对基频扫描值；可与 `base_multipliers` 二选一。 |
| `sensitivity.scaling_factor_values` | 绝对缩放因子扫描值；可与 `scaling_factor_multipliers` 二选一。 |

扫描值允许为 0 或负数，以便验证边界；此类值不会被当作正常方案配置，而是记录为边界敏感项。

## 2. 注意力样本 JSONL

推荐格式：

```json
{
  "sample_id": "s-extrap-001",
  "query_position": 64,
  "query_features": [0.1, -0.2, 0.3, 0.4],
  "key_positions": [20, 52, 63],
  "key_features": [[0.2, 0.1, -0.1, 0.0], [0.0, 0.2, 0.1, -0.1], [0.3, -0.2, 0.0, 0.1]]
}
```

也可使用 `keys` 数组：

```json
{
  "sample_id": "s-001",
  "query_position": 8,
  "query_features": [0.1, -0.2, 0.3, 0.4],
  "keys": [
    {"position": 4, "features": [0.2, 0.1, -0.1, 0.0]},
    {"position": 7, "features": [0.0, 0.2, 0.1, -0.1]}
  ]
}
```

约束：

- `sample_id` 在文件内唯一；最终样本按 ID 稳定排序。
- `query_features` 和每行 key 特征维度必须等于配置的 `dimension`。
- `key_positions` 与 `key_features` 数量必须一致且至少一个。
- 位置必须为整数；未注入时还需落在 `[0, extrapolation_target_length)`。
- 特征可以为负；若原始输入含 NaN/Inf，将进入 `E_NONFINITE_INPUT` 失败清单。

`tools/generate_samples.py` 使用固定种子生成这些字段，样本内容指纹由规范化 JSON 的 SHA-256 派生。

## 3. 异常注入清单

```json
{
  "version": 1,
  "injections": [
    {
      "injection_id": "INJ-001",
      "sample_id": "s-inj-nonfinite",
      "stage": "encoding",
      "fault": "nonfinite_feature",
      "target": "query_features",
      "index": 2,
      "value": "nan"
    },
    {
      "injection_id": "INJ-002",
      "sample_id": "s-inj-position",
      "stage": "scoring",
      "fault": "position_out_of_range",
      "target": "key_position",
      "key_index": 0,
      "position": 96
    },
    {
      "injection_id": "INJ-003",
      "sample_id": "s-inj-base",
      "scheme": "RoPE",
      "stage": "encoding",
      "fault": "base_zero"
    },
    {
      "injection_id": "INJ-004",
      "sample_id": "s-inj-scale",
      "scheme": "LinearInterpolation",
      "stage": "encoding",
      "fault": "scale_nonpositive",
      "value": -1.0
    }
  ]
}
```

故障映射：

| `fault` | 业务码 | 真实变异 |
| --- | --- | --- |
| `nonfinite_feature` | `E_NONFINITE_INPUT` | 将 query/key 特征指定位置写为 NaN 或 ±Inf。 |
| `position_out_of_range` | `E_POSITION_OUT_OF_RANGE` | 将指定 query/key 位置写为域外整数。 |
| `base_zero` | `E_BASE_ZERO` | 将当前样本/方案的有效基频写为 0。 |
| `scale_nonpositive` | `E_SCALE_INVALID` | 将当前样本/方案的有效缩放因子写为非正数。 |

可选 `scheme` 将注入限定到一种方案；缺省时作用于该样本的所有方案。所有命中项都会进入 `injection_hits`，并按样本汇总 `injection_hits_by_sample`。注入清单原始对象的规范化哈希写入报告的 `injection_hash`。
