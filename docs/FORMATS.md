# 交换格式与兼容口径

## 1. 统一中间表示

所有版本先解析为 `UnifiedResult`：

- `file_id`
- `total_micro`：`0..1_000_000` 的百万分之一刻度总分。
- `dimensions_micro`：`level`、`noise`、`clipping`、`continuity` 四维百分制分数。
- `level_code` / `level_name`
- `precision_tag`
- `frame_summaries`
- `bypass`

重评产生的运行时候选使用 `source_version=0`，表示它不是从某个交换文件直接解析而来；转换到 v1/v2/v3 后分别写回对应版本号。

解析器在每个版本内部校验文件标识唯一、四维字段完整、分值范围、等级码/名映射。格式错误产生 `E_EVAL_FORMAT`，映射缺失产生 `E_LEVEL_UNMAPPED`，均进入稳定排序的失败清单。

## 2. v1 定宽文本

每行 46 个 UTF-8 字符，固定宽度如下：

| 字段 | 宽度 | 格式 |
| --- | ---: | --- |
| file_id | 24 | 左对齐，空格补齐 |
| total | 4 | 千分比整数，`0000..1000` |
| level | 4 | 四维整数，`0000..1000` |
| noise | 4 | 四维整数，`0000..1000` |
| clipping | 4 | 四维整数，`0000..1000` |
| continuity | 4 | 四维整数，`0000..1000` |
| level_code | 2 | 左对齐，空格补齐 |

示例：

```text
sample_boundary         075506000700100007001 
```

读取时忽略空行和 `#` 注释。行宽、数字字段、标识字符集、重复标识均严格校验。

## 3. v2 JSON

```json
{
  "schema": "aqeval.v2",
  "precision_tag": "float6",
  "results": [
    {
      "file_id": "sample_clean",
      "total_score": 0.92,
      "dimensions": {
        "level": 0.6,
        "noise": 1.0,
        "clipping": 1.0,
        "continuity": 1.0
      },
      "level": "excellent",
      "precision_tag": "reeval_float6",
      "bypass": {}
    }
  ]
}
```

浮点用 `Decimal` 读取并按 half-up 转为微分数，保留精度标签。命名等级必须存在于等级映射表。

## 4. v3 二进制分区

头部 64 字节：

```text
<4sHHIIIIQQQQII
magic b"AQV3", format_version=3, header_size=64,
record_count, frame_count, fixed_bits, flags,
index_offset, frame_offset, metadata_offset, metadata_size,
checksum, reserved
```

分区顺序：

1. `index_offset..frame_offset`：文件定点记录，每条 92 字节。
2. `frame_offset..metadata_offset`：帧摘要，每条 26 字节。
3. `metadata_offset..metadata_offset+metadata_size`：旁路元数据 JSON 拼接区。

文件记录：

```text
<24s9I4s20s8s
file_id[24], total_fixed, level_fixed, noise_fixed,
clipping_fixed, continuity_fixed, frame_start, frame_count,
metadata_offset, metadata_length, level_code[4],
precision_tag[20], reserved[8]
```

帧摘要：

```text
<IIihHHHHI
file_index, frame_index, rms_db_centi, zcr_milli,
peak_q15, max_zero_run, zero_run_ratio_milli,
clipped_samples, reserved
```

校验和是 `sha256(records + frames + metadata)[0:4]` 的大端整数。解析时校验分区偏移、分区长度、记录引用范围与校验和。

定点换算由 `compat.py` 的兼容表控制：v3 分值刻度的最小单位为 `10^(6-fixed_bits)` 微分数；目标位宽不足时 half-up 取整并记录残差。

## 5. 转换与降级

| 转换 | 处理 |
| --- | --- |
| v1 → v2 | 千分比精确乘 1000 转浮点六位；`precision_tag=permille_to_float6`；等级码查表转名称 |
| v2 → v1 | half-up 还原千分比；总分和四维残差、精度标签、被丢弃的名称写入 bypass；产生降级项 |
| v1 ↔ v3 | 通过兼容表转定点整数；v3 分区偏移按全局记录/帧序号重算 |
| v2 ↔ v3 | 微分数与目标定点位宽互转；位宽不足产生 `DEG_V3_FIXED_WIDTH` |
| 无帧摘要的数据 → v3 | 写空帧分区并产生 `DEG_FRAME_SUMMARY_UNAVAILABLE` |
| v3 → v1 / v3 → v2 | 目标格式无法表达帧摘要，产生 `DEG_FRAME_SUMMARY_DROPPED` 并把数量写入 bypass |
| 等级不可映射 | `E_LEVEL_UNMAPPED`，该文件不进入目标产物 |

降级项绝不静默丢弃。`output/bypass_active.json` 保留所有旁路字段。
