# E_xxx 业务码

| 业务码 | 触发条件 | 主要产生位置 |
|---|---|---|
| `E_VOCAB_UNKNOWN_TERM` | 提示词含受控词表和停用词之外的词项 | 加载校验、文本编码、命令替换 |
| `E_VOCAB_EMPTY_PROMPT` | 提示词为空或没有受控词项 | 校验层/文本编码 |
| `E_COND_DIM_MISMATCH` | 文本、图像、运动、噪声或融合维度与配置不一致 | `validation.py`、条件编码器 |
| `E_PNG_SIGNATURE` | PNG 8 字节容器签名错误 | `pngcodec.py` |
| `E_PNG_HEADER` | IHDR、位深、颜色类型、过滤模式等容器头错误 | `pngcodec.py` |
| `E_PNG_RESOLUTION` | PNG 不是配置声明的固定分辨率 | `pngcodec.py` |
| `E_PNG_TRUNCATED` | chunk 长度/CRC/IDAT/IEND 或扫描线被截断 | `pngcodec.py`、PNG 注入 |
| `E_PNG_CRC` | PNG chunk CRC 不匹配 | `pngcodec.py` |
| `E_MOTION_GRID` | 运动网格与帧宽高/块尺寸不相容 | `validation.py` |
| `E_MOTION_BLOCK_MISMATCH` | 运动表实际分块数/形状与配置不符 | 校验层、运动注入 |
| `E_SEED_RANGE` | 种子不是声明范围内整数 | 配置校验 |
| `E_POSITIVE_INT` | 去噪步数、分块行数等要求正整数的字段非法 | 配置/命令校验 |
| `E_TILE_ROW_RANGE` | 分块行数超过帧高，无法形成非空像素带 | 配置/命令校验 |
| `E_QUANT_ENUM` | 量化档位不在 `Q8_FAST/Q12_BALANCED/Q16_DIAG` | 配置/命令校验 |
| `E_FRAC_BITS_ENUM` | 定点小数位宽不在 `8/12/16` | 配置校验 |
| `E_ROUNDING_ENUM` | 舍入模式不在允许枚举 | 配置校验 |
| `E_NOISE_VERSION` | 噪声表版本不在 `allowed_versions` 集合 | 噪声校验 |
| `E_NOISE_VERSION_MISSING` | 噪声表缺少 `version` 或版本为空 | 噪声校验/注入 |
| `E_NONFINITE_CONDITION` | 条件向量/统计量含 NaN 或 Inf | 图像编码注入 |
| `E_STALE_PREVIEW` | 交互预算内未得到新摘要，返回上一已提交摘要 | `RenderEngine.render_all` / `InteractiveSession` |
| `E_SNAPSHOT_SCHEMA` | 快照结构或整数版本不兼容，需要全量重建 | `snapshot.py` |
| `E_SNAPSHOT_CORRUPT` | 已提交快照 JSON/清单损坏 | 快照恢复 |
| `E_BUDGET_EXCEEDED` | 重算点数超过配置预算 | 预留账本控制 |
| `E_COMMAND_INVALID` | 命令结构、操作名或参数字段非法 | 命令应用层 |
| `E_FILE_NOT_FOUND` | 必需样本/资产缺失 | 加载层 |
| `E_CONFIG_INVALID` | YAML/JSON 解析或配置结构错误 | 加载层 |

失败清单由 `condrender.validation.write_failure_list` 输出为 JSON，字段顺序保持收集顺序；帧处理按帧号升序，因此组合注入的失败顺序稳定。
