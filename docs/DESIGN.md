# 设计与确定性语义

## 1. 定点数值域

配置的 `frac_bits` 定义 `Q = 1 << frac_bits`。所有条件向量、投影系数、潜变量、噪声项和去噪中间值均为 Python/NumPy `int64` 整数。十进制配置值通过 `Decimal` 转成定点整数，舍入统一为 half-away-from-zero；像素只在最终解码时裁剪到 `0..255` 并转成 `uint8`。

文本词袋按配置词表顺序遍历；每个词项的嵌入系数由 SHA-256 域名分隔派生。图像条件按图像块统计 RGB 均值与梯度。运动条件按帧、块行列和二维矢量投影到同一 `condition_dim`。融合按 `(condition_type, dimension)` 升序累加，每一项记录权重定点值、量化步长、量化后整数、残差、舍入方向和累加结果。

## 2. 阶段图

```text
text encode ─┐
image encode ├─> fusion/frame ─> latent init/frame ─> denoise step/frame
motion encode┘                                               │
                                                             v
                                        decode row/frame ─> decode frame
```

- 初始化随机源只由融合指纹、运行种子、帧号、潜变量形状和初始化配置指纹派生，不读取全局随机状态。
- 每步噪声表由种子、噪声种子、帧号、步号和总步数派生。
- 去噪步按固定顺序执行：`previous + noise + quantize(condition * guidance)`，再对结果量化。
- 每个分块行只读取由该行划分出的独立潜变量行带；空行带使用中性值，不借用相邻块像素。
- 帧内容指纹覆盖最终潜变量、投影参数、量化档位、全部去噪步指纹、全部行指纹和像素 SHA-256。

## 3. 缓存键与精确失效

统一缓存键：

```text
SHA256(domain, stage, upstream_fingerprint, parameter_fingerprint,
       quantization_level, snapshot_version)
```

原始文本、图像和运动编码发生在量化之前，其缓存键使用 `quantization_level = 0` 作为“未适用量化”的显式哨兵；融合及其下游使用真实档位。这样切换量化档位不会错误丢弃仍逐字段不变的原始编码，同时融合/初始化/去噪/解码会真实失效。

失效边界：

| 编辑 | 失效起点 | 保持复用 |
|---|---|---|
| 替换提示词 | 文本编码 | 图像、运动编码 |
| 替换/删除图像 | 图像编码或融合 | 文本、运动编码 |
| 替换运动表 | 运动编码 | 文本、图像编码 |
| 融合权重 | 融合 | 三个编码阶段 |
| 引导强度 | 去噪步 | 编码、融合、初始化 |
| 种子 | 初始化、去噪、投影/解码 | 编码、融合 |
| 去噪步数 | 去噪及下游 | 编码、融合、初始化 |
| 分块行数 | 行解码、帧组合 | 编码、融合、初始化、去噪 |
| 量化档位 | 融合及下游 | 原始编码 |

账本逐条记录 `stage/frame/step/row`、`reused|recomputed`、缓存键、上游指纹、参数指纹、量化档位、快照版本和失效字段。帧记录引用全部去噪步指纹；行记录引用潜变量行带、列带和通道带。

## 4. 调度与交互摘要

- 去噪任务按稳定键 `(stage_order, frame, step)` 排序；行解码按 `(stage_order, frame, row)` 排序。
- 线程池最大并发数严格取自配置。并发结果先按提交顺序收集，再由主线程按稳定顺序提交状态与缓存，因此完成顺序不影响输出。
- 每次 `advance()` 最多消耗 `batch_point_budget` 个去噪点。未处理步进入持久化待办，标记 `E_BUDGET_PENDING`；下一批优先处理上一批待办。
- 陈旧判定只由“全部帧完成且无故障”这一真实完成条件决定，不以等待时间推断完成。
- 请求未完成时，顶层响应为 `E_STALE_SUMMARY`，其 `summary` 来自最近一次完整快照，并带 `based_on_cursor`、`based_on_config_fp`；当前请求游标、当前配置指纹和当前进度放在 `requested_cursor`、`requested_config_fp`、`current_state`，不会冒充最新结论。
- 完整完成后顶层为 `OK`，`based_on_*` 与 `requested_*` 一致。

## 5. 快照与恢复

阶段产物发布到内容寻址目录：先写同父目录临时文件/临时目录，再原子重命名。硬件落盘 `fsync` 不是完成判据，状态完整性由内容指纹和快照摘要校验。

快照提交前计算整个会话状态的 `state_digest`；若与 `CURRENT` 相同则不改版本，否则写入临时快照目录并整体重命名为 `vNNNNNN`，最后原子替换 `CURRENT`。快照保存运行配置、命令游标、待办、已完成帧/行、已消耗预算、复用账本、失败清单、注入命中和最近完整摘要。重启时校验快照格式版本、状态摘要、运行参数、资源文件和枚举，再生成为内容指纹相同的任务键。

## 6. 异常隔离与注入

独立注入清单支持：

| 编号示例 | 阶段 | 真实故障 |
|---|---|---|
| `INJ-DIM` | fusion | 条件向量追加一个维度后走维度校验 |
| `INJ-NONFINITE` | denoise | 目标帧/步潜变量置入 `NaN` 后走有限性校验 |
| `INJ-ROW-OOB` | decode_row | 行号越界后走行边界校验 |
| `INJ-QUANT-OOB` | quantization | 档位 `9` 走允许枚举校验 |
| `INJ-STALE-SNAP` | snapshot | 恢复期望版本与快照版本不一致 |

单个帧、步或行失败只阻断其下游；其他帧、已完成的去噪步和分块行保留。报告按稳定顺序输出失败、账本、注入编号和命中任务。

## 7. 错误码

- `E_CONFIG_INVALID`：未知词项、维度不符、PNG/运动表不合约、枚举越界、非法命令等，响应包含具体 `entry`。
- `E_STALE_SUMMARY`：最新请求尚未完整完成，返回最近完整摘要。
- `E_BUDGET_PENDING`：去噪点预算耗尽后的待办条目。
- `E_PARTIAL_FAILURE`：存在真实阶段故障，但可处理部分已继续。
- `E_NONFINITE_LATENT`、`E_BLOCK_ROW_OUT_OF_RANGE`、`E_STALE_SNAPSHOT_VERSION`：注入或恢复故障。
- `E_SNAPSHOT_CORRUPT`、`E_CACHE_CORRUPT`：持久化产物完整性校验失败。
