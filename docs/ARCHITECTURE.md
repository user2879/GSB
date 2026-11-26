# 定点整数多模态条件渲染管线 — 技术说明

## 1. 目标与约束

当文本、图像、运动三类条件在同一帧争夺有限去噪步数预算时，系统必须给出
**可解释、可复核、可重复** 的结果。本实现满足：

- 全程定点整数运算，渲染数学不使用浮点（仅在解析人类输入的十进制字符串时
  做精确整数换算，见 `fixedpoint_render/fixedpoint.py`）。
- 不依赖任何网络、外部账号或 GPU；样本由固定整数种子确定性生成。
- 任意命令序列、任意并发度下，增量渲染与从空白全量渲染逐帧逐像素一致。
- 所有量化、预算仲裁、降级补齐、分块读写都有可定位、可复核的记录。

## 2. 目录与模块

```
fixedpoint_render/
  errors.py        E_ 业务码与可定位错误（文件 + 字段）
  fixedpoint.py    Q 定点内核：解析、舍入枚举、量化、残差、uint8 裁剪
  config.py        RenderConfig 模型与配置模式校验
  fingerprints.py  BLAKE2b 内容/参数指纹（不含时间、线程、文件顺序）
  rng.py           实例本地确定性随机源（计数器模式，无全局随机状态）
  workspace.py     目录扫描与 PNG/词表/运动表/命令流加载校验
  encoding.py      文本词袋、图像分块均值/梯度、运动矢量三类编码器
  fusion.py        权重归一 + 定点加权融合（类型名/维度序号升序）
  budget.py        逐帧去噪步预算仲裁、截断、重分配、降级标记
  latent.py        潜变量初始化、逐去噪步内核、帧间依赖、降级补齐
  decode.py        线性投影解码、分块行 PNG、E_TILE_BOUNDARY 守卫
  ledger.py        内容指纹缓存键、上游引用、复用/重算账本
  snapshot.py      阶段化快照、临时目录、原子提交、崩溃恢复
  state.py         可编辑状态与确定性命令应用
  pipeline.py      端到端编排、线程池、稳定排序键、账本驱动复用
  report.py        可复核诊断报告
  samplegen.py     确定性本地样本生成
  viewer.py        可选 Tkinter 查看器（管线本身无头安全）
  cli.py           命令行入口
scripts/
  check_env.py     运行前依赖/可写性检查
  acceptance.py    端到端验收并落盘 artifacts/ 记录
tests/             unittest 套件（35 个用例）
```

## 3. 数据布局与配置

工作区（由 `cli init` 生成）：

```
config.json
prompts/frame_XX.txt     受控词表词项，空格或逗号分隔
images/frame_XX.png      固定分辨率、固定像素模式（RGB/L）
motion/frame_XX.json     每个分块行一条 {vx, vy}，可选 condition_dim 维 vector
commands.json            可选编辑命令流
```

`config.json` 关键字段：`frac_bits ∈ {8,12,16}`、`qmode ∈ {trunc,nearest,floor}`、
`condition_dim`、`latent_dim`、`tile_rows`（必须整除 `height`）、
三类 `weights`、`guidance`、`per_frame_budget`（正整数）、
`global_step_budget`（正整数且不小于逐帧预算）、`noise_seed`、`max_workers`。

## 4. 定点整数与量化可复核性

- 内部数值统一为带符号 Python 整数的 Qm.n 定点数，`1.0 = 2^frac_bits`。
- 舍入枚举：`trunc`（向零）、`floor`（向负无穷）、`nearest`（半数偏离零），
  全部由显式整数除法实现（`q_div_round`）。
- 每次量化记录 `QuantTrace{op, in/out frac_bits, qmode, step, raw, value,
  residual}`。融合阶段每次乘加都产生一条轨迹，因此切换权重、位宽或档位会
  **真实改变中间整数与残差**，不是仅改展示。
- 像素解码 `q_to_uint8_clamp` 先裁剪到 `[0,255]*scale` 再按档位舍入为 uint8，
  同样记录残差。

## 5. 条件编码与融合

- 文本：按受控词表顺序构造词袋计数，归一化为 Q 向量（覆盖度=非零槽占比）。
- 图像：按分块行计算颜色均值与水平梯度均值，缩放到 Q 域；分块行与条件槽
  解耦（取模映射），支持任意 `tile_rows/condition_dim` 组合。
- 运动：读入分块矢量，逐槽累加；显式 `vector` 优先，否则按 `(vx,vy)` 确定性
  散布。
- 融合：条件类型名升序（`image < motion < text`），维度序号升序；权重先归一
  （末位吸收舍入，保证和恰为 `2^frac_bits`），再做定点乘加并逐级留痕。

## 6. 逐帧预算仲裁（纯定点、与调度无关）

- 每帧评分 = 条件覆盖度与运动强度的定点加权（权重 60/40，归一到百分制）。
- 原始份额 `global * score / score_sum` 先截断为整数并以逐帧需求 `b` 封顶；
- 剩余步数按“分数降序、帧号升序”的固定规则重分配，直到耗尽或全部饱和；
- 硬不变量：每帧 `allocated <= demand`，总和 `<= global_step_budget`；
- 未满足帧 `degraded=true`，`truncated=demand-allocated`，并记录取整依据。
仲裁结果只依赖输入整数，线程调度顺序无法影响。

## 7. 潜变量内核、帧间依赖与降级补齐

- 随机源为实例本地 `DeterministicRandom`（BLAKE2b 计数器模式），素材由
  融合向量指纹 + 配置指纹（不含 `max_workers`）+ 种子 + 帧号派生，禁止使用
  运行时全局随机状态。
- 帧 `f` 初始化混入帧 `f-1` 的融合向量指纹（帧 0 用固定 `GENESIS`），形成
  可被账本精确追踪的帧间依赖；编辑帧 `f` 会使帧 `f+1` 的潜变量链失效，其余
  帧不受影响。
- 每步固定运算顺序：噪声表条目 → 引导缩放条件项 → 叠加到上一步潜变量，
  前一步输出即后一步输入，串行链不可重排。
- 降级帧缺失的步数用“半引导延续”确定性补齐到可解码长度，每条补齐都写入
  `fill_basis`，与真实去噪步以 `kind=fill` 区分。

## 8. 解码、分块行与边界保护

- 确定性整数投影矩阵（由配置指纹派生的小尺度 Q 权重）+ 中灰偏置，将潜变量
  线性映射到像素，裁剪并按档位舍入为 uint8。
- 按 `tile_rows` 划分像素行区间，`TileBoundaryGuard` 校验每次像素行读取；
  越界读取立即抛 `E_TILE_BOUNDARY`，定位到帧号与分块行号。
- 每个分块行独立产出 PNG 字节（固定压缩参数，字节可复现），再按分块行号
  升序合成整帧 PNG。分块行数量真实改变划分与完成顺序。

## 9. 内容指纹账本与精确失效

缓存键（`Ledger.make_key`）由：快照版本 + 阶段名 + 帧/步/分块序号 +
上游内容指纹 + 参数指纹 + 量化档位（qmode、frac_bits）派生，并记录上游
产物引用。各阶段稳定指纹只依赖数据字段（向量、覆盖度、需求/分配等），
不含临时轨迹对象，保证序列化/反序列化后键不变。

典型失效半径：

- 改单帧运动矢量 → 仅该帧 motion 编码、该帧融合、预算、该帧与下一帧潜变量
  （帧间依赖）、对应分块行解码重算；其它帧直接复用。
- 改融合权重/量化档位 → 配置哈希变化、快照版本递增，融合及下游重算，编码
  层按上游指纹复用。
- 无变更重跑 → 全部产物命中（验收中 46/46 复用，0 重算）。

## 10. 并发确定性

- 线程池仅并行互不依赖的任务（跨帧去噪链、跨分块行解码）；帧内去噪步严格
  串行。`max_workers` 取自配置且作为硬上限。
- 每个任务带稳定排序键 `(stage, frame, index)`；结果在主线程按确定顺序归集，
  账本写入只发生在主线程。
- `max_workers` 是执行参数，被排除在配置内容指纹之外，因此串行（1）与并行
  （如 8）输出字节完全一致（见验收 `serial_parallel_pixel_identical`）。

## 11. 阶段化快照与崩溃恢复

阶段：`01_validate → 02_encode → 03_fuse → 04_budget → 05_denoise →
06_decode → 07_compose`。

- 版本目录预先建立，产物直接写入 `snapshots/v<N>/<stage>/`；提交只把清单
  经临时文件 `os.replace` 原子切换（根清单与版本清单同时更新），不逐文件
  覆盖。
- 版本身份 = 单调递增全局版本号 + 配置哈希；配置变化必升版本，相同配置沿用
  版本以稳定缓存键。
- 强制终止（`--crash-after` 模拟 SIGKILL）后，`tmp/` 半成品被丢弃，恢复只
  承认已提交清单；损坏清单抛 `E_SNAPSHOT_CORRUPT`。恢复后输出与全新干净运行
  逐像素一致。

## 12. 诊断报告

`diagnostic_report.json` 含：版本与恢复标记、配置哈希、量化档位、逐帧预算
分配依据、降级帧清单、补齐依据、按阶段复用/重算条目数、业务码计数，以及
逐帧误差归因（去噪残差绝对值之和、残差事件数、最终潜变量指纹）。

## 13. 错误码

| 代码 | 触发场景 |
| --- | --- |
| `E_CONFIG_PARSE` / `E_CONFIG_FIELD` | 配置非法 JSON 或字段缺失/类型错误 |
| `E_FILE_MISSING` / `E_FILE_IO` | 输入缺失或不可读 |
| `E_PNG_HEADER` / `E_PNG_RESOLUTION` / `E_PNG_MODE` | PNG 容器头、分辨率、像素模式 |
| `E_VOCAB_TERM` / `E_PROMPT_PARSE` | 提示词越出受控词表、空提示 |
| `E_MOTION_PARSE` / `E_MOTION_TILES` / `E_MOTION_FIELD` | 运动表结构、分块行数、字段 |
| `E_DIM_MISMATCH` | 条件向量维度/通道不一致 |
| `E_QBITS_RANGE` / `E_QMODE_ENUM` | 位宽/量化档位枚举越界 |
| `E_BUDGET_POSITIVE` / `E_WEIGHT_VALUE` | 预算非正、权重非法 |
| `E_COMMAND_PARSE` / `E_COMMAND_TARGET` | 命令流非法动词或目标越界 |
| `E_TILE_BOUNDARY` | 分块行越界读取（含帧号/分块行号） |
| `E_SNAPSHOT_VERSION` / `E_SNAPSHOT_CORRUPT` | 快照版本/清单损坏 |

任一校验错误都会聚合为 `ErrorBundle` 抛出，渲染阶段不会启动。
