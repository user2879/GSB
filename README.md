# 确定性多模态定点融合组件

本工作区实现一个可集成的图文样本联合特征、定点融合打分、误差传播诊断、预算调度和原子快照组件。运行过程不依赖网络、容器、真实账号或外部服务；PNG 由固定种子脚本生成，文本和配置均为本地文件。

## 快速运行

环境要求：Python 3.11+、NumPy、pytest、PyYAML。示例只使用这些本地依赖。

Windows PowerShell：

```powershell
python -m pip install numpy pytest PyYAML
python scripts/generate_demo_data.py
pytest -q
python run_example.py
```

Linux/macOS：

```bash
python -m pip install numpy pytest PyYAML
python scripts/generate_demo_data.py
pytest -q
python run_example.py
```

统一 CLI：

```bash
python -m multimodal_fusion run \
  --config examples/config.yaml \
  --input examples/input \
  --state .fusion_state \
  --report reports/latest.json
```

示例会生成或复用固定样本，执行全量/增量阶段读取，写出 `reports/latest.json`，并以原子目录替换方式提交 `.fusion_state/`。连续运行第二次时 `restore.status` 为 `resumed_same_version`，并按阶段内容键复用产物。

## 目录

```text
multimodal_fusion/
  config.py       配置解析、维度/位宽/档位校验和配置指纹
  png.py          容器签名、CRC、IHDR 与扫描线解码校验
  inputs.py       样本发现、E_SAMPLE_INVALID 聚合、失败清单排序
  cache.py        有界 LRU 块缓存和淘汰后重算统计
  features.py     分块颜色/梯度/边缘特征、固定投影与量化残差
  text.py         断言确定性稀疏编码、固定合并与默认补全
  fusion.py       稳定定点加权 softmax、得分、贡献分解和阈值判定
  diagnostics.py  配置精度与高精度档的逐阶段逐项差值
  budget.py       全局 MAC、单批点数的稳定顺序准入
  snapshot.py     临时目录、整体替换、内容寻址阶段缓存和清单
  pipeline.py     线程池调度、报告与全流程编排
scripts/generate_demo_data.py  固定种子 PNG/JSONL/区域真值生成器
examples/config.yaml           完整融合配置
examples/input/                6 个本地图文样本
tests/test_pipeline.py         13 个回归测试
reports/latest.json            最近一次实际运行报告
```

## 输入协议

`texts.jsonl` 每行一个对象：

```json
{"sample_id":"sample-001","assertions":[{"entity":"shape","attribute":"color","value":"red"}]}
```

图像目录使用 `<sample_id>.png`，区域真值目录使用 `<sample_id>.truth.json`：

```json
{"sample_id":"sample-001","width":64,"height":64,"regions":[{"label":"rectangle","bbox":[8,10,20,16]}]}
```

加载层检查 PNG 签名/块长度/CRC/IHDR、解码尺寸一致性、样本 ID 唯一性、断言字段、真值尺寸/区域字段、分块整除关系和文本维度。失败原因按样本 ID 聚合成一个 `E_SAMPLE_INVALID` 条目，最终失败清单按 `(sample_id, code)` 排序。`input.strict_assertions: false` 时，断言空字段由文本编码器确定性补全并记录 `filled_fields`；缺失或空断言列表始终由配置指纹默认向量补全并记录 `fill_positions`。

## 定点与量化语义

- 所有特征和中间量使用整数。设 `Q = 2^fractional_bits`，定点值为有理数乘 `Q` 后的整数。
- 舍入统一为“绝对值四舍五入、0.5 远离零”，实现见 `round_div`，不依赖浮点。
- 量化步长是 Q 单位整数：`max(1, base_step >> tier)`。档位越高，物理量化步长越小。
- 配置档位为 `(tier, fractional_bits, word_bits)`；诊断档自动变为 `(tier+1, fractional_bits+4, word_bits+8)`，并在诊断报告中完整记录。
- 每个视觉块记录 `quant_step`、`raw`、`values` 和 `residuals = raw - values`。
- 文本同样记录量化前后向量和逐维残差。
- 配置指纹通过规范 JSON + SHA-256 派生投影矩阵、哈希向量、默认向量和阶段缓存键，不使用 Python 运行时随机数或 `hash()`。

视觉块先计算 RGB 均值、梯度强度、边缘密度（Q8），再由固定投影矩阵映射到 `visual_dim`。块缓存键包含图像哈希、投影配置、量化档与块序号。

文本断言按 `(entity, attribute, value)` 固定排序。每个实体记录合并来源；每个 token 稳定映射到固定维度的稀疏位置，未覆盖维度用配置指纹默认向量填充并标注。

融合对每个模态按维度升序求均值并裁剪到 `[-1, 1]`；先减去所有模态均值的最大值，再以 `(mean-max)/temperature` 做定点指数近似和权重 softmax。最终激活为 `(mean+1)/2`，各模态贡献在固定顺序下求和并裁剪到 `[0,1]`。得分等于阈值时，默认 `tie_break: positive` 判正，同时标记 `boundary_sample`。

## 误差传播追踪

每条有效样本同时使用配置精度和高精度档完整执行四个阶段：

1. `quantization`
2. `text_encoding`
3. `weighted_fusion`
4. `threshold_decision`

每阶段输出逐项 `configured_q`、`higher_original_q`、对齐后的 `higher_aligned_q`、`delta_q`、绝对差和首个越差项。高精度值按小数位差确定性地对齐到配置精度再比较。

- `first_exceeded_stage` 标出首个超过配置容差的阶段。
- 两档结论不一致时，若量化或文本阶段已超差，归因为 `quantization_truncation`；否则归因为 `fusion_rounding_accumulation`。
- 示例 `sample-001` 的低精度得分恰好等于阈值，是高精度下仅因融合末位舍入累积发生翻转的边界样本。

## 预算、并发与缓存

预算准入按样本 ID 稳定升序执行，不依赖线程完成顺序：

- 全局预算：视觉投影、文本编码、融合估算乘加数，主运行和诊断运行都计费。
- 单批点数：按图像块、视觉维度、文本维度和断言数估算。
- 不满足预算的样本不计算、不改变已准入样本结果，保留为 `E_BUDGET_PENDING`。
- 预算计费与缓存命中无关，因此缓存状态不会使增量运行比全量重建多处理或少处理样本。
- 阶段内容键相同的产物从旧快照复用；块 LRU 容量由 `cache_blocks` 控制，被淘汰块再次引用会重算，结果与首次执行一致。
- `ThreadPoolExecutor(max_workers=threads)` 执行分析与融合；最终报告按样本 ID 排序，不依赖完成顺序。

## 快照与增量一致性

每次运行先在状态目录同级的临时目录写入视觉、文本、融合阶段产物和清单，全部完成后把旧目录改名备份、把临时目录整体替换为新状态目录。清单包含：

- 全局内容版本 `global_version`
- `config_hash`、量化档位、小数位和字宽
- 四阶段产物版本
- 产物数量与快照间隔

全局版本由最终配置和输入内容决定，不使用时间戳或编辑顺序。样本替换、断言增删、权重/温度调整和档位切换后，阶段缓存只复用内容键仍匹配的产物；最终逐样本得分、贡献顺序、边界清单、缓存键和失败清单与同最终输入的全量空白重建一致。`snapshot_interval` 控制处理期间的 fsync 检查点频率，最终可见提交始终是全量原子替换。

## 实际验证结果

命令 `pytest -q`：`13 passed`。

`python run_example.py` 的实际定点结果：

| 样本 | 得分（Q16 整数） | 得分 | 判定 | 边界 | 精度翻转归因 |
|---|---:|---:|---|---|---|
| sample-001 | 28998 | 0.442474365234375 | positive | 是 | fusion_rounding_accumulation |
| sample-002 | 26937 | 0.4110260009765625 | negative | 否 | 无翻转 |
| sample-003 | 29032 | 0.4429931640625 | positive | 否 | 无翻转 |
| sample-004 | 27710 | 0.422821044921875 | negative | 否 | 无翻转 |
| sample-005 | 29733 | 0.4536895751953125 | positive | 否 | 无翻转 |
| sample-006 | 27116 | 0.41375732421875 | negative | 否 | 无翻转 |

从空白状态运行时：

- 6 个样本有效并完成打分，失败清单为空。
- 全局 MAC 消耗 51,864；单批点数消耗 10,752。
- 配置为 12 块缓存，实际计算 768 个块（6 样本 × 64 块 × 配置/诊断两档），产生 756 次容量淘汰。
- 配置量化档 `tier=1`、Q16、32 位；诊断档 `tier=2`、Q20、40 位。
- `sample-001` 同时是边界样本和高低精度判定翻转样本，翻转被正确归因为融合舍入累积而非量化截断。
- 测试另用低档位和零容差样本验证了 `quantization_truncation` 分支。
- 测试比较了空白全量、同版本续跑、样本替换、断言增删、权重/温度/档位切换后的增量运行与最终全量运行，核心字段完全一致。

完整逐项数据、各阶段版本、预算和缓存统计在 `reports/latest.json` 及 `.fusion_state/manifest.json` 中。

## 约束

- PNG 解码支持非交错、8 位 RGB/RGBA；其他格式进入 `E_SAMPLE_INVALID`。
- 所有样本必须使用相同尺寸，且边长能被 `features.block_size` 整除。
- 阶段缓存使用规范 JSON 内容纠址；报告为了保留逐项误差，体积可能较大。
- 示例和测试只访问工作区内的本地文件。
