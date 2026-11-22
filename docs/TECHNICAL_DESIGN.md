# 多模态证据协同推理管线 — 技术说明

面向本地确定性样本的图文证据联合判定管线。全程不访问网络、不依赖真实模型权重；
所有结论在**定点整数域**计算，且由「输入内容 + 参数指纹 + 量化档位」唯一决定，
因此任意中断重启或任意编辑序列后的增量运行，都与从空白全量重建逐字段一致。

## 1. 目录结构

```
mm_pipeline/
  config.py       流转配置解析与严格校验（枚举/阈值/温度/位宽）
  errors.py       业务码与分类失败记录
  fixedpoint.py   定点整数运算（exp/softmax/除法/舍入，无浮点不确定性）
  hashing.py      规范化 JSON + SHA-256 内容指纹
  models.py       样本/断言/关系/特征/结论/ledger 数据模型与稳定排序键
  loader.py       目录扫描、PNG 头/尺寸、断言 schema、关系引用合法性校验
  features.py     视觉分块统计 + 配置指纹派生投影矩阵；文本断言整数编码与补齐
  fusion.py       关系语义定点融合、稳定 softmax、量化后边界决断
  snapshot.py     临时目录写入 + os.replace 原子整体提交、恢复/陈旧判定
  engine.py       线程池并发、稳定键排序、缓存复用/重算审计、阶段编排
  injection.py    独立注入配置、真实文件故障与参数故障、注入编号->业务码
  report.py       协作诊断报告、跨运行逐字段比对
  pipeline.py     顶层门面与 CLI
scripts/
  generate_samples.py  固定种子样本/配置生成
  run_validation.py    端到端验收，产出诊断报告
tests/                  28 个单元/集成测试
configs/injections.yaml 异常注入配置
```

## 2. 定点整数域

- 统一定点标度 `S = 2 ** frac_bits`，`frac_bits ∈ {8,16,24}`。
- 量化档位 `quant_level ∈ {8,12,16}`，映射到位移 `shift = quant_level // 4`，
  作用于视觉分块统计以及边界判定的量化后比较。
- `exp` 用 `z = k·ln2 + r` 区间归约 + 10 阶整数 Taylor 多项式；
  两模态 softmax 先减最大值（数值稳定），权重之和严格等于 `S`，
  残余舍入单位确定性地留给视觉模态。
- 除法区分「定点/定点」(`fdiv_scaled`) 与「原始整数比->定点」(`fdiv_round`)，
  半值采用远离零的确定性舍入，避免符号歧义。
- 模态标量先按维度归一化再乘模态权重，融合置信度经单位区间裁剪，
  不会因维度累积而恒为饱和值。

## 3. 特征抽取

- 视觉：每个网格区域计算 6 个精确整数统计量（RGB 均值/最大值），
  经量化位移后，与仅由**语义参数指纹 + 维度**派生的固定整数投影矩阵相乘，
  裁剪到 `[0, S]`。
- 文本：断言向量 = 配置种子 RNG 基向量 + 断言内容哈希 RNG 调制 + 数值属性项；
  关系权重在定点域相乘。无任何关系的区域用配置指纹派生的确定性默认向量补齐，
  并在 `backfilled_assertions` 标注补齐位置。
- 投影矩阵与默认向量的 RNG 种子只依赖语义参数，**不含** `max_workers`、
  `snapshot_interval` 等调度字段，也不含阈值/折减等决断字段。

## 4. 关系语义融合与边界

- 冲突关系：文本贡献乘 `conflict_discount` 固定折减。
- 补充关系：文本贡献乘 `supplement_boost` 叠加。
- 关联关系：按关系权重参与，贡献分解按固定 `relation_type` 顺序输出。
- 边界（真实依赖量化档位，非显示层）：
  - `threshold_tie`：定点置信度恰等于定点阈值 -> 固定决断为 `reject`。
  - `modality_tie`：两模态加权分经 `>> shift` 量化后相等 -> 决断为 `neutral`。
  - 其余按置信度与阈值比较给出 `support/reject`。
- 融合产物额外记录 `visual_quantized/text_quantized` 供审计。

## 5. 内容指纹与精确失效

缓存键分层派生，逐级包含上游指纹：

- 特征键 `feature`：图像哈希 + 断言哈希 + 关系哈希 + 区域号 +
  语义参数指纹 + 量化档位 + 位宽。
- 融合计算键 `fusion-compute`：特征键 + 关系哈希 + 融合温度等（不含阈值）。
- 融合结果键 `fusion-result`：计算键 + 置信度阈值。
  阈值变化只失效最终判定，特征与融合计算产物继续复用。
- ledger 键：全部 ledger 条目规范化哈希 + 语义参数指纹。

替换图像只失效该样本；改断言只失效引用它的区域；改关系表条目只失效对应区域；
切量化档位/位宽改变语义指纹而失效相关下游。`audit.reused/recomputed`
以 `阶段:样本:区域:键前缀` 形式输出可审计清单，未命中缓存的产物才会重算。

## 6. 阶段快照与恢复

阶段顺序固定：`load -> feature -> fusion -> ledger`。

- 每个产物先写入同目录临时文件，再 `os.replace` 原子替换；
  整目录重建采用「新树构建 + 原子改名」，不会出现半写阶段目录。
- 清单 `manifest.json` 仅在阶段产物全部提升后推进，记录
  `format_version/global_version/config_hash/relation_hash/completed_samples/stage_versions`。
- 启动恢复：格式版本或配置哈希不一致 -> 输出 `E_SNAPSHOT_STALE` 并干净重置，
  绝不静默复用；一致则按 `global_version` 续跑。
- 强制终止（测试中以 `os._exit(137)` 模拟 SIGKILL）后，只有完整替换过的
  快照可见，恢复报告输出 `recovery_version`。

## 7. 并发确定性

- 线程池大小取 `max_workers` 且不超过任务数；每个任务携带稳定键
  `(sample_id, region, stage_order)`。
- 结果按完成态收集后，统一按稳定键排序再落盘/审计；
  因此区域顺序、贡献分解顺序、复用/重算清单与失败清单顺序，
  与单线程串行执行完全一致（见 `serial_vs_concurrent` 校验）。
- 线程内不共享可变 RNG；所有随机量均由内容/参数指纹现场派生。

## 8. 异常注入与业务码

| 注入类型 | 业务码 | 阶段 |
| --- | --- | --- |
| truncate_image | E_IMAGE_FORMAT | load |
| missing_assertion_field | E_ASSERTION_SCHEMA | load |
| dangling_relation_region | E_RELATION_REF | load |
| nonfinite_fusion_param | E_CONFIG_INVALID | load（参数校验前置） |
| threshold_out_of_range | E_CONFIG_INVALID | load（参数校验前置） |
| crash_after_stage | E_INJECTION_HIT（恢复机制验证） | fusion |

文件级注入在样本目录的**隔离副本**上施加真实物理故障；参数级注入以内存覆盖
触发配置校验。单样本失败只记录 `样本/区域/阶段/业务码` 并继续其余样本，
报告的 `injections` 段回显注入编号、命中状态与匹配的失败记录。

## 9. 运行方式

```bash
pip install -r requirements.txt
python -m scripts.generate_samples --out samples/base --count 6
python -m mm_pipeline.pipeline \
  --samples samples/base --config samples/flow_config.yaml \
  --work-dir work/run --report work/run/report.json --reset
python -m scripts.run_validation \
  --samples samples/base --config samples/flow_config.yaml \
  --workspace work/acceptance --out work/acceptance_report.json
python -m unittest discover -s tests -t .
```

## 10. 验收结论

`scripts/run_validation.py` 覆盖 10 项逐字段一致性检查（基线、样本新增、
图像替换、断言/关系增删改、阈值/位宽/档位切换、串并行、崩溃恢复），
并输出跨档位置信度指纹、边界演示与注入命中映射；最新一次运行全部通过，
详见 `work/acceptance5_report.json`（随运行重新生成于 `work/`）。
