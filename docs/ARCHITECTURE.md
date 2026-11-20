# 架构与技术说明

## 1. 设计目标

面向本地确定性样本的指令微调数据构造诊断工作台，解决增量更新时的依赖失效
判定问题。核心约束：

- 全流程无随机性，同输入多次运行产物逐字节一致；
- 分批增量运行与从空白工作区按最终输入一次全量构造**逐字段一致**；
- 未失效候选直接复用，不得重新打分；
- 阶段化快照支持整体替换提交与强制终止后的一致性恢复；
- 并发度与线程调度不影响最终顺序与内容。

## 2. 模块分层

| 模块 | 职责 |
| --- | --- |
| `sft_diag/errors.py` | 业务错误码（`E_SEED_INVALID`、`E_CONFIG_INVALID`、`E_PARAMS_INVALID`、`E_BATCH_INVALID`、`E_CONFIG_MISMATCH`、`E_SNAPSHOT_CORRUPT`）与失败清单条目 |
| `sft_diag/utils.py` | 规范化 JSON、SHA-256 指纹、哈希派生种子/缓存键、float64 舍入与有效数字格式化、稳定 ID |
| `sft_diag/schemas.py` | 基于 jsonschema Draft 2020-12 的结构校验 |
| `sft_diag/loader.py` | JSONL/JSON 逐行逐项加载与业务校验，非法条目入失败清单不中断 |
| `sft_diag/models.py` | 种子、槽位来源、构造配置、打分参数、词表与累计工作区状态 |
| `sft_diag/tokenizer.py` | NFKC 归一 + 最长前缀子词匹配 + 未登录字符单字符回退 |
| `sft_diag/generator.py` | numpy PCG64 确定性加权采样，记录采样序号、模板出现次序、模板/槽位来源 |
| `sft_diag/scorer.py` | 四维度 float64 打分，线程池并发，缓存键命中即复用 |
| `sft_diag/dedup.py` | 子词集合 Jaccard、并查集近重复分组、代表样本选择、边界标注 |
| `sft_diag/batches.py` | 分批更新计划解析与累计状态纯数据变换 |
| `sft_diag/snapshot.py` | 临时目录写入 + `os.replace` 原子替换 + 清单指纹与恢复判定 |
| `sft_diag/pipeline.py` | 采样/打分/去重三阶段编排、精确失效 diff、报告 |
| `sft_diag/runner.py` | 加载、续跑判定、逐批执行、影子全量对照与差异诊断 |
| `sft_diag/cli.py` | `run` / `full` / `verify` 命令行入口 |

## 3. 确定性保证

1. **随机源**：仅使用 `numpy.random.Generator(PCG64(seed))`。PCG64 的流在所有
   平台上由整数种子唯一确定，不依赖系统熵。
2. **种子派生**：`derive_seed(seeds_fp, slots_fp, sampling_config_fp, vocab_fp,
   task_type, purpose)`，由上游内容 SHA-256 指纹混合得到 64 位整数。
3. **采样配置指纹排除并发字段**：`max_workers`、`snapshot_interval_batches`、
   去重阈值等非采样字段不参与 RNG 种子与采样缓存键，保证并发度调整不改变
   候选序列。
4. **加权采样**：每次按当前模板/来源 ID 的字典序构造概率向量，归一化后调用
   `rng.choice(p=...)`；槽位取值用 `rng.integers` 定序号。
5. **稳定 ID**：候选 ID 形如 `classification#000003`，定宽补零使字典序与
   数值序一致，所有平局按候选 ID 字典序决断。
6. **规范化序列化**：所有产物经 `canonical_json`（键排序、无多余空白、
   `allow_nan=False`）写入，重复运行逐字节一致。
7. **float64**：打分全程 Python/numpy IEEE-754 双精度，加权求和用
   `math.fsum`；产物同时输出数值字段与 `*_text` 六位有效数字文本字段。

## 4. 三阶段缓存键与精确失效

每个阶段的缓存键均由“上游内容指纹 + 参数指纹”派生：

- 采样阶段：任务级键 = `derive_cache_key("sampling", seeds_fp, slots_fp,
  sampling_config_fp, 候选内容指纹)`；
- 打分阶段：候选级键 = `derive_cache_key("score", 候选内容指纹, params_fp,
  vocab_fp, seeds_fp)`；
- 去重阶段：任务级键 = `derive_cache_key("dedup", 候选子词集合+分数指纹,
  去重参数指纹)`。

采样产物额外保存 `reference_index`：

```
reference_index.template[template_id]      = [candidate_id, ...]
reference_index.slot_source[source_id]     = [candidate_id, ...]
```

增量批次执行时：

1. 用最终累计输入重新确定性生成完整候选序列（保证序列与全量构造同源）；
2. 与上一版采样产物按候选 ID + 候选采样缓存键 diff，分为
   `new / changed / reused / removed`，并归因到模板替换删除、槽位来源
   增删、目标数/权重变化等原因；
3. 打分阶段仅对缓存键未命中的候选调用打分器，命中候选整记录复用；
4. 去重阶段按任务级缓存键决定整任务重算或整任务复用。

仅改打分权重时，采样 diff 全部为 `reused`，仅打分与去重重算；输入完全
不变的重跑则三阶段全部复用、零重算。

## 5. 近重复与边界规则

- 相似度 = 子词集合 Jaccard（空集对空集定义为 1.0）；
- `similarity > threshold` 必然合并；`similarity == threshold` 由
  `boundary_rule` 决断：`merge` 合并、`split` 不合并，两种情况下该边都在
  产物中标注 `boundary=true`，相关候选进入 `boundary_candidate_ids` 与
  分组的 `boundary_member_ids`；
- 并查集按候选 ID 字典序合并，组内代表样本取总分最高、总分相同取 ID 最小；
- 被剔除样本记录 `representative_id` 与对代表的相似度，分组显式列出全部成员。

## 6. 快照与恢复

磁盘布局：

```
<output>/snapshot/
  manifest.json
  stages/sampling.json
  stages/scoring.json
  stages/dedup.json
  tmp.<pid>/         # 阶段写入临时目录，提交后立即删除
```

- 每阶段先写临时目录文件，再 `os.replace` 原子替换正式文件，最后原子替换
  `manifest.json`；清单含全局版本、基础输入指纹、累计指纹、配置/参数哈希、
  已完成批次号、各阶段版本与缓存键，以及清单自身指纹；
- 启动时先清理残留 `tmp.*`，再读取清单：无清单则全新运行；基础输入指纹
  不一致返回 `E_CONFIG_MISMATCH` 拒绝续跑；清单指纹被篡改返回
  `E_SNAPSHOT_CORRUPT`；
- 强制终止发生在清单替换之前时，恢复到上一完整批次；发生在阶段提交之间时，
  仅依据已完整替换的阶段恢复并自动补齐缺失阶段（恢复点批次号写入报告
  `snapshot.recovered_from_batch`）。工作台不承诺生成“退出报告”。

## 7. 并发一致性

- 线程池仅用于无共享写的纯函数任务（单候选打分、候选对相似度），
  `max_workers = min(config.max_workers, 任务数)`，不会超出配置上限；
- 每个任务的稳定排序键由候选 ID 与阶段名派生；工作线程结果按原索引回填，
  最终一律再按候选 ID 排序输出；
- 并查集合并、分组、失败清单均在主线程按确定顺序执行，调度顺序不影响结果。

## 8. 诊断报告

`report.json` 包含：业务码、已完成批次、全局版本、五类内容指纹、各任务
候选数/去重前后数量/各维度平均分（全量与保留集两套）、失效与复用候选 ID
明细及原因、快照版本与恢复点、失败清单、影子全量对照差异（空数组表示
逐字段一致）。
