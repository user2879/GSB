# RELX：本地确定性中文关系抽取与增量失效判定应用

面向本地确定性中文语料的句子级关系抽取交互应用。核心解决句子、实体、关系模板
与后处理规则反复编辑时的**增量更新**与**阶段级精确失效判定**。所有样本由固定
种子脚本生成，运行不访问网络或外部服务，仅依赖 Python 3.11+ 与 NumPy。

## 1. 环境准备

```bash
python --version            # 需要 3.11 或更高
pip install -r requirements.txt
```

Windows 命令行直接运行：

```bat
run_demo.bat
run_tests.bat
```

Linux / macOS：

```bash
./run_demo.sh
./run_tests.sh
# 或
python -m relx data output
python -m unittest discover -s tests -v
```

首次运行会在 `data/` 下生成四类输入，在 `output/` 下写出全部 JSON 产物。

## 2. 输入数据（全部本地、固定种子）

| 文件 | 说明 |
| --- | --- |
| `data/sentences.jsonl` | 句级标注，每行含 `sentence_id`、空格分隔 `tokens`、`entities`、`relations`，并混入 5 类异常行 |
| `data/model_package.json` | 词项哈希表 `term_hash`、实体类型词表 `entity_lexicon`、关系模板权重、`feature_dim`、`hash_buckets`、模型版本号 |
| `data/config.yaml` | 实体/关系类型集合、最长词项长度、阈值、定点位宽与量化档位、线程数、单批特征点预算、快照提交间隔 |
| `data/commands.json` | 编辑命令流：改词、增删实体、模板权重、阈值、量化档位切换 |

生成器见 `relx/sample_data.py`，种子常量 `SEED = 20261124`，无随机网络来源。

## 3. 模块结构

```
relx/
  errors.py        业务错误码（E_CONFIG_INVALID、E_ENTITY_REF_INVALID 等）与失败清单
  fingerprint.py   FNV-1a 64 位内容指纹与规范化字节编码（禁止浮点入指纹）
  fixedpoint.py    量化档位 Q2/Q4/Q8、位宽 8/16/32、有符号饱和与整数四舍五入
  mini_yaml.py     受控 YAML 子集解析器（无第三方依赖）
  config.py        配置加载与枚举/正整数校验
  model.py         模型包加载、模板与关系类型一一对应校验
  corpus.py        JSONL 逐行校验，单句失败入清单不中断其余句子
  commands.py      编辑命令流加载与校验
  state.py         运行时状态与命令原地应用
  entities.py      词典候选生成与边界重叠消解（最长跨度>起点>类型名）
  features.py      FNV 哈希桶特征向量（NumPy 确定性运算）
  relation.py      模板触发、定点打分、阈值/边界决断与落空原因码
  cache.py         四阶段内容指纹缓存与模板/量化精确失效
  scheduler.py     单批特征点预算、待办保留、超预算码与预算账本
  engine.py        增量引擎：失效判定、波次调度、线程池并发与确定性合并
  pipeline.py      全量重建、增量回放、并发对照、量化对照与落盘
  sample_data.py   固定种子样本生成器
```

## 4. 关键设计

### 4.1 实体候选与边界重叠消解
- 候选仅按 `entity_lexicon` 声明的真实类型生成，避免通用词被误判为任意实体。
- 优先级：最长跨度 → 起始位置升序 → 实体类型名升序 → 来源 → 候选 ID。
- 同跨度同类型的标注实体与词典候选合并，标注 ID 保留；被抑制候选写入
  `suppressed` 并记录 `swallowed_by`。
- 词典候选 ID 按最终顺序稳定编号 `C01、C02 ...`，重复运行逐字段一致。

### 4.2 定点关系打分
得分完全在定点整数域、按固定运算顺序合成：

```
weight_q    = quantize(template.weight, width, quant)
threshold_q = quantize(config.score_threshold, width, quant)
score       = saturate(feature_energy + weight_q*hit_count
                       + compatibility - distance_q, width)
```

- `feature_energy` 由触发词哈希桶权重与实体对特征向量做整数点积。
- 距离衰减使用带符号整数四舍五入除法。
- `score >= threshold_q` 保留；**恰等于阈值记为边界关系** `boundary=true`。
- 切换位宽/档位会真实改变内部整数：例如同一关系在 Q4/16 位得 `26333`，
  Q8/32 位得 `263333333`，Q2/8 位饱和为 `127`。

### 4.3 落空处理
原因码：`MISS_NO_ENTITY`、`MISS_NO_COMPATIBLE_TEMPLATE`、`MISS_NO_TRIGGER`、
`MISS_BELOW_THRESHOLD`。无实体句与全落空句都保留在结果中；落空条目按
句子 ID、实体对起点、模板 ID 稳定排序，并记录依赖模板集合。

### 4.4 内容指纹与精确失效
四个阶段独立缓存：`entity → trigger → score → miss`。缓存键由上游指纹、
量化后的参数整数、模板量化权重与触发词派生；模板版本号仅用于审计，不进入
最终内容键，保证“编辑后增量”与“按最终态一次性全量”键值一致。

| 编辑 | 失效范围 |
| --- | --- |
| 改词 / 增删实体 | 仅该句四阶段 |
| 模板权重 | 仅实际引用该模板的句子的 trigger/score/miss |
| 阈值 | 全部句子的 score/miss，entity 缓存复用 |
| 位宽/档位 | 全部句子四阶段（整数域整体变化） |

### 4.5 预算调度与并发一致
- 调度器按句子 ID 稳定顺序扣减特征点；预算不足的任务以
  `E_FEATURE_BUDGET_PENDING` 进入待办，下一批优先；超整批预算以
  `E_FEATURE_BUDGET_OVERSIZE` 标记并移除，防止死循环。
- 账本逐行记录批次、句、阶段、扣点、批剩余与全局累计，并断言
  `sum(ledger.points) == global_consumed`。
- 线程池并发只负责计算，任务排序、账本扣减与结果合并全部在主线程按
  句子 ID 升序完成，因此并发度变化不改变任何数值与顺序。

## 5. 输出产物（`output/`）

| 文件 | 内容 |
| --- | --- |
| `full_rebuild.json` | 从空白全量运行的逐句实体、抑制、触发、关系、边界、落空、缓存键、预算账本 |
| `incremental_baseline.json` | 命令到达前的基线结果 |
| `incremental_commands.json` | 每条命令后的失效、复用/重算与完整报告 |
| `incremental_final.json` | 命令序列执行完的增量最终结果 |
| `consistency_check.json` | 增量最终态 vs 按最终句子集合/模板参数一次性全量重建的逐字段比对 |
| `concurrency_check.json` | 单线程 vs 8 线程产物比对 |
| `quant_variants.json` | 8/Q2、16/Q4、32/Q8 三档位整数与判定对照 |
| `summary.json` | 汇总指标、装载失败清单与缓存审计行 |

## 6. 验收覆盖

测试位于 `tests/`，共 37 个用例，覆盖：

- 定点量化、饱和、半数向上与档位差异；
- 实体最长跨度消解、抑制清单审计与重复运行一致性；
- 关系触发、定点得分、边界决断、落空原因与稳定排序；
- 五类异常输入失败码且不中断其余句子；
- 预算待办、超预算、账本核对与稳定顺序；
- 四类编辑的精确失效范围；
- 增量最终态与空白全量重建逐字段一致；
- 单线程与线程池并发逐字段一致、账本扣减序列一致。

## 7. 已知边界

- YAML 解析器仅支持本项目配置所需的子集（嵌套映射、块序列、基础标量）。
- 关系抽取为句子级、有序实体对（头起点先于尾起点），未做跨句共指。
