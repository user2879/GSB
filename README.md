# 中文知识图谱三元组抽取与置信度融合诊断工作台

这是一个完全本地、确定性、可恢复的中文三元组抽取工作台。它从本地合成句子、模型参数包、抽取配置、证据关系表和异常注入表出发，完成：

- 字级 BIESO 实体片段识别与实体类型分类；
- 证据约束的关系候选生成、触发词定位和关系类型分类；
- 定点整数温度融合、置信度千分比与阈值判定；
- 配置精度与高一档参考精度的逐级误差传播诊断；
- 固定并发度/恢复点下逐字段一致的报告；
- 四阶段原子快照、优雅中断和强制终止后的续跑；
- 单句异常隔离与可证明命中的异常注入。

运行时只使用 Python 标准库。Python 3.10+ 即可运行；Windows 上已安装的 NumPy 不影响运行。数值核心刻意使用 `Decimal`、`int` 和显式舍入规则，避免平台浮点差异，因此不需要联网安装任何依赖。

## 快速开始

Windows PowerShell：

```powershell
python --version                       # 应为 3.10+
python -m unittest discover -s tests -v
python -m kg_workbench --config config.json --fresh
```

Linux / macOS：

```bash
python3 -m unittest discover -s tests -v
python3 -m kg_workbench --config config.json --fresh
```

也可以使用入口脚本：

```powershell
python run_workbench.py --config config.json --fresh
```

默认结果：

- 完整报告：`artifacts/report.json`
- 当前快照指针：`artifacts/snapshot/CURRENT`
- 快照版本目录：`artifacts/snapshot/versions/v000001` 等

常用运行选项：

```text
--workers N              覆盖 runtime.workers
--snapshot-dir PATH      覆盖 paths.snapshot_dir
--report PATH            覆盖 paths.report
--fresh                  丢弃当前快照并从 load 阶段重新开始
--stop-after-stage NAME  测试钩子；在 load/entity/relation/fusion/diagnosis 后退出 0
```

不传 `--fresh` 时，若 `CURRENT` 指向完整快照，程序自动校验配置哈希、模型哈希和输入哈希并续跑。

## 项目结构

```text
kg_workbench/
  loader.py       配置、句子、模型、证据表、注入表的解析与校验
  entity.py       字级特征、固定哈希、量化发射分数、整数 BIESO 维特比
  relation.py     证据候选、触发词、关系特征、关系分类与排斥抑制
  fusion.py       定点温度融合、阈值判定、稳定三元组键
  diagnostics.py  两档精度逐级差值与边界敏感三元组
  snapshot.py     不可变版本目录、哈希校验、原子 CURRENT 提交
  pipeline.py     阶段并发、检查点、恢复、报告汇总
  report.py       最终 JSON 报告的稳定组装
  cli.py          命令行、信号处理和退出码
config.json       默认抽取配置
samples/          句子、模型、证据、异常注入样本
tests/            标准库 unittest 自动化验收
tools/            确定性样本/模型生成器
artifacts/        报告和运行快照（快照目录被 gitignore）
```

样本模型由 `python tools/generate_samples.py` 离线重建。生成器没有随机数；运行时的抽取代码不会把 `entities` 或 `relations` 真值当作输入特征。

## 输入契约

### 句子 JSONL

每一行一个对象。`words`、`pos` 既可以是空格分隔字符串，也可以是字符串数组。实体/关系真值只用于样本构造与校验。

```json
{"id":"S001","words":"张三 在 北京 大学 工作","pos":"NR P NS NN VV","entities":[{"start":0,"end":0,"type":"PER"},{"start":2,"end":3,"type":"ORG"}],"relations":[{"subject_entity":0,"object_entity":1,"type":"任职于"}]}
```

校验规则：

- `id` 必填且全局唯一；
- 词数与词性数一致且非空；
- 实体 `start/end` 是闭区间词位置，必须非负、`start <= end`、不越界、互不重叠；
- 关系真值引用已有实体下标，关系类型必须在配置集合内；
- 单行错误进入失败清单，其他行继续处理。

### 模型参数包

`samples/model.json` 包含：

- `feature_dim` 与 `hash_buckets`，两者必须相等；
- 五标签 `B/I/E/S/O` 的偏置、转移和长度等于桶数的权重数组；
- `entity_types` 及其实体类型偏置/权重；
- `relation_types` 及其关系偏置/权重，集合必须与配置一致；
- `feature_hashes`：特征键到桶号的显式固定映射；
- `lexicon`：用于“词表命中长度”特征。

未出现在 `feature_hashes` 的特征键使用 SHA-256 前 8 字节取模，绝不使用 Python 进程随机化的 `hash()`。所有权重先按当前定点档位量化，再参与整数维特比或关系打分。

### 抽取配置

路径相对于配置文件所在目录解析：

```json
{
  "paths": {
    "sentences": "samples/sentences.jsonl",
    "model": "samples/model.json",
    "evidence": "samples/evidence.json",
    "injections": "samples/injections.json",
    "snapshot_dir": "artifacts/snapshot",
    "report": "artifacts/report.json"
  },
  "entity": {"threshold": 0.5},
  "relation": {
    "relation_types": ["任职于", "位于", "创办"],
    "trigger_words": {
      "任职于": ["工作", "加入", "任教"],
      "位于": ["位于", "在"],
      "创办": ["创办"]
    }
  },
  "fixed": {
    "decimals": 4,
    "quantization_decimals": 3,
    "temperature": 0.75,
    "confidence_threshold": 0.62,
    "entity_weight": 0.45,
    "relation_weight": 0.55,
    "bias": -0.2,
    "threshold_equal_policy": "reject"
  },
  "diagnostic": {"tolerance_units": 200},
  "runtime": {"workers": 4, "snapshot_commit_interval": 2}
}
```

定点枚举：

- `fixed.decimals ∈ {2,3,4,5,6,8}`；
- `fixed.quantization_decimals ∈ {0,1,2,3,4,5}` 且不得大于 `decimals`；
- 当 `decimals=4, quantization_decimals=3` 时，分值网格步长为 `0.001`；
- 高一档参考精度为 `decimals+1`，量化档位为 `min(quantization_decimals+1, decimals+1)`；
- `diagnostic.tolerance_units` 以参考档整数单位计。

`threshold_equal_policy` 为 `accept` 或 `reject`。样本使用 `reject`，因此注入得到的“恰等于阈值”分数会被标记为边界样本且不输出，而高精度参考会输出同一条三元组。

### 证据关系表与异常注入

证据候选对按句子 ID 和两个实体词跨度精确声明，并限定允许关系类型；排斥关系按对称闭包处理：

```json
{
  "candidate_pairs": [
    {"sentence_id":"S001","subject_span":[0,0],"object_span":[2,3],"relations":["任职于","位于"]}
  ],
  "exclusive_relations": [{"relation":"任职于","excludes":["位于"]}]
}
```

注入表支持四类真实故障，并严格绑定阶段：

| 注入类型 | 阶段 | 行为 |
|---|---|---|
| `annotation_out_of_range` | `load` | 修改真实实体区间使其越界，产生 `E_ANNOTATION_RANGE` |
| `missing_pos` | `entity` | 删除一个词性标记，使实体阶段词/词性数量不符，产生 `E_CONFIG_INVALID` |
| `feature_hash_bucket_mismatch` | `relation` | 在关系特征路径注入越界桶号，产生 `E_MODEL_SHAPE_MISMATCH` |
| `threshold_equal` | `fusion` | 将该句融合分值置为配置阈值，触发边界规则并记录命中 |

注入编号、是否命中、所属阶段与业务码写入最终 `injection_hits`。

## 计算规范

### 实体识别

1. 每个词展开为字符槽，携带所属词位置、词内偏移、词性；
2. 位置特征为当前字、前字、后字、当前词性、前/后词性、词表命中长度；
3. 每个特征键先查 `feature_hashes`，否则用固定 SHA-256 映射；
4. 发射分数是模型偏置与特征权重之和，立即量化为整数单位；
5. 在整数对数域做 BIESO 维特比，结构上只允许 `B→I/E`、`I→I/E`、`E/S/O→B/S/O`，并以 `E/S/O` 结束；
6. 同一状态分数相同时按标签优先级 `B > I > E > S > O`；最终路径同分也按该顺序；
7. 字符边界映射回词位置整数闭区间；
8. 实体类型按字典序执行类型分类，同分取类型名字典序最小者；
9. 实体置信度由路径内选中标签相对每位置最强其他标签的平均间隔经 sigmoid 得到，再按定点网格量化；低于 `entity.threshold` 的片段不参与关系候选。

实体阶段产物包含标签序列、路径总分、每位置五标签量化得分、全部候选片段及类型分数。

### 关系抽取

1. 只枚举证据表中声明且两个实体均被识别/过阈的词跨度对；
2. 对每个候选关系类型，在实体跨度之外按词位置从左到右查找触发词；
3. 特征包括候选关系、实体类型、实体的词、触发词、实体间词序列和实体间逐词特征；
4. 特征键排序后按固定顺序求整数分；
5. 同分时关系类型名字典序最小者胜出；
6. 若证据表声明胜出关系与另一候选关系互斥，后者进入 `suppressed`；
7. 实体得分取主语/宾语实体置信度的较小值，关系得分取胜出类型的 sigmoid 定点置信度。

### 定点融合

令 `S=10^decimals`，实体、关系、权重、温度、偏置和阈值都转为整数单位。对每个候选：

```text
weighted_margin = round((w_entity * entity + w_relation * relation + bias * S) / S)
scaled_margin   = round(weighted_margin * S / temperature_units)
fusion          = round(sigmoid(scaled_margin / S) 到量化网格)
permille        = clamp(round(fusion * 1000 / S), 0, 1000)
```

阈值比较使用定点整数：

- 严格大于阈值：输出；
- 严格小于阈值：拒绝；
- 恰等于阈值：按 `threshold_equal_policy`，并标记 `boundary_sample=true`。

三元组按 `主语文本、谓语、宾语文本、句子 ID、主语跨度、宾语跨度` 的稳定键排序。

### 逐级误差诊断

每条完整成功句子同时执行配置档和高一档：

1. 实体量化；
2. 关系量化；
3. 融合加权；
4. 阈值判定。

低档分数单位乘以 `10^(high_decimals-low_decimals)` 对齐到参考单位，逐项输出配置值、参考值、差值、缺失项和是否超差。诊断会给出：

- `first_stage_over_tolerance`；
- `entity_boundary_shift`、`entity_type_shift`；
- `relation_classification_shift`；
- `primary_source`；
- 两档输出结论不一致的 `boundary_sensitive_triples`；
- 每个边界敏感三元组的融合主贡献来源和诊断主来源。

`primary_source` 可能为 `entity_boundary_shift`、`relation_classification`、`entity_quantization`、`relation_quantization`、`fusion_quantization`、`threshold_decision` 或 `none`。

## 并发、快照与恢复

### 并发一致性

- 句子按 ID 升序切分检查点批次；
- 每批用 `ThreadPoolExecutor` 并发执行，最大线程数来自 `runtime.workers` 或 `--workers`；
- 每个任务携带 `stable_task_key = stage + SHA-256(stage NUL sentence_id) 前 16 位`；
- 收集结果时按提交顺序回填，最终阶段产物按句子 ID 排序；
- 三元组、诊断、失败、注入和统计均从排序后的阶段产物构建。

因此线程完成顺序、并发度或恢复点不会改变核心报告字段。

### 原子快照

阶段为 `load → entity → relation → fusion → diagnosis`。每 `snapshot_commit_interval` 个句子写一个检查点，阶段完成时再提交完成状态。

提交过程：

1. 新建临时版本目录；
2. 复制上一版本完整产物；
3. 写入本阶段产物和新 `manifest.json`；
4. 对所有产物计算 SHA-256；
5. 将完整临时目录重命名为不可变版本目录；
6. 用临时文件 + `os.replace` 原子替换 `CURRENT` 指针。

强杀发生在步骤 1–5 时，`CURRENT` 仍指向旧完整版本；发生在步骤 6 前，新版本即使完整也不可见。恢复时只读取 `CURRENT` 指向的完整版本，并校验每个产物 SHA-256。

清单包含：

```json
{
  "global_version": 10,
  "config_hash": "...",
  "model_hash": "...",
  "input_hash": "...",
  "stages_done": ["load", "entity", "relation", "fusion", "diagnosis"],
  "completed_sentences": ["S001", "..."],
  "stage_completed_sentences": {"entity": ["S001", "..."]},
  "artifacts": {
    "load": {"version": 1, "sha256": "...", "file": "load.json"},
    "entity": {"version": 4, "sha256": "...", "file": "entity.json"}
  }
}
```

实际清单中的 `artifacts` 包含所有已完成或部分完成的阶段产物。

启动续跑时若配置、模型或输入哈希不一致，输出 `E_CONFIG_MISMATCH` 并退出码 6。

### 退出信号

安装 `SIGINT`/`SIGTERM` 处理器后，信号只设置停止事件；当前阶段会继续执行并完成其快照，然后以整数退出码 0 返回 `status=interrupted` 和当前快照摘要。再次运行同一命令即可恢复。

## 错误码与隔离

| 业务码 | 含义 |
|---|---|
| `E_CONFIG_INVALID` | 配置枚举/结构、句子 ID、词/词性数量等非法 |
| `E_MODEL_SHAPE_MISMATCH` | 维度/桶数/权重长度/特征桶/关系集合不匹配 |
| `E_ANNOTATION_RANGE` | 实体或关系真值的跨度非法 |
| `E_CONFIG_MISMATCH` | 快照与当前配置、模型或输入不一致 |
| `E_SNAPSHOT_INVALID` | 快照指针、清单或产物哈希损坏 |
| `E_RUNTIME` | 未归类的单句异常；仍进入失败清单 |

单句在任一阶段抛异常时，会记录 `sentence_id、stage、code、exception_type、message、injection_ids`，该句不再进入下游阶段，其余句子继续。

## 报告

`artifacts/report.json` 的主要字段：

- `triples`：排序后的三元组与置信度千分比；
- `score_histograms.configured/reference`：两档位的全部候选与输出三元组直方图；
- `boundary_sensitive_triples`：两档位输出结论不一致的三元组；
- `injection_hits`：全部注入项及命中证据；
- `diagnostics`：逐句四环节逐项差值和首次超差环节；
- `failures`：排序后的失败清单；
- `snapshot`：版本、哈希、阶段版本和已完成句子；
- `statistics`：句子、实体、候选、输出、边界与失败统计；
- `sentences`：逐句实体、位置分数、关系候选、融合候选、诊断和高精度参考产物。

## 实际验证结果

在 Python 3.11.0rc1、Linux 环境实际执行：

```text
$ ./verify.sh
Ran 8 tests in 5.038s
OK
```

样本完整运行摘要：

```json
{"boundary_sensitive_triples": 1, "diagnostics": 3, "failures": 5, "injection_hits": 4, "snapshot_version": 10, "status": "completed", "triples": 2}
```

具体验证：

- 输出三元组：`孙八—任职于—杭州公司`、`陈七—任职于—南京大学`；
- `S001` 的 `张三—任职于—北京大学` 在注入后融合分恰为阈值 `0.6200`，按拒绝规则不输出；高精度参考分为 `0.74390` 并输出，因此列入边界敏感三元组；
- 注入全部命中：`E_CONFIG_INVALID`、`E_ANNOTATION_RANGE`、`E_MODEL_SHAPE_MISMATCH` 及成功命中的阈值边界项；
- `S007` 词性数不符、`S008` 实体越界也各自隔离，未阻断其他句子；
- `workers=1` 与 `workers=5` 的报告核心字段逐字段相等；
- 在 `load/entity/relation/fusion/diagnosis` 每个阶段后中断再续跑，核心字段均与一次完整运行相等；
- 在 `relation` 阶段首个部分检查点模拟强杀，恢复后核心字段相等；
- 变更配置后的续跑返回 `E_CONFIG_MISMATCH` 和退出码 6；
- 篡改阶段产物后读取快照返回 `E_SNAPSHOT_INVALID`；
- 真实发送 `SIGTERM`：在 0.10 秒后发送时安全停在 `load`、0.15 秒后发送时安全停在 `entity`，两次退出码均为 0，快照均可续跑。

重新运行验收：

```bash
python3 -m unittest discover -s tests -v
python3 -m kg_workbench --config config.json --fresh
```

如需重建样本和模型：

```bash
python3 tools/generate_samples.py
```
