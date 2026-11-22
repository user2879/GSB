# 中文依存句法分析确定性工作台

这是一个不依赖外部服务的本地工作台，用移进-归约（arc-standard）转移体系解析中文词/词性序列。它同时提供：

- 贪心与穷举两种解析模式；
- 基于句子、转移状态和模型依赖的步骤级增量缓存；
- 可原子提交、可在句子和转移步游标处续跑的版本化快照；
- 线程池并发、稳定排序、批解析点数预算；
- 可复现的四类故障注入与 JSON 诊断报告。

样例、模型和配置全部由固定种子生成，不需要下载语料。开发验证环境为 Python 3.11.0rc1、NumPy 1.26.4；运行代码只使用 Python 标准库。

## 快速开始

```bash
python -m depworkbench generate --out samples --seed 20261123 --count 10
python -m depworkbench validate --samples samples
python -m depworkbench run --samples samples \
  --snapshot-dir .depworkbench-snapshots \
  --report artifacts/report.json
```

Windows 下可将 `python` 换成 `py -3`。所有输入输出均为 UTF-8。

生成后的样本目录包括：

```text
samples/
  vocabulary.json      # 受控词表与词性集合
  sentences.jsonl      # 每行一个句子
  model.json           # 四类动作权重、特征哈希、维度与桶数
  config.json          # 模式、约束、定点量化、并发、预算、快照间隔
  injections.json      # 基线默认无注入
  injections_demo.json # 四类故障注入演示
  generation.json      # 种子和生成元数据
```

## 输入格式

句子行支持空格分隔串或数组，例如：

```json
{"sentence_id":"S001","words":"我们 喜欢 数据","pos_tags":"r v n","gold_heads":[1, -1, 1],"gold_relations":["R","ROOT","R"]}
```

模型核心字段：

```json
{
  "feature_dim": 2,
  "num_buckets": 128,
  "action_weights": {
    "SHIFT": [0.25, -0.5],
    "LEFT_ARC": [0.1, 0.2],
    "RIGHT_ARC": [-0.3, 0.4],
    "FINISH": [0.0, 0.0]
  },
  "feature_hashes": {"p:n": 12, "a:SHIFT|bias": 31},
  "vocabulary": ["数据", "我们"],
  "pos_tags": ["n", "r", "v"]
}
```

每个动作权重数组长度为 `feature_dim`。特征桶为 `bucket`，实际权重下标是 `bucket % feature_dim`。分数先按
`10^fixed_point_decimals / quantization_level` 量化到有符号整数域，再逐特征求和比较。

配置核心字段：

```json
{
  "mode": "greedy",
  "max_dependency_distance": 7,
  "max_transition_steps": 40,
  "max_threads": 4,
  "batch_parse_point_budget": 80,
  "snapshot_commit_interval_sentences": 2,
  "fixed_point_decimals": 3,
  "quantization_level": 1,
  "constraints": {
    "max_arc_distance": 6,
    "max_stack_size": 9,
    "max_dependents_per_head": 5,
    "forbid_left_dependent_pos": [],
    "forbid_right_dependent_pos": [],
    "finish_requires_single_root": true
  },
  "max_exhaustive_nodes": 250000,
  "max_pruned_records": 200
}
```

约束规则支持：

- `max_arc_distance`、`max_stack_size`、`max_dependents_per_head`；
- `forbid_left_dependent_pos`、`forbid_right_dependent_pos`；
- `finish_requires_single_root`。

## 解析与缓存语义

动作编码固定为 `SHIFT=0`、`LEFT_ARC=1`、`RIGHT_ARC=2`、`FINISH=3`。贪心每步按 `(-score, action_code)` 选择动作，因此同分时编码小者胜出。穷举在最大转移步数内搜索，以总得分最高、动作编码序列字典序最小为准，并记录支配状态、最大步数和节点上限等剪枝依据。

每个转移步骤保存：

- 完整栈/缓冲区前后快照、可行动作与分数、排除动作及原因；
- `feasible_action_fingerprint`；
- 句子可见内容、局部模型参数、量化档位和前一转移步键派生的 `cache_key`；
- 引用到的词位、特征桶、权重单元和约束指纹。

贪心增量编辑只使首个输入依赖实际变化的转移步及其后继失效。例如只改未处理词性时，前面的动作和分数可原样复用；显示快照会从当前句子刷新，但不会重做转移计算。

穷举模式的目标是全局最优，未来未选分支上的权重也可能改变更早的最优前缀。为保证“增量结果等于空白全量重建”，穷举缓存使用句子级搜索环境指纹：解析相关模型、句子或约束一旦变化，就重搜该句；环境未变时全部步骤直接复用。这个保守边界是全局最优语义的必要条件，详见 [docs/DESIGN.md](docs/DESIGN.md)。

## 交互编辑与增量运行

```bash
python -m depworkbench repl --samples samples \
  --snapshot-dir .depworkbench-snapshots \
  --report artifacts/report.json
```

REPL 示例：

```text
run greedy
replace S001 我们 检查 数据 | r v n
weight LEFT_ARC 4 0.625
constraint forbid_left_dependent_pos ["d"]
mode exhaustive
run
quit
```

程序化接口：

```python
from depworkbench import Workbench

wb = Workbench("samples", snapshot_dir=".snapshots")
first = wb.run(fresh=True)
wb.replace_sentence("S001", ["我们", "检查", "数据"], ["r", "v", "n"])
incremental = wb.run()
full = Workbench("samples", snapshot_dir=".full").run(fresh=True)
assert incremental["semantic_digest"] == full["semantic_digest"]
```

## 调度、并发与预算

- 解析点数单位是词数；句子成本为 `max(1, len(words))`。
- 每批按“上一批待办优先、同组按句 ID 升序”选择；放不下的句子写入 `E_PARSE_BUDGET_PENDING` 并留给下一批。
- 线程池大小取自 `max_threads`，上限 32。
- 所有句子和转移步骤都有稳定任务键；最终句子、步骤、弧、树、失败和预算清单均按稳定键排序。
- 并发度、预算、快照间隔不会进入解析产物键。

## 快照与恢复

快照目录结构：

```text
snapshot-root/
  CURRENT                    # 原子替换的当前版本指针
  versions/v000001/
    manifest.json            # 版本、模型/配置/语料哈希、已完成句、步骤游标、阶段版本
    load.json
    parse.json
    arcs.json
    trees.json
```

每个版本先写入临时目录，再整体替换为不可变版本目录，最后原子替换 `CURRENT`。`run --resume` 在版本、模型、配置或语料不一致时输出 `E_SNAPSHOT_STALE`、丢弃当前指针并重新初始化；普通编辑运行则把旧快照作为增量缓存基线。中断后既可从句子完成集继续，也可从保存的部分转移步骤继续。

## 故障注入

`injections_demo.json` 覆盖四种真实输入故障：

| 编号 | 故障 | 业务码 |
|---|---|---|
| `INJ-001` | 空词序列 | `E_SENT_INVALID` |
| `INJ-002` | 词性未在哈希表声明 | `E_FEATURE_UNDECLARED` |
| `INJ-003` | 动作权重非有限数 | `E_MODEL_INVALID` |
| `INJ-004` | 最大转移步数为零 | `E_CONFIG_INVALID` |

故障只影响命中句子；注入项编号、命中状态和业务码写入报告。

## 验证与产物

```bash
python -m unittest discover -s tests -v
python -m depworkbench verify --samples samples --out artifacts --seed 20261123
```

验收脚本默认清理临时工作快照，仅保留 JSON 证据。主要产物：

- `artifacts/demo_report.json`：基线完整诊断报告；
- `artifacts/verification_report.json`：逐项验收结果；
- `artifacts/edit_sequence.json`：替换、增删、约束、权重、模式切换序列；
- `artifacts/injection_report.json`、`budget_report.json`、`resume_report.json`、`resume_manifest.json`、`stale_report.json`；
- `artifacts/final_incremental_report.json`：编辑序列后的最终增量报告。

实际验证结果记录见 [docs/DESIGN.md](docs/DESIGN.md#实际验证结果)。
