# 设计与验证说明

## 1. 模块划分

| 模块 | 职责 |
|---|---|
| `models.py` | 句子、模型、配置、约束、失败和注入的数据模型与校验 |
| `io.py` | UTF-8 JSON/JSONL 加载、受控词表校验、原子写出 |
| `quantize.py` | 浮点权重到定点整数域的确定性量化 |
| `parser.py` | 统一转移状态、打分、贪心/穷举搜索、步级缓存与失效判定 |
| `output.py` | 依存弧、扁平/嵌套树、参考树指标、产物键 |
| `snapshot.py` | 版本目录、临时写入、整体替换、`CURRENT` 指针 |
| `engine.py` | 调度、线程池、预算、注入隔离、增量运行、报告和快照提交 |
| `verify.py` | 一键验收场景与 JSON 证据生成 |

## 2. 统一转移状态

状态由五部分组成：栈、缓冲区游标、每个词的 head/relation、每个 head 的当前依存数、已选动作及累计分。两种模式共用同一 `_feasible_actions`、同一打分函数和同一 `_build_step` 快照构造。

动作约束在生成可行动作集时检查，排除项保留稳定业务式原因，例如：

- `E_LEFT_DISTANCE_LIMIT`
- `E_LEFT_POS_FORBIDDEN`
- `E_RIGHT_MAX_DEPENDENTS`
- `E_SHIFT_STACK_LIMIT`
- `E_FINISH_BUFFER_NOT_EMPTY`

完整 arc-standard 派生固定包含 `n` 次 SHIFT、`n-1` 次归约和 1 次 FINISH，共 `2n` 个选中转移。两种模式真正改变的是动作序列与搜索中实际求值的转移状态数；报告中的 `evaluated_transition_steps` 记录后者。

## 3. 定点打分

对每个可行动作，取偏置、栈顶/次栈顶/缓冲首词的词与词性、栈顶-缓冲首词性对等特征。特征表查到桶号后：

```text
weight_index = bucket % feature_dim
score(action) = sum(quantized_weight[action][weight_index])
```

量化公式为：

```text
integer = round_half_away_from_zero(value * 10^fixed_point_decimals / quantization_level)
```

所有比较都在整数域完成，不受线程调度或浮点求和顺序影响。

## 4. 步级缓存

每个步骤的公开键由以下内容派生：

- 该步可见句子内容指纹；
- 状态指纹和前一转移步键；
- 可行动作集合及分数指纹；
- 本步实际引用的特征哈希与权重单元；
- 模型形状、定点位宽、量化档位和模式。

缓存元数据同时记录 `token_refs`、`feature_hashes`、`weight_cells` 和约束指纹。贪心失效算法按以下顺序检测：

1. 模式、量化、模型形状或词数变化：从首步失效；
2. 句中引用词位内容变化：从首次引用该词位之步失效；
3. 特征桶或权重整数变化：从首次引用该单元之步失效；
4. 约束变化：只有重评该步可行动作集后确实变化，才从该步失效；
5. 全链无变化：整句不重算。

复用步骤只刷新展示用的栈/缓冲区快照，不重新计算分数或选择动作。这样即使未参与当前打分的未处理词被编辑，报告仍与全量运行逐字段一致。

### 穷举模式的全局边界

穷举目标是完整路径总分最大。若某权重只出现在旧的未选分支，沿旧选中前缀做依赖检查可能看不到变化，但它仍可能让另一条更早分叉的路径成为全局最优。因此穷举缓存额外保存解析搜索环境指纹。环境变化时重搜整句；环境不变时所有步骤直接复用。这样避免“增量选中路径”与“空白全量最优路径”不一致。

## 5. 快照协议

每次提交：

1. 在 `tmp/` 新建临时目录；
2. 写出 `load.json`、`parse.json`、`arcs.json`、`trees.json`、`manifest.json`；
3. `fsync` 文件后，把临时目录整体替换为 `versions/vNNNNNN`；
4. 原子替换 `CURRENT` 指针。

清单包含全局版本、模型/配置/语料哈希、已完成句子、逐句步骤游标与完成标记、四阶段版本。恢复时先比较版本和哈希；不一致输出 `E_SNAPSHOT_STALE` 并重新初始化。

崩溃测试钩子保留某句前 `k` 个转移步骤，再把快照提交到临时目录并整体替换。恢复时解析器从该步骤状态继续，剩余动作重新计算，因此最终语义与未中断运行一致。

## 6. 确定性排序

- 句子任务键：`sentence_id + 0 + "parse"`；
- 转移任务键：`sentence_id + step_index + "transition"`；
- 句子输出按 ID 升序；
- 弧按 `(sentence_id, head, dependent)`；
- 树按句子 ID；
- 失败按 `(sentence_id, line_no, code, injection_id)`；
- 注入按 `(sentence_id, injection_id)`；
- 预算按稳定选择顺序记录。

线程只影响实际执行时刻，结果先按任务键归集再写出，因此串行与并发逐字段一致。耗时字段是唯一允许随运行环境变化的诊断字段，不参与 `semantic_digest`。

## 7. 业务码

| 业务码 | 含义 |
|---|---|
| `E_SENT_INVALID` | 句子 JSONL 行、词/词性、重复 ID、gold 结构等非法 |
| `E_VOCAB_INVALID` | 受控词表缺失或非法 |
| `E_MODEL_INVALID` | 模型维度、桶、动作权重或特征表非法 |
| `E_CONFIG_INVALID` | 模式、定点、量化、并发、预算、步数等非法 |
| `E_FEATURE_UNDECLARED` | 句子的词或词性特征未在哈希表声明 |
| `E_PARSE_BUDGET_PENDING` | 本批预算不足，句子留在下一批优先队列 |
| `E_PARSE_DEAD_END` | 当前状态无可行动作 |
| `E_MAX_TRANSITION_STEPS` | 达到最大转移步数仍未完成 |
| `E_EXHAUSTIVE_NODE_LIMIT` | 穷举达到节点安全上限 |
| `E_SNAPSHOT_STALE` | 续跑快照版本或参数哈希不一致 |
| `E_INJECTION_INVALID` | 注入配置自身非法 |

## 8. 实际验证结果

验证日期：2026-11-23（Asia/Shanghai）。固定种子 `20261123`，10 个样本句。

自动化测试：

```text
Ran 18 tests in 45.654s
OK
```

验收脚本：

```text
all_checks_passed: true
check_count: 16
passed_count: 16
semantic_digest: 31005c5aeb18ec0597fc094d81f211cc45e21e5ff4416288c77e6657950bd17c
```

关键实测：

- 无编辑重跑复用全部 120 个主模式转移步骤，零重算；
- 词性编辑示例从第 7 步起失效，前 7 步复用，后 7 步重算；其余句零重算；
- 单权重编辑示例从第 13 步起失效，前 13 步复用，仅重算最后 1 步；
- 约束 `forbid_left_dependent_pos=["d"]` 触发 `CONSTRAINT_CHANGED`，而非整句无条件失效；
- 串行 1 线程与并发 4 线程的语义摘要、弧、树、失败清单完全相同；
- 注入 `INJ-001..004` 分别命中四个业务码，未注入句子继续成功解析；
- 快照在已完成 3 句、第 4 句保留 2 个转移步时中止；续跑后语义摘要与未中断参考运行完全相同；
- 修改模式后使用旧版本续跑得到 `E_SNAPSHOT_STALE`，重新初始化后的语义与空白全量一致；
- 编辑序列包含替换、新增、删除、约束修改、权重修改和模式切换，每一步均与同文件状态的空白全量运行一致；模式切换后 9 个句子选中动作序列改变，10 个句子的实际求值转移状态数不同。

精确报告与摘要位于 `artifacts/verification_report.json`、`artifacts/edit_sequence.json` 和 `artifacts/demo_report.json`。
