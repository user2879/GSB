# 查询理解服务技术设计与实验说明

## 1. 模块总览

| 模块 | 文件 | 职责 |
| --- | --- | --- |
| 错误码 | `qu/errors.py` | 7 个稳定业务码与类型化异常 |
| 配置/指纹 | `qu/config.py` | 冻结配置、规范化 JSON、SHA-256 指纹 |
| 归一化 | `qu/normalization.py` | 全半角/大小写/空白/标点四阶段，修改位置与阶段指纹 |
| 加载校验 | `qu/loader.py` | 四类 JSONL 读取、权重有限性、悬空规则、ID 唯一性 |
| 实体槽位 | `qu/entity.py` | 三条候选路径 + 统一最大匹配消解 |
| 意图打分 | `qu/intent.py` | 长度加权 × 位置衰减 × 规则加分，减最大值归一化 |
| 策略选择 | `qu/strategy.py` | 预置策略表、代价估计、可审计决策 |
| 改写 | `qu/rewrite.py` | 三类规则、收益排序、剪枝逐条诊断 |
| 流水线 | `qu/pipeline.py` | 单样本阶段编排、异常隔离、阶段耗时 |
| 并发运行 | `qu/runner.py` | 线程池、稳定排序键、串行/并发逐字段比对 |
| 快照缓存 | `qu/snapshot.py` | 临时文件 + `os.replace` 原子提交、增量失效 |
| 实验编排 | `qu/experiment.py` | 多档运行、穷举等价验证、报告、信号处理 |
| 合成数据 | `qu/synthetic.py` | 单种子确定性数据生成 |

## 2. 关键算法与确定性

### 2.1 归一化

阶段顺序固定为 `width_fold -> case_fold -> whitespace_merge ->
punctuation_strip -> final_trim`。NFKC 折叠全角字符，`str.casefold()`
处理大小写；每个输出字符都通过对齐数组回溯到原始偏移。所有修改记录
携带阶段名、归一化位置、原始位置、旧字符、新字符；每阶段对
`{"in", "out"}` 计算独立指纹。

### 2.2 意图打分

```
score(intent) = Σ weight_t × len(t) × decay^position   (每次出现)
              + rule_bonus_weight × Σ rule.weight
normalized(intent) = score(intent) − max(score)
```

分值统一四舍五入到小数点后 10 位，浮点累加顺序由词表加载行号固定。
同分按类目名字典序选主。减最大值归一化使主意图恒为 0，其余 ≤ 0，
避免 softmax 等引入跨分值耦合。全程无随机源。

### 2.3 实体最大匹配与消解

三条候选生成路径产出**同一个候选集合**，只改变探测工作量：

* `exhaustive`：对每个词典条目执行 `str.find`（参考路径）；
* `inverted`：首字倒排，仅探测首字出现在查询中的条目；
* `prefix_group`：前两字分组，仅探测查询中出现过的前缀组。

消解规则全局唯一：跨度最长 → 词典类型优先级
（brand > category > attribute）→ 起始位置升序。被吞并候选进入
`suppressed`，并在 `resolutions` 记录命中的具体规则。因此剪枝只影响
`match_checks` 与耗时，槽位/抑制/消解结果在任意规模档位逐字段相同。

### 2.4 策略选择器

选择器枚举 3×2×2 = 12 个候选策略，基于显式单位代价常数估计单次查询
延迟，选择预算内最低代价者；全部超预算时选最低代价并在理由中标注。
实测分层（本机）：

| 词典规模 | 实体模式 | 意图模式 | 缓存 |
| --- | --- | --- | --- |
| 300 | exhaustive | full_scan | 关 |
| 2,500 | inverted | first_char_pruned | 关 |
| 10,000 | inverted | first_char_pruned | 关 |
| 30,000 | prefix_group | first_char_pruned | 开 |

决策指纹只包含 `entity_mode / intent_mode / cache_enabled` 三个算法
选择，规模指标仅用于审计，避免单条样本变化导致整档缓存失效。

### 2.5 改写与剪枝诊断

同义归一、类目扩展走首字倒排；丢词补全规则数量少且语义是“缺失触发”，
始终全量检查，保证剪枝路径不丢任何建议。诊断块区分两种截断：
`dropped_by_cap_rule_ids`（条数上限截断）与
`missing_due_to_pruning_rule_ids`（剪枝遗漏），并给出与不剪枝路径的
逐条 `item_by_item_comparison`。

### 2.6 并发一致性

每条样本带 `query_id::stage` 稳定排序键；并发结果在汇总前按键排序。
比对时剔除 `*_timings_ms` 等调度相关字段，对意图、槽位、抑制项、
消解、贡献、建议、失败清单等语义字段做规范化 JSON 逐字段比较。

### 2.7 快照与增量缓存

* 快照写入同目录临时文件，`flush + fsync + os.replace` 原子替换；
  强制终止只能看到完整旧快照或完整新快照。
* 缓存键 = `fingerprint(query_id, scope)`，`scope` 由词表指纹、策略
  指纹、档位标识派生；记录内另存样本内容指纹。
* 复用时逐条校验内容指纹：仅被替换样本失效重算，其余直接复用。
* 恢复时校验配置指纹、档位标识、快照版本，不符抛 `E_CONFIG_MISMATCH`。
* `SIGINT/SIGTERM` 置位停止事件，当前档快照提交后以退出码 0 结束。

## 3. 规模放大方法

档位为嵌套超集，词表/词典的核心词条在各档保持稳定，填充词条按
`前缀+行号` 确定性生成，保证跨档可复用且无碰撞。预置四档：

| 档位 | 查询 | 词表 | 实体 | 规则 |
| --- | --- | --- | --- | --- |
| tiny | 300 | 200 | 300 | 24 |
| small | 1,000 | 1,500 | 2,500 | 60 |
| medium | 5,000 | 6,000 | 10,000 | 160 |
| large | 20,000 | 20,000 | 30,000 | 320 |

连续扫档用 `scripts/scale_up.py --custom 名:查询:词表:实体:规则`，
同一固定种子下任意重复生成字节级一致。

## 4. 实测验证结果

完整四档运行（4 线程，预算 25 ms/查询，验证子集 256 条）：

| 档位 | 策略 | 平均 ms | p99 ms | 实体探测（剪枝/穷举） | 穷举等价 |
| --- | --- | --- | --- | --- | --- |
| tiny | exhaustive/full_scan | 0.27 | 0.77 | 76,800 / 76,800 | 是 |
| small | inverted/pruned | 0.33 | 0.97 | 1,068 / 640,000 | 是 |
| medium | prefix_group/pruned | 0.61 | 1.11 | 714 / 2,560,000 | 是 |
| large | prefix_group/pruned | 1.20 | 2.86 | 705 / 7,680,000 | 是 |

* 四档全部 `equal_to_exhaustive=true`，并发比对全部 `equal=true`。
* large 档实体探测次数下降约 4 个数量级，平均延迟仍远低于 25 ms 预算。
* 同配置重跑：1,300 条缓存命中、0 失效；替换单条样本后仅 1 条 miss、
  1 条失效。
* 加载层四类失败码（`E_INVALID_JSON / E_EMPTY_QUERY /
  E_DUPLICATE_QUERY_ID / E_RULE_DANGLING`）与非有限权重
  `E_INVALID_WEIGHT` 均进入报告失败清单。
* 不同并发度（1/2/4/8 线程）逐字段一致；配置变更恢复触发
  `E_CONFIG_MISMATCH`。

## 5. Windows 兼容说明

仅使用标准库与 NumPy/pytest；所有文件读写显式 `encoding="utf-8"`、
`newline="\\n"`，标准输出走 UTF-8 JSON。原子替换使用
`tempfile.mkstemp + os.replace`（Windows 上 `os.replace` 为原子 MoveFile
语义）。信号处理在非主线程或不支持平台上静默跳过，不影响主流程。
