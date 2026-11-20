# featflow — 搜索推荐在线特征与样本链路组件

面向搜索/推荐场景的**在线特征计算 + 训练样本生成**参考组件，核心解决在
**并行线程池调度**与**双层资源预算约束**下，特征计算与样本生成的**状态一致性、
可复现性与精确失效**问题。

纯本地运行：仅依赖 Python 3.10+、NumPy、PyYAML；无 Docker、无外部服务、无网络请求，
所有输入由固定种子的脚本确定性合成，Windows 命令行可直接运行。

---

## 1. 目录结构

```
featflow/
├── src/featflow/
│   ├── common.py       # 业务码、定点量化(Decimal/ROUND_HALF_UP)、规范化JSON IO
│   ├── models.py       # Event / FeatureDef / LabelDef / InjectionRule 数据模型
│   ├── config.py       # YAML 链路配置加载与参数合法性校验
│   ├── loader.py       # 事件/定义表/标签/注入加载与 E_* 校验、DAG 拓扑(Kahn)
│   ├── storage.py      # 按实体存储事件，NumPy 做毫秒窗口边界检索
│   ├── compute.py      # 定点特征聚合(count/sum/mean/last)，基础+派生
│   ├── budget.py       # 全局预算 + 单特征配额双层调度，逐笔账本
│   ├── cache.py        # LRU 块缓存(访问序号+块键字典序决断)
│   ├── engine.py       # 点流编排、拓扑分层、缓存联动、注入隔离、快照提交
│   ├── sample.py       # 带标签生效延迟的样本生成、待定队列
│   ├── snapshot.py     # 临时目录原子替换、版本清单、过期识别
│   ├── recovery.py     # 断点恢复：校验清单、重建观测/账本/中止集合
│   ├── verifier.py     # 精确失效集合、增量vs全量逐字段比对、首差异定位
│   └── run.py          # CLI 入口(run / resume)
├── scripts/generate_inputs.py  # 确定性合成全部输入(LCG 整数随机数)
├── tests/              # 17 个 stdlib unittest 用例
├── data/               # 生成的输入(可随时重建)
├── out/                # 运行产物与报告
├── run_pipeline.py     # 便捷入口(Windows/macOS/Linux)
├── run_tests.py        # 一键测试入口
├── run_demo.bat / run_tests.bat
└── requirements.txt
```

## 2. 输入与规模档位

`scripts/generate_inputs.py --tier {smoke,small,medium} --seed 20261121`

| 档位 | 事件数 | 全局预算 | 单特征配额 | 用途 |
|---|---|---|---|---|
| smoke | 24 | 260 | 60 | 快速功能/预算压力 |
| small | 120 | 900 | 140 | 默认验收 |
| medium | 600 | 4500 | 700 | 性能与稳定性 |

固定种子 `20261121` 与整数 LCG 保证任何机器上生成的 JSONL 逐字节一致。
产出：`events.jsonl`、`feature_defs.json`、`label_defs.json`、`pipeline.yaml`、
`injections.json`、`edits.json`、`query_points.json`、`manifest.json`。

## 3. 快速开始

```bash
pip install -r requirements.txt
python scripts/generate_inputs.py --tier small --out data
python run_pipeline.py --data data --out out/small --serial      # 全量+增量审计
python run_pipeline.py --data data --out out/small --mode resume # 快照恢复
python run_tests.py                                               # 17 个测试
```

Windows：直接执行 `run_demo.bat` 与 `run_tests.bat`。

## 4. 输出产物约定（每个运行标签一个目录）

- `features.jsonl`   定点特征值（含 side/entity/anchor_ts/empty/empty_source/n_inputs）
- `samples.jsonl`    样本集合（ready/pending、判定时刻快照、缺失特征、标签）
- `budget_ledger.jsonl` 逐笔预算扣减账本（全局/单特征前后余额、决策原因）
- `failures.jsonl`   规范化失败清单（stage/target/code/detail/injection_id）
- `cache.json`       块注册表、LRU 命中/未命中/淘汰/重算遥测与事件流
- `diagnostics.json` 计数、业务码分布、注入命中、预算与缓存遥测
- 增量目录额外有 `invalidation_ledger.json`（失效/复用特征与块计数）
- 根目录有 `run_report.json`、`increment_equivalence.json`、`resume_report.json`、
  `snapshot_stale_check.json`、`concurrency_consistency.json`

---

## 5. 关键设计与确定性保证

### 5.1 定点量化（逐字节一致的核心）
- `FixedPoint(scale, quant_step)`：`steps = round_half_up(raw / step)`，
  再乘 `10^scale` 得到定点整数；全程 `decimal.Decimal` 精确运算，无浮点求和。
- mean 先把输入各自量化为定点整数，再做整数均值并 `ROUND_HALF_UP`，
  避免 numpy 并行归约的非结合性。
- 所有 JSON 经 `sort_keys + 固定分隔符 + UTF-8/LF` 规范化，重复运行产物哈希一致。

### 5.2 拓扑序增量计算
- 加载层用 Kahn 算法产出稳定拓扑序（同层按特征名字典序），并计算每个特征的 level。
- 每个计算点按 level 分层、层内按 `(topo_rank, name)` 稳定顺序执行。
- 基础特征在 `[anchor-window, anchor]`（**含边界**）内聚合原始事件，
  累加顺序固定为事件 ID 字典序；派生特征聚合上游**定点观测**。
- 空窗三态：`no_event`（实体无事件）、`no_source_value`（字段缺失/全空）、
  `no_upstream_obs`（派生特征无有效上游观测）。

### 5.3 双层预算调度（并发不影响结果）
- **所有预算决策与账本追加都在主编排线程**完成；线程池只执行无副作用的纯计算 callable。
- 全局预算 `global_event_budget` 与单特征配额 `feature_point_quota`（-1 为不限）。
- 单特征超额 → `E_FEAT_BUDGET_EXHAUSTED`；全局耗尽 → `E_GLOBAL_BUDGET_EXHAUSTED`，
  未完成任务进入 `pending_queue`；上游中止会级联标记下游。
- 账本 `seq` 单调连续，记录每次扣减前后余额，逐笔可审计。
- 因此 concurrency=1/4/8 的特征、样本、账本、失败清单、缓存注册表哈希完全相同。

### 5.4 LRU 块缓存与重算
- 块键 `feature|side|entity`；同访问序号按块键字典序决断淘汰。
- 滑动窗口随锚点变化，驻留不跨计算点；点内多个下游共享上游块。
- 被淘汰块在同点内再次被引用时触发重算，同配置下断言重算结果与首次逐元素一致；
  淘汰/命中/未命中/重算次数与完整事件流计入 `cache.json` 与诊断报告。

### 5.5 精确失效与增量/全量等价
- 独立 `IncrementalVerifier` 基于 DAG 后代闭包计算**精确失效集合**：
  - 定义变更（窗口/聚合）→ 该特征及其全部下游；
  - 事件回填 → 该用户侧与物品侧受影响基础特征及其全部下游（窗口会平移后续点）；
  - 仅定义变更、无回填时，不相关特征进入复用集合（见
    `test_definition_only_change_preserves_unrelated_features`）。
- 增量重放：复用块直接采用历史值且不重算，失效块经同一确定性引擎重算；
  预算规划仍对每个任务执行，保证账本/顺序/样本与全量一致。
- 比对覆盖 `features / samples / ledger / failures / cache.registry`，
  出现差异输出 `E_INCREMENT_MISMATCH` 并报告首个差异键路径。

### 5.6 样本与标签生效延迟
- `decision_ts = 事件 ts + delay_ms`；样本只读取 `point_seq <= 触发点` 且
  `ts <= decision_ts` 的可见快照，**不会读取未来事件**。
- 某观测特征因预算中止且无任何可见值 → 进入待定队列并记 `E_SAMPLE_PENDING`，
  绝不以空值或旧值静默填充。

### 5.7 快照原子提交、过期识别与恢复
- 加载/特征/样本三阶段，先写临时文件再 `os.replace` 原子替换。
- 清单记录全局版本号、定义表哈希、配置哈希、已完成事件游标与各阶段产物版本。
- 替换前校验哈希/游标漂移，不一致输出 `E_SNAPSHOT_STALE`（含自检测试用例）。
- `--mode resume` 从清单重建观测历史、预算已用量、剩余全局预算、中止集合与账本前缀，
  仅继续执行游标后的点；恢复结果与一次性全量运行逐字段一致。

### 5.8 异常注入与隔离
独立 `injections.json` 支持对指定事件/阶段注入：非有限上下文、未知行为类型、
回填时间戳早于游标。注入项编号（INJ-001…）写入失败清单与诊断报告证明命中；
单事件异常被隔离，其余事件继续处理。回填违规则输出
`E_BACKFILL_ORDER` 并拒绝写入。

## 6. 业务码映射

| 代码 | 触发场景 |
|---|---|
| E_EVENT_INVALID | 事件 ID 重复/字段缺失/时间戳非法/行为越界/非有限上下文 |
| E_DEF_CYCLE | 特征依赖图存在环 |
| E_DEF_PARAM | 窗口/聚合/上游/定点位宽/量化档位等参数非法 |
| E_BACKFILL_ORDER | 回填修正时间戳早于已完成游标 |
| E_SAMPLE_PENDING | 判定时刻所需特征不可用，样本待定 |
| E_FEAT_BUDGET_EXHAUSTED | 单特征计算点数配额耗尽（含上游级联） |
| E_GLOBAL_BUDGET_EXHAUSTED | 全局事件处理预算耗尽 |
| E_INCREMENT_MISMATCH | 增量与全量结果逐字段不一致（附首差异键） |
| E_SNAPSHOT_STALE | 快照版本/定义哈希/游标漂移 |
| E_INJECTION_HIT | 注入故障命中（携带 injection_id） |

代码与数字码的映射在 `pipeline.yaml: business_codes` 中声明。

## 7. 验收对照（详见 RUN_REPORT.md）

需求 1–8 全部由 17 个单测与 smoke/small/medium 三档端到端运行覆盖，
包括：加载校验与失败隔离、拓扑增量定点计算、精确失效与复用、标签延迟样本、
双层预算账本、LRU 淘汰重算、增量/全量逐字段等价、并发=串行一致性、
快照原子提交/过期识别/断点恢复、注入命中与隔离。
