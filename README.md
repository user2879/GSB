# 本地合成工单情感与风险判定工作台

这是一个只使用 Python 3.11+ 标准库的确定性工作台，用整数定点运算完成工单证据打分、半衰期衰减、互斥扣减、支持加成、裁决聚合、升级优先级判定、预算调度、增量复用和阶段化快照。

核心约束是：同一份最终证据集合、规则包与配置，无论采用串行还是线程池并行、无论缓存来自哪次编辑、无论快照以什么间隔提交，逐工单业务产物必须逐字段一致。

## 快速开始

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

python -m arbitration_workbench generate --out samples/base --seed 20261124
python -m arbitration_workbench run \
  --data samples/base \
  --state-dir artifacts/demo-state \
  --report artifacts/demo-report.json \
  --mode parallel

python -m unittest discover -v
python verify.py
```

本仓库没有第三方依赖。当前验证环境的 `ensurepip` 不可用，因此实际使用 `python -m venv --without-pip .venv` 创建了隔离环境；这不影响仅标准库运行。Windows 下所有路径均由 `pathlib.Path` 处理，快照写入同目录临时项后使用 `os.replace` 整体替换。

## 输入文件

`samples/base` 由固定种子 `20261124` 生成，包含 8 个工单、23 条证据、8 个编辑命令、5 个故障注入项。

### `evidence.jsonl`

每行一个 JSON 对象：

```json
{"workorder_id":"WO-0001","evidence_seq":1,"evidence_type":"login_anomaly","source_role":"system","event_ms":1700000000000,"terms":["refund","risk"],"confidence":0.8457}
```

校验规则：

- `workorder_id/evidence_seq` 组合必须唯一，`evidence_seq` 必须为正整数。
- `evidence_type` 必须存在于权重表，`source_role` 必须存在于角色可信度表。
- `event_ms` 必须为非负整数，`confidence` 必须落在 `[0,1]`。
- `terms` 必须为非空字符串数组。
- 空白行、坏 JSON、坏字段都会写入失败清单，不会静默跳过。

### `rules.json`

```json
{
  "evidence_type_weights": {"payment_fraud": 1.45},
  "source_role_credibility": {"system": 1.0},
  "mutual_exclusions": [
    {"id": "mx-001", "left_type": "payment_fraud", "right_type": "appeal_context"}
  ],
  "supports": [
    {"id": "sup-001", "left_type": "login_anomaly", "right_type": "device_risk"}
  ],
  "half_life_ms": 86400000,
  "priority_thresholds": [
    {"priority": "P1", "min_score": 0.0},
    {"priority": "P2", "min_score": 0.25},
    {"priority": "P3", "min_score": 1.0}
  ]
}
```

关系条目可带 `strength` 覆盖配置默认强度。关系 ID 必须全局唯一，端点类型必须存在。阈值按分值升序规范，最低阈值必须覆盖 0。

### `config.json`

```json
{
  "concurrency": 4,
  "batch_evidence_budget": 100,
  "workorder_evidence_quota": 10,
  "fixed_decimal_width": 4,
  "quantization_level": "Q5",
  "snapshot_commit_interval": 2,
  "exclusion_strength": 0.08,
  "support_strength": 0.05,
  "zero_lower_bound": 0.0,
  "fusion_mode": "SUM"
}
```

允许的定点位宽为 `{2,3,4,6}`，量化档位为 `Q1/Q2/Q5/Q10`（即整数步长 `1/2/5/10`）。`fusion_mode` 支持 `SUM` 和 `MEAN`。

### `input_meta.json`

记录输入版本。`apply-edits` 每次成功编辑后递增版本；快照启动时若当前输入版本小于快照版本，返回 `E_SNAPSHOT_VERSION_MISMATCH` 并要求全量重建。

## 整数定点算法

令 `Q = 10**fixed_decimal_width`，所有权重、角色可信度、标注可信度和衰减系数都先精确转成整数定点值。

### 半衰期衰减

以工单内最早 `event_ms` 为基准，`elapsed = event_ms - base_ms`。衰减系数使用纯整数的二进制分解与连续平方根保持在 32 位二进制小数定点：

```text
decay_bits = floor(2^32 * 2^(-elapsed / half_life_ms))
decay_q    = round_half_up(decay_bits * Q / 2^32)
```

当 `elapsed` 恰为半衰期整数倍时，走精确的 `2^-k` 右移路径，`half_life_boundary=true`，并把边界证据序号、算法、二进制位数和固定策略写入衰减产物元数据。每个衰减系数还记录 `decay_quantization_residual_num / 2^32`。

### 证据分

```text
raw_score_q = round_half_up(weight_q * role_q * confidence_q * decay_q / Q^3)
score_q     = floor(raw_score_q / quantization_step) * quantization_step
residual_q  = raw_score_q - score_q
```

固定运算顺序与残差都写入逐证据轨迹。

### 关系扣减与加成

- 先按关系 ID、证据序号对稳定排序执行全部互斥扣减，再执行全部支持加成。
- 每对互斥证据的两端分别扣减，支持关系两端分别加成。
- 每次结果使用 `max(zero_lower_bound_q, value)`，记录请求增量、实际增量、前后值、是否截断、关系 ID 和注入 ID。
- 这些中间轨迹全部进入关系阶段产物，并参与内容指纹。

### 裁决与优先级

证据按 `evidence_seq` 升序聚合。`SUM` 直接求和；`MEAN` 先定点四舍五入求均值，再按量化步长向下取整。每次累加值和最终量化残差可追踪。

阈值比较使用整数，边界策略固定为 `INCLUSIVE_UPPER`：裁决分恰好等于某阈值时归入该阈值对应档位，同时写入 `boundary=true`、匹配阈值与边界工单清单。

## 精确失效与缓存账本

三个下游阶段各有独立缓存键：

```text
decay_key       = H(证据内容, 工单涉及的权重子集, 涉及的角色子集,
                    半衰期, 位宽, 量化档, 衰减算法)
relations_key   = H(decay 内容指纹, 本工单实际适用的关系条目,
                    关系强度, 下界, 位宽, 量化档, 关系算法)
aggregation_key = H(relations 内容指纹, 阈值表, 融合模式,
                    位宽, 量化档, 边界策略, 聚合算法)
```

因此：

- 替换某条证据只失效该工单。
- 调整某证据类型权重只失效引用该类型的工单。
- 调整某角色可信度只失效使用该角色的工单。
- 增删关系只失效两端类型实际构成关系对的工单。
- 半衰期变化失效所有衰减；阈值变化失效所有聚合，但衰减与关系阶段仍可复用。
- 无变化的重复运行不重新打分、不扣减预算，三个阶段全部从持久化快照复用。

快照记录每个工单引用的证据序号集合、关系条目 ID，以及三阶段键和产物。复用/重算清单始终按工单 ID 升序。

## 预算与调度

- 一个证据打一分，成本为工单证据条数。
- 预算分配在提交线程前按稳定顺序完成，因此线程完成顺序不会改变预算结果。
- 上一批 `pending_workorders` 在下一批排到最前，其后按工单 ID 升序。
- 单工单成本超过配额：记 `E_WORKORDER_BUDGET_EXHAUSTED`，并扣减至多 `quota` 个可用点。
- 全局剩余预算不足但单工单不超配额：记 `E_EVIDENCE_BUDGET_PENDING`，不扣点，进入待办。
- 并行模式每个阶段使用 `ThreadPoolExecutor(max_workers<=concurrency)`；串行模式不启用线程池。
- 每个任务携带 `sha256(workorder_id:stage:stage-task-v1)` 稳定键。最终产物仍按工单 ID 排序。

## 阶段快照与恢复

快照目录结构：

```text
state/
  current.json
  snapshots/v00000001/
    manifest.json
    load.jsonl
    decay.jsonl
    relations.jsonl
    aggregation.jsonl
    records.jsonl
```

每次提交先在 `state/.tmp-snapshot-*` 写完全部阶段文件，再整体 `os.replace` 为不可变版本目录，最后原子替换 `current.json`。`manifest.json` 记录：

- 全局版本号、配置哈希、规则包指纹、输入版本；
- 已完成工单与待办工单；
- 本次和累计消耗预算；
- 阶段复用计数、提交原因。

启动时以 `config_hash` 检查快照是否可用于续跑。配置变化返回 `E_SNAPSHOT_CONFIG_MISMATCH`；输入版本回退返回 `E_SNAPSHOT_VERSION_MISMATCH`。两者都保留失败报告并要求显式全量重建（CLI 使用 `--rebuild-state`）。规则包变化不是恢复拒绝条件，而由内容键做工单级增量失效。

## 编辑命令

`python -m arbitration_workbench apply-edits --data DIR --edits edits.json` 支持：

- `replace_evidence`
- `delete_evidence`
- `set_evidence_type_weight`
- `set_role_credibility`
- `add_relation`
- `delete_relation`
- `set_relation_strength`
- `set_half_life_ms`
- `set_priority_threshold`

每条命令成功后才递增 `input_meta.json` 版本，并原子重写对应输入文件。

## 故障注入与诊断

`injections.json` 支持的真实故障：

| 故障 | 阶段 | 业务码 |
|---|---|---|
| `unknown_evidence_type` | load | `E_UNKNOWN_EVIDENCE_TYPE` |
| `confidence_out_of_range` | load | `E_CONFIDENCE_RANGE` |
| `negative_timestamp` | load | `E_EVENT_TIME_INVALID` |
| `zero_half_life` | decay | `E_HALF_LIFE_ZERO` |
| `relation_unknown_type` | relations | `E_RELATION_UNKNOWN_TYPE` |

注入作用域限定到指定工单和阶段。单个工单失败会记录工单 ID、阶段、业务码、异常类型和注入编号，其余工单继续执行。报告包含命中与未命中注入、优先级分布、边界工单统计、失败码分布和完整逐工单三阶段轨迹。

## 一致性比对口径

增量运行与空白全量运行的逐工单业务产物做逐字段比较，包括：

- 每条证据得分、衰减系数和量化残差；
- 关系扣减/加成轨迹；
- 裁决分、升级优先级和边界标记；
- 引用集合、三阶段缓存键、失败清单和汇总分布。

`reused_workorders / recomputed_workorders` 属于执行路径账本。全量运行天然没有复用，因此不能与增量账本逐项相等；本项目按以下方式分别验收：

- 增量账本自身固定顺序，并且复用集合与重算集合互斥、并集覆盖当前成功/失败工单；
- 空白全量账本固定为全部重算；
- 两条路径的业务投影逐字段相等。

## 实际验证结果

执行命令：

```bash
.venv/bin/python -m unittest discover -v
.venv/bin/python verify.py
```

结果：6 个单元测试全部通过，13 项验收检查全部为 `true`。详细机器可读结果在 `artifacts/verification_report.json`，逐次报告在 `artifacts/verification/`。

关键实测数据：

- 基线：8/8 工单成功，0 失败，消耗 23 点；串行与并行业务投影完全一致。
- 边界工单：`WO-0008` 裁决分 `0.2500`，恰等于 `P2` 阈值，归入 `P2` 并标记边界。
- 重复运行：8 个工单全部复用，三阶段复用计数均为 8，本批预算消耗 0。
- 首个编辑 `replace_evidence WO-0002/2`：仅 `WO-0002` 重算，其余 7 个工单复用。
- 8 个编辑步骤后的增量结果与最终输入的全量重建业务投影完全一致。
- 预算 7、每批固定上限时：首批消耗 6、剩余 1，待办为 `WO-0003..WO-0007`；第二批调度顺序以待办 ID 升序开头。
- 配额设为 2 时，6 个超配额工单按 ID 升序产生 `E_WORKORDER_BUDGET_EXHAUSTED`。
- 5 个注入项全部命中，分别映射到上表 5 个业务码，其余 3 个工单继续成功。
- 基线快照按间隔 2 提交 4 个全局版本；配置哈希或输入版本回退均返回对应 `E_SNAPSHOT_*` 并要求重建。

## 代码结构

```text
arbitration_workbench/
  canonical.py   规范化 JSON、SHA-256、定点与整数平方根
  codes.py       业务码、失败对象与稳定排序
  loading.py     证据/规则/配置/注入加载与校验
  models.py      输入与工单上下文模型
  scoring.py     衰减、关系、聚合与逐阶段内容指纹
  snapshot.py    阶段文件、临时目录和原子版本提交
  engine.py      失效键、预算调度、线程池阶段执行与报告
  editing.py     确定性编辑命令
  sample.py      固定种子样本生成器
  cli.py         命令行入口
tests/test_workbench.py
verify.py
samples/base/
artifacts/verification_report.json
```
