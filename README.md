# 确定性指令微调数据构造器

本应用面向本地、固定种子、可审计的指令微调数据构造。它从种子模板、受控词表、打分参数、构造配置和分批计划出发，完成：

- 无全局随机状态的确定性候选生成；
- 四维质量打分、定点量化和量化残差记录；
- 子词 Jaccard 近重复分组、代表样本选择；
- 任务配额、全局评分预算、单批生成点数三重约束调度；
- 内容指纹缓存、精确增量失效、批级快照与续跑；
- 与同最终配置一次性全量构造逐字段一致的增量结果；
- JSON 审计报告、调度账本、待办队列与业务失败码。

不使用 Docker、容器或网络服务。测试和样本全部位于工作区内。

## 快速开始

要求 Python 3.11+，并安装 numpy：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

Linux/macOS 对应命令：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

校验、构造和验收：

```powershell
python -m instruction_factory validate --input-dir data --output-dir out
python -m instruction_factory run --input-dir data --output-dir out --mode incremental --reset
python -m instruction_factory run --input-dir data --output-dir out/full --mode full --reset
python -m instruction_factory verify --input-dir data --output-dir out/verification
python -m pytest -q
```

主要输出路径：

| 路径 | 内容 |
|---|---|
| `out/reports/latest.json` | 最近一次增量运行的完整审计报告 |
| `out/reports/batch_0001.json` ... | 每批完成后的报告 |
| `out/reports/validation.json` | 加载校验结果 |
| `out/snapshot/manifest.json` | 全局版本、配置/参数哈希、预算、批次数、阶段版本 |
| `out/snapshot/sampling/artifacts.json` | 采样阶段产物 |
| `out/snapshot/scoring/artifacts.json` | 打分阶段产物 |
| `out/snapshot/dedupe/artifacts.json` | 去重阶段产物 |
| `out/snapshot/aggregate/` | 有效配置与聚合报告 |
| `out/cache/` | 内容寻址缓存对象 |
| `out/verification/equivalence.json` | 增量/全量/重复全量等价结论 |
| `artifacts/acceptance.json` | 已执行的本地验收摘要与哈希 |

`run` 默认不重置快照。需要从空白状态运行时加 `--reset`。

## 本地样本

`data/` 由固定程序 `tools/generate_samples.py` 生成，可用下面命令核对：

```powershell
python tools/generate_samples.py --check
```

主样本包含 8 个种子模板、4 个任务类型、3 个受控词表来源和 8 个计划批次：

- 调整目标数；
- 替换模板变体和模板正文；
- 删除槽位来源；
- 调整四维打分权重；
- 删除任务类型；
- 新增任务类型；
- 追加最终目标数调整。

最终有效配置为 `extract=3`、`rewrite=5`、`summarize=7`，共 15 条候选。

## 输入契约

默认文件名可由 `--seeds/--lexicon/--scoring/--config/--plan` 覆盖。

### `seeds.jsonl`

每行一条种子。建议字段：

```json
{
  "seed_id": "seed-summarize-concise",
  "task_type": "summarize",
  "template_variant_id": "summarize.concise",
  "instruction_template": "用一句话概括“{topic}”并点出关键风险。",
  "slots": ["topic"],
  "reference_answer": "“{topic}”需要简明概括并标注关键风险。"
}
```

- `seed_id` 全局唯一；
- `task_type` 必须属于基础配置任务与计划 `add_task` 任务的并集；
- `template_variant_id` 在同一任务内唯一；
- 模板占位符必须由 `slots` 声明；
- 字符串化 ID 只允许 `[A-Za-z0-9][A-Za-z0-9_.-]{0,127}`。

### `lexicon.json`

```json
{
  "sources": {
    "core": ["数据安全", "访问控制"],
    "support": ["备份恢复", "密钥轮换"]
  },
  "subword": {
    "lowercase": true,
    "ngram_size": 2,
    "drop_chars": " ，。！？；：,.!?;:",
    "min_chars": 1
  }
}
```

拉丁字母/数字按连续串切分，中文按连续块切分，再生成 `ngram_size` 大小的子词集合。近重复相似度为两个子词集合的 Jaccard 系数。

### `scoring.json`

```json
{
  "dimension_order": ["clarity", "correctness", "format", "length"],
  "dimensions": {
    "clarity": {"weight": 0.30, "baseline": 76.0},
    "correctness": {"weight": 0.35, "baseline": 78.0},
    "format": {"weight": 0.20, "baseline": 80.0},
    "length": {"weight": 0.15, "baseline": 74.0}
  },
  "pass_threshold": 87.6,
  "ideal_length_chars": 26,
  "length_penalty_scale": 1.2
}
```

权重必须非负、有限且总和为正；维度顺序固定为 `clarity → correctness → format → length`。

### `config.json`

关键字段：

```json
{
  "task_types": {
    "summarize": {
      "target_count": 6,
      "template_weights": {"summarize.concise": 3.0, "summarize.bullets": 1.0},
      "slot_source_weights": {"core": 3.0, "support": 1.0}
    }
  },
  "task_quotas": {"summarize": 7},
  "near_duplicate_threshold": 0.65,
  "fixed_decimal_places": 3,
  "quantization_step": 0.01,
  "max_concurrency": 4,
  "global_score_budget": 90,
  "per_batch_generation_points": 25,
  "snapshot_commit_interval_batches": 2
}
```

允许的小数位宽为 `0..6`；量化档位为 `0.001/0.002/0.005/0.01/0.02/0.05/0.1/0.2/0.5/1.0`，且必须能在给定位宽下精确表示。线程池硬上限为 16。

### `plan.json`

每个批次在配置副本上原子应用；任一操作失败则整批不提交。支持：

- `add_task`、`delete_task`
- `set_target`
- `set_template_weights`
- `replace_template`、`delete_template`
- `set_slot_source_weights`、`add_slot_source`、`remove_slot_source`
- `set_score_weight`
- `set_quality_threshold`、`set_near_duplicate_threshold`

## 核心语义

### 确定性采样

候选 ID 为 `cand-{task_type}-{sample_seq:06d}`，生成顺序为任务类型字典序、任务内采样序号升序。没有使用 `random` 或进程全局随机状态；权重选择的随机整数来自：

- 种子文件 SHA-256；
- 任务模板权重或来源权重的内容指纹；
- 任务名、采样序号和用途标签。

权重键排序后累计；边界并列时取字典序更早的模板变体或来源名。模板正文替换只改变实际引用该模板的候选缓存键，不影响其他候选。

### 质量打分与定点量化

四个维度的原始分夹取到 `[0,100]`，按固定顺序计算：

```text
raw_total = Σ(weight_i × dimension_i) / Σ(weight_i)
quantum_units = int(step × 10^places)
fixed_total = round_half_up(raw_total / step) × quantum_units
residual = raw_total - fixed_total / 10^places
```

每个维度也按同一规则量化为定点整数。改变位宽会真实改变整数尺度；改变档位会改变量化网格。`total_fixed >= threshold_fixed` 视为通过；相等时保留并标记 `boundary=true`，边界清单按候选 ID 字典序输出。

单条候选的评分数值成本是：

```text
1 + (len(instruction) + len(reference_answer) + len(canonical(slot_values))) // 24
```

### 近重复与代表样本

- 以任务内子词集合的 Jaccard 相似度构图，使用并查集形成传递闭包组；
- 相似度 `>= threshold` 连边，恰好相等时标记 `threshold_equal=true`；
- 每组保留 `total_fixed` 最大者，同分时取候选 ID 字典序最小者；
- 被剔除项记录代表 ID、与代表的相似度、是否阈值等号；
- 组按最小采样序号排序，成员与剔除清单按采样序号排序。

### 三重预算调度

调度顺序固定为任务类型名、采样序号。线程池仅用于打分阶段，任务结果按稳定键 `candidate_id|scoring` 的提交顺序汇总。

- 任务配额不足：该任务剩余候选全部记录 `E_TASK_BUDGET_EXHAUSTED`；
- 全局评分预算不足：剩余候选记录 `E_SCORE_BUDGET_EXHAUSTED`，写入待办队列；
- 单批生成点数不足：剩余候选记录 `E_GENERATION_BUDGET_EXHAUSTED`，写入待办队列。

账本逐项记录候选 ID、三类预算的前值、扣减量、后值、状态与所需评分成本。

### 增量等价与缓存

每个计划批次应用后，应用按“当前有效配置的规范最终视图”重建结果。内容寻址缓存使未受影响候选直接复用，不对其重新采样、打分或分组。

缓存键分别由以下内容派生：

- 采样：种子文件指纹、采样序号、模板/来源内容指纹、槽位值；
- 打分：候选内容指纹、候选 ID、打分参数/切分规则指纹、量化参数、阈值；
- 去重：任务、候选顺序与评分缓存键、相似度阈值、切分规则指纹。

每个候选记录模板引用、逐槽位来源引用、采样缓存键与评分缓存键。失效清单按规范对象顺序输出 `reuse/recompute` 动作。

为使“最终增量结果 = 同最终配置全量结果”，预算账本采用当前有效配置的规范视图口径，而不把已删除历史任务曾消耗的预算带入最终账本。每组批次的生成点数预算同样作用于该批完成后的规范视图。该口径消除了历史计划路径对最终预算序列的影响。

### 快照与续跑

快照阶段为 `sampling → scoring → dedupe → aggregate`。提交时先写入 `.snapshot-staging`，再整体替换 `snapshot/`，清单最后写入。

启动比较：

- 种子、词表、基础配置、计划、打分文件哈希；
- 有效配置哈希与有效打分哈希；
- 各阶段产物版本；
- 全局版本是否可由清单字段重算。

不匹配时报告 `E_SNAPSHOT_STALE`，并从对应阶段开始重建；缓存对象仍可被复用。匹配时从 `completed_batches` 继续。

## 业务码

| 业务码 | 含义 |
|---|---|
| `E_SEED_INVALID` | 种子行、ID、任务枚举、模板占位符或重复变体错误 |
| `E_WEIGHT_INVALID` | 权重非有限/负数/总和非正，或删除后无正权重 |
| `E_CONFIG_INVALID` | 配置、词表、计划、量化、阈值、并发、预算等结构错误 |
| `E_TASK_BUDGET_EXHAUSTED` | 单任务配额耗尽 |
| `E_SCORE_BUDGET_EXHAUSTED` | 全局评分数值预算不足 |
| `E_GENERATION_BUDGET_EXHAUSTED` | 单批候选生成点数耗尽 |
| `E_SNAPSHOT_STALE` | 快照版本或哈希不一致，已触发阶段重建 |

失败项包含 `code/message/stage/file/line/field/path/candidate_id/task_type/batch_id/fatal` 中适用的字段。

## 报告内容

`latest.json` 包含：

- `task_statistics`：逐任务计划数、生成数、评分数、质量通过/剔除数、去重前后数、分组数和四维/总分分布；
- `candidates`：候选序列、模板/来源、槽位、四维定点分、总分、残差、边界标记、保留状态；
- `near_duplicate_groups`、`removed_duplicates`、`quality_rejections`；
- `scheduler` 与顶层 `budget_ledger`：预算账本、剩余预算、待办队列；
- `upstream_artifacts`：模板与槽位来源内容引用；
- `invalidation`：复用/重算清单；
- `snapshot`：全局版本、阶段版本、已完成批次数、变更/复用阶段；
- `failures` 与 `failure_business_codes`。

## 本地验收结果

在当前 Linux/WSL 工作区实际执行：

```text
python -m pytest -q
18 passed

python -m instruction_factory verify --input-dir data --output-dir out/verification
incremental_vs_full.equal = true
full_repeat_determinism.equal = true
ok = true
```

主样本最终结果：

- 计划候选：15，成功评分：15，质量剔除：1；
- 去重前：14，去重后：11，剔除重复：3；
- 保留代表样本：11；
- 边界样本：5；
- 全局评分预算：90，消耗 45，剩余 45；
- 单批生成点数：25，消耗 15，剩余 10；
- 主计划失败业务码：无；
- 增量最终批次复用：15 条采样、15 条评分、3 个去重产物全部命中缓存。

异常路径由测试覆盖：任务配额中止、全局预算待办、生成点数中止、阈值等号、快照失效重建、非法字段业务码与字段位置。
