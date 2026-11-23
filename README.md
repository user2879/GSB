# CTR/CVR 联合预估定点诊断组件

这是一个从空仓库实现的、面向搜索推荐排序场景的确定性 CTR/CVR 联合预估组件。它不使用深度学习框架，核心整数训练、量化、误差传播、排序翻转归因、增量缓存和调度均由 Python 标准库实现；仓库固定唯一第三方依赖为 `numpy==1.26.4`，当前实现本身不依赖 NumPy API。

组件求解的核心问题是：曝光样本滚动到达，CTR/CVR 共享底层特征和隐层，转化标签存在延迟，两个塔的定点分数进入融合排序。切换量化档位、定点小数位宽或分箱边界后，需要判断误差首先在哪个环节超差、排序翻转由 CTR 塔、CVR 塔还是融合量化造成，并证明增量运行与全量重建完全一致。

## 目录

```text
ctr_cvr_pipeline/
  config.py         配置加载与校验
  samples.py        JSONL 样本校验
  commands.py       编辑命令流状态机
  binning.py        等宽/等频分箱
  fixed.py          Qm.n 舍入、裁剪、残差与 sigmoid LUT
  features.py       逐特征定点化与内容缓存
  model.py          共享隐层 + CTR/CVR 双塔定点训练及 float64 参考训练
  visibility.py     基于训练事件时刻的转化延迟可见性
  fusion.py         定点融合、排序、翻转比较与指标
  diagnostics.py    五级误差传播、首次超差与翻转归因
  cache.py          内容寻址持久化缓存与复用账本
  pipeline.py       流水线编排、缓存键、报告与确定性摘要
  runner.py         增量/全量回放及一致性验证
  snapshot.py       不可变快照目录和原子 latest 指针
  cli.py            `python -m ctr_cvr_pipeline`
samples/            固定种子生成的基础样本、配置和编辑命令流
scripts/            确定性样本生成脚本
tests/              17 项 unittest
artifacts/          运行产物（快照、缓存、验证摘要）
```

## 快速开始

Windows 10/11 PowerShell 下：

```powershell
python --version
python -m pip install numpy==1.26.4
python scripts/generate_samples.py
python -m unittest discover -s tests -v
python -m ctr_cvr_pipeline replay --mode incremental --output artifacts/demo_incremental
python -m ctr_cvr_pipeline replay --mode full --output artifacts/demo_full
python -m ctr_cvr_pipeline verify --output artifacts/verification
```

也可以把 `python -m ctr_cvr_pipeline` 写成 `python main.py`。

Linux/macOS 命令相同，将 `python` 换成对应的 Python 3.11 可执行文件即可。`verify` 返回码为 `0` 表示增量、全量、单线程和多线程四种执行方式逐稳定态一致。

默认输入路径为：

```text
samples/exposure_samples.jsonl
samples/feature_config.json
samples/train_config.yaml
samples/commands.json
```

也可以用 `--samples-dir` 或分别指定 `--samples`、`--feature-config`、`--train-config`、`--commands`。

`replay` 默认清空指定输出目录以获得干净的复现实验；在输出目录中加上 `--keep` 可保留上一次的内容寻址缓存，使新进程再次回放时直接复用未失效阶段产物。`verify` 总是为四条验证路径建立隔离目录。

## 输入契约

### 曝光样本 JSONL

每一行是一个候选曝光样本：

```json
{
  "request_id": "R0001",
  "item_id": "I0001_0",
  "dense": {"price": 82.39, "dwell": 0.5, "quality": 0.045},
  "sparse_counts": {"category": 1, "brand": 2, "device": 1},
  "click": 0,
  "conversion": 0,
  "conversion_delay_ms": 1200
}
```

校验规则：

- `request_id`、`item_id` 必须是非空字符串；`(request_id, item_id)` 在完整到达样本集中唯一。
- `dense` 可为特征名到数值的对象，也可为按特征配置中 dense 特征顺序排列的数组。数值必须有限且非负。
- `sparse_counts` 必须是有限非负计数对象；缺失计数按 0 使用。
- `click`、`conversion` 必须为整数 0/1。
- `conversion_delay_ms` 必须为非负整数或 `null`；缺省等同于 `null`。
- 非法样本不会进入状态，失败项统一记录为 `E_SAMPLE_INVALID`，包含 `sample_id`、`request_id`、`item_id`、`field` 和原因。

### 特征配置 JSON

示例见 `samples/feature_config.json`。关键字段：

```json
{
  "seed": 20261124,
  "model": {"hidden_dim": 4},
  "quantization": {"tier": "Q16", "fractional_bits": 8},
  "features": [
    {"name": "price", "kind": "dense", "binning": "equal_width", "bins": 5},
    {"name": "dwell", "kind": "dense", "binning": "equal_frequency", "bins": 4},
    {"name": "category", "kind": "sparse"}
  ],
  "crosses": [{"name": "category_x_brand", "left": "category", "right": "brand"}],
  "shared_features": ["price", "category", "category_x_brand"],
  "tower_features": {
    "ctr": ["quality", "device", "category"],
    "cvr": ["price", "brand", "category_x_brand"]
  },
  "fusion": {"mode": "weighted_sum", "ctr_weight": 0.65, "cvr_weight": 0.35}
}
```

约束：

- dense 特征必须声明 `equal_width` 或 `equal_frequency`，`bins` 为正整数。
- `shared_features`、`tower_features.ctr`、`tower_features.cvr` 只能引用已声明特征或 cross 名。
- 允许档位为 `Q8/Q12/Q16/Q24`，表示有符号定点总位宽；允许小数位宽为 `4/6/8/10/12/14/16`，且必须满足 `fractional_bits <= tier_bits - 2`。
- 融合口径为 `product` 或 `weighted_sum`；权重有限非负，加权和不得同时为 0。
- 特征数值先分箱，再以箱中心 `(bin + 0.5) / bins` 定点化；稀疏计数直接定点化；cross 为其上游稠密箱中心或稀疏计数的乘积。

### 训练配置 YAML

仓库附带轻量 YAML 子集解析器，同时接受 JSON。示例：

```yaml
seed: 71029
learning_rate: 0.08
batch_size: 4
epochs: 2
positive_weight: 2.0
negative_weight: 1.0
conversion_weight: 1.5
conversion_delay_cutoff_ms: 6000
global_train_sample_budget: 24
per_batch_sample_quota: 4
thread_pool_workers: 4
snapshot_commit_interval_batches: 2
initial_event_time_ms: 0
tolerances:
  feature_quantization: 0.18
  shared_hidden: 0.12
  ctr_tower: 0.10
  cvr_tower: 0.10
  fusion: 0.08
```

`positive_weight`、`negative_weight`、`conversion_weight` 仅作用于 CVR 塔。CTR 塔始终使用未加权的点击 BCE。全局预算按每个稳定态的唯一样本点计数，不按 epoch 重复计算；训练轮次会对同一选中集合重复使用这些点。

### 编辑命令流 JSON

支持以下命令，按数组顺序逐条应用，每条命令后产生一个稳定态报告：

```json
{"type": "append_batch", "batch_id": "b0001", "event_time_ms": 1000, "samples": []}
{"type": "correct_label", "request_id": "R0001", "item_id": "I0001_0", "click": 1, "conversion": 1, "conversion_delay_ms": 0, "event_time_ms": 1000}
{"type": "modify_bin_edges", "feature": "price", "edges": [0, 10, 30, 55, 80, 120], "event_time_ms": 3000}
{"type": "switch_quantization", "tier": "Q12", "fractional_bits": 6, "event_time_ms": 3000}
{"type": "set_conversion_cutoff", "cutoff_ms": 2500, "event_time_ms": 3000}
```

`modify_bin_edges` 的 `edges` 必须包含 `bins + 1` 个严格递增边界；传 `null` 表示恢复自动边界。命令事件时间不得回退。非法命令返回 `E_COMMAND_INVALID`、`E_SAMPLE_NOT_FOUND` 或 `E_EVENT_TIME_INVALID`，并保持后续报告中的失败记录稳定有序。

## 计算语义

### 定点格式与舍入

- 参数和中间值使用有符号 Q 格式；总位宽由档位决定，小数位数由 `fractional_bits` 决定。
- 所有除法式缩放使用“最近整数、恰好半值时远离零”规则。
- 每一步缩放、梯度平均、学习率更新、裁剪都记录 `numerator`、`denominator`、`rounded`、`residual` 和 `direction`（`UP/DOWN/EXACT/CLIPPED_HIGH/CLIPPED_LOW`）。
- 输出 sigmoid 在初始化时物化为确定性的 1/16 分辨率整数 LUT；之后前向计算只做整数索引。
- float64 参考路径使用相同初始权重的反量化值、相同样本顺序、相同标签/可见性、相同 batch 和 epoch，不使用任何定点舍入，作为五级误差比较的真值。

### 分箱规则

- 等宽：在 `[min, max]` 上等距切分；所有值相同时固定退化为单箱。
- 等频：按排序后秩的 `floor(k*n/bins)` 位置取内部边界；边界并列值固定进入上箱；最后一箱承接剩余样本。
- 显式边界必须严格递增。边界、箱计数、来源、退化/并列规则和指纹写入报告的 `binning_metadata`。
- 定点化顺序固定为“请求 ID、候选物品 ID、特征名”升序；相同输入与档位重复运行逐字段一致。

### 延迟转化可见性

对每个样本和当前训练事件时刻 `T`：

- 未点击：CVR 目标 0，原因 `NOT_CLICKED`。
- 已点击但转化标签为 0：目标 0，原因 `NO_CONVERSION`。
- 转化标签为 1、延迟缺失：目标 0，原因 `MISSING_DELIVERY`。
- 转化标签为 1、延迟大于当前截断：目标 0，原因 `DELAY_EXCEEDS_CUTOFF`。
- 曝光事件时刻 + 延迟 > 当前训练事件时刻：目标 0，原因 `LABEL_PENDING`。
- 已到可见时刻：目标 1，原因 `VISIBLE`。

不可见转化在 CVR 训练中严格按负样本处理，原始转化标签不会进入更早轮次。训练缓存键只使用有效 CVR 目标、可见性状态、原因和权重；如果训练事件推进但没有改变任何可见性结果，模型阶段可直接复用；一旦任一可见性变化，模型阶段键立即变化并使下游失效。

### 预算与调度

调度器对所有当前样本按 `(request_id, item_id)` 排序：

1. 前 `global_train_sample_budget` 个样本进入训练。
2. 其余样本保留在 `pending_samples`，每个产生一条 `E_TRAIN_BUDGET_PENDING`，字段为 `global_train_sample_budget`。
3. 训练 batch 大小取 `min(batch_size, per_batch_sample_quota)`。
4. 线程池并发数来自 `thread_pool_workers`，只并行确定性特征缓存任务；结果按固定顺序回填、写缓存和记账，因此串行/并发逐字段一致。
5. `snapshot_commit_interval_batches` 控制训练检查点周期；检查点包含当时的参数指纹、整数参数和 float64 参考参数。

## 误差、指标与翻转归因

每个样本输出五个环节的绝对误差、相对误差、比较分支和容差判定：

1. `feature_quantization`
2. `shared_hidden`
3. `ctr_tower`
4. `cvr_tower`
5. `fusion`

真值为 0 时不做除零，改用绝对误差，并标记 `ABSOLUTE_ZERO_REFERENCE`。按上述顺序找到第一个超过对应容差的环节，写入 `first_exceeded`。`stage_summary` 给出每级超差样本数、最大绝对/相对偏差、最大偏差样本和元素。

融合使用定点域乘积或加权和后重定标；同分排序键为请求 ID、候选物品 ID 字典序，报告同时给出请求内名次和全局名次。

对每个稳定态，组件自动运行四个相邻量化变体：档位前后相邻、位宽前后相邻。对每个排序确实互换的候选对输出：

- 翻转前后名次和对端候选；
- CTR/CVR 分数差、两塔贡献差；
- 融合重定标与理论两塔贡献之差 `fusion_quantization_component`；
- `cause = CTR_TOWER | CVR_TOWER | FUSION_QUANTIZATION`；
- 两个候选的 `first_exceeded` 环节。

加权和模式直接在线性贡献域比较；乘积模式在对数贡献域比较。绝对值最大的分量决定主因，完整分量保留在 `cause_detail`。另输出 float64 参考排序相对定点排序的翻转清单。

评估指标包含定点/参考 CTR AUC、CVR AUC、融合 NDCG@5、Top-1 一致率及指标绝对误差。

## 缓存、失效与原子提交

缓存目录默认为输出目录下的 `cache/`，每个对象以内容 SHA-256 键寻址。阶段为：

```text
feature_quantization
shared_hidden
tower_outputs
fusion_ranking
diagnostic_summary
```

每个缓存包记录 `key`、`upstream_key`、样本引用、特征引用、payload 指纹和 payload。账本逐项记录 `REUSED` 或 `COMPUTED`，但不让缓存状态影响确定性核心摘要。

失效范围：

- 单样本标签修正：该样本所有特征定点化复用；模型及其下游失效。
- 某稠密特征分箱边界修正：只有该特征的定点化及引用它的 cross 重算，其他特征级缓存直接复用；模型及其下游失效。
- 量化档位/位宽切换：所有定点化和下游失效。
- 仅训练事件推进且可见性不变：特征、模型、两塔和融合均可复用，只重算诊断汇总。
- 可见性变化：模型键变化，精确阻断旧 CVR 标签流入后续训练。

稳定态快照先写入同文件系统的临时目录，再整体 `os.replace` 为不可变 `snapshots/step_XXXX_<digest>/`；`latest.json` 和 `index.json` 原子替换。缓存对象也使用临时文件加原子替换写入。

## 报告结构

每个 `report.json` 包含：

- `run`：模式、命令序号、当前档位/位宽、训练事件、状态 ID、`core_digest`。
- `config`：完整特征/训练配置和模型维度。
- `binning_metadata`：每个 dense 特征的边界、计数、来源和指纹。
- `training`：选中/待办样本、两塔损失轨迹、最终整数/参考参数、参数指纹、完整整数算术轨迹、周期检查点。
- `pointwise`：逐样本两塔和融合定点/float64 分数、重定标残差、逐特征量化记录、五级误差与首次超差。
- `stage_summary`：五级超差统计。
- `quantization_flips` / `reference_flips`：相邻量化与 float64 参考翻转归因。
- `metrics`：AUC、NDCG、Top-1 一致率和指标误差。
- `cache`：阶段键、详细复用/重算账本、复用率、账本签名。
- `failures`：非法样本、非法命令和预算待办，顺序稳定。
- `deterministic_core_digest`：排除运行模式和缓存 `REUSED/COMPUTED` 状态后的确定性摘要。

## 实际验证结果

验证环境：Python 3.11，NumPy 1.26.4，Linux 工作区（代码不使用平台特定 API，Windows 路径同样使用 `pathlib`/`os.replace`）。

固定样本：基础 8 条 + 命令追加 24 条 = 32 条；11 条编辑命令；预算 24，最终 8 条待办。

```text
python -m unittest discover -s tests -v
Ran 17 tests
OK (约 8.2s)

python -m ctr_cvr_pipeline verify --output artifacts/verification
consistent: true
step_count: 11
step_counts: incremental=11, full=11, serial_workers_1=11, concurrent_workers=11
serial_concurrent_consistent: true
final_core_digest: sha256:1e032845aa0006f04478b81a4fce45e8c84e17eadb7eb5f3c8b260b780559da8
```

11 个稳定态的增量摘要与逐步全量重建摘要全部相同。实测耗时约：增量 11.89s、全量 18.39s、1 线程 10.11s、4 线程 10.40s。

最终态五级统计（Q24/F10，32 个样本；除共享隐层 1 个样本超差外，其余环节无超差）：

```text
feature_quantization: exceeded=0, max_deviation=0.00390625
shared_hidden:        exceeded=1, max_deviation=0.1326155428
ctr_tower:            exceeded=0, max_deviation=0.0144092934
cvr_tower:            exceeded=0, max_deviation=0.0168574218
fusion:               exceeded=0, max_deviation=0.0123232428
```

最终评估指标：定点 CTR AUC `0.5483091787`、CVR AUC `0.7098214286`、NDCG@5 `0.8125`；float64 参考 CTR AUC `0.5410628019`、CVR AUC `0.71875`、NDCG@5 `0.8125`；Top-1 一致率 `1.0`。CVR AUC 使用当前训练事件下可见的有效转化目标，未使用尚未回传的原始转化标签。

命令流中实际观测到：

- Q16/F8 阶段：2 条翻转，归因为 `CTR_TOWER=1`、`FUSION_QUANTIZATION=1`。
- Q12/F6 阶段：最多 7 条翻转，归因为 `CTR_TOWER=4`、`CVR_TOWER=2`、`FUSION_QUANTIZATION=1`。
- Q8/F4 阶段：最多 7 条翻转，归因为 `CTR_TOWER=3`、`CVR_TOWER=3`、`FUSION_QUANTIZATION=1`；同时出现 float64 参考相对定点的翻转。
- 示例 `R0001:I0001_0` 对 `I0001_1` 在 F4→F6 变体下名次 1→2，CTR 贡献差 `-0.040625`、CVR 贡献差 `-0.04921875`、融合量化分量 `0.01171875`，主因判定为 `CVR_TOWER`，首次超差环节为 `shared_hidden`。
- 缓存复用示例：Q12/F6 截断调整后 `143 REUSED / 1 COMPUTED`；Q8/F4 标签修正后 `182 REUSED / 4 COMPUTED`。
- 跨进程验证：同一输出目录连续回放两次并第二次使用 `--keep`，两次核心摘要仍完全相同；第二次最终态为 `228 REUSED / 0 COMPUTED`，快照通过后缀避让保持不可变。
- 最终失败清单只有 8 条 `E_TRAIN_BUDGET_PENDING`，无非法样本。

这些结果保存在 `artifacts/verification/verification.json`；`artifacts/sample_final_report.json` 和 `artifacts/sample_q8f4_flip_report.json` 分别保留了最终态与 Q8/F4 翻转态的完整报告，均可由上述命令完整复现。

## 测试覆盖

`tests/` 使用标准库 `unittest`，覆盖：

- 非法样本字段、重复候选、非法配置和非正权重；
- 半值舍入方向、裁剪、分箱退化、重复运行一致；
- 双塔训练与 float64 参考训练的损失/参数逐字段复现；
- 延迟未到、已到达、超截断和缺失回传；
- 可见性未变时模型复用、可见性变化时训练失效且早期 CVR 目标为 0；
- 标签修正时特征级缓存全部复用，边界修正时只重算目标特征；
- 五级误差字段、零真值绝对误差分支和预算待办；
- 增量/全量、1 线程/多线程核心结果一致；
- 排序翻转与 CTR/CVR/融合量化归因；
- `weighted_sum` 与 `product` 两种定点融合口径；
- 固定种子样本生成器两次输出逐字节一致。

## 设计约束说明

- 排序名次默认按请求分组；同时提供跨请求的全局名次，完全按 `(-score, request_id, item_id)` 决断。
- 预算按稳定态的唯一样本点计，不按 epoch 重复消耗。
- 全量模式为每个稳定态从基础输入重新回放命令并重建，因此是刻意保留的 `O(steps²)` 验证路径；增量模式复用持久化缓存。
- 组件聚焦可解释、可复现的定点诊断，不追求生产级模型精度。所有训练规模、模型维度、线程数和预算均由本地配置显式控制。
