# 本地确定性跨任务迁移学习管线

该工作区实现了一个完全本地、CPU 单机、固定顺序的跨任务迁移学习实验管线。它围绕“基座层与任务数据演进时，哪些适配器仍可复用、哪些必须失效”这一核心问题，提供：

- 共享编码器前向与逐层表示指纹；
- 线性探针 / 低秩适配器的确定性训练和验证；
- 任务级缓存键、基座层依赖指纹与精确失效；
- 增量运行与空白工作区全量重建的逐字段对比；
- 线程池并发训练与单线程结果顺序一致性；
- 原子代际快照、强制终止后的稳定前缀恢复；
- 独立异常注入、任务隔离和 JSON 诊断报告；
- 固定种子样本生成和可重复演示。

最终验证结果见 `artifacts/demo/demo-verification.json`，最终任务报告见
`artifacts/demo/reports/final.incremental.json`。

## 快速开始

需要 Python 3.10+、NumPy 和 PyYAML：

```bash
python3 -m pip install numpy pyyaml
```

生成固定种子本地样本：

```bash
python3 -m transfer_pipeline generate --out samples/demo --seed 20261120
```

执行一次完整演示（会清空并重建 `artifacts/demo`）：

```bash
python3 -m transfer_pipeline demo \
  --root artifacts/demo \
  --sample-dir samples/demo \
  --seed 20261120
```

执行单个输入状态：

```bash
python3 -m transfer_pipeline run \
  --input samples/demo \
  --workspace artifacts/run \
  --report artifacts/run-report.json
```

执行全部编辑批次，并在每批后与空白工作区重建比较：

```bash
python3 -m transfer_pipeline run-edits \
  --input samples/demo \
  --workspace artifacts/edit-run \
  --reports artifacts/edit-reports
```

运行测试：

```bash
python3 -m unittest -v
```

当前环境实测：9 个单元/回归测试通过；演示的 15 项检查全部通过。

## 目录结构

```text
transfer_pipeline/
  inputs.py       # 输入解析、校验、编辑操作、标准化配置
  encoder.py      # 共享编码器前向、表示指纹
  adapters.py     # 线性探针/低秩适配器、训练、评估
  pipeline.py     # 缓存键、失效判定、并发、报告编排
  snapshot.py     # 代际快照、原子 CURRENT 指针、稳定前缀
  samplegen.py    # 固定种子样本、编辑计划、注入配置生成
  workflow.py     # 增量/全量对比、崩溃恢复、并发与拒绝续跑验证
  cli.py          # 命令行入口
samples/demo/     # 固定种子生成的本地输入
artifacts/demo/   # 已执行的演示、快照、恢复报告与最终报告
tests/            # 回归测试
```

## 输入格式

`samples/demo` 由固定种子（默认 `20261120`）生成，包含 160 个样本、9 个任务，不依赖下载或真实模型权重。

### `tasks.json`

```json
{
  "schema_version": 1,
  "shared_feature_dim": 6,
  "target_layer_whitelist": ["layer_1", "layer_2", "layer_3"],
  "tasks": [
    {
      "name": "vision_alpha",
      "classes": ["alpha", "beta"],
      "train_ids": ["s0001"],
      "validation_ids": ["s0129"],
      "target_layer": "layer_1",
      "target_layer_whitelist": ["layer_1", "layer_2", "layer_3"]
    }
  ]
}
```

加载时校验：任务名唯一、类别非空且不重复、训练/验证样本存在且维度匹配、标签属于任务类别、训练/验证 ID 不重叠、目标层在白名单且存在于编码器。单任务校验失败写入 `E_TASK_INVALID`；验证集为空保留主业务码 `E_TASK_INVALID`，并以 `reason_code: E_VALIDATION_EMPTY` 给出细因。

### `features.jsonl`

每行一条：

```json
{"sample_id":"s0001","features":[0.1,0.2,0.3,0.4,0.5,0.6],"label":"alpha"}
```

### `encoder.json`

权重矩阵形状为 `[input_dim, output_dim]`：

```json
{
  "shared_feature_dim": 6,
  "normalization": {"mean": [0,0,0,0,0,0], "std": [1,1,1,1,1,1]},
  "layers": [
    {
      "name": "layer_1",
      "version": "1.0.0",
      "weights": [[0.1, 0.2]],
      "bias": [0.0, 0.1]
    }
  ]
}
```

校验层名唯一、输入/输出维度链连续、偏置维度匹配、参数全部有限。失败输出 `E_ENCODER_INVALID`，不会静默加载部分编码器。

### `config.yaml`

同时支持扁平字段和 `adapter` / `training` / `validation` / `execution` 嵌套字段。示例：

```yaml
activation: tanh
adapter_type: low_rank
rank: 2
scale: 0.9
learning_rate: 0.08
epochs: 16
batch_size: 4
validation_metric: macro_f1
validation_threshold: 0.45
max_concurrency: 3
snapshot_interval: 2
seed_offset: 20261120
```

适配器类型别名：`linear` / `linear_probe` / `probe`，以及 `low_rank` / `lora`。零秩不会在配置加载时被悄悄改正，而是进入适配器构造前的 `E_RANK_INVALID` 失败路径。

### `edit_plan.json`

每批包含 `batch_id` 和 `operations`。支持的编辑：

- `add_task`：新增完整任务；
- `delete_task`：按名称删除任务；
- `set_classes`：替换类别集合；
- `replace_sample_features`：替换真实特征库中的样本特征/标签；
- `update_adapter_config`：修改全局适配超参数，支持点号路径；
- `replace_base_layer`：替换基座层权重、偏置和版本。

样本与 `encoder.json` 位于真实输入路径；不是凭空构造的训练内存对象。

### `injections.json`

演示覆盖：

- `INJ-001` 缺失样本 ID → `E_TASK_INVALID`
- `INJ-002` 重复类别 → `E_TASK_INVALID`
- `INJ-003` 目标层不在白名单 → `E_TASK_INVALID`
- `INJ-004` 零秩 → `E_RANK_INVALID`
- `INJ-005` 空验证集 → `E_TASK_INVALID` + `reason_code: E_VALIDATION_EMPTY`
- `INJ-006` 训练阶段运行时异常 → `E_TRAINING_EXCEPTION`

报告中的 `injection_hits` 会给出注入编号、任务、阶段、业务码与命中状态；`injection_summary.unmatched_ids` 必须为空。

## 确定性设计

### 共享前向

1. 按样本 ID 字典序处理样本；
2. 使用编码器归一化统计量做 `(x - mean) / std`；
3. 层按层序号升序执行 `z = z @ W + b`，再应用固定激活 `identity` / `relu` / `tanh`；
4. 每层、每样本表示单独生成指纹；
5. 整体表示指纹由编码器版本指纹、样本内容指纹、激活函数与逐层表示指纹派生；
6. 编码器权重和归一化数组加载后标记为只读，前向不会写回权重数组。

重复运行使用同一输入与基座时，`representation_fingerprint` 完全一致。

### 适配器

线性探针：

```text
logits = z @ W + b
```

低秩适配：

```text
logits = scale * (z @ A) @ B + b
```

训练使用排序后的训练样本 ID、固定批大小、固定学习率和每轮固定批顺序。初始化种子由缓存键、适配器类型、秩和缩放系数稳定派生。类型、秩或缩放系数变化会改变参数初始化、参数指纹和训练损失轨迹；线性探针的旁路没有低秩矩阵，秩字段仍作为确定性初始化盐参与种子派生。

验证计算准确率和宏平均 F1。按配置选择比较指标，报告中同时包含 `passed` 与 `threshold_failures`，执行失败任务不会进入验证汇总。

### 排序与并发

每个任务使用 `SHA256(task_name + NUL + target_layer)` 作为稳定排序键。输入校验失败、任务结果、适配器摘要、损失轨迹、验证表和失败清单均按该键及确定性次级字段排序。

线程池仅用于彼此独立的任务适配器训练，最大并发数严格取自配置。任务结果在完成后先写入按稳定键索引的字典，再按稳定顺序生成报告，因此线程调度顺序不会泄漏到输出顺序。快照只提交按稳定排序键形成的已完成前缀，而不是完成顺序。

## 缓存键与失效

任务缓存键由以下内容派生：

- 任务名、排序后的类别集合；
- 训练与验证样本的 ID、特征和标签；
- 目标层、目标层白名单；
- 编码器归一化指纹；
- 从首层到目标层的所有层版本与内容指纹；
- 激活函数；
- 适配器类型、有效秩、缩放、学习率、轮数、批量、种子偏移；
- 验证指标与阈值。

因此：

- 替换某层权重只会使目标层位于该层之上（含该层）的任务失效；
- 修改某任务的样本或类别只改变该任务的缓存键；
- 超参数或适配器类型变化会使使用这些参数的任务失效；
- 基座层变化位于目标层之后时，目标表示未变化，缓存仍可复用；
- 同缓存键的任务直接从快照产物复用，不调用训练函数。

演示中的 `E1` 批次替换 `layer_2`：目标层为 `layer_1` 的 `inj_zero_rank` 被复用，目标层为 `layer_2/3` 的任务失效，验证了基座级失效的精度。

## 快照与恢复

每次提交生成一个新的完整代际目录：

```text
snapshots/g000009-<manifest-hash>/
  manifest.json
  shared_forward.json
  adapters.json
  evaluations.json
  task_cache.json
  failures.json
```

`CURRENT.json` 是唯一的活动指针，使用临时文件 + `os.replace` 原子替换。一个新代际只有在全部阶段文件写入并重命名为完整目录后才被指针引用；因此强制终止最多留下不可达临时目录，不会让半写快照成为恢复点。

`manifest.json` 记录：

- `global_version`；
- `config_hash`；
- 基座整体指纹与每层 `name / version / fingerprint`；
- 输入状态指纹；
- 已完成任务稳定前缀；
- `shared_forward` / `adapters` / `evaluations` 三个阶段版本；
- 每任务缓存键与状态。

启动时：

- 直接恢复要求快照 `config_hash` 与当前配置一致；
- 若调用方提供 `expected_global_version`，版本必须一致；
- 不一致抛出/输出 `E_CONFIG_MISMATCH` 并拒绝续跑，不修改当前指针；
- 强制终止后下一次运行读取最后完整替换的代际，并在报告 `recovery` 中输出恢复代际、版本、已完成任务和共享前向是否复用。

`run-edits` 遇到计划中的超参数修改时采用显式“新配置谱系”语义：记录旧快照与新配置哈希不一致、清除旧快照谱系，然后以新配置重建。直接 `run` 不会自动迁移；它严格拒绝续跑。这区分了“明确编辑”和“误用旧快照”。

## 报告字段

每次完整运行生成 JSON 报告，主要字段：

- `input_state_hash`、`config_hash`、`config`；
- `base`：基座指纹、层指纹、相对上一快照的变化层；
- `task_results`：每任务参数、参数形状、参数指纹、损失轨迹、验证细节；
- `adapter_summary`、`loss_trajectories`、`validation_metrics`；
- `validation_summary`：只汇总验证成功的任务；
- `threshold_failures`：阈值未达标但执行成功的任务；
- `failures`：加载、构造、训练、验证失败，含业务码、阶段、异常类型、注入编号；
- `lifecycle`：`invalidated`、`reused`、`retrained`、`new_tasks`、`removed`、`not_trained`；
- `injection_hits`、`injection_summary`；
- `snapshot`、`recovery`、`execution`。

生命周期不变量由测试保证：

- `reused ∩ retrained = ∅`；
- `invalidated ∩ reused = ∅`；
- 零秩任务出现在 `not_trained`，不进入训练；
- 失败任务不进入验证汇总。

### 与全量重建的比较口径

空白工作区重建没有历史缓存，因此其生命周期动作必然相对于“空缓存”：所有有效任务都是新建、复用列表为空。增量运行则需要保留真实的 `reused / invalidated` 证据。故比较器比较所有由最终输入和最终模型决定的物化字段：

- `input_state_hash`、`config_hash`；
- 基座指纹与逐层指纹；
- 每任务适配器参数、参数形状、参数指纹、损失轨迹；
- 验证指标、验证汇总、阈值失败；
- 失败清单及其顺序、注入命中与汇总。

生命周期字段作为“如何到达该状态”的独立层，通过上述互斥不变量、精确失效清单和幂等重跑验证；幂等重跑必须全部复用且零重训。该分层避免了把“空缓存重建天然没有复用”误判为模型输出差异。

## 实测结果

`python3 -m transfer_pipeline demo --root artifacts/demo --sample-dir samples/demo --seed 20261120` 的实际结果：

- 样本：160 条；任务描述：9 项；其中 4 个初始有效任务、5 个加载注入失败任务、1 个运行时训练异常注入。
- 15 项演示检查全部为 `true`，包括全量重建一致性、并发等价、崩溃恢复、配置哈希拒绝续跑、注入命中、零秩预构造拦截、基座层精确复用。
- 并发对比：`max_concurrency=1` 时观测最大在飞任务数为 1，`max_concurrency=5` 时为 3；两种调度下任务物化结果完全一致。
- 强制终止：在稳定前缀 2 的完整快照提交后抛出 `E_SIMULATED_CRASH`；恢复时读取同一代际，恢复任务 `vision_alpha`、`vision_gamma`，共享前向直接复用，并与全量重建逐字段一致。
- 配置不匹配：修改验证阈值后直接续跑被拒绝，返回 `E_CONFIG_MISMATCH`，原活动快照指针保持不变。
- 编辑批次 `E1`：`inj_zero_rank` 被复用，`vision_alpha / vision_gamma / signals_abg` 失效，`signals_mix` 删除；与全量重建一致。
- 编辑批次 `E2`：秩 `2→3`、缩放 `0.9→0.55`，记录显式配置迁移，全部任务重训并与全量重建一致。
- 编辑批次 `E3`：只修改 `signals_abg` 类别与 `vision_gamma/audio_alpha` 使用的 `s0003` 样本；`vision_alpha`、`inj_zero_rank` 直接复用，其余按依赖失效。
- 最终幂等重跑：5 个有效任务全部复用，`retrained=[]`，物化结果与上一轮完全一致。
- 最终报告版本：`global_version=9`，代际 `g000009-c09b6feaae`；6 个注入编号全部命中，`unmatched_ids=[]`。

## 可复现性说明

- 所有哈希使用稳定 JSON 序列化（键排序、紧凑分隔符、统一 UTF-8）。
- 所有随机数来自固定种子的 `numpy.random.default_rng`。
- 包导入时默认将 OpenBLAS/OpenMP/MKL/NumExpr 线程数设为 1，减少 BLAS 线程调度差异。
- 报告不写入当前时间、进程号或随机 UUID；输出路径和临时目录名不出现在物化比较字段中。
- 样本、编辑计划和注入配置都由 `samplegen.py` 程序化生成，不访问网络。
