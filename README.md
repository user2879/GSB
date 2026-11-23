# 长上下文位置编码外推诊断管线

这是一个仅依赖本地文件系统、Python 3.11+ 与 NumPy 的确定性诊断工具。它从确定性的注意力样本出发，完成 RoPE 位置编码、点积注意力、训练长度内外统计、逐级误差追踪、预算调度、内容指纹缓存、阶段快照与可复现报告。

## 交付内容

- `posdiag/`：校验、编码、打分、误差、调度、缓存、快照、流水线与 CLI 实现。
- `samples/generate_samples.py`：固定种子样本/配置/编辑计划生成器。
- `samples/`：已生成的本地确定性输入。
- `tests/`：22 项 `unittest` 测试。
- `scripts/run_acceptance.py`：端到端验收与证据生成脚本。
- `artifacts/`：三方案报告、增量一致性报告、预算边界报告、快照、测试日志和验收摘要。

实际验证环境：Python `3.11.0rc1`、NumPy `2.2.6`、16 核，配置并发度 `4`。

## 快速开始

```bash
python --version
python -m pip install -r requirements.txt

# 重新生成确定性样本（可选）
python samples/generate_samples.py --output samples --seed 20261124

# 单方案诊断
python -m posdiag run \
  --config samples/position_config.json \
  --samples samples/attention_samples.jsonl \
  --workspace .posdiag_state \
  --report artifacts/diagnostic_report.json

# 同一批输入运行三种方案
python -m posdiag run \
  --config samples/position_config.json \
  --samples samples/attention_samples.jsonl \
  --workspace .posdiag_state_all \
  --report artifacts/all_schemes_report.json \
  --all-schemes

# 应用编辑计划（方案切换、基频调整、样本替换、精度档位变更）
python -m posdiag run \
  --config samples/position_config.json \
  --samples samples/attention_samples.jsonl \
  --edit-plan samples/edit_plan.json \
  --workspace .posdiag_state_edited \
  --report artifacts/edited_report.json

# 测试与完整验收
python -m unittest discover -s tests -v
python scripts/run_acceptance.py
```

`--stop-after-stage loading|encoding|attention|error_trace` 提供可重复的优雅停止测试入口；CLI 同时安装 `SIGINT`/`SIGTERM` 处理器，收到信号后完成当前阶段写入再退出。`validate` 子命令只做输入校验。

## 输入格式

### 位置编码配置

```json
{
  "scheme": "rope_standard",
  "base_frequency": 10000.0,
  "training_context_length": 32,
  "extrapolation_target_length": 128,
  "dimension": 16,
  "scaling_factor": 2.0,
  "precision": "strict",
  "batch_point_budget": 64,
  "global_mac_budget": 200000,
  "thread_pool_concurrency": 4,
  "snapshot_commit_interval": 4,
  "tolerance": 0.35,
  "global_version": 1,
  "comparison_schemes": []
}
```

支持的方案名为 `rope_standard`、`rope_linear`、`rope_ntk`，也接受常见简写。`precision` 为 `strict`、`balanced` 或 `compact`。所有数学过程固定使用 float64；精度档位只控制最终 JSON 数值量化位数，并参与缓存键。`comparison_schemes` 可用于一次运行多个方案；CLI 的 `--all-schemes` 会覆盖为三种方案。

配置校验失败会产生 `E_CONFIG_INVALID`。并发数必须为正整数且不超过本机 CPU 核数。

### 注意力样本 JSONL

每行一个样本：

```json
{
  "sample_id": "s0001",
  "query_position": 4,
  "key_positions": [0, 1, 2, 3, 4],
  "query_vector": [0.1, -0.2, 0.3, 0.4],
  "key_vectors": [[0.2, 0.1, -0.1, 0.4]]
}
```

`key_positions` 可为空；`key_vectors` 必须与键数量一致。为方便手工输入，也支持 `key_vector` 作为所有键共享的向量。查询位置、键位置必须是非负整数，向量必须等长于配置维度并全部有限。坏行、重复 ID、维度错误或非有限数会产生 `E_SAMPLE_INVALID`，该行被跳过，其余样本继续处理。

空键列表固定产出空打分/空权重，并记录 `empty_key_list` 边界原因；查询位置出现在键位置中时正常计算自注意力，并记录 `self_attention`。

### 编辑计划

`samples/edit_plan.json` 同时演示四类核心编辑：

```json
{
  "operations": [
    {"op": "switch_scheme", "scheme": "rope_ntk"},
    {"op": "adjust_base_frequency", "factor": 1.5},
    {"op": "change_precision", "precision": "compact"},
    {"op": "replace_sample", "sample_id": "s0007", "sample": { "...": "full sample" }}
  ]
}
```

另支持 `change_scaling_factor`、`change_training_context`、`change_extrapolation_target`、`change_tolerance`、`change_budgets`、`change_concurrency`、`bump_version`。单个非法操作产生 `E_EDIT_INVALID`，不会阻止后续合法操作。

## 计算与诊断语义

### 三种位置编码

维度为 `d`，旋转频率索引为 `i=0..d/2-1`：

- 标准 RoPE：`freq_i = base^(-2i/d)`，角度为 `position * freq_i`。
- 线性内插：保持基频，将位置替换为 `position / scaling_factor`。
- NTK 感知缩放：`base_eff = base * scaling_factor^(d/(d-2))`（`d=2` 时指数取 1），角度仍由原位置生成。

统一接口 `encode_sequence()` 返回查询向量、键向量、有效位置与频率指纹。输入和运算均为 float64；重复运行逐元素一致。报告保存位置向量 SHA-256 摘要、范数与频率指纹。切换方案会改变实际旋转角度，继而改变点积分数，而不是只改方案标签。

### 注意力与分层统计

注意力分数为旋转后查询向量与每个旋转后键向量的 float64 点积；归一化权重使用减最大值的稳定 softmax。

距离定义为 `abs(query_position - key_position)`：

- `distance < training_context_length`：`in_training`。
- `distance >= training_context_length`：`extrapolation`，其中 `distance == training_context_length` 是训练边界外第一档。

每个样本记录 in/out 分层的 count/mean/min/max；全局报告进一步按方案、桶、同距离汇总 count/mean/min/max。

### 逐级误差

参考路径使用 `min(position, training_context_length - 1)` 将位置夹紧到训练上界，并使用同一方案重新编码；实际外推路径使用原位置。依次计算：

1. `position_encoding`：外推查询/键位置相对参考向量的最大相对 L2 偏差。
2. `attention_score`：含外推位置的键对的最大相对点积误差。
3. `attention_weight`：实际与参考归一化权重向量的最大绝对差。

按上述顺序标记第一个 `deviation > tolerance` 的环节；全部未超差时标记 `none`。报告为每个方案保存三阶段偏差的 min/mean/p50/p95/max、完整排序值，以及逐样本首次超差环节。

### 预算与调度

一个样本的点数定义为 `1 + 2 * len(key_positions)`：查询编码 1 点、每个键编码 1 点、每个键打分 1 点。`batch_point_budget` 是每个 sample-scheme 任务的单样本配额。全局 MAC 预算换算为全局点容量 `floor(global_mac_budget / dimension)`，每个计划点按 `dimension` 次乘加计费。

调度器先按 `(sample_id, scheme)` 稳定排序并串行预分配预算，再提交线程池执行。因此预算扣减序列不依赖线程完成顺序：

- 任务所需点数超过单样本配额：扣减配额，记 `E_SAMPLE_BUDGET_EXHAUSTED`，不产出该任务结果。
- 全局剩余点数不足以完成下一任务：从该任务起全部进入 `pending_queue`，记 `E_POINT_BUDGET_PENDING`。
- 线程池 `max_workers` 取配置并发数与本批未命中任务数的最小值，绝不超过配置。

### 确定性

样本任务和阶段任务键为 `(sample_id, scheme)`，报告内样本、方案、超差清单、边界清单、预算扣减序列和 pending/exhausted 清单均按稳定键排序。并发只改变执行完成时间，不改变数值或顺序。

报告的确定性诊断字段包括配置、样本结果、分层统计、误差分布、超差/边界清单、预算使用与校验信息。`run_metadata` 单独保存 `run_id`、时间、缓存命中、恢复判定、陈旧快照码、观测并发度与快照路径。增量一致性的“逐字段”比较针对完整报告去除 `run_metadata` 后的全部字段；测试对每一类编辑都执行该比较。

## 指纹缓存与增量失效

缓存键由规范 JSON 的 SHA-256 派生：

- 编码键：缓存 schema、方案参数（方案、基频、训练/目标长度、维度、缩放因子）、样本内容指纹、精度档位、规模档位。
- 注意力键：编码键 + 打分规则版本。
- 误差键：注意力键 + 容差 + 参考位置规则版本。

因此：

- 方案、基频、缩放因子或训练长度变化：所有样本的编码及下游失效。
- 样本替换：只有该样本的编码及下游失效，其余样本三个阶段全部直接复用。
- 精度档位变化：所有样本失效，但计算仍为 float64。
- 仅容差变化：编码和打分复用，只重算误差阶段。
- 预算或并发变化：不改变缓存键。

每条缓存结果是带 `sample_id`、`scheme`、`stage` 引用的 JSON 记录；报告为每个结果列出三个阶段缓存键和引用。

实际单样本替换验收中，`encoding`、`attention`、`error_trace` 均只计算 1 个样本、复用 13 个；增量结果与空白全量重建的报告哈希完全相同。

## 阶段快照、恢复与陈旧检测

阶段为 `loading`、`encoding`、`attention`、`error_trace`。每完成 `snapshot_commit_interval` 个任务以及阶段结束时：

1. 将完整阶段 payload 写入同一文件系统下的临时目录。
2. 以 payload SHA-256 作为不可变版本名，将临时目录原子重命名为 `stages/<stage>/<version>/`。
3. 原子替换 `manifest.json` 中的阶段指针。

这样读者只会看到完整版本；旧版本目录可保留作为历史产物。清单记录全局版本号、配置哈希、已完成方案/样本、已消耗预算、各阶段产物版本与路径、状态和当前阶段。

启动时同时比较 `global_version` 与配置哈希：

- 匹配：`resumed=true`，从阶段缓存续跑。
- 不匹配：输出 `E_SNAPSHOT_STALE`，不复用“运行状态”语义，但仍按内容键安全读取可匹配的旧阶段记录；随后生成新快照。

`SIGINT`/`SIGTERM` 只设置停止事件，不打断正在执行的阶段；阶段完整落盘后退出，报告状态为 `interrupted`。下一次同配置运行会复用已提交阶段。测试中在 `encoding` 后停止，恢复运行时编码复用 14/14，最终诊断与全新运行完全一致。

## 报告结构

核心顶层字段：

- `status`：`complete`、`interrupted` 或 `failed`。
- `config` / `config_hash` / `summary`。
- `samples`：逐样本方案结果、位置摘要、打分、权重、分层统计、逐级误差、缓存键和引用。
- `layer_statistics`：按方案/训练内外/同距离的分数统计。
- `error_distributions`：按方案/环节的偏差分布。
- `over_tolerance_samples`、`boundary_samples`。
- `budget_usage`：点容量、MAC、扣减序列、exhausted 样本和 pending 队列。
- `validation`：样本与编辑错误。
- `run_metadata`：运行期、缓存、并发观测与快照信息。

## 实际验证结果

`artifacts/acceptance_summary.json` 显示全部 16 项验收检查通过：

- 三方案 fresh run：42/42 任务成功；同一外推样本得到 3 个不同位置摘要和 3 组不同分数。
- 单样本替换增量：每阶段 computed=1、reused=13；与空白全量重建确定性哈希相同。
- 四类组合编辑：出现 `E_SNAPSHOT_STALE`，诊断结果仍与空白全量重建哈希相同。
- 阶段中断与恢复：中断后 attention 未计算；恢复后 encoding 复用 14/14，最终结果与干净运行相同。
- 预算：低全局预算产生 1 个 `E_POINT_BUDGET_PENDING`；低单样本配额产生 1 个 `E_SAMPLE_BUDGET_EXHAUSTED`。
- 并发：单线程与 4 线程的诊断结果和预算扣减序列一致。
- 测试：22 项 `unittest` 全部通过，日志见 `artifacts/test_results.txt`。

主要证据文件：

- `artifacts/all_schemes_report.json`
- `artifacts/incremental_sample_only_report.json`
- `artifacts/incremental_combined_report.json`
- `artifacts/full_combined_report.json`
- `artifacts/interrupted_after_encoding.json`
- `artifacts/resumed_after_interrupt.json`
- `artifacts/budget_pending_report.json`
- `artifacts/budget_exhausted_report.json`
- `artifacts/snapshot_all_schemes/manifest.json`

## 设计边界

- 本工具是离线诊断器，不加载真实 LLM 权重，也不访问网络。
- 点数/MAC 账本覆盖查询与键编码、点积打分；softmax、误差统计等诊断开销不计入配置的全局 MAC 预算。
- NTK 缩放按题面提供同一 `scaling_factor`；当 `scaling_factor == 1` 时三种方案在数学上可能重合，这是定义本身的结果。
- 快照是单机文件系统语义，依赖同目录原子重命名；不实现多写入者分布式锁。
