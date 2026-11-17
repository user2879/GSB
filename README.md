# 本地确定性增量迁移学习管线

本项目在本地 CPU 上完成共享编码器前向、任务适配器训练/评估、内容指纹缓存、精确失效判定、并发执行、阶段快照与异常注入。运行时不访问网络、外部数据集或 GPU，训练过程不使用随机数。

核心实现位于 `transfer_pipeline/`，可复现样本位于 `examples/`，验收与单元测试位于 `tests/` 和 `artifacts/verification.json`。

## 快速开始

要求 Python 3.10+、NumPy、PyYAML。联网环境可直接安装：

```bash
python3 -m pip install --user -r requirements.txt
```

离线环境使用预下载 wheel：

```bash
python3 -m pip install --user --no-index --find-links /path/to/wheelhouse \
  numpy PyYAML pytest
```

首次全量运行：

```bash
python3 -m transfer_pipeline run \
  --tasks examples/tasks.json \
  --features examples/features.json \
  --encoder examples/encoder.json \
  --config examples/adapter_config.json \
  --state-dir /tmp/transfer-state \
  --report /tmp/transfer-report.json
```

再次运行同一输入时所有任务都命中内容缓存，报告中的 `cache.training_invocations` 为 `0`。

## 输入格式

### 任务描述

```json
{
  "shared_feature_dim": 8,
  "tasks": [
    {
      "name": "sentiment",
      "classes": ["negative", "positive"],
      "train_ids": ["s001", "s002"],
      "val_ids": ["s019", "s020"]
    }
  ]
}
```

任务名必须唯一；类别集合非空且无重复；训练/验证样本 ID 必须存在；样本标签必须属于任务类别集合；任务共享维度必须同时匹配特征库和编码器输入维度。校验失败产生 `E_TASK_DESC_INVALID`，任务进入失败清单，其他任务继续执行。

### 合成特征库

```json
{
  "seed": 20261118,
  "feature_dim": 8,
  "samples": {
    "s001": {"features": [0.1, -0.3, 1.2, 0.0, 0.7, -0.2, 0.4, 1.0], "label": "negative"}
  }
}
```

### 编码器参数包

`weights` 形状为 `[output_dim, shared_feature_dim]`，标准前向为：

```text
z = (x - mean) / std
h = z @ weights.T + bias
```

```json
{
  "input_dim": 8,
  "output_dim": 6,
  "weights": [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]],
  "bias": [0.01],
  "normalization": {"mean": [0, 0, 0, 0, 0, 0, 0, 0], "std": [1, 1, 1, 1, 1, 1, 1, 1]}
}
```

样本按 ID 升序前向，表示缓存键由样本 ID、特征向量和编码器内容指纹派生。

### 适配器配置

```json
{
  "adapter_type": "linear_probe",
  "rank": 2,
  "learning_rate": 0.4,
  "epochs": 160,
  "batch_size": 8,
  "validation_threshold": 0.7,
  "concurrency": 4,
  "snapshot_commit_interval": 2,
  "convergence_tolerance": 1e-12,
  "lora_scale": 0.2,
  "task_overrides": {
    "intent": {"adapter_type": "low_rank", "rank": 2, "learning_rate": 0.2}
  }
}
```

支持 `linear_probe` 与 `low_rank`。训练严格按任务内样本 ID 升序和固定批量切分执行，线性探针和低秩适配均为零/确定性初始化，不使用 shuffle、随机种子或随机失活。

## 增量编辑

可编程编辑序列使用 `examples/edits.json` 的格式，由 `transfer_pipeline.edits.apply_edit` 执行：

```json
[
  {"op": "add_task", "task": {"name": "urgency", "classes": ["critical", "normal"], "train_ids": ["s087", "s099"], "val_ids": ["s095"]}},
  {"op": "delete_task", "name": "topic"},
  {"op": "set_classes", "name": "mood", "classes": ["high", "low"]},
  {"op": "replace_sample", "sample_id": "s010", "features": [0.0, 1.0], "label": "negative"},
  {"op": "set_adapter_params", "name": "sentiment", "changes": {"learning_rate": 0.22}},
  {"op": "set_adapter_params", "name": "*", "changes": {"epochs": 200}}
]
```

交互模式支持相同操作：

```bash
python3 -m transfer_pipeline repl \
  --tasks examples/tasks.json \
  --features examples/features.json \
  --encoder examples/encoder.json \
  --config examples/adapter_config.json \
  --state-dir /tmp/transfer-repl-state \
  --report /tmp/transfer-repl-report.json
```

命令：

```text
run
show
add-task TASK.json
delete-task NAME
set-classes NAME class1,class2
replace-sample REPLACEMENT.json
set-hyper TASK|* KEY VALUE
save DIR
quit
```

默认每次修改后自动执行增量运行；加 `--manual` 后需显式输入 `run`。`save DIR` 可导出当前任务、特征、编码器和配置，便于做全量重建对照。

## 缓存键与失效判定

实现分为三个内容指纹层级：

- 共享表示键：`sample_id + feature_fingerprint + encoder_fingerprint + algorithm_version`。
- 适配器键：任务名、类别集合、排序后的训练样本内容及表示键、有效训练超参数。
- 评估键：适配器键、排序后的验证样本内容及表示键、类别集合、验证阈值。

因此：

- 替换训练样本会使引用该样本的适配器及评估失效。
- 仅替换验证样本时复用适配器，只重算评估指标。
- 修改验证阈值只重算评估。
- 修改某任务超参数只失效该任务；修改全局超参数失效全部有效任务。
- 新增/删除任务不会使其他任务失效；删除任务写入 `deleted_tasks`。
- 缓存是内容寻址的，旧内容仍可保留；回退到旧输入时可再次命中。

报告 `incremental` 中的 `invalidated_tasks`、`reused_tasks`、`adapter_*`、`evaluation_*`、`trained_tasks` 均按任务名排序。

## 并发与确定性

训练/评估通过 `ThreadPoolExecutor` 并发执行，最大并发数取自配置。每个任务使用 `stable_sort_key = "task:<name>"`，最终任务列表、参数表、缓存清单和失败清单都按稳定键排序。单次任务内部数组顺序和批量顺序固定，因此并发度不会改变逐字段结果。

## 快照与恢复

状态目录包含原子替换的 `snapshot.json`；快照含：

- `global_version`、`config_hash`、`run_id`
- `completed_tasks`、`completed_stages`、`stage_versions`
- 共享表示缓存、适配器缓存、评估缓存
- 当前任务索引、样本引用表、失败清单和注入命中报告

共享前向完成后提交一次，训练/评估按 `snapshot_commit_interval` 提交，最终再提交终态。所有写入使用同目录临时文件、`fsync` 和 `os.replace`。

启动时若当前保存的核心配置哈希与快照不一致，或显式 `--expected-version` / `--expected-config-hash` 不匹配，返回报告失败码 `E_CONFIG_MISMATCH`、退出码 `2`，且不覆盖原快照。相同 `run_id` 的中断快照可用 `--resume` 强制检查并续跑；已完成共享阶段和内容化任务会被跳过或直接命中。

收到 `SIGINT`/`SIGTERM` 时，信号处理器只设置停止事件。当前已启动任务会完成并提交快照，尚未启动的任务跳过，最终以整数退出码 `0` 返回 `interrupted` 状态。

## 异常注入与业务码

`examples/injections.json` 演示缺失样本、非法类别、零秩和负学习率：

```bash
python3 -m transfer_pipeline run \
  --tasks examples/tasks.json \
  --features examples/features.json \
  --encoder examples/encoder.json \
  --config examples/adapter_config.json \
  --injections examples/injections.json \
  --state-dir /tmp/transfer-injection-state \
  --report /tmp/transfer-injection-report.json
```

报告中 `injection_hits` 的每个注入项带稳定编号（如 `INJ-001`）、目标任务、预期业务码、是否应用和 `hit` 布尔值。也支持 `train_error`、`evaluate_error` 注入；单任务训练或评估异常会记录任务名、阶段、业务码、Python 异常类型，并继续其他任务。

主要业务码：

| 业务码 | 含义 |
| --- | --- |
| `E_TASK_DESC_INVALID` | 任务结构、类别、样本 ID 或维度不合法 |
| `E_ADAPTER_CONFIG_INVALID` | 秩、学习率等任务有效适配配置不合法 |
| `E_CONFIG_MISMATCH` | 快照配置哈希或显式版本不匹配 |
| `E_SHARED_FORWARD_FAILED` | 共享前向阶段失败 |
| `E_TRAINING_FAILED` | 任务训练阶段异常 |
| `E_EVALUATION_FAILED` | 任务评估阶段异常 |
| `E_SNAPSHOT_INVALID` | 快照不可读或结构不合法 |

## 报告字段

完整报告包含：

- `tasks`：按任务稳定键排序的适配器摘要、参数内容摘要与摘要指纹、训练损失、收敛状态、准确率、宏平均 F1、缓存键与命中标记。
- `adapter_parameter_table`：按同一任务顺序排列的参数摘要表。
- `incremental`：精确失效/复用/训练任务清单、删除任务清单及对应任务数。
- `cache`：共享表示、适配器、评估的命中/重算计数与整任务缓存命中/失效数。
- `sample_references`：每个特征库样本被哪些有效任务引用。
- `failures`：按任务、阶段、业务码、异常类型、注入编号、消息稳定排序。
- `injection_hits`：注入项编号及命中证明。
- `snapshot`：全局版本、配置哈希、已完成任务、完成阶段和各阶段产物版本。
- `failure_count`：失败清单条数。

## 验证结果

本工作区实际执行结果（Python 3.11.0rc1，NumPy 2.4.6，PyYAML 5.4.1，CPU）：

```text
$ python3 -m pytest -q
16 passed in 18.12s

$ python3 -m transfer_pipeline verify --examples examples --output artifacts
{"outcomes_compared": [...], "passed": true, "step_count": 7}
EXIT=0
```

`artifacts/verification.json` 记录初始运行及 6 个连续编辑步骤。每一步均将增量结果与全新状态目录中的全量重建结果逐字段比较，包含：

- 适配器参数摘要与参数指纹
- 训练损失、收敛状态、训练轮数
- 适配器/评估内容缓存键
- 准确率与宏平均 F1
- 排序后的失败清单
- 样本引用表

实测关键步骤：

| 步骤 | 编辑 | 增量失效任务 | 复用任务 | 训练任务 | 全量一致性 |
| --- | --- | --- | --- | --- | --- |
| 1 | 新增 `urgency` | `urgency` | 其余 4 个 | `urgency` | 是 |
| 2 | 删除 `topic` | 无 | 其余 4 个 | 无 | 是 |
| 3 | 修改 `mood` 类别集合 | `mood` | 其余 3 个 | `mood` | 是 |
| 4 | 替换 `s010`（验证样本） | `sentiment` | 其余 3 个 | 无；`sentiment` 适配器复用、仅评估重算 | 是 |
| 5 | 替换 `s004`（训练样本） | `sentiment` | 其余 3 个 | `sentiment` | 是 |
| 6 | 调整 `sentiment` 学习率 | `sentiment` | 其余 3 个 | `sentiment` | 是 |

第 4 步的缓存计数为 `adapter_hits=4`、`training_invocations=0`，同时 `sentiment` 的 `evaluation_hits` 为 0，验证了适配器与评估的分离失效。

## 测试覆盖

`tests/test_pipeline.py` 覆盖：

- 不同空工作区的全量运行逐字段确定性
- 重复运行零训练、全命中
- 完整编辑序列的增量/全量逐步一致性
- 任务结构校验失败排序与有效任务隔离
- 四类注入编号命中
- 训练/评估异常隔离
- 1/2/8 并发下结果一致
- 中断快照恢复后与全量一致
- 实际 `SIGTERM` 下完成当前任务快照并以退出码 0 结束
- 版本与配置哈希 `E_CONFIG_MISMATCH`
- 仅验证样本变化时适配器复用、评估重算
- 仅验证阈值变化时适配器复用、评估重算
- 报告、参数表、样本引用和临时文件清理
- 常见字段别名、数值类别标签及可选超参数默认值
