# RoPE 外推诊断组件

这是一个仅依赖 Python 3.11+ 与 NumPy 的本地确定性诊断组件，用于在长上下文注意力部署前比较：

- 标准旋转位置编码 `rope`
- 线性内插 `linear_interpolation`
- NTK 感知缩放 `ntk_aware`

组件覆盖配置/样本/剪枝/注入校验、统一位置编码算子、稳定注意力打分、训练窗内外分层统计、逐级误差追踪、参数邻域敏感性、规模自适应选型、内容指纹缓存、单批点数预算调度、线程池并发、异常注入隔离和阶段化快照。

默认样本由固定种子生成，不访问网络、外部账号或 GPU；快照、临时文件和报告均位于工作区内。

## 快速开始

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy

python scripts\generate_samples.py --output-dir sample_data --count 8 --seed 20261124
python scripts\selfcheck.py --workspace .
python -m rope_diag --workspace . --output artifacts\latest\report.json
```

可直接执行 `scripts\windows_selfcheck.ps1 -Workspace .` 检查当前 Windows 虚拟环境中的 NumPy 导入、项目导入与样本校验。

### Linux / macOS

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt

python scripts/generate_samples.py --output-dir sample_data --count 8 --seed 20261124
python scripts/selfcheck.py --workspace .
python -m rope_diag --workspace . --output artifacts/latest/report.json
```

测试：

```bash
python -m unittest discover -s tests -v
```

可选的可安装入口：

```bash
python -m pip install . --no-deps
rope-diag --workspace . --output artifacts/latest/report.json
```

## 目录

```text
rope_diag/
  config.py         配置 JSON 校验
  samples.py        JSONL 样本逐行校验与内容指纹
  pruning.py        维度剪枝策略表
  injections.py     异常注入清单
  rope.py           三类 RoPE 统一算子接口
  attention.py      分块点积、稳定 softmax
  statistics.py     训练窗内/外与同距离分层统计
  trace.py          位置向量、打分、注意力权重三级误差
  sensitivity.py    基频/缩放因子邻域扫描与边界条件
  strategy.py       预置策略表与规模自适应选型
  fingerprints.py   阶段缓存键派生
  cache.py          结果缓存
  scheduler.py      单批点数预算调度
  snapshot.py       工作区内整目录原子替换快照
  pipeline.py       端到端编排、线程池、注入隔离和报告
  cli.py            命令行入口
scripts/
  generate_samples.py
  selfcheck.py
  windows_selfcheck.ps1
sample_data/        固定种子本地样本
tests/              15 项单元/集成测试
artifacts/latest/   验收报告
```

## 输入格式

### 位置编码配置 `config.json`

```json
{
  "schema_version": 1,
  "global_version": 1,
  "schemes": ["rope", "linear_interpolation", "ntk_aware"],
  "base": 10000.0,
  "training_context_length": 16,
  "extrapolation_target_length": 96,
  "dimension": 32,
  "scaling_factor": 4.0,
  "precision": "float64",
  "quantization": "none",
  "dimension_chunk_size": 32,
  "point_budget": 100000,
  "concurrency": 4,
  "snapshot_interval": 5,
  "error_tolerances": {
    "position_vector": 0.01,
    "score": 0.1,
    "attention": 0.001
  },
  "sensitivity": {
    "base_step": 2500.0,
    "scale_step": 0.5,
    "radius": 2
  }
}
```

约束：

- `dimension` 为正偶数，`base`、`scaling_factor` 为有限正数。
- `training_context_length < extrapolation_target_length`。
- `dimension_chunk_size`、`point_budget`、`concurrency`、`snapshot_interval` 为正整数。
- `precision` 允许 `float64`、`float32`；`quantization` 允许 `none`、`int8_symmetric`。
- `scheme` 可用于单方案快捷写法；同时提供 `scheme` 与 `schemes` 会报错。
- `global_version` 与配置哈希共同参与续跑判定。

### 注意力样本 `samples.jsonl`

每行一个对象：

```json
{
  "sample_id": "s0001",
  "query_position": 22,
  "key_positions": [0, 3, 7, 11, 15, 18, 22],
  "query_features": [0.12, -0.33, 0.91],
  "key_features": [[0.2, -0.1, 0.4], [0.0, 0.2, -0.5]]
}
```

`query_features` 长度为 `dimension`；`key_features` 行数必须等于 `key_positions` 数量，每行长度必须等于 `dimension`。所有元素必须为有限数。样本 ID 必须唯一，位置必须为非负整数，键位置不得重复。

上面的特征数组为结构示意；实际数组中每个向量都必须包含恰好 `dimension` 个有限数值，且 `key_features` 的每一行与 `key_positions` 项一一对应。

### 维度剪枝策略 `pruning_strategies.json`

策略按 RoPE 对数保留，保证同一旋转对的 `(2i, 2i+1)` 维度同时保留：

```json
{
  "policies": [
    {
      "policy_id": "full",
      "description": "all pairs",
      "pair_ranges": [{"start_ratio": 0.0, "end_ratio": 1.0}],
      "pair_stride": 1
    }
  ]
}
```

### 异常注入 `injections.json`

```json
{
  "injections": [
    {
      "injection_id": "INJ-SAMPLE-SCORING",
      "enabled": true,
      "scheme": "linear_interpolation",
      "sample_id": "s0006",
      "stage": "scoring",
      "action": "raise",
      "message": "deterministic scoring fault"
    }
  ]
}
```

`scheme` / `sample_id` 为 `null` 时表示通配。阶段允许 `position`、`scoring`、`statistics`、`trace`、`sensitivity`；动作允许 `raise`、`nonfinite`、`truncate`。执行时均被隔离为 `E_INJECTED_FAULT`，不中断其他任务。

## 数值定义

设位置 `p`、维度 `d`、旋转对索引 `i=0..d/2-1`，三种方案实际角度不同：

- 标准：`theta_i = p * base^(-2i/d)`
- 线性内插：`theta_i = (p / scaling_factor) * base^(-2i/d)`
- NTK 感知：`effective_base = base * scaling_factor^(d/(d-2))`，`theta_i = p * effective_base^(-2i/d)`；`d=2` 时指数退化为 1。

位置向量为每个旋转对写入 `(cos(theta_i), sin(theta_i))`，旋转对记录维度索引、角度、余弦和正弦。实际特征按二维旋转：

```text
y[2i]   = x[2i] * cos - x[2i+1] * sin
y[2i+1] = x[2i] * sin + x[2i+1] * cos
```

所有点积、softmax、统计与误差数值均使用 `float64`。`float32` 精度档位仅模拟输入精度舍入；`int8_symmetric` 仅模拟对称量化输入。最终诊断数值仍以 float64 计算。

注意力在面对每个样本的键列表时执行稳定归一化：先减最大打分，再取指数并归一化。

## 诊断语义

### 分层统计

相对距离定义为 `abs(query_position - key_position)`。距离 `< training_context_length` 属于 `in_training`，否则属于 `extrapolation`。报告分别给出总体及每个距离的 `score_count`、`sample_count`、均值、最小值、最大值。没有外推打分时输出 `E_NO_EXTRAPOLATION_SAMPLE`，该方案统计停止，但其他方案继续。

### 逐级误差

参考位置将每个查询/键位置截断到 `training_context_length - 1`，使用同一方案重新生成训练窗内编码。然后计算：

1. `position_vector`：外推位置编码向量与参考向量的归一化 L2 偏差。
2. `score`：外推位置实际点积与参考点积的绝对偏差均值。
3. `attention`：稳定 softmax 后外推位置权重的绝对偏差均值。

每个环节都写入偏差、容差和是否超差；`first_exceeded_stage` 记录首个超差环节。即使前一环节超差，后续环节仍完整计算并写入报告。

### 敏感性

对每种方案分别扫描 `base + k * base_step` 与 `scaling_factor + k * scale_step`，`k` 在 `[-radius, radius]` 内。每个候选值与当前方案基线在外推打分上比较，按绝对偏差降序排序，偏差相同按参数值升序。边界条件（无有限打分）优先展示，并记录：

- `scaling_overflow:adjusted_base`
- `dimension_truncation:inverse_frequency_underflow`
- `angle_overflow`
- 非正/非有限基频或缩放因子

边界项记录触发样本和条件，扫描步长写入报告。

## 规模自适应选型

内置策略表位于 `rope_diag/strategy.py`，同时参考外部剪枝策略表。决策使用样本数档位、维度档位、外推倍率档位和预算档位，输出：

- 方案优先级
- 选中的剪枝策略
- 实际参与打分和误差追踪的维度集合
- 实际分块边界与块大小
- 所参考的样本数、总点数、投影点数、预算和判定依据

维度集合与分块边界进入评分缓存键，因此选型变化会真实改变计算路径并让受影响下游缓存失效。

## 缓存、预算与快照

- 位置编码缓存键由方案参数、样本内容、精度、量化、剪枝口径派生。
- 打分缓存键再叠加选型指纹、维度集合和分块边界。
- 误差追踪与敏感性缓存键由方案参数、数据集内容指纹、容差/邻域、策略和剪枝口径派生。
- 单条样本替换只重算该样本的位置编码/打分；该方案下游的 trace 与 sensitivity 失效重算。
- 缩放因子变化时标准 RoPE 的位置编码、打分与误差追踪可复用，线性内插与 NTK 失效；敏感性因扫描锚点变化而重算。
- 报告输出 `cache.stage_events`、`cache.invalidated`、`cache.reused` 及每个样本决策。

调度按 `(scheme, sample_id)` 升序消耗键点数预算。预算不足时从当前项开始停止并输出 `E_POINT_BUDGET_PENDING`；已完成项在相同版本/配置/数据/注入清单下可续跑。线程池任务都带稳定排序键，预算扣减与最终报告顺序在串行和并发模式下一致。

快照写入 `<workspace>/.rope_diag/snapshots/staging-*`，完成后整目录替换到 `current`。清单包含：

- `global_version`
- `config_hash`
- `injection_hash`
- `completed_tasks`
- `completed_schemes`
- `completed_samples_by_scheme`
- `stage_keys`
- `commit_seq`

报告通过临时文件加 `os.replace` 原子写入；路径全部使用 `pathlib.Path`，Windows 与 POSIX 均可运行。

## 错误码

| 错误码 | 含义 |
|---|---|
| `E_CONFIG_INVALID` | 配置、剪枝或注入清单非法 |
| `E_SAMPLE_INVALID` | 样本行或字段非法 |
| `E_NO_EXTRAPOLATION_SAMPLE` | 某方案没有外推区间打分 |
| `E_POINT_BUDGET_PENDING` | 单批点数预算不足，任务待续跑 |
| `E_INJECTED_FAULT` | 注入故障已被隔离并记录 |
| `E_CACHE_CORRUPT` | 缓存/快照无法解码 |
| `E_IO_ERROR` | 输入读取失败 |
| `E_INTERNAL_ERROR` | 未处理内部异常 |

失败清单统一按样本 ID 升序；无样本 ID 的配置/聚合失败排在最前。

## 验收结果

本次实际执行环境：

- Python `3.11.0rc1`
- NumPy `1.26.4`
- 工作区：`/home/zsl/feedback/uidfc5909d3-default-210`

校验自检通过：配置、4 条剪枝策略、1 条默认注入项、8 条样本均合法；`E_CONFIG_INVALID` 与 `E_SAMPLE_INVALID` 自测可用。

默认 8 条本地样本（32 维、训练窗 16、目标 96、缩放因子 4）运行结果：

| 方案 | 状态 | 完成样本 | 外推打分点数 | 归一化前打分阶段平均偏差 |
|---|---:|---:|---:|---:|
| `rope` | `ok` | 8/8 | 50 | 4.040450 |
| `linear_interpolation` | `partial_fault` | 7/8 | 43 | 3.274033 |
| `ntk_aware` | `ok` | 8/8 | 50 | 3.391939 |

默认注入命中：

- `INJ-SAMPLE-SCORING`：仅隔离 `linear_interpolation/s0006/scoring`。

热缓存第二次运行：23 个样本任务直接复用，只有被强制注入的 1 个任务进入调度；10 个阶段复用，2 个阶段因注入故障不可复用。首次运行、热缓存增量和强制全量重建的语义摘要均为 `61f0aaf5a04e2936aeb44182eea7229eeaf6c5c43252e85f9bcefd277c04ba10`。

测试套件结果：`Ran 15 tests ... OK`，覆盖三方案真实角度差异、稳定归一化、无外推、错误追踪不截断、敏感性排序与溢出边界、维度级截断、策略选型、增量复用、全量一致性、单样本替换、预算待续、串并行一致性及注入隔离。

当前执行环境为 WSL/Linux，无法在本机启动 Windows 主机进程；Windows 兼容性通过 `pathlib.Path`、`os.replace` 目录替换、无 POSIX 专用路径拼接和 `scripts/windows_selfcheck.ps1` 提供。Windows 上建议按快速开始命令实际执行一次该脚本确认导入。
