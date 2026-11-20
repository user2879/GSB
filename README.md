# RoPE Extrapolation Diagnostics

面向长上下文 LLM 位置编码外推问题的可集成 Python 组件与完整本地示例。组件在同一个统一接口下实现：

- 标准旋转位置编码（standard RoPE）
- 线性内插（linear interpolation / PI）
- NTK 感知缩放（NTK-aware scaling）

它会对固定种子生成的查询/键特征计算点积注意力打分，分别统计训练长度内与外推区间，追踪位置向量、打分和 softmax 注意力权重三级误差，并在有限交互延迟预算内提供**真实已完成摘要**与**待完成队列**。

实现只使用 Python 3.10+ 标准库与 NumPy；不需要 Docker、容器、外部服务或网络访问。

## 快速开始

### Windows PowerShell

可一键执行环境检查、虚拟环境、依赖安装、测试和验收：

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_windows.ps1
```

或手工执行：

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

python -m rope_extrap generate --output samples
python -m rope_extrap verify --output-dir artifacts
```

### Linux / WSL

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python -m rope_extrap generate --output samples
python -m rope_extrap verify --output-dir artifacts
```

运行测试：

```bash
python -m unittest discover -s tests -v
```

本次实际验证环境为 Linux/WSL2、Python 3.11.0rc1、NumPy 1.26.4；代码只使用跨平台路径、原子文件替换和线程池 API，Windows 上的 Python 3.10+ 使用相同命令。完整实测见 [docs/verification_report.md](docs/verification_report.md)。

## 工作区布局

```text
rope_extrap/                 # 可集成组件
  loader.py                  # 配置、JSONL 样本、命令流加载与校验
  encoding.py                # 三种位置编码、精度档位、维度剪枝
  scoring.py                 # float64 点积打分、softmax、分层统计
  tracing.py                 # 三级误差追踪与超差判定
  cache.py                   # 内容指纹与分阶段缓存键
  engine.py                  # 线程池执行、稳定排序、异常隔离、报告
  session.py                 # 延迟预算交互、后台队列、命令语义
  snapshot.py                # 阶段产物临时写入、版本快照、恢复判定
  analysis.py                # 三方案统计、敏感性排序、一致性视图
  cli.py                     # 命令行入口
samples/                     # 固定种子生成的真实输入
  config_large_budget.json   # 大预算：3000 ms
  config_small_budget.json   # 小预算：20 ms
  samples.jsonl              # 64 条注意力打分样本
  commands.json              # 19 条交叉命令
  fault_manifest.json        # 故障注入清单
  fault_samples.jsonl        # 可隔离坏样本
artifacts/                   # 实测 JSON 产物与验收报告
runtime/tmp/                 # 可写临时目录
runtime/snapshots/           # 快照目录及实测版本
tests/                       # 12 个自动化测试
scripts/                     # 样本生成与完整验收脚本
```

## 输入格式

### 位置编码配置 JSON

```json
{
  "scheme": "standard_rope",
  "base": 10000.0,
  "training_context_length": 32,
  "extrapolation_target_length": 128,
  "dimension": 32,
  "scale_factor": 4.0,
  "precision": "float64",
  "dimension_chunk_size": 4,
  "interaction_budget_ms": 3000,
  "concurrency": 4,
  "snapshot_commit_interval": 4,
  "pruning_tolerance": 0.05,
  "pruning_policy": "energy",
  "error_tolerance": 0.02
}
```

| 字段 | 语义与约束 |
|---|---|
| `scheme` | `standard_rope`、`linear_interpolation`、`ntk_aware` |
| `base` | RoPE 基频，必须有限且大于 0 |
| `training_context_length` | 训练上下文长度，正整数，必须小于目标长度 |
| `extrapolation_target_length` | 外推目标位置上界 |
| `dimension` | 偶数维度，和每条特征向量长度一致 |
| `scale_factor` | 线性内插或 NTK 缩放因子，必须大于 0 |
| `precision` | `float64` 或 `float32`；打分始终以 float64 汇总 |
| `dimension_chunk_size` | 偶数、正整数，必须整除 `dimension` |
| `interaction_budget_ms` | 单条交互命令延迟预算，正整数毫秒 |
| `concurrency` | 线程池最大并发数，正整数 |
| `snapshot_commit_interval` | 每 N 条命令尝试提交快照 |
| `pruning_policy` | `none`、`energy`、`frequency` |
| `pruning_tolerance` | `[0, 1]`；能量阈值或低频保留比例参数 |
| `error_tolerance` | 三级误差超差阈值；缺省时沿用 `pruning_tolerance` |

### 合成注意力打分样本 JSONL

每行一个样本：

```json
{
  "sample_id": "s001",
  "query_position": 80,
  "key_positions": [80, 79, 72, 48, 47, 40],
  "query_features": [0.12, -0.08, 0.03],
  "key_features": [[0.10, 0.01, 0.04], [0.09, -0.02, 0.05]]
}
```

- `sample_id` 必须唯一且非空。
- `query_position`、`key_positions` 必须是非负整数；键位置不得大于查询位置或目标长度。
- `dimension` 必须为偶数，并与 `query_features`、每个 `key_features[i]` 长度一致。
- 特征不得包含 `NaN`/`Infinity`。
- `samples.jsonl` 覆盖相对距离 `0, 7, 31`、边界 `32` 和外部 `33, 40, 47, 63, 95, 119` 等区间。

### 交互命令流 JSON

顶层可为数组，或 `{"commands": [...]}`。支持：

```json
[
  {"type": "set_scheme", "scheme": "linear_interpolation"},
  {"type": "set_base", "base": 20000.0},
  {"type": "set_scale", "scale_factor": 2.5},
  {"type": "set_training_context_length", "training_context_length": 24},
  {"type": "set_pruning", "pruning_policy": "frequency", "pruning_tolerance": 0.1},
  {"type": "replace_sample", "sample": {"sample_id": "s007", "query_position": 120, "...": "..."}},
  {"type": "update", "base": 12000.0, "scale_factor": 3.25},
  {"type": "poll", "budget_ms": 120000}
]
```

命令可覆盖 `budget_ms`。`commands.json` 交叉覆盖方案切换、基频修改、缩放因子修改、训练长度调整、剪枝口径变更与单条样本替换。

## 数学与诊断语义

### 三种位置编码

对第 `i` 个二维旋转对，`theta_i = base^(-2i/d)`。

- 标准 RoPE：`angle = position * theta_i`
- 线性内插：`angle = (position / scale) * theta_i`
- NTK-aware：`base_eff = base * scale^(d/(d-2))`，随后 `angle = position * base_eff^(-2i/d)`，`d=2` 时退化处理为 `base_eff=base*scale`

每一对 `(x0, x1)` 使用二维旋转。三种方案共用同一个 `encode_sample(config, sample)` 输入接口，并返回同一个 `EncodedSample` 结构；测试验证三者的位置向量指纹和打分指纹都不同。

### 精度与维度剪枝

- `float32` 档位会在输入和旋转结果处量化到 float32，再提升为 float64 参与后续计算。
- 打分、softmax 和误差追踪统一使用 float64。
- `none`：保留全部维度块。
- `energy`：按查询与键在各维度块上的平方能量，保留不低于最大块能量 `pruning_tolerance` 比例的块；至少保留一个块。
- `frequency`：低频率对优先，保留 `ceil(chunk_count * (1 - pruning_tolerance))` 个块。
- 被剪枝维度写入 0；每个样本记录 `kept_chunks`、`kept_dimensions`、`pruned_dimensions`。

### 打分与区间

`score_j = dot(q_rot, k_rot_j)`。相对距离 `d_j = query_position - key_position_j`：

- `d_j <= training_context_length`：训练区间
- `d_j > training_context_length`：外推区间

注意力权重为减去样本内最大打分后的 softmax。报告分别输出训练区间/外推区间的均值、最大值、样本数、键数，以及按相对距离分组的统计。每个样本和每个键记录最大绝对贡献维度。

### 三级误差追踪

确定性参考使用同一特征行和同一剪枝掩码，将每个位置夹到训练边界：

```text
reference_position = min(position, training_context_length)
```

依次计算：

1. `position_vector`：查询/键旋转向量的相对 L2 偏差
2. `score`：实际点积与参考点积的相对偏差
3. `attention_weight`：实际 softmax 与参考 softmax 的相对偏差

按上述顺序标注首次超过 `error_tolerance` 的环节。外推键逐项记录 `relative_distance`、`first_exceeded_stage`、三级偏差和 `max_deviation`；样本超差清单按 `max_deviation` 降序、样本 ID、距离确定性排序。整个链路不含随机源。

## 增量缓存与失效

每个阶段都有独立内容键：

```text
position_encoding_key = H(scheme params, sample content, precision, pruning)
attention_score_key   = H(encoding_key, training length, target length, ...)
error_trace_key       = H(score_key, training length, target length, error tolerance, ...)
```

实际行为：

| 变更 | 复用 | 重算 |
|---|---|---|
| 切换方案 | 无 | 全部样本三阶段 |
| 修改 base / scale | 无 | 全部样本三阶段 |
| 修改训练长度 | 位置编码 | 打分统计 + 误差追踪 |
| 修改剪枝口径/容差 | 无 | 全部样本三阶段 |
| 替换单条样本 | 其他样本三阶段 | 仅该样本三阶段 |
| `poll` | 已存在结果全部 | 仅缺失阶段 |

每次命令都输出 `reuse_and_recompute.reused/recomputed`，包含样本、阶段、稳定排序键、缓存键、特征行来源。测试已验证训练长度变更复用 64 个编码阶段、替换 `s007` 只重算该样本的 3 个阶段。

## 延迟预算与待完成队列

交互会话使用一个持久后台工作线程推进最新内容寻址任务：

1. 命令线程应用参数并等待配置预算内的安全时间片。
2. 若最新结果未完成，立即返回真实已完成摘要和 `E_INTERACT_BUDGET_PENDING`。
3. 后台线程继续推进同一队列；下一次 `poll` 或命令继续检查/复用已完成阶段。
4. 新的变更命令会取代旧目标，但旧目标已提交的精确缓存键结果仍可复用。

摘要字段全部来自实际缓存结果：已完成样本数、当前方案、外推超差样本数、进度和待完成阶段数。不会用样本总数、历史平均值或占位值填充未完成结果。小预算样本配置为 20 ms；大预算为 3000 ms。

> 说明：1-10 ms 级微预算下，Python 线程调度和响应序列化本身可能竞态；示例采用 20 ms 小预算，使所有命令响应在预算内，同时仍稳定触发 pending。实测小预算 p95 为 17.692 ms，19/19 条变更命令返回 pending，最终大预算 poll 排空队列。

## 快照与恢复

```text
runtime/snapshots/<name>/
  manifest.json
  v00000007-<config-hash-prefix>/
    position_encoding.json
    attention_score.json
    error_trace.json
```

- 三个阶段产物先写入临时目录并 `fsync`。
- 临时目录重命名为不可变版本目录，再原子替换 `manifest.json` 指针。
- manifest 记录 `global_version`、`config_hash`、`completed_samples`、`failed_samples`、各阶段 `artifact_versions`。
- 读取时校验版本号、配置哈希和阶段内容哈希；不一致返回 `E_SNAPSHOT_STALE`。
- 恢复后完全相同的内容键直接命中；实测 36 个阶段产物全部复用、0 个重算。

## 故障码

| 业务码 | 场景 |
|---|---|
| `E_CONFIG_INVALID` | 非法 JSON、字段缺失/类型错误、维度奇数、块不整除、预算/并发非正整数 |
| `E_SAMPLE_INVALID` | 坏 JSON 行、空文件、重复 ID、位置非法/越界、非有限特征、空键列表 |
| `E_DIM_MISMATCH` | 配置维度或特征行长度不一致 |
| `E_BASE_INVALID` | 基频小于等于 0 或非有限 |
| `E_SCALE_INVALID` | 缩放因子小于等于 0 或非有限 |
| `E_COMMAND_INVALID` | 未知命令、非法命令参数、替换未知样本 |
| `E_INTERACT_BUDGET_PENDING` | 预算内最新完整结果尚未完成，队列保留 |
| `E_SNAPSHOT_STALE` | 快照版本/配置/内容哈希不一致 |
| `E_COMPUTE_NONFINITE` | 计算阶段出现非有限值 |
| `E_INTERNAL` | 未预期阶段异常（仍按样本/阶段隔离） |

坏样本不会阻断其他样本。运行期任一阶段异常会记录样本 ID、阶段和业务码；依赖该样本的后续阶段标记 skipped，其余样本继续。

## 集成 API

```python
from rope_extrap.loader import load_position_config, load_samples, load_commands
from rope_extrap.session import InteractiveSession

config, config_failures = load_position_config("samples/config_small_budget.json")
samples, sample_failures = load_samples(
    "samples/samples.jsonl",
    dimension=config.dimension,
    target_length=config.extrapolation_target_length,
)
commands, command_failures = load_commands("samples/commands.json", config=config)

with InteractiveSession(config, samples, commands) as session:
    responses = session.run_all()
    final_poll = session.drain(budget_ms=120_000)
    report = session.final_report()
```

一次性全量诊断：

```python
from rope_extrap.engine import run_full_diagnosis

report = run_full_diagnosis(config, samples)
```

## CLI

```bash
# 生成固定种子输入
python -m rope_extrap generate --output samples

# 一次性全量 + 三方案统计 + 敏感性
python -m rope_extrap diagnose \
  --config samples/config_large_budget.json \
  --samples samples/samples.jsonl \
  --extended --out artifacts/one_shot.json

# 交互命令流 + 增量/全量对比
python -m rope_extrap diagnose \
  --config samples/config_small_budget.json \
  --samples samples/samples.jsonl \
  --commands samples/commands.json \
  --extended --full-comparison \
  --out artifacts/command_sequence.json \
  --responses-out artifacts/command_responses.json \
  --full-out artifacts/full_rebuild.json

# 故障加载
python -m rope_extrap faults \
  --config samples/config_large_budget.json \
  --samples samples/fault_samples.jsonl \
  --out artifacts/faults.json

# 完整验收
python -m rope_extrap verify --output-dir artifacts
```

## 实际验证结果

本次运行 `python -m rope_extrap verify --output-dir artifacts`：

- 11/11 验收检查通过。
- 64 条样本、19 条交叉命令、固定种子 `20261121`。
- 三方案位置向量与打分指纹均不同；标准 RoPE 平均最大偏差 5.835631，线性内插 3.322718，NTK 3.777277。
- 大预算 19/19 响应完整且预算内，p95 426.563 ms。
- 小预算 20/20 响应预算内，19 条返回 `E_INTERACT_BUDGET_PENDING`，p95 16.421 ms。
- 队列探针剩余阶段任务 `[72, 36, 0]`，最终 24 个样本全部完成。
- 增量最终报告与空白全量重建在全部实质诊断字段上逐字段一致；执行审计因缓存命中历史天然不同，明确分离。
- 快照 version 7 恢复 36/36 阶段复用；version 8 返回 `E_SNAPSHOT_STALE`。
- 注入 `error_trace` 阶段故障后，5 个样本完成、1 个样本隔离失败，其他样本不受影响。
- 线程池观测并发不超过配置的 4。

详细报告：`docs/verification_report.md`、`artifacts/acceptance_report.json`、`artifacts/diagnostic_report.json`。

### 关于“逐字段一致”的边界

增量运行和空白全量重建的**诊断结果**、缓存键、样本顺序、超差顺序和失败顺序完全相同。执行审计中的 `reused/recomputed` 布尔历史不可能与空白重建相同：空白运行没有任何历史缓存，必然全部为 recomputed。实现将“结果一致性”和“执行历史审计”分开：`diagnostic_consistency_view` 比较全部实质字段，测试和验收同时单独验证缓存命中与失效路径。这是该需求中两个目标在逻辑上的必要区分。
