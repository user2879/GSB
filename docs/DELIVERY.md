# RoPE 定点推理诊断实验室 — 交付说明

## 快速开始

```bash
python3 -m venv .venv                 # 若镜像缺少 ensurepip，见 docs/VERIFICATION.md 的环境记录
source .venv/bin/activate
pip install numpy==1.26.4

python -m rope_lab init --root .       # 重新生成配置、样本、命令流、注入清单
python -m rope_lab validate            # 校验本地 JSON/JSONL
bash scripts/verify.sh                 # 单测 + 端到端验收 + Tk 冒烟
python -m rope_lab gui                 # 启动 Tkinter 工作台
```

无头环境可使用 `xvfb-run -a python scripts/tk_smoke.py`。当前 WSL 已检测到 `DISPLAY=:0`，真实 Tk 窗口创建与完整工作台填充均通过。

## 目录与职责

- `configs/lab_config.json`：维度、方案、定点位宽/量化档、容差、并发、预算和外推分档。
- `data/samples.jsonl`：固定整数种子生成的查询向量、键位置、键向量、查询位置和训练长度标记。
- `commands/edit_commands.jsonl`：方案增删、基频/缩放调整、训练/目标长度、样本替换、位宽/量化和预算命令。
- `injections/injections.json`：五类独立异常注入编号。
- `rope_lab/validation.py`：配置、样本 JSONL、枚举、有限性、唯一性和指定业务码校验。
- `rope_lab/rope.py`：多方案共用的统一 `RoPEScheme` 接口、角度/正余弦量化和单维度定点旋转。
- `rope_lab/fixedpoint.py`：二的幂量化、三种舍入、乘积重定标和二进制补码溢出模拟。
- `rope_lab/scoring.py`：维度升序点积、键顺序处理、稳定 softmax、分类分档、误差和排序翻转判别。
- `rope_lab/engine.py`：内容指纹缓存、维度级任务、线程池合并、误差矩阵、统计与复用账本。
- `rope_lab/scheduler.py`：样本稳定顺序、`dimension × key_count` 预算、单样本配额和待办优先队列。
- `rope_lab/snapshot.py`：临时目录写入、目录原子改名、`current` 符号链接整体替换和版本/哈希恢复判定。
- `rope_lab/injections.py`：五类真实故障变异、业务码命中和单任务失败隔离。
- `rope_lab/gui.py`：Tkinter 诊断工作台。
- `artifacts/verification/`：最终 JSON 报告、账本、注入报告、翻转归因和总验收摘要。
- `artifacts/snapshot-state/current/`：最近一次阶段化快照。

## 关键实现约定

### 定点路径

量化尺度为 `S = 2 ** fraction_bits`。角度、余弦、正弦和输入特征先转成有符号整数；旋转使用整数乘加，再按 `fraction_bits` 做确定性重定标。点积按维度 `0..D-1` 升序累加，键按 JSONL 中的列表顺序处理；没有使用 `numpy.sum`、数组默认归约或运行时全局随机状态参与计算。

允许枚举：

- `fraction_bits`: `4, 8, 12, 16`
- `signed_bits`: `16, 32, 64`
- `accumulator_bits`: `16, 32, 40, 64`
- `quant_level`: `truncate, nearest, ties_even`
- `frequency_correction`: `0..4`

切换 `fraction_bits`、`signed_bits` 或 `quant_level` 会改变缓存键和真实整数中间值，因此旋转整数、定点分数和归一化结果都会重算。
默认样本 `S000` 含精确半量化步长，`truncate`、`nearest`、`ties_even` 三种量化档两两之间均有可见整数和打分差异。

### 真值与误差

float64 真值路径不做定点量化，但仍使用相同的显式计算顺序。`position`、`rotation`、`dot`、`normalization` 四个环节均输出绝对误差、相对误差、容差判定和首次超容差标记。若选中的真值为 `0.0`，相对误差退化为绝对误差，并设置 `zero_truth_branch=true`。

### 缓存与精确失效

每个阶段键由方案参数指纹、样本几何/维度依赖指纹、定点配置指纹和上游缓存键派生。训练长度、目标长度和分档只进入统计键，因此修改它们时位置、旋转、点积和归一化条目直接复用。方案基频/缩放变化只使该方案命名空间下的任务键变化；样本单维变化通过相邻旋转对依赖指纹精确影响对应维度。

账本条目包含 `stable_key`、`cache_key`、`old_cache_key`、`cache_status`、`reuse_hit` 和 `invalidation_basis`。稳定键形如：

```text
S002|d0007|020:rotation|scheme=rope_standard
*SCHEME*|d-001|060:stats|scheme=rope_standard
```

### 预算与快照

样本成本严格按 `dimension × len(key_positions)` 计算。成本超过 `sample_quota` 时记 `E_POINT_QUOTA_EXCEEDED`；当前预算无法容纳队首样本时，该样本及其后候选进入待办并记 `E_BUDGET_PENDING`，下一批待办优先。

快照把报告、缓存产物、任务索引、待办队列和累计消耗写入同一临时目录，fsync 后改名为版本目录，再用原子替换 `current` 符号链接完成整体提交。恢复时核对 `global_version` 与 `config_hash`。

### 翻转归因

对每个外推键对同时计算四条路径：

1. float64 真值；
2. 仅量化角度后继续 float64；
3. 全定点旋转和宽累加器；
4. 按 `accumulator_bits` 模拟补码包装的累加器。

若路径 2 已翻转，归因为 `angle_quantization`；否则路径 3 翻转归因为 `rotation_rounding`；仅路径 4 翻转归因为 `accumulator_overflow`。`make_flip_attribution_cases()` 固化了三种均稳定命中的种子案例。
